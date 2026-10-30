"""Physical Score stream counts, common density initialization and strict resume metadata."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from globalsplat.compression import CodecConfig, ObservableLowRank1DCodec
from globalsplat.compression.bitstream import SceneBitstream, ScoreContextBitstream
from globalsplat.compression.checkpoint import (
    convert_score_path_state, infer_config, load_feature_codec_checkpoint,
    resize_registered_buffers, validate_feature_codec_checkpoint,
)
from globalsplat.compression.entropy import FactorizedScoreEntropy
from globalsplat.compression.score_context import ContextualScoreEntropy
from test_score_path_ablation import config, MINIMAL


def arm(schedule):
    values=config(**MINIMAL,token_order="hilbert",score_spatial_context=False,
        score_context_schedule=schedule,score_spatial_stages=1 if schedule=="single" else 4,
        score_spatial_kernel=5,score_context_quantization="ste").to_dict()
    values["score_slice_channels"]=values["rank"]
    return ObservableLowRank1DCodec(CodecConfig.from_mapping(values))


class ContextOffBlockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_invalid_single_schedule_context_and_metadata_are_rejected(self):
        good=arm("single").config
        for values in (dict(score_spatial_context=True),dict(score_channel_context=True),
                       dict(score_spatial_stages=2),dict(score_context_schedule="legacy")):
            with self.subTest(values=values),self.assertRaises(ValueError):
                replace(good,**values)
        codec=arm("single")
        state=codec.state_dict()
        self.assertEqual(infer_config(state,codec.config.to_dict()),codec.config)
        wrong=dict(state)
        wrong["score_context.single_block_marker"]=torch.tensor(0,dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError,"single-block"):
            infer_config(wrong,codec.config.to_dict())
        wrong=dict(state)
        del wrong["score_context.single_block_marker"]
        with self.assertRaisesRegex(ValueError,"configuration mismatch"):
            infer_config(wrong,codec.config.to_dict())

    def test_actual_rank56_streams_are_one_or_four_and_no_predictor_executes(self):
        for schedule,stages,count in (("single",1,1),("dyadic4",4,4)):
            with self.subTest(schedule=schedule),torch.no_grad():
                context=ContextualScoreEntropy(rank=56,scene_channels=736,slice_channels=56,hidden=64,
                    mean_condition=False,channel_context=False,spatial_context=False,spatial_stages=stages,
                    spatial_kernel=5,context_quantization="ste",context_schedule=schedule).eval()
                base=FactorizedScoreEntropy(56)
                context.update(base,force=True)
                def fail(*args):
                    raise AssertionError("Context OFF predictor executed")
                handles=[]
                for name in ("mean_conditioner","channel_predictors","spatial_predictors",
                             "split_even_predictors","split_odd_predictors","anchor_predictors"):
                    module=getattr(context,name,None)
                    if module is not None:
                        for child in module.modules():
                            handles.append(child.register_forward_pre_hook(fail))
                score=torch.randn(1,56,1,4096)
                mean=torch.zeros(1,736)
                try:
                    packet,hat=context.compress(score,mean,base)
                    received=context.decompress(packet,mean,4096,base)
                    forward,likelihood=context(score,mean,base,training=False)
                    altered=score.clone()
                    altered[...,0]+=100
                    other,other_likelihood=context(altered,mean,base,training=False)
                finally:
                    for h in handles:
                        h.remove()
                packed=ScoreContextBitstream.unpack(packet)
                self.assertEqual(len(packed.strings),count)
                self.assertEqual((packed.rank,packed.slice_channels),(56,56))
                self.assertFalse(packed.flags&context.FLAG_SPATIAL)
                self.assertEqual([len(ids) for _,ids in context.stage_indices(4096,'cpu')],
                                 [4096] if count==1 else [512,512,1024,2048])
                torch.testing.assert_close(hat,received,rtol=0,atol=0)
                torch.testing.assert_close(hat,forward,rtol=0,atol=0)
                torch.testing.assert_close(forward[...,1:],other[...,1:],rtol=0,atol=0)
                torch.testing.assert_close(likelihood[...,1:],other_likelihood[...,1:],rtol=0,atol=0)

    def test_regrouped_warm_start_preserves_every_channel_density_and_matches_arm_reconstruction(self):
        parent=ObservableLowRank1DCodec(config()).eval()
        # Distinguishable learned channel parameters/medians exercise concatenation order.
        with torch.no_grad():
            start=0
            for entropy in parent.score_context.group_entropies:
                width=entropy.entropy_bottleneck.quantiles.shape[0]
                entropy.entropy_bottleneck.quantiles[:,0,1]=torch.arange(start,start+width)*.05+.1
                start+=width
        parent.update(force=True)
        checkpoint=dict(state_dict=parent.state_dict(),feature_codec_config=parent.config.to_dict())
        outputs=[]
        inputs=tuple(torch.randn(1,65,c) for c in (8,8,3))
        for schedule,count in (("single",1),("dyadic4",4)):
            codec=arm(schedule).eval()
            with self.assertRaisesRegex(ValueError,"configuration mismatch"):
                validate_feature_codec_checkpoint(checkpoint,codec.config)
            validate_feature_codec_checkpoint(checkpoint,codec.config,allow_score_path_conversion=True)
            state=convert_score_path_state(parent.state_dict(),codec)
            resize_registered_buffers(codec,state)
            codec.load_state_dict(state,strict=True)
            for key,value in parent.state_dict().items():
                if not key.startswith("score_context."):
                    torch.testing.assert_close(state[key],value,rtol=0,atol=0)
            for entropy in codec.score_context.active_entropies(codec.score_entropy):
                for name,parameter in entropy.named_parameters():
                    expected=torch.cat([dict(e.named_parameters())[name] for e in parent.score_context.group_entropies],0)
                    torch.testing.assert_close(parameter,expected,rtol=0,atol=0)
            codec.update(force=True)
            compressed=codec.compress(*inputs)
            packed=ScoreContextBitstream.unpack(SceneBitstream.unpack(compressed.data).score)
            self.assertEqual(len(packed.strings),count)
            output=torch.cat(codec.decompress(compressed.data),-1)
            outputs.append(output)
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/"codec.ckpt"
                torch.save(dict(state_dict=codec.state_dict(),feature_codec_config=codec.config.to_dict()),path)
                loaded=load_feature_codec_checkpoint(path).codec.eval()
                self.assertEqual(loaded.config,codec.config)
                actual=torch.cat(loaded.decompress(compressed.data),-1)
                torch.testing.assert_close(actual,output,rtol=0,atol=0)
            codec.train().set_trainable_scope("all")
            trained=codec(*inputs,training=True)
            (trained.reconstruction_sorted.square().mean()+trained.estimated_bits/10000+codec.aux_loss()).backward()
            self.assertIsNotNone(codec.shared_basis.grad)
            self.assertTrue(any(p.grad is not None for p in codec.residual_codec.parameters()))
            active={id(p) for e in codec.score_context.active_entropies(codec.score_entropy) for p in e.parameters()}
            for name,p in codec.score_context.named_parameters():
                if "predictors." in name or ("entropies." in name and id(p) not in active):
                    self.assertFalse(p.requires_grad,name)
                    self.assertIsNone(p.grad,name)
        torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)

    def test_regrouping_is_not_silently_allowed_with_enabled_context(self):
        parent=ObservableLowRank1DCodec(config())
        checkpoint=dict(state_dict=parent.state_dict(),feature_codec_config=parent.config.to_dict())
        requested=replace(arm("dyadic4").config,score_spatial_context=True)
        with self.assertRaisesRegex(ValueError,"regrouping requires"):
            validate_feature_codec_checkpoint(checkpoint,requested,allow_score_path_conversion=True)


if __name__=="__main__":
    unittest.main()

"""Exercise preparation, real train-derived loading, and experiment configs."""
from io import BytesIO
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from PIL import Image
import torch
from torch import nn

from globalsplat.compression import CodecConfig, ObservableLowRank1DCodec
from globalsplat.dataset.holdout import file_hash, read_holdout, ExcludingViewSampler
from globalsplat.dataset.re10k import ZPressorRE10KDataModule

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import run_nfcgs_architecture_pilot as runner
import prepare_nfcgs_architecture_pilots as prepare


def dataset_fixture(root):
    buf=BytesIO();Image.new("RGB",(640,360),(120,80,60)).save(buf,format="JPEG")
    image=torch.frombuffer(bytearray(buf.getvalue()),dtype=torch.uint8)
    cameras=torch.zeros(80,18);cameras[:,:4]=torch.tensor([.8,.8,.5,.5])
    transforms=torch.eye(4).repeat(80,1,1)
    transforms[:,0,3]=torch.arange(80)*.1
    cameras[:,6:]=transforms[:,:3].flatten(1)
    for split,names in (("train",["train_a","train_b","train_c"]),("test",["test_only"])):
        folder=root/split;folder.mkdir(parents=True)
        items=[{"key":name,"cameras":cameras,"images":[image]*80} for name in names]
        torch.save(items,folder/"000000.torch")
        (folder/"index.json").write_text(json.dumps({name:"000000.torch" for name in names}))


class TinyModel(nn.Module):
    """Use the real 736-D codec/QR boundary without an expensive image encoder."""
    def __init__(self,feature_codec=None,**kwargs):
        super().__init__()
        self.backbone=nn.Linear(8,8)
        self.feature_codec=ObservableLowRank1DCodec(CodecConfig.from_mapping(feature_codec)) if feature_codec else None
        self.gaussian_decoder=nn.Module()
        width=224 if self.feature_codec else 512
        for name,rows in (("geo_pos_readout",48),("geo_param_readout",112),("gate_readout",64)):
            setattr(self.gaussian_decoder,name,nn.Linear(width,rows))


class ArchitecturePilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_twelve_hydra_configs_and_training_validation_separation(self):
        for arm in runner.ARMS:
            for mode in ("train","validation","test"):
                with self.subTest(arm=arm,mode=mode):
                    argv=[mode,"--arm",arm,"--preparation","prep","--output","out"]
                    if mode!="train":argv += ["--checkpoint","trained.ckpt"]
                    command,_=runner.build_command(runner.parse_args(argv))
                    with initialize_config_dir(config_dir=str(ROOT/"config"),version_base=None):
                        cfg=compose(config_name="main",overrides=command[command.index("globalsplat.main")+1:])
                    config=CodecConfig.from_mapping(cfg.model.feature_codec)
                    from globalsplat.compression.architecture import pilot_config
                    self.assertEqual(config,pilot_config(arm))
                    self.assertTrue(cfg.architecture_entropy_determinism)
                    self.assertTrue(cfg.model.freeze_globalsplat)
                    if mode=="train":
                        self.assertTrue(cfg.dataset.train_holdout_manifest.endswith("holdout.json"))
                        self.assertEqual(cfg.trainer.limit_val_batches,0)
                        self.assertEqual(cfg.trainer.max_steps,50000)
                        self.assertEqual(cfg.optimizer.batch_size*cfg.trainer.devices*cfg.trainer.accumulate_grad_batches,8)
                    else:
                        self.assertTrue(cfg.test.actual_bitstream)
                        self.assertEqual((cfg.dataset.num_context_views,cfg.dataset.num_target_views),(12,8))
                        if mode=="validation":
                            self.assertTrue(cfg.dataset.eval_train_holdout_manifest.endswith("holdout.json"))
                            self.assertFalse(cfg.dataset.eval_all_scenes)
                        else:
                            self.assertIsNone(cfg.dataset.eval_index_path)

    def test_train_holdout_is_excluded_and_validation_reads_training_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"dataset";output=Path(directory)/"prep";output.mkdir()
            dataset_fixture(root)
            manifest=prepare.create_holdout(root,output,count=2)
            selected=set(manifest["scenes"])
            common=dict(mvsplat_root=str(ROOT/"third_party/ZPressor/mvsplat"),dataset_roots=[str(root)],
                        num_workers=0,persistent_workers=False,batch_size=1,
                        min_distance_between_context_views=45,max_distance_between_context_views=79,
                        context_gap_warm_up_steps=0,augment=False)
            train=ZPressorRE10KDataModule(**common,num_context_views=24,num_target_views=12,
                                        train_holdout_manifest=str(output/"holdout.json"))
            batches=list(train.train_dataloader())
            self.assertEqual({b["scene_info"]["scene"][0] for b in batches},{"train_a","train_b","train_c"}-selected)
            with self.assertRaisesRegex(ValueError,"default val loader"):
                train.val_dataloader()
            validation=ZPressorRE10KDataModule(**common,num_context_views=12,num_target_views=8,
                         eval_index_path=str(output/"validation_index.json"),eval_train_holdout_manifest=str(output/"holdout.json"))
            batches=list(validation.test_dataloader())
            self.assertEqual({b["scene_info"]["scene"][0] for b in batches},selected)
            for batch in batches:
                scene=batch["scene_info"]["scene"][0]
                self.assertEqual(batch["inputs"]["frame_ids"][0].tolist(),manifest["scenes"][scene]["context"])
                self.assertEqual(batch["targets"]["frame_ids"][0].tolist(),manifest["scenes"][scene]["target"])
            changed=json.loads((root/"test/index.json").read_text())
            changed[next(iter(selected))]="bad.torch"
            (root/"test/index.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,"index does not match"):
                read_holdout(output/"holdout.json",root)

    def test_actual_preparation_shares_fresh_modules_and_preserves_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);root=folder/"dataset";output=folder/"prepared"
            dataset_fixture(root)
            vanilla=folder/"vanilla.ckpt"
            model=TinyModel()
            torch.save({"state_dict":{f"model.{k}":v for k,v in model.state_dict().items()}},vanilla)
            original_holdout=prepare.create_holdout
            argv=["prepare","--vanilla",str(vanilla),"--dataset-root",str(root),"--output",str(output),
                  "--arms","score56_only","msh_only","two_path56"]
            with patch("globalsplat.model.globalsplat.GlobalSplat",TinyModel),patch.object(prepare,"create_holdout",side_effect=lambda r,o:original_holdout(r,o,count=2)),patch.object(sys,"argv",argv):
                prepare.main()
                before=file_hash(output/"preparation.json")
                prepare.main()
                self.assertEqual(file_hash(output/"preparation.json"),before)
            ckpts={arm:torch.load(output/f"{arm}.ckpt",weights_only=False) for arm in ("score56_only","msh_only","two_path56")}
            for arm in ("score56_only","msh_only"):
                a=ckpts[arm];b=ckpts["two_path56"]
                for key in set(a["state_dict"])&set(b["state_dict"]):
                    if a["state_dict"][key].shape==b["state_dict"][key].shape:
                        torch.testing.assert_close(a["state_dict"][key],b["state_dict"][key],atol=0,rtol=0)
            for arm,value in ckpts.items():
                self.assertEqual(value["architecture_pilot_initialization"]["signature"]["arm"],arm)
                self.assertNotIn("optimizer_states",value)
                smoke=subprocess.run([sys.executable,str(ROOT/"scripts/smoke_nfcgs_architecture_pilot.py"),
                                      "--checkpoint",str(output/f"{arm}.ckpt"),"--device","cpu"],
                                     capture_output=True,text=True,env={**os.environ,"PYTHONPATH":str(ROOT)})
                self.assertEqual(smoke.returncode,0,smoke.stdout+smoke.stderr)
                self.assertIn("PILOT_SMOKE_OK",smoke.stdout)

    def test_checkpoint_hook_keeps_pilot_provenance_and_handles_absent_score(self):
        from globalsplat.model.model_wrapper import GlobalSplatModule
        from globalsplat.compression.architecture import pilot_config
        codec=ObservableLowRank1DCodec(pilot_config("msh_only"))
        metadata={"signature":{"arm":"msh_only","holdout_sha256":"fixed"}}
        module=SimpleNamespace(model=SimpleNamespace(feature_codec=codec),architecture_pilot_initialization=metadata)
        checkpoint={"state_dict":{}}
        GlobalSplatModule.on_save_checkpoint(module,checkpoint)
        self.assertEqual(checkpoint["architecture_pilot_initialization"],metadata)
        self.assertEqual(checkpoint["feature_codec_config"]["codec_mode"],"msh_only")
        self.assertTrue(checkpoint["score_mean_offset_enabled"])
        restored=SimpleNamespace()
        GlobalSplatModule.on_load_checkpoint(restored,checkpoint)
        self.assertEqual(restored.architecture_pilot_initialization,metadata)


if __name__=="__main__":
    unittest.main()

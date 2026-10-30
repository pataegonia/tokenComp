"""Schedule diagnostics must reuse the original input cache and causal receiver."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import test_mechanism_runner as fixture

analysis, reporting = fixture.analysis, fixture.reporting


class ScheduleAnchorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def source(self, folder):
        manifest,args,cache,load=fixture.MechanismRunnerTests().fixture(folder)
        source_manifest=copy.deepcopy(manifest)
        source_manifest["points"]=source_manifest["points"][:4]
        source_manifest["reference"]="context_only_0p0128"
        args.scope="legacy3-anchors"
        args.prediction_ablation=False
        with patch.object(analysis.profile,"load_model",side_effect=load), \
             patch.object(analysis.profile,"cache_scenes",return_value=cache), \
             contextlib.redirect_stdout(io.StringIO()):
            analysis.run(source_manifest,args)
        source=args.output
        manifest["points"]=[p for p in manifest["points"] if p["config"]["token_order"]=="hilbert"]
        args=copy.deepcopy(args)
        args.scope="hilbert-schedule-anchors"
        args.input_run=source
        args.expected_input_fingerprint=json.loads((source/"inputs.json").read_text())["fingerprint"]
        args.output=Path(folder)/"schedules"
        return manifest,args,cache,load,source

    def test_same_cached_inputs_reference_repeatability_partial_resume_and_schedule_report(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest,args,cache,load,source=self.source(folder)
            before=(source/"inputs.pt").read_bytes()
            before_reference=(source/"points/hilbert_legacy3_0p0128/scene_0000.json").read_bytes()
            original=analysis.analyze_scene
            calls=0
            def stop(*a,**kw):
                nonlocal calls
                calls+=1
                if calls==2:
                    raise RuntimeError("interrupted")
                return original(*a,**kw)
            with patch.object(analysis.profile,"load_model",side_effect=load), \
                 patch.object(analysis.profile,"cache_scenes",side_effect=AssertionError("resampled")), \
                 patch.object(analysis,"fixed_score_relations",side_effect=AssertionError("out of scope")), \
                 patch.object(analysis,"analyze_scene",side_effect=stop), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,"interrupted"):
                    analysis.run(manifest,args)
            first=args.output/"points/hilbert_legacy3_0p0128/scene_0000.json"
            first_bytes=first.read_bytes()
            args.resume=True
            with patch.object(analysis.profile,"load_model",side_effect=load), \
                 patch.object(analysis.profile,"cache_scenes",side_effect=AssertionError("resampled")), \
                 patch.object(analysis,"fixed_score_relations",side_effect=AssertionError("out of scope")), \
                 patch.object(analysis,"analyze_scene",wraps=original) as analyzed, \
                 contextlib.redirect_stdout(io.StringIO()):
                analysis.run(manifest,args)
            self.assertEqual(analyzed.call_count,5)
            self.assertEqual(first.read_bytes(),first_bytes)
            self.assertFalse((args.output/"inputs.pt").exists())
            self.assertEqual((source/"inputs.pt").read_bytes(),before)
            self.assertEqual((source/"points/hilbert_legacy3_0p0128/scene_0000.json").read_bytes(),before_reference)
            metadata,results=reporting.read_run(args.output)
            self.assertEqual(metadata["identity"]["input_run"]["inputs"]["fingerprint"],args.expected_input_fingerprint)
            for rows in results.values():
                self.assertEqual([r["scene"] for r in rows],[r["scene"] for r in cache])
            for row in results["hilbert_legacy3_0p0128"]:
                self.assertTrue(row["source_reference_check"]["packet_sha256_equal"])
                self.assertEqual(row["source_reference_check"]["total_bytes_delta"],0)
            summary=reporting.report(args.output,args.output/"report")
            self.assertEqual(summary["reference_repeatability"]["identical_packet_scenes"],2)
            self.assertEqual([len(m["stages"]) for m in summary["models"].values()],[3,2,4])
            for label,model in summary["models"].items():
                self.assertEqual(sum(s["tokens"] for s in model["stages"]),65)
                self.assertIsNone(model["stages"][0]["anchor_normalized_l1"]["mean"])
                totals=[]
                for row in results[label]:
                    stages=row["measurements"]["trained"]["stages"]
                    totals.append(sum(g["raw_abs_error_sum"] for s in stages for g in s["groups"])/(65*7))
                self.assertAlmostEqual(model["whole"]["raw_mae"]["mean"],sum(totals)/len(totals))
            self.assertEqual(len(summary["candidate_minus_reference_whole_scene"]),2)
            for name in ("01_schedule_stages","02_whole_scene_cost"):
                for extension in ("png","pdf"):
                    self.assertGreater((args.output/"report"/f"{name}.{extension}").stat().st_size,1000)
            with self.assertRaises(FileExistsError):
                reporting.report(args.output,args.output/"report")

    def test_source_missing_wrong_fingerprint_or_changed_cache_never_resamples(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest,args,cache,load,source=self.source(folder)
            args.expected_input_fingerprint="wrong"
            with self.assertRaisesRegex(ValueError,"original 256"):
                analysis.input_run_info(args,manifest)
            args.expected_input_fingerprint=None
            args.seed=5
            with self.assertRaisesRegex(ValueError,"seed"):
                analysis.input_run_info(args,manifest)
            args.seed=0
            args.input_run=None
            with self.assertRaisesRegex(ValueError,"requires --input-run"):
                analysis.input_run_info(args,manifest)
            args.input_run=source
            cache[0]["inputs"]["texture"][0,0,0]+=1
            torch.save(cache,source/"inputs.pt")
            with patch.object(analysis.profile,"load_model",side_effect=load), \
                 patch.object(analysis.profile,"cache_scenes",side_effect=AssertionError("resampled")), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,"inputs.pt content changed"):
                    analysis.run(manifest,args)


if __name__=="__main__":
    unittest.main()

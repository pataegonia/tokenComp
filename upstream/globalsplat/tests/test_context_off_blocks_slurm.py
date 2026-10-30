"""Validate matched 1/4-stream submissions using fake sbatch and interrupted retries."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import test_token_order_context_slurm as existing

ROOT=Path(__file__).resolve().parents[1]
SCRIPT="submit_nfcgs_context_off_blocks_rd.sh"


@unittest.skipUnless(existing.BASH,"bash unavailable")
class ContextOffBlocksSlurmTests(unittest.TestCase):
    def fixture(self,directory):
        helper=existing.OrderContextSlurmTests()
        repo,env,log=helper.rd_fixture(directory)
        for name in ("score_context.py","checkpoint.py"):
            dest=repo/"globalsplat/compression"/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/"globalsplat/compression"/name,dest)
        fake=repo/"fake_bin/sbatch"
        text=fake.read_text()
        text=text.replace('EVAL_DIR=%s\\n','EVAL_DIR=%s SLICE=%s\\n')
        text=text.replace('"${EVAL_RUN_DIR:-}" >>', '"${EVAL_RUN_DIR:-}" "${SCORE_SLICE_CHANNELS:-16}" >>')
        fake.write_text(text,newline="\n")
        return helper,repo,env,log

    def test_six_hilbert_points_have_full_channel_streams_shared_parent_and_idempotent_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            helper,repo,env,log=self.fixture(directory)
            env.update(TOKEN_ORDER="nn_score",CONTEXT_SCHEDULE="legacy",SPATIAL_CONTEXT="true",
                       SCORE_SLICE_CHANNELS="16",STAGES="3",EVAL_RUN_DIR="wrong-eval")
            run=helper.rd_run(repo,env,script=SCRIPT)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            rows=log.read_text().splitlines()
            self.assertEqual(len(rows),12)
            state=repo/"logs/slurm/context_off_blocks_rd_20261017"
            records=[r.split('\t') for r in (state/"submissions.tsv").read_text().splitlines()[1:]]
            self.assertEqual({tuple(r[:4]) for r in records},{
                ("hilbert",schedule,str(stages),rate) for schedule,stages in (("single",1),("dyadic4",4))
                for rate in ("0.0064","0.0128","0.0256")})
            self.assertEqual(len({r[6] for r in records}),1)
            self.assertEqual(len({r[4] for r in records}),6)
            for j,record in enumerate(records):
                self.assertIn("_context_off_c56/",record[7])
                self.assertIn(f"--dependency=afterok:{record[4]}",rows[j*2+1])
                self.assertTrue(all("SLICE=56" in row and "ORDER=hilbert" in row for row in rows[j*2:j*2+2]))
            self.assertEqual((repo/"contexts.txt").read_text().splitlines(),["SPATIAL_CONTEXT=false"]*12)
            self.assertIn("SINGLE_STRINGS=1 DYADIC4_STRINGS=4",run.stdout)
            retry=helper.rd_run(repo,env,script=SCRIPT)
            self.assertEqual(retry.returncode,0,retry.stderr)
            self.assertEqual(log.read_text().splitlines(),rows)
            self.assertEqual(retry.stdout.count("REUSING TRAIN_JOB="),6)

    def test_preview_and_partial_eval_retry_do_not_duplicate_accepted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            helper,repo,env,log=self.fixture(directory)
            preview=helper.rd_run(repo,env,"--dry-run",script=SCRIPT)
            self.assertEqual(preview.returncode,0,preview.stdout+preview.stderr)
            self.assertIn("PREVIEW ONLY: 6 training points",preview.stdout)
            self.assertFalse(log.exists())
            self.assertFalse((repo/"logs").exists())
            env["FAKE_FAIL_CALL"]="2"
            failed=helper.rd_run(repo,env,script=SCRIPT)
            self.assertNotEqual(failed.returncode,0)
            state=repo/"logs/slurm/context_off_blocks_rd_20261017/hilbert_single_0p0064.state"
            self.assertEqual(state.read_text().split()[-2:],["90001","-"])
            env.pop("FAKE_FAIL_CALL")
            retry=helper.rd_run(repo,env,script=SCRIPT)
            self.assertEqual(retry.returncode,0,retry.stdout+retry.stderr)
            rows=log.read_text().splitlines()
            self.assertEqual(len(rows),13)
            self.assertEqual(sum("--dependency=" not in r for r in rows),6)
            self.assertIn("--dependency=afterok:90001",rows[2])

    def test_group_count_change_cannot_reuse_a_previous_submission_id(self):
        with tempfile.TemporaryDirectory() as directory:
            helper,repo,env,log=self.fixture(directory)
            env.update(TOKEN_ORDER="hilbert",SPATIAL_CONTEXT="false",CONTEXT_SCHEDULE="dyadic4",STAGES="4",
                       SCORE_SLICE_CHANNELS="16",SUBMISSION_STATE_FILE="same.state")
            first=helper.rd_run(repo,env,script="submit_nfcgs_order_context.sh")
            self.assertEqual(first.returncode,0,first.stderr)
            before=log.read_bytes()
            env["SCORE_SLICE_CHANNELS"]="56"
            changed=helper.rd_run(repo,env,script="submit_nfcgs_order_context.sh")
            self.assertNotEqual(changed.returncode,0)
            self.assertIn("mismatched submission state",changed.stderr)
            self.assertEqual(log.read_bytes(),before)

    def test_settings_and_real_cli_emit_one_and_four_stream_architectures(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location("blocks_cli",ROOT/"scripts/run_nfcgs.py")
        runner=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        for schedule,stages in (("single",1),("dyadic4",4)):
            for mode in ("train","eval"):
                argv=[mode,"--checkpoint","parent.ckpt","--score-path","minimal","--token-order","hilbert",
                      "--no-spatial-context","--score-context-schedule",schedule,"--score-slice-channels","56"]
                if mode=="train":
                    argv.append("--allow-score-path-conversion")
                args=runner.parse_args(argv)
                command,_,_=runner.build_command(args)
                self.assertIn(f"model.feature_codec.score_spatial_stages={stages}",command)
                self.assertIn("model.feature_codec.score_slice_channels=56",command)
                self.assertIn("model.feature_codec.score_spatial_context=false",command)
                self.assertIn("model.feature_codec.score_channel_context=false",command)
        with self.assertRaises(SystemExit):
            runner.parse_args(["eval","--score-context-schedule","single"])


if __name__=="__main__":
    unittest.main()

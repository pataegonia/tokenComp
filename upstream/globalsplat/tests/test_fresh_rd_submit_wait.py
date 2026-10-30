"""Login-node retries must never hide other errors or request a new GPU job."""
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import wait_nfcgs_provisional_submission as wait


def state_file(tmp_path):
    path=tmp_path/"state.json"
    value=dict(version="nfcgs_fresh_rd_v1",inputs=dict(root="/server/custom_sweep",dataset="/data/re10k",
        preparation="/server/preparation",native_checkpoint="/server/native.ckpt"),
        provisional_stage_b=dict(status="provisional"),jobs={"accepted":dict(id="461656")})
    path.write_text(json.dumps(value),encoding="utf-8")
    return path


def test_quota_then_success_retries_controller_without_changing_state(tmp_path,capsys):
    path=state_file(tmp_path);before=path.read_bytes()
    responses=[subprocess.CompletedProcess([],1,"REUSING=461656\n","sbatch: error: QOSMaxSubmitJobPerUserLimit\n"),
        subprocess.CompletedProcess([],0,"ACCEPTED msh=470000\n","")]
    with patch.object(wait.subprocess,"run",side_effect=responses) as call,patch.object(wait.time,"sleep") as sleep:
        assert wait.wait_for_submission(path,600)==0
    assert call.call_count==2
    sleep.assert_called_once_with(600)
    command=call.call_args.args[0]
    assert command[1]==str(ROOT/"scripts/submit_nfcgs_fresh_rd.py")
    assert "--provisional" in command and "/server/custom_sweep" in command
    assert not any(x in command for x in ("sbatch","scancel"))
    assert path.read_bytes()==before
    assert "PROVISIONAL_SUBMISSION_COMPLETE" in capsys.readouterr().out


def test_non_quota_error_stops_without_sleep_or_retry(tmp_path,capsys):
    path=state_file(tmp_path)
    with patch.object(wait.subprocess,"run",return_value=subprocess.CompletedProcess([],1,"","runtime changed\n")) as call, \
            patch.object(wait.time,"sleep") as sleep:
        assert wait.wait_for_submission(path,600)==1
    call.assert_called_once();sleep.assert_not_called()
    assert "non_quota_error=True" in capsys.readouterr().err


def test_selected_budget_branch_is_not_resubmitted(tmp_path):
    path=state_file(tmp_path);value=json.loads(path.read_text())
    value["active_budget_branch"]=dict(budget=120000)
    path.write_text(json.dumps(value))
    with patch.object(wait.subprocess,"run") as call:
        with pytest.raises(ValueError,match="extended budget branch"):
            wait.wait_for_submission(path,600)
        call.assert_not_called()


def test_invalid_interval_never_launches_controller(tmp_path):
    with patch.object(wait.subprocess,"run") as call:
        with pytest.raises(ValueError,match="positive"):
            wait.wait_for_submission(state_file(tmp_path),0)
        call.assert_not_called()

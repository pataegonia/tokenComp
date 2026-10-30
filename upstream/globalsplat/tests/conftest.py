"""Make the repo root importable so the tests can ``import globalsplat``
regardless of where pytest is launched from."""
import sys
import os
import stat
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

WINDOWS_REPLACE_RETRIES = 0


@pytest.fixture(autouse=True)
def opt_in_windows_temp_replace_retry(monkeypatch, request):
    """Test-host workaround for transient Windows file sharing, never production IO.

    Opt-in only; restricted to this test's writable temporary files. Read-only
    checkpoint protection and persistent permission failures remain failures.
    """
    if (os.name != "nt" or os.environ.get("NFCGS_WINDOWS_REPLACE_RETRY_TESTS") != "1"
            or "tmp_path" not in request.fixturenames):
        return
    root=request.getfixturevalue("tmp_path").resolve()
    original=os.replace
    def replace(source,destination,*args,**kwargs):
        global WINDOWS_REPLACE_RETRIES
        for attempt in range(11):
            try:return original(source,destination,*args,**kwargs)
            except PermissionError as error:
                target=Path(destination)
                if (attempt==10 or getattr(error,"winerror",None) not in (5,32)
                        or not target.resolve().is_relative_to(root)
                        or target.exists() and not target.stat().st_mode & stat.S_IWUSR):
                    raise
                WINDOWS_REPLACE_RETRIES+=1
                time.sleep(.02)
    monkeypatch.setattr(os,"replace",replace)

# Checkpoint ENOSPC 후속 확인 — 2026-10-15

10월 15일 09:03 다운로드 로그에서 451821(quarter2 .0064, 약10k), 451825(quarter2 .0256, 약5k), 451827(dyadic4 .0064, 약5k)이 ariel-v8의 checkpoint 쓰기 중 `OSError: [Errno 28] No space left on device`로 중단됐다. 이는 GPU OOM 판정이 아니다. 저장 완료나 checkpoint 무결성은 아직 확인하지 못했다.

이후 사용자가 마스터와 ariel-v8 양쪽에서 확인한 출력:

- 대상 `outputs`는 두 노드 모두 Ceph `/ceph_data`이며 205T 중39T 사용, 166T 여유다.
- `df -i`는 IUsed/IFree/IUse%를 `-`로 표시하므로 inode 소진 여부를 판정하지 못한다.
- 실패한 세 `version_0` 경로와 조상26개 디렉터리의 xattr 조회에서 `/ceph_data/clue9986`에 다음 값이 있다.
  - `ceph.quota.max_bytes = 1429365116108`
  - `ceph.dir.rbytes = 785385164873`
  - `ceph.quota.max_files = 0`
  - `ceph.dir.rfiles = 2812404`
- 바이트 quota는 약1331.2GiB 중731.4GiB 사용으로 약599.8GiB가 남아 있다. `max_files=0`은 파일 수 quota가 설정되지 않았다는 뜻이다. 다른 경로의 quota 출력이나 조회 오류는 보고되지 않았다.

따라서 현재 전체 Ceph 용량 또는 확인한 디렉터리 quota의 초과 증거는 없다. 이 출력은 실패 시각의 상태를 재현하지 않는다. Ceph backend/OSD full 또는 client 문제는 후속 확인 대상이며, 현재 원인으로 확정하지 않는다. Ceph는 특정 OSD가 full일 때 해당 pool의 쓰기를 차단할 수 있다([공식 OSD troubleshooting](https://docs.ceph.com/en/reef/rados/troubleshooting/troubleshooting-osd/)). Quota의 0 값 해석은 [CephFS quota 문서](https://docs.ceph.com/en/quincy/cephfs/quota/)에 따른다.

## 추가 확인: 최종 경로 쓰기 성공과 시스템 임시 경로

사용자가 ariel-v8의 quarter2 .0064 `version_0`에서 확인한 파일은 `last.ckpt` 398,082,611B와 `step000005000.ckpt` 398,082,419B다. 10k 파일은 해당 glob 출력에 없다. 같은 디렉터리의 64MiB 실제 쓰기·flush·fsync는 성공했다. 파일 크기만으로 checkpoint 정상 로드나 optimizer 상태를 검증한 것은 아니다.

사용자가 제공한 filtered dmesg에는 9월14일 mount 관련 기록, 9월20일과10월12일의 inode 조회 실패만 있다. 10월15일의 실패 시각을 설명하는 기록은 이 출력에 없다.

저장 구현을 재확인하면서 최종 Ceph 경로만 조회한 진단의 누락을 발견했다. Lightning `_atomic_save`는 `fs.transaction`으로 저장하고, fsspec `LocalFileOpener._open`의 transaction 분기는 `tempfile.mkstemp()`로 시스템 임시 경로에 먼저 쓴다. commit 때 최종 checkpoint 경로로 옮긴다. 따라서 traceback의 `fsspec local write`가 최종 Ceph 파일에 대한 직접 쓰기라고 해석해서는 안 된다. 임시 경로는 Python `tempfile.gettempdir()`로 확인해야 한다.

이 구조는 [Lightning 2.4 소스](https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/fabric/utilities/cloud_io.py)와 [fsspec 공식 소스](https://filesystem-spec.readthedocs.io/en/latest/_modules/fsspec/implementations/local.html)에서 확인했고, 로컬 설치 fsspec 2025.9.0의 실제 소스도 같은 동작이다. 서버 설치 버전의 소스와 기본 temp 경로 용량은 아직 사용자 출력으로 확인하지 않았다. 현재 유력한 가설은 ariel-v8의 시스템 임시 저장소 부족이며, 확정 원인은 아니다. 작은 최종경로 probe 성공은 이 임시 저장소 경로를 시험한 것이 아니다.

재제출용 `train_nfcgs_order_context_recovery.slurm`은 Python 실행 전에 `TMPDIR=/ceph_data/${USER}/.tmp/nfcgs/job_${SLURM_JOB_ID}`를 생성·export하도록 수정했다. `NFCGS_TMP_ROOT`로 root를 바꿀 수 있다. DataLoader/multiprocessing의 Unix socket 경로 제한을 고려해 긴 run output 하위 경로 대신 짧은 사용자 Ceph 경로를 쓴다. 다른 job의 임시 파일을 지우지 않으며, 이 변경은 새로 제출하는 job에만 적용된다. 서버 업로드·실행은 아직 하지 않았다.

로컬 검증: `python -X utf8 -m unittest discover -s tests -p test_sort_context_recovery.py -v`의 9개 테스트가 통과했다. Git Bash 실행에서 restart/resume 양쪽의 학습 인자와 출력 경로를 유지하고, checkpoint 복사와 학습 subprocess가 지정한 TMPDIR를 상속함을 확인했다. `artifacts/check_fsspec_checkpoint_staging.py`는 새 Python subprocess의 실제 fsspec transaction으로 기본 temp와 TMPDIR로 지정한 temp가 각각 선택되고 commit 결과가 같은지 검증했다. 이는 로컬 설치 버전의 검증이며 서버 temp 용량이나 서버 checkpoint 무결성을 대신하지 않는다.

기존 recovery helper는 최신 checkpoint 후보가 깨졌으면 검증 중 중단되며, 종료된 train에 연결된 eval이 여전히 PENDING인 경우에도 중단된다. 따라서 helper를 그대로 재실행하면 자동으로 유효한 이전 checkpoint를 건너 찾아 복구할 것이라고 가정하지 않는다. 이번 후속 확인에서 서버 job·checkpoint·quota 설정을 변경하지 않았다.

## 서버 검증 결과와 재제출 경로

이후 사용자가 `globalsplat` 환경에서 확인한 `PYTHON_TMPDIR`는 `/tmp`다. 서버의 `_atomic_save` 소스에도 `fs.transaction`과 `f.write(bytesbuffer.getvalue())`가 있어 조사한 Lightning 저장 경로와 일치한다. 이 출력에는 `/tmp`의 `df` 결과가 포함되지 않아 임시 디스크의 실제 공간 부족 자체는 여전히 확정하지 않는다.

서버 helper의 `latest_checkpoint` 검증 결과:

| 실험 | 확인 결과 | 복구 계획 |
|---|---|---|
| Context-only quarter2 .0064 | `recovery_job_451821/.../step000005000.ckpt`, step5000 정상 검증 | 모델·optimizer·scheduler 전체 상태로 5k부터 총50k까지 resume |
| Context-only quarter2 .0256 | `(None, 0)` | 원래 공통 FullSplit .0256 부모 가중치에서 새50k restart |
| Context-only dyadic4 .0064 | `(None, 0)` | 원래 공통 FullSplit .0256 부모 가중치에서 새50k restart |

사용자의 `squeue -j 451822,451826,451867` 출력에는 헤더만 있어 세 eval은 조회 당시 큐에 없다. 취소·실패·완료 중 어느 accounting 상태인지는 이 출력만으로 정하지 않는다. recovery helper가 재실행 시 `sacct`와 `squeue`로 최종 확인한다. 따라서 stale pending eval로 인한 기존 사전검사 중단은 현재 출력에서는 관찰되지 않는다.

서버의 `scripts/slurm/train_nfcgs_order_context_recovery.slurm`에 수정본을 반영한 뒤 master에서 기존 recovery helper를 재실행할 수 있다. `--dry-run`으로 계획을 출력하고 성공하면 실제 실행한다. 기존 recovery state를 유지하며, 살아 있거나 완료된 작업은 재사용하고 필요한 train/eval만 새로 연결한다. 새 job stdout의 `CHECKPOINT_TMPDIR=/ceph_data/clue9986/.tmp/nfcgs/job_<ID>`로 새 경로 적용을 확인한다. 서버 파일 반영 및 새 제출 결과는 아직 사용자로부터 받지 않았다.

## 사용자 제공 재제출 접수 결과

사용자가 수정본 marker 확인 후 dry-run과 실제 recovery를 실행했다. dry-run 및 실제 PLAN은 일치했고, 아래 train3개·eval3개가 접수됐다.

| 실험 | 이전 train / eval | 새 train / eval | 제출된 복구 방식 | 다음 주기적 checkpoint |
|---|---|---|---|---|
| Context-only quarter2 .0064 | 451821 / 451822 | 453064 / 453065 | step5000 전체 상태 resume, 목표50k | step10000 |
| Context-only quarter2 .0256 | 451825 / 451826 | 453066 / 453067 | 같은 공통 부모 weights-only restart, 새50k | step5000 |
| Context-only dyadic4 .0064 | 451827 / 451867 | 453068 / 453069 | 같은 공통 부모 weights-only restart, 새50k | step5000 |

나머지12개 point는 기존 train/eval ID로 `MODE=reuse`였다. helper는 서버의 `logs/slurm/sort_context_recovery_20261014.json`에 새 ID를 저장했다고 출력했다. 로컬 `slurm/sort_context_recovery_20261014.json`은 이전 다운로드이므로 이 접수 결과를 반영한 서버 원본인 것처럼 수동 수정하지 않았다.

접수 결과는 실행 시작·checkpoint 복원 완료·쓰기 오류 해결을 확인한 것이 아니다. 다음에 새 로그를 내려받으면 세 작업의 `CHECKPOINT_TMPDIR`가 Ceph 경로인지, 453064의 실제 복원 step이5000인지, 위 다음 checkpoint가 정상 저장되는지 확인한다. 후속 eval은 복구 helper에서 새 train의 `afterok` 의존성으로 제출하는 경로다. 새 job의 실제 queue 상태와 runtime 로그는 아직 받지 않았다.

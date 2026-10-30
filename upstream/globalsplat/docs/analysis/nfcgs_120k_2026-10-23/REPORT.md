# 120k 설계 구현과 로컬 검증 — 2026-10-23

재개 경계의 checkpoint 재저장 방지, 읽기 전용 hard link, 엄격한 `--accept-resaved-60k` 분기,120k 단일 Wave 2 job을 구현했다. quantile step 저장·복원 및 갱신 조건은 기존 코드가 이미 처리하므로 바꾸지 않았다. 서버의 학습·checkpoint·Slurm job은 이 작업에서 변경하지 않았다. 실행 절차는 [새 runbook](../../NFCGS_120K_RUNBOOK_2026-10-23.md), 서버 반영 대상은 [파일 목록](../../nfcgs_120k_20261023_files.txt)이다. 최초8개에 holdout 복구 파일1개를 추가했다.

## 실제 서버 dry-run 후 추가한 holdout 복구

사용자가 붙인 실제 출력에서 core12개가 모두 `resaved_after_first_microbatch` 조건을 통과했다. 다음 `h_on/lambda0/s0`는 평가의 checkpoint SHA가 현재60k SHA와 달라 중단됐다. S56/MSH는 그 뒤이므로 이 출력으로 판정할 수 없다. 기존 B 중단·120k 분기 생성 전의 실패이며, 모니터 종료 후 늦은 λ0 Stage B 시작으로 경계 재저장이 발생한 상황에 부합한다.

`scripts/repair_nfcgs_stage_a_holdout.py`를 추가했다. 원 state·run manifest·checkpoint·learner job은 보존하고15개를 사전 검사하여 stale60k 평가만1GPU allocation에서 다시 실행한다. 원본의 완료 평가를 보관한 뒤, 현재 코드와 기존 protocol의 차이가 승인된 runtime3개에 한정되는 별도 audit manifest를 사용한다. 현재 checkpoint SHA에 묶인 실제 재평가와 과거 평가의 장면별 frame ID·MSE·LPIPS·actual bytes가 같아야 Gate 1을 다시 만든다. 평가 실패 후 재시도 및 불일치 후 다음 실행에서도 거부되는 경로를 포함한다. SHA를 relabel하거나50k로 자동 전환하지 않는다. 신규 GPU 사전 테스트가 아니라 Gate 1에 필요한 실제 holdout 평가를 복구한다.

[복구 검사 결과](pytest_holdout_repair.xml): 신규9개와 기존 identity6개, 총15개 CPU 검사 통과. 실제 GPU 평가 자체는 mock했으며, 실제 launcher의 audit manifest 재구성·제한된 parent protocol·원 파일 보존·strict120k preflight와의 연결을 검사했다. 서버 GPU 평가는 사용자가 `--submit`으로 실행해야 한다. 아래 기존118개 검증 기록은 원 구현에 대한 기록으로 유지한다.

## 검증 결과

관련 **118개 고유 검사에 대해 통과 결과를 확보했다.** 전체 실행은114개 통과,3개 Windows 경로 길이 오류였으며, 그3개를 짧은 임시 경로로 재실행해 모두 통과했다. 이후 수정한 실제 CLI dry-run·분기 의존성·Wave 2·선택4개 취소 경로5개도 다시 통과했다. milestone 반복 횟수 변경을 거부하는 검사를1개 추가하고 관련4개 경로를 다시 통과했다. 원 보고서를 보존하고 test별 최신 결과를 합친 [118개 결과](pytest_effective.xml)를 생성했다. 이는118개가 한 번의 실행에서 모두 통과했다는 뜻은 아니다.

- [전체 실행](pytest.xml):117개,114개 통과,3개 Windows hard-link 경로 길이 실패.
- [단축 경로 재실행](pytest_retry.xml):해당3개 통과.
- [최종 수정 경로 재검사](pytest_final_changes.xml):5개 통과.
- [엄격한 milestone 및 부모 복원·CLI 재검사](pytest_strict_milestones.xml):4개 통과.
- [검증·코드 SHA 기록](verification.json).

테스트는 Lightning2.3.0과 로컬 CPU 환경에서 실행했다. TensorBoard2.20.0과 absl-py는 workspace의 별도 dependency 폴더에만 내려받았다. 설치 패키지와 기본 임시 폴더의 계정 간 권한 차이 때문에 한 실행은 fixture setup에서 중단됐으며, 작업 폴더의 새 임시 경로로 재실행했다. Windows 임시 JSON의 일시적 `os.replace` 공유 위반은 명시적 환경 변수로 켠 test-only 재시도로 처리했다. production IO·읽기 전용 checkpoint 보호는 변경하지 않았다. py_compile, train Slurm의 bash syntax, 수정된 tracked 파일의 diff whitespace 검사도 통과했다.

## 주요 검증 범위

작은 실제 Lightning 학습을 N step에서 저장한 뒤 같은 저장 주기로 재개했다. 첫 microbatch 뒤 경계 파일의 추가 저장이 없고, SHA·mtime이 유지되며 이후2N 저장 후에도 원본과 분기 hard link가 유지됨을 확인했다. 기존 full-state 재개 sample/RNG/LR 검사3개도 포함한다.

재저장본 fixture는 가중치를 유지하고 processed를 정확히239979→239980으로 바꿨다. 수락과 다음 거부 조건을 검증했다: 플래그 없음, LR 감소, milestone 변경, step·epoch·scheduler last_epoch 불일치, processed 원값/+2, 평가 SHA 불일치, superseded MSE·LPIPS·bytes 불일치, quantile step 불일치, loader state 누락. 실패 시 scancel·sbatch 전에 중단한다.

실제 데이터를 읽는 공개 CLI 경로의 dry-run을 모의 데이터로 실행했다. [출력](dry_run_fixture.txt)에는 run별 source 판정, 모의 중단 ID, 파일 연결, B120k/test/final monitor 계획이 있다. 이 파일은 서버 checkpoint의 판정 결과가 아니다. 서버 실제 입력은 업로드 후 runbook의 동일 dry-run 명령으로 검증한다.

runtime migration 검사에서는 원 sweep manifest와 코드 hash를 보존하고 새 분기의 현재 protocol을 생성했다. 선언한3개 runtime 파일 외의 코드나 recipe가 달라지면 거부한다. 고정한60k 부모 SHA와 protocol을 확인한 뒤 모델의 실제 on_load 경로에서 LR milestone을84k/108k로 바꾸고 LR1e-4와 저장된 quantile step을 유지함을 확인했다.

Wave 2는20개0→120k train+test allocation과 최종 monitor1개다. 앞16개와 뒤4개의 순서·의존성을 확인했다. 선택4개 취소는 core16개를 보존하며, monitor inventory를16개로 기록하고 의존성을 먼저 해제한 뒤4개만 취소한다. 따라서 사이트의 invalid-dependency 자동 취소 정책에서도 monitor를 잘못 취소하지 않는다. TensorBoard LR event 검사에서는84k 전의1e-4와 조기 감소 오류를 구분했다.

## 구현상 확인 사항

설치된 Lightning2.3.0의 `_atomic_save`는 `fsspec.open(filepath, "wb")`에 직접 쓴다. 임시 파일 rename 보호를 가정하지 않았다. [설치 소스 확인](lightning_checkpoint_io.txt). `on_train_start`의 저장 step 동기화와 읽기 전용 hard link를 함께 사용한다.

새 분기는 검증이 끝난 과거 Stage A/monitor ID에 새 `afterok`를 연결하지 않는다. 완료한 ID는 `MinJobAge` 이후 새 의존성에 사용할 수 없기 때문이다. [Slurm 공식 문서](https://slurm.schedmd.com/sbatch.html#OPT_dependency). core16개 완료를 확인한 optional 취소에서도 새 의존성을 다시 걸지 않고 기존 의존성을 비운다. [scontrol 설명](https://slurm.schedmd.com/scontrol.html).

Wave 1의 B15개+test15개+monitor1개는31개이므로30-job 한도에서 마지막 접수가 지연될 수 있다. 기존 supervisor에 `--restart-120k`를 추가했다. QOS 제출 한도만 재시도하고 accepted ID를 재사용한다. 다른 오류는 중단한다. Wave 2는 test를 같은 allocation 끝에 실행하여21개로 제출한다.

## 아직 서버에서 수행하지 않은 항목

첨부된 dry-run의 core12개 통과는 실제 서버 출력에 근거한다. λ0 재평가, 나머지2개 checkpoint의 조건, 최종 Slurm ID identity는 복구 후 서버 dry-run이 판정한다. 사용자가 알려준3개 평가 차이0을 로컬에서 실제 파일로 재검증했다고 주장하지 않는다. 현재 상태에서120k 접수·기존 B 취소를 수행하지 않았다. 시작 후 SHA·LR은 제공한 `verify_nfcgs_fresh_120k.py`로70k 저장 전후에 확인한다.

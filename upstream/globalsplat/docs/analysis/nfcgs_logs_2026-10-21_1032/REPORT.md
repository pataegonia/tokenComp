# 2026-10-21 최신 fresh Wave 1 로그 확인

workspace 루트 `slurm/`에 10:32 무렵 내려받은 파일을 확인했다. 실시간 Slurm 조회·서버 제출·취소는 수행하지 않았다.

## 현재 관찰

- Wave 1 학습 15개 `460311~460325`와 monitor `460326`의 접수 ID가 새 state에 있다. 모든 학습의 dependency는 비어 있어 Gate 0에 막힌 상태가 아니다.
- **학습 15개 모두 같은 `FileNotFoundError`로 모델 학습 시작 전에 종료됐다.** `validate_preparation → read_holdout`에서 `/data3/local_datasets/re10k/train/index.json`을 읽지 못했다. 학습 progress·optimizer step·checkpoint 생성 기록은 없다.
- monitor는 `MONITOR_PENDING phase=A evaluated=0`이며 stderr는 비어 있다. 실제 queue 상태나 할당 노드는 이 파일로 확정하지 않는다.
- 다운로드 state에 기록된 42개 코드 hash가 수정 전 로컬 파일과 모두 같았다. 파일 전달 누락이나 서로 다른 코드 버전 때문에 생긴 실패는 아니다.

기존 정상 architecture/order-context 학습·평가 Slurm은 `--exclude=ariel-v7`을 사용했으나 새 fresh Slurm에는 이 조건이 빠져 있었다. 같은 제외 조건을 복원했다. 현재 로그에는 실제 NodeList가 없어 15개가 ariel-v7에 배정됐다고 단정하지 않는다. 복구 도구는 실행 시 sacct에서 실제 실패 노드를 확인하고 새 접수의 exclude 목록에도 포함한다. 데이터 경로를 임의의 다른 위치로 바꾸지 않는다.

## 복구 동작

`scripts/recover_nfcgs_fresh_start_20261021.py`는 기존 state를 보존하며 다음 경우만 다시 접수한다.

1. 정확히 Wave 1의 15개 run이고 Stage B/test는 아직 접수되지 않았다.
2. 원래 job의 Slurm 상태가 실패·취소·시간 초과 등 종료 상태다.
3. stderr가 실제 학습 전의 train index 누락을 증명한다.
4. 해당 run의 `run.json`, checkpoint, 학습 invocation이 없다. 이미 학습이 시작된 run은 fresh 재시작을 거부한다.
5. 코드 차이는 이번 제출·노드 배치 수정으로 한정된다. codec·학습 설정·초기화는 바꾸지 않는다.

기존 접수 ID는 각 job history에 보존하고 새 ID를 즉시 기록한다. 같은 복구 명령을 다시 실행하면 이미 교체한 작업은 재사용한다. 같은 run 폴더와 초기화로 step 0부터 시작하며, 없는 `last.ckpt`를 요구하는 resume를 사용하지 않는다.

`--restart-monitor`를 주면 기존 monitor 한 개만 종료한 뒤 같은 holdout monitor를 새 배치로 접수한다. 종료 상태를 확인한 뒤 이전 monitor의 정확한 lock 파일만 정리한다. 학습 job·Gate 0 job을 취소하지 않는다. Gate 0를 재실행하거나 학습 선행 조건으로 추가하지 않는다.

## 서버에 반영할 파일

- `scripts/recover_nfcgs_fresh_start_20261021.py`
- `scripts/submit_nfcgs_fresh_rd.py`
- `scripts/slurm/train_nfcgs_fresh_rd.slurm`
- `scripts/slurm/eval_nfcgs_fresh_rd.slurm`
- `scripts/slurm/monitor_nfcgs_fresh_rd.slurm`
- `scripts/slurm/fresh_rd_environment.sh`
- `scripts/slurm/prepare_nfcgs_architecture_pilots.slurm`

```bash
cd /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
python scripts/recover_nfcgs_fresh_start_20261021.py --restart-monitor
```

일반 `--wave 1` 재실행은 실패 ID를 재사용하므로 이 상황의 재접수 방법이 아니다. state를 삭제하지 않는다.

## 로컬 검증

- fresh 및 배치 복구 검사 **64개 통과**. 실제 재시작/취소는 mock으로 검증했다.
- 15개 교체 후 재실행이 추가 접수를 만들지 않음, 실제 학습 흔적이 있거나 원래 job이 active이면 교체 거부, monitor 재시작의 명시성·중복 방지를 확인했다.
- 실제 다운로드 state와 stderr 15개로 dry-run을 수행했다. 원본 다운로드 state는 변경하지 않았다. 이 dry-run은 서버의 실제 filesystem·sacct를 검사하거나 job을 접수한 결과가 아니다.

[검증 요약 및 원본 hash](verification.json), [기존 state 복구 dry-run](recovery_dry_run.txt).

# 2026-10-21 로그 확인 및 본학습 선행 조건 해제

근거는 workspace 루트 `slurm/`의 다운로드 파일이다. 서버의 실시간 squeue/sacct 조회나 job 제출·취소는 수행하지 않았다.

## 새 fresh 작업

다운로드한 `slurm/fresh_rd/state.json`에는 진단 6개만 접수되어 있고 본학습 `runs`는 비어 있다.

| 작업 | Job | 확인한 결과 |
|---|---|---|
| 초기화 공통 tensor 검사 | 459887 | `GATE0_PASS=initialization` |
| fresh QR 경계 | 459888 | 수치 허용치 실패. 상세 `complete.json`은 다운로드되지 않아 차이의 크기·원인은 미확정 |
| eval forward / 실제 coder 비교 | 459889 | 수치 허용치 실패. 상세 JSON 미확인 |
| native 기준 평가 | 459890 | codec OFF인데 양의 rate lambda가 남은 실행 명령 오류로 생성자에서 중단 |
| 연속/분할 재개 | 459891 | 연속 2k·분할 1k 및 2k 재개 학습은 종료. trace 비교 다음 단계에서 가변 길이 CDF buffer(185/193)를 빼다가 비교 코드 중단 |
| 진단 집계 | 459892 | 접수 ID와 afterok 의존성만 확인. 로컬 stdout 없음 |

경계/forward 실패를 정상으로 판정하거나 허용치를 올리지 않았다. 학습을 막는 선행 조건에서 진단을 분리했다. native 명령은 rate lambda를 0으로 고쳤다. 재개 결과 비교는 entropy buffer의 길이 차이를 별도 기록하고 모델 parameter의 shape 불일치는 계속 오류로 처리한다.

## 바로 학습을 접수하는 변경

- `--wave 1`은 기본적으로 Gate 0 결과를 읽지 않는다. 선행 검사는 명시적인 `--require-gate0` 옵션으로만 적용한다.
- 옛 Gate 0 GPU ID가 lane에 있어도 본학습의 afterok 의존성으로 연결하지 않는다.
- 아직 본학습이 없는 diagnostic-only state는 기존 ID와 이전 코드 hash를 보존하고 새 코드로 갱신한다. state 삭제·새 이름으로 중복 제출할 필요가 없다.
- 기본 monitor는 본학습 holdout 곡선만 평가한다. native/warm-start 기준 평가는 선택적인 `--baselines`다.
- full test의 scene/frame 기준도 native 진단 완료를 요구하지 않는다. 기존 native 또는 warm-start frame을 재사용하거나 최초 완료 fresh 평가에서 frame 집합을 고정하며, 품질 수치는 기준 선택에 사용하지 않는다.
- Stage A 60k 이후의 공통 budget 선택 및 Wave 2의 holdout 검토 절차는 유지한다.

서버 반영 파일은 `scripts/submit_nfcgs_fresh_rd.py`, `scripts/run_nfcgs_fresh_rd.py`, `scripts/monitor_nfcgs_fresh_rd.py`, `scripts/check_nfcgs_fresh_rd_gate0.py`다. 이전 Slurm 파일을 그대로 쓴다.

```bash
cd /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
bash scripts/slurm/submit_nfcgs_fresh_rd.sh --wave 1
```

## 검증

`tests/test_fresh_rd.py`: **60 passed**. 기존 다운로드 state를 복사해 실행한 별도 dry-run에서도 진단 ID 459887~459892 전부 보존, 본학습 15개+monitor 1개만 추가, 진단에 대한 학습 의존성 없음, 원본 다운로드 state 변경 없음을 확인했다. [실제 state의 오프라인 재현](wave1_existing_state_dry_run.txt).

## 기존 H+D4 OFF16 완료

복구 학습458783 stderr의 `max_steps=50000` 종료와 평가457100의 최종 6991장 처리·지표를 확인했다. 치명적 평가 오류는 없다. nominal progress 분모7286과 실제6991을 구분한다.

| 모델 | PSNR | SSIM | LPIPS | B/scene | KiB/scene |
|---|---:|---:|---:|---:|---:|
| H+D4 OFF16 .0128, 기본 seed | 23.6338 | .7338 | .2695 | 45690.114147 | 44.6192521 |

기존 같은 seed ON16의 23.7021dB·37.17881KiB와 비교하면 같은 λ에서 ON은 약16.6754% 작은 bytes와 +.0683dB다. 이 결과는 기존 warm-start 계보이며 새 fresh 실험 결과로 합치지 않는다.

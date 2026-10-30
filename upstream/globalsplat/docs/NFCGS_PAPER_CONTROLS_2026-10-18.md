# 다음 제출: L3 대조군 끝점과 D4 OFF16

실험 설계안의 기존 codec 대조군 7개를 제출하는 launcher를 준비했다. 학습 7개와 각각 성공 후 실행되는 actual full-test 평가 7개다. 원격 제출은 아직 하지 않았다.

## 제출 목록

| 순서 | 모델 | λ | Score strings | 목적 |
|---|---|---:|---:|---|
| 1 | None+legacy3, 공간 Context ON | .0032 | 12 | 원래 순서에서 Context 효과 |
| 2 | Hilbert+legacy3, 공간 Context OFF12 | .0032 | 12 | 정렬 단독 효과 |
| 3 | None+legacy3, 공간 Context OFF12 | .0032 | 12 | 정렬·공간 Context 모두 OFF |
| 4 | None+legacy3, 공간 Context ON | .0512 | 12 | 원래 순서에서 Context 효과 |
| 5 | Hilbert+legacy3, 공간 Context OFF12 | .0512 | 12 | 정렬 단독 효과 |
| 6 | None+legacy3, 공간 Context OFF12 | .0512 | 12 | 정렬·공간 Context 모두 OFF |
| 7 | **Hilbert+dyadic4, 공간 Context OFF16** | **.0128** | **16** | 최종 D4의 단계·그룹을 맞춘 공간 Context 비교 |

전부 rank56, MSH ON, 채널 그룹16/16/16/8이다. OFF16은 기존 채널56 통합의 OFF4와 다른 조건이다. 기존 확장 13개에 들어 있는 Hilbert+L3 ON 끝점과 Hilbert+single OFF1 끝점은 이 launcher에 넣지 않았다.

## 서버에 반영할 파일

로컬에서 다음 파일 **1개를 서버의 같은 상대 경로에 업로드**한다.

```text
upstream/globalsplat/scripts/slurm/submit_nfcgs_paper_controls.sh
→ /ceph_data/clue9986/tokencomp/upstream/globalsplat/scripts/slurm/submit_nfcgs_paper_controls.sh
```

현재 확장 13개에 사용 중인 `submit_nfcgs_order_context.sh`, settings 및 train/eval Slurm, `run_nfcgs.py`를 그대로 재사용한다. 별도 manifest는 필요 없다. 새 구조 T07/T08/T09/T18은 이 명령으로 제출하지 않는다.

## 실행 명령

```bash
cd -P /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
if bash scripts/slurm/submit_nfcgs_paper_controls.sh --dry-run; then
    bash scripts/slurm/submit_nfcgs_paper_controls.sh
fi
```

`bash`로 실행한다. `sbatch`는 launcher 안에서 각 train/eval에 호출한다. 사전 확인만 하려면 첫 번째 `--dry-run` 명령까지만 실행한다. Dry run은 job이나 state를 생성하지 않는다.

## 학습·평가 조건

- 공통 부모: 기존 FullSplit λ=.0256 probability10k checkpoint. 가중치만 로드하며 optimizer/global step을 이어받지 않는다.
- 새 50k optimizer steps, checkpoint 매5k, 시간 제한6일, GPU2×batch2×누적2, BF16, LR1e-4, 35k/45k 감소.
- 기존 seed111123/data-loader seed403와 view·subset consistency 레시피, frozen GlobalSplat 및 codec 전체 학습을 유지한다.
- Minimal, kernel5, STE, 지정 정렬·스케줄·공간 Context만 변경. 정렬은 Score와 MSH 모두에 적용된다.
- 평가는 각 train에 `afterok:<train ID>`로 연결한다. GPU1, 최종50k checkpoint 검증 후 실제 encode→decode→render, test 전체6991 scene, C12/T8, batch1, seed0.
- 부모를 직접 지정하려면 `SOURCE_CHECKPOINT`를 사용할 수 있다. 새 비교에서도 기존 실험과 같은 부모를 사용한다. 이미 접수한 state와 다른 부모·설정은 재사용을 거부한다.

## 제출 제한과 재실행

기본 state 폴더는 `logs/slurm/paper_controls_20261018/`다. point별 `.state`가 접수 ID의 기준이고 `submissions.tsv`에는 train/eval 쌍을 기록한다. Train ID는 eval 제출 전에 저장한다.

QoS 제출 제한으로 멈추면 접수 용량이 생긴 뒤 같은 명령을 다시 실행한다.

```bash
bash scripts/slurm/submit_nfcgs_paper_controls.sh
```

이미 접수된 train/eval ID는 재사용하고 빠진 것만 제출한다. state 폴더를 지우거나 다른 폴더로 바꾸지 않는다. 이는 제출 재시도 기능이며, 실패·취소된 학습을 자동 resume하는 기능은 아니다. 기존 13개나 다른 작업은 취소하지 않는다.

동시 제출을 막는 `.submit-lock`이 있으면 새 제출을 중단한다. 파일을 삭제하기 전에 다른 제출 프로세스가 실행 중인지 확인한다.

## 기록 위치

- State와 제출 요약: `logs/slurm/paper_controls_20261018/`.
- Train 이름: `gs-p-nl3on`, `gs-p-hl3off`, `gs-p-nl3off`, `gs-p-hd4off` 뒤에 λ tag.
- Eval 이름: 같은 이름의 `gs-p-e-*`.
- 출력: `outputs/nfcgs_order_context/<order>/<schedule>_stages<3 또는 4>_k5_ste[_context_off]/lambda<tag>/job_<TRAIN_ID>/`.
- 최종 checkpoint: 출력 아래 `checkpoints/nfcgs_main/version_0/step000050000.ckpt`.
- 평가 결과: 출력 아래 `eval_all/`.
- 로그: 기존과 같은 `logs/slurm/slurm-gs-order-context-<TRAIN_ID>.out/.err`, `slurm-gs-order-context-eval-<EVAL_ID>.out/.err`.

## 로컬 검증

자동 검사 4개를 모두 통과했다. 계획 JSON의 대조군 7개와 launcher의 실제 가짜 sbatch 호출을 대조했다. 반복 실행의 ID 재사용, 마지막 eval 제출 거부 후 train 중복 방지, 마지막 state 오류·부모 누락·기존 lock의 사전 탐지, 환경에 남은 다른 정렬·그룹·학습량 값의 덮어쓰기를 확인했다. 실제 Hydra로 train/eval 14개 설정의 MSH·공간 Context·단계·채널·학습량·실제 평가 조건도 확인했다. 실제 GPU 학습·서버 sbatch 실행은 로컬 검증에 포함하지 않는다.

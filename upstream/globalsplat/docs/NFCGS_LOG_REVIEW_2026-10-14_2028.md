# NN-score·정렬/context ablation 로그 점검 — 2026-10-14 20:28

20:27~20:28에 내려받은 workspace `slurm/` 로그와 `sort_context_recovery_20261014.json`을 확인했다. 아래는 다운로드된 로그의 마지막 기록이며 실시간 Slurm 상태가 아니다. 기존에 취소된 450977~450995와 재제출한 451809 이후 ID를 구분했다.

## NN-score .0256의 50k 학습 완료

학습 450934의 `.err` 끝에 다음 정상 종료 메시지가 있다.

```text
`Trainer.fit` stopped: `max_steps=50000` reached.
```

원래 NN-score step10k에서 이어 학습한 GPU knn64/input 경로가 목표50k에 도달했다. `.out`에는 epoch3까지의 진행이 있으며 Numba import 오류, CUDA OOM, 새 traceback, 취소 메시지는 없다. 후속 eval 450935의 로그는 이번 다운로드에 없으므로 평가의 시작·완료 및 실제 RD 성능은 확인하지 못했다. 최종 checkpoint 파일 자체도 로컬에 내려받지 않아 별도로 검사하지 않았다.

## 진행 기록이 있는 학습

| 설정 | λ | Train / eval ID | 마지막 epoch | epoch 내 batch / 33,017 | 표기 속도 batch/s |
|---|---|---|---:|---:|---:|
| NN-score legacy3 | .0064 | 450937 / 450938 | 2 | 19,765 | .97 |
| NN-score legacy3 | .0128 | 450939 / 450940 | 2 | 21,018 | .98 |
| Context only, legacy3 | .0064 | 451809 / 451810 | 1 | 1,420 | 1.61 |
| Context only, legacy3 | .0128 | 451811 / 451812 | 1 | 2,011 | 1.64 |
| Context only, legacy3 | .0256 | 451813 / 451814 | 1 | 1,621 | 1.66 |
| Hilbert only | .0064 | 451815 / 451816 | 1 | 1,540 | 1.66 |
| Hilbert only | .0128 | 451817 / 451818 | 1 | 794 | 1.63 |
| Hilbert only | .0256 | 451819 / 451820 | 0 | 4,490 | 1.63 |

Epoch 내 batch 번호와 퍼센트는 전체 optimizer 학습 진행률이 아니다. 유효배치8과 accumulate2를 사용하며, subset consistency로 실제 codec forward의 입력은 A/B를 합친 2B다. 위 속도는 progress bar의 배치 처리 속도이며 정렬 함수만의 GPU benchmark 시간이 아니다. Hilbert only .0256도 4,490 batch가 진행돼 초기화에 멈춘 상태를 보여주지 않는다.

## 설정 검증

로그가 있는 9개 학습에서 공통 조건을 대조했다.

- GlobalSplat 고정, codec scope all/MSH 유지, C24/T12, subset consistency ON.
- GPU2·batch/GPU2=2·accumulate2·유효배치8, BF16, 목표50k, checkpoint5k 간격.
- 각 job의 λ가 recipe와 일치한다.
- Score minimal: centering·norm·mean conditioner·channel context OFF, linear transform, spatial legacy3/kernel5/STE.
- NN-score는 knn64/input으로 거리 계산을 입력 장치에서 수행한다.
- Context-only 451809/451811/451813은 `token_order=none`, `use_morton=false`, spatial context ON이다.
- Hilbert-only 451815/451817/451819는 `token_order=hilbert`, spatial context OFF다. Mean/channel context도 OFF다.

재제출한 여섯 ablation은 `RECOVERY_MODE=restart`, `checkpointing.resume=false`와 공통 FullSplit .0256 step10k 부모의 weights-only 로드가 확인된다. 부모의 optimizer/step 상태를 이어받은 실행이 아니다. 학습 log의 `missing=38 unexpected=0`은 재생성된 frozen render loss-network tensor에 관한 것으로, model weight 누락은 검증 코드에서 거부하며 epoch 진행 기록이 있다.

## 이번 다운로드에서 실행 상태를 확인하지 못한 학습

| 설정 | λ | Train / eval ID |
|---|---|---|
| Context only, quarter2 | .0064 | 451821 / 451822 |
| Context only, quarter2 | .0128 | 451823 / 451824 |
| Context only, quarter2 | .0256 | 451825 / 451826 |
| Context only, dyadic4 | .0064 | 451827 / 451867 |
| Context only, dyadic4 | .0128 | 451868 / 451869 |
| Context only, dyadic4 | .0256 | 451870 / 451871 |

이 여섯 학습은 recovery state에 제출 ID가 있지만 해당 `.out`·`.err` 파일이 로컬에 없다. 이를 pending, 실패, 미제출로 확정하지 않는다. Queue 상태 확인이 필요하면 서버에서 다음처럼 조회한다.

```bash
squeue -j 450935,451821,451823,451825,451827,451868,451870 -o '%.18i %.24j %.10T %.30R'
```

## 오류·평가 상태와 다음 확인

읽은 9개 학습 로그에서 fatal error와 CUDA OOM·새 취소 메시지는 발견하지 못했다. Torchvision deprecated 인자, DDP의 unused-parameter 검사와 gradient stride 경고가 있다. 이 경고들로 학습이 중단된 기록은 없다. 450934에는 기존 non-resumable DataLoader 경고도 남아 있어 이어 학습 후의 입력 순서가 원래 실행과 같다는 의미로 해석하지 않는다.

Recovery state에는 NN-score 3개와 새 ablation 12개, 총15개 학습·15개 eval의 서로 다른 ID가 저장돼 있다. 이번 다운로드에서 이 eval 15개의 실행 로그는 모두 확인되지 않았다. 완료된 450934의 후속 eval 450935와 아직 로그가 없는 quarter2/dyadic4의 상태 확인이 우선이다. 새 NN-score 및 단독 ablation의 PSNR·실제 bitstream 용량·encode/decode 시간은 아직 이번 자료로 비교할 수 없다.

구조화된 분석 결과는 workspace [artifacts/nfcgs_log_review_2026-10-14_2028.json](../../../artifacts/nfcgs_log_review_2026-10-14_2028.json)에 저장했다. 확인 과정에서 모델 코드나 서버 job을 변경하지 않았다.

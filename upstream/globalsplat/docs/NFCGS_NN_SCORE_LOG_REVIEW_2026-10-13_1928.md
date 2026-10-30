# NN-score resume 오류 점검 — 2026-10-13 19:28 다운로드

이 기록은 내려받은 Slurm 로그와 resume state를 기준으로 한다. 서버 queue를 실시간 조회하거나 job을 제출하지 않았다. 다른 세션의 Hyper1D 실험은 분석 대상에서 제외했다.

## 확인 결과

| 작업 | 결과 | 근거 |
|---|---|---|
| 기존 NN-score .0256 학습 448650 | 19:23:15 취소 | Slurm JOB/STEP CANCELLED, SIGTERM |
| NN-xyz .0256 학습 448652 | 18:25:00 취소 | Slurm JOB/STEP CANCELLED, SIGTERM |
| GPU kNN resume .0256 학습 450904 | 19:26:59 첫 batch에서 실패 | 양쪽 rank의 Numba import AttributeError, srun exit 1 |
| 후속 eval 450905 | state에 ID 존재, 실행 로그 없음 | 450904에 afterok 의존하므로 이 실패한 학습으로 실행 조건 충족 불가 |
| 새 NN-score .0064 / .0128 | 확인 불가 | 받은 로그·제출 state에 해당 작업 없음 |
| 기본 Morton legacy2 .0128 학습/eval 449561 / 449562 | 50k 학습·전체 eval 완료 | max_steps 도달, 최종 metric 요약 |

기본 Morton legacy2 .0128 결과는 이전 확인과 같다. PSNR 23.6313 dB, SSIM .7339, LPIPS .2703, 실제 크기 41,815.163496 B = 40.835 KiB/scene이다. 6,991 scene을 평가했고 시간 평균은 워밍업 5개를 제외한 6,986 scene 기준이다. Codec encode/decode는 141.4/81.4 ms, sender/receiver는 189.7/84.0 ms다.

## 실패 원인과 수정

450904는 원래 448650의 `step000010000.ckpt`를 새 폴더에 복사하고, optimizer·scheduler를 포함한 전체 학습 상태를 정상 복원했다. GPU2, batch/GPU2, accumulate2, 유효배치8, `nn_score / legacy3 / kernel5 / knn64 / input` 설정도 로그에 확인된다. 복원 직후 첫 forward에서 멈췄으므로 추가 optimizer step을 완료하지 못했다. 목표 50k에 대한 잔여 학습은 40k step이다.

```text
token_order._compiled_greedy_knn -> from numba import njit
numba.core.typing.builtins -> @infer_global(print)
AttributeError: module '__main__' has no attribute '_filtered'
```

`globalsplat/main.py`의 `_quiet_upstream_logs()`가 `builtins.print`를 지역 함수 `_filtered`로 바꾼다. Numba를 처음 불러오면 이 함수의 `__module__`과 `__name__`으로 전역 접근을 검사하는데, `__main__._filtered`는 없어서 실패했다. 제출 단계의 별도 Python preflight에서는 이 필터를 설치하지 않으므로 해당 충돌을 찾지 못했다.

필터 함수에 `functools.wraps(_orig)`를 적용해 접근 가능한 `builtins.print` 이름 정보를 보존했다. 장면 건너뜀 메시지 필터와 다른 출력의 동작은 유지한다. 거리 계산, 정렬 규칙, context, bitstream, checkpoint 가중치는 변경하지 않았다.

수정 전 코드로 같은 AttributeError를 로컬에서 재현했다. 수정 후 새 Python 프로세스에서 필터 설치 → 첫 Numba import → JIT 실행 → exact kNN 정렬을 확인했다. GPU 서버의 실제 재실행은 아직 수행하지 않았다.

체크포인트 폴더 이동 경고는 저장 callback의 경로 기록에 관한 것이며 이 실패의 원인은 아니다. 데이터 로더가 epoch 중간의 위치까지 복원되지 않는다는 경고도 별개다. 모델·optimizer·scheduler는 복원됐지만 이후 장면의 순서까지 원래 실행과 같다는 의미는 아니다. CUDA OOM이나 GPU 거리 계산 오류는 이번 실패 로그에 없다.

## 서버 반영과 재제출

이번 수정은 [globalsplat/main.py](../globalsplat/main.py) 한 파일을 서버의 같은 위치에 반영하면 된다. 기존 정렬 최적화·resume 파일은 이미 반영된 상태를 전제로 한다.

기존 기본 state에 450904/450905가 저장돼 있으므로 같은 명령만 반복하면 실패한 ID를 재사용한다. 아래처럼 새 state 경로로 제출한다. 원래 step10k checkpoint를 검증한 뒤 남아 있는 450904/450905 중 본인의 active job만 취소하고 새 학습·eval을 연결한다.

```bash
cd -P /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat

RESUME_CHECKPOINT=outputs/nfcgs_order_context/nn_score/legacy_stages3_k5_ste/lambda0p0256/job_448650/checkpoints/nfcgs_main/version_0/step000010000.ckpt \
NN_ORDER_DISTANCE_DEVICE=input \
NN_SCORE_RESUME_STATE_FILE=logs/slurm/nn_score_resume_retry_450904.state \
bash scripts/slurm/submit_nfcgs_nn_score_resume.sh --replace-jobs 450904,450905
```

GPU2·유효배치8·총50k를 유지하며, 10k부터 40k를 추가 학습한다. 성공 종료 후 전체 scene eval이 `afterok`로 실행된다. 이 명령을 다시 실행하면 새 state의 ID를 재사용하므로 중복 제출하지 않는다. 새 .0064/.0128 job에는 관여하지 않는다.

원본 로그는 workspace의 `slurm/slurm-gs-ns-resume-450904.out`, `.err`, `slurm/nn_score_resume_448650_input_knn.state`에 있다.

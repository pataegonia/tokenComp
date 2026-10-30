# Context 실험의 시간·복잡도 측정 현황

2026-10-12 저녁까지 받은 로그와 과거 overhead 보고서를 기준으로 정리했다. 이 정리를 위해 새로운 GPU 작업은 실행하지 않았다.

## 최근 정렬·단계 실험의 실제 bitstream 시간

아래는 λ=.0256, C12/T8, batch1, BF16, 실제 bitstream eval의 평균이다. 단위는 ms/scene이다. CUDA 동기화를 측정 전후에 수행해 GPU 연산이 완료될 때까지 기다린 벽시계 시간이다. CPU 작업도 측정 구간에 포함된다.

| 설정 | Codec encode | Codec decode | 전체 sender | 전체 receiver |
|---|---:|---:|---:|---:|
| Morton 기본 legacy2, kernel5/STE | 142.2 | 81.4 | 190.5 | 84.0 |
| Morton legacy3 | 139.9 | 80.8 | 188.1 | 83.4 |
| Morton legacy4 | 148.0 | 84.9 | 196.1 | 87.4 |
| Morton quarter2 | 142.6 | 82.6 | 190.8 | 85.1 |
| Morton dyadic4 | 144.2 | 84.8 | 191.9 | 87.7 |
| Hilbert legacy3 | 152.8 | 85.0 | 201.2 | 87.6 |
| Hilbert quarter2 | 147.1 | 82.3 | 195.4 | 85.0 |
| Hilbert dyadic4 | 161.0 | 92.7 | 209.2 | 95.9 |

- Codec encode는 로그의 `entropy_encode`다. 위치 투영과 정렬, low-rank 분석, Score 압축 및 encoder 내부의 causal 복원, MSH residual 압축, bitstream 생성 등을 함께 포함한다. 순수한 arithmetic/rANS coder 시간만을 뜻하지 않는다.
- Codec decode는 `entropy_decode`다. Score와 MSH의 실제 bitstream 복호 및 token 재구성을 포함한다.
- 전체 sender는 GlobalSplat scene encoder + codec encode다. 전체 receiver는 codec decode + Gaussian decoder다.
- GlobalSplat scene encoder는 이 표에서 약 48 ms/scene, Gaussian decoder는 약 2.5~3.2 ms/scene이다. Renderer는 별도로 약 .4 ms/view이며 sender/receiver 합에 포함하지 않았다.
- 시간 평균은 처음 5 scene의 워밍업을 제외한 6,986 scene에 대한 값이다. 화질·용량 평가 대상은 6,991 scene이다. 이전 확인에서 6,986을 전체 평가 수로 설명한 부분을 정정한다.
- 서로 다른 job과 노드에서 측정해 GPU 종류와 부하가 같다고 보장할 수 없다. 같은 GPU에서 같은 scene으로 반복한 실험 전에는 작은 차이를 알고리즘만의 효과로 확정하지 않는다.

서버의 각 `EVAL_OUTPUT` 디렉터리에 `pipeline_timing.json`은 평균·중앙값·합계·호출 수, `benchmark.json`은 구간별 시간 목록, `peak_memory.json`은 eval 프로세스의 CUDA allocated peak를 저장하도록 구현돼 있다. 현재 이 최근 실험들의 timing/memory JSON은 로컬에서 발견하지 못했으며 위 표는 Slurm stdout의 평균값을 사용했다. Eval의 메모리 peak는 구간별 증분 메모리와 다르다.

## 과거에 완료한 통제된 overhead 측정

별도로 2026-09-15의 job 423729에서 18 checkpoint의 profile을 완료했다. 동일한 RTX A5000 한 개에서 같은 32 test scene을 순서대로 사용하고, 4 scene을 워밍업했다. 현재 정렬·단계 모델이 아니라 이전 Mean/Channel/Spatial/Full 및 Shared/Split/Gaussian/Conditional Scale 모델의 측정이다.

그 결과에는 다음이 있다.

- Score 단독 encode/decode와 feature codec 전체 encode/decode의 시간 분포·중앙값.
- Score path, context predictor, probability model의 parameter count.
- Parameter와 buffer의 실제 저장 바이트 및 직렬화 모델 크기.
- 각 측정 구간의 증분 CUDA peak allocation. CPU/rANS의 native allocation은 포함하지 않는다.
- Score payload와 container overhead, 송수신 Score 복원값의 일치 확인.

로컬 원본은 저장소 루트의 `overhead-REPORT.md`, `overhead-aggregate.json`이다. 예전 Full+Split λ=.0256의 Score 단독 encode/decode 중앙값은 49.11/29.57 ms, 전체 codec은 133.50/77.71 ms였다. 최근 표와 모델·scene 표본·집계 방법이 달라 절대 시간을 직접 비교하지 않는다.

## 아직 없는 측정

최근 legacy2/3/4, quarter2/dyadic4, Morton/Hilbert/NN 설정 전체에 대한 같은 GPU의 통제된 profile은 아직 없다. 최신 모델의 정렬 단독 시간, Score context 단독 시간, MSH 단독 시간, 구간별 증분 메모리와 FLOPs/MACs도 이번 로그만으로는 확보되지 않는다.

NN 정렬은 아직 eval이 완료되지 않았다. 받은 학습 로그에는 nn_score .22 batch/s, nn_xyz .46 batch/s가 기록돼 있지만 이는 forward/backward가 포함된 학습 속도여서 추론 encode/decode 시간과 섞지 않는다.

## 저장한 정리와 측정 구간 코드

- 로그에서 모은 수치: `artifacts/order_context_timing_2026-10-12.json`
- 정리 재실행: `python artifacts/summarize_context_timing_2026-10-12.py`
- 최신 시간 측정: `globalsplat/model/model_wrapper.py`의 `_time_test_stage`, `test_step`, eval 종료 시 `pipeline_timing.json` 저장 부분.
- 예전 profile: [측정 방법](NFCGS_OVERHEAD_PROFILE_2026-09-15.md).

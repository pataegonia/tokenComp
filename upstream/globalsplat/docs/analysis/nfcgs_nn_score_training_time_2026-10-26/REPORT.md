# Fresh 이전 NN-score 학습 시간 비교 — 2026-10-26

최적화된 NN-score L3는50k 학습 루프에 약28.2~28.6시간이 걸렸다. 같은λ·L3의 None/Morton/Hilbert는16.9~17.2시간으로, NN-score가 약1.66~1.70배(+66~70%), 약11~12시간 더 걸렸다. Hilbert D4와 비교하면 약1.53~1.55배(+53~55%), 약10시간 더 걸렸다.

## 비교 조건과 계산

- 새 fresh-init 실험을 제외한 기존 FullSplit .0256 step10k 부모의 weights-only 후속50k다.
- 주 비교12개 모두 실제 실행 명령의 GPU2 × microbatch2 × accumulation2 = 유효 batch8, 목표50k, kernel5를 확인했다. 같은λ .0064/.0128을 비교한다.
- NN-score는 `knn`·후보64개·거리장치 `input`의 최적화된 학습450937/450939다. L3 계열끼리의 비교를 우선하며 D4/L2는 schedule이 다른 참고 비교다.
- tqdm의 epoch별 마지막 batch에서 가장 큰 elapsed clock을 합했다. epoch 전환 때 출력되는 이전 epoch의0/33017·00:00 행은 제외했다. 주 비교 모두 batch 수33017+33017+33017+946, 최종50k 종료 footer를 확인했다. accumulation과 epoch 끝의 마지막 불완전 그룹 때문에 batch 수를 단순히2로 나누어 정확한 최종 global step을 추정하지 않는다.
- 아래 시간은 학습 progress 경과시간이다. Queue와 test eval은 포함하지 않으며, 초기화·epoch 전환 등의 일부 overhead가 빠져 Slurm allocation 전체 elapsed와 같지 않다. Clock 안의 checkpoint/quantile 등의 비용은 포함될 수 있다. GPU 모델·동일 노드·부하를 통제한 benchmark는 아니다.

## 완료한50k 학습 로그

| 설정 | λ=.0064 | λ=.0128 | Job(.0064 / .0128) |
|---|---:|---:|---|
| None L3 | 17.22h | 16.89h | 451809 / 451811 |
| Morton L3 | 17.13h | 16.94h | 447719 / 447721 |
| Hilbert L3 | 16.87h | 16.95h | 447723 / 447725 |
| Hilbert D4 | 18.73h | 18.17h | 447744 / 447746 |
| Morton L2 | 16.11h | 16.31h | 449559 / 449561 |
| NN-score L3 | 28.62h | 28.19h | 450937 / 450939 |

| NN-score / 기준 | λ | 시간 배수 | 추가 시간 | 증가율 |
|---|---:|---:|---:|---:|
| None L3 | 0.0064 | 1.662× | 11.41h | +66.2% |
| Morton L3 | 0.0064 | 1.671× | 11.50h | +67.1% |
| Hilbert L3 | 0.0064 | 1.697× | 11.75h | +69.7% |
| Hilbert D4 | 0.0064 | 1.529× | 9.90h | +52.9% |
| Morton L2 | 0.0064 | 1.777× | 12.52h | +77.7% |
| None L3 | 0.0128 | 1.669× | 11.30h | +66.9% |
| Morton L3 | 0.0128 | 1.664× | 11.25h | +66.4% |
| Hilbert L3 | 0.0128 | 1.663× | 11.24h | +66.3% |
| Hilbert D4 | 0.0128 | 1.551× | 10.02h | +55.1% |
| Morton L2 | 0.0128 | 1.728× | 11.88h | +72.8% |

## 초기 구현·재개 실행은 별도

초기 NN-score448650은 취소 시 약11.42k optimizer step까지29.02시간이 걸렸다. 그 속도를50k로 환산하면 약127.0시간(5.29일)이지만, 완료한50k 실측 시간이 아니다.

최적화 후450934는 step10k에서50k까지40k를 재개해 약23.64시간이다.50k 환산은 약29.55시간으로, 새50k의 주 비교와 구분한다. 초기 취소 전 미저장분·실패450904·대기·재시도 시간을 완료본에 합치지 않았다.

## Step당 시간

| 설정 | λ | 학습 루프 평균 s/optimizer-step |
|---|---:|---:|
| NN-score L3 | 0.0064 | 2.061 |
| NN-score L3 | 0.0128 | 2.030 |
| None L3 | 0.0064 | 1.240 |
| None L3 | 0.0128 | 1.216 |
| Morton L3 | 0.0064 | 1.233 |
| Morton L3 | 0.0128 | 1.219 |
| Hilbert L3 | 0.0064 | 1.215 |
| Hilbert L3 | 0.0128 | 1.220 |
| Hilbert D4 | 0.0064 | 1.348 |
| Hilbert D4 | 0.0128 | 1.309 |
| Morton L2 | 0.0064 | 1.160 |
| Morton L2 | 0.0128 | 1.175 |

## 근거

[계산·로그 SHA](timing.json), [재계산 코드](analyze.py). 로그는 workspace 루트 `slurm/`의 다운로드본이다. 서버 조회·제출·취소·GPU 실험을 수행하지 않았다.

- `slurm/slurm-gs-order-context-450937.out` — SHA256 `9bb878af15ec5b1a902d38ec93d6f75559c2424b7b6a8e8172ae223f6ee4bef8`
- `slurm/slurm-gs-order-context-450939.out` — SHA256 `efca49f298e46822fba1a6519f5ecdf54162043f53d1f23c54deb4d877e37a79`
- `slurm/slurm-gs-oc-recovery-451809.out` — SHA256 `273d5d11d04d05b4909e84790640bf27e6ba75c7c7dc1a25e38b82dc3c29c899`
- `slurm/slurm-gs-oc-recovery-451811.out` — SHA256 `ef60eca79de9327037a8bea11d83fed712d2b0f28e5fba44d20e83ddabbf0a77`
- `slurm/slurm-gs-order-context-447719.out` — SHA256 `50de40bad05c04c4ce20ef719cecb95e4fa9fa1a30230de355657883170d7c3b`
- `slurm/slurm-gs-order-context-447721.out` — SHA256 `37ca0c324efec745eb9d25d6c94bcfb8321bedca5284b501d2ef9eb89749a440`
- `slurm/slurm-gs-order-context-447723.out` — SHA256 `6b97fa11ec79e4f4bb632041616c73118a1596a0ef6ab2d9ffa85a2a8abb5318`
- `slurm/slurm-gs-order-context-447725.out` — SHA256 `71cb6997ca74b572091345204d2951f6a749d705cef1bcf0019c824f44a96016`
- `slurm/slurm-gs-order-context-447744.out` — SHA256 `239753445730ce916b5a2af10990cb7830512c2ff6204be40351aabc2b9db9c1`
- `slurm/slurm-gs-order-context-447746.out` — SHA256 `7fa2ca37e26d09d237ab4196fec24eea69fcae2d17fe4ab8a0e6c7250c3a6c9b`
- `slurm/slurm-gs-order-context-449559.out` — SHA256 `760c6dc14a7cb9bffed30b7807d1279dd6ed3d64d379d87b28388f4cbab323b5`
- `slurm/slurm-gs-order-context-449561.out` — SHA256 `892ed604cb4ed06186d7e5ce50f4d2e142d591c5d0e16b1303c74d92e66dc857`
- `slurm/slurm-gs-order-context-448650.out` — SHA256 `318695a1580e9a66b7a43808428b5d2a3a36ab32022a69909b2119ffa494221d`
- `slurm/slurm-gs-ns-resume-450934.out` — SHA256 `2c95311a7d034cdd9289f8ac71d9f198279a34d6ed0e4ca8e9bee91bde9d2674`

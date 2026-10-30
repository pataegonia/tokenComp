# 정렬·context 실험 복잡도 — 2026-10-16

2026-10-17 후속: 실제 GPU job455251에서 기존 37개 point의 측정을 모두 완료했다. [같은 GPU·같은 입력의 실제 시간·메모리 검토](analysis/nfcgs_complexity_455251/REPORT.md)에 검증 결과와 전체 수치가 있다. 아래 문서는 10월 16일의 static 계산 및 측정 계획을 보존한 기록이다. 실제 속도 판단에는 후속 결과를 사용한다. 특히 통제된 새 측정에서는 Hilbert legacy3가 quarter2보다 빨랐다.

최근 정렬/context 실험 37개 point를 13개 설정으로 묶었다. 기존 7개 RD 곡선의 21개 point, 최근 NN-score/context-only/Hilbert-only의 15개 point, Morton legacy4 .0256의 1개 point가 대상이다. 10월 16일 12:31 자료에서 35개 point의 eval이 완료됐고, context-only quarter2 .0256과 dyadic4 .0064는 학습 중이었다. 오래된 mean/channel/probability ablation은 이번 목록에 포함하지 않았다.

## 지금 계산한 파라미터와 신경망 연산량

Batch1·4096 token·rank56·MSH 유지·minimal linear/kernel5/STE 조건으로 현재 코드를 실행해 계산했다. 각 λ에서 구조가 같으므로 파라미터 수와 shape 기반 MAC은 같다. 초기화한 codec과 합성 입력을 사용했으며, 학습된 checkpoint의 실제 GPU 시간·GPU 메모리는 별도 측정 대상이다.

활성 파라미터는 실행하는 구조의 가중치와 사용하는 entropy model 파라미터를 합한 값이다. 학습 호환성을 위해 보존한 미사용 MLP·mean/channel predictor·구형 entropy·anchor schedule의 미사용 convolution은 제외한다. CDF 생성용 확률모델 가중치는 포함하므로 매 symbol마다 그 신경망을 실행한다는 뜻은 아니다. 저장 파라미터는 codec에 남아 있는 호환성 가중치를 포함한다. requires_grad와 실제 사용 여부가 다를 수 있어 두 개를 분리했다.

MAC은 PyTorch FlopCounterMode로 실제 compress/decompress 경로의 등록된 행렬곱·convolution을 세었다. 1 multiply-add=1MAC, 해당 등록 연산의 FLOPs=2×MAC으로 표기한다. AnchorPairPredictor의 einsum/bmm, functional shared-basis 행렬곱, GDN 내부 convolution, ConvTranspose2d를 포함한다. 정렬 비교·거리 계산·indexing·CPU 전송·rANS·원소별 연산·CDF 생성은 이 MAC에 포함되지 않으므로 실제 시간을 별도로 측정해야 한다. [PyTorch 카운터 설명](https://docs.pytorch.org/docs/main/generated/torch.utils.flop_counter.FlopCounterMode.html)

아래 encoder는 위치가 준비된 후 `feature_codec.compress`를 호출하는 codec core다. 현재 구현에서 sender 내부의 Score causal 복원과 MSH 복원도 실행하므로 그 비용을 포함한다. Decoder는 `feature_codec.decompress`다. GlobalSplat scene encoder·token center 준비·Gaussian decoder·renderer는 아래 core 열에 포함하지 않는다.

| 설정 | 활성 Params (M) | 저장 Params (M) | Context predictor Params | Encoder GMAC | Decoder GMAC | Score decode MMAC |
|---|---:|---:|---:|---:|---:|---:|
| Morton legacy2 | 4.585288 | 4.696560 | 4,216 | 6.967001 | 3.393454 | 17.039360 |
| Morton legacy3 | 4.592920 | 4.704192 | 8,432 | 6.984040 | 3.410493 | 34.078720 |
| Hilbert legacy3 | 4.592920 | 4.704192 | 8,432 | 6.984040 | 3.410493 | 34.078720 |
| Morton quarter2 | 4.586400 | 4.701888 | 5,328 | 6.955074 | 3.381527 | 5.111808 |
| Hilbert quarter2 | 4.586400 | 4.701888 | 5,328 | 6.955074 | 3.381527 | 5.111808 |
| Morton dyadic4 | 4.593232 | 4.717152 | 5,328 | 6.955926 | 3.382378 | 5.963776 |
| Hilbert dyadic4 | 4.593232 | 4.717152 | 5,328 | 6.955926 | 3.382378 | 5.963776 |
| NN-score legacy3 | 4.592920 | 4.704192 | 8,432 | 7.152861 | 3.410493 | 34.078720 |
| Context-only legacy3 | 4.592920 | 4.704192 | 8,432 | 6.984040 | 3.410493 | 34.078720 |
| Context-only quarter2 | 4.586400 | 4.701888 | 5,328 | 6.955074 | 3.381527 | 5.111808 |
| Context-only dyadic4 | 4.593232 | 4.717152 | 5,328 | 6.955926 | 3.382378 | 5.963776 |
| Hilbert-only/context OFF | 4.584488 | 4.704192 | 0 | 6.949962 | 3.376415 | 0.000000 |
| Morton legacy4 | 4.600552 | 4.711824 | 12,648 | 7.001080 | 3.427533 | 51.118080 |

Quarter2/dyadic4의 Score predictor 연산량은 legacy3보다 작지만, MSH와 공통 변환이 전체 codec 연산의 큰 부분이어서 전체 GMAC 차이는 작다. NN-score는 정렬 전 Score 분석 행렬곱이 추가되어 legacy3보다 encoder가 .168820736GMAC 크고, decoder 신경망 MAC은 같다. NN의 거리·tour 비용은 이 차이에 포함되지 않는다.

## 정렬의 크기별 비용

N은 token 수, C는 feature 폭, r은 Score 폭 56, b는 Morton/Hilbert 정밀도 10bit다. 아래는 현재 구현을 읽어 정리한 비용 모델이며 측정된 walltime이 아니다.

| 정렬 | 순서 생성 | 별도 특징 |
|---|---|---|
| none | identity index 생성 O(N) | 현재 코드는 공통 feature gather를 실행하므로 그 비용은 O(NC)로 남음 |
| Morton/Hilbert | key 계산 O(Nb) + argsort | 비교정렬 모델에서는 argsort가 O(N log N). 실제 CPU/CUDA sort 구현의 알고리즘은 별도 |
| NN-score knn64/input | Score 분석 O(NCr), 전체 L1 거리 O(N²r) | 후보 선택과 CPU greedy tour를 추가 실행. knn64여도 현재 전체 거리행렬을 만듦 |

NN 거리행렬은 4096²개의 FP32만으로 scene당 64MiB다. CPU로 복사하는 경로에서는 CPU 복사본과 GPU 원본 및 후보 배열이 별도로 존재할 수 있다. 이 64MiB는 전체 GPU peak가 아니다. Decoder는 정렬된 bitstream을 읽으며 NN tour를 다시 계산하지 않는다. Stage 수 S와 channel group 수 4에 따라 Score entropy string은 8/12/16개이고, context OFF 실험도 legacy3의 12개 split 구조를 유지한다.

## 기존 실제 eval에서 관찰한 시간

단위는 ms/scene, actual bitstream eval의 평균이다. 아래 encode는 `model.compress_scene_tokens`여서 앞의 codec core 연산량에 더해 token center 준비도 포함한다. 서로 다른 job/노드에서 측정한 시간이므로 같은 GPU에서 통제한 순위로 해석하지 않는다. 35개 완료 point의 원래 로그와 수치를 모두 보존했다.

| 설정 | λ=.0064 enc/dec | λ=.0128 enc/dec | λ=.0256 enc/dec |
|---|---:|---:|---:|
| Morton legacy2 | 142.7 / 82.3 | 141.4 / 81.4 | 142.2 / 81.4 |
| Morton legacy3 | 157.9 / 90.3 | 151.8 / 86.9 | 139.9 / 80.8 |
| Hilbert legacy3 | 162.6 / 90.5 | 156.3 / 87.2 | 152.8 / 85.0 |
| Morton quarter2 | 144.2 / 83.8 | 141.6 / 82.4 | 142.6 / 82.6 |
| Hilbert quarter2 | 143.9 / 81.3 | 139.3 / 78.4 | 147.1 / 82.3 |
| Morton dyadic4 | 162.9 / 95.2 | 156.6 / 91.5 | 144.2 / 84.8 |
| Hilbert dyadic4 | 165.4 / 94.2 | 156.9 / 89.9 | 161.0 / 92.7 |
| NN-score legacy3 | 261.6 / 85.1 | 258.1 / 89.6 | 258.4 / 87.6 |
| Context-only legacy3 | 146.9 / 85.7 | 145.7 / 84.8 | 146.4 / 84.6 |
| Context-only quarter2 | 144.0 / 84.9 | 149.5 / 88.1 | eval 전 |
| Context-only dyadic4 | eval 전 | 155.8 / 92.1 | 154.3 / 91.2 |
| Hilbert-only/context OFF | 152.4 / 84.6 | 149.5 / 83.3 | 160.4 / 87.9 |
| Morton legacy4 | 실험 없음 | 실험 없음 | 148.0 / 84.9 |

## 준비한 같은 GPU의 실제 profile

`scripts/profile_nfcgs_complexity.py --mode actual`은 37개 최종50k checkpoint를 같은 GPU 할당에서 순서대로 검증·측정한다. 첫 실행의 C12 입력 scene 36개를 CPU에 cache하고 모든 checkpoint에 같은 영상·카메라 tensor를 재사용한다. 앞 4개는 warmup, 다음 32개는 3회 반복 측정한다. 입력 fingerprint와 scene 목록을 기록한다. Image encoder와 token position 준비는 scene당 1회, 나머지 구간은 3회 측정하며 집계 count도 구분한다.

다음 구간별 시간의 mean/median/p90과 CUDA allocated peak, 측정 전 할당량, 증분 peak를 기록한다: scene encoder, token position 준비, 순서 생성(필요한 NN Score 분석 포함), feature gather, Score encode/decode, MSH encode/decode, codec core encode, eval과 같은 pipeline entropy encode, codec decode, Gaussian decoder. 메모리는 PyTorch CUDA allocator 기준으로 CPU rANS/native 메모리나 NVML 전체 VRAM은 포함하지 않는다. 각 구간은 따로 실행하므로 구간 시간을 단순 합해 end-to-end 측정값으로 부르지 않는다.

FLOP 카운터는 시간 측정과 분리해 첫 측정 scene에서만 실행한다. Score 단독과 전체 bitstream의 일치, 송수신 Score 복원의 완전 일치를 검증하고, 실제 CDF buffer bytes와 직렬화된 codec state_dict 크기도 저장한다. Renderer와 PSNR 재평가는 실행하지 않는다. λ마다 학습된 값·bitstream 크기·NN fallback 양상이 달라질 수 있으므로 시간·메모리는 37개 point 모두 측정한다.

로컬 CPU에서 9개 테스트를 통과했다: functional 행렬곱·전치 convolution 카운트, 미사용 가중치 제외, 6개 schedule/ON-OFF의 isolated/full bitstream 일치, decoder 정렬 미실행, warmup/반복 count, 입력 fingerprint, 부분 재개와 이전 결과 보존, Slurm 구문 및 plan. 실제 GPU·실제 RE10K 입력으로는 아직 실행하지 않았다.

## 서버 반영 파일

- `scripts/profile_nfcgs_complexity.py`
- `scripts/slurm/profile_nfcgs_complexity.slurm`
- [측정 manifest](../../../artifacts/nfcgs_complexity_manifest_2026-10-16.json)을 서버 `logs/slurm/nfcgs_complexity_manifest_2026-10-16.json`에 저장

master의 globalsplat 저장소에서 계획 확인:

```bash
conda activate globalsplat
bash scripts/slurm/profile_nfcgs_complexity.slurm plan
```

남은 두 학습이 진행 중일 때 완료 후 시작하도록 예약:

```bash
sbatch --dependency=afterok:453066:453068 scripts/slurm/profile_nfcgs_complexity.slurm
```

두 학습이 이미 끝나 최종 checkpoint가 있으면 의존성 없이 제출:

```bash
sbatch scripts/slurm/profile_nfcgs_complexity.slurm
```

GPU1개·CPU8개·RAM48G·최대6시간 할당이다. 이는 scheduler 요청 한도이며 실측 실행시간 예측은 아니다. TMPDIR는 짧은 사용자 Ceph 경로를 사용한다. 훈련·checkpoint 수정·job 취소는 하지 않는다. 최종 파일이 없으면 기본 동작은 측정 시작 전 중단이다. 완료된 point만 먼저 측정하려면 `PROFILE_SKIP_MISSING=1`을 설정할 수 있고, 누락 목록을 결과에 남긴다.

출력은 `outputs/nfcgs_complexity/job_<ID>/`다. point별 JSON과 `summary_<개수>points_<hash>.json`, cache된 입력과 run 환경이 있다. 결과를 다운로드할 때 cache 입력 `.pt`는 분석에 필요하지 않고 JSON들과 Slurm 로그면 된다. 같은 출력으로 재개하려면 `PROFILE_OUTPUT=<기존폴더> PROFILE_RESUME=1`로 제출하되 GPU UUID/host·환경·프로토콜·manifest가 같아야 한다. 다른 장비에서는 새 출력으로 전체 비교를 다시 측정한다.

## 로컬 산출물

- [구조별 원본 카운트](../../../artifacts/nfcgs_complexity_static_2026-10-16/static.json)
- [37개 point 비교 데이터](../../../artifacts/nfcgs_complexity_summary_2026-10-16.json)
- manifest 생성: `python -X utf8 artifacts/build_nfcgs_complexity_manifest_2026-10-16.py`
- static 계산: `python -X utf8 upstream/globalsplat/scripts/profile_nfcgs_complexity.py --manifest artifacts/nfcgs_complexity_manifest_2026-10-16.json --mode static --output <새 출력 폴더>`
- 정리 재생성: `python -X utf8 artifacts/summarize_nfcgs_complexity_2026-10-16.py`

# Both-off 로컬 구조 복잡도 — 2026-10-27

Sorting=None, Score spatial context OFF인 기존 L3 OFF12 구조를 로컬 CPU에서 실행해 파라미터 수와 등록 신경망 MAC을 계산했다. 실제 학습 checkpoint를 로드한 latency·GPU 메모리 측정은 아니다.

## 결과

Batch1,4096 tokens,F736,rank56,MSH N192/M320/adapter96,linear minimal,채널 그룹16/16/16/8,L3의12개 Score entropy stream을 유지한다.

| 항목 | 로컬 계산값 |
|---|---:|
| 활성 구조 파라미터 |4,584,488 (4.584488M)|
| 저장 파라미터 |4,704,192 (4.704192M)|
| 활성 context predictor 파라미터 |0|
| Codec core encode |6.949961728 GMAC|
| Codec decode |3.376414720 GMAC|

활성 파라미터는 실제 사용하는 구조와 entropy model의 파라미터이며, 저장 파라미터에는 checkpoint 호환성을 위해 남긴 미사용 가지도 포함한다. 원 profiler의 등록된 행렬곱·convolution FLOPs를2로 나눠1 multiply-add=1MAC으로 계산했다. 정렬 비교·거리·gather/indexing·전송·rANS·원소별 연산·CDF 생성·GlobalSplat backbone·Gaussian decoder·renderer는 MAC 범위에서 제외한다. Encode는 sender 내부의 causal Score 복원과 MSH 복원까지 포함하는 기존 codec core 경로다.

기존 Hilbert+L3 OFF12의 파라미터·신경망 MAC과 같다. 두 구조에서 정렬은 달라지지만 등록 신경망의 shape은 같기 때문이다. 이 일치를 latency·메모리 동일성으로 해석하지 않는다. 공간 context OFF여도 MSH의 token 축 convolution은 유지한다.

## 로컬 환경과 가능한 측정

- 로컬 GPU: NVIDIA GeForce GTX1660 SUPER,6GiB. CUDA 사용 가능.
- 로컬 Python 환경: `nfcgs-local-profile`,PyTorch2.5.1+cu121.
- 이번 계산은 CPU8 threads와 초기화한 codec·합성 입력을 사용했다.
- 학습된 both-off checkpoint는 로컬 checkpoint/outputs 경로에서 발견하지 못했다. 공개 GlobalSplat checkpoint와 Hyper1D 초기화용 checkpoint는 별도다.
- 현재 이 환경에서 `gsplat` 모듈은 발견되지 않았다.

실제 codec latency와 GPU memory도 로컬 측정 자체는 가능하지만, 학습 checkpoint와 측정 입력 및 해당 실행 경로의 의존성이 필요하다. 기존 actual profiler는 통합 모델과 동일 C12 장면 입력을 사용하므로 필요한 파일·환경을 먼저 준비해야 한다. 로컬 GTX1660 SUPER 측정값을 기존 RTX A5000·BF16 결과와 같은 장비의 비교 표로 합치지 않는다. 같은 장비에서 both-off와 기준 모델을 함께 재측정해야 한다.

## 근거와 실행

- [측정 원본](static_measurement/static.json)
- [Manifest](manifest.json): 기존 완료 RD inventory의 `both_off_0p0128` 설정과 현재 CodecConfig로 정규화해 동일함을 확인했다. Manifest의 서버 checkpoint 경로는 출처이며 static 모드에서는 그 파일을 열지 않는다.
- 실행 코드: `scripts/profile_nfcgs_complexity.py --mode static`을 수정 없이 사용했다.

Repo root에서 다음과 같이 재계산할 수 있다. Output은 아직 없는 새 경로를 사용한다.

```powershell
& 'C:/Users/a0103/miniconda3/envs/nfcgs-local-profile/python.exe' `
  scripts/profile_nfcgs_complexity.py `
  --manifest docs/analysis/nfcgs_both_off_local_2026-10-27/manifest.json `
  --mode static --device cpu --tokens 4096 --threads 8 `
  --output docs/analysis/nfcgs_both_off_local_2026-10-27/static_repeat_01
```

서버 학습·checkpoint·Slurm job을 변경하지 않았다. 초기화된 codec에서 나온 bytes·실행시간을 학습 모델의 압축 성능이나 실제 GPU latency로 보고하지 않는다.

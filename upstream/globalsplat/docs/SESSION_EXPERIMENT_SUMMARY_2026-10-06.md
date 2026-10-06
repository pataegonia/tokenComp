# GlobalSplat token compressor 실험 정리 — 2026-10-06

## 1. 핵심 결과

- 기존 기준 모델은 **Nonlinear32 / rank56 / Morton ON / MSH residual ON / Full context P0 / Split entropy**다. λ=0.0256에서 **36.920 KiB, PSNR 23.5707 dB**다.
- 최근 **minimal Score + 3단계 / kernel5 / STE**는 **30.807 KiB, PSNR 23.4892 dB**다. 기존 기준보다 전체 bytes는 **16.56% 감소**, PSNR은 **0.0815 dB 감소**했다. 더 작은 용량의 후보이며, 같은 bitrate에서의 우열을 검증한 결과는 아니다.
- GlobalSplat까지 모두 새로 학습한 joint는 220k → 500k 연장 후에도 **PSNR 19.2332 dB**였다. 이번 학습 recipe에서는 pretrained GlobalSplat을 고정한 기존 기준의 화질에 도달하지 못했다.
- `b_mean` 값을 측정한 진단과 encoder/decoder를 별도 job으로 실행한 평가는 완료됐다. **`b_mean` ON/OFF 재학습 대조 실험은 저장 공간 오류로 실패**했으므로 효과에 대한 최종 결론이 없다.
- **2단계 / kernel5 / STE와 4단계 / kernel5 / STE는 실행 명령을 전달한 상태**다. 현재 내려받은 로그에는 결과가 없다. 서버에서 제출됐는지, 실행 중인지도 이 자료만으로 확인할 수 없다.

### 정리 기준과 근거

이 문서는 기존 실험 문서, 현재 로컬의 Slurm stdout 54개와 대응 stderr, 다운로드한 JSON을 대조했다. 서버의 실시간 queue나 checkpoint 파일 존재 여부를 조회한 기록은 아니다. 학습 job, eval job, 재시도와 중복 평가는 별도 실험 개수로 합산하지 않았다.

- [9월 15일까지의 완료 실험 인수인계](SESSION_HANDOFF_EXPERIMENTS_2026-09-15.md)
- [이전 48개 full-test eval의 지표·checkpoint 계보](NFCGS_CONTEXT_EXPERIMENT_HISTORY_2026-09-14_metrics.json)
- [이번에 대조한 54개 Slurm 로그의 추출 지표](SESSION_EXPERIMENT_SUMMARY_2026-10-06_metrics.json)
- [초기 아카이브까지 포함한 프로젝트 지도](../../../EXPERIMENTS_OVERVIEW.md)

## 2. 비교 조건

| 항목 | 주력 Score 계열의 full-test 조건 |
| --- | --- |
| 데이터 | RE10K, 256×256 |
| 입력 / 평가 시점 | context 12 views / target 8 views |
| 장면 수 | 실제 처리 6,991개 |
| 추론 | batch1, seed0, bf16-mixed, test-time 최적화 없음 |
| rate | 실제 bitstream을 encode → decode한 bytes/scene |
| 용량 단위 | KiB = bytes / 1024 |
| 포함 비용 | Score, residual y/z, scene mean, container |
| 제외 비용 | 공유 신경망 가중치 |
| 출력 | scene당 token 4,096개, Gaussian 32,768개 |
| timing | 최초 5개를 제외한 6,986개; 개별 로그의 작은 시간 차이는 반복 benchmark 결과가 아님 |

`6991/7286`에서 정상 종료한 full eval은 6,991개를 평가한 것이다. 7,286은 데이터 index의 수이며 실제 처리 장면 수와 다르다. 32/100-scene 진단과 validation 결과는 full-test 표에 섞지 않았다.

**같은 λ도 같은 bitrate를 보장하지 않는다.** 구조 변경 실험에서 bytes와 화질이 함께 달라지면 RD trade-off로 읽어야 한다. 여러 요인을 동시에 바꾼 실험에서 특정 모듈 하나의 효과를 분리할 수도 없다.

## 3. 주력 실험의 계보와 학습량

```text
pretrained GlobalSplat 고정
 └─ Nonlinear32 factorized codec 50k
     ├─ factorized continuation +50k
     └─ Full context P0 +50k
         ├─ P0/P1/P2 +25k
         ├─ low-LR 전체 codec +10k
         └─ probability-only Shared/Split/Gaussian/Conditional +10k
             └─ Full + Split 기준 모델
                 ├─ b 진단 / sender·receiver 분리 eval
                 ├─ b_mean ON/OFF 추가20k 시도 → 저장 오류
                 ├─ minimal Score, 2단계/k3/noise +50k
                 └─ minimal Score, 3단계/k5/STE +50k

GlobalSplat + codec 전체 random initialization
 └─ joint 220k → full-state resume +280k → 총500k
```

Probability-only와 low-LR 실험은 원래 Full P0 100k 부모에서 각각 갈라졌다. P0/P1/P2의 추가25k 모델에서 이어간 것이 아니다.

### step 숫자를 읽는 방법

- 초기 Nonlinear checkpoint의 `step000050000.ckpt`: 해당 codec 단계의 50k.
- 기존 Split의 `step000010000.ckpt`: **Nonlinear 50k + context 50k 이후 probability-only 10k**. 해당 계보의 codec 학습량은 110k다.
- minimal·3단계의 `step000050000.ckpt`: 기존 Split weight를 로드한 뒤 **추가50k**. 활성 모듈이 부모 weight를 이어받았으며 optimizer/step state는 새로 시작했다.
- 따라서 minimal·3단계 로그의 `Starting training from scratch`는 optimizer/step 시작 방식이다. **GlobalSplat과 codec을 모두 random initialization으로 학습했다는 뜻이 아니다.**
- joint의 500k: 220k에서 학습 상태를 이어받아 추가280k를 실행한 총 step이다.

## 4. 기존 기준 모델을 고른 실험

아래는 9월 15일 인수인계와 이전 48개 eval JSON에 근거한 완료 실험이다. 세부 arm 수와 전체 지표는 원 문서에 보존돼 있다.

| 실험 묶음 | 변경·학습량 | 관측과 판단 |
| --- | --- | --- |
| 초기 Linear grid | rank56/80, residual·Morton ON/OFF, λ .0064/.0256; 50k | residual ON과 Morton ON이 유용했고 rank80은 일관된 이득이 없었다. 사용자 제공 12개 표는 원본 eval 로그가 현재 로컬에 없음 |
| 초기 Nonlinear32 grid | rank56/80 × residual ON/OFF × 2λ; 50k, 8개 | rank56 + residual ON을 후속 부모로 사용 |
| Factorized continuation | 같은 Linear/Nonlinear 부모에서 +50k | context 구조 효과와 추가 학습 효과를 구분하는 대조군 |
| Linear context | Mean / Mean+Channel / Mean+Spatial / Full, +50k | Full은 matched factorized 대비 bytes −6.50%/.0064, −7.16%/.0256. Spatial의 기여가 컸음 |
| Nonlinear32 + context | Mean+Spatial / Full, +50k | Full은 nonlinear factorized 대비 bytes −6.09%/.0064, −7.99%/.0256, 화질도 개선 |
| Spatial predictor 확대 | P0 linear k3 / P1 residual k3 / P2 residual k7, +25k | 모델을 더 크게 만든 효과가 두 λ에서 일관되지 않음. 대조 P0도 추가 학습으로 RD 지점이 이동 |
| Probability-only | Shared / Split / Static Gaussian / Conditional Scale, +10k | 복원 경로를 고정. Split은 Shared보다 bytes −1.56%/.0064, −2.72%/.0256, 집계 화질 동일 |
| Full low-LR | 원래 Full 부모에서 LR 1e-6, +10k | 완료 arm의 개선이 0.1% 미만. .0064/P0는 checkpoint 저장 실패 |

### 주요 완료 지점

모두 rank56, residual/Morton ON이며 아래 표는 full-test 결과다.

| λ | 모델 | 학습 계보 | KiB/scene ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| .0064 | Nonlinear32 factorized 초기 | 50k | 98.183 | 24.5082 | .7665 | .2343 |
| .0064 | Nonlinear32 factorized 대조군 | 50k+50k | 87.840 | 24.3022 | .7602 | .2376 |
| .0064 | Nonlinear32 Full P0 | 50k+50k | 82.491 | 24.3238 | .7611 | .2371 |
| .0064 | **Nonlinear32 Full + Split** | 50k+50k+10k | **81.174** | **24.3238** | **.7611** | **.2371** |
| .0256 | Nonlinear32 factorized 초기 | 50k | 46.695 | 23.6943 | .7381 | .2673 |
| .0256 | Nonlinear32 factorized 대조군 | 50k+50k | 41.256 | 23.5210 | .7328 | .2693 |
| .0256 | Nonlinear32 Full P0 | 50k+50k | 37.958 | 23.5707 | .7342 | .2683 |
| .0256 | **Nonlinear32 Full + Split** | 50k+50k+10k | **36.920** | **23.5707** | **.7342** | **.2683** |

초기50k보다 continuation 이후 PSNR이 낮은 경우도 있다. 더 오래 학습했다고 같은 화질에서 bytes만 줄어든 것이 아니므로, 각 checkpoint의 용량과 화질을 함께 보존했다.

### Entropy·overhead 진단

| 진단 | 조건 | 결과 |
| --- | --- | --- |
| Fixed-symbol entropy probe | 128 TRAIN fit / 별도128 TEST; 복원·symbol 고정 | 단순 decoder-known score context table로 전체 bytes 약0.736%/.0064, 0.759%/.0256 절감. 재학습 codec의 full-test 결과는 아님 |
| Overhead profile, job423729 | 동일 GPU, warm-up4 + paired32 TEST, 18 checkpoint | Split의 score-path tensor 증가는 Shared 대비 약60.09/45.72 KiB, 절감 bytes로 상각되는 장면 수 약46.0/43.4. paired full codec decode는 81.68/77.71 ms |
| Scene CI | 기존 scene별 결과에 bootstrap을 적용하는 분석 도구 | 실행 방법 문서는 있으나 이번 로컬 자료에서 완료 결과 JSON/REPORT는 확인하지 못함 |

근거: [entropy probe](NFCGS_ENTROPY_PROBE_2026-09-14.md), [실측 overhead 보고서](../../../overhead-REPORT.md), [Scene CI의 설계·실행 방법](NFCGS_SCENE_CI_2026-09-15.md).

## 5. 최신 Score 단순화와 복호 단계 실험

### 5.1 minimal 설정

MSH residual 경로를 유지하면서 Score 쪽에서 아래 요소를 **동시에** 껐다.

| 요소 | 기존 Full + Split | minimal |
| --- | --- | --- |
| scene centering | scene mean을 빼고 736채널 FP16 mean 전송 | centering OFF, mean payload 0 B |
| channel-wise score norm | 학습한 채널별 scale | OFF |
| Mean context | mean으로 offset·step 예측 | OFF |
| Channel context | 앞 channel slice 참조 | OFF |
| Spatial context | Morton even → odd | 유지; 후속 실험에서 단계 세분화 |
| Nonlinear transform | hidden32 analysis/synthesis | OFF, linear 경로 |
| MSH residual | ON | ON, 활성 codec과 함께 학습 |
| GlobalSplat | pretrained 고정 | 동일 |

공통 시작점은 λ=.0256의 Full + Split checkpoint다. v11 GPU4, GPU당 batch2, accumulation1로 유효 batch8을 유지했고, LR1e-4로 추가50k를 학습했다.

### 5.2 단계 정의와 확인 상태

Morton 정렬 **이후 token 순번**을 기준으로 A=0 mod4, B=2 mod4, C=1 mod4, D=3 mod4다. 모든 token을 유지하며, 같은 단계의 token 값은 context에 넣지 않는다.

| 설정 | 복호 순서 | kernel / 학습 context 양자화 | 학습·eval job | 상태 |
| --- | --- | --- | --- | --- |
| minimal 기존2단계 | A+B → C+D | k3 / noise | 441307 / 441558 | 추가50k + full eval 완료 |
| minimal matched2단계 | A+B → C+D | k5 / STE | 결과 job ID 없음 | 실행 명령 전달, 완료 확인 전 |
| minimal3단계 | A → B → C+D | k5 / STE | 441552_1 / 441553_1 | 추가50k + full eval 완료 |
| minimal4단계 | A → B → C → D | k5 / STE | 결과 job ID 없음 | 실행 명령 전달, 완료 확인 전 |

3/4단계 구현은 앞 단계의 복호된 token 값만 참조한다. k5는 원래 sequence에서 ±2까지 볼 수 있다. Score hyperprior를 새로 추가한 실험은 아니다. STE는 학습 context에 양자화 복원값을 쓰는 방식이며 실제 bitstream 평가에서는 정수 양자화로 복호한다.

### 5.3 λ=.0256, full-test 6,991 scenes

| 모델 | KiB/scene ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 기존 Full 대비 bytes | 기존 Full 대비 PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 기존 Full + Split | 36.920 | 23.5707 | .7342 | .2683 | 기준 | 기준 |
| minimal2 / k3 / noise | 32.524 | 23.3109 | .7223 | .2845 | −11.91% | −0.2598 dB |
| **minimal3 / k5 / STE** | **30.807** | **23.4892** | **.7294** | **.2765** | **−16.56%** | **−0.0815 dB** |

3단계 설정은 기존 minimal2보다 전체 bytes **5.28% 감소**, Score bytes **7.71% 감소**, PSNR **0.1783 dB 증가**했다. SSIM/LPIPS도 개선됐다.

**해석 한계:** 이 비교에서는 단계 수(2→3), kernel(3→5), 학습 context 양자화(noise→STE)가 함께 바뀌었다. 3단계만의 이득을 확인하려면 2단계/k5/STE 대조 결과가 필요하다. 네 요소를 한 번에 제거했으므로 centering, norm, mean/channel context, nonlinear 각각의 기여도도 아직 분리하지 못했다.

### 5.4 바이트 분해

| 모델 | 전체 B | Score B | Residual y B | Residual z B | Mean B | Container B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 기존 Full + Split | 37,806.496 | 30,093.048 | 5,998.320 | 111.128 | 1,472 | 132 |
| minimal2 / k3 / noise | 33,304.705 | 21,508.002 | 11,540.893 | 123.810 | 0 | 132 |
| minimal3 / k5 / STE | 31,546.630 | 19,849.413 | 11,433.598 | 131.620 | 0 | 132 |

단순화 후 Score·mean 비용은 줄었지만 residual y 비용은 크게 늘었다. 전체 절감은 mean 1,472 B 제거만으로 설명되지 않으며, Score와 MSH 사이의 용량 배분도 달라졌다.

Eval 로그의 entropy decode 평균은 minimal2 81.6 ms, minimal3 80.8 ms다. 별도 부하 통제·반복 측정이 없어 3단계가 더 빠르다고 결론내리지는 않는다. 단계별 B/D 비트 절감 집계도 현재 다운로드 자료에 없다.

근거: [minimal 학습](../../../slurm/slurm-gs-score-minimal-441307.out), [minimal eval](../../../slurm/slurm-gs-score-minimal-eval-EEE441558.out), [3단계 학습](../../../slurm/slurm-gs-score-stages-441552_1.out), [3단계 eval](../../../slurm/slurm-gs-score-stages-eval-441553_1.out).

## 6. GlobalSplat + codec joint from-scratch

GlobalSplat을 고정하지 않고 codec과 함께 처음부터 학습했다. v10 GPU8, GPU당 batch1, accumulation1로 유효 batch8이었다. λ=.0064다.

| 모델 | 학습 job | Eval job | step | KiB/scene | PSNR | SSIM | LPIPS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 기존 pretrained Full + Split | 기존 계보 | 기존 full eval | codec 계보110k | 81.174 | 24.3238 | .7611 | .2371 |
| Joint from-scratch | 425420 | 426725 | 220,000 | 25.498 | 18.8065 | .5701 | .4366 |
| Joint 연장 | 426699 | 428944 | 총500,000 | 26.545 | 19.2332 | .5814 | .4142 |

220k→500k로 **PSNR +0.4267 dB** 개선됐지만 기존 기준보다 **5.0906 dB 낮았다**. bitstream도 훨씬 작으므로 같은 bitrate의 품질 비교는 아니다.

두 joint eval 모두 residual y와 z가 각각 **8 B**였다. residual stream이 매우 작게 사용되는 현상은 확인되지만, 이것만으로 모든 residual parameter가 0이라거나 from-scratch 학습 자체가 불가능하다고 단정할 수 없다. 현재 결론은 **이번 initialization·loss·schedule 조합이 원하는 화질에 도달하지 못했다**는 것이다.

근거: [220k 학습](../../../slurm/slurm-gs-joint-v10-425420.out), [500k 연장](../../../slurm/slurm-gs-joint-ext-v10-426699.out), [220k eval](../../../slurm/slurm-gs-main-eval-426725.out), [500k eval](../../../slurm/slurm-gs-joint500k-eval-428944.out).

## 7. b 값 진단과 b_mean 제거 재학습

### 7.1 서로 다른 세 작업

| 작업 | 실제로 바꾸거나 측정한 것 | 상태 |
| --- | --- | --- |
| b diagnostics | 기존 Full + Split에서 예측 offset b와 step d의 크기, prediction gain 측정 | 428943 / 429783, 32-scene 진단 완료 |
| b_mean ON/OFF 재학습 | mean conditioner의 **offset head만** 비교. scene mean 전송, step d, channel/spatial, MSH 유지 | 433719와 434543 재시도 모두 저장 공간 오류 |
| minimal | scene centering·mean 전송 자체를 없애고 norm·mean/channel context·nonlinear도 함께 OFF | 441307 완료; b_mean만의 ablation이 아님 |

`b_mean`은 Score를 예측하는 offset이다. 1,472 B로 전송하는 scene mean feature와 서로 다른 값이다. b는 수신기가 가진 mean·복호 context·공유 weight로 계산하므로 b 벡터 자체를 별도 payload로 보내는 것은 아니다.

### 7.2 기존 모델의 32-scene 진단

λ=.0064 Full + Split, 추가 학습 없음. 해당 subset의 PSNR은 24.8145, 전체 크기는 80.879 KiB다. full-test 수치와 직접 비교하지 않는다.

| 측정량 | 결과 |
| --- | ---: |
| b_mean 관측 범위, 모든 scene/channel | −14.5529 ~ 8.3589 |
| step d 관측 범위 | 0.3140 ~ 3.9276 |
| effective b_even pooled MAE / RMS | 1.0847 / 2.0565 |
| effective b_odd pooled MAE / RMS | 2.7194 / 7.0140 |
| odd Score RMS → 전체 context를 뺀 RMS | 7.5608 → 3.1050, 58.93% 감소 |
| b_mean만 뺀 odd RMS | 7.7505, 원 Score보다 약2.51% 증가 |

이 값은 score norm을 적용한 Score 공간의 값이다. b는 scene/channel/token과 참조 가능한 context에 따라 달라지며, 단일 상수가 아니다. odd 전체 예측에는 상당한 효과가 있었지만 **b_mean만의 역할은 작거나 방향이 불리한 통계도 있었다**. MAE/RMS 변화가 실제 coding bits 변화와 같지는 않고, step d의 기여도 남아 있다.

### 7.3 재학습 ablation의 완료 여부

- 두 arm은 공통 Full + Split .0064 부모에서 시작하고 b_mean head를 0으로 초기화했다. baseline은 다시 학습하고, no_mean_offset은 b_mean을 0으로 유지하는 설계였다. 추가20k, 각 GPU1, batch2×accumulation4로 유효 batch8이다.
- **433719_0/1:** progress는 20k에 도달했지만 final checkpoint 저장에서 `Errno 28` 발생. 성공한 최종 checkpoint나 후속 평가 결과는 확인되지 않았다.
- **434543_0/1:** 재시도 중 `/tmp/pymp-*` 생성에서 `Errno 28`, 학습 초반에 취소. 완료되지 않았다.
- 따라서 이 두 arm을 근거로 b_mean 제거의 성능을 결론내릴 수 없다.

근거: [b 수치 원본 JSON](../../../score_context_b_summary.json), [prediction 진단 로그](../../../slurm/slurm-gs-score-pred-429783.out), [ablation 설계](NFCGS_MEAN_OFFSET_ABLATION.md), [433719 오류](../../../slurm/slurm-gs-mean-offset-433719_0.err), [434543 오류](../../../slurm/slurm-gs-mean-offset-434543_0.err).

## 8. Encoder / decoder 별도 job 평가

기존 Full + Split .0064, b는 기존 setting ON으로 유지했다.

1. **433727 encoder job:** 전체 scene을 bitstream 파일로 저장하고 종료.
2. **433728 decoder job:** 저장된 bitstream을 읽어 복호하고 Gaussian 생성·렌더링·평가.

| 항목 | 결과 |
| --- | ---: |
| Encoder / decoder 처리 장면 | 각각6,991 |
| 평균 bytes | 83,121.710 B = 81.174 KiB |
| Decoder PSNR / SSIM / LPIPS | 24.323778 / .761096 / .237059 |
| Encoder 계산, scene encoder + entropy encode | 223.13 ms/scene |
| Bitstream write | 526.02 ms/scene |
| Encoder 계산 + 파일 저장 | 749.15 ms/scene |
| Bitstream read | 73.91 ms/scene |
| Decoder 계산, entropy decode + Gaussian decoder | 120.86 ms/scene |

분리 실행 후에도 기존 full eval의 화질을 재현했다. 파일 저장/읽기 시간은 Ceph I/O를 포함하므로 순수 codec 계산 시간과 구분했다. 이 결과는 encoder 프로세스가 만든 실제 파일만으로 decoder 프로세스가 평가를 완료했다는 근거다.

서버 bitstream 경로:

```text
outputs/nfcgs_main/sender_receiver_full/existing_b/20260927_171640/bitstreams
```

근거: [encoder_summary.json](../../../encoder_summary.json), [decoder_summary.json](../../../decoder_summary.json), [encode 로그](../../../slurm/slurm-gs-encode-full-433727.out), [decode 로그](../../../slurm/slurm-gs-decode-full-433728.out).

## 9. 초기 Nonlinear32 50k의 100-scene 재평가

원래 nonlinear32/rank56/residual ON 50k checkpoint를 사용했다. test index에서 validation 128개를 제외하고 고정 seed111123으로 100개를 선택하는 별도 subset 평가다.

| Job | λ | 상태 | 결과 |
| --- | ---: | --- | --- |
| 436182 | .0256 | 실패 | 11-frame scene에서 C12+T8의 분리된20 frame을 만들 수 없어 종료 |
| 436236 | **.0064** | 100개 완료 | PSNR24.6772, SSIM.7634, LPIPS.2389, 101,396.88 B = 99.020 KiB |

성공한 로그는 짧은 scene을 미리 제외했다. 최초 요청했던 **λ=.0256의 성공 결과는 현재 자료에 없다**. 재시도436236은 실제 checkpoint와 output 모두 .0064다. 또한 이100개는 full-test 6,991개와 다른 표본이다.

근거: [.0256 실패](../../../slurm/slurm-gs-nonlinear50k-100-436182.err), [.0064 완료](../../../slurm/slurm-gs-nonlinear50k-100-436236.out).

## 10. 같은 workspace의 별도 CleanToken / Hyper1D·MSH 실험

아래 로그의 서버 작업 디렉터리는 `/ceph_data/clue9986/CleanToken/upstream/globalsplat`이다. 주력 `tokencomp` Score 계열과 별도의 학습 계보다.

pretrained GlobalSplat은 고정하고 새 Hyper1D codec을 초기화했다. 입력은 appearance512 + observable geometry224이며 input norm은 none이다. single MSH는 feature 전체를 한 경로로 압축하고, dual MSH는 base 복원과 residual을 두 MSH로 압축한다. `lowrank_base`는 base 쪽 rank56 변환을 넣은 별도 변형으로, 기존 Full Score + Split을 그대로 사용한 모델이 아니다.

### 10.1 완료 학습과 final validation

validation 표의 수치는 train 로그 마지막 validation 집계다. full-test 수치와 비교하지 않는다.

| 설정 | Job | step | λ | Morton | Final val PSNR | Final val LPIPS | Final val bytes |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| Legacy single MSH pilot B | 434938 | 16k | .0256 | OFF | 18.2211 | .4912 | 5,810.281 |
| Legacy single MSH 별도 초기화 | 434951 | 16k | .0256 | OFF | 18.0395 | .5051 | 5,705.156 |
| 위 모델 이어 학습 | 436008 | 총32k | .0256 | OFF | 19.5891 | .4401 | 7,635.656 |
| 위 모델 이어 학습 | 436859 | 총50k | .0256 | OFF | 20.2361 | .3936 | 8,835.594 |
| Plain4, adapter 없음 | 436321 | 50k | .0256 | OFF | 18.7142 | .4572 | 5,735.969 |
| Legacy single MSH | 437624_0 | 50k | .0128 | OFF | 19.9112 | .3988 | 11,057.531 |
| Plain4 | 437624_1 | 50k | .0128 | OFF | 17.7147 | .5124 | 7,926.781 |
| Legacy single MSH | 437624_2 | 50k | .0064 | OFF | 20.1607 | .4027 | 13,071.031 |
| Plain4 | 437624_3 | 50k | .0064 | OFF | 19.0721 | .4396 | 7,322.406 |
| Single MSH | 438115_0 | 50k | .0256 | ON | 20.8614 | .3551 | 9,928.812 |
| Dual MSH, base+residual | 438115_1 | 50k | .0256 | ON | 21.6057 | .3338 | 10,716.656 |
| Dual MSH, base+residual | 441056 | 50k | .0256 | OFF | 20.7853 | .3712 | 9,231.531 |
| Low-rank base56 + MSH | 441332 | 50k | .0256 | ON | 로컬 집계 없음 | 로컬 집계 없음 | 로컬 집계 없음 |

마지막 모델은 대응하는 full eval이 있어 아래 표에 결과를 적었다. 고화질 λ sweep의 네 학습은 50k를 완료했지만, 현재 자료에는 그 full-test eval 결과가 없다.

### 10.2 λ=.0256의 full-test 6,991 scenes

| 구조 | Morton | Train / eval job | KiB/scene | PSNR | SSIM | LPIPS |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Single MSH | OFF | 436859 / 441057 | 8.594 | 20.1189 | .6000 | .4030 |
| Single MSH | ON | 438115_0 / 440685_0 | 9.712 | 20.7369 | .6290 | .3660 |
| Dual MSH | OFF | 441056 / 442219_1 | 8.961 | 20.6579 | .6209 | .3827 |
| Dual MSH | ON | 438115_1 / 440685_1 | 10.462 | 21.4372 | .6521 | .3446 |
| Low-rank base56 + MSH | ON | 441332 / 442219_0 | 8.973 | 20.9457 | .6340 | .3730 |

이 별도 계열에서 Morton ON의 single/dual은 OFF보다 화질이 높고 bytes도 많았다. Dual ON은 Single ON보다 PSNR +0.7003 dB, bytes 약7.72% 증가였다. 주력 Full이나 minimal3보다 훨씬 작은 bitrate이므로 화질만으로 동일 RD 지점의 우열을 판단할 수 없다. z side bits도 상당한 비용을 차지하며, stream별 내역을 추출 JSON에 보존했다.

### 10.3 Early subset eval

| Checkpoint | Eval job | 선택 수 / 실제 처리 수 | PSNR | KiB/scene |
| --- | --- | --- | ---: | ---: |
| Legacy Morton OFF 16k | 436141 | 32 / 31 | 18.8527 | 6.035 |
| Legacy Morton OFF 23k | 436157 | 32 / 31 | 18.8870 | 6.338 |
| Legacy Morton OFF 23.5k | 436171 | 100 / 97 | 18.7793 | 6.030 |

32/100개를 선택했어도 실제 집계는 31/97개였다. 완료 지표와 최종 progress에서 확인한 수이며, 주력 모델의 32-scene b 진단과 다른 표본·모델이다.

근거: [MSH 비교 설계](NFCGS_HYPER1D_MSH_COMPARE_50K.md), [λ sweep 설계](NFCGS_HYPER1D_HIGH_QUALITY_LAMBDA.md), [Single ON full eval](../../../slurm/slurm-gs-msh-fulltest-440685_0.out), [Dual ON full eval](../../../slurm/slurm-gs-msh-fulltest-440685_1.out), [Low-rank base full eval](../../../slurm/slurm-gs-hyper1d-50k-eval-442219_0.out), [Dual OFF full eval](../../../slurm/slurm-gs-hyper1d-50k-eval-442219_1.out). 개별 학습의 validation 수치는 [이번 추출 JSON](SESSION_EXPERIMENT_SUMMARY_2026-10-06_metrics.json)에 있다.

## 11. 실패·취소·확인 전 실행

아래 작업은 성능 결과를 얻은 실험으로 세지 않는다.

| Job / 시도 | 기록된 원인·상태 |
| --- | --- |
| 424117, 428792 | 본 처리 전에 cancelled, 성능 집계 없음 |
| 424118 v12 joint | 5k checkpoint 저장 시 `No space left on device` |
| 424577 / 424688 v10 | 20-step 저장 probe 완료. 품질 평가용 학습 아님 |
| 425438 extension | `resolve_nfcgs_slurm_checkpoint.py` 미동기화로 시작 실패. 426699에서 재실행 완료 |
| 428798 b probe | `NameError: name 'codec' is not defined`. 428943에서 재실행 완료 |
| 433719_0/1 b_mean train | 20k 도달 후 checkpoint 저장 실패 |
| 434543_0/1 b_mean retry | `/tmp/pymp-*` 생성에서 Errno28, 초반에 중단 |
| 436182 nonlinear test100 .0256 | context/target용 frame 부족 |
| 434935 Hyper1D | 필수 checkpoint 인자 누락 |
| 434941 Hyper1D | 기존 initial checkpoint 덮어쓰기를 거부 |
| 436140 Hyper1D eval | 0/32 상태에서 cancelled. 436141에 완료 기록 있음 |
| 438123_0/1 MSH 재실행 | Errno28. 438115_0/1의 완료 학습과 별도 출력 |
| 440681_0/1 MSH eval | `conda: command not found`. 440685_0/1에서 재실행 완료 |
| Score context 완전 OFF안 | 롤백 지시 있음. 실행·완료 결과 없음 |
| minimal2 k5/STE·minimal4 k5/STE | 제출 명령을 전달했지만 현재 자료에 job ID·결과 없음 |

Slurm의 node/GRES, GPU type 미지정, shebang이나 동기화 누락으로 생긴 제출 오류는 실행 환경의 문제다. 모델 성능에 대한 부정적 결과는 아니다. 또한 `Errno28`은 checkpoint 저장과 node의 `/tmp`에서 각각 관측됐으므로 모두 같은 quota 문제라고 단정할 수 없다.

## 12. 그 이전의 탐색 실험

주력 context 계보 이전에도 아래 탐색이 있다. 초기 아카이브의 76개 설정은 SGA+ 등 test-time 최적화를 포함하므로 현재의 zero-update full-test 결과와 직접 비교하지 않는다.

| 세대·실험 | 검증한 질문 | 확인된 주요 사항 |
| --- | --- | --- |
| 초기76개 설정 아카이브 | 2D/1D, shared/scene basis, rank, tied/untied, nonlinear, codec 위치, decoder adapter, e2e 범위 | 현행 구조의 후보 탐색. 현재 active path와 별도. 76은 설정 수이며 이번에 확인한 완료 job 수가 아님 |
| Ablation17 | observable 변환→low-rank→양자화→residual의 누적 비교 | rank56 projection에서 −4.0791 dB, residual 추가에서 +3.1132 dB. 양자화 전 표현 손실도 큼 |
| Analysis16 | channel covariance·low-rank spectrum·token lag 상관 | observable concat의 Morton lag1 상관0.4739, 미정렬에서는 작음. 높은 에너지 보존율만으로 화질을 보장하지 못함 |
| Ablation14 | 같은 fresh50k budget에서 linear/no residual, nonlinear/no residual, linear/residual | nonlinear/no residual은 linear/no residual보다 +0.8882 dB이고 bytes도 감소. PCA 초기화를 쓰는 후속 grid와 별도 recipe |
| Ablation15 | 순서 의존 module 없는 추론 재정렬 | none/Morton/random에서 집계 화질 동일 |
| Ablation15b | Morton 학습된 1D residual의 추론 재정렬 | Morton이 유리. 추론 순서 민감도이며 각 순서로 재학습한 비교는 아님 |
| 기존 full RD sweep | λ .0004/.0016/.0064/.0256 | 4개 지점의 full-test 측정 있음. 후속 Full+Split이나 minimal3의 4점 RD 곡선은 아직 없음 |

세부 결과와 원 표: [초기·통합 ablation 기록](ABLATION_PROGRESS_2026-09-02.md), [아카이브 전체 지도](../../../EXPERIMENTS_OVERVIEW.md).

## 13. 현재 비교에 남기는 후보와 미해결 사항

1. **Full + Split**: 기존 화질·압축·encoder/decoder 분리 동작의 기준.
2. **minimal3 / k5 / STE**: MSH를 유지한 작은 용량의 후보. 기존 기준보다 화질이 조금 낮고 용량도 작음.
3. **minimal2 / k5 / STE**: 3단계 효과를 분리하기 위해 필요한 대조군. 현재 결과 확인 전.
4. **minimal4 / k5 / STE**: C→D 추가 참조의 효과와 복호 비용을 확인할 후보. 현재 결과 확인 전.

미해결 사항은 같은 kernel/STE에서의 2·3·4단계 비교, 단계별 실제 bits, 제거 요소 각각의 효과, 여러 λ의 실제 RD 곡선, 여러 seed에서의 재현성이다. 실패한 b_mean20k ablation의 결과는 아직 없다. joint의 낮은 성능도 from-scratch 학습 전체에 대한 결론으로 확장하지 않는다.

### 주력 checkpoint 기록

서버 project root `/ceph_data/clue9986/tokencomp/upstream/globalsplat` 기준 상대 경로:

```text
# Full + Split .0256 (.0064는 경로의 두 lambda 부분을 교체)
outputs/nfcgs_score_probability10k/20260913_112316/e1_split/rank56/lambda0p0256/residual_on/checkpoints/nfcgs_score_probability10k_e1_split_rank56_lambda0p0256_residual_on_morton_on_m1c1s1_split/version_0/step000010000.ckpt

# minimal2 / k3 / noise
outputs/nfcgs_score_path_ablation/minimal/lambda0p0256/job_441307/checkpoints/nfcgs_main/version_0/step000050000.ckpt

# minimal3 / k5 / STE
outputs/nfcgs_score_stages/lambda0p0256/job_441552/stages3/checkpoints/nfcgs_main/version_0/step000050000.ckpt

# joint500k
outputs/nfcgs_main/joint_extension/lambda0p0064/from_job_425420_to_500000/checkpoints/nfcgs_main_joint/version_0/step000500000.ckpt
```

이 경로들은 로그에 기록된 경로다. 과거 서버 outputs 정리 이후에도 현재 남아 있는지는 이번 자료로 확인하지 않았다.

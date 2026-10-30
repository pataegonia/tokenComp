# GlobalSplat / NFC-GS 이전 실험 전체 종합 정리

통합일: **2026-10-24 (Asia/Seoul)**. 기존 완료 실험의 근거는10월20~21일 다운로드 로그와 보존된 과거 결과다. **이번에 진행 중인 새 fresh-init RD 설정과 그 Stage A/B·120k 연장·monitor 복구는 집계 대상에서 제외한다.** 이전의50k 구조 pilot4개는 종전 실험에 포함한다.

초기 rank·projection 탐색, linear/nonlinear·residual·context 학습, 정렬·entropy schedule RD, 대조군과 추가 seed, 구조 pilot, 경로·정렬·수신 진단, 복잡도·시간 측정, 실패·재개 이력을 이 문서 하나에 모았다. 원 종합본에서 미완료로 남아 있던 None+L3 OFF12 .0512, H+D4 OFF16 .0128, Score736-only의 완료 결과를 반영했다. 기존 snapshot 문서는 수정하지 않았다.

1. [완료 현황과 집계 범위](#status)
2. [공통 모델·평가 조건과 명칭](#protocol)
3. [전체 실험 계보](#lineage)
4. [기존26개 학습·평가 목록](#recent)
5. [주 seed RD 전체65점·17계열](#rd)
6. [λ0·두 seed D4 ON/OFF·구조 pilot](#special)
7. [비압축·경로·정렬·독립 수신 진단](#diagnostics)
8. [복잡도·시간·메모리](#complexity)
9. [초기 실험·과거 context48개](#legacy)
10. [기타 평가와 CleanToken 별도 분기](#other)
11. [실패·재개 이력](#failures)
12. [기존 후속 계획과 결과 미확인 항목](#pending)
13. [과거 Slurm 로그 색인](#ledger)
14. [출처와 검증 범위](#sources)

<a id="status"></a>

## 1. 완료 현황과 집계 범위

기존26개 묶음은 **26개 모두 최종50k 평가까지 완료**됐다. 이 묶음의 완료 평가는 **전체 test6991장22개 + train-derived holdout128장4개**다. 실패한 학습을 복구한 job은 같은 모델의 이력으로 기록하며 독립 실험으로 더하지 않는다.

| 범위 | 완료 기록 | 평가 집합과 해석 |
|---|---:|---|
| 정렬·schedule·context 주 seed, 양의λ | **65점·17계열** | 전체 test6991; H+D4 OFF16의 한 점을 별도 계열로 포함 |
| λ=0 |1점| 실제 양자화·entropy coding 유지; 비압축 기준 아님 |
| H+D4 추가 seed |2점| ON/OFF 각1점, λ=.0128; 같은 warm-start 부모 공유 |
| 구조 pilot |4점| 같은 train-derived holdout128; 전체 test와 분리 |
| 위 네 범위 합계 |72개 모델별 완료 평가 기록| 평가 집합·계보가 달라 단일 RD 곡선으로 합치지 않음 |
| 기존26개 제출 묶음 |26개 완료| 위72개와 겹치는 부분집합 |
| 복잡도 profile |37 checkpoint| 별도 측정; 학습37개를 추가했다는 뜻 아님 |
| 과거 context 보존 JSON |48개 평가 기록|9.3절에 전부 수록; 현재 원본 stdout가 없어 보존 자료 기준 |
| 초기 Linear 사용자 표 |12개 조건|9.2절; 원래 kB 단위를 유지 |
| 기타 현존 평가 |37개 기록|10절; subset·다른 계보·중복 checkpoint 포함 |

집계 범위끼리 겹치므로 위 숫자를 더해 독립 학습의 총수로 사용하지 않는다. 초기 탐색·분석 전용 작업·실패한 시도까지 아래 각 절에 포함하되 완료 모델과 구분한다.

<a id="protocol"></a>

## 2. 공통 모델·평가 조건과 명칭

| 항목 | 현행 조건 |
|---|---|
| 기반 모델 | 공개 GlobalSplat RE10K-32K; 4,096 scene tokens |
| 압축 경계 | appearance512 + raw geometry512→observable geometry224 = F736 |
| 기본 codec | rank56 LR/Score 복원 + MSH residual; MSH N192/M320/adapter96 |
| 현행 minimal | centering·Score norm·mean conditioning·channel context OFF, linear transform; 공간 Context는 별도 스위치 |
| Gaussian | stage3, token당8개 = scene당32,768개; M_max16과 구분 |
| 전체 test | RE10K C12/T8, 256×256, BF16, eval seed0, 실제 처리6,991장 |
| 실제 용량 | Score + MSH y/z + mean + 모든 container/header; 평균 B/scene ÷1,024 = KiB/scene |
| 공유 모델 | checkpoint/신경망 weight는 scene payload에서 제외 |
| 평가 중 최적화 | 현행 평가에서 per-scene fitting 또는 test-time 학습 없음 |
| 후속50k의 일반 설정 | GPU2 × batch/GPU2 × accumulation2 = 유효 batch8, LR1e-4, 35k/45k decay |
| 학습 초기화 | 기존 codec 대조군은 부모 weights warm start; 구조 pilot은 공개 backbone/QR 경계에서 codec fresh init |
| 학습되는 경계 | codec 학습에는 geometry projection이 포함됨; backbone/Gaussian decoder는 고정 |

전체 test loader의 표기상 분모는 7,286일 수 있지만 실제 유효 처리 수는 6,991이다. 시간 측정의 warmup5를 뺀 6,986회는 평가 장면 수가 아니다. 구조 pilot의 holdout128, 진단256, profile32, 옛 subset31/32/97/100 결과를 전체 test와 섞지 않는다.

`context_only`는 **기본 token 순서(None)+공간 Context ON인 two-path 모델**이다. Score-only를 뜻하지 않는다. `hilbert_only`는 **Hilbert+L3 OFF12**이며 MSH를 유지한다. 공간 Context OFF는 LR 또는 MSH 제거와 다르다.

| 표기 | Token 단계 | 채널 그룹 | Score strings | 공간 예측 |
|---|---|---|---|---|
| OFF1 | 전체4096 한 단계 | 전체56 | 1 | OFF |
| OFF4 | D4 4단계 | 단계마다 전체56 | 4 | OFF |
| L3 OFF12 | legacy 3단계 | 16/16/16/8 | 12 | OFF |
| D4 OFF16 | D4 4단계 | 16/16/16/8 | 16 | OFF |
| L2 ON8 / Q2 ON8 | 각2단계; 참조 topology는 다름 | 16/16/16/8 | 8 | ON |
| L3 ON12 | 0mod4 → 2mod4 → 홀수 | 16/16/16/8 | 12 | ON |
| D4 ON16 | 0mod8 → 4mod8 → 2mod4 → 홀수 | 16/16/16/8 | 16 | ON |

D4의 token 수는 512/512/1024/2048이며 실제 anchor 간격은 ±4/±2/±1이다. 오른쪽 anchor가 없으면 왼쪽을 복제한다. Legacy4와 D4는 다른 schedule이다. 정렬은 Score와 MSH 양쪽에 적용된다.

SCCTX001 Score wrapper는 `24+8K`B이며 K는 strings 수다. D4 ON16/OFF16은152B다. `actual_score_bytes`에 이 wrapper가 이미 포함되므로 순수 Score와 모든 header를 분리할 때만 이동한다.
<a id="lineage"></a>

## 3. 전체 실험 계보

```text
초기 rank / projection / residual / token 순서 탐색
  └─ Linear / Nonlinear factorized + context 추가학습
       ├─ P0 / P1 / P2 predictor
       ├─ 복원 고정 Shared / Split / Gaussian / Conditional probability-only
       ├─ low-LR 후속 / 고정-symbol entropy probe
       └─ Full P0+Split 부모에서 weights-only 후속50k
            └─ None / Morton / Hilbert / NN-score
                 × L2 / L3 / legacy4 / Q2 / D4 / spatial OFF
                 ├─ 주 seed 양의λ65점, λ0 한 점
                 ├─ stage/group을 맞춘 OFF12 / OFF16
                 ├─ D4 ON/OFF 추가 seed
                 └─ anchor / B0 / T05 / T06 / T15 / complexity 진단

공개 backbone + 공통 QR 경계 + 새 codec 초기화
  └─ 기존50k 구조 pilot4개, holdout128
       Score56-only / MSH-only / two-path56 / Score736-only

별도 CleanToken repository
  └─ Hyper1D single / dual / lowrank-base / Morton / λ 실험
```

앞의 warm-start RD와 기존 구조 pilot은 초기화와 평가 집합이 다르다. 추가 seed도 같은 부모 가중치를 공유하므로 독립 fresh-init 반복으로 해석하지 않는다. CleanToken 분기는 구현·압축 경계·계보가 달라 별도 표로 유지한다.

<a id="recent"></a>

## 4. 기존26개 학습·평가 전체 목록

이 표는 기존26개의 최종 완료 기록이다. 여기서 완료된 전체 test 결과는 뒤의 RD·λ0·seed 표와 중복된다. `→`는 실패한 원래 train에서 새 full-state resume job으로 이어졌다는 뜻이다.

| 묶음 | 설정 | λ | Train / 재개 | Eval | 최신 상태 |
|---|---|---|---|---|---|
| rd_extensions | H+D4 ON16 | 0 | 455327 | 455328 | 완료 / test6991 |
| rd_extensions | H+D4 ON16 | 0.0032 | 455329 | 455330 | 완료 / test6991 |
| rd_extensions | M+D4 ON16 | 0.0032 | 455331 | 455332 | 완료 / test6991 |
| rd_extensions | H+L3 ON12 | 0.0032 | 455333 | 455334 | 완료 / test6991 |
| rd_extensions | M+L3 ON12 | 0.0032 | 455335 | 455336 | 완료 / test6991 |
| rd_extensions | None+D4 ON16 | 0.0032 | 455337 | 455338 | 완료 / test6991 |
| rd_extensions | H+OFF1 | 0.0032 | 455339 | 455340 | 완료 / test6991 |
| rd_extensions | H+D4 ON16 | 0.0512 | 455341 | 455342 | 완료 / test6991 |
| rd_extensions | M+D4 ON16 | 0.0512 | 455343 | 455344 | 완료 / test6991 |
| rd_extensions | H+L3 ON12 | 0.0512 | 455928 | 455929 | 완료 / test6991 |
| rd_extensions | M+L3 ON12 | 0.0512 | 455930 | 455931 | 완료 / test6991 |
| rd_extensions | None+D4 ON16 | 0.0512 | 455932 | 455933 | 완료 / test6991 |
| rd_extensions | H+OFF1 | 0.0512 | 455934 | 455935 | 완료 / test6991 |
| controls | None+L3 ON12 | 0.0032 | 456895 | 456896 | 완료 / test6991 |
| controls | H+L3 OFF12 | 0.0032 | 456897 | 456898 | 완료 / test6991 |
| controls | None+L3 OFF12 | 0.0032 | 456899 | 456900 | 완료 / test6991 |
| controls | None+L3 ON12 | 0.0512 | 456901 | 456948 | 완료 / test6991 |
| controls | H+L3 OFF12 | 0.0512 | 456949 | 456950 | 완료 / test6991 |
| controls | None+L3 OFF12 | 0.0512 | 456951 → 458782 | 457098 | 완료 / test6991 |
| controls | H+D4 OFF16 | 0.0128 | 457099 → 458783 | 457100 | 완료 / test6991 |
| architecture | score56_only | 0.0128 | 457741 | 457742 | 완료 / holdout128 |
| architecture | msh_only | 0.0128 | 457743 | 457744 | 완료 / holdout128 |
| architecture | two_path56 | 0.0128 | 457745 | 457746 | 완료 / holdout128 |
| architecture | score736_only | 0.0128 | 457828 | 457829 | 완료 / holdout128 |
| seed_pair | H+D4 on16 추가 seed | 0.0128 | 457786 | 457787 | 완료 / test6991 |
| seed_pair | H+D4 off16 추가 seed | 0.0128 | 457788 | 457789 | 완료 / test6991 |

구조 초기화 job457740은 공통 holdout128 선정과 네 arm 초기화를 완료했다. 구조 네 arm의 별도 전체6991장 test job은 아직 제출한 기록이 없다. 기존 seed·대조군의 full test와 혼동하지 않는다.
<a id="rd"></a>

## 5. 주 seed RD 전체65점·17계열

양의λ는 .0032/.0064/.0128/.0256/.0512다. None/Hilbert×L3 OFF/ON 네 곡선은 각각5점으로 총20점이다. H+D4 OFF16은 λ=.0128의 한 점이므로 ON과의 단일 점 대조로만 사용한다. λ0와 추가 seed는6절에 분리한다.

| Family | 설정 | 완료λ | 점 수 |
|---|---|---|---:|
| both_off | None + L3 OFF12 | 0.0032, 0.0064, 0.0128, 0.0256, 0.0512 | 5 |
| context_dyadic4 | None + D4 ON16 | 0.0032, 0.0064, 0.0128, 0.0256, 0.0512 | 5 |
| context_only | None + L3 ON12 | 0.0032, 0.0064, 0.0128, 0.0256, 0.0512 | 5 |
| context_quarter2 | None + Q2 ON8 | 0.0064, 0.0128, 0.0256 | 3 |
| hilbert_dyadic4 | Hilbert + D4 ON16 | 0.0032, 0.0064, 0.0128, 0.0256, 0.0512 | 5 |
| hilbert_dyadic4_off16 | Hilbert + D4 OFF16 | 0.0128 | 1 |
| hilbert_legacy3 | Hilbert + L3 ON12 | 0.0032, 0.0064, 0.0128, 0.0256, 0.0512 | 5 |
| hilbert_off1 | Hilbert + OFF1 | 0.0032, 0.0064, 0.0128, 0.0256, 0.0512 | 5 |
| hilbert_off4 | Hilbert + OFF4 | 0.0064, 0.0128, 0.0256 | 3 |
| hilbert_only | Hilbert + L3 OFF12 | 0.0032, 0.0064, 0.0128, 0.0256, 0.0512 | 5 |
| hilbert_quarter2 | Hilbert + Q2 ON8 | 0.0064, 0.0128, 0.0256 | 3 |
| morton_dyadic4 | Morton + D4 ON16 | 0.0032, 0.0064, 0.0128, 0.0256, 0.0512 | 5 |
| morton_legacy2 | Morton + L2 ON8 | 0.0064, 0.0128, 0.0256 | 3 |
| morton_legacy3 | Morton + L3 ON12 | 0.0032, 0.0064, 0.0128, 0.0256, 0.0512 | 5 |
| morton_legacy4 | Morton + legacy4 ON16 | 0.0256 | 1 |
| morton_quarter2 | Morton + Q2 ON8 | 0.0064, 0.0128, 0.0256 | 3 |
| nn_score | NN-score + L3 ON12 | 0.0064, 0.0128, 0.0256 | 3 |

| 설정 | λ | PSNR | SSIM | LPIPS | KiB/scene | Checkpoint run | Eval |
|---|---|---|---|---|---|---|---|
| None + L3 OFF12 | 0.0032 | 23.7435 | 0.7385 | 0.2684 | 67.29040 | 456899 | 456900 |
| None + L3 OFF12 | 0.0064 | 23.6356 | 0.7343 | 0.2726 | 56.58601 | 454665 | 454666 |
| None + L3 OFF12 | 0.0128 | 23.5784 | 0.7324 | 0.2751 | 47.46689 | 454667 | 454668 |
| None + L3 OFF12 | 0.0256 | 23.3517 | 0.7253 | 0.2823 | 38.82390 | 454669 | 454670 |
| None + L3 OFF12 | 0.0512 | 22.9042 | 0.7114 | 0.2980 | 30.14873 | 456951 | 457098 |
| None + D4 ON16 | 0.0032 | 23.7153 | 0.7370 | 0.2693 | 67.20204 | 455337 | 455338 |
| None + D4 ON16 | 0.0064 | 23.6162 | 0.7329 | 0.2719 | 55.73304 | 453068 (recovery) | 453069 |
| None + D4 ON16 | 0.0128 | 23.5838 | 0.7326 | 0.2748 | 46.73730 | 451868 (recovery) | 451869 |
| None + D4 ON16 | 0.0256 | 23.3711 | 0.7255 | 0.2816 | 38.30672 | 451870 (recovery) | 451871 |
| None + D4 ON16 | 0.0512 | 22.9360 | 0.7120 | 0.2951 | 29.49763 | 455932 | 455933 |
| None + L3 ON12 | 0.0032 | 23.7288 | 0.7371 | 0.2697 | 67.64892 | 456895 | 456896 |
| None + L3 ON12 | 0.0064 | 23.6369 | 0.7344 | 0.2726 | 55.88247 | 451809 (recovery) | 451810 |
| None + L3 ON12 | 0.0128 | 23.5665 | 0.7319 | 0.2756 | 46.52462 | 451811 (recovery) | 451812 |
| None + L3 ON12 | 0.0256 | 23.3521 | 0.7251 | 0.2825 | 38.26433 | 451813 (recovery) | 451814 |
| None + L3 ON12 | 0.0512 | 22.9142 | 0.7104 | 0.2983 | 29.40673 | 456901 | 456948 |
| None + Q2 ON8 | 0.0064 | 23.6703 | 0.7358 | 0.2718 | 55.67799 | 453064 (recovery) | 453065 |
| None + Q2 ON8 | 0.0128 | 23.5874 | 0.7327 | 0.2746 | 46.80049 | 451823 (recovery) | 451824 |
| None + Q2 ON8 | 0.0256 | 23.3691 | 0.7255 | 0.2817 | 38.40935 | 453066 (recovery) | 453067 |
| Hilbert + D4 ON16 | 0.0032 | 23.8965 | 0.7433 | 0.2590 | 56.59803 | 455329 | 455330 |
| Hilbert + D4 ON16 | 0.0064 | 23.8170 | 0.7403 | 0.2623 | 46.74118 | 447744 | 447745 |
| Hilbert + D4 ON16 | 0.0128 | 23.7021 | 0.7363 | 0.2673 | 37.17881 | 447746 | 447899 |
| Hilbert + D4 ON16 | 0.0256 | 23.5173 | 0.7298 | 0.2752 | 29.39856 | 447900 | 447901 |
| Hilbert + D4 ON16 | 0.0512 | 23.0675 | 0.7146 | 0.2942 | 21.90789 | 455341 | 455342 |
| Hilbert + D4 OFF16 | 0.0128 | 23.6338 | 0.7338 | 0.2695 | 44.61925 | 457099 → 458783 | 457100 |
| Hilbert + L3 ON12 | 0.0032 | 23.8682 | 0.7421 | 0.2601 | 58.50404 | 455333 | 455334 |
| Hilbert + L3 ON12 | 0.0064 | 23.7994 | 0.7395 | 0.2628 | 46.89417 | 447723 | 447724 |
| Hilbert + L3 ON12 | 0.0128 | 23.7032 | 0.7363 | 0.2670 | 38.26852 | 447725 | 447726 |
| Hilbert + L3 ON12 | 0.0256 | 23.5050 | 0.7295 | 0.2764 | 30.28974 | 445516 | 445517 |
| Hilbert + L3 ON12 | 0.0512 | 23.0360 | 0.7140 | 0.2977 | 22.78907 | 455928 | 455929 |
| Hilbert + OFF1 | 0.0032 | 23.8639 | 0.7424 | 0.2608 | 64.42357 | 455339 | 455340 |
| Hilbert + OFF1 | 0.0064 | 23.7795 | 0.7393 | 0.2642 | 55.03013 | 455235 | 455236 |
| Hilbert + OFF1 | 0.0128 | 23.6710 | 0.7354 | 0.2688 | 44.63527 | 455237 | 455238 |
| Hilbert + OFF1 | 0.0256 | 23.4012 | 0.7260 | 0.2812 | 35.45799 | 455239 | 455240 |
| Hilbert + OFF1 | 0.0512 | 22.8594 | 0.7091 | 0.3034 | 25.55945 | 455934 | 455935 |
| Hilbert + OFF4 | 0.0064 | 23.7720 | 0.7389 | 0.2640 | 54.97323 | 455241 | 455242 |
| Hilbert + OFF4 | 0.0128 | 23.6312 | 0.7338 | 0.2694 | 44.76900 | 455243 | 455244 |
| Hilbert + OFF4 | 0.0256 | 23.4001 | 0.7261 | 0.2814 | 35.49336 | 455245 | 455246 |
| Hilbert + L3 OFF12 | 0.0032 | 23.8513 | 0.7421 | 0.2611 | 64.54988 | 456897 | 456898 |
| Hilbert + L3 OFF12 | 0.0064 | 23.8130 | 0.7405 | 0.2629 | 55.10968 | 451815 (recovery) | 451816 |
| Hilbert + L3 OFF12 | 0.0128 | 23.6831 | 0.7358 | 0.2686 | 44.80749 | 451817 (recovery) | 451818 |
| Hilbert + L3 OFF12 | 0.0256 | 23.4072 | 0.7264 | 0.2793 | 35.78297 | 451819 (recovery) | 451820 |
| Hilbert + L3 OFF12 | 0.0512 | 22.8816 | 0.7098 | 0.3021 | 25.42800 | 456949 | 456950 |
| Hilbert + Q2 ON8 | 0.0064 | 23.8168 | 0.7405 | 0.2626 | 48.26934 | 447738 | 447739 |
| Hilbert + Q2 ON8 | 0.0128 | 23.7032 | 0.7363 | 0.2674 | 38.81943 | 447740 | 447741 |
| Hilbert + Q2 ON8 | 0.0256 | 23.5015 | 0.7295 | 0.2765 | 30.61704 | 447742 | 447743 |
| Morton + D4 ON16 | 0.0032 | 23.8564 | 0.7418 | 0.2606 | 57.31014 | 455331 | 455332 |
| Morton + D4 ON16 | 0.0064 | 23.7791 | 0.7390 | 0.2638 | 47.44991 | 447727 | 447728 |
| Morton + D4 ON16 | 0.0128 | 23.7023 | 0.7362 | 0.2677 | 37.93216 | 447729 | 447730 |
| Morton + D4 ON16 | 0.0256 | 23.5089 | 0.7299 | 0.2753 | 29.98379 | 445518 | 445519 |
| Morton + D4 ON16 | 0.0512 | 23.0479 | 0.7142 | 0.2952 | 22.39766 | 455343 | 455344 |
| Morton + L2 ON8 | 0.0064 | 23.7620 | 0.7386 | 0.2650 | 50.46029 | 449559 | 449560 |
| Morton + L2 ON8 | 0.0128 | 23.6313 | 0.7339 | 0.2703 | 40.83512 | 449561 | 449562 |
| Morton + L2 ON8 | 0.0256 | 23.4172 | 0.7265 | 0.2793 | 32.67135 | 442209 | 442210_0 |
| Morton + L3 ON12 | 0.0032 | 23.8505 | 0.7418 | 0.2614 | 59.05459 | 455335 | 455336 |
| Morton + L3 ON12 | 0.0064 | 23.7910 | 0.7390 | 0.2634 | 47.52542 | 447719 | 447720 |
| Morton + L3 ON12 | 0.0128 | 23.6971 | 0.7361 | 0.2678 | 39.00322 | 447721 | 447722 |
| Morton + L3 ON12 | 0.0256 | 23.4892 | 0.7294 | 0.2765 | 30.80726 | 441552 | 441553_1 |
| Morton + L3 ON12 | 0.0512 | 23.0172 | 0.7136 | 0.2989 | 23.29139 | 455930 | 455931 |
| Morton + legacy4 ON16 | 0.0256 | 23.4358 | 0.7271 | 0.2782 | 30.76891 | 443968 | 443969_2 |
| Morton + Q2 ON8 | 0.0064 | 23.7865 | 0.7391 | 0.2634 | 48.34018 | 448644 | 448645 |
| Morton + Q2 ON8 | 0.0128 | 23.6325 | 0.7335 | 0.2691 | 39.00955 | 448646 | 448647 |
| Morton + Q2 ON8 | 0.0256 | 23.4844 | 0.7292 | 0.2771 | 31.15064 | 448648 | 448649 |
| NN-score + L3 ON12 | 0.0064 | 23.8274 | 0.7405 | 0.2607 | 45.01227 | 450937 | 450938 |
| NN-score + L3 ON12 | 0.0128 | 23.7431 | 0.7375 | 0.2642 | 36.95083 | 450939 | 450940 |
| NN-score + L3 ON12 | 0.0256 | 23.5447 | 0.7304 | 0.2737 | 29.30698 | 450934 (resume) | 450935 |

### L3 네 곡선의 공통 품질 구간 비교

공통 PSNR23.0360–23.7288dB에서 각5점의 PCHIP log-rate를 적분했고 외삽하지 않았다.

| 후보 / 기준 | BD-rate |
|---|---:|
| None ON / None OFF |−1.2648%|
| Hilbert ON / Hilbert OFF |−20.4325%|
| Hilbert OFF / None OFF |−14.7991%|
| Hilbert ON / None ON |−31.3393%|
| Hilbert ON / None OFF |−32.2077%|

이 값은 반올림된 집계 지표와 주 seed 한 개의 RD 보간에 근거한다. Scene별 paired CI나 training-seed 유의성은 확인하지 않았다. Hilbert+L3 .0256에는 GPU4 학습 예외도 있어 완전히 맞춘 학습 조건의 순수 인과 효과로 표현하지 않는다. [계산 근거](analysis/nfcgs_logs_2026-10-20_2159/REPORT.md).

<a id="special"></a>

## 6. λ0·두 seed D4 ON/OFF·구조 pilot

### 6.1 λ=0

| 설정 | Train / Eval | PSNR | SSIM | LPIPS | KiB/scene |
|---|---|---:|---:|---:|---:|
| H+D4 λ0 |455327 /455328|24.0734|0.7493|0.2512|831.64219|

실제 양자화·entropy coding이 있는 모델이다. 비압축 또는 native 화질 상한으로 사용하지 않는다. H+D4 .0032 대비 PSNR+0.1769dB, bytes 약14.69배이며 용량 증가의 큰 부분은 MSH y다.

### 6.2 H+D4 ON16/OFF16 두 seed의 완료 비교

모두 λ=.0128, 전체 test6991장이다. OFF16은 공간 예측을 끄되 LR/MSH와 stage/group stream 구성을 유지한다.

| Seed(model/data) | Context | Eval | PSNR | SSIM | LPIPS | KiB/scene |
|---|---|---|---:|---:|---:|---:|
|111123/403|ON16|447899|23.7021|0.7363|0.2673|37.178812|
|111123/403|OFF16|457100|23.6338|0.7338|0.2695|44.619252|
|111124/404|ON16|457787|23.7036|0.7362|0.2669|37.075100|
|111124/404|OFF16|457789|23.6310|0.7336|0.2692|44.768936|

| Seed | ON의 전송량 감소율 | ON−OFF PSNR |
|---|---:|---:|
|기본|16.6754%|+0.0683dB|
|추가|17.1857%|+0.0726dB|

두 seed 모두 같은λ에서 ON이 더 작은 actual bytes와 더 높은 PSNR을 보였다. 같은 warm-start 부모를 공유하는 반복이며 동일 품질 BD-rate 또는 독립 fresh-init의 seed 유의성 검정은 아니다. [완료 근거](analysis/nfcgs_d4_context_2026-10-21/REPORT.md).

### 6.3 기존50k 구조 pilot 네 arm

Hilbert·λ=.0128·model111123/data403, 공개 backbone/QR 경계에서 공통 codec 초기화 후50k를 학습했다. 같은 train-derived holdout128을 학습에서 제외했으며 일반 test를 읽는 validation loader는 학습 중 끈다. Warm-start 주 RD의 전체 test 결과와 분리한다.

| Arm | 활성 경로 | Train / Eval | PSNR | SSIM | LPIPS | KiB/scene |
|---|---|---|---:|---:|---:|---:|
| score56_only | rank56 Score만 | 457741 / 457742 | 21.7889 | 0.6653 | 0.3295 | 65.53055 |
| msh_only | F736 직접 MSH | 457743 / 457744 | 20.6696 | 0.6123 | 0.3700 | 14.48434 |
| two_path56 | rank56+MSH residual | 457745 / 457746 | 23.0067 | 0.7157 | 0.2796 | 56.73486 |
| score736_only | 736→736 Score만 | 457828 / 457829 | 21.7001 | 0.6603 | 0.3315 | 67.81793 |

Score736-only는 Score56-only보다 PSNR−0.0888dB·SSIM−0.0050·LPIPS+0.0020이고 bytes도3.4906% 많아 이번 한 점에서는 지배된다. Two-path56은 Score736-only보다 PSNR+1.3066dB이고 bytes16.3424% 적다. MSH-only는14.48KiB의 저율 점이므로 같은λ의 품질 차이만으로 제외하지 않는다.

Score736은 channel group184×4와 큰 predictor를 사용하므로 rank만의 순수 대조군은 아니다. Backbone/Gaussian decoder는 고정하고 codec과 geometry projection을 학습했다. 네 구조의 전체 test와 추가λ는 이 정리의 완료 결과에 없다. [구조 비교 근거](analysis/nfcgs_logs_2026-10-20_2159/REPORT.md).

<a id="diagnostics"></a>

## 7. 비압축·경로·정렬·독립 수신 진단

| 실험 | Job | 대상/규모 | 확인 상태와 범위 |
|---|---|---|---|
| L3 정렬별 실제 anchor·잔차 | 454995 | None/Morton/Hilbert/NN-score,4모델×256장 | ANALYSIS_COMPLETE; 현재 후속진단의 원본 C12 cache |
| Hilbert schedule anchor·잔차 | 455166 | L3/Q2/D4,3모델×같은256장 | ANALYSIS_COMPLETE; 모델별 basis/projection이 달라 frozen-order 개입과 구분 |
| T04 native B0 및 B1 bridge | 458024 | native geometry512 vs current geometry224,256장 | 완료; 같은 encoder output·정렬·카메라 |
| T05 경로·양자화 우회 | 458004 | 현행 H+D4 .0128,7경로×256장 | 완료; actual만 packet rate 있음 |
| T15 현행 독립 수신 | 458784 | T05 packet256개; 새 codec+Gaussian decoder 프로세스 | 완료; stdout PSNR delta256장 모두0, 검사 허용치 통과 |
| T06 frozen order 개입 | 458818 | 같은 F/P/basis/Score/XYZ,13순서×256장 | 완료; Hilbert packet은 T05와 SHA 일치; 재학습/PSNR 측정 아님 |
| 과거 Full nonlinear Split 독립 수신 | 433727 /433728 | 기존 mean-offset 설정,전체6991장 encode→decode/render | 두 프로세스 완료; PSNR24.3238/SSIM.7611/LPIPS.2371; 현재 T15와 다른 모델 |

현재256장 공통 fingerprint는 `069b40a74daef588e7ac01be40231dce159bb304b588e7669c6f4801e4241f29`다. 주 checkpoint는 `outputs/nfcgs_order_context/hilbert/dyadic4_stages4_k5_ste/lambda0p0128/job_447746/checkpoints/nfcgs_main/version_0/step000050000.ckpt`다.

### 7.1 B0와 T05의 모든 경로

B0는 공개 native geometry512 decoder, B1은 현재 학습된 projection 이후 F736을 직접 넣는 우회다. B1은 현행 checkpoint의 화질 상한이 아니다.

| 경로 | PSNR | SSIM | LPIPS | KiB/scene |
|---|---|---|---|---|
| B0 native | 24.90270 | 0.77651 | 0.24100 | — |
| B1 직접 F | 16.76771 | 0.56521 | 0.48654 | — |
| 연속 LR만 | 19.80692 | 0.60636 | 0.41663 | — |
| 실제 복호 Score의 LR만 | 18.56430 | 0.51692 | 0.51549 | — |
| Score 양자화 우회 | 23.26808 | 0.73005 | 0.27009 | — |
| MSH y/z 양자화 우회 | 23.92326 | 0.74670 | 0.25720 | — |
| 양쪽 양자화 우회 | 23.31259 | 0.73187 | 0.26772 | — |
| 실제 packet | 23.87649 | 0.74468 | 0.25992 | 37.2592 |

Actual−B0는 이 matched256에서−1.02621dB다. MSH 양자화 우회 개선은+0.04677dB로 작지만 Hcont에 분석/합성 손실은 남는다. Score 우회는−0.60840dB로 내려가며, 이는 fixed checkpoint의 residual 분포·공동 적응을 포함한다. 양자화의 독립 손실이나 각 경로의 가산 기여로 해석하지 않는다. 재학습한 pilot 순위를 대신하지도 않는다.

Native slot-order 검사에서 Gaussian field 차이는0이고 평균 PSNR 차이는 약−2.61e−7dB지만 최대 pixel 차이는0.0124051이었다. 동일 Gaussian 반복 렌더에서는 B0/B1/actual의 pixel 차이가0이었다. 순열 불변 Gaussian 값과 GPU pixel bit-exact를 구분한다.

T15 수신기는 scene encoder·입력 영상·원래 순열 없이 packet과 공유 checkpoint로 Gaussian을 만든다. Target 카메라·영상 크기는 렌더, 실제 target RGB는 metric에만 쓴다. T05 원본 pixel/feature와의 bit-exact 증명은 원본 tensor가 저장되지 않아 주장하지 않는다.

### 7.2 T06 전체 aggregate 결과

| 순서 | 전체 KiB | Raw Score KiB | MSH y KiB | MSH z KiB | Header B | Anchor XYZ 거리 | Score 잔차 정규화 MAE |
|---|---|---|---|---|---|---|---|
| slot | 56.67001 | 41.06366 | 15.19284 | 0.13617 | 284.0 | 2.3873300 | 1.6707784 |
| morton | 37.91431 | 23.28050 | 14.22141 | 0.13506 | 284.0 | 0.2421056 | 1.5297684 |
| hilbert | 37.25922 | 22.66045 | 14.18643 | 0.13499 | 284.0 | 0.2092603 | 1.5245647 |
| random10_mean | 56.72330 | 41.11120 | 15.19830 | 0.13645 | 284.0 | 2.3901985 | 1.6718214 |

| 기준−Hilbert | 평균 B/scene | 95% paired scene bootstrap CI |
|---|---|---|
| slot | +19876.656 | [+19598.986, +20176.601] |
| morton | +670.812 | [+648.911, +696.355] |
| random10_mean | +19931.219 | [+19648.911, +20233.803] |

Slot→Hilbert는 전체−34.25%, raw Score−44.82%, anchor 거리−91.23%, 예측 잔차 MAE−8.75%다. 관찰된 전체 byte 차이의 약94.8%는 Score stream에서 나온다. MSH도 변하므로 이를 독립 인과 기여율로 읽지 않는다. Morton→Hilbert의 추가 전체 절감은 약1.73%다.

Random10은 scene 안에서 먼저 평균하고1000회 scene bootstrap을 사용했다. Training seed CI가 아니다. Anchor 거리는 유효 anchor pair의 가중 평균이고 MAE는 첫 단계까지 포함한 전체 symbol 가중 평균이라 분모가 다르다. 같은 Hilbert 학습 weight에 대한 분포 변화 진단이며 feature distortion도 달라질 수 있어 **동일 화질 RD 절감률이 아니다**.

주요 서버 출력: `outputs/nfcgs_fixed_order/job_458818/REPORT.md`, `outputs/nfcgs_independent_receiver/job_458784/receiver/REPORT.md`, `outputs/nfcgs_path_diagnostics/job_458004/REPORT.md`, `outputs/nfcgs_b0_matched/job_458024/REPORT.md`.
<a id="complexity"></a>

## 8. 복잡도·시간·메모리 측정

### 8.1 현행 정렬/context 37 checkpoint: job455251 완료

같은 RTX A5000(ariel-v10), BF16, threads8, 같은 C12 cache에서 warmup4+측정32장, codec 구간3회 반복이다.32장 profile bytes를 전체6991장 RD bytes와 섞지 않는다. Encode는 token-position 준비를 포함한 `compress_scene_tokens`, decode는 `feature_codec.decompress`이며 image encoder·Gaussian decoder·renderer는 제외한다.

| 설정 | Encode ms | Decode ms | 정렬 ms | Encode 총 peak MiB | Encode 추가 peak MiB | Decode 추가 peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| Hilbert legacy3 | 150.583 | 84.279 | 6.054 | 511.957 | 83.220 | 48.503 |
| Hilbert quarter2 | 155.149 | 87.423 | 6.242 | 511.901 | 83.220 | 48.503 |
| Hilbert dyadic4 | 163.698 | 93.299 | 6.361 | 512.078 | 83.220 | 48.503 |
| NN-score legacy3 | 214.619 | 84.715 | 67.662 | 515.012 | 86.303 | 48.503 |
| Hilbert context OFF | 156.028 | 86.213 | 6.219 | 511.962 | 83.220 | 48.503 |
| None legacy3 | 150.710 | 87.644 | 0.109 | 511.853 | 83.188 | 48.503 |

이 표의 Hilbert Context OFF는 **L3 OFF12**다. D4 OFF16이 아니다. D4는 L3 대비 encode+8.71%, decode+10.70%였다. NN-score는 L3/Hilbert 대비 encoder가 약42.52% 느렸다. MAC만으로 이 시간 순위를 예측할 수 없다.

총 peak는 전체 모델이 올라간 harness의 allocated VRAM이며 codec-only 독립 수신기의 최소 메모리가 아니다. 추가 peak는 호출 직전 allocated를 뺀 증가량이다. CUDA context/driver·NVML·CPU rANS/native 메모리와 다르다.

37개 point의 실제 profile 목록:

| Point | Encode ms | Decode ms | Encode p90 ms | Encode 총 peak | Encode 추가 peak | Decode 추가 peak | Profile bytes KiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| hilbert_dyadic4_0p0064 | 162.312 | 92.101 | 172.223 | 510.404 | 82.296 | 49.485 | 46.967 |
| hilbert_dyadic4_0p0128 | 163.698 | 93.299 | 174.459 | 512.078 | 83.220 | 48.503 | 37.388 |
| hilbert_dyadic4_0p0256 | 160.102 | 90.374 | 170.450 | 512.056 | 83.220 | 48.503 | 29.560 |
| hilbert_legacy3_0p0064 | 151.097 | 84.864 | 153.915 | 510.006 | 81.237 | 48.985 | 47.120 |
| hilbert_legacy3_0p0128 | 150.583 | 84.279 | 153.327 | 511.957 | 83.220 | 48.503 | 38.469 |
| hilbert_legacy3_0p0256 | 150.578 | 83.755 | 153.045 | 511.941 | 83.220 | 48.503 | 30.457 |
| hilbert_quarter2_0p0064 | 155.415 | 88.045 | 158.663 | 511.908 | 83.220 | 48.503 | 48.504 |
| hilbert_quarter2_0p0128 | 155.149 | 87.423 | 156.789 | 511.901 | 83.220 | 48.503 | 39.048 |
| hilbert_quarter2_0p0256 | 155.364 | 87.018 | 157.439 | 511.890 | 83.220 | 48.503 | 30.779 |
| morton_dyadic4_0p0064 | 158.214 | 92.313 | 161.024 | 512.091 | 83.220 | 48.503 | 47.708 |
| morton_dyadic4_0p0128 | 157.982 | 91.910 | 161.107 | 512.068 | 83.220 | 48.503 | 38.154 |
| morton_dyadic4_0p0256 | 156.310 | 91.909 | 158.881 | 512.054 | 83.220 | 48.503 | 30.145 |
| morton_legacy2_0p0064 | 145.638 | 83.779 | 147.604 | 511.881 | 83.220 | 48.503 | 50.748 |
| morton_legacy2_0p0128 | 143.712 | 82.493 | 146.247 | 511.866 | 83.220 | 48.503 | 41.078 |
| morton_legacy2_0p0256 | 147.106 | 82.620 | 145.914 | 511.866 | 83.220 | 48.503 | 32.904 |
| morton_legacy3_0p0064 | 150.281 | 86.692 | 152.245 | 511.975 | 83.220 | 48.503 | 47.811 |
| morton_legacy3_0p0128 | 148.875 | 85.480 | 150.972 | 511.954 | 83.220 | 48.503 | 39.244 |
| morton_legacy3_0p0256 | 147.643 | 84.682 | 149.610 | 511.945 | 83.220 | 48.503 | 31.008 |
| morton_quarter2_0p0064 | 152.587 | 88.560 | 154.939 | 511.910 | 83.220 | 48.503 | 48.620 |
| morton_quarter2_0p0128 | 152.079 | 88.118 | 154.400 | 511.894 | 83.220 | 48.503 | 39.215 |
| morton_quarter2_0p0256 | 150.939 | 87.556 | 152.699 | 511.889 | 83.220 | 48.503 | 31.346 |
| nn_score_0p0256 | 211.560 | 83.145 | 216.035 | 515.001 | 86.303 | 48.503 | 29.515 |
| nn_score_0p0064 | 250.549 | 85.532 | 255.922 | 515.023 | 86.303 | 48.503 | 45.288 |
| nn_score_0p0128 | 214.619 | 84.715 | 219.854 | 515.012 | 86.303 | 48.503 | 37.186 |
| context_only_0p0064 | 152.336 | 90.339 | 157.556 | 511.870 | 83.188 | 48.503 | 56.203 |
| context_only_0p0128 | 150.710 | 87.644 | 153.310 | 511.853 | 83.188 | 48.503 | 46.919 |
| context_only_0p0256 | 149.774 | 87.035 | 151.611 | 511.835 | 83.188 | 48.503 | 38.640 |
| hilbert_only_0p0064 | 156.037 | 86.961 | 158.219 | 511.974 | 83.220 | 48.503 | 55.573 |
| hilbert_only_0p0128 | 156.028 | 86.213 | 158.176 | 511.962 | 83.220 | 48.503 | 45.194 |
| hilbert_only_0p0256 | 154.282 | 85.772 | 156.439 | 511.959 | 83.220 | 48.503 | 36.108 |
| context_quarter2_0p0064 | 154.604 | 90.900 | 156.330 | 511.801 | 83.188 | 48.503 | 55.961 |
| context_quarter2_0p0128 | 153.689 | 89.892 | 155.983 | 511.790 | 83.188 | 48.503 | 47.191 |
| context_quarter2_0p0256 | 152.328 | 89.222 | 154.676 | 511.778 | 83.188 | 48.503 | 38.774 |
| context_dyadic4_0p0064 | 162.469 | 96.380 | 165.164 | 511.992 | 83.188 | 48.503 | 56.133 |
| context_dyadic4_0p0128 | 159.527 | 94.187 | 162.656 | 511.972 | 83.188 | 48.503 | 47.168 |
| context_dyadic4_0p0256 | 158.653 | 93.400 | 161.217 | 511.949 | 83.188 | 48.503 | 38.686 |
| morton_legacy4_0p0256 | 154.590 | 88.510 | 157.691 | 512.019 | 83.220 | 48.503 | 30.932 |

설정별 parameter 수와 codec core 연산량도 함께 보존한다. 저장 parameter에는 비활성 모듈이 포함될 수 있으며, codec core GMAC는 CPU 정렬·entropy coder·renderer의 실행 시간을 나타내지 않는다.

| 설정 | 활성 Params M | 저장 Params M | Codec core encode GMAC | Decode GMAC |
|---|---:|---:|---:|---:|
| Morton legacy2 | 4.585288 | 4.696560 | 6.967001 | 3.393454 |
| Morton legacy3 | 4.592920 | 4.704192 | 6.984040 | 3.410493 |
| Hilbert legacy3 | 4.592920 | 4.704192 | 6.984040 | 3.410493 |
| Morton quarter2 | 4.586400 | 4.701888 | 6.955074 | 3.381527 |
| Hilbert quarter2 | 4.586400 | 4.701888 | 6.955074 | 3.381527 |
| Morton dyadic4 | 4.593232 | 4.717152 | 6.955926 | 3.382378 |
| Hilbert dyadic4 | 4.593232 | 4.717152 | 6.955926 | 3.382378 |
| NN-score legacy3 | 4.592920 | 4.704192 | 7.152861 | 3.410493 |
| None legacy3 | 4.592920 | 4.704192 | 6.984040 | 3.410493 |
| None quarter2 | 4.586400 | 4.701888 | 6.955074 | 3.381527 |
| None dyadic4 | 4.593232 | 4.717152 | 6.955926 | 3.382378 |
| Hilbert context OFF | 4.584488 | 4.704192 | 6.949962 | 3.376415 |
| Morton legacy4 | 4.600552 | 4.711824 | 7.001080 | 3.427533 |

### 8.2 과거 overhead·probe

- Overhead job423729: **18 profile 완료**. 기존 context/predictor/probability 계열, warmup4·측정32장. `OVERHEAD_AGGREGATE_COMPLETE profiles=18`과 `OVERHEAD_ALL_COMPLETE`를 확인했다.
- Fixed-symbol entropy probe array423055_0/1: λ=.0064/.0256, TRAIN128에서 table fit→TEST128 평가. 원본 symbol/feature 복원을 고정하고 context histogram이 전체 payload 약0.736%/0.759% 절감했다. 이는 새 full codec 학습이나 전체 test/render 결과가 아니다.
- Complexity job454488 및 broad mechanism job454797은 현재 stdout가 비어 있어 성공으로 집계하지 않는다. 실제 완료 근거는455251,454995,455166이다.

새 OFF1/OFF4/Both-OFF·OFF16·구조 모델의 빠진 profile 보완은 아직 별도 완료 결과가 없다. 새 할당에서는 H+L3/H+D4/NN-score .0128 reference를 함께 재측정하고 새 output으로 기록해야 한다.
<a id="legacy"></a>

## 9. 9월 이전/초기 실험 전체 기록

### 9.1 초기 rank·projection·residual·정렬 탐색

9월2일 보존 보고서 기준이다. 현재 H+D4 checkpoint의 B0/T05 결과와 다른 모델·초기화·시점이다. 당시의 `upper`는 역사적 vanilla 참고명이며 수학적 화질 상한이 아니다.

| 실험 | 수행 내용 | 당시 확인 상태 |
|---|---|---|
| Experiment17 | rank40/56/72/80에서 projection·quantization·full residual의 누적 우회 | 전체 test 결과 보존; 같은 full-trained checkpoint에서 경로를 끈 진단 |
| Experiment16 | 6,991장 covariance spectrum·채널/토큰 상관 | job409784 완료; covariance용64 tokens, lag통계32 channels |
| Experiment14 | Linear-no-residual / Nonlinear32-no-residual / Linear-residual의 새50k | 세 모델 완료; 기존 PCA artifact와 다른 초기화, 최초409785 실패 후 수정 |
| Experiment15 | no-residual 모델에서 none/Morton/random inference 순서 | eval array410949 완료 |
| Experiment15b | frozen Morton-trained residual 모델에서 순서 변경 | eval array411709 완료 |
| 초기 Full RD | λ=.0004/.0016/.0064/.0256와 codec-free | eval array411698 완료; 초기30장 sweep과 구분 |

Experiment17 전체 수치:

| Condition | PSNR | SSIM | LPIPS | Bytes/scene | bits/Gaussian | BPGA |
|---|---:|---:|---:|---:|---:|---:|
| upper | 24.7004 | 0.7682 | 0.2480 | -- | -- | -- |
| concat_only | 24.6376 | 0.7664 | 0.2494 | -- | -- | -- |
| rank40_projection | 18.3502 | 0.6700 | 0.3621 | -- | -- | -- |
| rank40_projection_quantized | 17.9304 | 0.6428 | 0.4119 | 43,797 | 10.6927 | 0.181232 |
| rank40_full | 23.3062 | 0.7272 | 0.2889 | 54,735 | 13.3631 | 0.226493 |
| rank56_projection | 20.5585 | 0.7015 | 0.3264 | -- | -- | -- |
| rank56_projection_quantized | 20.2568 | 0.6813 | 0.3611 | 47,894 | 11.6929 | 0.198184 |
| rank56_full | **23.3700** | 0.7290 | 0.2870 | 58,444 | 14.2686 | 0.241841 |
| rank72_projection | 19.4533 | 0.6853 | 0.3081 | -- | -- | -- |
| rank72_projection_quantized | 19.2073 | 0.6670 | 0.3424 | 52,069 | 12.7121 | 0.215460 |
| rank72_full | 23.1918 | **0.7307** | 0.2864 | 94,217 | 23.0022 | 0.389867 |
| rank80_projection | 20.1058 | 0.6912 | 0.3021 | -- | -- | -- |
| rank80_projection_quantized | 19.8696 | 0.6756 | 0.3303 | 54,283 | 13.2527 | 0.224622 |
| rank80_full | 23.0444 | 0.7305 | **0.2836** | 97,568 | 23.8202 | 0.403733 |

위 projection 경로들은 독립적으로 재학습한 모델이 아니라 같은 full-trained checkpoint의 누적 개입이다. 연속 경로에는 packet rate가 없다. 해당 손실을 현행 모델의 독립 양자화/경로 기여로 재사용하지 않는다.

Experiment16의 covariance energy:

| Stream | Mean abs. off-diagonal correlation | Rank 40 | Rank 56 | Rank 72 | Rank 80 |
|---|---:|---:|---:|---:|---:|
| appearance | 0.1241 | 88.80% | 91.00% | 92.49% | 93.08% |
| raw geometry | 0.0905 | 98.63% | 98.90% | 99.09% | 99.16% |
| observable geometry | 0.4233 | 99.45% | 99.62% | 99.72% | 99.76% |
| observable concat | 0.1291 | 97.31% | 97.93% | 98.29% | 98.42% |

Joint concat rank56의97.93% energy가 모든 지각적 appearance 정보를 보존한다는 뜻은 아니다. Appearance rank56은91.00%, observable geometry는99.62%여서 covariance scale의 영향도 고려해야 한다.

Experiment14의 새50k 구조 결과:

| Condition | PSNR | SSIM | LPIPS | Bytes/scene | bits/Gaussian | BPGA |
|---|---:|---:|---:|---:|---:|---:|
| linear_no_residual | 20.5875 | 0.6347 | 0.3990 | 55,625 | 13.5804 | 0.230177 |
| nonlinear32_no_residual | **21.4757** | **0.6649** | **0.3387** | 51,391 | 12.5467 | 0.212655 |
| linear_residual | 20.1332 | 0.5971 | 0.4011 | **9,581** | **2.3391** | **0.039645** |

당시 구조 실험은 matched PCA 초기화 대조군이 아니므로 현재 fresh pilot과도 직접 합치지 않는다.

Experiment15 /15b의 inference 정렬 결과:

| Ordering | PSNR | SSIM | LPIPS | Bytes/scene | bits/Gaussian | BPGA |
|---|---:|---:|---:|---:|---:|---:|
| none | 20.5875 | 0.6347 | 0.3990 | 55,625.397797 | 13.580419 | 0.230177 |
| Morton | 20.5875 | 0.6347 | 0.3990 | 55,625.396653 | 13.580419 | 0.230177 |
| deterministic random | 20.5875 | 0.6347 | 0.3990 | 55,625.397225 | 13.580419 | 0.230177 |

| Inference ordering | PSNR | SSIM | LPIPS | Bytes/scene | bits/Gaussian | BPGA |
|---|---:|---:|---:|---:|---:|---:|
| none | 22.8059 | 0.7232 | 0.3068 | 73,978.984981 | 18.061276 | 0.306123 |
| **Morton** | **23.3700** | **0.7290** | **0.2870** | **58,444.196824** | **14.268603** | **0.241841** |
| deterministic random | 22.7935 | 0.7230 | 0.3071 | 74,491.973108 | 18.186517 | 0.308246 |

초기 protocol-matched Full RD:

| Condition | Bytes/scene | bits/Gaussian | BPGA | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| codec-free GlobalSplat | -- | -- | -- | 24.7004 | 0.7682 | 0.2480 |
| lambda 0.0004 | 385,759 | 94.1795 | 1.596262 | **24.9867** | **0.7832** | **0.2180** |
| lambda 0.0016 | 354,285 | 86.4953 | 1.466022 | 24.8686 | 0.7792 | 0.2223 |
| lambda 0.0064 | 128,360 | 31.3378 | 0.531149 | 24.5563 | 0.7682 | 0.2353 |
| lambda 0.0256 | 58,444 | 14.2686 | 0.241841 | 23.3700 | 0.7290 | 0.2870 |

### 9.2 초기 Linear 사용자 표12개

초기50k의 사용자 제공 결과다. 원표의 크기 단위는 `kB`로 보존한다. 원본 bytes가 없어 KiB로 변환하지 않았고 job ID도 확정하지 않는다. 역사적 codec-OFF는24.7004/.7682/.2480이고 scene packet 크기는 없다.

| Rank | λ | Residual | Morton | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 크기(원표 kB 표기) ↓ |
|---:|---:|---|---|---:|---:|---:|---:|
| 56 | .0064 | ON | ON | 24.5021 | 0.7663 | 0.2342 | 102.68 |
| 56 | .0064 | OFF | ON | 24.2426 | 0.7582 | 0.2460 | 111.30 |
| 56 | .0064 | ON | OFF | 24.4920 | 0.7662 | 0.2356 | 105.45 |
| 56 | .0256 | ON | ON | 23.6659 | 0.7368 | 0.2701 | 48.53 |
| 56 | .0256 | OFF | ON | 23.4941 | 0.7311 | 0.2750 | 58.58 |
| 56 | .0256 | ON | OFF | 23.7240 | 0.7393 | 0.2668 | 53.08 |
| 80 | .0064 | ON | ON | 24.4776 | 0.7653 | 0.2371 | 103.84 |
| 80 | .0064 | OFF | ON | 24.2327 | 0.7577 | 0.2457 | 115.63 |
| 80 | .0064 | ON | OFF | 24.4845 | 0.7661 | 0.2370 | 110.21 |
| 80 | .0256 | ON | ON | 23.5391 | 0.7346 | 0.2764 | 48.12 |
| 80 | .0256 | OFF | ON | 23.4823 | 0.7313 | 0.2746 | 58.53 |
| 80 | .0256 | ON | OFF | 23.6440 | 0.7386 | 0.2690 | 54.54 |

### 9.3 보존 JSON의 완료 평가48개

9월15일에 원본 완료 eval 로그에서 재추출한 기록을 사용한다. **현재 다운로드 폴더에는 이48개 원본 stdout가 없으므로 이번에 다시 raw log를 검산한 결과는 아니다.** 정확한 bytes·checkpoint·당시 training lineage는 보존 JSON에 있다.

이전50k+추가50k, 추가25k/10k는 서로 다른 stage다. Stage step과 누적 lineage를 함께 표시한다. Weights-only continuation은 optimizer-state resume와 다르며, probability-only stage는 복원 관련 parameter를 고정했다.

| 과거 묶음 | 완료 평가 수 |
|---|---|
| Linear factorized +50k | 4 |
| Nonlinear32 초기50k | 8 |
| Nonlinear factorized +50k | 4 |
| Nonlinear Spatial/Full +50k | 4 |
| Linear mean/channel/spatial +50k | 8 |
| 복원 고정 probability-only +10k | 8 |
| P0/P1/P2 +25k | 9 |
| Full low-LR +10k | 3 |

| 실험/arm | Rank | λ | Stage/누적 steps | PSNR | SSIM | LPIPS | KiB/scene | Eval |
|---|---|---|---|---|---|---|---|---|
| Full low-LR +10k / ARM=p2_residual7 | 56 | 0.0064 | 10,000/110,000 | 24.3256 | 0.7611 | 0.2370 | 82.43005 | 422944_1 |
| Full low-LR +10k / ARM=p0_linear | 56 | 0.0256 | 10,000/110,000 | 23.5701 | 0.7342 | 0.2684 | 37.93242 | 422944_2 |
| Full low-LR +10k / ARM=p2_residual7 | 56 | 0.0256 | 10,000/110,000 | 23.5716 | 0.7343 | 0.2684 | 37.92130 | 422944_3 |
| Linear factorized +50k / task0 | 56 | 0.0064 | 50,000/100,000 | 24.3024 | 0.7602 | 0.2374 | 90.80475 | 420718_0 |
| Linear factorized +50k / task2 | 80 | 0.0064 | 50,000/100,000 | 24.2805 | 0.7597 | 0.2399 | 90.94448 | 420718_2 |
| Linear factorized +50k / task1 | 56 | 0.0256 | 50,000/100,000 | 23.4875 | 0.7307 | 0.2729 | 42.25802 | 420718_1 |
| Linear factorized +50k / task3 | 80 | 0.0256 | 50,000/100,000 | 23.4740 | 0.7310 | 0.2758 | 42.64204 | 420718_3 |
| Linear mean/channel/spatial +50k / PRIOR=m1c0s0 | 56 | 0.0064 | 50,000/100,000 | 24.3179 | 0.7610 | 0.2371 | 89.82306 | 419616_0 |
| Linear mean/channel/spatial +50k / PRIOR=m1c1s0 | 56 | 0.0064 | 50,000/100,000 | 24.3151 | 0.7608 | 0.2371 | 88.24939 | 419616_2 |
| Linear mean/channel/spatial +50k / PRIOR=m1c0s1 | 56 | 0.0064 | 50,000/100,000 | 24.3222 | 0.7611 | 0.2369 | 86.07254 | 419616_4 |
| Linear mean/channel/spatial +50k / PRIOR=m1c1s1 | 56 | 0.0064 | 50,000/100,000 | 24.3232 | 0.7611 | 0.2368 | 84.90535 | 419616_6 |
| Linear mean/channel/spatial +50k / PRIOR=m1c0s0 | 56 | 0.0256 | 50,000/100,000 | 23.4950 | 0.7309 | 0.2723 | 41.60841 | 419616_1 |
| Linear mean/channel/spatial +50k / PRIOR=m1c1s0 | 56 | 0.0256 | 50,000/100,000 | 23.4943 | 0.7312 | 0.2727 | 41.49185 | 419616_3 |
| Linear mean/channel/spatial +50k / PRIOR=m1c0s1 | 56 | 0.0256 | 50,000/100,000 | 23.5304 | 0.7319 | 0.2716 | 39.17162 | 419616_5 |
| Linear mean/channel/spatial +50k / PRIOR=m1c1s1 | 56 | 0.0256 | 50,000/100,000 | 23.5352 | 0.7323 | 0.2716 | 39.23263 | 419616_7 |
| Nonlinear factorized +50k / PRIOR=factorized | 56 | 0.0064 | 50,000/100,000 | 24.3022 | 0.7602 | 0.2376 | 87.83998 | 420304_0 |
| Nonlinear factorized +50k / PRIOR=factorized | 80 | 0.0064 | 50,000/100,000 | 24.2920 | 0.7601 | 0.2395 | 88.16874 | 420383_10 |
| Nonlinear factorized +50k / PRIOR=factorized | 56 | 0.0256 | 50,000/100,000 | 23.5210 | 0.7328 | 0.2693 | 41.25607 | 420304_1 |
| Nonlinear factorized +50k / PRIOR=factorized | 80 | 0.0256 | 50,000/100,000 | 23.4264 | 0.7319 | 0.2756 | 42.30569 | 420383_11 |
| Nonlinear Spatial/Full +50k / PRIOR=m1c0s1 | 56 | 0.0064 | 50,000/100,000 | 24.3244 | 0.7612 | 0.2371 | 83.63341 | 420919_6 |
| Nonlinear Spatial/Full +50k / PRIOR=m1c1s1 | 56 | 0.0064 | 50,000/100,000 | 24.3238 | 0.7611 | 0.2371 | 82.49134 | 420919_8 |
| Nonlinear Spatial/Full +50k / PRIOR=m1c0s1 | 56 | 0.0256 | 50,000/100,000 | 23.5867 | 0.7348 | 0.2678 | 38.23606 | 420919_7 |
| Nonlinear Spatial/Full +50k / PRIOR=m1c1s1 | 56 | 0.0256 | 50,000/100,000 | 23.5707 | 0.7342 | 0.2683 | 37.95804 | 420919_9 |
| Nonlinear32 초기50k / RESIDUAL=true | 56 | 0.0064 | 50,000/50,000 | 24.5082 | 0.7665 | 0.2343 | 98.18345 | 419630_8 |
| Nonlinear32 초기50k / RESIDUAL=false | 56 | 0.0064 | 50,000/50,000 | 24.2580 | 0.7587 | 0.2440 | 105.78999 | 419630_9 |
| Nonlinear32 초기50k / RESIDUAL=true | 80 | 0.0064 | 50,000/50,000 | 24.4890 | 0.7659 | 0.2367 | 99.15207 | 419630_12 |
| Nonlinear32 초기50k / RESIDUAL=false | 80 | 0.0064 | 50,000/50,000 | 24.2579 | 0.7584 | 0.2438 | 109.91254 | 419630_13 |
| Nonlinear32 초기50k / RESIDUAL=true | 56 | 0.0256 | 50,000/50,000 | 23.6943 | 0.7381 | 0.2673 | 46.69461 | 419630_10 |
| Nonlinear32 초기50k / RESIDUAL=false | 56 | 0.0256 | 50,000/50,000 | 23.5011 | 0.7303 | 0.2738 | 55.33627 | 419630_11 |
| Nonlinear32 초기50k / RESIDUAL=true | 80 | 0.0256 | 50,000/50,000 | 23.5833 | 0.7378 | 0.2699 | 47.31401 | 419630_14 |
| Nonlinear32 초기50k / RESIDUAL=false | 80 | 0.0256 | 50,000/50,000 | 23.4970 | 0.7301 | 0.2728 | 55.41493 | 420263_15 |
| 복원 고정 probability-only +10k / ARM=e0_shared, ENTROPY=shared | 56 | 0.0064 | 10,000/110,000 | 24.3238 | 0.7611 | 0.2371 | 82.46314 | 422559_0 |
| 복원 고정 probability-only +10k / ARM=e1_split, ENTROPY=split | 56 | 0.0064 | 10,000/110,000 | 24.3238 | 0.7611 | 0.2371 | 81.17353 | 422559_2 |
| 복원 고정 probability-only +10k / ARM=e2_gaussian, ENTROPY=gaussian | 56 | 0.0064 | 10,000/110,000 | 24.3238 | 0.7611 | 0.2371 | 88.16696 | 422559_4 |
| 복원 고정 probability-only +10k / ARM=e3_conditional_scale, ENTROPY=conditional_scale | 56 | 0.0064 | 10,000/110,000 | 24.3238 | 0.7611 | 0.2371 | 81.13547 | 422559_6 |
| 복원 고정 probability-only +10k / ARM=e0_shared, ENTROPY=shared | 56 | 0.0256 | 10,000/110,000 | 23.5707 | 0.7342 | 0.2683 | 37.95195 | 422559_1 |
| 복원 고정 probability-only +10k / ARM=e1_split, ENTROPY=split | 56 | 0.0256 | 10,000/110,000 | 23.5707 | 0.7342 | 0.2683 | 36.92041 | 422559_3 |
| 복원 고정 probability-only +10k / ARM=e2_gaussian, ENTROPY=gaussian | 56 | 0.0256 | 10,000/110,000 | 23.5707 | 0.7342 | 0.2683 | 44.51828 | 422559_5 |
| 복원 고정 probability-only +10k / ARM=e3_conditional_scale, ENTROPY=conditional_scale | 56 | 0.0256 | 10,000/110,000 | 23.5707 | 0.7342 | 0.2683 | 37.02079 | 422559_7 |
| P0/P1/P2 +25k / ARM=p0_linear | 56 | 0.0064 | 25,000/125,000 | 24.1634 | 0.7569 | 0.2400 | 79.16599 | 422182_0 |
| P0/P1/P2 +25k / ARM=p1_residual3 | 56 | 0.0064 | 25,000/125,000 | 24.1858 | 0.7576 | 0.2394 | 79.19481 | 422182_2 |
| P0/P1/P2 +25k / ARM=p2_residual7 | 56 | 0.0064 | 25,000/125,000 | 24.1591 | 0.7570 | 0.2401 | 79.05495 | 422182_4 |
| P0/P1/P2 +25k / ARM=p0_linear | 56 | 0.0256 | 25,000/125,000 | 23.4764 | 0.7324 | 0.2681 | 37.44576 | 422182_1 |
| P0/P1/P2 +25k / ARM=p1_residual3 | 56 | 0.0256 | 25,000/125,000 | 23.4515 | 0.7316 | 0.2676 | 37.46507 | 422182_3 |
| P0/P1/P2 +25k / ARM=p2_residual7 | 56 | 0.0256 | 25,000/125,000 | 23.4566 | 0.7317 | 0.2674 | 37.32333 | 422182_5 |
| P0/P1/P2 +25k / ARM=p0_linear | 56 | 0.0256 | 25,000/125,000 | 23.4284 | 0.7312 | 0.2683 | 37.34785 | 422221_6 |
| P0/P1/P2 +25k / ARM=p1_residual3 | 56 | 0.0256 | 25,000/125,000 | 23.4286 | 0.7315 | 0.2684 | 37.31322 | 422221_7 |
| P0/P1/P2 +25k / ARM=p2_residual7 | 56 | 0.0256 | 25,000/125,000 | 23.4556 | 0.7318 | 0.2679 | 37.14938 | 422221_8 |

주요 과거 판단: factorized 대비 decoder-causal context가 유효했고, 복원을 고정한 probability-only에서는 even/odd Split이 Shared 추가학습 대조군보다 실제 bytes를 줄였다. P1/P2의 추가 이득은 두λ에서 일관되게 크지 않았다. Full low-LR의 `.0064/P0`은 최종 저장 실패로 완료3개에 포함하지 않는다.

### 9.4 외부 스크리닝 기록

E82/E83 정렬·context 스크리닝은 보존 문서에 완료 결과가 있다. GlobalSplat16K·2048 tokens·train55/test12씬·고정 PCA 심볼·정적 histogram/교차 entropy 기반이다. 현행4096-token closed-loop 양자화·actual full-test codec과 다른 대리 진단이므로 위 RD 숫자에 합치지 않는다. 여기서 제안된 정렬 후보는 이후 실제 codec 학습/평가로 별도 검증했다.
<a id="other"></a>

## 10. 그 밖의 평가 수치와 CleanToken 별도 분기

현존 로그를 추가로 검색해 위 목록에 없는 **37개 평가 파일**을 찾았다. 아래는 기록 누락을 막기 위한 별도 목록이다. `CleanToken/` 경로는 현재 `tokencomp/`와 다른 repository·구조·훈련 recipe이며 현재 MSH-only/two-path pilot과 맞춘 대조군이 아니다.

Checkpoint step은 saved metadata가 출력된 경우 그 값을 쓰고, 그렇지 않으면 **파일명 근거**임을 표시했다. Eval Hydra의 `trainer.max_steps` 기본값으로 과거 학습 step을 추정하지 않았다.31/32/97/100장 subset은6,991장 전체 test와 구분한다. 일부32장 평가는 같은 checkpoint의 반복/진단이다.

### tokencomp 과거 실행

| 설정/변형 | λ | 평가 장면 | Step 증거 | PSNR | SSIM | LPIPS | KiB/scene | 로그 |
|---|---|---|---|---|---|---|---|---|
| Full nonlinear Split joint500k | 0.0064 | 6991 | 500000 | 19.2332 | 0.5814 | 0.4142 | 26.54527 | slurm-gs-joint500k-eval-428944.out |
| Full nonlinear Split joint checkpoint220k | 0.0064 | 6991 | 220000 | 18.8065 | 0.5701 | 0.4366 | 25.49765 | slurm-gs-main-eval-426725.out |
| 기존 Split checkpoint 32장 재평가 | 0.0064 | 32 | 10000 | 24.8145 | 0.7953 | 0.2105 | 80.87854 | slurm-gs-main-eval-428943.out |
| 초기 minimal L2/kernel3/noise | 0.0256 | 6991 | 50000 | 23.3109 | 0.7223 | 0.2845 | 32.52413 | slurm-gs-score-minimal-eval-EEE441558.out |
| 같은 Split checkpoint score 진단32장 | 0.0064 | 32 | 10000 | 24.8145 | 0.7953 | 0.2105 | 80.87854 | slurm-gs-score-pred-429783.out |
| Nonlinear32 기존50k test100 | 0.0064 | 100 | 50000 | 24.6772 | 0.7634 | 0.2389 | 99.02039 | slurm-gs-nonlinear50k-100-436236.out |

### CleanToken 별도 분기

| 설정/변형 | λ | 평가 장면 | Step 증거 | PSNR | SSIM | LPIPS | KiB/scene | 로그 |
|---|---|---|---|---|---|---|---|---|
| single_off; paths=1; base_rank=0; Morton=false | 0.0128 | 6991 | 50000 (파일명) | 19.8166 | 0.5891 | 0.4090 | 10.70241 | slurm-gs-h1d-rd4-eval-446171_1.out |
| dual_off; paths=2; base_rank=0; Morton=false | 0.0064 | 6991 | 50000 (파일명) | 20.9980 | 0.6333 | 0.3691 | 15.05392 | slurm-gs-h1d-rd4-eval-446171_10.out |
| lowrank_off; paths=2; base_rank=56; Morton=false | 0.0128 | 6991 | 50000 (파일명) | 20.2736 | 0.6056 | 0.3896 | 10.62706 | slurm-gs-h1d-rd4-eval-446171_17.out |
| single_off; paths=1; base_rank=0; Morton=false | 0.0064 | 6991 | 50000 (파일명) | 20.0586 | 0.5975 | 0.4130 | 12.65733 | slurm-gs-h1d-rd4-eval-446171_2.out |
| plain4; paths=1; base_rank=0; Morton=false | 0.0256 | 6991 | 50000 (파일명) | 18.6180 | 0.5522 | 0.4659 | 5.58526 | slurm-gs-h1d-rd4-eval-446171_24.out |
| plain4; paths=1; base_rank=0; Morton=false | 0.0128 | 6991 | 50000 (파일명) | 17.6461 | 0.5303 | 0.5192 | 7.75222 | slurm-gs-h1d-rd4-eval-446171_25.out |
| plain4; paths=1; base_rank=0; Morton=false | 0.0064 | 6991 | 50000 (파일명) | 19.0060 | 0.5639 | 0.4476 | 7.14387 | slurm-gs-h1d-rd4-eval-446171_26.out |
| single_off; paths=1; base_rank=0; Morton=false | 0.0032 | 6991 | 50000 (파일명) | 20.1589 | 0.5996 | 0.3949 | 20.08312 | slurm-gs-h1d-rd4-eval-446171_3.out |
| single_on; paths=1; base_rank=0; Morton=true | 0.0128 | 6991 | 50000 (파일명) | 20.7294 | 0.6275 | 0.3644 | 12.13725 | slurm-gs-h1d-rd4-eval-446171_5.out |
| single_on; paths=1; base_rank=0; Morton=true | 0.0064 | 6991 | 50000 (파일명) | 21.1963 | 0.6411 | 0.3566 | 19.82998 | slurm-gs-h1d-rd4-eval-446171_6.out |
| single_on; paths=1; base_rank=0; Morton=true | 0.0032 | 6991 | 50000 (파일명) | 21.3039 | 0.6448 | 0.3455 | 28.78144 | slurm-gs-h1d-rd4-eval-446171_7.out |
| dual_off; paths=2; base_rank=0; Morton=false | 0.0128 | 6991 | 50000 (파일명) | 20.8375 | 0.6310 | 0.3769 | 11.02984 | slurm-gs-h1d-rd4-eval-446171_9.out |
| dual_off; paths=2; base_rank=0; Morton=false | 0.0032 | 6991 | 50000 (파일명) | 21.3255 | 0.6472 | 0.3636 | 21.10988 | slurm-gs-h1d-rd4-eval-446669_11.out |
| dual_on; paths=2; base_rank=0; Morton=true | 0.0128 | 6991 | 50000 (파일명) | 21.6812 | 0.6630 | 0.3363 | 14.62718 | slurm-gs-h1d-rd4-eval-446669_13.out |
| dual_on; paths=2; base_rank=0; Morton=true | 0.0064 | 6991 | 50000 (파일명) | 22.0580 | 0.6752 | 0.3178 | 22.75951 | slurm-gs-h1d-rd4-eval-446669_14.out |
| dual_on; paths=2; base_rank=0; Morton=true | 0.0032 | 6991 | 50000 (파일명) | 22.4730 | 0.6916 | 0.3023 | 43.52813 | slurm-gs-h1d-rd4-eval-446669_15.out |
| plain4; paths=1; base_rank=0; Morton=false | 0.0032 | 6991 | 50000 (파일명) | 17.0720 | 0.5214 | 0.5439 | 7.34954 | slurm-gs-h1d-rd4-eval-446669_27.out |
| lowrank_off; paths=2; base_rank=56; Morton=false | 0.0064 | 6991 | 50000 (파일명) | 20.4064 | 0.6109 | 0.3883 | 14.34568 | slurm-gs-h1d-rd4-eval-447711_18.out |
| lowrank_off; paths=2; base_rank=56; Morton=false | 0.0032 | 6991 | 50000 (파일명) | 20.5801 | 0.6168 | 0.3852 | 18.18074 | slurm-gs-h1d-rd4-eval-447711_19.out |
| lowrank_on; paths=2; base_rank=56; Morton=true | 0.0128 | 6991 | 50000 (파일명) | 21.3799 | 0.6509 | 0.3429 | 13.90212 | slurm-gs-h1d-rd4-eval-447711_21.out |
| lowrank_on; paths=2; base_rank=56; Morton=true | 0.0064 | 6991 | 50000 (파일명) | 21.4821 | 0.6527 | 0.3428 | 17.45181 | slurm-gs-h1d-rd4-eval-447711_22.out |
| lowrank_on; paths=2; base_rank=56; Morton=true | 0.0032 | 6991 | 50000 (파일명) | 22.1995 | 0.6820 | 0.3141 | 36.48779 | slurm-gs-h1d-rd4-eval-447711_23.out |
| lowrank_base_morton_on; paths=2; base_rank=56; Morton=true | 0.0256 | 6991 | 50000 (파일명) | 20.9457 | 0.6340 | 0.3730 | 8.97332 | slurm-gs-hyper1d-50k-eval-442219_0.out |
| msh2_morton_off; paths=2; base_rank=0; Morton=false | 0.0256 | 6991 | 50000 (파일명) | 20.6579 | 0.6209 | 0.3827 | 8.96086 | slurm-gs-hyper1d-50k-eval-442219_1.out |
| hyper1d; Morton=false | 0.0256 | 31 | 21500 (파일명) | 18.8527 | 0.5758 | 0.4687 | 6.03528 | slurm-gs-hyper1d-eval-436141.out |
| hyper1d; Morton=false | 0.0256 | 31 | 23000 (파일명) | 18.8870 | 0.5764 | 0.4650 | 6.33795 | slurm-gs-hyper1d-eval-436157.out |
| hyper1d; Morton=false | 0.0256 | 97 | 23500 (파일명) | 18.7793 | 0.5641 | 0.4718 | 6.03028 | slurm-gs-hyper1d-eval-436171.out |
| single_msh; paths=1; Morton=true | 0.0256 | 6991 | 50000 (파일명) | 20.7369 | 0.6290 | 0.3660 | 9.71217 | slurm-gs-msh-fulltest-440685_0.out |
| base_residual_msh; paths=2; Morton=true | 0.0256 | 6991 | 50000 (파일명) | 21.4372 | 0.6521 | 0.3446 | 10.46190 | slurm-gs-msh-fulltest-440685_1.out |
| legacy; paths=1; Morton=false | 0.0256 | 6991 | 50000 (파일명) | 20.1189 | 0.6000 | 0.4030 | 8.59442 | slurm-gs-msh-off-eval-441057.out |
| lowrank_off; paths=2; base_rank=56; Morton=false | 0.0256 | 6991 | 50000 (파일명) | 19.9208 | 0.5917 | 0.4186 | 8.08557 | slurm-gs-rd4-lr0256-eval-443942.out |

`slurm-gs-msh-fulltest-440685_0.out / _1.out`은6,991장과 최종 PSNR/SSIM/LPIPS/bytes를 모두 출력했다. 다만 stderr에는 후속 shell의 `cannot create temp file for here-document: No space left on device`가 있어 **평가 지표 산출 완료·후속 shell 오류**로 기록한다. 정상 job 종료를 확인한 것으로 쓰지 않는다.

예전 Hyper1D single/dual/lowrank-base의 수치나 joint500k 결과를 현재 fresh 구조 pilot의 우열 근거로 쓰지 않는다. 이 목록은 출력된 평가 지표를 보관한 것이며, 동일 초기화·학습 budget·scene/frame 정합까지 새로 맞춘 재실험이 아니다.
<a id="failures"></a>

## 11. 실패·중단·완료 여부 미확인 이력

| 작업 | 근거/상태 | 현재 취급 |
|---|---|---|
| 456951 /457099 원래 대조군 | FAILED 및 step signal9; QOSMaxGRESPerUser만으로 강제종료 원인을 확정하지 않음 | 458782/458783으로 full-state 복구 후 최종50k 평가 완료 |
| 451821 /451825 /451827 | 과거 ENOSPC 중단 | 453064/453066/453068 재개 후 완료점은 현재 RD에 포함; 옛 실패를 현재 진행으로 세지 않음 |
| NN-score 448650 계보 /450904 /450934 | 과거 재개 실패 및 재시도 기록 | 최종 eval450935의 완료 RD 점을 사용; 실패 job을 추가 모델로 세지 않음 |
| Mean-offset 433719_0/1 | OSError28 No space left; 최종 성공 평가 없음 | 20k 설정의 시도 기록, 완료 ablation 결과로 사용하지 않음 |
| Mean-offset 재시도434543_0/1 | /tmp/pymp ENOSPC·취소 오류; 시작 직후 진행만 있음 | 두 arm의 완성된 비교 결과 없음; 현재 진행 작업으로 세지 않음 |
| Nonlinear test100 job436182 | 11frames scene에서20개의 disjoint context/target을 요구해 ValueError | 실패 subset 평가; 다른 λ의436236 test100 완료와 구분 |
| MSH fulltest440685_0/1 | 6,991장 지표 출력 후 shell here-document 임시 파일 ENOSPC | 평가 수치는 보존; 정상 job 종료와 구분 |
| 과거 Full low-LR .0064/P0 | 최종 checkpoint 저장 실패 | 완료48개 중 low-LR3개와 구분 |
| 454797 broad mechanism /454488 complexity | 현재 stdout 비어 있음 | 접수/완료를 추정하지 않음; 명시적 완료 후속 job을 사용 |
| 그 밖의 옛 array/eval 시도 | 아래 전체 로그 목록에 포함; 완료 지표 없는 시도도 존재 | 오래된 progress만으로 지금 RUNNING/PENDING이라 하지 않음 |

파일명의 array ID와 실제 stderr의 Slurm child job ID가 다를 수 있다. 실패 원인 확정이나 현재 취소 여부에는 해당 job의 accounting이 필요하다. 이번 정리는 옛 작업을 재제출하거나 취소하지 않는다.
<a id="pending"></a>

## 12. 기존 후속 계획과 결과 미확인 항목

아래는 과거에 제안·구현한 항목의 상태 기록이다. 이번 새 fresh-init RD 계획은 싣지 않는다. 이미 완료된457098·457100·457829는 이 대기 목록에서 제거했다.

| 항목 | 확인 가능한 범위 |
|---|---|
| None+D4 OFF16·N01·native 전체 test·T06 렌더 후속 묶음 |10월20일 실행 도구 구현; 현재 로컬에 `slurm-gs-followup20-*` 로그가 없어 접수·완료를 확정하지 않음 |
| 구조 pilot 전체6991장 test |네 arm 모두128장 평가 완료; 전체 test 완료 기록 없음 |
| 구조별 추가λ·bitrate 맞춤 |MSH-only .0032·two-path56 .0512 우선 탐색을 제안했으나 이 과거 묶음의 완료 결과 없음 |
| Morton L3 OFF12 .0128(T17) |과거 계획 deferred; 완료 결과 없음 |
| Hilbert L3 ON/OFF12 추가 seed |과거 계획 deferred; D4 seed 비교로 대체했다고 간주하지 않음 |
| B0/T05 전체6991장 확장 |기존 matched256 결과만 완료 |
| OFF/주요 구조 complexity 보완 |기존37-checkpoint profile 이후 추가 완료 자료 없음 |

결과 미확인은 서버에서 미접수라는 뜻이 아니다. 이 정리 작업에서는 기존 job을 재제출하거나 취소하지 않았다.

<a id="ledger"></a>

## 13. 과거 Slurm 로그 색인 — 10월20일16:47 보존본

아래284개 stdout·49개 prefix의 목록은 원 snapshot의 이력이다. 현재 로컬 전체 파일 수가 아니며, 이번 새 fresh-init job은 추가하지 않았다. 이후 완료 평가457098·457100·457829는 앞의 결과 표에 반영했다.

현재 로컬 `slurm/`의 stdout284개를 파일명 prefix별로 전부 나열했다. stderr도284개가 있다. 이 표는 **파일/시도 색인**이며 독립 모델 수나 현재 queue 상태가 아니다. CleanToken의 다른 분기, 재시도, 실패, 빈 로그도 포함된다. 오래된48개 평가 JSON의 원본 로그는 현재 폴더에 없으므로 이284개와 별도다.

| 로그 prefix | stdout 수 | 파일명에 나타난 job/task 식별자 |
|---|---|---|
| slurm-gs-arch-eval | 3 | 457742, 457744, 457746 |
| slurm-gs-arch-prepare | 1 | 457740 |
| slurm-gs-arch-train | 4 | 457741, 457743, 457745, 457828 |
| slurm-gs-b0-matched | 1 | 458024 |
| slurm-gs-complexity | 2 | 454488, 455251 |
| slurm-gs-decode-full | 1 | 433728 |
| slurm-gs-encode-full | 1 | 433727 |
| slurm-gs-fixed-order | 1 | 458818 |
| slurm-gs-h1d-rd4-1g | 11 | 442942_0, 442942_1, 442942_10, 442942_2, 442942_3, 442942_4, 442942_5, 442942_6, 442942_7, 442942_8, 442942_9 |
| slurm-gs-h1d-rd4-eval | 34 | 445929_1, 445929_10, 445929_17, 445929_2, 445929_24, 445929_25, 445929_26, 445929_3, 445929_5, 445929_6, 445929_7, 445929_9, 446171_1, 446171_10, 446171_17, 446171_2, 446171_24, 446171_25, 446171_26, 446171_3, 446171_5, 446171_6, 446171_7, 446171_9, 446669_11, 446669_13, 446669_14, 446669_15, 446669_27, 447711_18, 447711_19, 447711_21, 447711_22, 447711_23 |
| slurm-gs-h1d-rd4-lr | 4 | 442943_0, 442943_1, 442943_2, 442943_3 |
| slurm-gs-h1d-rd4-resume | 12 | 445307_0, 445307_1, 445307_2, 445307_3, 445307_4, 445307_5, 445307_6, 445308_0, 445308_1, 445308_2, 445308_3, 445308_4 |
| slurm-gs-hsched-anchors | 1 | 455166 |
| slurm-gs-hyper1d | 4 | 434935, 434938, 434941, 434951 |
| slurm-gs-hyper1d-50k-eval | 2 | 442219_0, 442219_1 |
| slurm-gs-hyper1d-eval | 4 | 436140, 436141, 436157, 436171 |
| slurm-gs-hyper1d-hq50k | 4 | 437624_0, 437624_1, 437624_2, 437624_3 |
| slurm-gs-hyper1d-plain4 | 1 | 436321 |
| slurm-gs-hyper1d-resume | 2 | 436008, 436859 |
| slurm-gs-independent-receiver | 1 | 458784 |
| slurm-gs-joint-ext-v10 | 2 | 425438, 426699 |
| slurm-gs-joint-v10 | 3 | 424577, 424688, 425420 |
| slurm-gs-joint-v12 | 2 | 424117, 424118 |
| slurm-gs-joint500k-eval | 1 | 428944 |
| slurm-gs-l3-anchors | 1 | 454995 |
| slurm-gs-lrbase-50k | 1 | 441332 |
| slurm-gs-main-eval | 4 | 426725, 428792, 428798, 428943 |
| slurm-gs-mean-offset | 4 | 433719_0, 433719_1, 434543_0, 434543_1 |
| slurm-gs-mechanism | 1 | 454797 |
| slurm-gs-msh-compare50k | 4 | 438115_0, 438115_1, 438123_0, 438123_1 |
| slurm-gs-msh-fulltest | 4 | 440681_0, 440681_1, 440685_0, 440685_1 |
| slurm-gs-msh-off-eval | 1 | 441057 |
| slurm-gs-msh2-off50k | 1 | 441056 |
| slurm-gs-nonlinear50k-100 | 2 | 436182, 436236 |
| slurm-gs-ns-resume | 2 | 450904, 450934 |
| slurm-gs-oc-recovery | 15 | 451809, 451811, 451813, 451815, 451817, 451819, 451821, 451823, 451825, 451827, 451868, 451870, 453064, 453066, 453068 |
| slurm-gs-order-context | 62 | 445516, 445518, 447719, 447721, 447723, 447725, 447727, 447729, 447738, 447740, 447742, 447744, 447746, 447900, 448644, 448646, 448648, 448650, 448652, 449559, 449561, 450937, 450939, 450977, 450979, 450981, 450983, 450985, 450987, 450989, 450991, 450993, 450995, 454665, 454667, 454669, 455235, 455237, 455239, 455241, 455243, 455245, 455327, 455329, 455331, 455333, 455335, 455337, 455339, 455341, 455343, 455928, 455930, 455932, 455934, 456895, 456897, 456899, 456901, 456949, 456951, 457099 |
| slurm-gs-order-context-eval | 61 | 445517, 445519, 447720, 447722, 447724, 447726, 447728, 447730, 447739, 447741, 447743, 447745, 447899, 447901, 448645, 448647, 448649, 449560, 449562, 450935, 450938, 450940, 451810, 451812, 451814, 451816, 451818, 451820, 451824, 451869, 451871, 453065, 453067, 453069, 454666, 454668, 454670, 455236, 455238, 455240, 455242, 455244, 455246, 455328, 455330, 455332, 455334, 455336, 455338, 455340, 455342, 455344, 455929, 455931, 455933, 455935, 456896, 456898, 456900, 456948, 456950 |
| slurm-gs-overhead | 1 | 423729 |
| slurm-gs-paper-control-resume | 2 | 458782, 458783 |
| slurm-gs-path-diagnostics | 1 | 458004 |
| slurm-gs-rd4-lr0256-eval | 1 | 443942 |
| slurm-gs-score-minimal | 1 | 441307 |
| slurm-gs-score-minimal-eval | 1 | EEE441558 |
| slurm-gs-score-pred | 1 | 429783 |
| slurm-gs-score-stages | 4 | 441552_1, 442209_0, 442211_2, 443968_2 |
| slurm-gs-score-stages-eval | 3 | 441553_1, 442210_0, 443969_2 |
| slurm-gs-seed-pair-eval | 2 | 457787, 457789 |
| slurm-gs-seed-pair-train | 2 | 457786, 457788 |
<a id="sources"></a>

## 14. 출처와 검증 범위

- [기존 전체 종합본](NFCGS_EXPERIMENTS_MASTER_2026-10-20.md): 초기 탐색·과거48개·기타37개·진단·profile·실패·로그 색인의 원본. 원 snapshot을 보존했다.
- [10월20일21:59 완료 검토](analysis/nfcgs_logs_2026-10-20_2159/REPORT.md), [원 inventory](analysis/nfcgs_logs_2026-10-20_2159/inventory.json): L3 마지막 점과 구조4개. 원 JSON의 `completed_positive_rate_points`는63으로 남아 있지만 실제 `results`는64개다. 이 문서는 실제 행 수를 세어64개를 읽고 최종 H+D4 OFF16 한 점을 추가했다.
- [H+D4 두 seed 최종 비교](analysis/nfcgs_d4_context_2026-10-21/REPORT.md): 기본 OFF 완료와 네 평가 원본 SHA.
- [통합 데이터·검증 결과](analysis/nfcgs_legacy_summary_2026-10-24/summary.json), [생성·검산 코드](analysis/nfcgs_legacy_summary_2026-10-24/build_summary.py).
- [기존 통합 기계 판독 자료](analysis/nfcgs_experiment_master_2026-10-20/all_experiments.json), [과거 context48개 지표](NFCGS_CONTEXT_EXPERIMENT_HISTORY_2026-09-14_metrics.json).
- [T06 수치 원문](analysis/nfcgs_logs_2026-10-20_1636/provided_t06_report.md), [복잡도37개](analysis/nfcgs_complexity_455251/REPORT.md), [초기 Linear 표](NFCGS_EXPERIMENT_RESULTS_2026-09-10.md).

이번 통합에서 주 RD65개·λ0 한 개·추가 seed2개·구조4개, 총72개 완료 평가의 stdout SHA·PSNR·SSIM·LPIPS·actual bytes·처리 장면 수를 다시 대조했다. 검증 범위는 위72개 평가 기록이며, 독립 학습의 전체 개수를 뜻하지 않는다. 과거48개 보존 자료·초기 사용자 표·기타37개와 진단/profile의 수치는 기존 종합본에서 범위와 한계를 유지해 옮겼다. 그 모든 원본을 이번에 새로 재검증한 것은 아니다.

서버의 실시간 상태나 raw checkpoint를 확인한 문서가 아니다. 다른 초기화·checkpoint 계보·장면 집합·seed·평가 프로토콜의 수치를 한 곡선에 섞지 않는다. 집계 지표 검증을 장면별 frame/tensor bit-exact 검증이나 training-seed 통계 검정으로 확대하지 않는다. 이 작업에서는 학습 코드·서버 job·dependency를 변경하지 않았다.

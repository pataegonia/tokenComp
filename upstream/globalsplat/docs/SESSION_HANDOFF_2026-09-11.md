# GlobalSplat 압축 연구 — 다음 세션 인수인계 (2026-09-11)

> **2026-09-15:** 실험 결과만 이어받을 때는 최신
> [실험 전용 handoff](SESSION_HANDOFF_EXPERIMENTS_2026-09-15.md)를 먼저 읽는다.
> 이 문서의 과거 실행 예정/미확인 표기보다 최신 handoff의 완료 상태를 우선한다.

## 2026-09-14 후속 — 고정-symbol entropy 진단 구현

사용자가 score의 큰 bit 비중이 실제 확률 모델 병목인지 검증하는 작업을 승인했다.
[고정-symbol 진단](NFCGS_ENTROPY_PROBE_2026-09-14.md)을 구현했다. 두 λ 모두 기존
Nonlinear32 Full(P0)+Split probability run `20260913_112316`의 step10k가 부모다.
Codec은 동결하고 TRAIN 128장면으로 확률표를 추정한 뒤 TEST 128장면의 동일 정수
symbol을 재부호화한다. Score/residual별 예상 비트·CDF·bypass·실제 비트, 단독 경로
절감량 및 paired bootstrap, 복원 feature 완전 일치를 보고한다. 더 강한 모델을
위한 정수 symbol 캐시는 선택 사항이며 저장 공간 문제를 고려해 기본 OFF다.

이후 서버 job array `423055`도 완료했다. `.0064`는 output 내부 job `423056_0`,
`.0256`은 `423055_1`이며 둘 다 128 TRAIN fit + 128 TEST eval, exact feature
round-trip, 빈 stderr를 확인했다. Context table은 전체 scene bytes를 각각 0.736%
`[0.656,0.821]`, 0.759% `[0.673,0.855]` 줄였다. Score-only 효과는 0.713/0.738pp,
residual marginal 효과는 0.024/0.021pp였다. rANS 자체의 NLL 초과는 두 경로 합쳐
장면당 약159B/101B이고 그중 coder gap은 약60B라 주 병목은 coder가 아니다.
개선량은 초기 score group, 특히 odd에 집중됐다. 세부 수치와 해석은 위 진단
문서의 Completed result를 본다. 이는 128장면 screening이고 새 full-test 결과나
score가 전역적으로 최대 병목이라는 증명은 아니다. 기존 완료 실험과 미커밋
변경은 보존했다.

## 2026-09-14 업데이트 — 전체 실험 이력 통합

첫 Linear context부터 P0/P1/P2 및 probability-only까지의 설정·구현·동기·결과는
[전체 실험 기록](NFCGS_CONTEXT_EXPERIMENT_HISTORY_2026-09-14.md)을 우선 참조한다.
로컬 완료 eval 45개(초기 nonlinear 배경 8개 포함), 최신 train 17개를 대조했으며,
[원시 지표와 checkpoint 경로](NFCGS_CONTEXT_EXPERIMENT_HISTORY_2026-09-14_metrics.json)도 보존했다.
이전 사용자 표 12개를 포함하면 문서화된 codec 결과는 57개이고 codec-OFF는 별도다.

P0/P1/P2 9개와 probability-only 8개는 완료 로그가 있다. Full low-LR 10k는
구현은 있으나 현재 로컬 로그에서 결과를 확인하지 못했다. 아래 과거의 “진행 중/미제출”
표기는 해당 날짜의 스냅샷이며 현재 서버 상태를 뜻하지 않는다.
현재 결과상 두 λ 공통의 실용적 후속 기준점은 Nonlinear32 + Full(P0) + Split이다.
이는 추천이며 코드의 기본 entropy 설정은 여전히 shared다.

정정: 6,986은 warm-up 5회를 제외한 시간 측정 횟수이고 eval 진행 수는 6,991이다.
원래 Full의 step50k는 초기 transform50k에 더한 context 단계 번호로 누적100k다.
Probability-only의 집계 화질 일치와 서버 전체 복원 텐서의 bit-exact 검증은 구분한다.
이번 정리에서는 실험 재실행·서버 제출·기존 코드 변경을 하지 않았다.

## 2026-09-12 후속 업데이트 — Full 추가

사용자는 두 λ에서 Full을 공통 기본 구조로 개선하기로 결정했다.
기존 spatial-predictor task 0–5는 실행 중이라고 알렸으며, mapping과 출력 경로를 보존했다.
새 task `6=P0 / 7=P1 / 8=P2`는 모두 λ=.0256 Full의 완료된 nonlinear context
task9 checkpoint를 부모로 사용한다. `.0064 Full`은 기존 task 0/2/4를 활용한다.

train/eval/submitter 기본 array는 새 `6–8`이며, 추가 25k와 기존 batch/seed/LR recipe를
유지한다. 서버에서 반영할 파일과 제출 명령은
[실험 문서](NFCGS_SPATIAL_PREDICTOR_EXPERIMENT_2026-09-11.md)에 정리했다.
관련 테스트 58개 통과. 이 후속 작업에서는 서버 queue 조회나 job 제출을 하지 않았다.
아래 내용은 9월11일 결과/설계 스냅샷이며, 다음 실험 선택에는 이 업데이트가 우선한다.

## 먼저 읽을 요약 — 9월11일 스냅샷

목표는 GlobalSplat 기반을 유지하면서 실제 scene bitstream의 RD 성능을 개선하는 것이다.
**추적하던 학습/eval은 모두 완료됐다.** 최신 Nonlinear32+Spatial/Full 네 개의 결과까지 확인했다.
현재 결과가 있는 codec 설정 40개(사용자 초기 표 12개 + 로컬 완료 eval 로그 28개), codec-OFF reference 별도.

다음 추천은 **Spatial predictor 개선: 기존 P0 / 비선형 kernel3 보정 P1 / 비선형 kernel7 보정 P2**다.
두 λ에서 대조군을 포함해 총 6개, 선택 checkpoint에서 동일 추가25k를 학습하는 설계다.
**P1/P2 코드와 6조건 Slurm은 후속 작업에서 로컬 구현·검증했으며, 새 실험은 아직 제출하지 않았다.**
마지막 사용자 요청은 다음 세션을 위한 정리였으며, 이 문서는 자동 실행 명령이 아니다.

우선 읽을 문서:
1. [최신 Nonlinear+Context 결과](NFCGS_NONLINEAR_CONTEXT_RESULTS_2026-09-11.md)
2. [이전 전체 실험 36개와 원본 로그](NFCGS_EXPERIMENT_RESULTS_2026-09-10.md)
3. [기존 다음 실험 설계](NFCGS_NEXT_EXPERIMENT_2026-09-10.md)
   - 이 파일의 nonlinear 결합 실험은 이제 완료됐다.
   - 당시 다음으로 제안했던 residual conditioning보다 **Spatial predictor 개선을 우선**하기로 추천을 조정했다.
4. [초기 탐색 보고서](ABLATION_PROGRESS_2026-09-02.md)
   - 초기 탐색과 최근 paper24 결과는 학습 초기화/프로토콜을 확인하고 비교해야 한다.

## 사용자 선호와 운영 환경

- 한국어로 간결하게 설명하며, 실험을 왜 하는지와 결과가 무엇을 검증했는지 설명한다.
- GPU 서버에 코드를 올려 Slurm으로 학습한다. 로컬 변경이 서버에 자동 동기화되는 것은 아니다.
- GPU마다 독립 실험을 병렬 실행하는 방식을 선호한다. 이전에 최대8개 언급 후,
  가능한 실험을 병렬로 제출하고 자원 수는 Slurm이 스케줄하도록 요청했다.
- 모델당 batch/step을 유지한다. GPU 수를 늘린다고 effective batch를 늘리는 DDP 실험으로 바꾸지 않는다.
- 이전 명시 요청: 향후 GPU job은 ariel-v12, gpu:1.
- 변경 파일은 클릭할 수 있는 파일 링크로 제공한다.
- 현재까지 결과 확인/문서 작성 턴에서는 모델 수정이나 서버 job 제출을 하지 않았다.
- 로컬 로그는 서버에서 받은 스냅샷이다. 서버 실시간 queue 상태를 확인했다고 주장하지 않는다.

로컬 workspace:
    C:\Users\a0103\Desktop\school\vml\global

실제 integrated repository:
    C:\Users\a0103\Desktop\school\vml\global\upstream\globalsplat

로컬 서버 로그:
    C:\Users\a0103\Desktop\school\vml\global\slurm

서버 repository:
    /ceph_data/clue9986/tokencomp/upstream/globalsplat

서버 conda 환경: globalsplat
데이터 경로(로그): /data3/local_datasets/re10k

workspace의 src/, experiments/에는 오래된 별도 코덱/아카이브가 많다.
최근 통합 모델을 수정할 때는 upstream/globalsplat/globalsplat/compression/이 대상이다.
최신 서버 checkpoint 및 per-scene 평가 JSON은 현재 로컬 outputs에 없다.

## Git 상태 — 2026-09-11 로컬 확인

- 현재 branch: session-after-score-context
- 변경 전 보존 branch: session-before-score-context
- main도 있음.
- 사용자 레포 remote 이름은 tokencomp:
    https://github.com/pataegonia/tokenComp.git
- origin은 여전히 upstream:
    https://github.com/R-Itk/globalsplat.git
- 사용자 레포가 origin이라고 가정하지 않는다. push는 이 인수인계 턴에서 하지 않았다.

최근 commit:
    0804f8e Enforce LF for Slurm scripts
    3f034fe Fix score-context training entrypoints
    66dccb7 Add contextual score entropy experiments
    cc707c2 Snapshot before contextual score entropy

작업 트리가 dirty하다. 코드/설치/Slurm/test 등 기존 미커밋 변경이 있고 신규 follow-up
스크립트와 결과 문서들도 untracked다. 다음 세션 시작에 git status/diff를 읽고,
사용자 변경을 reset/checkout하거나 전체 파일을 과거 버전으로 덮어쓰지 않는다.
이 보고서들은 파일로 저장됐지만 commit/push된 것으로 취급하면 안 된다.

## 현재 모델 구조와 중요한 구현 의미

GlobalSplat encoder
→ appearance512 + observable geometry224 =736 channel, scene tokens4096
→ 전송하는 FP16 scene mean을 제거
→ Morton 순서, low-rank analysis (rank56/80, linear/nonlinear32)
→ quantized score를 entropy coding
→ low-rank synthesis
→ residual hyperprior로 남은 오차 복원
→ 원래 GlobalSplat Gaussian decoder.

- Nonlinear32는 analysis와 synthesis에 hidden32 MLP를 사용하는 옵션이다.
- Residual은 multiscale1D adapter, N192/M320, mean/scale hyperprior.
- Rank는 채널 폭이며 중요도 점수나 분산학습 process rank가 아니다.
  Slurm 공통 runner 전에 export -n RANK를 유지한다.
- Score context:
  - Mean: 이미 전송한 scene mean에서 offset/step 예측.
  - Channel: 먼저 복호화한 channel slices 사용.
  - Spatial: Morton even anchor → odd 예측, two-pass.
  - Full: Mean+Channel+Spatial.
- Rank56 Full의 channel slices는16/16/16/8. Spatial만 쓰면2개 entropy 단계,
  Full이면4개 slices×2=8단계. 실행시간이 반드시4배가 되는 것은 아니다.
- 현재 spatial predictor는 단일 선형 Conv2d(width,width,kernel=(1,3)).
  odd에서는 좌우 인접 even anchor 두 개가 유효 입력이다.
- Channel predictor는1×1 선형 Conv.
- scene mean conditioner는 LayerNorm→Linear→GELU→Linear.
  SCORE_CONTEXT_HIDDEN=64를 바꾸는 것만으로 spatial predictor가 비선형이 되지 않는다.
- 이 context는 예측값뿐 아니라 양자화 간격도 바꾼다.
  동일 symbol에 probability model만 교체한 lossless recoding 실험이 아니다.
- Residual entropy는 z_hat에만, synthesis는 y_hat에만 조건화돼 있다.
  decoded base conditioning은 현재 integrated 경로에 아직 없다.

핵심 파일:
- globalsplat/compression/score_context.py
- globalsplat/compression/codec.py
- globalsplat/compression/config.py
- globalsplat/compression/checkpoint.py
- globalsplat/compression/bitstream.py
- globalsplat/compression/residual.py
- globalsplat/model/model_wrapper.py (실제 bitstream 평가/JSON/timing)
- scripts/initialize_nfcgs_from_vanilla.py (weight warm-start)
- scripts/check_nfcgs_checkpoint.py (eval 전 metadata/shape 검사)
- tests/test_score_context_codec.py
- tests/test_nfcgs_codec.py
- tests/test_transform_slurm.py
- tests/test_slurm_eval.py

## 학습과 eval 공통 조건

최근 continuation/context 실험:
- 각 λ의 원래50k checkpoint에서 weight warm-start, optimizer/scheduler 새로 시작.
- 추가50,000 optimizer steps, seed111123, LR1e-4와 기존 schedule.
- micro-batch2 × accumulate4 = effective batch8.
- paper24 subset consistency: context pool24, 두13-view branches, shared targets12.
- codec-wide joint training scope, 기존 GlobalSplat backbone freeze 정책.
- bf16-mixed, train data workers8. 독립 task당 GPU1개.
- Residual ON, Morton ON, nonlinear hidden32, slice16, context hidden64.
- 50k+50k는 uninterrupted optimizer-state resume100k와 다르다.

평가:
- RE10K all-test, context12 / target8, batch1, actual bitstream encode→decode→render.
- 원본 index7,286개, 완료 로그에서 실제 처리6,991개.
- PSNR/SSIM/LPIPS는 scene 평균.
- 크기는 actual_bytes 평균/1024 = KiB. Mean1472B와 container132B 포함
  (현재 residual-ON 조건), shared model weights는 scene payload에서 제외.
- 초기 사용자 baseline 표는 kB 표기였다. 원시 bytes를 아직 확인하지 못했으므로
  초기 표와 비교한 절감률은 단위가 같은지 확인해야 한다.
- actual_rate_per_scene.json에는 scene ID, context/target frame ID, quality, stream별bytes가 저장된다.
- scores_all_avg.json, benchmark.json, peak_memory.json도 생성된다.
- entropy_decode timer는 decompress_scene 전체(복원 포함)이고 pure entropy 연산만이 아니다.
- 현재 비교는 집계 로그 기준. 최신 per-scene JSON의 scene/frame 완전 일치나 paired bootstrap은 미실시.
- 작은 차이를 통계적 유의성으로 단정하거나 같은 λ 절감률을 BD-rate라고 표현하지 않는다.

## 완료된 job/task 지도

| 실험 | Train job | Eval job | Tasks | RUN |
|---|---|---|---|---|
| Linear context | 419615 | 419616 | 0–7 | 20260908_092102 |
| Linear factorized continuation | 420267 | 420718 | 0–3 | 20260908_184640 |
| Nonlinear factorized Rank56 | 420303 | 420304 | 0,1 | 20260908_195856 |
| Nonlinear factorized Rank80 replacement | 420382 | 420383 | 10,11 | 20260908_221227 |
| Nonlinear Spatial/Full | 420916 | 420919 | 6–9 | 20260910_092000 |

Nonlinear 초기50k transform grid:
- tasks8–14의 eval은419630, task15의 eval은420263. 전8조건 완료.
- task8/9: Rank56 .0064 residual ON/OFF
- task10/11: Rank56 .0256 ON/OFF
- task12/13: Rank80 .0064 ON/OFF
- task14/15: Rank80 .0256 ON/OFF

Linear context grid:
- 0/1 Mean, 2/3 Mean+Channel, 4/5 Mean+Spatial, 6/7 Full.
- 짝수 .0064 / 홀수 .0256, 모두 Rank56.

Nonlinear context grid:
- 0/1 Rank56 factorized (완료)
- 2/3 Rank56 Mean (최근 후속으로 실행 안 함)
- 4/5 Rank56 Mean+Channel (최근 후속으로 실행 안 함)
- 6/7 Rank56 Mean+Spatial (완료)
- 8/9 Rank56 Full (완료)
- 10/11 Rank80 factorized (완료)
- 짝수 .0064 / 홀수 .0256.

과거420303_10/11의 'expects task id 0 through 9' 오류는 구버전 grid를 사용한 실패다.
최신 grid는0–11을 지원하며420382_10/11이 정상 replacement로 완료됐다.
실패한 구job은 별도 결과로 세지 않는다.
Rank80 학습의 copied/recovered 로그는 종료 전에 끊겨 있지만, 새 eval이 실제
최종 global_step50000 checkpoint를 읽고 평가 완료한 것이 확인됐다.

## 핵심 결과

공식 codec-OFF: PSNR24.7004 / SSIM.7682 / LPIPS.2480.

Nonlinear32 초기50k, Residual ON / Morton ON:

| Rank | λ | PSNR | SSIM | LPIPS | KiB |
|---:|---:|---:|---:|---:|---:|
| 56 | .0064 | 24.5082 | .7665 | .2343 | 98.18 |
| 56 | .0256 | 23.6943 | .7381 | .2673 | 46.69 |
| 80 | .0064 | 24.4890 | .7659 | .2367 | 99.15 |
| 80 | .0256 | 23.5833 | .7378 | .2699 | 47.31 |

최근50k+50k, Rank56 / Residual ON / Morton ON:

| λ | Transform/context | PSNR | SSIM | LPIPS | KiB |
|---:|---|---:|---:|---:|---:|
| .0064 | Linear factorized | 24.3024 | .7602 | .2374 | 90.80 |
| .0064 | Linear Spatial | 24.3222 | .7611 | .2369 | 86.07 |
| .0064 | Linear Full | 24.3232 | .7611 | .2368 | 84.91 |
| .0064 | Nonlinear factorized | 24.3022 | .7602 | .2376 | 87.84 |
| .0064 | Nonlinear Spatial (task6) | 24.3244 | .7612 | .2371 | 83.63 |
| .0064 | Nonlinear Full (task8) | 24.3238 | .7611 | .2371 | 82.49 |
| .0256 | Linear factorized | 23.4875 | .7307 | .2729 | 42.26 |
| .0256 | Linear Spatial | 23.5304 | .7319 | .2716 | 39.17 |
| .0256 | Linear Full | 23.5352 | .7323 | .2716 | 39.23 |
| .0256 | Nonlinear factorized | 23.5210 | .7328 | .2693 | 41.26 |
| .0256 | Nonlinear Spatial (task7) | 23.5867 | .7348 | .2678 | 38.24 |
| .0256 | Nonlinear Full (task9) | 23.5707 | .7342 | .2683 | 37.96 |

Rank80 continuation:

| Transform | λ | PSNR | SSIM | LPIPS | KiB |
|---|---:|---:|---:|---:|---:|
| Linear | .0064 | 24.2805 | .7597 | .2399 | 90.94 |
| Linear | .0256 | 23.4740 | .7310 | .2758 | 42.64 |
| Nonlinear | .0064 | 24.2920 | .7601 | .2395 | 88.17 |
| Nonlinear | .0256 | 23.4264 | .7319 | .2756 | 42.31 |

결과 해석:
- Rank80은 nonlinear와 추가 학습을 제공해도 Rank56보다 좋지 않았다.
  Nonlinear continuation에서는 두 λ 모두 Rank56이 크기와 세 화질 지표에서 우세다.
- Linear context Full vs matched Linear factorized:
  .0064에서 −6.50% / +.0208 dB; .0256에서 −7.16% / +.0477 dB.
- 최신 Nonlinear context vs matched Nonlinear factorized:
  task6 −4.79% / +.0222 dB, task8 −6.09% / +.0216 dB,
  task7 −7.32% / +.0657 dB, task9 −7.99% / +.0497 dB.
  네 조건 모두 SSIM/LPIPS도 개선됐다.
- 같은 context의 Linear→Nonlinear 비교:
  .0064에서는 약−2.84%, PSNR 거의 동일, LPIPS는 .0002–.0003 높아졌다.
  .0256에서는 Spatial/Full 모두 크기와 세 화질 지표가 개선됐다.
- 최신 score 비중은 전체 약80–84%, residual은 약14–16%, 나머지는 고정 비용이다.
- 초기50k 고화질 모델과 추가 학습 저율 모델은 operating point가 다르다.
  더 작다는 이유만으로 모든 rate에서 최종 모델이 개선됐다고 하지 않는다.

## 다음 실험 출발점 추천과 정확한 checkpoint

추천은 실용적인 선택이지 유일한 Pareto winner 확정이 아니다.

- λ=.0064: Nonlinear Full(task8).
  Spatial보다1.14 KiB 작고 PSNR 차이는−.0006 dB라 거의 같다.
- λ=.0256: Nonlinear Spatial(task7), 화질을 중시한 선택.
  Full보다 .28 KiB 크지만 PSNR+.0160, SSIM+.0006, LPIPS−.0005다.
  최소 payload 후보 Full(task9)도 보존한다.

서버 공통 parent:
    /ceph_data/clue9986/tokencomp/upstream/globalsplat/outputs/nfcgs_nonlinear_score_context/20260910_092000

λ=.0064 Full(task8):
    /ceph_data/clue9986/tokencomp/upstream/globalsplat/outputs/nfcgs_nonlinear_score_context/20260910_092000/m1c1s1/rank56/lambda0p0064/residual_on/checkpoints/nfcgs_nonlinear_score_context_joint_rank56_lambda0p0064_residual_on_morton_on_m1c1s1/version_0/step000050000.ckpt

λ=.0256 Spatial(task7):
    /ceph_data/clue9986/tokencomp/upstream/globalsplat/outputs/nfcgs_nonlinear_score_context/20260910_092000/m1c0s1/rank56/lambda0p0256/residual_on/checkpoints/nfcgs_nonlinear_score_context_joint_rank56_lambda0p0256_residual_on_morton_on_m1c0s1/version_0/step000050000.ckpt

## 다음 Spatial predictor 개선 설계 — 로컬 구현 완료, 서버 미제출

| Arm | 변경안 | 목적 |
|---|---|---|
| P0 | 현재 predictor 유지 + 동일 추가 학습 | 추가 학습 대조군 |
| P1 | 기존 predictor + Conv3→GELU→Conv1×1 보정 | 같은 범위에서 비선형 예측 |
| P2 | 기존 predictor + Conv7→GELU→Conv1×1 보정 | 넓은 Morton 문맥 추가 |

- P1/P2는 `score_spatial_predictor=residual3/residual7`, P0는 기본값 `linear`로 구현됐다.
  보정 hidden은 `score_spatial_hidden=32`다.
- 작은 hidden(예:32)을 사용하고 보정 branch 마지막 Conv를0으로 초기화한다.
  현재 공간 예측기의 weight/key를 보존하고 초기 출력을 부모와 같게 만든다.
- 입력은 기존 anchor_delta(복호된 even minus base)다.
  원래 입력 feature, 미복호 odd score, encoder만 아는 원래 center를 사용하지 않는다.
- Conv3은 odd 기준 좌우±1 anchor, Conv7은±1/±3 anchor를 본다.
  P1을 Conv3 두 개로 만들면 RF5가 되어 같은 범위 비교가 아니다.
- P2는 weight 수도 늘어나므로 개선을 순수한 문맥 범위만의 효과로 단정하지 않는다.
- even/odd 두 단계와 Full의 channel 순서를 유지한다. 추가 직렬 복호 단계는 없다.
- 두 λ 각각 동일한 선택 parent에서 P0/P1/P2를 시작한다.
  모든 arm에 추가25k와 동일한 새 optimizer/scheduler, batch/seed/freeze 정책을 적용한다.
- **총6개 신규 training + 6개 dependent eval**이다.
  기존 parent eval은 추가25k를 받지 않았으므로 P0 대조군 대신 쓸 수 없다.
- 새 run/output prefix를 써서 기존 완료 결과와 충돌하지 않게 한다.
- 중요 구현 범위:
  config/default, checkpoint infer/validation, initializer warm-start,
  forward/compress/decompress 전체, metadata, output naming을 함께 연결한다.
  old checkpoint를 기본 P0로 strict-load할 수 있고 variant 불일치를 감지해야 한다.
  entropy CDF buffer와 group별 EntropyBottleneck warm-start 동작을 보존한다.
- 의미 있는 검증:
  P1/P2 zero-init 출력이 부모의 eval reconstruction과 일치하는지,
  미복호 odd 정보가 predictor에 새지 않는지,
  홀수/짝수 token 길이의 실제 bitstream roundtrip,
  보정 branch gradient, checkpoint save/load와 old defaults,
  Slurm plan/dry-run의 parent/task/node/batch/step 매핑을 확인한다.
- 결과는 actual bytes, PSNR/SSIM/LPIPS, 수신측 시간으로 비교한다.
  접전이면 비교하는 두 arm 모두 같은 추가 seed로 반복한다.

## Slurm 파일과 이미 겪은 운영 문제

기존 파일군:
- scripts/slurm/nfcgs_score_context_grid.sh
- scripts/slurm/train_nfcgs_score_context.slurm
- scripts/slurm/eval_nfcgs_score_context.slurm
- scripts/slurm/submit_nfcgs_score_context.sh
- scripts/slurm/nfcgs_factorized_continuation_grid.sh
- scripts/slurm/train_nfcgs_factorized_continuation.slurm
- scripts/slurm/eval_nfcgs_factorized_continuation.slurm
- scripts/slurm/submit_nfcgs_factorized_continuation.sh
- scripts/slurm/nfcgs_nonlinear_score_context_grid.sh
- scripts/slurm/train_nfcgs_nonlinear_score_context.slurm
- scripts/slurm/eval_nfcgs_nonlinear_score_context.slurm
- scripts/slurm/submit_nfcgs_nonlinear_score_context.sh
- 공통 training: scripts/slurm/train_nfcgs_paper_recipe.slurm
- 공통 eval: scripts/slurm/eval_nfcgs_rank56.slurm

Submitter는 training과 aftercorr dependent eval을 함께 제출한다.
기존 nonlinear 기본 TASKS=0-1,10-11은 이미 완료한 대조군이다.
TASKS=6-9도 이제 완료했으므로 재실험 요청이 없으면 다시 제출하지 않는다.
task 번호는 grid 간 다르므로 Linear task6과 Nonlinear task6을 혼동하지 않는다.

**현재 노드 설정:** 9월11일 인수인계 시점 로컬 nonlinear train/eval의 SBATCH 헤더는
ariel-v10이다. 이전 설계 문서의 v11 기술은 당시 스냅샷이다.
Submitter는 명령줄 --nodelist=ariel-v12 --gres=gpu:1로 둘 다 override한다.
echo 로그의 NODE=ariel-v12만 보고 실제 배정을 단정하지 않는다.
사용자 수정일 수 있으므로 기존 헤더를 이번 handoff 작업에서 덮어쓰지 않았다.
새 script는 이전 명시 요청대로 v12/gpu1을 반영하고, 다음 세션의 최신 사용자 지시를 따른다.

- high_perf 요구 에러는 ariel-v12를 명시하고 gpu:1을 요청하는 방식으로 대응했다.
- CRLF 때문에 set: pipefail invalid option 문제가 있었다.
  .sh/.slurm은 LF, UTF-8 BOM 없이 저장한다. .gitattributes는 둘 다 text eol=lf다.
  PowerShell Out-File 기본 인코딩/CRLF와 비ASCII 오염에 주의한다.
- common train에는 DUMMY_GPU_LOAD default=true 코드가 있지만 최근 follow-up launcher는
  false로 override한다. 후속 실험에서도 불필요한 dummy load를 켜지 않는다.
- eval 중복 제출을 피하려면 기존 dependent eval queue를 확인한다.
  GPU를 사용 중인 job 목록만 보고 pending eval이 없다고 판단하지 않는다.
- array master ID와 개별 task job ID를 구분한다.
  결과 path의 job ID는 array master ID와 다를 수 있다.
- 이 세션에서 서버로 원격 접속하거나 직접 sbatch를 실행하지 않았다.
  실행 명령과 파일을 제공하고 사용자가 서버에서 제출한 로그를 받아 확인했다.

## 다음 세션에서 바로 할 일

1. 이 문서와 최신 결과 보고서를 읽고 git status/diff로 현재 작업 상태를 확인한다.
2. P0/P1/P2 구현과 Slurm6조건은 로컬에서 완료됐다.
   [실험 문서](NFCGS_SPATIAL_PREDICTOR_EXPERIMENT_2026-09-11.md)를 읽고, 서버 제출 전
   최신 로그/queue를 확인해 중복을 방지한다.
3. 출발 checkpoint는 위의 완료된 context 모델들이다. 이전 nonlinear context 실험이
   사용했던 original50k warm-start 경로와 혼동하지 않는다.
4. 다음 실험 추천만 필요하면 더 기다릴 이유는 없다.
   현재 handoff 요청 자체는 새 학습 실행 요청이 아니다.
5. 추가 분석에는 서버 per-scene JSON으로 scene/frame 정합성, paired quality/rate,
   validation 복호 시간을 확인한다. 학습/validation/test의 역할을 구분한다.
6. Base-conditioned residual2×2, 추가λ RD curve, rank 확대는 현재 새 score predictor보다
   후순위다. Eval에서 λ만 바꾸는 것으로 새 RD point가 생기지는 않는다.
   새로운 λ 조건은 학습이 필요하다.

## 문서 변경 추적

- NFCGS_EXPERIMENT_RESULTS_2026-09-10.md: 과거 스냅샷과 최신 보고서 링크
- NFCGS_NONLINEAR_CONTEXT_RESULTS_2026-09-11.md: 최신4개 결과
- NFCGS_NEXT_EXPERIMENT_2026-09-10.md: 당시 계획, 일부는 이제 완료
- SESSION_HANDOFF_2026-09-11.md: 이 인수인계 문서

원래 handoff 턴에서는 인수인계 문서만 작성했다. 후속 작업에서 predictor 코드,
검증, 학습/평가 script를 로컬에 추가했으며 서버 job, git branch/remote는 변경하지 않았다.

# 다음 세션 인수인계 — 2026년 10월 29일

## 0. 먼저 읽을 내용

현재 작업은 **NFC-GS fresh-init RD 실험(§13–15 추가 15개 포함)의 진행 관리와 판정**이다. 기준 문서는 [NFCGS_FRESH_RD_PLAN_2026-10-20.md](NFCGS_FRESH_RD_PLAN_2026-10-20.md)다. 이 문서의 §13–15와 그 뒤 기록(427–451행)이 최신 결정이다.

상태 근거는 **10/29 사용자가 붙여 준 `squeue -u clue9986`**다. 실시간 상태가 아니다.

일정:

| 날짜 | 항목 |
|---|---|
| 10/30 13:00 | EXT 6개 40k 학습 곡선 확인(§3 명령) |
| **11/1** | 새 구조·설정 동결(사용자 일정) |
| 11/1–2 | Wave 2 추가 4개와 monitor 464634 종료 예상 → main 저장소 해시 잠금 해제 |
| 11/3 정오 | TK-H·TK-imp·TK-NN 종료 목표. **TK가 10/29에 아직 대기 중이라 지킬 수 없다**(120k × 약 3.8 s/step ≈ 5.3일) |
| ~11/6 | 최종 집계 |
| **11/10** | 제출 |

## 1. 사용자 작업 방식

- 응답 형식: `/i-have-adhd` 모드가 켜져 있다. 사용자가 "stop adhd mode"라고 할 때까지 유지한다. 첫 줄에 답이나 다음 행동을 두고, 단계는 번호로 매기고, 목록은 5개 이하로 쓴다. 시간 추정은 구체적으로 쓰고 서두·요약·맺음말은 쓰지 않는다. 매 답변에서 상태를 다시 밝히고, 다음 행동 하나로 끝낸다.
- 로컬 파일 읽기·수정·PowerShell·분석 선택은 묻지 않고 한다("이런거 나한테 물어보지마 그냥 해").
- 서버 명령은 사용자가 실행한다. job 제출·취소, git push, checkpoint·결과 삭제는 먼저 확인받는다. commit은 요청할 때만 한다.
- 로컬 repo: `C:\Users\a0103\Desktop\school\vml\global\upstream\globalsplat`.
- 서버 main: `/ceph_data/clue9986/tokencomp/upstream/globalsplat`.
- 서버 EXT 사본: `/ceph_data/clue9986/tokencomp_ext/upstream/globalsplat`. outputs와 checkpoints는 main으로 가는 symlink다.
- 서버 환경: `conda activate globalsplat`. VS Code `uploadOnSave`는 false다.

## 2. 잠금 규칙 (어기면 test 직전 code hash 검사에서 run이 실패한다)

1. **main 저장소의 runtime 파일**([run_nfcgs_fresh_rd.py](../scripts/run_nfcgs_fresh_rd.py) `RUNTIME_FILES`)은 Wave 2와 monitor 464634가 끝날 때까지(약 11/2) 서버에 올리라고 하지 않는다.
2. **EXT 사본**은 자기 run(468866–71, 469899·469900·469902, 470017–21)이 도는 동안 runtime 파일을 바꾸지 않는다. runtime 수정이 필요하면 새 사본(`tokencomp_ext2`)을 만든다.
3. 분석 스크립트(예: [analyze_nfcgs_fixed_order.py](../scripts/analyze_nfcgs_fixed_order.py))는 runtime 목록 밖이라 잠금과 무관하다.

## 3. Job 상태 (10/29 squeue)

### 실행 중 8개

| Job | Cell | λ | 저장소 | 경과 |
|---|---|---|---|---|
| 468866 | X-ON `nx_on` (NN-xyz) | .0128 | ext | 11.8h |
| 468867 | S56 H-OFF `s56_off` | .0128 | ext | 11.5h |
| 468868 | S56 N-ON `s56_n_on` | .0128 | ext | 11.2h |
| 468869 | S56 N-OFF `s56_n_off` | .0128 | ext | 9.6h |
| 468870 | MSH-N `msh_n` | .0032 | ext | 7.0h |
| 468871 | S56 `s56` | .0512 | ext | 6.7h |
| 464614 | MSH | .0016 | main | 6.0h |
| 464615 | MSH | .0064 | main | 5.6h |

### 대기 (위에서부터 대기열 순서)

| Job | Cell | λ | 저장소 | 비고 |
|---|---|---|---|---|
| 469899 | decoder finetune(H-ON λ0 120k에서 20k) | 0 | ext | §13.4 |
| 469900 | 대조군(codec만 학습) | 0 | ext | §13.4 |
| 469902 | S-ON `ns1_on` (NN-score, L1 graph_gemm) | .0128 | ext | §13.3 |
| 470017 | `tk_h` | .0032 | ext | §14 |
| 470018 | `tk_h` | .0128 | ext | |
| 470019 | `tk_h` | .0512 | ext | |
| 470020 | `tk_imp` | .0128 | ext | smoke에 없던 cell |
| 470021 | `tk_ns1` (TK-NN) | .0128 | ext | |
| 464616 | M-ON (Morton) | .0128 | main | **nice=100 적용됨** |
| 464617 | H-L3 | .0128 | main | **nice=100 적용됨** |
| 464634 | monitor | — | main | Wave 2 20개 afterok 대기 |

- 사용자가 10/29에 464616·464617에 nice=100을 걸어 TK를 Wave 2 추가 2개보다 앞으로 당겼다. 되돌리려면 `scontrol update jobid=464616 nice=0`(464617도 같음).
- 전체가 Priority로 대기하는 이유는 클러스터 fairshare다. 최근 GPU 사용량이 많아 우선순위가 낮다. nice는 우리 job끼리의 순서만 바꾼다.

## 4. 다음에 할 일 (시간순)

1. **TK job이 시작되면 10분 안에** 470020(`tk_imp`) 로그를 확인한다. smoke(469898)는 `tk_h`·`tk_ns`·`tk_ns1`만 돌렸고 `tk_imp`는 돌리지 않았다.
   ```bash
   cd /ceph_data/clue9986/tokencomp_ext/upstream/globalsplat
   ls logs/slurm/ | grep -E '47001[7-9]|47002[01]'
   tail -n 40 logs/slurm/*470020*.err
   ```
2. **10/30 13:00 40k 확인**: 사용자가 아래를 실행하고 출력을 붙이면 판정한다. 기준은 계획 문서 440행(같은 λ 정상 run 대비 train PSNR이 0.5dB 넘게 낮고, 25k 대비 Score 비트가 줄면 seed 추가).
   ```bash
   cd /ceph_data/clue9986/tokencomp_ext/upstream/globalsplat
   conda activate globalsplat
   B=outputs/nfcgs_fresh_rd/branches/B120000
   python scripts/check_nfcgs_train_curve.py --runs $B/nx_on/*/s0/job_* $B/ns1_on/*/s0/job_* --references $B/h_on/lambda0p0128/s*/job_*
   python scripts/check_nfcgs_train_curve.py --runs $B/s56_off/*/s0/job_* $B/s56_n_on/*/s0/job_* $B/s56_n_off/*/s0/job_* --references $B/s56/lambda0p0128/s0/job_*
   python scripts/check_nfcgs_train_curve.py --runs $B/msh_n/*/s0/job_* --references $B/msh/lambda0p0032/s0/job_*
   ```
3. **TK 10k 확인**(시작 후 약 11시간): K와 인코딩당 bit를 읽는다. `estimated_bits`는 인코딩 4개의 합이므로 4로 나눈다(§5).
   ```bash
   cd /ceph_data/clue9986/tokencomp_ext/upstream/globalsplat
   python - <<'EOF'
   import glob
   from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as EA
   B = "outputs/nfcgs_fresh_rd/branches/B120000"
   tags = ("train_codec_selected_tokens", "train_codec_estimated_bits", "train_psnr")
   for run in sorted(glob.glob(f"{B}/tk_*/*/s0/job_*")):
       vals = {t: {} for t in tags}
       for f in sorted(glob.glob(f"{run}/**/events.out.tfevents*", recursive=True)):
           ea = EA(f, size_guidance={"scalars": 0}); ea.Reload()
           for t in tags:
               if t in ea.Tags()["scalars"]:
                   vals[t].update({e.step: e.value for e in ea.Scalars(t)})
       k = vals[tags[0]]
       if not k: print(run, "no data"); continue
       last = max(k)
       mean = lambda t: (lambda xs: sum(xs) / len(xs) if xs else float("nan"))(
           [v for s, v in vals[t].items() if s > last - 500])
       K, bits = mean(tags[0]), mean(tags[1]) / 4
       print(f"{run} step {last} K {K:.0f} bits/enc {bits:.3g} bits/token {bits / K:.0f} psnr {mean(tags[2]):.2f}")
   EOF
   ```
   판정(계획 문서 433행):
   - .0128과 .0512가 하한 256 근처에 있는 것은 실패가 아니다.
   - **세 λ의 K가 모두 256–300에 붙어 λ에 반응하지 않으면** router가 갇힌 것으로 보고 취소를 제안한다. 대안은 `tokencomp_ext2`에서 router warmup 판을 만드는 것이다.
   - .0032의 K가 .0128보다 뚜렷이 높으면 계속한다.
4. **TK-fixed 후보 문제**: 미리 정의한 후보 `tk_f2048`…`tk_f3840`은 TK-H .0128의 30k 평균 K를 2048–3840으로 가정했다. K가 약 270에 머물면 맞는 후보가 없다. 그 경우 새 사본에서 cell을 새로 정의해야 한다. 30k 시점에 결정한다.
5. **11/2 이후**:
   - C3G 평가 스크립트를 runtime과 분리해 만든다(`dataset=re10k_eval_ctx{12,24,36}`, §5 평가 프로토콜 참고).
   - T06·T15를 H-ON .0128로 다시 맞춘다.
   - complexity를 다시 잰다(N-ON·X-ON·S-ON·TK 포함, §13.5).
   - scorer 점수와 Gaussian opacity의 상관을 본다.
   - 이것들을 ~11/6 최종 집계로 이어 간다.

### 사용자 결정을 기다리는 것

- **T06에 정렬 후보 추가(10/29 제안, 답 없음)**: [analyze_nfcgs_fixed_order.py](../scripts/analyze_nfcgs_fixed_order.py) `ORDERS`에 다음 후보를 더한다. 로컬 구현 2–3시간, GPU 1–2시간이다.
  - nn_score, nn_xyz
  - 혼합 거리 `α·d_score + (1−α)·d_xyz`(α 0.25·0.5·0.75)
  - score PCA-3 위의 Hilbert
  
  이 스크립트는 순서별로 `score_raw`·`msh_y`를 따로 내므로 MSH와 Context 경로 사이의 비트 배분을 볼 수 있다. Hilbert로 학습된 고정 모델이라 Hilbert 쪽으로 치우친다. Hilbert와 직접 비교하지 말고 다른 후보끼리만 비교한다. 사용자가 "ㄱ"이라고 하면 구현한다.
- **NN으로 정렬을 학습하는 아이디어(사용자 제안)**: 11/1 동결 이후라 이번 논문에서는 제외했다. 핵심 근거는 아래와 같고, 다음 논문 후보로 남긴다.
  - 정렬은 encoder 전용이라 비트 비용이 0이다.
  - D4는 순서상 ±4/±2/±1 anchor로 예측한다. 그래서 바로 옆 쌍만 최적화하는 greedy NN보다 score 공간에서 여러 척도로 근접성을 지키는 순서가 유리하다.
  - hard sort라 미분이 안 된다(SoftSort는 4096² 크기). 순서와 ctx 모델이 서로 맞춰 학습되는 닭과 달걀 문제도 있다.
  - MSH와 ctx 사이 balance는 S-ON(α=1)과 X-ON(α=0)의 경로별 비트로 먼저 판단한다. S-ON이 score 비트는 줄이고 msh_y 비트는 늘리는 경우에만 중간 α가 의미 있다.
- **선택(답 없음)**: H-ON .0128의 step 300 이전 train PSNR(smoke 비교용). 명령은 이전 세션 답변에 있다. TK 판정에는 필수가 아니다.

## 5. 이번 세션에서 확인·정정한 사실

### rate loss 정규화 (정정)

- `rate_loss = λ · estimated_bits / (4 × 32,768 × 59)` ([model_wrapper.py:388](../globalsplat/model/model_wrapper.py#L388), 547–548행).
  - 4는 인코딩 수(micro batch 2 × 13-view branch 2, `preds_2b`)다.
  - `estimated_bits`는 이 4개의 합이다(Top-K는 [globalsplat.py:319-320](../globalsplat/model/globalsplat.py#L319-L320)의 `rate.sum()`).
  - Gaussian은 32,768개(token 4096 × 8)이고, 속성은 59개다.
  - smoke 로그로 확인했다: bits ÷ bpga = 7,733,248.
- Top-K에서도 분모는 K와 무관하다. 32,768개 전부를 gate로 렌더하기 때문이다. TokenCodecGS 원본도 장면당 평균 bit × λ이고 token 수로 나누지 않는다(`TokenCodecGS/tokencodecgs/codecs.py` `rate_loss`).
- **인코딩당 bit 계수**: .0032 1.66e-9, .0128 6.6e-9, .0512 2.6e-8. 로그의 `rate_loss/estimated_bits`(.0128에서 1.66e-9)는 4개 합 기준이다. 처음에는 이 값을 장면당 계수로 잘못 써서 원본 대비 K 환산이 4배 어긋났다. 계획 문서 431·433행은 수정했다.
- 원본 발표 기준 정상 K(수정값):

  | λ | 정상 K |
  |---|---|
  | .0032 | 약 390–540 |
  | .0128 | 약 125–145 |
  | .0512 | 90 미만 |

  우리 token이 원본보다 싸다는 점은 K를 키우고, 원본 L1에 비해 MSE loss 크기가 작다는 점은 K를 줄인다. 그래서 대략값이다.
- smoke step 300의 token당 bit는 약 380이다(이전에 쓴 1550은 4개 합 기준). H-ON .0128 holdout의 101 bit/token(50.58KiB/scene)보다 약 4배 많다.

### Top-K smoke 469898 (tk_h, .0128, 300 step)

| step | K | train PSNR |
|---|---|---|
| 0–24 | 3904 | 10.2 |
| 125–149 | 720 | 11.8 |
| 275–299 | 283 | 11.9 |

K가 내려가도 PSNR과 render loss(0.10–0.11)는 거의 변하지 않았다. codec이 덜 학습된 시점이라 router에는 rate 항만 작용한다. 원시 dump는 세션 scratchpad에 있었다(임시 경로, 사라질 수 있다). 핵심 수치는 계획 문서 431·433행에 있다.

### 평가·학습 프로토콜 출처 (질문에 답한 내용)

- 우리 평가 `re10k_eval_all_ctx12`는 직접 만든 C12/T8이다. 카메라 위치 farthest-point로 context 12개를 고르고, 남은 frame에서 다시 farthest-point로 target 8개를 고른다. 대상은 6,991장이고 commit cc707c2(9/8)에서 추가됐다. 12/8은 ZPressor/MVSplat의 학습 설정(C12/T8)에서 온 것으로 보인다.
- GlobalSplat 원본 평가는 C3G 고정 index다(`assets/eval_index/re10k/c3g_re10k_ctx_{12,24,36}v_trg_3v.json`, 각 5,601장, T=3). 11/2 이후 별도 스크립트로 대표 checkpoint와 codec이 없는 GlobalSplat 기준을 함께 평가할 수 있다.
- 우리 학습은 GlobalSplat 쪽이다(24-view pool, 13-view branch 2개, bounded_v2, subset consistency). 원본과 다른 점:

  | 항목 | 우리 | GlobalSplat 원본 |
  |---|---|---|
  | 학습 대상 | codec만(backbone 고정) | 전체 |
  | Optimizer | Adam 1e-4, milestone 84k/108k | AdamW 5e-4, warmup 뒤 cosine |
  | Batch | 2×4 | 4 |
  | Curriculum | 없음 | 있음 |
  | Loss | MSE + 0.05 LPIPS | 2×(MSE + 0.5 VGG) |

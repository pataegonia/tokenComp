# 다음 세션 인수인계 — 2026년 10월 19일

## 0. 다음 세션에서 먼저 읽을 내용

현재 작업은 GlobalSplat scene-token 압축 논문의 실험 결과 수집·검증이다. **최근 계획한26개 학습은 모두 제출됐다. 새 학습을 추가하기보다 남은11개와 연결 평가를 수집하고, 비압축·경로 진단·독립 수신·복잡도 보완을 진행할 단계다.**

최신 확인은 **10월19일20:20경 다운로드한 로그**다. 26학습 중 **15개는50k 학습·전체 평가 완료, 8개는 진행 로그, 3개는 접수됐으나 시작 로그 없음**이다. 실시간 Slurm 상태가 아니므로 다음 세션에서 새 로그가 있으면 먼저 갱신한다. 시작 로그가 없는 job을 PENDING 또는 실패로 확정하지 않는다.

누적 actual full-test 결과는 **양의λ60점·16계열 + 별도λ0 한 점**, 모두6991 scene이다. 과거40점 또는46점 문서와 당시의 “미제출” 문구를 최신 상태로 사용하지 않는다. **기존26개를 중복 제출하거나 임의로 취소하지 않는다.**

사용자의 마지막 요청은 첨부 실험 검토문을 확인한 뒤 다음 세션용 인수인계를 정리하는 것이었다. 검토문의 수치와 판단은 확인했고, 아래 자료에 보관했다. 새로운 실험 제출을 요청받은 상태가 아니다.

## 1. 사용자 작업 방식과 경로 — 반드시 유지

- **ZIP으로 전달하지 않는다. 목록 파일 링크만 주는 것도 안 된다.** 서버에 반영할 변경이 있으면 필요한 **각 파일의 직접 클릭 가능한 하이퍼링크**와 서버에서 실행할 명령을 채팅에 준다.
- 사용자는 로컬 **`upstream/` 아래 파일만 서버에 반영**한다. 새 코드·문서·분석 산출물은 `upstream/globalsplat/` 안에 둔다. 다른 `artifacts/`, 작업 루트 `outputs/`, `Documents/Codex/...`에만 두지 않는다.
- 이미 만들어진 옛 ZIP은 최신 전달 경로가 아니다. ZIP 업로드·압축 해제 명령을 다시 안내하지 않는다.
- 로컬 workspace: `C:/Users/a0103/Desktop/school/vml/global`
- 로컬 repo: `C:/Users/a0103/Desktop/school/vml/global/upstream/globalsplat`
- 사용자가 다운로드한 로그: **workspace 루트의 `slurm/`**. repo 안의 `slurm/`가 아니다.
- 서버 repo: `/ceph_data/clue9986/tokencomp/upstream/globalsplat`
- 서버 Python 환경: `conda activate globalsplat`.
- 사용자가 서버 명령을 실행하고 로그를 다운로드한다. 이번 세션에서 에이전트가 원격 SSH·Slurm 제출·취소를 직접 수행한 적은 없다.
- 기존 미커밋 변경이 많다. 무관한 변경을 되돌리거나 `git reset`하지 않는다.

## 2. 가장 신뢰할 최신 자료

repo 기준 상대 경로다. 아래 자료를 먼저 읽으면 원본 대화 전체를 다시 복구할 필요가 없다.

| 자료 | 경로 |
|---|---|
| 최신 현황·모든 신규 완료 결과·남은 job | `docs/analysis/nfcgs_logs_2026-10-19_2020/REPORT.md` |
| 26학습 상태·설정·마지막 progress·149개 source 해시 | `docs/analysis/nfcgs_logs_2026-10-19_2020/review.json` |
| 양의λ60점 + λ0의 원본 수치·config·checkpoint·출처 | `docs/analysis/nfcgs_logs_2026-10-19_2020/inventory.json` |
| 같은 로컬 로그를 재집계하는 분석 | `docs/analysis/nfcgs_logs_2026-10-19_2020/audit_snapshot.py` |
| 첨부 검토문 재검산·표기 보완 | `docs/analysis/nfcgs_logs_2026-10-19_2020/review_crosscheck/CHECK.md` |
| 첨부 검토문 원본 | 같은 `review_crosscheck/provided_review.txt` |
| 재계산 비교 수치 | 같은 `review_crosscheck/derived_comparisons.json` |
| RD 그림 | 같은 `review_crosscheck/rd-snapshot-2026-10-19.png` 및 `.pdf` |
| 기계 판독용 실험 계획·최신 job ID/status | `config/plans/nfcgs_paper_experiments_20261018.json` |
| 설계 배경 | `docs/NFCGS_PAPER_EXPERIMENT_PLAN_2026-10-18.md` |

기존46점의 원본 로그 SHA-256과 최신 audit의149개 source 해시를 검증했다. 신규15평가에서 최종50k·6991 scene·actual bitstream·용량 합계를 검산했다. 이 검증이 장면별 ID/프레임 목록의 완전한 일치 확인이나 원본 tensor 재복호를 뜻하지는 않는다.

## 3. 최근26개와 job 수

| 묶음 | 학습 수 | 학습·평가 완료 | 학습 진행 로그 | 시작 로그 없음 |
|---|---:|---:|---:|---:|
| 양끝점·λ0 확장 | 13 | 13 | 0 | 0 |
| L3 대조군6개 + D4 OFF16 | 7 | 2 | 5 | 0 |
| 구조 T07/T08/T18/T09 | 4 | 0 | 3 | 1 |
| T11 추가 seed 쌍 | 2 | 0 | 0 | 2 |
| 합계 | **26** | **15** | **8** | **3** |

학습26 + 연결 평가26 + 공통 prepare1 = **53개 접수 job**이다. 26개를 전부 동시에 실행 중이라는 뜻은 아니다. 기본 구조3개는 prepare1+학습3+검증3=7개였고, T09 확장으로 학습1+검증1이 추가됐다. seed 쌍은 prepare 없이 학습2+평가2=4개다.

### 3.1 양끝점·λ0: 모두 완료

모든 행이50k 학습과6991장 실제 압축·복호·렌더 평가까지 완료됐다.

| 모델 | λ | Train | Eval | PSNR | KiB/scene |
|---|---:|---:|---:|---:|---:|
| Hilbert+D4 | 0 | 455327 | 455328 | 24.0734 | 831.64 |
| Hilbert+D4 | .0032 | 455329 | 455330 | 23.8965 | 56.60 |
| Morton+D4 | .0032 | 455331 | 455332 | 23.8564 | 57.31 |
| Hilbert+L3 | .0032 | 455333 | 455334 | 23.8682 | 58.50 |
| Morton+L3 | .0032 | 455335 | 455336 | 23.8505 | 59.05 |
| None+D4 | .0032 | 455337 | 455338 | 23.7153 | 67.20 |
| Hilbert+OFF1 | .0032 | 455339 | 455340 | 23.8639 | 64.42 |
| Hilbert+D4 | .0512 | 455341 | 455342 | 23.0675 | 21.91 |
| Morton+D4 | .0512 | 455343 | 455344 | 23.0479 | 22.40 |
| Hilbert+L3 | .0512 | 455928 | 455929 | 23.0360 | 22.79 |
| Morton+L3 | .0512 | 455930 | 455931 | 23.0172 | 23.29 |
| None+D4 | .0512 | 455932 | 455933 | 22.9360 | 29.50 |
| Hilbert+OFF1 | .0512 | 455934 | 455935 | 22.8594 | 25.56 |

위 여섯 모델 계열은 기존 중앙3λ와 합쳐 양의λ5점이 모두 확보됐다.

### 3.2 대조군7개

| 모델 | λ | Train / Eval | 20:20 로그 상태 |
|---|---:|---|---|
| None+L3 Context ON | .0032 | 456895 / 456896 | **학습·전체 평가 완료** |
| Hilbert+L3 OFF12 | .0032 | 456897 / 456898 | **학습·전체 평가 완료** |
| None+L3 OFF12 | .0032 | 456899 / 456900 | 46,657/50k, 93.3% |
| None+L3 Context ON | .0512 | 456901 / 456948 | 47,438/50k, 94.9% |
| Hilbert+L3 OFF12 | .0512 | 456949 / 456950 | 44,722/50k, 89.4% |
| None+L3 OFF12 | .0512 | 456951 / 457098 | 17,459/50k, 34.9% |
| Hilbert+D4 OFF16, T10 | .0128 | 457099 / 457100 | 8,184/50k, 16.4% |

완료한 두 점은 각각23.7288dB/67.65KiB와23.8513dB/64.55KiB다. 나머지 연결 평가 시작 로그는 아직 없다. **Eval ID를 train+1로 추정하면 안 된다.** QoS 재시도 때문에 번호가 떨어진 경우가 있다.

### 3.3 구조4개: 같은 holdout128 평가

| 모델 | 경로 | Train / 검증 Eval | 상태 |
|---|---|---|---|
| T07 `score56_only` | rank56 + D4 Context, MSH 없음 | 457741 / 457742 | 2,765/50k, 5.5% |
| T08 `msh_only` | direct MSH, rank56 경로 없음 | 457743 / 457744 | 3,456/50k, 6.9% |
| T18 `two_path56` | rank56 + MSH residual | 457745 / 457746 | 1,279/50k, 2.6% |
| T09 `score736_only` | learned736→736 + D4 Context, MSH 없음 | 457828 / 457829 | 접수·시작 로그 없음 |

공통 prepare **457740 완료**. train에서128장을 고정했고 후보11개를 건너뛰었다. 네 모델 초기화를 모두 만들고 `PREPARATION_COMPLETE`를 출력했다. T07/T08/T18은 **실제 CUDA의4096-token BF16 압축·별도 프로세스 복호·backward smoke를 통과하고 학습에 진입**했다. T09는 초기화가 만들어졌고 job 접수는 확인했지만 CUDA 실행 로그는 아직 없다.

네 모델은 Hilbert·λ=.0128·새50k·GPU2×batch2×누적2·BF16·model111123/data403이다. 같은 공개 GlobalSplat과 QR 초기화에서 시작하고 codec을 새로 초기화했다. 호환 tensor는 같은 초기값을 복사·해시 검사한다. backbone/GaussianDecoder는 고정하고 공통 geometry projection은 학습한다.

**구조4개는 train에서 제외한 같은128장 검증 평가이며, 아직 전체 test job을 제출하지 않았다.** 기본 val이 test split을 읽는 문제를 피하려고 이 학습에서는 val을 끄고 별도 train-derived 검증을 사용한다. 아직 검증 결과가 없으므로 구조 우열이나 다음λ를 확정하지 않는다. 같은λ가 구조마다 같은 실제 bitrate를 보장하지 않는다.

T09는 채널 그룹184×4로 D4의16 Score strings를 유지한다. rank56의 slice16을 그대로 써서184 strings를 만드는 설정이 아니다. symbol 수와 predictor 용량은 rank56보다 크므로 동일 용량의 순수 차원 비교로 해석하지 않는다.

### 3.4 T11 추가 seed: 기존 대조군과 같은 전체 train/test

| 모델 | Model / data seed | Train / 전체 Test Eval | 상태 |
|---|---|---|---|
| Hilbert+D4 ON16 | 111124 / 404 | 457786 / 457787 | 접수·시작 로그 없음 |
| Hilbert+D4 OFF16 | 111124 / 404 | 457788 / 457789 | 접수·시작 로그 없음 |

T11은 같은 기존 FullSplit 부모 가중치에서 시작하는 후속50k의 seed 반복이다. 구조 pilot처럼 codec을 새로 초기화하는 실험이 아니다. 기존 seed111123/data403의 H+D4 ON16과 T10 OFF16에 대응한다. 전체 train split을 유지하고 최종50k를 현재 C12/T8 전체6991장으로 평가한다. **구조 pilot의 holdout128 프로토콜과 섞지 않는다.**

## 4. 현재 모델·용어·대조군 정의

- 4096 scene tokens. 공통 압축 대상 `F736 = appearance512 + observable geometry224`다.
- 공통 geometry projection은512→224다. rank56 경로를 제거한 direct MSH도 이 projection은 유지한다.
- 현행2-path는 rank56의 low-rank/Score 복원 + MSH residual이다. MSH는 N192/M320/adapter96이다.
- minimal은 centering·Score norm·mean/channel Context를 끄고 linear transform을 사용한다. 공간 Context는 별도 스위치다.
- **Context OFF는 Score 공간 예측만 OFF이며 MSH는 유지**한다. MSH 없는 T07/T09와 다르다.
- JSON family **`context_only`는 기본 token 순서+L3 Context ON인2-path**이고 `use_residual=true`다. T07의 단일 경로 Context-only 결과로 읽지 않는다.
- 사용자는 과거 다른 조건에서 direct MSH·low-rank 없는2-path·low-rank를 비교했을 때 단순2-path가 좋았다고 설명했다. 현재 T07/T08/T18/T09는 위의 네 경로 비교이며, 별도의 “low-rank 없는2-path” 모델을 추가로 제출한 것은 아니다.

| 명칭 | token 단계 | 채널 그룹 | Score strings | 공간 예측 |
|---|---:|---|---:|---|
| OFF1 | 전체4096 한 단계 | 전체56 | 1 | OFF |
| OFF4 | D4의4단계 | 단계마다 전체56 | 4 | OFF |
| OFF12 | L3의3단계 | 16/16/16/8 | 12 | OFF |
| OFF16 | D4의4단계 | 16/16/16/8 | 16 | OFF |
| L3 ON | 3단계 | 16/16/16/8 | 12 | ON |
| D4 ON | 4단계 | 16/16/16/8 | 16 | ON |

L3는0 mod4 → 2 mod4 → 홀수, 실제 anchor 간격±2/±1이다. D4는0 mod8 → 4 mod8 → 2 mod4 → 홀수, 단계별 token 수512/512/1024/2048, 실제 anchor 간격±4/±2/±1이다. 오른쪽 anchor가 없으면 왼쪽을 복제한다. 분석에는 앞 단계에서 실제 양자화·복호된 anchor를 사용한다.

정렬은 **Score와 MSH 양쪽에 적용**된다. 전체 용량 차이를 전부 Score Context의 효과로 돌리지 않는다.

## 5. 검증한 연구 판단과 한계

### 5.1 현행 후보

**Hilbert+D4를 최종 RD 후보**, L3 matched2×2를 정렬·공간 Context의 주 분석 기준으로 유지한다.

- H+D4 대 None+D4는 다섯λ 모두 bytes15.78~25.73% 감소, PSNR0.1183~0.2008dB 증가다. 공간 정렬 유무의 차이가 H/M 사이의 작은 차이보다 크다.
- H+D4 대 H+L3는 다섯λ 모두 bytes0.326~3.867% 감소다. PSNR은4점 증가, .0128에서만−0.0011dB다.
- D4의 H 대 M은 bytes1.24~2.19% 감소다. PSNR은4점 증가, .0128에서만−0.0002dB다.
- H+L3 .0032 ON 대 OFF12는 bytes9.366% 감소·PSNR+0.0169dB다. 중앙3λ의14.6~15.4% 절감을 전λ의 공통 효과로 쓰지 않는다.
- Context가 모든λ에서 PSNR까지 개선한다고 일반화하지 않는다. H+L3 .0064는 bytes−14.91%이나 PSNR−0.0136dB다.
- 기존 OFF1→OFF4는 중앙3λ에서 일관된 RD 개선이 없었다. D4 공간 예측의 맞춘 대조군은 OFF16이다.

모두 같은λ의 학습 모델 비교다. 정렬만의 순수 인과효과나 동일 화질 BD-rate·통계적 유의성으로 확대하지 않는다. D4 ON/OFF 추가 seed가 L3 대 D4 스케줄 차이나 L3 정렬×Context 상호작용 전체의 seed 반복을 대신하지 않는다.

### 5.2 λ0와 비압축 기준

H+D4 λ0는24.0734dB·831.6422KiB, .0032는23.8965dB·56.5980KiB다. **PSNR+0.1769dB에 bytes14.69배**다. 증가분의72.35%는 MSH y 증가다. λ0에도 양자화·entropy coding이 있으므로 비압축 또는 구조적 화질 상한이 아니다.

역사적 non-comp24.7004dB는 과거 ablation 표의 `upper` 참고값이다. λ0와의0.6270dB 차이를 현재 codec의 확정 손실이나 전부 양자화 손실로 쓰지 않는다. 현재와 같은 장면·카메라·정밀도 및 B0/B1 경계 확인이 필요하다.

### 5.3 용량 분해의 헤더

`actual_score_bytes`에는 Score 내부 wrapper가 포함되고, `actual_container_bytes`에는 그 wrapper가 없다. SCCTX001 wrapper는 **24+8K bytes**이며 K는 Score strings 수다.

- 순수 Score payload = `actual_score_bytes−(24+8K)`.
- 모든 헤더 = `actual_container_bytes+(24+8K)`.
- D4 K16이면 wrapper152B다. λ0의 Score 스트림240.818764KiB는 내부 헤더 포함이고 raw Score는 **240.670326KiB**다.
- Score + MSH y + MSH z + mean + container 합계의61점 최대 오차는 약1e−6B로 출력 반올림 범위다.

## 6. 다음 작업 순서

### 우선1: 최신 로그 수집과 미완료11개 평가

새 로그를 받으면 상태·footer·실제 checkpoint step·scene count부터 갱신한다. 20:20 시점 잔여 시간 추정은 현재 시각부터의 ETA가 아니다. 필요하면 서버에서:

```bash
squeue -j 456899,456900,456901,456948,456949,456950,456951,457098,457099,457100,457741,457742,457743,457744,457745,457746,457786,457787,457788,457789,457828,457829 -o '%.18i %.24j %.10T %.30R %E'
```

1. L3 대조군 나머지4개로2×2 네 곡선의 양의λ5점을 완성한다.
2. T10과 T11로 D4 ON16/OFF16 두 seed를 비교한다.
3. 구조4개는 같은128장 검증으로 수렴·실제 rate를 보고, 살아남은 구조에만 추가λ/전체 test를 결정한다.

### 우선2: T04 B0/B1 및 T05 경로·양자화 진단 구현/검증

**과거에 유사한 진단을 이미 했다. 이를 누락하고 “경로·양자화 실험을 한 적이 없다”고 설명하지 않는다.** `docs/ABLATION_PROGRESS_2026-09-02.md`의 완료 실험17은 `upper → concat_only → projection → projection_quantized → full`의 누적 단계별 평가다. rank40/56/72/80, λ=.0256,50k의 과거 Morton/factorized 계열을 사용했다. rank56은 upper24.7004, concat_only24.6376, 연속 projection20.5585, projection_quantized20.2568, full23.3700dB였다. 실험14의 독립 학습 구조 비교도 별도로 있었다.

과거 `concat_only`는 vanilla 초기화의 별도 identity checkpoint이고 현재 학습된 Hilbert+D4의 B1과 같지 않다. 과거 `projection_quantized−projection`에는 mean FP16 양자화도 포함돼 순수 Score 양자화 손실로 읽으면 안 된다. 과거 실험17은 residual 경로를 켜면서 그 분석·합성과 양자화를 함께 추가한 것이므로, MSH를 유지한 상태에서 Score Q/MSH y·z Q를 각각 우회하는 아래7경로와도 다르다.

**현재 Hilbert+D4 조건의 B1/7경로 완료 기록은 받은 로그·현재 진단 코드에서 확인하지 못했다.** 원격의 모든 실행 이력을 조회한 것은 아니다. 과거 구현·결과를 먼저 찾아 재사용하고, 현재 checkpoint에서 빠진 조건만 보완한다. 이미 완료된 anchor/잔차/bits 분석과 λ0 평가는 렌더 PSNR의 경로별 양자화 우회 평가를 대신하지 않는다.

- **B0:** 공개 원본 GlobalSplat 비압축 경로, native geometry decoder 입력512.
- **B1:** 현재 checkpoint에서 projection 이후 F736을 현행 geometry224 GaussianDecoder에 직접 전달해 LR/MSH/양자화를 우회.
- 현재 checkpoint에서 단순히 `feature_codec=null`로 바꾸면 decoder의 geometry 차원이 달라질 수 있어 B1이 되지 않는다.
- QR 초기화는 native readout을 보존하지만 geometry projection은 이후 학습되므로 B0→B1과 B1→codec 변화를 분리한다.

T05의7개 경로는 정렬된 연속 Score s, 실제 인과적 양자화·복호 Score ŝ, LR 복원 L, 연속 MSH 분석·합성 Hcont, 양자화 포함 Hhat으로 정의한다.

| 모드 | GaussianDecoder 입력 |
|---|---|
| B1 비압축 우회 | F |
| 연속 LR만 | L(s) |
| 복호 Score만, residual0 | L(ŝ) |
| Score 양자화 우회 | L(s)+Hhat(F−L(s)) |
| MSH y/z 양자화 우회 | L(ŝ)+Hcont(F−L(ŝ)) |
| 양쪽 양자화 우회 | L(s)+Hcont(F−L(s)) |
| 실제2-path packet | L(ŝ)+Hhat(F−L(ŝ)) |

Score를 바꾸면 MSH 입력 residual도 다시 계산한다. Hcont도 identity가 아니며 분석·합성 손실이 남는다. 연속 우회 모드에는 actual packet rate가 없으므로 RD 점으로 만들지 않는다. PSNR 차이를 더해 독립 경로 기여로 해석하지 않는다. 먼저 기존 같은256장, 필요한 핵심 모드만 전체6991장으로 확장한다.

### 우선3: T15 독립 수신과 T06 정렬 진단

- 기존 `scripts/eval_nfcgs_sender_receiver.py`가 있으므로 재사용 가능 여부·현행 checkpoint/packet 호환성을 먼저 확인한다. 이 스크립트가 있다는 이유만으로 최종 방법의 독립 Gaussian 복호·렌더 검증이 완료됐다고 하지 않는다.
- codec smoke는 packet+checkpoint만 받는 별도 프로세스에서 **feature 복호**를 확인했다. 전체 Gaussian/renderer 수신 검증과 다르다.
- GaussianDecoder는 token당 Gaussian8개다. 순열 검증은8개 묶음을 맞추며 실제 GPU 렌더 반복 오차도 측정한다. 과거 CPU 임의 입력의 순열 검사 통과를 GPU pixel bit-exact 검증으로 쓰지 않는다.
- 현재 H+D4 .0128의 같은 F/P/basis/Score/XYZ를 고정해 slot·Morton·Hilbert·random10 seeds의 실제 anchor·잔차·bits를 연결하는 T06 보완은 가능하다. 고정 predictor에 정렬만 바꾸는 결과는 분포 밖 진단이며 각 정렬 재학습의 RD와 다르다.

### 우선4: 복잡도와 최종 집계

- 기존37 checkpoint의 A5000 profile은 완료됐다. `docs/analysis/nfcgs_complexity_455251/REPORT.md`를 본다.
- .0128 H+L3 encode/decode150.583/84.279ms, H+D4 163.698/93.299ms, NN-score214.619/84.715ms다. D4는 L3보다8.71%/10.70% 느렸다. 이는 특정λ의 codec 구간이며 전체 RD 범위나 전체 시스템 시간으로 확대하지 않는다.
- 기존46점 중 빠진 Both OFF3점+OFF1/OFF4 6점=9점과 새 OFF16·주요 구조를 보완한다. 필요 없이 과거37개 전부 재측정하지 않는다.
- 새 할당에서는 H+L3/H+D4/NN-score .0128 reference도 함께 측정하고 같은 cache·32장·warmup4·반복3·threads8·BF16을 맞춘다. manifest/환경/GPU UUID가 바뀌면 기존 profile output에 resume해서 섞지 않는다.
- 전체 모델이 올라간 harness의 peak VRAM과 독립 receiver 최소 메모리를 구분한다.
- RD/BD는 공통 구간과 단조성·외삽 금지·scene/frame 정합을 확인한다. paired scene bootstrap CI는 scene 변동이며 seed 변동 CI가 아니다.

## 7. 이미 완료된 분석·프로토콜 주의

- L3 네 정렬의 실제 anchor·잔차·bits: job454995, 같은256장.
- Hilbert L3/quarter2/D4 anchor·잔차·단계별 bits: job455166, 같은256장.
- 복잡도37 checkpoint: job455251.
- 과거 OFF1/OFF4 중앙6점도 완료다. 이번26개에 다시 넣지 않는다.
- 현재 전체 test는 **C12/T8, 6991장, 256×256, eval seed0**다. 원본 GlobalSplat의 C3G target3 프로토콜과 다르다. 임의로 target view를 바꾸지 않는다.
- 외부 CodecSplat/FeCoSplat/MFGS 프로토콜 조사: `docs/NFCGS_EXTERNAL_EVAL_PROTOCOLS_2026-10-19.md`. FeCoSplat target 수를 원문에 명시된 사실처럼 단정하지 않는다. 현재 비교 조건으로의 별도 맞춤 평가 없이 논문 숫자끼리 직접 순위를 만들지 않는다.
- 사용자 제공 원래26학습 목록과 최신 구성은 다르다. T17 Morton+L3 OFF12 대신 T10 H+D4 OFF16, L3 추가 seed 대신 D4 ON/OFF16 쌍을 사용한다. Morton matched OFF나 L3 추가 seed를 제출된 것으로 세지 않는다.
- 사용자 계획의11월1일 구조/설정 동결·11월10일 제출을 참고한다. 새 구조를 늘리는 것보다 현재 결과와 논문 근거를 완성한다.

## 8. 운영·검사·구현 위치

### 제출 state와 launcher

| 묶음 | 서버 state | launcher |
|---|---|---|
| 양끝점 | `logs/slurm/order_context_extremes_20261017/` | `scripts/slurm/submit_nfcgs_order_context_extremes.sh` |
| 대조군7개 | `logs/slurm/paper_controls_20261018/` | `scripts/slurm/submit_nfcgs_paper_controls.sh` |
| 구조4개 | `logs/slurm/architecture_pilots_20261019/state.json` | `scripts/slurm/submit_nfcgs_architecture_pilots.sh` |
| seed 쌍 | `logs/slurm/context_seed_pair_20261019/state.json` | `scripts/slurm/submit_nfcgs_context_seed_pair.sh` |

대응하는 다운로드본은 로컬 루트 `slurm/` 아래에 있다. 접수 ID는 제출 직후 저장하므로 QoS 뒤 같은 launcher를 재실행하면 미접수분만 추가한다. 실패·취소 학습의 자동 resume 기능은 아니다. **현재는26개 모두 접수돼 재제출할 이유가 없다.**

구조 `--include-fullwidth`도 이미 실행돼 T09가 접수됐다. `--with-test`는 검증 결과를 보고 전체 test를 추가할 때 쓰는 옵션이며 아직 실행된 기록은 없다. 판단 없이 추가하지 않는다.

### 중요한 구현과 수정

- 구조 경로: `globalsplat/compression/architecture.py`, `config.py`, `codec.py`, `checkpoint.py`. 명시적 `two_path/score_only/msh_only`를 지원한다. 비활성 branch 모듈·payload가 실제로 없고 packet flag로 구분한다.
- 공통 초기화: `scripts/prepare_nfcgs_architecture_pilots.py`.
- 구조 실행/검사: `scripts/run_nfcgs_architecture_pilot.py`, `scripts/smoke_nfcgs_architecture_pilot.py`.
- holdout: `globalsplat/dataset/holdout.py`, `data_module.py`, `re10k.py`. train에서 image decode 전 제외하고 검증은 명시적 train chunk와 고정 C12/T8 index를 읽는다.
- seed 쌍: `scripts/run_nfcgs_context_seed_pair.py`, `scripts/submit_nfcgs_context_seed_pair.py`, Slurm3개. 전용 wrapper로 seed를 전달하며 기존 공통 runner는 바꾸지 않았다. 평가 전에 실제 train Hydra config·최종50k·6991장을 검사한다.
- `/data3/local_datasets/re10k`는 계산 노드용이다. 로그인 노드 `ariel-master`에서 train index가 없다고 submission을 막던 버그를 수정했다. 데이터 검사는 prepare/계산 job 안에서 한다. 임의로 공유 경로로 바꿀 필요가 없다.
- checkpoint 쓰기 문제 대응으로 Slurm에서 Ceph `TMPDIR=/ceph_data/<USER>/.tmp/nfcgs/job_<ID>`를 사용한다. 기존 수정은 유지한다.
- 구조 state의20개, seed state의17개 실행 파일 해시는 최신 로컬과 모두 일치했다. 제출 이후 학습 코드를 바꾸면 signature mismatch와 실행 중 job의 코드 혼용을 고려해야 한다.

### 로컬 환경과 검사

- PowerShell. 기본 Python3.13은 Hydra가 없을 수 있다. 설정/codec 검사는 `C:/Users/a0103/miniconda3/envs/nfcgs-local-profile/python.exe`를 사용한다. Torch CPU/Hydra/Lightning/CompressAI가 있다.
- Git Bash: `C:/Program Files/Git/bin/bash.exe`. Slurm/shell 파일 LF와 shebang을 유지한다.
- 구조 관련 기존/회귀60개, seed 쌍8개 검사를 통과했다. 마지막 구조 제출 검사4개에서 기존7개 후 T09만2개 추가·prepare/ID 재사용을 확인했다. 새 기능 변경이 없으면 무거운 검사 전체를 반복할 필요는 없다.
- T09도 로컬4096-token BF16 CPU 압축·독립 복호·backward를 통과했다. 실제 GPU 학습이나 원본 backbone 평가를 로컬에서 했다는 뜻은 아니다.
- PowerShell here-string을 Python stdin으로 보내면 한글 literal이 `?`로 깨질 수 있다. 한글 문서는 `apply_patch`, Python snippet은 ASCII를 쓴다.
- `rg`는 우선 사용하되 Windows에서 wildcard 경로 인자를 그대로 넘기지 말고 `-g`를 사용한다. training log는 carriage return으로 progress가 매우 길 수 있으므로 광범위 검색 결과를 통째로 출력하지 말고 Python `splitlines()`와 마지막 완전한 progress를 사용한다.

## 9. 다음 세션 첫 행동

1. 이 인수인계와 최신 `REPORT.md`, `review.json`, `inventory.json`을 읽는다.
2. 루트 `slurm/`에20:20 이후 새 파일·내용이 있는지 확인한다. 있으면 미완료11개와 연결 평가부터 갱신한다.
3. 새 로그가 없다면 기존 작업을 다시 제출하지 않고, 사용자 의도에 따라 B0/B1·T05·T15 중 다음 분석의 기존 코드와 구현 공백을 확인한다.
4. 변경이 생기면 `upstream/globalsplat/`에 만들고 **변경된 개별 파일 하이퍼링크와 서버 실행 명령을 채팅에 제공**한다.

**핵심: 새 학습을 더 늘리는 세션이 아니라, 이미 접수한 비교의 결과와 “왜 좋아지는가”의 근거를 완성하는 세션으로 이어간다.**

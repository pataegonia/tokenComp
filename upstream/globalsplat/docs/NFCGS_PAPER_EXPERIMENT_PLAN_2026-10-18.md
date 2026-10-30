# Scene-token 실험 설계안 — 2026-10-18

**현황 갱신: 10월19일20:20경 받은 로그에서 최근26학습이 모두 접수됐고, 15개는 학습·평가 완료, 8개는 학습 진행, 3개는 시작 로그가 없었다. 누적 actual 결과는 양의λ60점과 별도λ0 한 점이다. 아래9/13회는 원래 추가 학습 예산이며 현재 미접수 학습 수가 아니다. 최신 상태는 [10월19일 로그 정리](analysis/nfcgs_logs_2026-10-19_2020/REPORT.md)를 따른다.**

사용자가 제공한 2026-10-17 계획을 10월18일 로컬 로그와 코드에 대조한 설계안이다. 당시 이 문서 작성으로 새 학습을 제출하지 않았다. 아래 실행 순서와 변경 사항은 제안이며, 접수된 기존 작업은 유지한다. 대조군7개는 [제출 안내](NFCGS_PAPER_CONTROLS_2026-10-18.md)의 launcher로 준비했다.

## 1. 확인된 상태와 예산

| 구분 | 확인 결과 | 다음 조치 |
|---|---:|---|
| 기존 actual RD | 40점·14계열, 각 6991 scene 평가 완료 | 재학습 없이 재사용 |
| Hilbert OFF1/OFF4 | 6점 모두 actual 평가 완료 | 기존 40점에 합쳐 총 46점·16계열로 집계 |
| 양끝점·λ0 확장 | 13개 모두 train/eval 접수, 학습 완료 3개 | 진행 로그 8개, 시작 로그 없는 2개와 연결 eval 확인 |
| 새 기본안 | 9회 학습 | 아래 T16 6회 + T10 1회 + 구조 pilot 2회 |
| 새 권장안 | 13회 학습 | 기본 9회 + 구조 2회 + seed 쌍 2회 |

10월 18일 21:11경 다운로드 자료에서 기존 train 455327/329/331/333/335/337/339/341/343 외에 **λ=.0512의 Hilbert+L3 455928, Morton+L3 455930, None+D4 455932, Hilbert+OFF1 455934**와 각각 다음 번호의 eval 접수를 확인했다. 13개 모두 접수돼 중복 제출하지 않는다. 학습 완료 3개는 455327/455329/455339이며 연결 eval 결과 로그는 아직 없다. 세부 상태는 [최신 로그 검토](analysis/nfcgs_logs_2026-10-18_2111/REPORT.md)에 정리했다. 이는 실시간 큐 조회 결과가 아니다.

원문과 같은 계산으로 확장 계획 13회를 포함하면 기본 22회·권장 26회다. 이는 완료된 46점의 수가 아니며, 앞으로 반드시 22/26회를 새로 제출한다는 뜻도 아니다. 확장 13개는 모두 접수됐으므로 현재 설계상 새로 필요한 학습은 기본 9회·권장 13회다.

근거와 각 로그의 SHA-256은 [완료 결과 목록](analysis/nfcgs_paper_inventory_2026-10-18.json), 설정별 새 실행 목록은 [실험 계획 JSON](../config/plans/nfcgs_paper_experiments_20261018.json)에 저장했다. JSON은 제출용 manifest가 아니다.

## 2. 같은 학습 수 안에서 바꿀 항목

최종 후보인 Hilbert+dyadic4의 효과를 직접 설명하도록 세 항목을 조정한다.

1. 원문의 **T17 Morton+L3 OFF12 한 점**을 **T10 Hilbert+D4 OFF16 한 점**으로 대체한다. Morton의 보조 검증보다 최종 방법의 공간 예측 기여를 먼저 분리한다. T17은 선택 항목으로 남긴다.
2. 구조 비교 기준선 T18은 **Hilbert+L3 대신 Hilbert+D4**를 사용한다. T07/T09의 Context도 D4로 고정한다.
3. 추가 seed 쌍 T11은 **Hilbert+D4 ON16/OFF16**으로 맞춘다. 기존 seed와 새 seed에서 각각 ON/OFF 차이를 비교한다. L3의 추가 seed 쌍은 선택 항목이다.

| 구분 | ID | 새 학습 | λ | 수 | 답하려는 질문 |
|---|---|---|---|---:|---|
| 기본 | T16 | None+L3, Context ON | .0032/.0512 | 2 | 기본 순서에서도 Context가 주는 이득 |
| 기본 | T16 | Hilbert+L3, 공간 Context OFF12 | .0032/.0512 | 2 | 정렬 단독 효과 |
| 기본 | T16 | None+L3, 공간 Context OFF12 | .0032/.0512 | 2 | 정렬·공간 Context 모두 OFF인 기준 |
| 기본 | T10 | Hilbert+D4, 공간 Context OFF16 | .0128 | 1 | 최종 D4에서 공간 Context의 기여 |
| 기본 | T07 | Hilbert+rank56 transform+Context, MSH 없음 | .0128부터 검증 | 1 pilot | 저차원 경로 하나로 충분한가 |
| 기본 | T08 | Hilbert+direct MSH, rank56 경로 없음 | .0128부터 검증 | 1 pilot | MSH 경로 하나로 충분한가 |
| 권장 추가 | T09 | Hilbert+736→736 learned transform+Context, MSH 없음 | 검증 후 확정 | 1 pilot | rank56 병목을 제거하면 달라지는가 |
| 권장 추가 | T18 | Hilbert+D4, rank56+MSH, 새 구조와 공통 초기화 | 검증 후 확정 | 1 pilot | 같은 초기화·예산에서도 2-path가 유리한가 |
| 권장 추가 | T11 | Hilbert+D4 ON16/OFF16, 새 seed 한 쌍 | .0128 | 2 | 공간 Context 효과가 다른 seed에서도 유지되는가 |
| | | **기본 / 권장 총 추가 학습** | | **9 / 13** | |

T16의 Hilbert+L3 ON 양끝점 2개는 기존 확장 계획에 있으므로 중복 제출하지 않는다. 이미 완료된 중앙 3λ와 합치면 L3의 None/Hilbert×공간 Context OFF/ON 네 곡선에 모두 양의 λ 5점이 생긴다.

T18은 예산표에서는 권장 항목이지만, **구조 간 우열을 주장할 때는 필수 기준선**이다. 기본안의 T07/T08만 끝났다면 구조 선별 결과로 한정한다. 구조 우열을 논문에 넣을 목적이라면 T18을 T09·추가 seed보다 먼저 실행한다.

## 3. OFF 설정과 비교 범위

여기서 Context OFF는 Score의 공간 예측을 끈다는 뜻이다. MSH는 계속 유지하며, minimal 설정의 Score 채널 Context는 원래 OFF다.

| 명칭 | token 단계 | 채널 그룹 | 실제 Score entropy strings | 해석 |
|---|---:|---|---:|---|
| OFF1 | 전체 4096 token 한 단계 | 56 전체 | 1 | 현재 사용자가 지정한 한 덩어리 기준선 |
| OFF4 | dyadic4의 4단계 | 단계마다 56 전체 | 4 | token 분할만 바꾼 기준선 |
| OFF12 | legacy3의 3단계 | 16/16/16/8 | 12 | L3 ON과 단계·그룹을 맞춘 공간 예측 OFF |
| OFF16 | dyadic4의 4단계 | 16/16/16/8 | 16 | D4 ON과 단계·그룹을 맞춘 공간 예측 OFF |

OFF1/OFF4와 D4 ON의 차이에는 단계·채널 그룹·스트림 수 차이도 포함된다. **D4 ON/OFF16 비교**에서 이 조건들을 맞춘다. 이 비교도 예측과 예측 잔차용 확률 모델을 포함한 공간 Context 경로 전체의 효과이며, 예측기 한 모듈만의 기여로 좁혀 해석하지 않는다.

완료된 OFF1/OFF4의 세 λ에서 4분할이 일관되게 유리하지 않았다. OFF4 대비 추가 Context 이득은 아직 OFF16 평가 전에는 확정하지 않는다.

L3 2×2는 전체 시스템의 정렬·공간 Context 효과를 비교한다. 정렬은 Score와 MSH 모두에 작용하므로 전체 용량 차이를 Score Context의 변화만으로 설명하지 않는다. 두 요인의 결합 효과는 동일 화질의 log-rate 등 미리 정한 척도에서 평가하고, 같은 λ에서 PSNR과 용량이 함께 변한다는 점을 별도로 표시한다.

## 4. 기존 codec 대조군의 공통 조건

T16 여섯 개와 T10 한 개는 기존 코드로 구성할 수 있다. 각 train/eval 설정을 Hydra로 구성해 계획 JSON과 대조했다. OFF16은 실제 Context codec의 4096-token CPU 압축·복호와 16개 strings를 확인했다. 새 Slurm 제출이나 50k GPU 학습 검증을 한 것은 아니다.

- 부모: 기존 FullSplit λ=.0256 probability10k checkpoint의 가중치만 로드. optimizer는 새로 시작.
- rank56, MSH ON, minimal, linear predictor, kernel5, STE. 지정한 정렬·단계·공간 Context만 변경.
- 50,000 optimizer steps, GPU 2개×GPU당 batch2×누적2, BF16, LR 1e-4, 35k/45k milestone.
- 기존 model seed 111123, data loader seed 403 및 기존 데이터·view 선택 레시피 유지.
- 평가: actual encode→decode→render, test 6991 scene, C12/T8, batch1, seed0. 최종 50k checkpoint 사용.
- 추가 seed T11은 model seed 111124/data seed 404를 ON/OFF 쌍에 동일하게 사용. 10월 19일 전용 runner와 학습2개·전체 test 평가2개의 제출 명령을 준비했고, 실제 train config를 평가 전에 검사한다. [실행 안내](NFCGS_CONTEXT_SEED_PAIR_2026-10-19.md)를 따른다.

진행 중인 작업을 취소하거나 기존 중앙 λ 모델의 학습 레시피를 소급 변경하지 않는다. 기존 Morton+L3 .0256의 GPU4 예외는 결과표에 기록한다. 이를 통일하는 재학습은 핵심 결론이 그 차이에 민감할 때 추가한다.

## 5. 구조 비교: 네 경로의 입출력을 통일

공통 압축 대상은 `F=[appearance 512, observable geometry 224]`, 총 736채널이다. direct MSH의 ‘low-rank 없음’은 **rank56 경로가 없다**는 뜻이다. 공통 512→224 geometry projection까지 제거한다는 뜻은 아니다.

| ID | 압축·복원 경로 | MSH | Score Context |
|---|---|---|---|
| T07 | F736→Score56→복원 F736 | 없음 | Hilbert+D4, 16/16/16/8, 16 strings |
| T08 | F736→MSH→복원 F736 | direct MSH, N192/M320/adapter96 | 없음 |
| T09 | F736→learned Score736→복원 F736 | 없음 | Hilbert+D4, **184×4**, 16 strings |
| T18 | F736→LR56 복원 + MSH residual 복원 | residual MSH, N192/M320/adapter96 | Hilbert+D4, 16/16/16/8, 16 strings |

10월 18일 검토 당시 config는 `use_residual=True`를 요구하고 checkpoint 로더도 MSH를 전제했다. **10월 19일 명시적 single-path mode·공통 초기화·train holdout과 제출 묶음을 구현했다.** T07/T08/T18 세 모델의 기본 제출과 T09 선택 옵션은 [구조 pilot 실행 안내](NFCGS_ARCHITECTURE_PILOTS_2026-10-19.md)를 따른다. 이는 기존 checkpoint에서 branch를 지우는 방식이 아닌 새 학습 실험이다.

공정한 선별 조건은 다음과 같다.

1. 같은 공개 GlobalSplat backbone과 QR 기반 observable geometry 초기화를 사용한다. backbone·GaussianDecoder는 고정하고 geometry projection은 현행과 같이 모든 구조에서 학습한다. 특정 구조에서만 고정하지 않는다.
2. codec은 네 구조 모두 새로 초기화한다. 기존 2-path의 학습된 codec만 T18에 재사용하지 않는다. 공통 모듈은 같은 초기 tensor를 복사해 해시로 확인한다. seed만 같으면 모듈 생성 순서 때문에 가중치까지 같아지는 것은 아니다.
3. T07/T18은 호환되는 rank56 transform·확률 모델 초기값, T08/T18은 호환되는 MSH 초기값을 공유한다. 구조 고유 모듈은 별도로 초기화하고 정책을 기록한다.
4. 같은 Hilbert 순서, view 선택, batch, 50k step, optimizer, 평가 프로토콜을 사용한다. 새 구조 검증용 holdout은 T18에도 똑같이 적용한다.
5. 실제 packet roundtrip과 packet-only 수신 복호, 학습 경로의 gradient, 비활성 branch의 optimizer 제외를 확인한 뒤 본 학습을 시작한다. 재구성 tensor를 저장해 용량으로 대신하지 않는다.

T09의 기본 slice16을 그대로 쓰면 46그룹×4단계=184 strings가 되어 다른 실험이 된다. 184×4그룹으로 맞추어 16 strings를 유지한다. 그래도 rank56과 용량·연산량이 동일한 모델은 아니다. 실제 Context 생성자로 확인한 D4 anchor predictor parameter는 rank56에서 5,328개, rank736/그룹184에서 816,960개다. predictor 용량과 부호화 symbol 수도 증가하므로 순수한 병목 차원 효과로만 해석하지 않는다. 구조별 params/MAC·actual bytes·시간·메모리를 함께 제시한다.

### 검증 데이터와 λ 선정

현재 `third_party/ZPressor/mvsplat/src/dataset/dataset_re10k.py`의 `data_stage`는 **val을 test로 매핑**한다. 현재 val loss로 새 구조의 λ·checkpoint를 고르면 test를 선택에 사용하게 된다.

새 구조 네 모델에는 **train에서 고정한 128 scene을 별도 검증용으로 제외**하는 프로토콜을 제안한다. scene/frame 목록·해시를 먼저 고정하고 test 6991 scene과 중복이 없는지 확인한다. 이 128 scene은 T07/T08/T09/T18의 codec 학습에서 모두 제외한다. 평가 view는 C12/T8, seed0으로 고정한다. 기존 T16/T10은 이전 RD와 비교하는 기존 학습 프로토콜을 유지한다.

λ=.0128은 첫 pilot의 출발점이다. 구조마다 같은 λ가 같은 rate를 만들지는 않는다. validation의 actual rate와 수렴으로 λ 범위를 정하고 test에서 λ를 고르지 않는다. pilot 한 점으로 구조 전체의 RD 우위를 선언하지 않는다.

**9/13회는 선별 단계 예산이다.** 짧은 λ 보정 학습, 발산 재시도, pilot 이후 RD 확장은 별도 비용이다. 이미 pilot 한 점씩 있는 K개 구조를 L개 λ로 확장하면 추가 본 학습은 `K×(L−1)`회다. 예를 들어 살아남은 두 구조를 3점으로 확장하면 4회, 5점으로 확장하면 8회가 더 필요하다. λ는 서로 달라도 공통 실제 화질 구간을 충분히 덮어야 한다.

## 6. 추가 학습 없이 할 핵심 분석

### T04: non-comp 기준선을 둘로 분리

| 기준선 | 경로 | 확인하는 손실 |
|---|---|---|
| B0 | 공개 원본 GlobalSplat, geometry decoder 입력512 | 원래 비압축 기준 |
| B1 | 현재 checkpoint의 F736을 기존 geometry224 GaussianDecoder에 직접 입력 | 현행 projection 이후, LR/MSH/양자화를 모두 우회한 기준 |

codec을 끄면 decoder geometry 입력이 224에서 512로 바뀌므로 `feature_codec=null`만 설정해 현재 checkpoint를 그대로 비교할 수 없다. 초기 QR 재매개화는 선형 geometry readout을 보존하지만 현재 geometry projection은 학습 중 변할 수 있다. **B0→B1 변화와 B1→actual codec 변화를 따로 측정**한다. 기존 0.8834dB 차이 전체를 양자화 손실로 부르지 않는다. 최소 H+D4 .0128에서 B1을 측정하고, 모든 기준선에 같은 test scene·camera·정밀도 조건을 적용한다.

### T05: 일곱 경로로 압축 손실 진단

정렬한 F의 연속 Score를 s, 실제 인과적 양자화·복호 Score를 ŝ, low-rank 복원을 L, MSH 연속 분석·합성을 Hcont, 양자화 포함 복원을 Hhat으로 쓴다.

| 모드 | GaussianDecoder에 전달할 feature |
|---|---|
| 1. 현행 비압축 B1 | F |
| 2. 연속 low-rank만 | L(s) |
| 3. 실제 복호 Score만, residual0 | L(ŝ) |
| 4. Score 양자화 우회 | L(s) + Hhat(F−L(s)) |
| 5. MSH y/z 양자화 우회 | L(ŝ) + Hcont(F−L(ŝ)) |
| 6. 양쪽 양자화 우회 | L(s) + Hcont(F−L(s)) |
| 7. 실제 2-path packet | L(ŝ) + Hhat(F−L(ŝ)) |

Score 경로가 바뀔 때마다 MSH 입력 residual을 다시 계산한다. Hcont에서도 분석·합성의 정보 손실은 남으므로 identity로 간주하지 않는다. 가중치·정규화·entropy buffer는 고정한다. 먼저 기존 동일 256 scene에서 확인하고 핵심 모드는 6991 scene으로 확장한다. 모드 간 PSNR 차이는 상호작용을 포함하므로 더해서 경로별 독립 기여로 해석하지 않는다. 연속값 우회 모드에는 실제 packet rate가 없으며 actual RD 점으로 표시하지 않는다.

### T06: 정렬→실제 anchor→잔차→bits 연결

기존 학습 모델 분석은 그대로 활용한다. 추가 진단에서는 H+D4 .0128의 **같은 F·projection·basis·Score·XYZ**를 고정하고 slot/Morton/Hilbert와 random 순서 10 seeds를 비교한다. 이웃 간 3D 거리, 채널별 정규화 Score 거리·상관, 단계별 실제 anchor와 대상의 거리, 예측 잔차, bits/token을 연결한다.

D4 단계는 0 mod8 → 4 mod8 → 2 mod4 → 홀수이며, 뒤 세 단계의 참조 간격은 각각 ±4/±2/±1이다. 오른쪽 anchor가 없으면 왼쪽을 복제한다. 입력 anchor는 앞 단계에서 실제 양자화·복호한 값을 사용한다. 고정 predictor에 순서만 바꾸는 결과는 학습 분포 밖 민감도 진단이며, 각 순서로 재학습한 모델의 RD 비교를 대신하지 않는다.

채널별 정규화 기준은 순서마다 다시 맞추지 않고 공통 reference에서 고정한다. raw 거리·양자화 단위 거리도 함께 본다. 단계별 raw bytes와 `8×raw bytes/token 수`, 전체 Score·MSH y/z·header를 함께 보고한다. 앞 단계 token 수가 적다는 이유만으로 예측 성능이 좋다고 해석하지 않는다.

### T15: 순열과 packet-only receiver

GaussianDecoder는 token마다 독립적으로 Gaussian을 생성한다. 현재 stage3에서는 token당 Gaussian 8개이므로 Gaussian 결과는 **8개 묶음의 순열**로 대조한다. 이번 로컬 CPU float64 임의 입력 검사에서는 means/rotations/scales/SH/opacities 모두 순열 복원 후 최대 차이 0이었다. 이는 decoder 모듈 검사이며 실제 checkpoint·GPU 렌더·독립 수신 검증은 아직 아니다.

실제 수신 검증은 새 프로세스에서 packet bytes와 공유 codec/GaussianDecoder 가중치·설정만 로드한다. 입력 이미지·encoder token·원래 순열·encoder의 정렬 함수를 사용할 수 없게 한다. target camera는 렌더러에 별도로 제공한다. Gaussian 집합 오차와 렌더 max-abs/RMSE를 측정하고, 같은 입력의 반복 GPU 렌더 차이도 함께 측정한다. 깊이 동률과 수치 연산 때문에 픽셀 단위 완전 동일함을 미리 가정하지 않는다. encoder가 없는 이 수신 프로세스의 메모리를 별도로 잰다.

### T12: 복잡도 보완

기존 37 checkpoint의 A5000 profile은 유지한다. 완료 46점 중 빠진 것은 Both OFF 3점과 OFF1/OFF4 6점, 총 9점이다. 여기에 새 OFF16·주요 구조를 보완한다. 중앙 .0128을 우선하고 전체 양끝점 profile은 필요에 따라 확장한다.

새 GPU 할당에서 빠진 모델만 측정한 값을 과거 시간과 바로 섞지 않는다. **H+L3 / H+D4 / NN-score .0128 reference 세 개를 같은 새 할당에서 함께 재측정**해 새 모델과 비교한다. 32 scene·warmup4·반복3·threads8·BF16·batch1 및 입력 cache 해시를 맞춘다. 과거 37개 전체 재측정은 필요 없다.

현재 profile resume는 manifest·환경·GPU UUID를 엄격하게 확인한다. 수정한 manifest를 과거 결과에 이어 쓰지 말고 새 output을 사용한다. 기존 입력 cache를 명시적으로 재사용하는 경로를 확인·준비한다. params/MAC와 실제 encode/decode 시간을 구분하고, CPU rANS/CDF 갱신 및 native kernel의 연산량 집계 범위를 쓴다. CUDA allocated/reserved, codec 추가 peak, 전체 프로세스 peak, 독립 receiver peak를 구분한다.

### T13/T19: payload·RD·범위 표

46점 모두 actual bytes가 Score+MSH y+MSH z+mean+container 합계와 로그 출력 정밀도 내에서 일치한다. 다만 현재 `actual_score_bytes`에는 Score 내부 wrapper가 들어 있고, `actual_container_bytes`에는 그것이 들어 있지 않다.

현재 SCCTX001 packet의 Score wrapper는 **24+8K bytes**다. K는 실제 Score entropy strings 수로 OFF1=1, OFF4=4, L3=12, D4=16이다. 모든 header를 분리하려면 `raw Score=Score bytes−(24+8K)`, `all headers=container bytes+(24+8K)`로 집계한다. config에서 K를 유도한 값은 유도값이라고 표시하고 대표 packet을 실제 재파싱해 확인한다. MSH y/z는 실제 entropy string 길이를 사용하며 mean은 별도 항목으로 둔다.

최종 RD는 양의 λ 다섯 점 .0032/.0064/.0128/.0256/.0512를 사용하고 λ0는 별도 화질 진단으로 둔다. λ0에서도 양자화와 entropy coding은 동작한다. 같은 λ 비교와 같은 화질 비교를 구분한다.

scene/frame 목록이 같은지 확인한 후 scene 단위 paired bootstrap 1,000회로 95% CI를 계산한다. 같은 scene 표본을 모델·λ에 공통으로 적용한다. 이 CI는 scene 변화의 불확실성이며 seed 변화의 신뢰구간은 아니다. 추가 seed 한 쌍은 총 두 seed의 효과 반복으로 보고한다.

BD-rate는 공통 PSNR 구간에서 log-rate를 적분하고, BD-PSNR은 공통 log-rate 구간을 별도로 사용한다. PCHIP 보간 전 단조성·중복점을 확인하고 외삽하지 않는다. 비단조·누락점 처리 규칙을 기록하며 불리한 점을 조용히 제거하지 않는다. 서로 다른 공통 구간으로 얻은 BD 수치를 직접 종합 순위로 쓰지 않는다.

적용 범위 표에는 정렬×스케줄×공간 Context×분할을 함께 쓴다. 없는 matched OFF는 빈칸으로 두며 L3 또는 D4 한 점의 결론을 모든 스케줄로 확대하지 않는다.

## 7. 실행 순서와 논문 산출물

| 순서 | 작업 | 완료 기준 |
|---|---|---|
| 1 | 기존 확장 13개 접수·완료 확인, 미접수점만 보완 | train50k + 연결 actual eval 완료 |
| 2 | T16 6개 + T10 OFF16 1개 준비·실행 | L3 2×2 다섯 점 + D4 matched Context 비교 |
| 병행 | T04/T05/T15, 기존 46점 payload·범위 표 | projection/양자화 손실 분리와 독립 수신 검증 |
| 3 | 새 구조 경로·검증 holdout 구현, T07/T08 pilot | 실제 packet roundtrip + 수렴·실제 rate 확인 |
| 4 | T18을 우선하고 T09, T11 seed 쌍 | 공통 초기화의 구조 비교, 핵심 효과 반복 |
| 5 | 살아남은 구조의 추가 λ만 확장, 부족한 profile·정성 평가 | 공통 화질 구간 actual RD와 시간/메모리 |

현재 rank56 50k 작업의 과거 관측치를 이용한 대략적 자원 계획은 2 GPU 기준 16–24시간/회, 즉 32–48 GPU-hours/회다. 큐 대기·장비·경로별 차이는 별도이며 full-width736의 시간은 아직 측정하지 않아 같은 추정치를 적용하지 않는다. QoS 제한 시 이미 접수된 ID를 저장하고 미접수분만 재시도한다.

사용자가 제시한 **11월 1일 새 구조·설정 동결, 11월 10일 제출**을 기준으로 한다. 새 구조가 공통 기준선·실제 packet·공통 rate 구간까지 준비되지 않으면 그 구조의 우열 주장을 늘리지 않고 현행 방법의 검증을 우선한다.

최종 산출물은 주 RD, L3 2×2와 D4 ON/OFF16, 실제 anchor·잔차·단계별 bits, Score/MSH/header 분해, 시간·메모리, 공통 레시피 구조 비교, 같은 view/crop의 정성·실패 사례로 연결한다. 정성 예시는 전체 정량 결과를 대체하지 않으며 rate 또는 quality를 맞춘 조건과 선택 규칙을 표시한다.

## 8. 선택 항목으로 남길 실험

Morton+L3 OFF12 .0128, L3 추가 seed 쌍, 새 seed의 None ON/OFF 쌍, geometry projection 자체의 고정/학습 비교는 기본 9·권장 13회에 넣지 않는다. 정렬×Context 상호작용 전체의 seed 재현성이 핵심 주장이 되면 None 쌍을 추가한다.

identity feature에 직접 Context를 적용하는 대안, MSH×2, LR×2, predictor 용량을 맞춘 비교, 다른 backbone·view 수·외부 codec은 구조 pilot 결과가 설명하지 못하는 구체적 질문이 남을 때 선택한다. 기존 NN-score RD와 복잡도는 재사용하며 새 NN-score 양끝점이나 가속기 개발을 필수 목록에 늘리지 않는다.

## 9. 이번에 준비한 범위

- 완료 로그 46점의 평가 수·용량 합계와 출처 목록을 정리했다. 개별 scene/frame 목록 일치 여부는 별도 검증이 남아 있다.
- 새 9/13회 실행 목록과 기존 codec 대조군 7개의 train/eval 설정 조합을 검증했다.
- OFF16 CPU entropy roundtrip, full-width Context 생성 시 그룹·parameter 수, decoder의 token 순열 성질을 확인했다.
- 기존 codec 대조군 7개용 launcher와 10월 19일 구조 pilot용 codec·공통 초기화·train holdout·제출 묶음을 준비했다. 실제 서버 holdout 생성·GPU 학습·평가는 아직 실행하지 않았다. 이 문서와 계획 JSON 자체는 제출 명령이 아니다.
- T11 Hilbert+D4 ON16/OFF16 추가 seed 쌍도 준비했다. 기존 부모·전체 train split을 유지하고 model111124/data404로 각각 새50k를 학습한다. 별도 prepare 없이 총4개 job이며 실제 서버 제출은 아직 수행하지 않았다.
- T09 full-width Context-only의 추가 실행도 요청받았다. 기존 구조 launcher의 `--include-fullwidth`를 사용하며 기본7개가 접수된 state 기준으로 학습1개·128장 평가1개만 추가한다. [T09 실행 안내](NFCGS_FULLWIDTH_CONTEXT_PILOT_2026-10-19.md)를 따른다.

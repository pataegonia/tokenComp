# 정렬·context 단독 RD 실험 — 2026-10-13

정렬 없이 context만 사용하는 legacy3·quarter2·dyadic4와, context 없이 Hilbert 정렬만 사용하는 설정을 λ=.0064/.0128/.0256에서 각각 학습한다. Sorting-only 정렬은 사용자가 선택한 Hilbert다. 모든 설정에 kernel5/STE를 사용하며 quarter2/dyadic4의 예측은 kernel convolution 대신 anchor gather로 수행한다.

| 실험 | Token 순서 | Score 공간 예측 | λ | 학습 수 |
|---|---|---|---|---|
| Context only, legacy3 | GlobalSplat의 원래 token slot 순서 | ON, 3단계 masked convolution | .0064/.0128/.0256 | 3 |
| Context only, quarter2 | GlobalSplat의 원래 token slot 순서 | ON, 2단계 anchor gather | .0064/.0128/.0256 | 3 |
| Context only, dyadic4 | GlobalSplat의 원래 token slot 순서 | ON, 4단계 anchor gather | .0064/.0128/.0256 | 3 |
| Hilbert only | Hilbert, 10bit | OFF | .0064/.0128/.0256 | 3 |

전체 목록은 학습 12개이며 각 학습의 성공 종료 후 전체 scene eval 1개를 `afterok`로 연결한다. 총 24 Slurm jobs다. 학습당 GPU2·유효배치8이며 학습 12개가 모두 동시에 실행되면 GPU24개를 사용한다. 실제 동시 실행 수는 Slurm 자원 배정에 따른다. 이전 legacy3 context-only와 Hilbert-only의 6개 학습이 이미 제출됐다면 기존 state의 ID를 재사용하고 quarter2/dyadic4 학습 6개와 후속 eval 6개만 추가한다.

## OFF의 의미와 비교 범위

- `token_order=none`은 identity 순열이다. Morton/Hilbert/NN 정렬과 거리 계산을 호출하지 않으며 Score와 MSH 모두 원래 token 순서를 사용한다.
- `score_spatial_context=false`는 공간 예측 모듈을 우회하고 동결한다. 먼저 복원한 token 값으로 다른 token의 Score를 예측하지 않는다. Hilbert only는 기존 stage/group별 독립적인 factorized entropy model, 학습된 median을 포함한 양자화와 legacy3의 3단계 stream 분할을 유지한다. 기존 legacy3 대조군과 entropy 모델 수와 stream 수를 맞추기 위한 선택이다.
- 공통 minimal 설정에서 scene centering·Score norm·mean conditioner·channel context·추가 nonlinear MLP는 OFF다. 따라서 Hilbert only에는 mean/channel/spatial context가 모두 없다.
- MSH의 내부 nonlinear transform·normalization·hyperprior 및 y/z stream은 유지한다. 이 실험의 context OFF는 Score 예측 모델에 대한 ablation이다.
- Spatial OFF는 Score stream의 기존 spatial flag를 해제한다. 잘못된 ON/OFF 설정으로 복호하면 거부한다. 순열 stream을 추가하지 않는다.
- 기존 metadata는 정렬/context ON으로 해석되며 기본 동작은 유지한다. OFF는 checkpoint metadata에 저장한다. 다른 설정으로 바꾸는 것은 명시적인 weights-only warm start에서만 허용하고 strict resume/eval에서는 거부한다.

각 context-only 곡선은 같은 context schedule을 쓰는 기존 Morton/Hilbert 정렬+context ON 곡선과 비교한다. 예를 들어 정렬 없는 dyadic4는 Hilbert dyadic4와 비교해 정렬의 영향을 확인한다. Hilbert only는 기존 Hilbert legacy3와 비교해 공간 예측의 영향을 확인한다. Hilbert only와 quarter2/dyadic4는 독립적인 entropy 모델·stream 분할도 다르므로 그 차이를 공간 예측 ON/OFF 효과 하나로 해석하지 않는다. Sorting-only는 Hilbert legacy3 OFF 한 곡선으로 유지하며, 정렬/context를 모두 끈 학습은 이번 제출 목록에 포함하지 않는다.

## 공통 학습·평가 조건

- 열두 학습 모두 같은 FullSplit λ=.0256 step10k 부모의 가중치에서 독립적으로 시작한다. 부모의 optimizer·scheduler·global step은 이어받지 않는다.
- GlobalSplat 고정, codec 전체와 MSH 학습, rank56.
- 학습 C24/T12, subset consistency ON, GPU2·batch/GPU2·accumulate2, 유효배치8.
- 새50k optimizer steps, LR1e-4, 35k/45k decay, checkpoint 5k 간격, seed111123, BF16, 시간 제한 6일.
- 전체 test scene 실제 bitstream encode/decode 및 렌더링, C12/T8·batch1·seed0·BF16, test-time optimization 없음.
- 기본 부모 경로와 λ는 helper에 고정돼 있다. 부모가 이동했다면 `SOURCE_CHECKPOINT=/absolute/path/to/parent.ckpt`로 지정한다.

## 서버에 반영할 파일

| 파일 | 용도 |
|---|---|
| [compression/config.py](../globalsplat/compression/config.py) | 정렬·공간 예측 OFF 허용 및 검증 |
| [compression/codec.py](../globalsplat/compression/codec.py) | identity 순열, 공간 예측 설정·동결 |
| [compression/score_context.py](../globalsplat/compression/score_context.py) | 공간 예측 우회, OFF flag와 복호 검증 |
| [compression/checkpoint.py](../globalsplat/compression/checkpoint.py) | OFF metadata·weights-only 변환·strict 검증 |
| [globalsplat_nfcgs_rank56.yaml](../config/model/globalsplat_nfcgs_rank56.yaml) | 옵션 주석; 기존 기본값 유지 |
| [run_nfcgs.py](../scripts/run_nfcgs.py) | `--token-order none`, `--no-spatial-context` |
| [nfcgs_order_context_settings.sh](../scripts/slurm/nfcgs_order_context_settings.sh) | train/eval의 동일 설정과 OFF 출력 경로 |
| [submit_nfcgs_order_context.sh](../scripts/slurm/submit_nfcgs_order_context.sh) | OFF state 서명 분리·설정 preview |
| [submit_nfcgs_sort_context_ablation_rd.sh](../scripts/slurm/submit_nfcgs_sort_context_ablation_rd.sh) | 12 point 제출·중복 방지·부분 실패 재시도 |

기존 `train_nfcgs_order_context.slurm`, `eval_nfcgs_order_context.slurm`은 그대로 사용한다. 이 두 파일과 앞서 적용한 `main.py` Numba 로그 필터 수정이 서버에 존재해야 한다.

## 제출

```bash
cd -P /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
bash scripts/slurm/submit_nfcgs_sort_context_ablation_rd.sh --dry-run
bash scripts/slurm/submit_nfcgs_sort_context_ablation_rd.sh
```

Helper는 열두 설정과 부모·기존 state를 모두 preview한 뒤 제출한다. 다시 실행하면 저장된 train/eval ID를 재사용하며, eval 제출만 실패했으면 해당 eval만 재시도한다. 각 학습은 별도 실행 폴더에 저장된다. 이전 파일들을 이미 서버에 반영했다면 이번 확장에서는 `submit_nfcgs_sort_context_ablation_rd.sh`만 교체하면 된다.

| 설정 | Train name | Eval name | Output |
|---|---|---|---|
| Context only, legacy3 | `gs-co-<λtag>` | `gs-co-e-<λtag>` | `outputs/nfcgs_order_context/none/legacy_stages3_k5_ste/lambda<λtag>/job_<TRAIN_ID>/` |
| Context only, quarter2 | `gs-cq2-<λtag>` | `gs-cq2-e-<λtag>` | `outputs/nfcgs_order_context/none/quarter2_stages2_k5_ste/lambda<λtag>/job_<TRAIN_ID>/` |
| Context only, dyadic4 | `gs-cd4-<λtag>` | `gs-cd4-e-<λtag>` | `outputs/nfcgs_order_context/none/dyadic4_stages4_k5_ste/lambda<λtag>/job_<TRAIN_ID>/` |
| Hilbert only | `gs-ho-<λtag>` | `gs-ho-e-<λtag>` | `outputs/nfcgs_order_context/hilbert/legacy_stages3_k5_ste_context_off/lambda<λtag>/job_<TRAIN_ID>/` |

`λtag`는 `0p0064`, `0p0128`, `0p0256`이다. 각 실행의 최종 checkpoint는 `checkpoints/nfcgs_main/version_0/step000050000.ckpt`, 평가는 `eval_all/`에 저장한다.

State는 `logs/slurm/sort_context_ablation_rd_20261013/{context_only,hilbert_only,context_quarter2,context_dyadic4}_<λtag>.state`와 `submissions.tsv`에 저장한다. 기존 두 arm의 state 이름·설정 서명은 유지한다. 상태 폴더는 `SORT_CONTEXT_ABLATION_STATE_DIR`로 변경할 수 있다.

## 서버 제출 출력 확인 — 2026-10-13

사용자가 제공한 dry-run·제출·squeue·재시도 출력에서 아래 상태를 확인했다. 이 기록은 해당 출력 시점의 snapshot이며 서버를 직접 조회하지 않았다.

| 설정 | λ=.0064 학습 / eval | λ=.0128 학습 / eval | λ=.0256 학습 / eval |
|---|---|---|---|
| 정렬 없는 legacy3 | 450977 / 450978 | 450979 / 450980 | 450981 / 450982 |
| Hilbert only | 450983 / 450984 | 450985 / 450986 | 450987 / 450988 |
| 정렬 없는 quarter2 | 450989 / 450990 | 450991 / 450992 | 450993 / 450994 |
| 정렬 없는 dyadic4 | 450995 / 제출 거부 | 미제출 / 미제출 | 미제출 / 미제출 |

12개의 dry-run 설정과 공통 FullSplit .0256 부모가 일치했다. 새 학습 10개와 후속 eval 9개, 총 19개 job이 접수됐다. dyadic4 .0064의 학습 450995는 접수되고 state에 저장됐지만, 그 eval 제출에서 `QOSMaxSubmitJobPerUserLimit`으로 거부됐다. Helper는 여기서 중단했으므로 dyadic4 .0128/.0256의 학습·eval 제출에는 도달하지 않았다. 남은 제출은 5개다.

붙여준 squeue에는 해당 사용자의 job 30개가 있다. 이번 ablation 19개 외에 기존 NN-score 학습·eval 6개와 별도 MVSplat job 5개가 포함됐다. 계정/QOS에 설정된 정확한 제출 한도 값은 조회하지 않았다. 같은 사용자의 서로 다른 실험 job도 이 queue 출력에 함께 나타나므로, 출력의 총 job 수와 이번 ablation의 job 수는 구분한다.

- 새 학습 10개는 `PD (QOSMaxGRESPerUser)`다. 자원 배정을 기다리는 상태이며, 학습 코드의 첫 step 실패를 보여주는 출력은 아니다.
- 접수된 eval 9개는 `PD (Dependency)`다. 각 학습의 성공 종료를 기다린다. 실제 afterok 연결은 제출 helper의 경로와 로컬 검증을 근거로 하며, 이 squeue 출력에 상세 dependency 값은 없다.
- 기존 NN-score 학습 450934/450937/450939는 `R`이다. 대응 eval ID는 450935/450938/450940이며 `Dependency`로 대기 중이다. `R` 상태만으로 optimizer step이나 처리 속도를 확정하지 않는다.
- 반복 실행에서 기존 학습·eval ID가 재사용됐다. 추가 학습의 중복 제출은 출력에 없다. 다시 시도한 450995의 eval은 같은 QOS 제출 한도로 거부됐다.
- `slurbash: command not found`는 별도의 명령 오타다. 이후의 `bash` 명령은 정상적으로 helper를 실행했으나 같은 QOS 한도에서 멈췄다.

이 출력에서 코드 오류나 설정 변경이 필요한 근거는 없다. 기존 job이 종료돼 제출 슬롯이 확보되면 같은 state 폴더를 유지하고 아래 명령을 다시 실행한다. 남은 5개를 한 번에 접수하려면 그만큼의 제출 여유가 필요하다. 일부만 확보된 경우에도 접수된 ID는 state에 남아 다음 실행에서 재사용된다.

```bash
bash scripts/slurm/submit_nfcgs_sort_context_ablation_rd.sh
```

재시도는 먼저 450995의 eval을 제출하고, 이어서 dyadic4 .0128/.0256의 학습·eval을 제출한다. 새 state 폴더를 지정하면 이미 접수된 실험이 다시 제출될 수 있으므로, 현재 기록을 이어서 사용하는 것이 이 제출 흐름의 복구 절차다. 이번 확인에서는 서버 job을 취소하거나 코드를 수정하지 않았다.

## quarter2/dyadic4 확장 후 추가 검증

- 확장한 ablation 검증: 5개 테스트와 13개 subtest 통과. legacy3·quarter2·dyadic4 모두 정렬 호출 없이 원래 token 순서를 유지했고, nonzero 예측기를 사용한 실제 bitstream 왕복, 같은 FullSplit 부모의 weights-only 변환과 strict 로드, CUDA/BF16 학습·역전파가 통과했다.
- 실제 CLI/Hydra 구성: 1개 테스트와 8개 subtest 통과. 네 설정의 학습·평가에서 정렬·context·schedule·단계 수가 일치했다.
- 제출 helper: 5개 테스트 통과. 12개 point와 개별 afterok 연결, 중복 제출 방지, 부분 eval 제출 실패 복구, 마지막 point의 잘못된 state를 제출 전에 검출하는 것을 확인했다. 원래 6개 학습/eval 쌍의 state가 있는 환경에서는 해당 state와 ID를 유지하고 quarter2/dyadic4의 6개 학습/eval 쌍만 추가했다. 가짜 Slurm 명령을 사용했다.

## 기존 두 설정의 로컬 검증 결과

- 새 ablation 및 기존 main codec 기준 구현 비교: 15개 테스트와 12개 subtest 통과. 실제 bitstream 왕복, 잘못된 context 설정의 복호 거부, MSH 가중치 보존, strict checkpoint 검증, 원래 token 순서 유지, OFF 예측 모듈 우회·동결, 다른 token에 대한 Score 확률·복원 독립성을 확인했다. 두 새 설정의 CUDA/BF16 학습·역전파 smoke test도 통과했다.
- CLI/Hydra와 Slurm 제출·resume: 45개 테스트와 102개 subtest 통과. 여섯 point의 공통 부모·개별 설정·학습/eval ID·afterok 연결, 반복 실행의 ID 재사용, 부분 eval 실패 복구, OFF state 서명 분리를 확인했다. Slurm 검증에는 가짜 명령을 사용했다.
- 기존 Score/context/정렬/resume/log-filter 회귀: 37개 통과, CUDA+Numba 검사 1개 생략. 해당 테스트 실행 환경의 PyTorch가 CPU 빌드여서 이 검사만 생략했다.
- Python 구문과 Slurm shell의 LF·BOM 없는 shebang, shell 구문 검사를 통과했다.

서버 job은 사용자가 제출했으며, 이 세션에서는 제공된 출력과 로컬 파일을 확인했다. 실제 RD 성능과 서버의 학습·부호화 시간은 실행 로그·평가 결과로 확인해야 한다.

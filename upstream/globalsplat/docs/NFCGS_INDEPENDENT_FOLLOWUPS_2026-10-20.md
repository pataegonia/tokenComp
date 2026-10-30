# 독립 후속 실험 실행 안내 — 2026-10-20

준비일: 2026-10-20. 이 문서는 구현·실행 안내다. 로컬에서 서버 job을 제출한 기록은 아니다.

21:59 업데이트: **736의 최종50k holdout128 평가는 완료됐다.** 아래5개 job은 736 대기 중 준비한 독립 후속이며 실행 조건은 유지한다. 접수 state/로그는 아직 다운로드되지 않았다. 최신 결과는 [21:59 검토](analysis/nfcgs_logs_2026-10-20_2159/REPORT.md)를 따른다.

## 실행 범위

| 작업 | 새 job | 자원·조건 | 출력 |
|---|---:|---|---|
| 기본 seed None+D4 OFF16 .0128 | 학습1 | GPU2, batch/GPU2×accumulation2, 50k, LR1e-4, milestone35k/45k | 기존 order-context run 구조 |
| 위 모델 전체 actual test | 평가1 | 새 학습의 afterok, GPU1, C12/T8, 최종50k | 해당 run의 eval_all |
| N01 경계 metadata 검토 | CPU1 | 기존 B0/T05 matched256, 새 모델·렌더 없음 | followup/boundary |
| N03 native 전체 test | GPU1 | 공개 native geometry512, 기존 codec6991개 scene/frame 목록과 정합 | followup/native |
| N02 T06 품질 보완 | GPU1 | 기존 slot/Morton/Hilbert packet256×3, 기존 T15 target bundle | followup/fixed-render |
| 합계 | **5개** | 새 학습은1개 | |

학습 조건은 기존 FullSplit λ=.0256 부모의 weights-only warm start, model seed111123/data seed403, rank56+MSH, minimal linear, D4 4단계·16/16/16/8·16 Score strings다. 기존 None+D4 ON16과 비교할 OFF16이며 구조 fresh-init pilot과 섞지 않는다.

**736 학습457828·연결 평가457829, 복구458782/458783과 기존 eval457098/457100은 변경하지 않는다.** MSH-only/two-path의 추가λ, 추가 seed, fresh 2×2는 이번 묶음에 포함하지 않는다. 736 결과를 반영한 추가λ 우선순위는 최신 검토를 따르며 실행 도구 확장은 별도다.

## 서버에 올릴 파일

저장소의 동일한 상대 경로에 개별 파일을 반영한다. 기존 codec/model/loader/config 파일은 이번 구현에서 수정하지 않았다.

- [제출 도구](../scripts/submit_nfcgs_followup_20261020.py)
- [평가·분석 도구](../scripts/run_nfcgs_followup_20261020.py)
- [Slurm 실행 파일](../scripts/slurm/run_nfcgs_followup_20261020.slurm)
- [테스트](../tests/test_followup_20261020.py) — 검증용
- 이 문서

기존 `train_nfcgs_order_context.slurm`, `eval_nfcgs_order_context.slurm`, `nfcgs_order_context_settings.sh`, B0/T05/T06/T15 runner와 현재 codec runtime은 이미 올린 버전을 그대로 사용한다. 제출 도구는 필요한 파일·LF shebang·Ceph checkpoint TMPDIR 수정·산출물 정합을 먼저 확인한다.

## 서버 명령

```bash
cd /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
python scripts/submit_nfcgs_followup_20261020.py --dry-run
python scripts/submit_nfcgs_followup_20261020.py
```

첫 명령은 기존 source artifact와 설정을 검증하고 제출 계획만 출력한다. 두 번째 명령이5개 job을 접수한다. CPU audit에는 GPU를 요청하지 않는다. GPU 평가는1개씩, 새 학습은2개를 요청한다. 실제 시작 시점은 scheduler와 GPU 여유에 따른다.

ID 저장 위치:

```text
logs/slurm/followup_20261020/state.json
```

QoS 등의 이유로 일부 접수가 거부되면 **같은 명령을 다시 실행**한다. 접수 ID를 매번 저장하고, 저장된 ID는 재사용하며 미접수 job만 시도한다. 기존 eval을 새 학습에 연결하는 것이 아니라 이 새 학습의 eval 하나를 `afterok:<새 train ID>`로 접수한다. 기존 job을 취소·재연결하지 않는다. State 폴더를 지우거나 다른 폴더로 바꿔 재제출하지 않는다.

제출 도구가 실제 ID를 넣은 `squeue` 명령을 출력한다. 새 분석/평가 stdout는 `logs/slurm/slurm-gs-followup20-<JOB>.out`, 학습·연결 평가는 기존 order-context stdout 이름을 사용한다.

## 기존 산출물의 기본 경로

```text
B0:      outputs/nfcgs_b0_matched/job_458024
T05:     outputs/nfcgs_path_diagnostics/job_458004
T06:     outputs/nfcgs_fixed_order/job_458818
Targets: outputs/nfcgs_independent_receiver/job_458784/bundle
Native:  checkpoints/pretrained/globalsplat-re10k-32k.ckpt
Reference:
  outputs/nfcgs_order_context/hilbert/dyadic4_stages4_k5_ste/lambda0p0128/job_447746/
  eval_all/nfcgs_main/actual_rate_per_scene.json
```

Reference는 PSNR/SSIM/LPIPS·실제 bytes·context/target frame ID가 있는6,991개 장면 목록이어야 한다. Native 평가는 이 목록의 장면과 frame을 맞추며 누락된 장면을 다른 장면으로 채우지 않는다. 기존 reference codec의 지표를 재사용하고 native만 새로 렌더한다. Native에는 압축 packet rate를 부여하지 않는다.

N02는 기존 T06 packet SHA·stream 길이와 target bundle SHA/frame을 검증하고 **재인코딩 없이** 최소 수신기로 복호한다. Hilbert 품질은 T05의 기존 independent-receiver 허용치 PSNR1e-3/SSIM1e-5/LPIPS1e-5 안에 있어야 한다. 기존 T05와 GPU 종류·소프트웨어·수치 정책도 검사한다. 원본 데이터셋과 scene encoder는 이 렌더 작업에서 사용하지 않는다.

N02의 inverse-permutation feature MSE는 T06 기록을 재사용한다. 전체 Gaussian의 역순열 오차를 새로 측정한 결과로 표시하지 않는다. Frozen Hilbert-trained 모델의 정렬 개입이며 재학습 RD 또는 동일 화질 절감률이 아니다.

N01은 기존 `model_pair.geometry_boundary_effective_readout_errors`와 경로별 feature 오차·B1 drift를 표로 만든다. 초기 QR operator를 새로 복원한 실험이나 고정 projection 재학습은 아니다. Metadata가 설명하는 범위를 넘어 B1 급락을 버그나 독립 압축 손실로 단정하지 않는다.

경로가 실제로 변경된 경우에는 `FOLLOWUP_B0_RUN`, `FOLLOWUP_PATH_RUN`, `FOLLOWUP_FIXED_RUN`, `FOLLOWUP_TARGET_BUNDLE`, `FOLLOWUP_CHECKPOINT`, `FOLLOWUP_NATIVE_CHECKPOINT`, `FOLLOWUP_REFERENCE`로 지정할 수 있다. 경로·코드가 저장된 state와 달라지면 재접수를 막는다. 데이터셋 index는 node-local 경로이므로 compute job에서 확인한다.

## 결과 받기

새 진단/평가 결과는 다음 위치에 생성된다.

```text
outputs/nfcgs_followup_20261020/boundary/job_<CPU_JOB>/REPORT.md
outputs/nfcgs_followup_20261020/native/job_<GPU_JOB>/REPORT.md
outputs/nfcgs_followup_20261020/fixed-render/job_<GPU_JOB>/REPORT.md
```

각 디렉터리에 `metadata.json`, `summary.json`, `complete.json`이 있고 GPU 평가에는 `scenes/`가 있다. 전체 결과 해석에는 scene별 원자료도 함께 내려받는다. N03/N02는 중단된 같은 출력에 한해 명시적 `--resume`를 지원한다. 실패 job을 제출 도구가 자동으로 다시 만들지는 않는다. 재개할 때는 원래 output과 task를 명시하고 `FOLLOWUP_OUTPUT`·`FOLLOWUP_RESUME=1`을 설정한 별도 `sbatch`를 사용한다. 새 제출 묶음은 이 두 환경변수를 물려받지 않는다.

## 로컬 검증 범위

새 테스트는 실제 entropy packet과 최소 수신기·Gaussian decoder를 사용하며 렌더/품질 함수는 CPU fixture로 대체한다. Scene/frame 정합, source metadata schema, packet 손상·Hilbert 불일치 시 완료 방지, 완료 scene 재사용, 부분 제출 실패 후 ID 재사용을 검사한다. Slurm 파일은 Bash 구문과 LF를 확인한다.

실제 공개 checkpoint·서버 데이터셋·CUDA rasterizer의 본 평가는 서버 job에서 수행한다. 새 학습을 로컬에서50k 실행한 것으로 보고하지 않는다.

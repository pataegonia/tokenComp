# T09: 736채널 Context-only 구조 pilot 추가

T07/T08/T18 구조 비교와 T11 추가 seed 쌍 다음으로 요청한 구조 실험이다. T09 구현은 기존 구조 pilot 파일에 이미 포함돼 있으므로 실행 코드를 추가로 업로드할 필요가 없다.

## 경로와 비교 대상

| 모델 | 경로 |
|---|---|
| T07 | F736 → learned Score56 → dyadic4 Context → F736, MSH 없음 |
| T09 | F736 → learned Score736 → dyadic4 Context → F736, MSH 없음 |
| T18 | F736 → rank56 복원 + MSH residual 복원 |

T09는 rank56 차원 병목을 제거한 Context-only 대안이다. identity feature에 바로 확률 모델을 적용하는 실험과는 다르며, 736→736의 학습 가능한 분석·합성 transform을 유지한다. 공통 geometry512→224 projection도 유지한다.

T09는 Hilbert, λ=.0128, 새50k, GPU2×batch2×누적2, BF16, model seed111123/data seed403, minimal/linear/kernel5/STE다. 공통 공개 backbone과 QR 초기화, train에서 제외한 동일128장 검증을 T07/T08/T18과 공유한다. 기존 FullSplit 부모에서 후속 학습하는 T11과 초기화·학습 데이터가 다르므로 T11을 구조 기준선 대신 쓰지 않는다.

## 스트림 수와 해석

Score 채널 그룹은 **184/184/184/184**, dyadic4 token 4단계다. 실제 Score entropy strings는 **16개**로 유지한다. rank56에서 쓰는 slice16을 그대로 적용해 184 strings를 만드는 설정이 아니다.

rank56보다 symbol 수와 predictor 용량이 커진다. 차원만 바꾼 동일 용량 모델로 해석하지 않는다. 같은 λ도 같은 실제 bitrate를 보장하지 않으므로 이번 한 점은 구조 선별용이며, 실제 PSNR·bytes·학습 시간·메모리를 함께 확인한다.

## 실행과 job 수

기존 구조 prepare는 처음부터 네 모델의 초기화를 모두 만든다. **기존 기본 7개 job이 모두 접수된 state가 있으면 새 학습1개(GPU2) + holdout 평가1개(GPU1), 총2개 job만 추가**된다. T09 학습도 기존 prepare의 `afterok`에 연결된다. 기존 prepare가 아직 실행 중이어도 확장 제출할 수 있다.

```bash
cd -P /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
if bash scripts/slurm/submit_nfcgs_architecture_pilots.sh --include-fullwidth --dry-run; then
    bash scripts/slurm/submit_nfcgs_architecture_pilots.sh --include-fullwidth
fi
```

dry-run에서 기존 세 모델은 접수된 숫자 ID로, `score736_only`는 `TRAIN=new VALIDATION=new TEST=deferred`로 표시되는 것이 정상이다. 기존 세 모델의 미접수 job이 남아 있으면 이 명령은 그것도 함께 보완한다. state가 없는 처음 실행이라면 전체가 prepare1+학습4+검증4=9개 job이므로 추가2개라는 계산을 적용하지 않는다.

기존 `ARCH_PILOTS_STATE_DIR`, `ARCH_PILOTS_PREPARATION`, `ARCH_PILOTS_OUTPUT_ROOT`, `VANILLA_CHECKPOINT`, `DATASET_ROOT`를 따로 지정했다면 같은 값을 유지한다. 접수된 ID는 재사용하며 QoS 제출 제한 후에도 같은 명령으로 미접수분만 재시도한다. 전체 test는 이번 명령에 포함하지 않는다.

## 출력

| 항목 | 기본 경로 |
|---|---|
| 제출 state | `logs/slurm/architecture_pilots_20261019/state.json`의 `arms.score736_only` |
| 초기화 | `outputs/nfcgs_architecture_pilots/initialization_20261019/score736_only.ckpt` |
| 학습 RUN | `outputs/nfcgs_architecture_pilots/lambda0p0128/score736_only/job_<TRAIN_ID>/` |
| 최종 checkpoint | `RUN/checkpoints/nfcgs_main/version_0/step000050000.ckpt` |
| 128장 평가 | `RUN/eval_validation/nfcgs_main/` |
| 로그 | 기존 `slurm-gs-arch-train-<ID>` 및 `slurm-gs-arch-eval-<ID>` |

이번 준비에서 원격 job을 직접 제출하거나 취소하지 않았다. 제출 검사4개와 네 모델의 train/validation/test Hydra 설정12개를 확인했다. T09도 4096-token BF16 CPU 압축·별도 프로세스의 정확한 복호·유한 gradient backward를 통과했다. 이 CPU 검사는 무작위 codec 가중치를 사용한 기능 검사이며 실제 backbone 평가나 학습된 모델의 rate 결과는 아니다.

확인 범위는 `docs/analysis/nfcgs_fullwidth_context_pilot_20261019/validation.json`에 기록한다. 실제 GPU에서는 기존 train job이 같은 4096-token BF16 압축·독립 수신 복호·gradient smoke를 통과한 뒤 학습을 시작한다.

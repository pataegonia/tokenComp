# Morton 순번에 따른 2·3·4단계 score context

MSH residual path는 유지한다. Score hyperprior는 추가하지 않는다. 기존
`minimal` 설정(centering, score norm, mean/channel context, nonlinear transform
OFF)에 spatial 복호 순서만 세분한 실험을 추가했다. 일반 실행의 기본값은
기존 Full P0 + 2단계, kernel 3, noise 학습이다.

## 분할과 참조

Morton 정렬 **이후**의 순번으로 `A=0 mod 4`, `B=2 mod 4`,
`C=1 mod 4`, `D=3 mod 4`를 정의한다. 모든 token과 채널을 유지한다.

| 단계 수 | 복호 순서 | 사용 가능한 이전 token |
| --- | --- | --- |
| 2 | even → odd | odd는 even 전체 참조 |
| 3 | A → B → C+D | B는 A, C+D는 A+B 참조 |
| 4 | A → B → C → D | B는 A, C는 A+B, D는 A+B+C 참조 |

3·4단계는 한 token 단계의 모든 channel slice를 완료한 뒤 다음 token
단계로 진행한다. Channel slice마다 기존 형태의 factorized entropy model과
spatial convolution을 유지하고, 다른 slice에 대한 channel context는 사용하지
않는다. Predictor 입력은 원래 Morton sequence 길이를 유지한 canvas다.
이전 단계의 복원값만 채우고 현재·미래 단계 위치는 0으로 가린다.
Predictor에 원본 token, 현재 단계의 다른 token 값 또는 미래 token 값은
들어가지 않는다. Mean context를 별도로 켜면 이전 복원값에서 공통 mean
offset을 뺀 값을 canvas에 넣는다.

Kernel 5의 ±2 범위로 B가 A를, D가 C를 참조할 수 있다. Padding은 0이다.
Morton 순번의 인접 관계를 사용하며, 3D 거리 기준 이웃을 새로 찾지는 않는다.
같은 token 단계 내의 위치는 병렬 처리한다. 같은 단계의 원본 값에 접근하는
attention이나 convolution은 추가하지 않았다.

## 설정

```text
--score-spatial-stages 2|3|4
--score-spatial-kernel 3|5
--score-context-quantization noise|ste
```

3·4단계는 kernel 5, channel context OFF가 필요하다. CLI에서 3·4단계를
선택하면 kernel은 5, 학습 quantization은 `ste`로 기본 설정한다.
`ste`는 learned entropy median을 포함한 실제 양자화 복원값을 다음 단계에
전달한다. Gradient는 straight-through 방식으로 전달하고, likelihood 학습에는
기존 noise relaxation을 유지한다. 2단계의 기존 기본값 `noise`는 변경하지
않았다. 실제 압축과 복호화는 두 모드 모두 동일한 양자화 규칙을 사용한다.

비교 스크립트는 **2·3·4단계 모두 kernel 5 + STE**를 사용한다. 따라서
기존 2단계 noise 학습 결과와 비교할 때는 학습 quantization 차이도 존재한다.
실험 간 비교는 새로 학습한 2단계 control을 기준으로 한다.

## 체크포인트와 bitstream

세 실험은 같은 `lambda=0.0256` main Split 10k checkpoint에서 weights-only로
시작한다. Optimizer와 step은 새로 시작하고 GlobalSplat은 고정한다.
활성 codec 파라미터와 MSH는 학습한다. 변환 시:

- 기존 spatial kernel 3을 kernel 5 중앙에 복사하고 새 ±2 weight는 0으로 둔다.
- B의 entropy model은 기존 even model을 복사하고 B predictor는 0으로 시작한다.
- D의 entropy model과 predictor는 기존 odd model과 predictor를 복사한다.
- Score context 밖의 가중치·buffer는 그대로 가져온다. MSH는 초기화하지 않는다.

변환은 `--allow-score-path-conversion`을 명시한 weights-only 학습에서만
허용한다. Eval과 optimizer-state resume는 저장된 설정과 일치해야 한다.
새 설정은 `feature_codec_config`에 저장하고, stage 수와 kernel은 bitstream
flags에도 기록한다. 호환되지 않는 decoder는 복호화 전에 거부한다.
3·4단계 entropy string은 stage 순서, 그 안에서는 channel slice 순서로 저장한다.
8개 미만 token인 비어 있지 않은 단계는 native entropy coder에만 0을 padding하고
복원 직후 잘라낸다. Padding은 token context와 reconstruction에 포함되지 않는다.
빈 단계는 빈 string으로 저장한다. 일반적인 4096-token scene에는 이 padding이 없다.

## v11 실행

서버 repository에서 실행한다. Train은 normal GPU 4개, effective batch 8
(`4 × batch 2 × accumulation 1`), lr `1e-4`, 추가 50k optimizer step,
35k/45k LR milestone, 5k checkpoint 간격이다. λ 기본값은 `0.0256`이다.
Eval은 normal GPU 1개로 전체 eval scene에 actual bitstream 평가를 수행한다.

먼저 3단계만 학습하고, 성공 후 자동 eval:

```bash
STAGE_TASKS=1 bash scripts/slurm/submit_nfcgs_score_stages_v11.sh
```

2·3·4단계를 순차 학습하고 전체 성공 후 자동 eval:

```bash
bash scripts/slurm/submit_nfcgs_score_stages_v11.sh
```

Array task `0=2단계`, `1=3단계`, `2=4단계`다. 기본 `0-2%1`은 train job을
한 번에 하나만 실행한다. Eval array에는 같은 task 선택과 `afterok:<train array>`
dependency를 건다. 모든 선택된 train task가 성공해야 eval이 시작된다.
Job을 제출한 뒤 학습 array ID를 출력한다. Eval 추가 제출이 실패하면 이미
제출된 학습을 중복 제출하지 말고 아래 명령으로 eval만 다시 제출한다.

```bash
mkdir -p logs/slurm
sbatch --array=1 --dependency=afterok:<TRAIN_ID> \
  --export=ALL,TRAIN_ARRAY_JOB_ID=<TRAIN_ID> \
  scripts/slurm/eval_nfcgs_score_stages_v11.slurm
```

`--array`는 학습 때 선택한 task와 맞춘다. λ 또는 MAX_STEPS를 변경했다면
같은 값을 eval에도 export한다. 예를 들어 짧은 제출 확인은 다음처럼 설정한다.

```bash
STAGE_TASKS=1 MAX_STEPS=20 CHECKPOINT_EVERY=10 \
  bash scripts/slurm/submit_nfcgs_score_stages_v11.sh
```

위 짧은 실행의 eval도 전체 scene을 평가한다. 실제 학습 제출 전 각 Slurm
파일을 LF, BOM 없는 UTF-8로 동기화한다. Submit helper는 파일 존재와 첫 줄을
검사한다. 서버에서 이 문서 작성 과정에 job을 제출하지는 않았다.

## 저장 위치와 평가

`<TAG>`는 기본 `0p0256`, `<ID>`는 training array ID다.

```text
outputs/nfcgs_score_stages/lambda<TAG>/job_<ID>/
  stages2|stages3|stages4/
    checkpoints/nfcgs_main/version_0/step000050000.ckpt
  stages2_eval_all|stages3_eval_all|stages4_eval_all/nfcgs_main/
    scores_all_avg.json
    actual_rate_per_scene.json
    score_stage_summary.json
    scores_psnr_all.json
    scores_ssim_all.json
    scores_lpips_all.json
    pipeline_timing.json
```

`actual_rate_per_scene.json`에 각 단계의 raw entropy bytes와 score wrapper bytes를
별도로 저장한다. `score_stage_summary.json`에는 단계 순서, 단계별 합계·평균,
wrapper 평균을 저장한다. Header/length overhead는 단계 raw bytes에 포함되지
않는다. 최종 rate 비교에는 전체 actual bytes/bpga를 사용한다.
3단계 B 절감과 4단계 D 절감을 보려면 기존 단계의 대응 집합 크기를 고려한다.
2단계 even/odd와 세분한 한 단계의 bytes 자체를 직접 비교하면 안 된다.
전체 PSNR/SSIM/LPIPS, total bytes, entropy decode 시간도 함께 비교한다.

CPU 검증은 기본 Full 경로와 archived 구현의 일치, 단계별 참조 mask,
±2 이웃 활용, STE 복원값과 실제 복호값의 일치 및 gradient, 홀수/짧은 토큰열,
MSH 보존 warm start, 독립 decoder 및 BF16 sender, checkpoint·bitstream 불일치
거부를 포함한다. GPU DDP 학습과 실제 데이터셋 성능은 서버 실행 후 확인한다.

# 로그 확인 — 2026-10-20 21:59 KST

19:40과21:59에 내려받은 로그를 합쳐 확인했다. 실시간 Slurm 조회는 아니다. 직전 통합 정리의16:47 snapshot 이후 완료된 것은 **L3 마지막 OFF 점과 Score736 pilot 평가**다.

## 1. 현재 상태

기존26개 학습 묶음에서 **25개는 최종50k 평가까지 완료**, **1개는 학습 진행 로그**가 있다. 완료 평가25개는 **full-test6991 21개 + train-derived holdout128 4개**다.

| 작업 | Train / Eval | 최신 근거 |
|---|---|---|
| None+L3 OFF12, λ=.0512 | 원래456951 → 복구458782 /457098 | 복구 학습50k 종료 footer와6991장 최종 평가 완료 |
| Score736-only, λ=.0128 |457828 /457829| 최종50k checkpoint를 검증하는 기존 runner의 holdout128 평가 완료 |
| H+D4 OFF16, λ=.0128 | 원래457099 → 복구458783 /457100 | **48,946/50,000, 97.892%**; 최종 평가 시작 로그 없음 |

736의 다운로드된 train stdout/err 자체는 아직47,614 step 기록에 머물러 있다. 새 eval은 최종50k checkpoint와 설정·공통 초기화 provenance를 검사하는 기존 runner를 통과하고128장 지표를 출력했다. 따라서736을 계속 학습 중으로 세지 않되, 학습 완료의 근거를 새 평가와 기존 runner 검증으로 기록한다. 원본 checkpoint를 로컬에서 직접 열었다는 뜻은 아니다.

진행 step은 마지막 완전한 progress에서 추정한 값이다. H+D4 OFF16의 현재 queue 상태나 남은 시간을 확정하지 않는다.

앞서 준비한 신규 후속5개 job의 접수 state와 `slurm-gs-followup20-*` 로그는 이번 다운로드에서 발견하지 못했다. 위26개 집계와 구분하며, 서버에서 미접수라고 단정하지 않는다.

## 2. 새로 완료된 Score736 결과

네 구조 모두 λ=.0128, 같은 설계의 train-derived holdout128, fresh 공통 초기화,50k 기준이다. 전체 test6991장 결과가 아니다.

| 구조 | PSNR | SSIM | LPIPS | 실제 KiB/scene | Eval |
|---|---:|---:|---:|---:|---|
| Score56-only |21.7889|0.6653|0.3295|65.53055|457742|
| MSH-only |20.6696|0.6123|0.3700|14.48434|457744|
| two-path56 |23.0067|0.7157|0.2796|56.73486|457746|
| **Score736-only** |**21.7001**|**0.6603**|**0.3315**|**67.81793**|**457829**|

Score736-only는 Score56-only보다 PSNR−0.0888dB, SSIM−0.0050, LPIPS+0.0020이며 실제 bytes도3.4906% 많다. **이번 한 점에서는 Score56-only에 지배된다.**

Two-path56은 Score736-only보다 PSNR+1.3066dB, SSIM+0.0554, LPIPS−0.0519이며 bytes16.3424% 적다. Full-width 단일 Score 경로로 확장하는 것이 이번 초기화·λ·학습 budget에서 이득을 주지는 않았다. Rank56이 모든 조건에서 최적이라는 결론이나736 구조 전체의 RD 열세로 확대하지 않는다.

MSH-only는 여전히14.48KiB의 저율 점이다. Score736보다 약4.68배 작고 two-path보다 약3.92배 작아서, 이 결과로 MSH-only를 탈락시킬 수 없다.

736 eval의 logged codec encode/decode 평균은597.4/348.0ms다. Warmup5를 제외한123개 호출 평균이며, 같은 GPU 할당에서 통제한 profile이 아니므로 다른 구조와의 정밀한 시간 비교는 별도 profile로 수행한다.

## 3. 새로 완료된 L3 마지막 점

| 설정 | λ | PSNR | SSIM | LPIPS | B/scene | KiB/scene |
|---|---:|---:|---:|---:|---:|---:|
| None+L3 OFF12 |.0512|22.9042|0.7114|0.2980|30872.300386|30.14873|

최종50k·C12/T8·BF16·actual bitstream·6991장 처리를 확인했고 Score+MSH y/z+mean+container 합계도 출력 정밀도 안에서 일치했다. 현재 job의 Traceback/OOM/ENOSPC 등 치명적 오류는 확인되지 않았다.

현행 주 seed의 양의λ RD는 **63→64점**,16계열이다. 별도λ0와 추가 seed2점은 유지한다. L3의 None/Hilbert×Context OFF/ON 네 계열이 모두 `.0032/.0064/.0128/.0256/.0512`의 **5점**, 총20점을 갖췄다.

새 .0512에서 Context ON의 같은λ 효과는 다음과 같다.

| 순서 | ON의 실제 bytes 감소 | ON−OFF PSNR |
|---|---:|---:|
| None |2.4611%|+0.0100dB|
| Hilbert |10.3781%|+0.1544dB|

같은λ 비교이며 동일 화질 절감률과 구분한다.

## 4. L3 네 곡선의 공통 구간 비교

네 곡선 모두에 공통인 **PSNR23.0360–23.7288dB**에서 PCHIP로 log-rate를 적분했다. 각 곡선의5점이 PSNR·rate에서 단조이며 외삽하지 않았다. 아래 모든 값에 같은 구간을 사용했다.

| 후보 / 기준 | BD-rate |
|---|---:|
| None ON / None OFF |−1.2648%|
| H ON / H OFF |−20.4325%|
| H OFF / None OFF |−14.7991%|
| H ON / None ON |−31.3393%|
| H ON / None OFF |−32.2077%|

이 고정 구간의 집계 RD 보간에서는 Hilbert 뒤의 Context 이득이 None에서보다 크다. 원본 stdout의 반올림된 평균 지표와 주 seed 하나의 곡선으로 계산한 값이다. Scene별/frame별 정합·paired CI·training-seed 유의성을 확인한 결과는 아니다. 기존 Hilbert+L3 .0256의 GPU4 학습 예외도 남아 있어 완전히 통일된 학습 조건의 순수 인과 효과로 표현하지 않는다.

## 5. 다음 판단

- **736 추가λ와 그룹 폭 변경의 우선순위를 낮춘다.** 수렴 문제가 드러나거나 다른 rate 구간을 검사할 구체적 이유가 생기면 다시 판단한다.
- **구조 RD 확장은 MSH-only와 two-path56에 우선 배정한다.** 처음에는 MSH-only λ=.0032, two-path56 λ=.0512로 rate 간격을 줄이는 방향을 탐색하고 그 결과로 다음λ를 고른다. 실제 중첩 구간 확보는 보장하지 않는다.
- 기존 H+D4 OFF16 최종 평가를 수집하면 기본 seed와 추가 seed의 H ON/OFF 비교가 완성된다.
- 이미 준비한 None+D4 OFF16·N01·native 전체 test·T06 렌더는 별도 목적의 독립 후속으로 유지한다. 이번 검토에서 새 학습을 접수하거나 기존 job/dependency를 변경하지 않았다.

## 6. 근거·검증 범위

- [최신 수치·상태](review.json), [RD64점 inventory와 구조4점](inventory.json), [구조 비교 계산](architecture_comparison.json).
- [L3 곡선·공통 구간 계산](../nfcgs_logs_2026-10-20_1940/l3_comparison.json), [19:40 검산](../nfcgs_logs_2026-10-20_1940/review.json).
- [재집계 코드](audit_snapshot.py), [직전 전체 실험 정리](../../NFCGS_EXPERIMENTS_MASTER_2026-10-20.md).

기존 기록 source를 재해시했고 이전 완료 평가70개의 원본 stdout는 동일했다. 새736 평가의128장·설정·payload 합계와 남은 복구 학습을 다시 확인했다. 기존3개 제출 state의 runtime 파일 해시도 일치했다. 새 평가 파일과 현재 progress에서 나온 결론이며 서버 `squeue/sacct` 실시간 확인은 아니다.

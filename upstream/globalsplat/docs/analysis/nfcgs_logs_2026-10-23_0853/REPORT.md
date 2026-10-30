# Fresh 로그와 checkpoint identity 복구 — 2026-10-23 08:53

근거는 workspace `slurm/`에08:48~08:53 내려받은 파일이다. 실시간 Slurm 조회·서버 job 제출·취소·checkpoint 변경은 수행하지 않았다.

## 진행

- Stage A15개 중10개가 `max_steps=60000`으로 종료됐다. 나머지5개는 약54.4k~59.8k다. 최근1,000 microbatch 기준 가장 늦은 λ0의 종료 예상은13:25경이며 전체 확인은 오늘13~14시가 적절하다.
- 임시 Stage B15개가 전부 접수됐다. 마지막 MSH는462078이다.9개의 시작과 약61.7k~67.0k 진행을 확인했고 나머지6개는 로컬 stdout이 없어 queue 상태를 단정하지 않는다. 다운로드한 학습 로그에 치명적 오류 표식은 없다.
- 시작된 B9개만 현재 속도를 적용하면100k 종료는10월24일07~12시경이다. 이 값은100k를 확정하는 경우이며 나머지6개는 별도다. 재개한 epoch의 progress 평균은 재개 전 batch index를 포함하므로 ETA에는 최근1,000 microbatch 시간 차이를 썼다.
- monitor460774는 평가74/90건 뒤 `checkpoint identity changed after evaluation`로 종료됐다. Gate 1을 확정할 근거는 아직 없다.

## 원인 재현과 한계

로컬 Lightning2.3.0에서 checkpoint 주기를 동일하게 유지하고 accumulate4로 stage를 재개했다. 원래 step4 파일이 재개 첫 microbatch에서 같은 global_step4로 다시 저장됐다. 작은 재현 모델의 가중치는 동일했지만 SHA-256과 batch progress가16→17로 바뀌었다. [재현 JSON](resume_boundary_reproduction.json).

실제 서버의 어떤60k 파일이 바뀌었는지, 원래 full state가 보존됐는지는 아직 서버 파일로 확인하지 못했다. 파일 변경을 단순 serialization 차이로 무시하거나 가중치·RNG·loader가 모두 같다고 처리하지 않는다. 현재100k 학습을 유지하되120k/150k로 돌아갈 원래60k full state의 보존 여부는 audit이 필요하다. 기존 연장 절차의 Stage A completion hash 검사는 유지한다.

## 수정과 실행

서버 반영 파일은 두 개다.

- `scripts/monitor_nfcgs_fresh_rd.py`: 바뀐 checkpoint에 이전 평가를 재사용하지 않는다. 이전 결과를 `.superseded_<시각>`으로 보존하고 현재 파일의 안정성을 확인한 뒤 새로 평가한다. metric 자체의 손상은 계속 오류이며 근거가 사라진 Gate 1 파일도 별도 보존해 무효화한다.
- `scripts/recover_nfcgs_monitor_identity_20261023.py`: 실패한 monitor만 확인·재접수한다. 학습30개 job 기록과 checkpoint는 유지한다. 학습·평가 runtime30개 hash는 그대로이며 monitor 코드 hash만 이력과 함께 갱신한다. 완료한 각 Stage A60k의 원 completion hash 및 같은 step backup을 감사 JSON으로 남기며 checkpoint를 자동 복원·변경하지 않는다.

```bash
cd /ceph_data/clue9986/tokencomp/upstream/globalsplat
conda activate globalsplat
python scripts/recover_nfcgs_monitor_identity_20261023.py
```

audit: `logs/slurm/fresh_rd/checkpoint_identity_audit_<시각>.json`. read-only 재점검은 위 명령에 `--audit-only`를 추가한다. hash 일치 backup 미발견은 외부 backup 부재까지 증명하지 않는다.

## 검증

관련8개 검사가 통과했다. checkpoint 변경 시 이전 평가 보존 및 재평가 요구, 다른 metric 손상 거부, active output 이동 거부, stale Gate 1 무효화, learner30개 기록 보존, backup의 read-only 탐색을 검증했다. 원본 보존 fork 검사와 기존 완료 평가 검사도 함께 통과했다.

```text
python -m pytest tests/test_fresh_rd_monitor_identity.py tests/test_fresh_rd_provisional.py::test_extension_forks_same_60k_and_preserves_all_original_outputs tests/test_fresh_rd.py::test_monitor_completed_fake_checkpoint -q --disable-warnings
8 passed, 3 warnings in 8.95s
```

앞선 조합에서는74개가 통과했고1개가 Windows 임시 state의 `os.replace` PermissionError로 실패했다. 해당 fork 검사는 위 재실행에서 통과했다. 학습 runtime 파일은 수정하지 않았다. [로그와 계산값](verification.json).

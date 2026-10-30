# NN-score 후속 상태 확인 — 2026-10-13

사용자는 수정 반영 후 .0256 재시도와 .0064/.0128 학습을 모두 제출했다고 확인했다. 아래 job ID와 자원 배정은 사용자가 제공한 출력에서 확인한 값이다. 서버를 직접 조회하지 않았다.

| 학습 | Job ID | 노드 | GPU 번호 | GPU 수 |
|---|---|---|---|---|
| .0256 전체 상태 resume | 450934 | ariel-v12 | 5,6 | 2 |
| .0064 새 학습 | 450937 | ariel-v8 | 1,4 | 2 |
| .0128 새 학습 | 450939 | ariel-v11 | 2,3 | 2 |

세 학습에 배정된 GPU는 총 6개다. 각 행의 CPU와 메모리 표시는 16 / 96 GiB다. 자원 배정 출력만으로 첫 optimizer step의 성공이나 현재 처리 속도를 확정하지 않는다. 후속 eval ID는 아직 제공되지 않았다.

제출 recipe는 .0256을 원래 NN-score step10k의 모델·optimizer·scheduler에서 총50k까지 이어 학습하고, 두 새 λ는 같은 FullSplit .0256 부모의 가중치에서 각각 새50k를 학습하는 방식이다. 세 실행의 실제 적용 설정은 새 로그로 확인해야 한다.

## 로컬 검증

- `test_quiet_upstream_logs`, `test_nn_order_backend`, `test_token_order_context`: 20개 통과, CUDA 검사 1개 생략. 테스트에 사용한 기본 Python의 PyTorch는 CPU 빌드다.
- 새 Python 프로세스에서 로그 필터 설치 → 첫 Numba import → JIT → exact kNN 실행이 통과했다.
- CPU의 기존 정렬과 최적화 정렬 사이에서 실제 bitstream bytes 동일성 및 checkpoint 호환 검증이 통과했다.
- `test_nn_score_resume.py`, `test_nn_score_resume_slurm.py`, `test_token_order_context_slurm.py`: 28개 테스트와 18개 subtest 통과. 제출 검증은 가짜 Slurm 명령을 사용했으며 서버 job을 제출하지 않았다.
- resume checkpoint 복사·검증, 새 state를 이용한 실패한 job 쌍 교체, `afterok` 의존성, 두 λ의 공통 부모·독립 초기화, 반복 실행의 ID 재사용, eval 제출 실패 후 재시도를 확인했다.

이번 후속 확인에서는 모델 코드와 제출 스크립트를 추가 수정하지 않았다. 기존 `main.py`의 `functools.wraps(_orig)` 수정이 로컬에 유지돼 있다.

## 다음 확인에 필요한 자료

현재 로컬 `slurm/` 로그는 19:28 다운로드본이다. 새 450934/450937/450939의 `.out`와 `.err`, resume retry state 및 `nn_score_rd_gpu_20261013/` 제출 state를 내려받으면 다음을 확인한다.

1. 450934가 step10k 상태를 복원한 뒤 추가 optimizer step을 완료했는지.
2. 450937/450939가 새 λ, weights-only 초기화, GPU2·유효배치8, `knn64/input`, minimal·MSH·legacy3/kernel5/STE 설정을 적용했는지.
3. 새 eval ID와 각 학습에 대한 `afterok` 연결이 맞는지.
4. GPU 정렬 benchmark의 제출·완료 여부와 거리·top-k·CPU 전달·greedy·전체 정렬 시간. 현재 GPU 시간의 새 실측값은 없다.

기존 450904의 실패와 재제출 절차는 [19:28 로그 검토](NFCGS_NN_SCORE_LOG_REVIEW_2026-10-13_1928.md)에 남겨둔다. 새 job의 정상 학습 확인 전까지 첫 batch 오류가 서버에서 해결됐다고 확정하지 않는다.

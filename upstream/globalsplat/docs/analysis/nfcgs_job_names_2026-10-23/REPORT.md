# Fresh RD job 이름 — 2026-10-23

새 접수 이름을 모델·λ·seed·단계/목표 step으로 구성한다.

| 이름 | 의미 |
|---|---|
| `gs-fr-H-on-l0p0128-s0-A60k` | Hilbert, context ON, λ=0.0128, seed index 0, Stage A 60k |
| `gs-fr-N-off-l0p0512-s0-B100k` | None, context OFF, λ=0.0512, seed index 0, Stage B 100k |
| `gs-fr-H-on-l0p0128-s0-test120k` | 해당 모델의 120k checkpoint test 평가 |
| `gs-fr-monitor-w1-A60k` | Wave 1 Stage A holdout monitor |
| `gs-fr-monitor-w1-final150k` | Wave 1 최종 150k holdout monitor |

`M-on`은 Morton, `H-L3`는 Hilbert legacy 3-stage, `S56`/`MSH`는 비교 모델이다. seed index를 표시하며 재시도는 원래 이름을 유지하고 job ID로 구분한다.

서버 반영 파일은 `scripts/submit_nfcgs_fresh_rd.py`, `scripts/recover_nfcgs_fresh_start_20261021.py`, `scripts/recover_nfcgs_monitor_identity_20261023.py`다. 이전 monitor identity 수정 파일은 앞선 안내대로 반영되어 있어야 한다. 학습·평가 runtime 30개 파일의 SHA는 다운로드한 state와 같음을 확인했다.

`python scripts/submit_nfcgs_fresh_rd.py --rename-active-jobs`는 state에 등록된 내 활성 job만 대상으로 원 제출 comment·WorkDir·Command를 먼저 확인한다. `scontrol update JobId=... JobName=...`만 실행하며 재제출, 취소, 우선순위·의존성 변경은 하지 않는다. 원 sbatch 명령은 이력으로 유지하고 새 표시 이름과 변경 이력을 별도로 기록한다. monitor 탐색은 예전 hash 이름과 새 이름을 모두 인식한다. 기존 identity 복구 도구는 submission controller만 바뀐 경우 활성 monitor를 유지하며 controller hash를 갱신할 수 있다.

서버 사용:

```bash
cd /ceph_data/clue9986/tokencomp/upstream/globalsplat
# identity 복구를 아직 반영하지 않았다면 앞선 안내의 monitor 코드도 업로드하고 먼저 실행한다.
# python scripts/recover_nfcgs_monitor_identity_20261023.py
python scripts/submit_nfcgs_fresh_rd.py --rename-active-jobs
squeue --me -o "%.18i %42j %.10T %.10M %R"
```

실제 서버 명령은 여기서 실행하지 않았다. 이름 변경과 출력 폭은 [scontrol](https://slurm.schedmd.com/scontrol.html), [squeue](https://slurm.schedmd.com/squeue.html)의 문서에 맞췄다.

검증: 2개 wave와 3개 최종 budget의 B/test 이름 210개가 중복되지 않았다. mock Slurm으로 이름 변경 시 ID·의존성·원 제출 명령 보존, 미등록 identity 거부, 새 monitor 이름 탐색, 활성 monitor의 controller 변경 시 재제출하지 않음을 확인했다. 기존 관련 검사 27개 중 최초 25개가 통과했고 2개는 Windows `atomic_json/os.replace`의 WinError 5였다. 재실행에서 extension 검사는 통과했으며 partial-submission 검사는 동일한 Windows 파일 권한 오류로 남았다. 실패는 이름·Slurm 동작 assertion이 아니며, 이 문제 때문에 실행 중인 실험의 runtime 파일을 바꾸지는 않았다. 중간의 첫 재실행은 새 basetemp 부모 디렉터리 누락으로 setup 오류가 났고 디렉터리 생성 후 재실행했다. [검증 기록](verification.json).

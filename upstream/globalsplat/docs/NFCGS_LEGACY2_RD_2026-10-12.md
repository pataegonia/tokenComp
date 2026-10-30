# Morton 기본 2단계의 RD 대조군 보완 — 2026-10-12

사용자가 선택한 추가 학습은 Morton 기본 even/odd 2단계, kernel5/STE의 λ=.0064와 .0128 두 개다. 같은 설정의 λ=.0256은 학습 442209_0, eval 442210_0에서 완료돼 있다. 기존 .0256의 PSNR은 23.4172 dB, scene당 평균 실제 bitstream은 32.671 KiB다. 세 point를 합쳐 기본 2단계의 RD 곡선을 완성한다.

| 정렬 | Context | 단계 | Kernel | λ | 학습 job name | eval job name |
|---|---|---:|---:|---:|---|---|
| Morton | legacy even/odd | 2 | 5 | .0064 | gs-oc-m2-0p0064 | gs-oc-e-m2-0p0064 |
| Morton | legacy even/odd | 2 | 5 | .0128 | gs-oc-m2-0p0128 | gs-oc-e-m2-0p0128 |

## 학습과 평가 조건

- 공통 부모는 기존 .0256 실험과 같은 FullSplit rank56, residual_on, λ=.0256, step10k checkpoint다. .0064도 이 부모를 사용한다.
- GlobalSplat 고정, codec 전체 학습, Score minimal/linear/STE/spatial-only, MSH 유지.
- 50k optimizer steps, LR1e-4, 35k/45k decay, checkpoint 5k마다.
- 학습당 normal GPU2, batch/GPU2, accumulation2, 유효배치8. 두 학습은 독립적으로 제출하며 동시에 학습하면 GPU4를 사용한다.
- 기존 .0256은 GPU4, batch/GPU2, accumulation1이었다. 유효배치8과 나머지 recipe를 유지하지만 GPU 수·누적 방식의 변경으로 완전히 같은 수치 경로를 보장하지는 않는다.
- 각 학습이 성공한 뒤 별도 `afterok` eval이 시작한다. Eval은 normal GPU1, 기존 C12/T8, 전체 처리 가능한 test scene, 실제 bitstream encode/decode다.
- 제출 총량은 학습2 + eval2 = Slurm job4개다. 노드 선택은 기존 order/context Slurm 파일 설정을 따른다.

## 서버에 반영할 파일과 실행

기존 `submit_nfcgs_order_context.sh`와 관련 설정·train/eval 파일이 최신 서버에 있다는 기준으로 새 파일 하나만 추가한다.

`scripts/slurm/submit_nfcgs_legacy2_rd.sh`

```bash
cd -P /ceph_data/clue9986/tokencomp/upstream/globalsplat
bash scripts/slurm/submit_nfcgs_legacy2_rd.sh --dry-run
bash scripts/slurm/submit_nfcgs_legacy2_rd.sh
```

부모가 이동했다면 `SOURCE_CHECKPOINT=/absolute/parent.ckpt`로 지정한다. 전용 helper는 정렬·단계·kernel·λ 두 개·학습량을 고정한다. 상태와 TSV는 `logs/slurm/order_context_legacy2_rd_20261012`에 저장한다. 같은 명령을 다시 실행하면 기록된 job ID를 재사용하고 빠진 eval만 제출할 수 있다. 부모·설정이 상태 파일과 달라지면 제출 전에 중단한다.

학습 출력은 `outputs/nfcgs_order_context/morton/legacy_stages2_k5_ste/lambda<tag>/job_<trainID>`이고, 최종 checkpoint는 `checkpoints/nfcgs_main/version_0/step000050000.ckpt`, eval 출력은 `eval_all`이다.

로컬에서 제출 스크립트를 준비했다. 서버에는 직접 제출하지 않았다.

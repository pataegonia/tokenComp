# 코드 정리 계획 (2026-10-28 논의, 10/30 기록)

10/28 대화에서 견적만 내고 문서로 남기지 않았던 계획을 기록한다. 아직 실행하지 않았다. 10/28 이후 추가된 실험(§13–14)에 맞춰 남길 범위를 고쳤다.

## 목표

minimal 설정을 코드 기본값으로 만들고, fresh 실험에서 쓰지 않는 모듈·스크립트·테스트를 지운다. 논문 실험에서 켜고 끄는 축은 남긴다.

## 남길 것 (10/30 기준)

| 축 | 값 | 쓰는 실험 |
|---|---|---|
| 정렬 | 없음 / Hilbert / Morton / NN-score / NN-xyz | N, H, M-ON, S-ON·TK-NN, X-ON |
| NN backend | knn, graph, graph_gemm | 옛 NN-score(knn), X-ON(graph), S-ON(graph_gemm) |
| Score context | ON / OFF | 2×2 주 실험 |
| context 스케줄 | D4 / L3 | 기본, H-L3·옛 NN-score |
| 코덱 경로 | Score+MSH / Score만(rank 56, 736) / MSH만 | 기본, S56·Score736 pilot, MSH |
| Top-K | `token_selection`, rate surrogate | TK fine-tune(§14 재설계) |
| decoder readout 학습 | `train_decoder_readouts`, `decoder_lr` | decoder finetune(§13.4), TK fine-tune |

10/28 견적과 달라진 점:

- NN-xyz와 graph·graph_gemm backend는 X-ON·S-ON·TK에 쓰이므로 남긴다. 10/28에는 NN-xyz가 삭제 대상이었다.
- Top-K와 decoder readout 학습도 남긴다.
- legacy·row backend는 knn과 같은 순서를 내는 느린 구현이라 지운다. 다만 NN이 아닌 정렬의 설정값과 checkpoint config에 `nn_order_backend=legacy`가 기록돼 있으므로, 설정 호환 처리와 함께 지운다.

## 지울 것 (10/28 견적)

| 항목 | 위치 |
|---|---|
| `--score-path full/minimal` 스위치와 변환 기능 | `run_nfcgs`, `checkpoint` |
| channel context | `score_context` |
| quarter2·single·옛 2단계 스케줄, kernel 3·7 | `score_context` |
| mean conditioner와 mean offset | `score_context`, `checkpoint` |
| score norm(`score_log_scale`) | `codec` |
| 비선형 변환(analysis/synthesis MLP) | `codec` |
| centering과 FP16 mean 전송 | `codec`, 비트스트림 |
| 옛 `score_entropy`(지금도 안 씀) | `codec`, `score_context` |
| noise 양자화 | `score_context` |
| `score_order_scale`(모든 실행에서 빈 값) | `config`, `codec` |
| 옛 실험 전용 스크립트(recover, resume, legacy, order_context, joint v10–12 등) | `scripts/` |

## 기본값 변경

지금 yaml 기본값은 옛 full 설정(2단계 스케줄, kernel 3, noise 양자화, Morton 정렬, 비선형·centering)이다([globalsplat_nfcgs_rank56.yaml](../config/model/globalsplat_nfcgs_rank56.yaml)). 실험은 매번 명령줄에서 설정을 덮어썼다. 기본값을 **Hilbert + D4 + kernel 5 + STE + minimal**로 바꿔 실제 설정과 맞춘다.

## 규모 (10/28 견적)

- 코드 본체 약 15개 파일, 500–700줄 삭제
- 테스트 58개 중 37개 수정 또는 삭제
- 옛 실험 전용 스크립트 py 약 14개, slurm/sh 약 47개 삭제
- 코드 작업 반나절–하루, 서버 검증 GPU 1–2시간

10/28 이후 스크립트가 늘었으므로 수치는 대략값이다.

## 주의할 점

1. **옛 checkpoint 로드**: Wave 1/2와 초기값 checkpoint(`two_path56.ckpt`)에는 안 쓰는 텐서가 있다. 지금 로딩은 키가 하나라도 다르면 실패하므로, 옛 키를 버리는 변환을 추가한다. weight만 불러오기는 지원하고 학습 재개는 지원하지 않는다.
2. **압축 결과 동일성**: 비트스트림 flag 값을 그대로 둔다. 같은 checkpoint를 새 코드로 압축했을 때 바이트와 PSNR이 완전히 같아야 한다. H-ON, H-OFF, N-ON, MSH, S56, H-L3, 옛 NN-score, X-ON, S-ON, TK에서 하나씩 확인한다.
3. **해시 잠금**: main 저장소는 Wave 2 추가 4개와 monitor 464634가 끝날 때(약 11/3)까지, `tokencomp_ext`는 자기 run(S-ON 469902 약 11/3–4, TK fine-tune 약 11/2–3)이 끝날 때까지 올리지 않는다. 정리 뒤에는 기존 state 파일의 해시 검증이 실패하므로, 정리 전 상태를 commit과 tag(예: `fresh-rd-v1`)로 남겨야 Wave 1/2를 재현할 수 있다. 지금은 commit하지 않은 변경이 많다.

## 진행 순서

1. 현재 상태 commit과 tag(사용자 확인 후)
2. 로컬 브랜치에서 본체 정리와 checkpoint 변환 추가, 로컬 pytest 통과
3. 옛 스크립트·테스트 삭제(삭제 권장, git 기록에는 남는다)
4. 모든 서버 run이 끝난 뒤 압축 결과 동일성 확인, 같으면 반영

## 정할 것

- **OFF1·L2·Q2 스케줄 코드**: 지우고 tag로 재현(권장)하거나, 부록 숫자를 정리된 코드로 다시 평가해야 하면 남긴다.
- **시기**: 모든 run이 끝나는 11/4 이후에야 서버 검증을 할 수 있다. 11/10 제출 전에 하려면 11/4–11/6 사이에 검증까지 끝내야 한다. 제출에 코드가 필요 없으면 제출 뒤에 하는 편이 안전하다.

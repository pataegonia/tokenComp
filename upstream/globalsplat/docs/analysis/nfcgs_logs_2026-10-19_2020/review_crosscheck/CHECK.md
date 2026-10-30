# 첨부 검토문 확인 — 2026년 10월 19일

첨부 검토문의 수치와 다음 판단 순서는 현재 확보한 자료와 일치한다. 누적 inventory에서 rate/PSNR 비교19개와 λ0 수치9개를 다시 계산했다. 그림 PNG도 확인했다. 이 확인은 같은20:20 로그 스냅샷에 대한 재검산이며, 새 서버 상태나 학습 결과를 확인한 것은 아니다.

## 확인된 내용

- 양의λ60점·16계열과 별도λ0 한 점이며, 모든 평가의 장면 수는6991이다. 장면별 ID·프레임 목록 일치까지 이 재검산에서 확인한 것은 아니다.
- Hilbert+D4와 None+D4의 다섯λ 비교, Hilbert+D4와 L3/Morton 비교, L3 matched ON/OFF12, OFF1/OFF4 비교가 원본 집계와 일치한다.
- λ0의 bytes14.6938배, PSNR+0.1769dB와 MSH y가 전체 bytes 증가의72.3501%를 차지한다는 계산이 맞다.
- 과거 .0128 GPU profile에서 계산한 D4의 encode/decode 시간 증가율8.71%/10.70%도 맞다. 이 측정은 같은 GPU·입력의 codec 구간 비교다.
- `context_only` family는 `use_residual=true`인 기본 순서 L3 2-path다. T07의 MSH 없는 단일 경로와 다르다.
- 최근26학습의15완료·8진행 로그·3시작 로그 없음, 미완료11개를 우선 수집하자는 판단과 일치한다.

## 표현을 구체화할 두 곳

**1. 역사적 non-comp와의 차이는 잠정 비교로 유지한다.**

24.7004dB는 [과거 ablation 표](../../../ABLATION_PROGRESS_2026-09-02.md)의 `upper` 값으로 확인된다. 이 값과의 차이0.6270/0.8039dB는 산술적으로 맞지만, 현재와 같은 장면·view·정밀도·모델 경계의 비압축 기준을 확인하기 전에는 확정된 현재 codec 손실로 쓰지 않는다. 현재 checkpoint의 projection 이후 비압축 우회 B1과 원본 backbone 기준 B0도 구분한다. 첨부문은 이미 이를 잠정 기준이라고 명시했으므로 이 한정을 보존하면 된다.

**2. “계수240.819KiB”는 내부 헤더를 포함한 Score 스트림 용량이다.**

D4의16 strings에는 Score wrapper152B(`24+8×16`)가 있다. λ0의 `actual_score_bytes/1024=240.818764KiB`이며, 순수 Score entropy payload는152B를 뺀 **240.670326KiB**다. 따라서 해당 문구는 “Score 스트림240.819KiB, 내부 헤더 포함”으로 적는 것이 정확하다. 전체 용량 증가와 MSH y의72.35% 기여 계산에는 변화가 없다.

## 해석 범위와 다음 작업

현재 결론은 공간 정렬의 이득과 Hilbert+D4의 후보 지위가 강화됐다는 것이다. D4 matched OFF16과 추가 seed는 최종 방법의 공간 Context 효과를 검증한다. D4 seed 쌍만으로 L3 대 D4 스케줄 차이의 반복성까지 검증되는 것은 아니다.

Context ON이 모든λ에서 PSNR을 높였다고 일반화하지 않는다. 예를 들어 기존 Hilbert+L3 .0064에서는 OFF12 대비 bytes가14.91% 작지만 PSNR은0.0136dB 낮다. 첨부문이 동시 우세를 확인한 .0032 한 점의 설명은 정확하다.

남은11개 학습·평가를 수집하면서 B0/B1 기준선, 경로별·양자화 우회 진단, 독립 Gaussian 수신 검증과 누락 profile을 보완한다. 구조4개의 결과가 없으므로 구조 우열이나 low-rank가 화질 갭의 원인이라는 결론은 아직 내리지 않는다. 이 확인 과정에서는 새 job을 제출하지 않았다.

## 보관한 자료

원래 `Documents/Codex/...`에 있던 검토 그림과 수치를 `upstream/globalsplat/` 안에 동일한 bytes로 복사했다.

- [원본 첨부 검토문](provided_review.txt)
- [RD 그림 PNG](rd-snapshot-2026-10-19.png)
- [RD 그림 PDF](rd-snapshot-2026-10-19.pdf)
- [재계산 비교 JSON](derived_comparisons.json)
- [검증 범위·파일 해시](verification.json)
- [원본 누적 inventory](../inventory.json)

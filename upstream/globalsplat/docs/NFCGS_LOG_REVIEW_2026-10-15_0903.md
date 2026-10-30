# NN-score·정렬/context ablation 로그 점검 — 2026-10-15 09:03

`slurm/`에 09:03경 내려받은 로그와 recovery state의 train/eval 15쌍을 확인했다. 아래는 다운로드된 로그의 마지막 기록이며 실시간 Slurm 상태가 아니다. 기존에 취소한 450977~450995는 현재 재제출 실험과 구분했다.

## 저장 공간 오류로 중단된 학습 3개

세 작업 모두 `ariel-v8`에서 체크포인트 저장 중 `OSError: [Errno 28] No space left on device`로 중단됐다. 호출 경로는 `ModelCheckpoint → trainer.save_checkpoint → _atomic_save → fsspec local write`다. 이후 rank 종료와 srun exit 137이 기록됐다. 로그가 가리키는 직접 원인은 저장 실패다.

| 설정 | λ | Train / eval | 마지막 약 optimizer step | 로그의 중단 시각 |
|---|---:|---|---:|---|
| Context only / quarter2 | 0.0064 | 451821 / 451822 | 10,000 | 2026-10-15 04:01:03 |
| Context only / quarter2 | 0.0256 | 451825 / 451826 | 5,000 | 2026-10-15 05:52:48 |
| Context only / dyadic4 | 0.0064 | 451827 / 451867 | 5,000 | 2026-10-15 07:54:02 |

ログだけでは対象ファイルシステムの容量不足・inode不足・quota・ノード固有のmount状態を区別できない。同じノードで3回発生しているため、マスターに加えて問題が出た計算ノードから保存先を確認する。保存途中の checkpoint が残った可能性があり、10k/5kに到達したことはその checkpoint の保存成功を意味しない。451821にはそれ以前の5k checkpointが残っている可能性があるが、ローカルにcheckpoint本体がなく検証していない。

まずサーバーで状態と対象パスを確認する。次の操作は読み取りのみ。

```bash
sacct -X -j 451821,451825,451827 --format=JobID,State,ExitCode,NodeList,End
df -hT /ceph_data/clue9986/tokencomp/upstream/globalsplat/outputs
df -i /ceph_data/clue9986/tokencomp/upstream/globalsplat/outputs
quota -s
```

`ariel-v8`に入れる場合は、同じ`df`と`quota`、`findmnt -T /ceph_data/clue9986/tokencomp/upstream/globalsplat/outputs`で計算ノード側も照合する。

既存 recovery helper は最新候補 checkpoint の破損を検出すると止まる。また、失敗した train に対する eval が依存待ちのまま残っている場合も停止する設計だ。空き容量を解決し、checkpointと後続evalの状態を確認してから復旧する。今回、job取消・再提出・checkpoint削除は実行していない。

## 完了した実際の bitstream eval 8件

以下の8学習はすべて `max_steps=50000` 到達を確認し、evalも `global_step=50000` のcheckpointをロードして最終metric・時間要約まで出力している。C12/T8、batch1、seed0、BF16、actual bitstream ON。評価6,991 scene、時間平均はwarmup5 sceneを除く6,986 scene。進捗表示の分母7,286は評価scene数として使用しない。

| 設定 | λ | Train / eval | PSNR dB | SSIM | LPIPS ↓ | KiB/scene ↓ | Codec encode / decode ms |
|---|---:|---|---:|---:|---:|---:|---:|
| NN-score + legacy3 | 0.0064 | 450937 / 450938 | 23.8274 | 0.7405 | 0.2607 | 45.012 | 261.6 / 85.1 |
| NN-score + legacy3 | 0.0128 | 450939 / 450940 | 23.7431 | 0.7375 | 0.2642 | 36.951 | 258.1 / 89.6 |
| NN-score + legacy3 | 0.0256 | 450934 / 450935 | 23.5447 | 0.7304 | 0.2737 | 29.307 | 258.4 / 87.6 |
| Context only / legacy3 | 0.0064 | 451809 / 451810 | 23.6369 | 0.7344 | 0.2726 | 55.882 | 146.9 / 85.7 |
| Context only / legacy3 | 0.0128 | 451811 / 451812 | 23.5665 | 0.7319 | 0.2756 | 46.525 | 145.7 / 84.8 |
| Context only / legacy3 | 0.0256 | 451813 / 451814 | 23.3521 | 0.7251 | 0.2825 | 38.264 | 146.4 / 84.6 |
| Hilbert only / context OFF | 0.0064 | 451815 / 451816 | 23.8130 | 0.7405 | 0.2629 | 55.110 | 152.4 / 84.6 |
| Hilbert only / context OFF | 0.0128 | 451817 / 451818 | 23.6831 | 0.7358 | 0.2686 | 44.807 | 149.5 / 83.3 |

NN-score의 sender는 305.4~309.3 ms/scene, receiver는 87.6~92.4 ms/scene이다. Context-only legacy3의 sender는 193.7~195.1 ms/scene, receiver는 87.2~88.2 ms/scene이다. Codec encode는 `model.compress_scene_tokens` 호출 전체를 측정하며, 정렬 함수만의 GPU benchmark 시간이 아니다.

## NN-scoreと既存のHilbert結果

同じλの既存50k evalと比べた差。負の容量変化はNN-scoreが小さいことを表す。BD-rateや同一PSNRへの補間ではなく、seed反復による有意差も未測定。

| λ | 比較元 | 全体容量変化 | PSNR変化 dB | encode ms: 比較元 → NN |
|---:|---|---:|---:|---:|
| 0.0064 | hilbert_legacy3 | -4.013% | +0.0280 | 162.6 → 261.6 |
| 0.0064 | hilbert_dyadic4 | -3.699% | +0.0104 | 165.4 → 261.6 |
| 0.0128 | hilbert_legacy3 | -3.443% | +0.0399 | 156.3 → 258.1 |
| 0.0128 | hilbert_dyadic4 | -0.613% | +0.0410 | 156.9 → 258.1 |
| 0.0256 | hilbert_legacy3 | -3.245% | +0.0397 | 152.8 → 258.4 |
| 0.0256 | hilbert_dyadic4 | -0.311% | +0.0274 | 161.0 → 258.4 |

NN-scoreは同じλのHilbert legacy3・dyadic4の両方に対し、今回の3点すべてでPSNRが高く実際の総bitstreamが小さい。一方、encode時間は増加している。別jobの壁時計測定なので、時間差をsorting手法だけの効果には帰属しない。特にdyadic4との容量差は.0128で約0.61%、.0256で約0.31%と小さい。

Context-only legacy3は、同じλの既存Hilbert+legacy3より全3点で容量が大きくPSNRが低い。Hilbert-onlyは完了した2点で容量が大きく、.0064のPSNRはHilbert+legacy3よりわずかに高く、.0128は低い。今回の測定範囲ではsortingとcontextの併用が圧縮効率に有利と読めるが、quarter2/dyadic4のcontext-onlyは未評価なのでその結論は保留する。Hilbert-onlyはlegacy3の3段階entropy splitを維持した対照実験であり、quarter2/dyadic4とentropy model数まで一致する対照ではない。

![RD comparison](../../../artifacts/nfcgs_rd_2026-10-15_0903.png)

線は測定点を結ぶだけで、未評価のHilbert-only .0256やquarter2/dyadic4は補完していない。

## 完了記録がない残り4学習

| 設定 | λ | Train / eval | 最後の約step / 50k | batch/s |
|---|---:|---|---:|---:|
| Hilbert only / context OFF | 0.0256 | 451819 / 451820 | 39,950 | 1.67 |
| Context only / quarter2 | 0.0128 | 451823 / 451824 | 26,294 | 1.68 |
| Context only / dyadic4 | 0.0128 | 451868 / 451869 | 4,502 | 1.55 |
| Context only / dyadic4 | 0.0256 | 451870 / 451871 | 4,502 | 1.55 |

上記4件には現在のダウンロード範囲でfatal errorはなく、学習進捗がある。実際の現在状態はサーバーで確認する。stepはepoch内batch・1epoch33,017 batch・accumulation2からの近似値で、progress barのpercentは全体学習進捗ではない。これらの後続evalログも未取得。

## 設定と検証

15学習すべてでGPU2・batch/GPU2=2・accumulation2・有効batch8、C24/T12、subset consistency ON、GlobalSplat固定、codec scope all、MSH維持、50k・checkpoint5k間隔を照合した。Scoreのcentering/norm/mean conditioner/channel contextはOFF、linear/kernel5/STE。none+legacy3/quarter2/dyadic4、Hilbert+context OFF、NN-score+knn64/inputの設定が各armと一致する。8件のevalはcheckpoint設定も照合し、Score/MSH y/MSH z/mean/container各payloadの和が総bytesに一致することも確認した。

構造化データ: [監査JSON](../../../artifacts/nfcgs_log_review_2026-10-15_0903.json)、[比較JSON](../../../artifacts/nfcgs_rd_comparison_2026-10-15_0903.json)。

再実行: `python -X utf8 artifacts/review_nfcgs_logs_2026-10-14.py --snapshot 2026-10-15_0903` → `python -X utf8 artifacts/report_nfcgs_logs_2026-10-15.py`。

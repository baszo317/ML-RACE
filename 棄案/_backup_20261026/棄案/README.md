# 棄案：目前沒有在用的檔案

2026-10-23 整理。v2 和整夜實驗都不會讀這裡的東西。

> 2026-10-25：v9（live 正式版）用到的 `v3\features_v3.py` 已複製一份到 `v9\features_v3.py`（內容完全相同），v9 每天的流程不再讀這個資料夾。但 `v9\research\` 的研究程式仍會讀 `棄案\v3`，要重算 v9 的研究結果時請保留。

所有檔案都保留原本的相對路徑，例如 `棄案\data\pit\` 原本是 `data\pit\`。要拿回來，移回同樣位置即可，例如在 PowerShell：

```
Move-Item "D:\ML RACE\棄案\data\pit" "D:\ML RACE\data\pit"
```

## 搬過來的東西

| 原路徑 | 內容 | 為什麼不用了 |
|---|---|---|
| `feature_analysis\` | 10/22 最早的 feature IC 分析（Excel、各股 IC） | 結論已整理成 `transformer\comp_features.py` 的 B 組 67 個 feature |
| `best_features\` | 含外部資料的 123 個候選 feature、F 組、LightGBM 比較、`train_best.csv` | F 組（加外部資料）沒有比 B 組好（見 `transformer\results\summary.txt`），v2 只用 B 組 |
| `data\raw\` | FinMind、TWSE 原始下載（9,735 個檔） | 只有 `best_features` 和舊 pipeline 用到 |
| `data\pit\` | 外部資料 point-in-time 表：費半、匯率、法人總額、營收、本益比、融資融券等，到 2026-07-31 | 同上。之後若要做外部資料的 timing feature，可以從這裡拿歷史資料 |
| `data\out\`、`data\model\`、`data\reports\` | 第 4–6 週舊 pipeline（預測 5 日報酬、150 支 universe）的 panel、模型比較、資料稽核 | 題目設定不同（H = 5），和現在的盤中預測無關 |
| `data\ref\stock_master.*`、`rules.json`、`corporate_actions.csv` | 舊 pipeline 的股票主檔、規則設定，以及除權息表的 CSV 副本 | v2 只讀 `corporate_actions.parquet`、`calendar.parquet`、`exdiv_upcoming.csv`，這三個還在 `data\ref` |
| `transformer\models\` | v1 static 模型（凍結在 2026-05-29） | static leaderboard 已上傳；v2 有自己的 `models_v2_static` |
| `transformer\models_live\` | v1 每日重訓的模型 | 已由 v2 取代 |
| `transformer\submission_*.csv`（4 個） | v1 static leaderboard 的上傳檔、機率、資料來源 | 已上傳的歷史檔 |
| `transformer\train_and_predict.py`、`walkforward_nn.py`、`train_best.py` | v1 的訓練、模型比較、B/F 組比較程式 | v2 不用。結果記錄在 `transformer\README.md` 和 `transformer\results\summary.txt` |
| `transformer\results\` 的 6 個 `wf_*.pkl` 和 `train_best.log` | v1 實驗（B+margin、B+breadth、F、recency、quarterly、Transformer F）的 walk-forward 預測 | 結論已寫在 `summary.txt` |
| `v3\`（2026-10-23 新增後移入） | v2 + 選股觀點加入 5 個新 feature（gap_bias_60/250、resid_vol_60、dow_bias_e、hi52），含篩選程式、walk-forward 比較和結果 | walk-forward 沒有比 v2 好（見 `v3\README_v3.md` 第三節）。程式用相對路徑，要用時整個資料夾移回 `D:\ML RACE\v3\` |

注意：`best_features\` 和 `transformer\` 裡被搬過來的程式，用的是 `ROOT\data\raw`、`ROOT\data\pit` 這類原路徑，放在棄案裡不能直接執行，要先移回原位。

## 沒有搬的（仍在使用）

- `data\market`、`data\global`、`data\info.json`：競賽資料，`update_data.py` 每天更新。
- `data\ref` 剩下的 3 個檔：除權息規則與交易日曆。
- `data_backup_original\`：`update_data.py` 第一次執行時自動做的原始資料備份。如果搬走，它會用「已經更新過的資料」重做一份，所以留在原位。
- `downloads\`：`update_data.py` 從這裡找每天的 zip。
- `transformer\` 剩下的檔案：
  - `comp_features.py`、`tfmodel.py`、`update_data.py`、`final_feature_lists.json`：v2 用到的共用程式。
  - `results\wf_transformer_B.pkl`：v2 的整夜實驗會讀。
  - `results\wf_mlp_B.pkl`：`v3\research\screen_candidates.py` 會讀（v3 移回原位後）。

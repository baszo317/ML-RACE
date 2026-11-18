# ML RACE：預測台股「收盤價 > 開盤價」

最後整理：2026-10-25。

## 現在用哪一版

- **live（每天上傳）：v9**。晚上跑 `v9\evening.bat`，早上跑 `v9\daily.bat`，上傳 `v9\out\submission_v9_MMDD.csv`（見下面「每天的流程」）。
- **static leaderboard**：`v9\out\submission_v9_static_w025.csv`（10/25 決定；為什麼和風險見 `v9\README_v9.md` 8.1）。w = 0.5 的 `v9\out\submission_v9_static.csv` 保留。
- 緊急備案：`v5\daily.bat`。注意 v5 從沒在這台電腦跑過，第一次會訓練 Transformer（約 5–10 分鐘）。

## 每天的流程（課程 zip 約 17:00 收盤後釋出，隔天 9:00 前上傳）

### 晚上（17:00 以後）

1. 從課程網站下載當天的 zip，放進 `downloads\`（留在自己的「下載」資料夾也可以，程式會找最新的 `data*.zip`）。
2. 雙擊 `v9\evening.bat`：合併 zip、抓除權息預告、訓練當天的新模型（筆電約 7 分鐘）、寫出**預備檔** `v9\out\submission_v9_MMDD_prelim.csv`，最後備份（`v9\backup_v9.py`）。
   - 預備檔只少了隔天的夜盤（05:00 才收盤），其他和正式檔相同。網站如果允許重新上傳，可以先傳預備檔當保險。

### 早上（05:00 以後、9:00 以前）

3. 雙擊 `v9\daily.bat`：抓夜盤和除權息；模型昨晚已經訓練好，只做預測（約 2 分鐘），寫出 `v9\out\submission_v9_MMDD.csv`，並列出和預備檔差幾支。
4. 上傳 `v9\out\submission_v9_MMDD.csv`。
5. 截 daily board 和 accumulated board 的圖，存進 `leaderboard\`（說明在 `leaderboard\README.md`）。

### 出狀況時

| 狀況 | 怎麼做 |
|---|---|
| 時間不夠 | 上傳昨晚的 `v9\out\submission_v9_MMDD_prelim.csv` |
| 只想用現有模型預測（約 1 分鐘） | 在 `v9\` 執行 `python train_and_predict_v9.py --dates auto --retrain none` |
| 出現「is in the past」 | 用到的是舊 zip：下載最新的 zip 再跑 |
| update_data.py 顯示 CHECK THESE，看過後確認是誤報 | 在 cmd 執行 `"D:\ML RACE\v9\daily.bat" accept`（晚上用 `evening.bat accept`） |
| 其他 FATAL，看過訊息後確認是誤報 | 在 `v9\` 執行 `python train_and_predict_v9.py --dates auto --force` |
| 晚上忘了跑 | 早上照常跑 `daily.bat`，它會自己訓練（約 7 分鐘） |

## 版本

| 版本 | 資料夾 | 做法 | 狀態 |
|---|---|---|---|
| v1 | `transformer\` | 67 個 B 組 feature；MLP 與 Transformer 機率平均，p > 0.5 猜 1 | 停用；共用程式放在這裡 |
| v2 | `v2\` | 大盤觀點與殘差選股觀點分開，score = 0.5 × (pbar − 0.5) + (p_e − 當天平均)；除權息日猜 0 | 停用（之後各版的基礎） |
| v3 | `棄案\v3\` | v2 + 5 個新選股 feature | 未採用；它的 `features_v3.py` 已複製到 `v9\` 給 v9 用 |
| v4 | `v4\` | v2 + rolling pool（每天只用到前一個交易日的資料訓練） | 舊的 static 上傳檔（15.36 bp/日） |
| v5 | `v5\` | 預測同 v4，加上防呆 `guards.py` 和監控 | 緊急備案；v9 用到它的共用程式 |
| v6 | `v6\` | static 專用：`compare_static.py`、排行榜 leak 對照 | 研究與工具 |
| v7 | `v7\` | live 研究：擴充股票池、夜盤、w_T 風險設定 | 研究；v9 用到它的夜盤抓取 |
| v8 | `v8\` | 大盤觀點診斷、架構比較、beta 對沖 | 研究；v9 用到 `hedge_beta.py`、`monitor_market.py` |
| v9 | `v9\` | 11 個 val t > 0 的改法合成一個模型 | **live 正式版**（10/25 上線） |

各版本的細節、證據和數學證明在各自資料夾的 `README_vN.md`（v1 是 `transformer\README.md`，v3 是 `棄案\v3\README_v3.md`）。10/25 上線前做的修改在 `v9\README_v9.md` 第十節。

## 資料夾

| 資料夾或檔案 | 內容 | 注意 |
|---|---|---|
| `data\` | 競賽資料（`market\`、`global\`、`info.json`）和 `ref\`（交易日曆、除權息） | 每天由 `update_data.py` 更新 |
| `data\ref\exdiv_fetch_log.csv` | 每次抓除權息預告的成功或失敗紀錄 | 防呆會讀 |
| `data_backup_original\` | `update_data.py` 第一次執行時做的原始資料備份 | 不要移動 |
| `downloads\` | 每天的 zip；`archive\` 留著每一個用過的 zip（`data_<預測日>.zip`） | |
| `leaderboard\` | live 排行榜的截圖和表 | 每天記，11/23 定版要用 |
| `v9\` | live 正式版：程式、模型池 `models_v9\`、輸出 `out\`、`data\panel_ext.parquet`、`features_v3.py` | |
| `v7\data\tx_daily.csv` | 夜盤資料，`daily.bat` 每天更新；`v7\data\archive\` 每天一份快照 | v9 在用，不要刪 |
| `_backup_20261025\` | 10/25 修改前的原始檔 | 確認新版沒問題後可以刪 |
| `棄案\` | 沒在用的舊檔 | v9 每天的流程不讀；`v9\research\` 仍會讀 `棄案\v3` |
| `docs\` | 課程投影片 | |
| `sample_submission.csv` | 上傳格式範本（v9 用它的 50 支股票代號和順序） | 不要移動 |

## v9 每天的流程用到的檔（不要刪除、移動或改名）

- `v9\`：`rolling9.py`、`train_and_predict_v9.py`、`monitor_v9.py`、`features_v3.py`、`data\panel_ext.parquet`、`models_v9\`
- `v5\`：`ens_lib.py`、`rolling.py`、`guards.py`、`features_extra.py`、`fetch_exdiv.py`
- `transformer\`：`comp_features.py`、`tfmodel.py`、`update_data.py`、`final_feature_lists.json`
- `v7\`：`fetch\fetch_night_session.py`、`data\tx_daily.csv`
- `v8\`：`hedge_beta.py`、`monitor_market.py`
- `data\`、`sample_submission.csv`

## 備份

`evening.bat` 最後會執行 `v9\backup_v9.py`，把整個資料夾（不含 `棄案` 和研究用的 cache）複製到 `%USERPROFILE%\ML_RACE_backup`（C 槽）。只複製新的或改過的檔，不會刪檔。想備份到隨身碟或私人雲端：`python backup_v9.py E:\ML_RACE_backup`。競賽資料不能分享給沒修課的人，備份請放在私人的地方。

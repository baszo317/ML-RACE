# ML RACE：預測台股「收盤價 > 開盤價」

最後整理：2026-10-24。

## 版本

| 版本 | 資料夾 | 做法 | 狀態 |
|---|---|---|---|
| v1 | `transformer\` | 67 個 B 組 feature；MLP 與 Transformer 機率平均，p > 0.5 猜 1 | 停用。static leaderboard 用過；v2、v4 用到的共用程式還放在這裡 |
| v2 | `v2\` | 大盤觀點與殘差選股觀點分開，score = 0.5 × (pbar − 0.5) + (p_e − 當天平均)；除權息日猜 0 | 已上傳 static（用資料裡的 label 算，6–9 月 AR 11.27 bp/day）。live 可繼續用，作為 v4 的備援 |
| v3 | `棄案\v3\` | v2 + 選股觀點加入 5 個新 feature（gap_bias_60/250、resid_vol_60、dow_bias_e、hi52） | 未採用：walk-forward 沒有比 v2 好（w_T = 0.5 時 val 12.59 對 12.54，t = 0.1；只選股時 hold 變差，t = −2.1） |
| v4 | `v4\` | v2 的規則 + rolling pool：每一天只用「到前一個交易日為止」的資料訓練的模型（課程新規定）；除權息日補到 2026/10 | **static 上傳版**；live 的備援 |
| v5 | `v5\` | 預測和 v4 完全相同；加上每天早上的防呆（`guards.py`）、成績拆解監控（`monitor_live.py`）、事先寫好的決策規則。feature 去留、以 v4 為基礎的 ablation、三個風險的診斷都在 README_v5 | **live 正式版**（11/23 前定版） |
| v6 | `v6\` | **第一階段（static）專用**，和 live 分開：`compare_static.py`（成對比較、95% 區間、MDE、試 N 個留最好的偏差）、排行榜分數的 leak 對照 | 研究與工具；static 上傳檔仍是 v4 |
| v7 | `v7\` | **第二階段（live）專用**：擴充股票池（H1）、當天 gap 上限、w_T 風險設定、除權息前後日、夜盤（H2）。都沒過門檻，預測沒改 | 研究；live 仍用 v5 |

各版本的細節、證據和數學證明：

- v1：`transformer\README.md`
- v2：`v2\README_v2.md`
- v3（未採用）：`棄案\v3\README_v3.md`
- v4：`v4\README_v4.md`
- v5：`v5\README_v5.md`（最前面有簡短說明）
- v6（static 專用）：`v6\README_v6.md`
- v7（live 專用）：`v7\README_v7.md`

## Static leaderboard

上傳 `v4\out\submission_v4_static.csv`（6–9 月 AR 15.36 bp/day，v2 是 11.27；細節與誠實說明見 `v4\README_v4.md`）。模型已在雲端全部算好（`v4\models_v4_static\`），`v4\run_static.bat` 只會重新組合，約 1 分鐘。

## 每天早上（9:00 前）

1. 從課程網站下載當天的 zip，放進 `downloads\`。
2. 雙擊 `v5\daily.bat`。它會併入資料、更新除權息預告、檢查輸入（防呆）、把新模型加入 pool、預測、檢查輸出，最後用 `monitor_live.py` 列出已實現成績。
   - 第一次執行會把 v2 的正式模型複製進 pool（和 v4 相同）。
   - 之後每天約 1 分鐘（MLP）加 5–15 分鐘（Transformer）。
   - 防呆出現 FATAL 時不會寫上傳檔，畫面和 `v5\out\checks_v5_MMDD.txt` 會寫原因；確認是誤報後才用 `python train_and_predict_v5.py --dates auto --force`。
   - 時間不夠時：`python train_and_predict_v5.py --dates auto --retrain none`，只預測，幾秒完成。
3. 上傳 `v5\out\submission_v5_MMDD.csv`。

備援：`v4\daily.bat`（同樣的預測，沒有防呆）、`v2\daily.bat`。

## 資料夾

| 資料夾或檔案 | 內容 | 誰會用 |
|---|---|---|
| `data\` | 競賽資料（`market\`、`global\`、`info.json`），以及 `ref\`（交易日曆、除權息） | 全部版本 |
| `data\ref\exdiv_fetch_log.csv` | 每次執行 `v5\fetch_exdiv.py` 記一行成功或失敗；`v5\guards.py` 用它確認今天的除權息預告有抓到 | v5 |
| `data\ref\exdiv_history.csv` | TWSE 除權除息計算結果表（TWT49U），2026/08 以後的除權息日；`v4\fetch_exdiv.py --history` 產生 | v4 |
| `data_backup_original\` | `update_data.py` 第一次執行時自動做的原始資料備份 | `update_data.py`，不要移動 |
| `downloads\` | 每天下載的 zip | `update_data.py` |
| `transformer\` | v1，以及共用的 `comp_features.py`、`tfmodel.py`、`update_data.py`、`final_feature_lists.json` | v2、v4 |
| `v2\` | v2 的程式、模型（`models_v2\`、`models_v2_static\`）、輸出（`out\`） | v4 第一次執行時會複製它的模型 |
| `v4\` | static 上傳版的程式、模型（`models_v4_static\`）、輸出（`out\`）；live 備援 | |
| `v5\` | live 正式版的程式、模型 pool（`models_v5\`，第一次執行時建立）、輸出（`out\`）、研究程式與結果（`research\`） | |
| `docs\` | 課程投影片 | |
| `棄案\` | 沒在用的檔案（含未採用的 v3）；保留原路徑，附 README 說明怎麼還原 | 沒有程式會讀 |
| `sample_submission.csv` | 上傳格式範本 | 程式會讀，不要移動 |

## 版本之間的依賴（不要刪除或改名）

- v2 和 v4 都會 import `transformer\` 的 `comp_features.py`、`tfmodel.py`、`final_feature_lists.json`，並執行 `update_data.py`。
- v4 第一次執行時，會從 `v2\models_v2_static\`（static）和 `v2\models_v2\`（live）複製模型；複製後就不再需要它們。
- v5 第一次執行時，會從 `v2\models_v2\` 複製模型（和 v4 相同）。v5 的研究程式會讀 `v2\results_v2\wf\`、`transformer\results\`、`棄案\transformer\results\` 和 `v5\research\cache\`。
- v2 的整夜實驗會讀 `transformer\results\wf_transformer_B.pkl`。

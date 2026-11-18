# 棄案：目前沒有在用的檔案

## GitHub 版（2026-11-18）

這個 repo 另外把下面這些「只有研究、每天的程式不讀」的資料夾搬進棄案，放在同樣的相對位置（例如 `v13\` → `棄案\v13\`）。**筆電上的 `D:\ML RACE` 沒有搬**，這些資料夾在筆電上還在最上層。

| 原路徑 | 內容 | 結果 | 還有誰會讀 |
|---|---|---|---|
| `v12\` | ranking objective 的實驗 | 沒通過 | `棄案\v14` 的 E5 讀它的 cache（沒有 cache 時 E5 跳過） |
| `v13\` | LSTM、1D CNN、F-LOAM | 沒通過 | 同上 |
| `v14\` | 量化方法對照表和五組實驗 | 沒通過 | |
| `v15\` | 去噪方法、Random Forest | 沒通過 | |
| `v16\` | Kalman filter、smoother | 沒通過 | |
| `overfit\` | overfit 檢查 | 只有研究 | |
| `reg_svm\` | regularization、SVM | 沒通過 | `棄案\v14` 的 E5 |
| `race ver\race_mkt\` | 任何數量 0/1 的大盤觀點、期貨籌碼 | 沒通過 | |
| `race ver\race_news\` | 鉅亨網新聞 feature | 沒通過 | |
| `race ver\race_agg\` | aggregation 方法 | 沒通過 | |

- 這些資料夾的程式用自己的位置往上找專案根目錄，放在棄案裡不能直接執行；要重跑時整個資料夾移回最上層。
- `v10\` 沒有搬：`v11\research\v11b_lib.py` 會 import `v10\research`（W0 的 `custom_v11.bat` 第一步用到）。
- `race ver\race_std\` 沒有搬：race 版每天的程式（`race ver\race_live\`）會 import 它的 `research\`，也讀 `cache\features.json`。
- GitHub 版沒有上傳競賽資料、cache、模型和 `.pkl` / `.parquet` / `.zip`，所以下面表格提到的 `棄案\data\`、`models_v2\`、`results_v2\` 等不在 repo 裡；組員的檔案（下面 10/26 表格裡的 `submission-1-phase1*.csv`）也沒有放上來。

## 筆電版的紀錄

2026-10-23 整理；2026-10-26 再搬進 v2、v4、v9 的測試輸出、v6 的重複檔和兩個舊備份；2026-10-27 搬進 `ML_RACE_1027_phase2\`（下面第一節）。v9 每天的流程（`daily.bat`、`evening.bat`）不讀這個資料夾；`v9\backup_v9.py` 不備份這個資料夾。

## 2026-10-27 搬過來的東西

| 原路徑 | 內容 | 為什麼不用了 | 還有誰會讀 |
|---|---|---|---|
| `ML_RACE_1027_phase2\` | 10/27 雲端給的更新檔 `ML_RACE_1027_phase2.zip` 解壓出來的 41 個檔（Phase 2 本地檢查 `v6\phase2.bat` 等、`v6\README_v6.md`、最上層 `README.md`、`v11\results\rerun_1027_us\`） | 全部已經裝到原本的位置（逐檔 md5 比對）；其中 `README.md`、`v11\results\rerun_1027_us\README.txt`、`v6\phase2\scores.csv` 之後又更新過，這裡的是舊版，**不要複製回去** | 沒有程式讀它 |

## 2026-10-26 搬過來的東西

| 原路徑 | 內容 | 為什麼不用了 | 還有誰會讀 |
|---|---|---|---|
| `v2\` | v2 的程式、模型（`models_v2\`、`models_v2_static\`）、static 上傳檔、整夜實驗的 walk-forward 結果（`results_v2\`，216 MB） | v4 起改成 rolling pool，v5、v9 都有自己的一份程式 | `v5\daily.bat`（緊急備案）第一次執行會用 `棄案\v2\models_v2\` 當模型池的起點（`v5\train_and_predict_v5.py` 先找原位置、再找這裡）；`v5\research\rlib.py` 讀 `棄案\v2\results_v2\wf\`；`v6\compare.bat` 的預設比較讀 `棄案\v2\out\` |
| `v4\` | v4 的程式、static 模型（`models_v4_static\` 和兩個 zip）、舊的 static 上傳檔 `out\submission_v4_static.csv`（15.36） | v5 的預測和 v4 完全相同；v5 的 `ens_lib.py`、`rolling.py`、`features_extra.py` 和 v4 的只差第一行說明 | `v6\leak_check.py`、`v6\compare.bat`、`v9\static_coef_v9.py`、`v9\train_and_predict_v9.py --static` 的報告用 `棄案\v4\out\submission_v4_static.csv` 當對照（先找原位置、再找這裡）；`v5\research\rlib.py` 改用 v5 自己那份相同的程式 |
| `v9\models_dry\`、`v9\out_dryrun\` | 10/25 筆電實測用的模型池和輸出（README_v9 10.1） | 測試完了 | |
| `v9\models_v9.zip` | 雲端預先訓練到 10/22 的模型池壓縮檔 | 已解壓在 `v9\models_v9\`，之後每天會加新模型 | 想還原成最初的模型池時用 |
| `v9\out\*_1023cloud.*`（4 個） | 雲端模擬 10/23 live 的輸出（README_v9 第九節） | 只是測試，不能上傳 | |
| `v9\cache\adapt_chosen_static.npy` | `v9\research\adapt_eval.py` 第一次執行留下的中間檔 | 程式已經不寫這個檔 | |
| `v6\` 根目錄的 4 個 csv | `submission-1-phase1 (4).csv`、`submission-1-phase1_4.csv`（同一個檔，就是有 leak 的 phase1_4）、`submission_v9-3_static.csv`、`submission_v9_static_wt0_lam05.csv` | 全部是重複的檔（md5 相同）；正本是 `v6\others\submission-1-phase1_4.csv`、`v9\out\submission_v9_static_max.csv`、`v6\others\submission_v9.2_static.csv`（= `v9\out\submission_v9_static_wt0_lam05.csv`） | |
| `_backup_20261025\`、`_backup_20261025_typhoon\` | 10/25 修改前的程式、README，以及颱風休市處理前的 `data\market`、`data\global`、交易日曆 | 修改後的版本已驗證；要還原時從這裡複製回原位置 | |
| `_backup_20261026\`（新增） | 這次為了搬移而修改的 6 個程式（`v5\train_and_predict_v5.py`、`v5\research\rlib.py`、`v6\compare.bat`、`v6\leak_check.py`、`v9\static_coef_v9.py`、`v9\train_and_predict_v9.py`）和 3 個 README 的原檔 | | |

修改的內容都只是「找檔案的位置」：先找原本的位置，找不到再找 `棄案\` 底下同樣的相對位置，所以把 v2、v4 移回原位也能照常執行。驗證（2026-10-26，使用者電腦 VM）：`train_and_predict_v9.py --static --from-members` 和 `static_coef_v9.py` 的輸出和修改前逐 byte 相同，報告裡的 v4 對照 15.36 照常出現（另外也找到了 `v6\others\` 的組員 V3 檔，20.31）；`leak_check.py others\*.csv` 的輸出和搬移前完全相同；`compare_static.py` 用棄案裡的 v4、v2 檔照常執行；`v5\train_and_predict_v5.py` 找得到 `棄案\v2\models_v2\`（50 個模型檔）。

## 2026-10-23 搬過來的東西

> 2026-10-25：v9（live 正式版）用到的 `v3\features_v3.py` 已複製一份到 `v9\features_v3.py`（內容完全相同），v9 每天的流程不再讀這個資料夾。但 `v9\research\` 的研究程式仍會讀 `棄案\v3`，要重算 v9 的研究結果時請保留。

所有檔案都保留原本的相對路徑，例如 `棄案\data\pit\` 原本是 `data\pit\`。要拿回來，移回同樣位置即可，例如在 PowerShell：

```
Move-Item "D:\ML RACE\棄案\data\pit" "D:\ML RACE\data\pit"
```

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

# ML RACE：預測台股「收盤價 > 開盤價」

最後整理：2026-10-26（沒在用的 v2、v4、測試輸出、舊備份移到 `棄案\`；各版本在做什麼見「版本」一節）。
2026-10-27：新增 `v6\phase2.bat`（收盤後自己下載價格、算當天 live 上傳檔的 AR，見下面早上的第 6 步）；10/27 AR 差的原因見 `v11\results\rerun_1027_us\README.txt`。

## 現在用哪一版

- **live（每天上傳）：v9**。晚上跑 `v9\evening.bat`，早上跑 `v9\daily.bat`，上傳 `v9\out\submission_v9_MMDD.csv`（見下面「每天的流程」）。係數固定（w_T 0.5、λ 0.75）。
- **v11（10/26 新增，可選）**：改雙擊 `v11\evening_v11.bat`（晚上）、`v11\daily_v11.bat`（早上）。它們先跑 v9 的完整流程，再跑 v11（多約 1 分鐘），同時寫出 `v9\out\submission_v9_MMDD.csv` 和 `v11\out\submission_v11_MMDD.csv`，上傳其中一個。證據和風險見 `v11\README_v11.md` 第 0 節（hold 差 +0.01、10 月 −8.63）。
- **static leaderboard**：`v9\out\submission_v9_static_w025.csv`（10/25 決定；為什麼和風險見 `v9\README_v9.md` 8.1）。w = 0.5 的 `v9\out\submission_v9_static.csv` 保留。
- 緊急備案：`v5\daily.bat`。注意 v5 從沒在這台電腦跑過，第一次會訓練 Transformer（約 5–10 分鐘）；它的模型池起點是 `棄案\v2\models_v2\`（v2 移到棄案後，程式會自己去那裡找）。
- 10/26 測過「係數每天依最近 15 天自動調整」：做得到，但 9 種調法在五年 walk-forward 全部比固定係數差，沒有採用（`v9\README_v9.md` 第十一節）。

## 每天的流程（課程 zip 約 17:00 收盤後釋出，隔天 9:00 前上傳）

### 晚上（17:00 以後）

1. 從課程網站下載當天的 zip，放進 `downloads\`（留在自己的「下載」資料夾也可以，程式會找最新的 `data*.zip`）。
2. 雙擊 `v9\evening.bat`：合併 zip、抓除權息預告、訓練當天的新模型（筆電約 7 分鐘）、寫出**預備檔** `v9\out\submission_v9_MMDD_prelim.csv`，最後備份（`v9\backup_v9.py`）。
   - 預備檔只少了隔天的夜盤（05:00 才收盤），其他和正式檔相同。網站如果允許重新上傳，可以先傳預備檔當保險。

### 早上（05:00 以後、9:00 以前）

3. 課程網站早上也會打包 zip（10/23 看到 07:00 一份，當天那一 row 已經有前一晚的美股收盤）：有的話先下載到 `downloads\`。
   然後雙擊 `v9\daily.bat`：抓夜盤和除權息；模型昨晚已經訓練好，只做預測（約 2 分鐘），寫出 `v9\out\submission_v9_MMDD.csv`，並列出和預備檔差幾支。
   （10/27 沒有下載早上的 zip，美股少了兩晚；事後重算只差約 4 bp，見 `v11\results\rerun_1027_us\README.txt`。）
4. 上傳 `v9\out\submission_v9_MMDD.csv`。
5. 截 daily board 和 accumulated board 的圖，存進 `leaderboard\`（說明在 `leaderboard\README.md`）。
6. 收盤（13:30）後約 1–2 小時，雙擊 `v6\phase2.bat`：自己下載當天 50 支的開盤、收盤價，算每個上傳檔的 AR，和排行榜的兩個 baseline 列在一起，可以直接和 daily board 對照。在 cmd 執行 `"D:\ML RACE\v6\phase2.bat" --uploaded 上傳的檔.csv` 記下那天上傳的是哪個檔（10/27 已經記好），累積表就會多一行 uploaded，對照 accumulated board（`v6\README_v6.md` 第七節）。

### 出狀況時

| 狀況 | 怎麼做 |
|---|---|
| 時間不夠 | 上傳昨晚的 `v9\out\submission_v9_MMDD_prelim.csv` |
| 只想用現有模型預測（約 1 分鐘） | 在 `v9\` 執行 `python train_and_predict_v9.py --dates auto --retrain none` |
| 出現「is in the past」 | 用到的是舊 zip：下載最新的 zip 再跑 |
| evening 停在「no -1 placeholder day」 | zip 裡還沒有隔天要預測的 -1 row（10/27 15:52 打包的那份只有 10/27 的答案）：晚一點或隔天早上再下載一次 zip，重跑 |
| 出現「is in the past」，但已經是課程網站最新的 zip，而且那天其實沒開盤（颱風假等臨時休市） | 在 `data\ref\closures_manual.csv` 加一行 `日期,原因`，再跑 `daily.bat`：那天會從資料和交易日曆拿掉，要預測的日子改成下一個交易日（`v9\README_v9.md` 10.8；10/23 薔琵颱風就是這樣處理） |
| update_data.py 顯示 CHECK THESE，看過後確認是誤報 | 在 cmd 執行 `"D:\ML RACE\v9\daily.bat" accept`（晚上用 `evening.bat accept`） |
| 其他 FATAL，看過訊息後確認是誤報 | 在 `v9\` 執行 `python train_and_predict_v9.py --dates auto --force` |
| 晚上忘了跑 | 早上照常跑 `daily.bat`，它會自己訓練（約 7 分鐘） |

## 版本：每一版在做什麼

分數是 Top50 的 bp/日（競賽算法）。walk-forward：val 2022–24、hold 2025-01 到 2026-05、post 2026-06 以後。

| 版本 | 現在在哪 | 在做什麼 | 主要結果 | 狀態 |
|---|---|---|---|---|
| v1 | `transformer\`（舊的訓練程式、模型、上傳檔在 `棄案\transformer\`） | 67 個 B 組 feature（只用競賽資料）；MLP 和 Transformer（20 天序列）的機率平均，> 0.5 猜 1 | val 12.30、hold 12.04，post 只剩 0.76（大盤方向押太重） | 停用。資料夾留著 v9 每天要用的共用程式（`comp_features.py`、`tfmodel.py`、`update_data.py`、`final_feature_lists.json`） |
| v2 | `棄案\v2\` | 拆成大盤觀點（pbar = 100 支「收 > 開」機率的平均）和選股觀點（殘差 label：贏過當天 100 支平均）；score = 0.5 × (pbar − 0.5) + (p_e − 當天平均)；除權息日猜 0 | val 13.27、hold 13.45、post 12.22 | 停用，之後每一版的骨架。10/26 移到棄案（`v5\daily.bat` 第一次執行會讀它的 `models_v2\`） |
| v3 | `棄案\v3\` | v2 的選股觀點加 5 個新 feature | 沒有比 v2 好 | 未採用；`features_v3.py` 已複製到 `v9\` |
| v4 | `棄案\v4\` | v2 + rolling pool：每天只用到前一個交易日的資料加訓新模型、平均最近 5 個訓練點；除權息表補到 10 月 | static 15.36（舊的 static 上傳檔） | 停用。10/26 移到棄案；它的 static 檔仍是 `v6` 工具和 v9 static 報告的對照（程式會到棄案找） |
| v5 | `v5\` | 預測和 v4 完全相同；加上每天早上的防呆（`guards.py`，FATAL 就不寫檔）和監控；feature 去留、ablation、三個風險（selection bias、drift、檢定力） | 全期 13.38；21 個改法都沒過門檻 | 緊急備案（`v5\daily.bat`）；v9 每天用它的 `ens_lib.py`、`rolling.py`、`guards.py`、`features_extra.py`、`fetch_exdiv.py` |
| v6 | `v6\` | static 專用工具：`compare_static.py`（組內比較，附雜訊和「挑過幾組」的修正）、`leak_check.py`（檢查上傳檔有沒有 leak）；`others\` 放組員的檔。10/27 加上 live（Phase 2）的本地檢查 `phase2.bat`：自己下載當天價格、算每個上傳檔的 AR 和累積 | phase1_4 查出 6–8 月等於答案（FATAL）；phase2 和排行榜的 baseline 逐位相同 | 工具，在用 |
| v7 | `v7\` | live 研究：擴充股票池、台指期夜盤 feature、開盤 gap 上限、除權息前後日規則、w_T 風險設定 | 都沒過當時的門檻（val t > 0 的幾項後來併進 v9） | 研究；v9 每天用它的 `fetch\fetch_night_session.py` 和 `data\tx_daily.csv` |
| v8 | `v8\` | 大盤觀點診斷（pbar 從哪來、選股觀點暗藏的低 beta 押注）、架構比較（CNN、GRU、LSTM、LightGBM、logistic）、beta 對沖 | 5 個大盤改法都沒過；LightGBM 進選股（val +0.36）、對沖 λ 0.75（val +0.28）後來進了 v9 | 研究；v9 每天用 `hedge_beta.py`、`monitor_market.py`，v9 的研究程式用 `research\mlib.py` 和 `cache\` |
| v9 | `v9\` | 把 v1–v8 裡 val t > 0 的 11 個改法混成一個：夜盤、擴充股票池、LightGBM、4 年 window、σ 權重、v3 成員、拿掉那斯達克、除權息前一天／後一天／後兩天猜 0、beta 對沖 λ 0.75 | walk-forward 14.31（val 13.99、hold 14.33、post 16.59；v5 13.38）；rolling static 16.42，w 0.25 版 18.84 | **live 正式版**（10/27 起）；static 上傳 `out\submission_v9_static_w025.csv` |
| v9 第十一節 | `v9\research\adapt_eval.py` | 係數（w_T、λ、LightGBM 權重、Transformer 權重、門檻）每天依最近 15 天的成績自動調整 | 9 種調法全部輸固定係數（主要假設 −1.67，t −1.94）；視窗 5–250 天都輸 | 不採用，live 沒改 |
| v10 | `v10\` | Qlib Alpha158（158 個、23 組）加進 v9 選股觀點的 ablation；val（AR）t > 0 的群組全部加進去 | L 入選 19 組、M 5 組；hold −0.45（t −0.75）沒過事先寫的檢驗 | 不採用，live 沒改（`v10\README_v10.md`） |
| v11 | `v11\` | 第一部分：用低雜訊的方法重挑 Alpha158（rank IC、20 個 seed、嚴格規則、nested walk-forward），沒有群組通過。第二部分：v9 + 夜盤反轉（T2）+ 每月前 3 天多 5 支猜 1（T3）+ 選股第二個 LightGBM（X1），照「val bp > 0」從 7 個候選入選 | walk-forward 15.02（v9 14.31）：val +0.62、hold +0.01、post +3.76，事先寫的檢驗通過；但 10 月 15 天 −8.63（t −3.3） | 和 v9 並行：`v11\daily_v11.bat`、`evening_v11.bat` 同時寫出 v9 和 v11 的上傳檔，上傳哪一個由你決定（`v11\README_v11.md`） |
| v12 | `v12\` | 用實驗檢驗 v11 第 9 節對 ranking objective 的解讀：把「label 的資訊」（正負、十分位、報酬大小）和「損失函數」（pointwise、同一天 pairwise、lambdarank）分開，新訓練 6 種 LightGBM；照「val bp > 0」入選的再加到 v11 上檢驗 | 解讀：lgb_res 本來就是排序學習（成立）；X5 差是因為 NDCG 的位置折扣（成立），不是十分位 label（十分位反而比報酬大小好）。v12 = v11 + 4 個入選：hold −1.05（t −2.4）沒過事先寫的檢驗 | 只有研究，沒有每天的程式，live 沒改（`v12\README_v12.md`） |

`v9\out\` 裡的 static 檔：`submission_v9_static_w025.csv`（上傳中，18.84）、`submission_v9_static.csv`（w 0.5，16.42）、`submission_v9_static_wt0_lam05.csv`（21.53，`v6\others\` 裡的檔名是 v9.2）、`submission_v9_static_max.csv`（32.18，組內檔名 v9-3；係數是在 static 上搜尋出來的，同一組係數在 walk-forward 反而比 live 設定低 3.7）。

各版本的細節、證據和數學證明在各自資料夾的 README（v1 是 `transformer\README.md`，v2、v3、v4 是 `棄案\v2\README_v2.md`、`棄案\v3\README_v3.md`、`棄案\v4\README_v4.md`）。舊 README 裡寫的 `v2\`、`v4\`、`_backup_20261025\`、`_backup_20261025_typhoon\`、`v9\models_dry\`、`v9\out_dryrun\`，現在都在 `棄案\` 底下同樣的相對位置（例如 `v4\out\` → `棄案\v4\out\`）。

## 資料夾

| 資料夾或檔案 | 內容 | 注意 |
|---|---|---|
| `data\` | 競賽資料（`market\`、`global\`、`info.json`）和 `ref\`（交易日曆、除權息） | 每天由 `update_data.py` 更新 |
| `data\ref\exdiv_fetch_log.csv` | 每次抓除權息預告的成功或失敗紀錄 | 防呆會讀 |
| `data\ref\closures_manual.csv` | 臨時休市日（日曆上有、實際沒開盤，例如 10/23 颱風）；`update_data.py` 和夜盤對齊會讀 | 手動維護；`calendar_official.parquet` 是拿掉休市日之前的官方日曆 |
| `data_backup_original\` | `update_data.py` 第一次執行時做的原始資料備份 | 不要移動（`v6\leak_check.py` 也用它找原始資料的最後一天） |
| `downloads\` | 每天的 zip；`archive\` 留著每一個用過的 zip（`data_<預測日>.zip`） | |
| `leaderboard\` | live 排行榜的截圖和表 | 每天記，11/23 定版要用 |
| `v9\` | live 正式版：程式、模型池 `models_v9\`、輸出 `out\`、`data\panel_ext.parquet`、`features_v3.py`、研究 `research\` | |
| `transformer\`、`v5\`、`v7\`、`v8\` | 舊版本，但裡面有 v9 每天要用的程式或資料（見下一節），所以整個資料夾留在原位 | 不要移動 |
| `v7\data\tx_daily.csv` | 夜盤資料，`daily.bat` 每天更新；`v7\data\archive\` 每天一份快照 | v9 在用，不要刪 |
| `v6\` | static 工具和組員的檔（`others\`）；live 的本地檢查 `phase2.bat`，輸出在 `phase2\`（`files\` 放要一起算的檔） | |
| `v10\`、`v11\`、`v12\` | v10：Alpha158 研究（沒通過）。v12：ranking objective 的實驗（只有研究）。v11：Alpha158 重挑（沒通過）＋ v11 模型（`train_and_predict_v11.py`、`models_v11\`、`data\pbar_history.csv`，用 v9 的模型池） | 用 v11 時不要刪 `v11\models_v11\`、`v11\data\`；`v11\cache\_stage\` 是傳檔到雲端用的暫存 tar，可以刪 |
| `棄案\` | 沒在用的東西：v2、v3、v4、v1 的舊檔、v9 的測試輸出、舊備份（`_backup_20261025\`、`_backup_20261025_typhoon\`、`_backup_20261026\`）；每一項為什麼不用、還有誰會讀，寫在 `棄案\README.md` | v9 每天的流程不讀；`backup_v9.py` 不備份 |
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

`evening.bat` 最後會執行 `v9\backup_v9.py`，把整個資料夾（不含 `棄案` 和研究用的 cache）複製到 `%USERPROFILE%\ML_RACE_backup`（C 槽）。只複製新的或改過的檔，不會刪檔（所以之前備份過的 v2、v4 還留在備份裡）。想備份到隨身碟或私人雲端：`python backup_v9.py E:\ML_RACE_backup`。競賽資料不能分享給沒修課的人，備份請放在私人的地方。

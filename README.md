# ML RACE：預測台股「收盤價 > 開盤價」

課程的機器學習競賽：每個交易日 09:00 前，替 Top50 的 50 支股票各猜 1（收盤 > 開盤）或 0，上傳一個 CSV。
分數 AR = Top50 平均的 (2·pred − 1) × r，r = (close − open) / open（還原價的盤中報酬），單位 bp/日。

這個 repo 是筆電上 `D:\ML RACE` 的整理版：

- **沒有上傳**：競賽資料和其他資料、cache、訓練好的模型、`.pkl` / `.parquet` / `.zip`（課程規定競賽資料不能分享給沒修課的人）。data 裡有什麼、用在哪裡，寫在下面「資料」一節。
- **目前用不到的**（只有研究、每天的程式不讀）都在 `棄案/`。v12–v16、overfit、reg_svm、race_mkt、race_news、race_agg 只在這個 repo 搬進 `棄案/`，筆電上還在最上層。
- 組員的系統和文件、leaderboard 截圖沒有放上來（見最後「出處」）。
- `README_原版.md` 是筆電上 `D:\ML RACE\README.md` 的原文，沒有改，裡面的路徑是筆電上的位置。每天的操作細節、出狀況時怎麼辦、每一版的結果都在那裡。
- `FEATURES_AND_ARCHITECTURE.md` 是模型、111 個 feature、架構的完整說明。

## 現在在用的三套

| | v9（live 正式版） | v11 / W0 | race 版 |
|---|---|---|---|
| 晚上（課程 zip 出來後） | `v9/evening.bat` | `v11/evening_v11.bat` | `race ver/race_live/race_evening.bat`（21:30 以後，可省略） |
| 早上（07:00 以後下載最新 zip，09:00 前） | `v9/daily.bat` | `v11/daily_v11.bat`；W0 再跑 `v11/custom_v11.bat` | `race ver/race_live/race_daily.bat`（08:01 以後） |
| 上傳檔 | `v9/out/submission_v9_MMDD.csv` | `v11/out/submission_v11_MMDD.csv`；W0 是 `submission_v11_MMDD_wt0_lam0.5.csv` | `race ver/race_live/out/submission_race_MMDD.csv` |
| 模型 | 大盤觀點（3 個 MLP ＋ Transformer）＋ 選股觀點（4 個 MLP ＋ LightGBM），除權息規則，beta 對沖 λ 0.75 | v9 ＋ T2 夜盤反轉 ＋ T3 月初多猜 5 支 ＋ X1 第二個選股 LightGBM；W0 = w_T 0、λ 0.5 | C 組 169 個 feature ＋ LightGBM，每天重訓、最近 3 個模型平均，Top50 機率前 25 支猜 1 ＋ 除權息規則 |
| walk-forward | 14.31 bp/日 | v11 15.02、W0 12.22 | 16.35 bp/日 |

- v9、v11、W0 的 walk-forward 是 2022-01 到 2026-10-22（1,164 天）；race 版是 2022–2024（727 天）。期間不同，不能直接比。
- 收盤（13:30）後 1–2 小時跑 `v6/phase2.bat`：自己下載當天 50 支的開盤、收盤價，算每個上傳檔的 AR，和排行榜的兩個 baseline 對照。

## 架構（v9 / v11）

```
競賽資料（market 100 支、global）   台指期夜盤（FinMind）   除權息預告   擴充股票池（只用來訓練）
                │
                ▼
feature panel：每個（日期, 股票）一個 row，111 個 feature，只用 D 日 09:00 前拿得到的資料
                │
                ├── 大盤觀點（label：收 > 開）：3 個 MLP ＋ Transformer → pbar = 當天 100 支機率的平均
                ├── 選股觀點（label：盤中報酬贏過當天 100 支平均）：4 個 MLP ＋ LightGBM → p_e − 當天平均
                ▼
score = w_T × (pbar − 0.5) + (p_e − 當天平均)   （v11：T2、T3）
                │
                ▼
除權息當天、前一天、後一天、後兩天猜 0 → beta 對沖（λ）→ Top50 的 0/1 → submission csv
```

- 每個成員每天用「到前一個交易日為止」的資料訓練新模型，預測時平均最近 5 個訓練點的模型（rolling pool）。
- race 版的 feature 是 C 組 169 個（價量、籌碼、Yahoo 的國際行情與 ADR、FinMind 的融資融券／借券／當沖），一個 LightGBM，不加大盤觀點、不對沖。

## 資料夾

| 資料夾 | 功能 |
|---|---|
| `v9/` | live 正式版：每天的程式、`out/`（上傳檔、機率、檢查結果）、`research/`（walk-forward、係數自動調整的研究） |
| `v11/` | v11 和 W0：`train_and_predict_v11.py`、三個 bat、`out/`、`results/`（AR 計算、補跑紀錄）、`research/` |
| `transformer/` | v1。留著 v9 每天要用的共用程式：`update_data.py`、`comp_features.py`、`tfmodel.py`、`final_feature_lists.json` |
| `v5/` | v5。留著 v9 每天要用的 `ens_lib.py`、`rolling.py`、`guards.py`、`features_extra.py`、`fetch_exdiv.py`；`daily.bat` 是緊急備案 |
| `v6/` | static 工具（`compare_static.py`、`leak_check.py`）和每天算 AR 的 `phase2.bat`（報告在 `phase2/`） |
| `v7/` | live 研究；留著每天要用的夜盤下載 `fetch/fetch_night_session.py` |
| `v8/` | 大盤觀點診斷、架構比較；留著每天要用的 `hedge_beta.py`、`monitor_market.py` |
| `v10/` | Qlib Alpha158（沒採用），但 `v11/research` 會 import `v10/research`（W0 的 `custom_v11.bat` 用到），所以留在原位 |
| `race ver/race_live/` | race 版每天的程式、`out/`、`results/`（每天重訓的 walk-forward） |
| `race ver/race_std/` | 四套系統合併的驗證標準 RACE-STD v1：feature、learner、爬蟲的程式（race_live 直接 import）、`results/`、`part2/` |
| `leaderboard/` | 每天記錄排行榜的方式（截圖沒有上傳） |
| `棄案/` | 目前用不到的版本和研究，說明在 `棄案/README.md` |

## 主要程式的功能

| 程式 | 功能 |
|---|---|
| `transformer/update_data.py` | 把課程網站下載的 `data*.zip` 合併進 `data/`；檢查日期、要預測的 −1 row、臨時休市日 |
| `v5/fetch_exdiv.py` | 從 TWSE 抓 Top100 已公告的除權息日（TWT48U）→ `data/ref/exdiv_upcoming.csv`；`--history` 抓過去的（TWT49U） |
| `v7/fetch/fetch_night_session.py` | 從 FinMind 抓台指期日盤、夜盤（15:00–05:00）→ `v7/data/tx_daily.csv` |
| `transformer/comp_features.py` | B 組 67 個 feature，只用競賽資料 |
| `v5/features_extra.py` | XR（當天 100 支裡的排名）、日曆、除權息旗標、殘差 label |
| `v9/features_v3.py` | NEW 5 個選股 feature（gap_bias_60/250、resid_vol_60、dow_bias_e、hi52） |
| `transformer/tfmodel.py` | MLP、Transformer、scaler、訓練迴圈 |
| `v5/ens_lib.py`、`v5/rolling.py` | panel 和 Transformer 的訓練；模型池的訓練點和 seed 編號 |
| `v9/rolling9.py` | v9 的核心：成員定義、NIGHT / SIG feature、MLP / LightGBM 訓練、rolling 模型池、合成與決策規則 |
| `v9/train_and_predict_v9.py` | v9 每天的訓練和預測（evening 加 `--prelim`）；也能跑 static walk-forward |
| `v5/guards.py`、`v9/check_zip_fresh.py` | 上傳前的防呆：FATAL 就不寫上傳檔（舊 zip、缺前一晚美股、缺值太多、模型池是空的、除權息預告太久沒抓到、輸出不合理） |
| `v8/hedge_beta.py` | beta 對沖：抵銷選股觀點偏好低 beta 股票、等於暗中押大盤跌的部分 |
| `v9/monitor_v9.py`、`v8/monitor_market.py`、`v5/monitor_live.py` | live 成績監控，拆出大盤和選股各賺賠多少 |
| `v9/backup_v9.py` | 把整個資料夾備份到 `%USERPROFILE%\ML_RACE_backup`（不備份棄案和研究的 cache） |
| `v9/static_coef_v9.py` | static leaderboard：用不同係數重組存好的成員機率 |
| `v11/train_and_predict_v11.py` | v11 每天的預測；`--w-t 0 --lam 0.5` 就是 W0 |
| `v11/research/v11_try.py` | 自訂參數在 walk-forward 的表現（`custom_v11.bat` 的第一步） |
| `race ver/race_live/race_live.py` | race 版：`train`（晚上訓練缺的模型）、`predict`（早上建 panel、預測、Top25 ＋ 除權息規則、檢查） |
| `race ver/race_live/livelib.py` | race 版共用：C 組 panel、模型池、決策 |
| `race ver/race_live/fetch_live.py` | race 版每天的爬蟲：Yahoo 28 個序列、FinMind 融資融券／借券／當沖 → `race_live/ext/` |
| `race ver/race_live/walkforward.py` | race 版每天重訓的 walk-forward |
| `race ver/race_std/research/` | `build_panel.py`（U、P、C、M 四組 feature）、`learners.py`（5 個 learner）、`fetch_ext.py`（爬蟲）、`rslib.py`（folds、指標）、`stage1_eval.py` / `freeze.py` / `stage2_validate.py`（事先寫好的比較與一次性驗證） |
| `v6/phase2_check.py`（`phase2.bat`） | 收盤後下載當天價格（FinMind，失敗改 TWSE / TPEx），算每個上傳檔的 AR 和累積 |
| `v6/compare_static.py`、`v6/leak_check.py` | static 上傳檔的比較（附雜訊和「挑過幾組」的修正）；leak 檢查（答案段、日期錯位、同一 row 的 leak、大盤 timing） |
| `transformer/check_submission.py`、`transformer/score_static.py` | 上傳前檢查格式；用競賽算法算分 |

## 資料（都沒有上傳）

### `data/`：競賽資料

每天由 `transformer/update_data.py` 從課程 zip 合併。要預測的那一天，課程 zip 會放一個值都是 −1 的 row，程式把它當成要預測的日子。

| 檔案 | 內容 | 用在 |
|---|---|---|
| `data/info.json` | `Top100@Y`：100 支股票代號；`Top50@Y`：要預測的 50 支 | 100 支用來算 feature 和「當天平均」，50 支是上傳的股票 |
| `data/market/<代號>.csv`（100 個檔） | 每支股票一個檔，每個 row 一個交易日，2020-01-02 起（到 11/18 共 1,669 個 row），21 個 column，見下表 | 個股 feature、label |
| `data/global/global_data.csv` | 每個 row 一個交易日，2020-01-03 起（1,558 個 row）：`^VIX`、`^IXIC`（Nasdaq）、`^GSPC`（S&P 500）、`^TWII`（加權指數）、`^KS11`（KOSPI）、`TX_OI`（外資台指期淨未平倉口數） | 美股、台股、韓股、期貨籌碼 feature |
| `data/ref/calendar.parquet` | 交易日曆（date、pos），2009-01-05 到 2026-12-31，已扣掉臨時休市 | 找下一個交易日、對齊 |
| `data/ref/calendar_official.parquet` | 扣掉臨時休市之前的官方日曆 | 對照 |
| `data/ref/closures_manual.csv` | 手動維護的臨時休市日（date、reason），目前 5 天：10/23 颱風，9/25、9/28、10/9、10/26 國定假日 | `update_data.py`、夜盤對齊 |
| `data/ref/corporate_actions.parquet` | 除權息的還原因子（stock_id、date、f、ref_price、source），7,379 筆，2009 起 | 除權息旗標 |
| `data/ref/exdiv_history.csv` | 過去的除權息事件（stock_id、date、kind 權／息、name、value、prev_close、source），TWSE TWT49U | 除權息旗標 |
| `data/ref/exdiv_upcoming.csv` | 已公告、還沒到的除權息（stock_id、date、kind、name、cash_dividend、first_seen），`fetch_exdiv.py` 每天更新 | 除權息前一天猜 0 |
| `data/ref/exdiv_fetch_log.csv` | 每次抓除權息預告的結果（run_at、status、n_events、n_top100） | `guards.py`：超過 3 個交易日沒抓成功就 FATAL |

`data/market/<代號>.csv` 的 21 個 column：

| column | 內容 |
|---|---|
| `date` | 交易日 |
| `etl:adj_close`、`etl:adj_open`、`etl:adj_high`、`etl:adj_low` | 還原收、開、高、低 |
| `price:成交筆數`、`price:成交金額`、`price:成交股數` | 成交筆數、金額、股數 |
| `foreign_investors_shareholding:全體外資及陸資持股比率` | 外資持股比率（%） |
| `institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)` | 外資買賣超（股） |
| `institutional_investors_trading_summary:投信買賣超股數` | 投信買賣超（股） |
| `institutional_investors_trading_summary:自營商買賣超股數(自行買賣)` | 自營商（自行買賣）買賣超（股） |
| `institutional_investors_trading_summary:自營商買賣超股數(避險)` | 自營商（避險）買賣超（股） |
| `RSI`、`SMA`、`MACD`、`ATR`、`BBANDS_upper`、`BBANDS_mid`、`BBANDS_lower` | 課程提供的技術指標（SMA 是 30 日均線） |
| `y` | label：1 = 收盤 > 開盤 |

時間規則：row D 的 feature 只用 D 日 09:00 以前公布的資料。個股資料和 `^TWII`、`^KS11`、`TX_OI` 用到 D−1；global 的美股在 row D 已經是前一晚的收盤（早上 07:00 以後打包的 zip 才有）。

### 其他資料

| 位置 | 內容 | 用在 |
|---|---|---|
| `data_backup_original/` | `update_data.py` 第一次執行（10/22）時備份的原始 `data/` | `v6/leak_check.py` 用它找原始資料的最後一天 |
| `downloads/` | 課程網站下載的 zip（`data (N).zip`）；`archive/` 留著每個用過的 zip（`data_<預測日>.zip`） | `update_data.py` 找最新的 `data*.zip` |
| `v7/data/tx_daily.csv` | 台指期每日行情（FinMind TaiwanFuturesDaily）：date、futures_id、contract_date、open、max、min、close、spread、spread_per、volume、settlement_price、open_interest、trading_session（position = 日盤、after_market = 夜盤），2020-01-02 起約 45,000 個 row；`archive/` 每天一份快照 | NIGHT feature（night_ret、night_range）、v11 的 T2 |
| `v7/data/mis_symbols.csv` | TWSE MIS 盤前試撮的代號表 | `v7/fetch/record_mis.py`（研究） |
| `v9/data/panel_ext.parquet` | 擴充股票池：約 210 支其他上市股的 panel，資料到 2026-07（77 MB） | 只用來訓練 mlp_res_B9 |
| `v11/data/pbar_history.csv` | 每天的 pbar（date、pbar、source），2022-01 起，前段是 walk-forward、10 月底起是 live | v11 的 T2 要用前 250 天的 pbar |
| `race ver/race_live/ext/` | race 版每天爬的資料。`yahoo/`：ADR（TSM、UMC、ASX、CHT）和對應的台股、`^SOX`、`^N225`、`^KS11`、`^IXIC`、`^GSPC`、`TWD=X`、`^TNX`、`DX-Y.NYB`、美股產業 ETF（SPY、SOXX、XLK、XLF 等 12 支）；`finmind/`：融資融券、借券賣出餘額、當沖 | C 組裡爬來的 40 個 feature |
| `race ver/race_std/ext/` | 第一部分凍結的爬蟲資料（race_live 第一次執行時複製到 `race_live/ext/`） | race_std 的研究 |
| `v6/phase2/prices/` | `phase2.bat` 下載的當天 50 支開盤、收盤價 | 算每天的 AR |
| `docs/` | 課程投影片 | |

### 模型和 cache

| 位置 | 內容 |
|---|---|
| `v9/models_v9/`（208 個檔，12 MB） | v9 的 9 個成員的模型池，每個成員一個資料夾 |
| `v11/models_v11/`（11 個檔，2.8 MB） | X1（lgb_res_xrN） |
| `race ver/race_live/models/` | race 版每天重訓的 LightGBM（`race_YYYYMMDD_seed*.txt` 和 `.json`） |
| 各資料夾的 `cache/` | 研究的中間結果（panel、各模型的預測），重跑會重建 |

模型沒有上傳：第一次執行時，evening（或沒跑 evening 時的 daily）會自己訓練。

## 在另一台電腦跑

1. Windows、Python 3，套件：numpy、pandas、torch、lightgbm、pyarrow（bat 會自動裝 lightgbm、pyarrow）；race_std 的研究另外要 scikit-learn。
2. 把上面「資料」一節的檔案放回原位：至少要有 `data/`（market、global、info.json、ref），每天的 zip 放進 `downloads/`。
3. 夜盤和 race 版的 FinMind 爬蟲可以設 `FINMIND_TOKEN` 環境變數（免費帳號，提高請求上限）。token 只從環境變數讀，不要寫進檔案。
4. 照「現在在用的三套」的 bat 執行。

## 棄案

| 資料夾 | 內容 | 結果 |
|---|---|---|
| `棄案/v2`、`v3`、`v4` | v2：拆成大盤、選股兩個觀點（之後每一版的骨架）；v3：5 個新 feature；v4：rolling pool | 已由 v5、v9 取代 |
| `棄案/transformer`、`best_features`、`feature_analysis` | v1 的舊訓練程式和輸出；最早的 feature 分析 | 結論已整理成 B 組 67 個 feature |
| `棄案/v12` | ranking objective 的實驗 | 沒通過 |
| `棄案/v13` | LSTM、1D CNN、F-LOAM | 沒通過 |
| `棄案/v14` | 其他量化方法：PCA 統計套利、因子中性、季底作帳、1/N | 沒通過 |
| `棄案/v15` | 去噪方法、Random Forest | 沒通過 |
| `棄案/v16` | Kalman filter、smoother | 沒通過 |
| `棄案/overfit` | overfit 檢查 | 只有研究 |
| `棄案/reg_svm` | regularization、SVM | 沒通過 |
| `棄案/race ver/race_mkt` | 任何數量 0/1 的大盤觀點、期貨籌碼 | 沒通過 |
| `棄案/race ver/race_news` | 鉅亨網新聞 feature | 沒通過 |
| `棄案/race ver/race_agg` | aggregation 方法（blending、bagging、AdaBoost、單一決策樹、RF、換 loss 的 GBDT） | 沒通過 |
| `棄案/_backup_*`、`ML_RACE_1027_phase2`、`v6`、`v9` | 舊備份、重複的檔、測試輸出 | |

每一版的細節、證據和證明在各自資料夾的 README（例如 `棄案/v15/README_v15.md`）。棄案裡的程式用自己的位置往上找專案根目錄，要重跑時把整個資料夾移回最上層。

## 出處

- race 版的 C 組 169 個 feature 來自組員陳聖翰的 V5 架構；race_std 比較的 P 組（FinPred v2.3）、M 組（NTU M2）是組內另外兩套系統。這些 feature 在 `race ver/race_std/research/build_panel.py` 依原文件重寫；組員的原始文件和系統沒有放在這個 repo。
- `v10/alpha158.py` 依 Microsoft Qlib 的 Alpha158 定義。
- 其他文獻列在各版本的 README。

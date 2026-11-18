# v6：第一階段（static leaderboard）工具＋第二階段（live）的本地檢查

2026-10-27：新增第七節，第二階段（dynamic leaderboard）的本地檢查 `phase2.bat`：收盤後自己下載 50 支的開盤、收盤價，算每個上傳檔當天的 AR 和累積分數，和 daily board、accumulated board 對照；不產生、不修改任何上傳檔。以下第〇到六節是 10/24 整理的第一階段內容，沒有改。

整理日期：2026-10-24。v6 只處理第一階段（static，2026/06/01–09/30 共 84 天），和第二階段（live）的 `v7\` 完全分開，兩邊不共用程式。static 上傳檔維持 `v4\out\submission_v4_static.csv`（15.36 bp/日）：這次沒有找到能讓 static 合法變好的改法。v1–v5 的程式、模型和結果都沒有動（執行研究程式時 Python 在 `v4\`、`v5\research\` 自動產生了 `__pycache__`，不影響結果）。

## 〇、簡短說明

- 我們的計分和排行榜完全一致：兩個 baseline 都重現（Previous day 7.22、Always down −2.04 bp/日）。
- 合規的上限大約在 15–17 bp/日：v4 是 15.36；就算把 9:00 的真實開盤價放進模型（違規，只當上限），也只有 17.37。
- 排行榜前段 26–45 bp/日，落在 leakage 的範圍：用到當天收盤的簡單規則（同一列的 RSI、MACD 都含當天收盤）有 112–160；只用當天開盤也有 20–30。
- 84 天的雜訊很大：v4 和「全部猜 0」的差 17.4 bp/日，t = 1.44，分不出來。組內比較請用 `compare_static.py`。
- 不做「用 static 答案挑設定」的版本：那違反課程 static 規則（用 6/1 預測 6/2，不能反過來），而且平均只會多 4–9 bp/日（第五節），也追不上 45。

## 怎麼執行

v6 不會產生新的上傳檔：static 上傳檔仍是 `v4\out\submission_v4_static.csv`，每天的 live 仍用 `v5\daily.bat`。

| 要做的事 | 怎麼做 | 時間 |
|---|---|---|
| 比較 static 上傳檔 | 雙擊 `v6\compare.bat`：v4、v2 和 v2 plain 互相比較 | 幾秒 |
| 和組員的檔案比較 | 把他的上傳檔（0/1 的 csv，格式同 sample_submission）拖到 `v6\compare.bat` 上 | 幾秒 |
| 他是在 static 上挑過設定的 | 在 cmd 執行 `"D:\ML RACE\v6\compare.bat" 他的檔.csv --tried 他的檔.csv=20`（20 換成他試過的設定數） | 幾秒 |
| 重算第二節的 leak 對照表 | 雙擊 `v6\research\rerun.bat`，結果寫到 `research\results\s1_static_leaks.txt` | 幾秒 |
| 第二階段：收盤後算當天 live 上傳檔的 AR | 雙擊 `v6\phase2.bat`（第七節） | 約 20 秒（要下載時） |
| 第二階段：記下當天上傳的是哪個檔 | cmd：`"D:\ML RACE\v6\phase2.bat" --uploaded 上傳的檔.csv` | 約 20 秒 |

只需要 numpy 和 pandas。

## 一、Static 的規則

課程規定 static 每一天只能用到前一個交易日為止的資料。

- 可以：每天用到前一天為止的資料重訓（v4 的 rolling pool）；所有設定在 6/1 以前決定。
- 不行：用第 t 天那一列的任何欄位（收盤、最高、最低、成交量、RSI、SMA、MACD、ATR、布林通道都含第 t 天收盤後的資訊），或第 t 天開盤價；看過 static 成績再回頭改設定。
- 誠實說明：v2 選 w_T = 0.5 時參考過 6–10 月（README_v2 第 4.2 節），所以 v4 的 15.36 也不是完全乾淨。

## 二、排行榜分數的組成（`research\s1_static_leaks.py`）

static 期間、Top50、bp/日：

| 規則 | 用到的資訊 | 分數 |
|---|---|---|
| Previous day（猜 y_{t−1}） | 前一天 | 7.22（= 排行榜 baseline） |
| 全部猜 0 | 無 | −2.04（= 排行榜 baseline） |
| 完全知道答案 | 當天收盤 | 198.18 |
| 收盤_t > 收盤_{t−1} | 當天收盤 | 159.80 |
| RSI_t > RSI_{t−1} | 同一列的 RSI | 158.99 |
| MACD_t > MACD_{t−1} | 同一列的 MACD | 111.80 |
| 跳空向下就猜 1 | 當天開盤 | 29.91 |
| 跳空低於當天中位數就猜 1 | 當天開盤 | 20.35 |
| RSI_{t−1} > RSI_{t−2} | 前一天（合規） | −6.99 |
| v4 上傳檔 | 前一天（合規） | 15.36 |
| v4 + 真實開盤價當 feature | 當天開盤（上限，v7 e6） | 17.37 |

判讀：

- 45 分需要每天比 v4 多約 30 bp。v4 和其他合規設定在 static 的成對差標準誤是 1–8 bp，差 30 bp 至少是 3.7 個標準誤；v5、v7 在 static 期間比過的合規改法，和 v4（或同條件對照組）的差都在 −10 到 +1.4 之間（最大的是 v7 的擴充股票池，+1.37，t 0.61）。
- 連真實開盤價都只多 2 bp，所以 45 分大概不是來自開盤價，而是用到當天收盤的資訊。最常見的原因是 row 對齊：競賽資料第 t 天那一列的技術指標都用到第 t 天收盤，直接拿同一列的 feature 預測同一列的 y 就會 leak。`transformer\comp_features.py` 的做法是先 `shift(1)`。
- 這類模型在 live 會失效：9:00 前，要預測的那一天在資料裡是 −1 佔位列，只能用前一天的列。
- 不能確定任何一隊的做法；以上只說明這些分數落在哪個範圍。如果組員的 static 分數遠高於 20，可以請他檢查 feature 有沒有 shift。

## 三、84 天的雜訊

`python compare_static.py ..\v4\out\submission_v4_static.csv ..\v2\out\submission_v2_static.csv ..\棄案\transformer\submission_ensemble.csv`

| 檔案 | AR | 每日 sd | SE | t（NW） | timing | xsec |
|---|---|---|---|---|---|---|
| v4 static | 15.36 | 81.8 | 8.92 | 2.05 | −3.13 | 18.49 |
| v2 static | 11.27 | 79.5 | 8.67 | 1.42 | −2.36 | 13.63 |
| v1 ensemble | 4.23 | 95.0 | 10.37 | 0.37 | −3.33 | 7.56 |
| 全部猜 0 | −2.04 | 115.7 | 12.63 | −0.17 | −2.04 | 0 |

和 v4 的成對差（同一批 84 天）：

| 對 v4 | 差 | SE | t | 95% 區間 | MDE80 | 19 天內贏過 v4 的機率 |
|---|---|---|---|---|---|---|
| v2 static | −4.09 | 4.91 | −0.83 | −9.32 至 0.75 | 13.8 | 0.35 |
| v1 ensemble | −11.13 | 8.02 | −1.39 | −27.57 至 3.00 | 22.5 | 0.22 |
| 全部猜 0 | −17.40 | 12.10 | −1.44 | −36.96 至 3.58 | 33.9 | 0.22 |

- SE 取 iid 和 Newey–West（5 lags）較大者；95% 區間是 stationary bootstrap（平均區塊 10 天）；MDE80 = 2.80 × SE（證明見 README_v5 7.3）。
- static 期間大盤（Top50 平均盤中報酬）每日標準差 115.7 bp，比 2022–2026/05 高很多。

## 四、建議的組內比較規則

1. 每個人交上傳檔和產生它的程式；用同一份 `compare_static.py` 計分（同樣 50 支、同樣 84 天）。
2. 看過 static 成績才改的設定，寫出總共試了幾個（N），比較時加 `--tried 檔名=N`。
3. 成對差的 95% 區間含 0，或差小於 MDE80：判定分不出來。
4. 分不出來時，比 10/12 以後的 live 累積成績，那是唯一沒有人能事先看到答案的資料。
5. 檢查 feature 是否 shift(1)（第二節）。

## 五、`compare_static.py`

```
python compare_static.py A.csv B.csv [C.csv ...] [--base A.csv] [--tried B.csv=20] [--start 2026-06-01] [--end 2026-09-30] [--out 結果.csv]
```

- base 預設是第一個檔，自動加上「全部猜 0」「全部猜 1」當對照。
- label 讀 `data\market\*.csv`（etl:adj_open、etl:adj_close），預測日的 −1 佔位列略過。只用 numpy 和 pandas。
- 輸出和 `transformer\score_static.py`、README_v4 的數字相同（v4 15.36、v2 11.27）。

**試 N 個設定、留最好的，平均會高估多少**

直觀：每個設定的分數 = 實力 + 運氣；挑最高的就是挑運氣最好的，運氣不會延續到 live。

證明：設 N 個設定一樣好，和 base 的成對差 d_k = δ + s·Z_k，Z_k 獨立標準常態，s > 0。

1. max_k d_k = δ + s·max_k Z_k。
2. P(max Z ≤ z) = Φ(z)^N，所以 max Z 的密度是 Nφ(z)Φ(z)^(N−1)，E[max d] = δ + s·c_N，c_N = ∫ z·Nφ(z)Φ(z)^(N−1) dz。數值：c_2 = 0.56、c_5 = 1.16、c_10 = 1.54、c_20 = 1.87、c_40 = 2.16、c_100 = 2.51。
3. 上界：對任何 t > 0，由 Jensen，exp(t·E[max Z]) ≤ E[exp(t·max Z)] ≤ Σ_k E[exp(t Z_k)] = N·exp(t²/2)。取對數得 E[max Z] ≤ (ln N)/t + t/2，取 t = √(2 ln N) 得 E[max Z] ≤ √(2 ln N)。∎

補充：雜訊等相關 ρ 時（Z_k = √ρ·W + √(1−ρ)·V_k），max Z = √ρ·W + √(1−ρ)·max V，所以 E[max Z] = √(1−ρ)·c_N。設定越像，高估越少，但不會是 0。static 的 s 約 2–5 bp/日，N = 20 時約 4–9 bp/日。

## 六、檔案

| 檔案 | 用途 |
|---|---|
| `compare.bat` | 一鍵執行 `compare_static.py`（雙擊或把檔案拖上去） |
| `compare_static.py` | static 上傳檔的比較（第三、五節） |
| `research\rerun.bat` | 重算第二節的表 |
| `others\` | 空的。組員的 static 上傳檔實際放在 `v6\` 根目錄：`submission-1-phase1_1.csv`（V2）、`submission-1-phase1 (2).csv`（V3）；`v9\train_and_predict_v9.py --static` 的報告從那裡讀 V3 |
| `research\s1_static_leaks.py` | 第二節；在 `research\` 執行 `python s1_static_leaks.py > results\s1_static_leaks.txt` |
| `research\compat.py` | 舊版 pandas 讀新版 pickle 用（目前 v6 沒有用到，保留給之後的研究程式） |
| `phase2.bat` | 第七節：雙擊、把檔案拖上去，或在 cmd 加參數 |
| `phase2_check.py` | 第七節的程式（下載價格、計分、累積） |
| `phase2\` | 第七節的輸出：`report_YYYYMMDD.txt`、`scores.csv`、`uploaded.csv`、下載的價格 `prices\`；`files\` 放要一起算的檔（例如組員每天的上傳檔） |

## 七、第二階段（Phase 2，dynamic leaderboard）的本地檢查（2026-10-27）

收盤後不用等課程 zip，就能算出當天每個上傳檔的 AR，和 daily board、accumulated board 對照。

### 7.1 怎麼用

| 要做的事 | 怎麼做 |
|---|---|
| 看今天（或最近一個交易日）的 AR | 13:30 收盤後約 1–2 小時，雙擊 `v6\phase2.bat`。各版本 `out\`（v9、v11 等）裡標頭是那一天的上傳檔全部會算，`v6\phase2\files\` 裡的檔也會算（例如組員每天的檔） |
| 算某幾個檔 | 把檔案拖到 `phase2.bat` 上 |
| 記下當天上傳的是哪個檔 | cmd：`"D:\ML RACE\v6\phase2.bat" --uploaded "D:\ML RACE\v11\out\submission_v11_1028.csv"`。之後累積表多一行 uploaded，可以和 accumulated board 上我們隊的總得分、平均分數對照（10/27 已經記好：`v11\out\submission_v9.4_1027_wt0_lam0.5.csv`） |
| 指定日期 | `phase2.bat --date 2026-10-27` |
| 重算所有 live 日、只看累積 | `phase2.bat --all`、`phase2.bat --board` |
| 課程資料已經有那天，仍要下載來核對 | `phase2.bat --check` |

輸出：畫面上的表；`phase2\report_YYYYMMDD.txt`（同一張表，加上每一支的報酬、預測和貢獻）；`phase2\scores.csv`（每天每個檔一個 row，累積表從這裡算）。

表的欄位：AR score（和排行榜一樣的小數，6 位）、AR bp、timing、xsec（拆法同第三節）、猜 1 的比例、猜對幾支。最上面兩行是排行榜的兩個 baseline。

### 7.2 價格從哪裡來

- 課程資料已經有那一天：用課程資料（`data\market\`，或 `downloads\`、自己的「下載」資料夾裡最新的 `data*.zip`；zip 只讀，不會改 `data\`），和排行榜用的是同一份。
- 沒有的話自動下載：FinMind TaiwanStockPrice（每支一次，約 20 秒）；FinMind 失敗才用 TWSE MI_INDEX（上市）加 TPEx 每日收盤行情（上櫃：8069、3293）。完整的下載存在 `phase2\prices\prices_YYYY-MM-DD.csv`，之後不再重抓；課程 zip 進來以後，會自動和存下來的價格比對，報酬不一樣的股票會列出來。
- FinMind 的原始價和課程資料一致：8069、3293、3661 在 10/20、10/21、10/22、10/27 的 close/open，和課程的 adj_close/adj_open 差 0 到 2×10⁻¹⁶（同一天的還原因子相同，比例不受影響）。
- 13:30 以前不下載。收盤後價格還沒公布時，FinMind 先問第一支，沒有就停（省下另外 49 次），改顯示前一個有上傳檔的交易日。
- FinMind 一小時有次數上限，超過時會提示設定 `FINMIND_TOKEN`（和夜盤用的同一個）。

### 7.3 算法和驗證

- R = (2p − 1)(close − open)/open；AR score = 50 支（`sample_submission.csv`）的平均，就是 daily board 上的數字；accumulated board 的總得分 = 每天 AR score 的和，平均分數 = 總得分 / 天數。
- baseline：Always down = 全部猜 0；Previous day = 每支猜它前一個交易日的 y（close > open）。前一個交易日取資料裡最近、有 8 成以上股票有價格的一天，所以 10/27 的前一天是 10/22（10/23 颱風休市沒有資料）。
- 和排行榜截圖（`leaderboard\20261022_*.png`）逐位相同：10/22 Always down 0.015102、Previous day 0.015561；10/12–10/22 九天累積 Always down −0.014105、Previous day 0.005455。
- 雲端用模擬的網路回應測過：FinMind 正常、FinMind 連不上改用 TWSE＋TPEx、價格還沒公布、收盤前、課程 zip 晚到再比對、同一個檔用相對和絕對路徑各給一次、缺股票或不是 0/1 的檔、未來日期的檔，結果都符合設計。FinMind 和 TWSE 的回應格式是用實際回應確認的（FinMind 10/27、TWSE 10/22）；TPEx OpenAPI 的欄位名稱沒有實際確認過，只有 FinMind 失敗時才會用到，對不上時會印出它的欄位名稱，不會算錯。

### 7.4 第一筆：2026-10-27

當天 Top50 平均盤中報酬 −107.30 bp（50 支只有 18 支收高於開），是 2020 年以來 1,654 天裡第 135 低。

| 檔案 | AR score | bp | timing | xsec | 猜 1 | 猜對 |
|---|---|---|---|---|---|---|
| baseline Always down | 0.010730 | 107.30 | 107.30 | 0 | 0 | 32/50 |
| baseline Previous day | 0.004795 | 47.95 | 30.04 | 17.91 | 0.36 | 28/50 |
| 上傳的 `v11\out\submission_v9.4_1027_wt0_lam0.5.csv`（w_T 0、λ 0.5） | −0.006980 | −69.80 | −25.75 | −44.05 | 0.62 | 19/50 |
| `v9\out\submission_v9_1027.csv` | −0.007953 | −79.53 | −55.80 | −23.74 | 0.76 | 18/50 |
| v11 預設（雲端試跑的檔，不在 `v11\out\`） | −0.008429 | −84.29 | −77.26 | −7.03 | 0.86 | 19/50 |

原因（美股晚了兩晚只差約 4 bp，主要是跳空開高後一路殺低）見 `v11\results\rerun_1027_us\README.txt`。

## 參考文獻

- Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality. *Journal of Portfolio Management*, 40(5), 94–107.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
- Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313.
- 課程投影片 `docs\Financial_Prediction_with_SDAI_Motivation.pdf`：Data Leakage（Future information in features、In-sample upstream selection）、Walk-Forward Evaluation。

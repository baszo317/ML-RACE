# v6：第一階段（static leaderboard）專用

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
| `others\` | 組員的 static 上傳檔（比較用，例如 `submission-1-phase1_1.csv`） |
| `research\s1_static_leaks.py` | 第二節；在 `research\` 執行 `python s1_static_leaks.py > results\s1_static_leaks.txt` |
| `research\compat.py` | 舊版 pandas 讀新版 pickle 用（目前 v6 沒有用到，保留給之後的研究程式） |

## 參考文獻

- Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality. *Journal of Portfolio Management*, 40(5), 94–107.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
- Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313.
- 課程投影片 `docs\Financial_Prediction_with_SDAI_Motivation.pdf`：Data Leakage（Future information in features、In-sample upstream selection）、Walk-Forward Evaluation。

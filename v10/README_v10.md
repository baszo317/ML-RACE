# v10：把 Qlib Alpha158 加進 v9 的 ablation（2026-10-26 完成：V10 沒有通過事先寫好的檢驗，live 維持 v9）

使用者的要求：「爬 Alpha158，加進 v9 的 ablation 實驗，看哪些能改善效能；t 值大於 0 的都可以當 V10」。
規則在訓練前寫好：`research\PREREG_v10.md`。live 仍是 v9，這個資料夾只有研究。結果見第 5 節。

## 1. Alpha158 從哪來、怎麼算

- 來源：microsoft/qlib（commit 5435523，2026-10-08）`qlib/contrib/data/loader.py` 的 `Alpha158DL.get_feature_config`，
  用 `handler.py` 裡 `Alpha158` 的預設設定：kbar 9 個、price 4 個（OPEN0 HIGH0 LOW0 VWAP0）、rolling 29 個 operator × 5 個
  window（5、10、20、30、60）= 158 個。原始碼文字存在 `research\qlib_alpha158_source.txt`、`research\qlib_ops_source.txt`（只當規格讀，沒有執行）。
- 實作：`alpha158.py`（只要 numpy、pandas）。輸入只用競賽檔：`$open $high $low $close` = `etl:adj_*`；`$volume` = `price:成交股數`；
  `$vwap` = `price:成交金額 / price:成交股數` × 還原因子。
- 時間：第 t 天的值用到第 t 天收盤，所以每個 feature 往後移一個 row（同一支股票）：row D 只用 D−1 以前的資料，和 B 組 feature 相同。
- 和 Qlib 不同的兩處：
  - volume 用原始股數（Qlib 會除以還原因子；這裡只會讓比例型 feature 在除權息日跳幾 %）。
  - VWAP 的還原因子要估：課程的 adj 價格是後復權（2330 在 2020 年約 1.92 倍、2026 年約 2.21 倍），每天
    adj_low / 原始 VWAP ≤ 因子 ≤ adj_high / 原始 VWAP；把除權息以後的天數交集起來，取中點（只用過去的天數）。已知的除權息因子
    （`data\ref\corporate_actions.parquet`、`exdiv_history.csv`）用來跨過除權息日；和前面矛盾的單日先擱置，連兩天矛盾才重來。
    誤差上界（交集半寬）中位數 0.1%、99% 0.8%；VWAP 最後夾在當天 [low, high] 之內。

## 2. 驗證（`research\check_alpha158.py`，輸出 `results\check_alpha158.txt`）

- 公式：158 個 expression 的模板逐字出現在 Qlib 原始碼裡，名稱和順序相同。
- 數值：另外寫一個照 `qlib/data/ops.py` 與 `rolling.pyx` 逐行移植的版本（pandas rolling min_periods=1、Corr/Rsquare 的 std≈0 遮罩、
  Cython deque 的 Slope/Resi/Rsquare），6 支股票全期比對：143 個 feature 完全相同；BETA、RSQR、RESI 的 15 個差 < 4e-9 個標準差，
  用有理數精確計算確認差異來自 Qlib 累加式更新的捨入（Qlib 誤差 7e-13 × 價格，這裡 1e-17）。
- 時間對齊：row D 等於 D−1 的值；和 v9 panel 的 B feature 一致（KMID = exp(intraday_ret) − 1、ROC5 = exp(−ret5)，差 < 2e-15）。

## 3. 實驗（PREREG_v10.md）

- 23 個資訊群組：KBAR、PRICE、ROC、MA、STD、BETA、RSQR、RESI、MAX、MIN、QTLU、QTLD、RANK、RSV、IDX（IMAX+IMIN+IMXD）、CORR、CORD、
  CNT（CNTP+CNTN+CNTD）、SUM（SUMP+SUMN+SUMD）、VMA、VSTD、WVMA、VSUM（VSUMP+VSUMN+VSUMD）。互為線性轉換的 operator 合成一組。
- 兩個 family：L = 選股 LightGBM `lgb_res`（B 67 個 + 群組，seed 1–5）；M = 選股 MLP `mlp_res_xr9`、`mlp_res_BN9`、`mlp_res_xrN9`
  （各自 v9 的 feature + 群組，seed 1–5）。每個候選只換該 family 的成員，其他用 v9 的 walk-forward 預測，v9 的規則（w_T 0.5、除權息 4 天、
  對沖 λ 0.75）算 Top50 bp/日，和同 family 重訓的 baseline 成對比較。
- 入選：val（2022–24）t > 0。V10 = L、M 各自加上所有入選群組，完整 seed 重訓；檢驗：V10 對 v9 replica 的 hold、post 差都 > 0。

## 4. 怎麼跑（2026-10-26 實際的做法）

| 步驟 | 在哪 | 指令 | 實際時間 |
|---|---|---|---|
| L 篩選（750 個 LightGBM） | 雲端 | `python a158_screen.py --family lgb --workers 2` | 134 分鐘（2 核） |
| M 篩選（2250 個 MLP） | 筆電 | 雙擊 `screen_mlp.bat`（可中斷、再雙擊會接著跑） | 125 分鐘（4 個 worker） |
| 評估、寫入選名單 | 雲端 | `cd research`，`python a158_eval.py > ..\results\a158_screen_eval.txt` | 2 分鐘 |
| V10 的 lgb_res（30 個） | 雲端（和 L 篩選同一台） | `python a158_screen.py --family v10 --members lgb_res --workers 2` | 13 分鐘 |
| V10 的 MLP、v9 replica 的 MLP seed 6–10（270 個）與檢驗 | 筆電 | 雙擊 `stage2.bat`（雲端的 LightGBM 檔已放進 `cache\screen\`，所以會跳過） | 18.5 分鐘 + 1 分鐘 |

- 第一次執行會用 `v9\cache\core9.pkl` 和 `data\` 建 `cache\panel10.pkl`（約 370 MB）。訓練輸出在 `cache\screen\`（每個 unit 一個 .npz）。
- M 的結果用 `a158_eval.family_preds('mlp')` 寫成 `cache\screen_mlp_preds.csv.gz`（seed 平均）再拿到雲端評估；L 的 seed 平均在 `cache\screen_lgb_preds.csv.gz`。
- 筆電上只有 L 的 base 和 V10 兩組單一 seed 檔；其他 24 組（23 個群組和 ALL）只有 seed 平均。需要單一 seed 的檔可以在雲端用同一個指令重跑（雲端同一個 unit 重訓兩次差 0）。
- `--members` 是 10/26 加的選項：只訓練指定的成員，不給就和原本一樣。
- 晚上 17:00 以後要跑 `v9\evening.bat`，screen 同時在跑會讓它變慢，可以先關掉 screen 視窗，之後再雙擊接著跑。

## 5. 結果（2026-10-26）

**結論：照事先寫好的檢驗，V10 沒有推廣出去（hold 差 −0.45 bp/日 ≤ 0），live 維持 v9。** 全期（2022-01 到 2026-10）V10 對 v9 replica
是 −0.13 bp/日（t −0.37）。輸出：`results\a158_screen_eval.txt`（篩選表）、`results\v10_selection.json`、`results\v10_eval.txt`（檢驗）。

### 5.1 執行與檢查

- L 篩選：第一次在雲端跑完的結果沒有寫回筆電（雲端環境已回收），用筆電的 `cache\panel10.pkl` 在雲端重跑：750 個、0 失敗
  （lightgbm 4.7.0、pandas 3.0.5）。M 篩選：筆電 2250 個、0 失敗。Stage 2：雲端 30 個 LightGBM、筆電 270 個 MLP，0 失敗。
- v9 從 cache 重算：val 13.99、hold 14.33、post 16.59、all 14.31，和 `v9\results\v9_eval.txt` 相同。
- L baseline 對 v9 的 `lgb_res`：2022、2023、2024、2026-01 四個 fold 完全相同（差 5e-8，是 cache 的捨入），2025、2026-06 有小差異
  （corr 0.99976、0.99963），和第一次雲端的結果一樣，原因仍不明。這點差異讓 hold 從 14.33 變 13.93（−0.41，t −2.04）。
- M baseline（seed 1–5）對 v9 的成員（seed 1–10）：corr 0.996–0.997。
- 兩個雜訊尺度（都不是 feature 造成的）：
  - LightGBM 在 2 個 fold 有極小差異（上面 L baseline 那條，原因不明）：hold 差 0.41 bp/日。
  - v9 replica 的 MLP 用 seed 1–5 或 seed 1–10：post 差 1.21 bp/日（t −1.44）。

### 5.2 篩選（val 2022–24 成對 t > 0 入選；粗體 = 入選）

| 群組 | 個數 | L val 差（t） | L hold | L post | M val 差（t） | M hold | M post |
|---|---|---|---|---|---|---|---|
| KBAR | 9 | **+0.39（+1.36）** | +0.04 | +1.08 | −0.08（−0.35） | +0.41 | +1.12 |
| PRICE | 4 | −0.13（−0.43） | −0.18 | +0.74 | −0.44（−1.88） | −0.34 | +1.17 |
| ROC | 5 | −0.03（−0.12） | −0.08 | −0.70 | −0.22（−1.01） | −0.01 | +0.46 |
| MA | 5 | **+0.34（+1.15）** | −0.03 | +0.21 | −0.12（−0.61） | −0.08 | +1.09 |
| STD | 5 | **+0.29（+1.08）** | +0.36 | −0.42 | −0.04（−0.18） | +0.64 | +2.97 |
| BETA | 5 | −0.01（−0.04） | −0.20 | −0.22 | −0.08（−0.38） | −0.05 | +1.45 |
| RSQR | 5 | **+0.21（+0.82）** | +0.09 | −0.17 | **+0.04（+0.16）** | +0.11 | +1.74 |
| RESI | 5 | **+0.23（+0.74）** | +0.19 | −0.08 | −0.01（−0.05） | −0.09 | +1.14 |
| MAX | 5 | **+0.47（+1.81）** | −0.03 | −0.15 | −0.10（−0.46） | +0.48 | +1.02 |
| MIN | 5 | **+0.45（+1.60）** | +0.47 | +0.34 | **+0.08（+0.30）** | −0.05 | +1.00 |
| QTLU | 5 | **+0.24（+0.84）** | +0.34 | +0.99 | −0.14（−0.66） | +0.32 | +1.87 |
| QTLD | 5 | **+0.01（+0.04）** | −0.17 | +0.93 | −0.09（−0.37） | −0.32 | +0.85 |
| RANK | 5 | **+0.27（+1.08）** | −0.21 | +0.16 | −0.06（−0.25） | −0.49 | +1.45 |
| RSV | 5 | **+0.43（+1.76）** | −0.07 | −0.71 | **+0.00（+0.002）** | −0.16 | +1.61 |
| IDX | 15 | **+0.05（+0.21）** | +0.10 | +0.14 | **+0.16（+0.61）** | +0.01 | −0.02 |
| CORR | 5 | **+0.09（+0.35）** | +0.45 | −1.07 | −0.05（−0.22） | +0.07 | +0.08 |
| CORD | 5 | **+0.10（+0.38）** | +0.35 | −1.45 | −0.10（−0.46） | +0.28 | +2.08 |
| CNT | 15 | **+0.34（+1.27）** | +0.26 | +0.56 | −0.22（−0.89） | −0.00 | +0.97 |
| SUM | 15 | −0.04（−0.14） | +0.21 | −0.84 | −0.23（−1.00） | +0.72 | +2.56 |
| VMA | 5 | **+0.59（+2.57）** | +0.18 | −0.81 | −0.17（−0.70） | +0.28 | +1.30 |
| VSTD | 5 | **+0.14（+0.59）** | +0.41 | −0.74 | −0.00（−0.02） | +0.79 | +0.23 |
| WVMA | 5 | **+0.39（+1.53）** | +0.39 | +0.13 | **+0.37（+1.48）** | +0.33 | +1.24 |
| VSUM | 15 | **+0.34（+1.47）** | −0.23 | +0.01 | −0.14（−0.59） | +0.18 | +2.30 |
| ALL（只報告） | 158 | −0.46（−1.21） | −0.70 | +3.08 | −0.00（−0.01） | +0.61 | +4.25 |

單位 bp/日，和同 family 的 baseline 成對比較（L baseline：val 13.99、hold 13.93、post 16.73；M baseline：14.03、13.93、15.72）。

- 入選：L 19 組（沒入選：PRICE、ROC、BETA、SUM）；M 5 組（RSQR、MIN、RSV、IDX、WVMA；RSV 的 t 只有 0.002）。
- val 的排名幾乎預測不了其他期間：t_val 和 t_hold 的相關係數 L +0.16、M +0.14；和 t_post 的相關係數 L +0.05、M −0.12。
  入選組的 hold 平均差 L +0.15（沒入選 −0.06）、M +0.05（沒入選 +0.16）。
- 同一個 family 會一起偏：L 有 19/23 組 val 為正，M 有 18/23 組 val 為負，因為候選都和同一個 baseline 比。
  M 有 22/23 組 post 為正，這大部分可以用 baseline 解釋：5 個 seed 的 M baseline 在 post 剛好偏低（比 10 個 seed 低約 0.9–1.2）。

### 5.3 V10 檢驗（`results\v10_eval.txt`）

V10 = `lgb_res` 加 L 的 19 組（67 → 196 個 feature），三個 MLP 加 M 的 5 組（96 → 131、71 → 106、106 → 141 個）。
LightGBM seed 1–5、MLP seed 1–10；v9 replica 用同樣的機器與 seed（LightGBM 在雲端、MLP 在筆電）。

| 設定 | val | hold | post | all |
|---|---|---|---|---|
| v9 replica | 13.99 | 13.93 | 16.73 | 14.20 |
| V10 | 13.78 | 13.47 | 18.35 | 14.08 |
| V10 − replica（t） | −0.21（−0.50） | **−0.45（−0.75）** | **+1.61（+1.07）** | −0.13（−0.37） |
| 只改 L − replica（t） | −0.08（−0.21） | −0.68（−1.55） | +1.38（+1.08） | −0.13（−0.46） |
| 只改 M − replica（t） | +0.11（+0.38） | −0.11（−0.22） | +1.44（+1.49） | +0.16（+0.64） |
| ALL 158（M seed 1–5，只報告）− replica | −0.89（−1.74） | −0.03（−0.04） | +2.54（+1.66） | −0.35（−0.86） |

- 事先寫好的檢驗：hold −0.45、post +1.61，hold 不大於 0 → 沒有推廣，live 維持 v9。
- 單組 t > 0 的群組合起來，連挑選用的 val 都沒有變好（−0.21）。
- 依年份（V10 − replica）：2022 −0.21、2023 −0.11、2024 −0.31、2025 −0.73、2026 年 1–5 月 +0.26、post +1.61。
- post 的 5 個月都是正的（6 月 +1.87、7 月 +1.32、8 月 +3.49、9 月 +0.20、10 月 +0.93），static（6/1–9/30）17.49 對 15.76、
  10 月 23.12 對 22.19。不過 post 只有 99 天，而且是同一組凍結的模型；每日差的 sd 約 15 bp，t 1.07。
  光是 MLP 的 seed 數不同，post 就能差 1.2。要用這個幅度得到 t = 2，大約需要 350 個交易日。
- 只報告，不改變結論：hold、post 在 v1–v9 都看過，乾淨的只有 10/27 以後的 live。

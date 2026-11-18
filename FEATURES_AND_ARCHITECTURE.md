# 模型、feature 與架構

整理日期：2026-11-11。依 `D:\ML RACE` 的程式整理：`v9\rolling9.py`、`v9\train_and_predict_v9.py`、`v11\train_and_predict_v11.py`、`transformer\comp_features.py`、`transformer\tfmodel.py`、`v5\ens_lib.py`、`v5\features_extra.py`、`v9\features_v3.py`、`v8\hedge_beta.py`。每天跑的程式最後一次修改是 10/27；之後的 v12–v16 都只是研究，沒有改到這套。

名詞：
- 觀點：大盤觀點（明天整體偏漲還是偏跌）和選股觀點（哪些股票比當天平均好）。
- 成員：一個觀點裡的單一模型。
- 訓練點：某一天用「到前一個交易日為止」的資料訓練出來的一批模型。

## 0. 總覽

```
競賽資料（market 100 支、global）   台指期夜盤（FinMind）   除權息預告   擴充股票池（只用來訓練）
                │
                ▼
feature panel：每個（日期, 股票）一個 row，共 111 個 feature，只用 D 日 09:00 前拿得到的資料
                │
                ├── 大盤觀點（label：收 > 開）
                │     MLP：mlp_raw_B9、mlp_raw_xr9、mlp_raw_w4_9
                │     Transformer：tf_raw_B
                │     p_raw = (3 個 MLP 的平均 + Transformer) / 2
                │     pbar = 當天 100 支 p_raw 的平均
                │
                ├── 選股觀點（label：盤中報酬贏過當天 100 支的平均）
                │     MLP：mlp_res_B9、mlp_res_xr9、mlp_res_BN9、mlp_res_xrN9
                │     LightGBM：lgb_res（v11、W0 再加 X1 = lgb_res_xrN）
                │     p_e = (4 個 MLP 的平均 + LightGBM 的平均) / 2
                │     p_e_c = p_e − 當天 100 支的平均
                │
                ▼
score = w_T × (pbar − 0.5) + p_e_c
        （v11：T2 用夜盤調整 pbar；T3 每月前 3 個交易日多 5 支猜 1）
                │
                ▼
除權息當天、前一天、後一天、後兩天 → 猜 0
                │
                ▼
beta 對沖（λ）→ Top50 的 0/1 → submission csv
```

## 1. 版本

| | v9 | v11 預設 | W0 |
|---|---|---|---|
| 怎麼產生 | `v9\daily.bat` | `v11\daily_v11.bat` | `v11\custom_v11.bat`，或 `train_and_predict_v11.py --w-t 0 --lam 0.5` |
| 上傳檔 | `v9\out\submission_v9_MMDD.csv` | `v11\out\submission_v11_MMDD.csv` | `v11\out\submission_v11_MMDD_wt0_lam0.5.csv` |
| 大盤觀點權重 w_T | 0.5 | 0.5 | 0 |
| beta 對沖 λ（1 = 不對沖） | 0.75 | 0.75 | 0.5 |
| T2 夜盤反轉 | 無 | 權重 0.5 | 權重 0.5，但 w_T = 0，所以沒作用 |
| T3 月初多 5 支猜 1 | 無 | 有 | 有 |
| X1 第二個選股 LightGBM | 無 | 有 | 有 |
| 會影響 0/1 的成員 | 9 個 | 10 個 | 6 個（只有選股觀點） |
| walk-forward 全期（Top50 bp/日） | 14.31 | 15.02 | 12.22 |
| 每日 sd（bp） | 48.3 | 48.0 | 38.4 |

- walk-forward 是 2022-01 到 2026-10-22，共 1,164 天，全部是樣本外。
- 10/27 起上傳的是 W0。
- static leaderboard 上傳的是 `v9\out\submission_v9_static_w025.csv`（v9 的 rolling static，w_T 0.25）。

## 2. 資料

| 來源 | 檔案 | 內容 | 什麼時候能用 |
|---|---|---|---|
| 競賽個股 | `data\market\<代號>.csv`，Top100 共 100 個檔 | 還原開高低收、成交筆數 / 金額 / 股數、全體外資及陸資持股比率、外陸資 / 投信 / 自營商（自行買賣）/ 自營商（避險）買賣超股數、RSI、SMA（30 日）、MACD、ATR、Bollinger 上中下軌、y | D 日的值收盤後才有，D+1 才能用 |
| 競賽全球 | `data\global\global_data.csv` | ^VIX、^IXIC（Nasdaq）、^GSPC（S&P 500）、^TWII（加權指數）、^KS11（KOSPI）、TX_OI（外資台指期淨未平倉口數） | 美股三個：row D 已經是前一晚的收盤，D 日就能用。^TWII、^KS11、TX_OI：row D 是當天收盤，D+1 才能用 |
| 股票清單 | `data\info.json` | Top100@Y：算 feature 和「當天平均」用的 100 支；Top50@Y：要預測的 50 支 | |
| 交易日曆、除權息 | `data\ref\` 的 `calendar.parquet`、`closures_manual.csv`、`corporate_actions.parquet`、`exdiv_history.csv`、`exdiv_upcoming.csv`、`exdiv_manual.csv` | 交易日（扣掉臨時休市）、除權息日 | 除權息日數週前就會公告 |
| 台指期夜盤 | `v7\data\tx_daily.csv`（FinMind，`daily.bat` 每天更新） | 台指期日盤、夜盤的開高低收和成交量 | D−1 15:00 到 D 05:00 的夜盤，D 日早上可用 |
| 擴充股票池 | `v9\data\panel_ext.parquet` | 約 210 支其他上市股（每年取前一年底市值前 300 名），資料到 2026-07 | 只用來訓練 mlp_res_B9 |

- 要預測的那一天，課程 zip 會放一個所有欄位都是 −1 的 row。程式把它當成要預測的日子，數值當缺值。
- 目前（11/10 的 daily 之後）有 label 的資料：2020-03-02 到 2026-11-09，共 1,629 天；100 支合計 162,848 row。

## 3. Label、樣本權重、時間規則

- r = (close − open) / open，用還原價計算的盤中報酬。
- 大盤觀點的 label：y = 1{close > open}。
- 選股觀點的 label：y_resid = 1{e > 0}，e = r − 當天 100 支 r 的平均（殘差報酬）。
- 比賽分數 AR：Top50 平均的 (2·pred − 1) × r，單位 bp/日。
- 樣本權重（最後都除以平均，讓平均等於 1）：
  - MLP：min(|r|, 5%) / σ^0.5。大盤成員的 r 是盤中報酬、σ = absr_60；選股成員把 r 換成 e，σ = absres_60。
  - Transformer：min(|r|, 5%)。
  - LightGBM：min(|e|, 5%)。
  - 權重和 |r| 成正比，等於把 AR 這個分數直接變成加權分類問題。
- 時間規則：row D 的每個 feature 只用 D 日 09:00 以前公布的資料。
  - 個股資料用到 D−1：整張 feature 表往後 shift 一天。
  - ^TWII、^KS11、TX_OI 用 D−1 收盤。
  - 美股用前一晚，名稱有 `_lag1` 的是前兩晚。
  - 夜盤用 D−1 15:00 到 D 05:00。

## 4. 成員

| 成員 | 觀點 | 模型 | feature（數量） | 訓練資料 | 樣本權重 | 每天新訓練 | 預測時平均幾個模型 |
|---|---|---|---|---|---|---|---|
| mlp_raw_B9 | 大盤 | MLP | B9 + NIGHT + SIG（68） | 2020-03-02 到前一個交易日，100 支 | min(\|r\|, 5%) / absr_60^0.5 | 2 個 seed | 10 |
| mlp_raw_xr9 | 大盤 | MLP | B9 + XR + NIGHT + SIG（98） | 同上 | 同上 | 2 | 10 |
| mlp_raw_w4_9 | 大盤 | MLP | B9 + NIGHT + SIG（68） | 只用最近 4 年 | 同上 | 2 | 10 |
| tf_raw_B | 大盤 | Transformer（20 天序列） | B（67） | 2020-03-02 起，100 支 | min(\|r\|, 5%) | 1 | 5 |
| mlp_res_B9 | 選股 | MLP | B9 + SIG（66） | 2020-03-02 起，100 支 + 擴充股票池 | min(\|e\|, 5%) / absres_60^0.5 | 2 | 10 |
| mlp_res_xr9 | 選股 | MLP | B9 + XR + SIG（96） | 2020-03-02 起，100 支 | 同上 | 2 | 10 |
| mlp_res_BN9 | 選股 | MLP | B9 + NEW + SIG（71） | 同上 | 同上 | 2 | 10 |
| mlp_res_xrN9 | 選股 | MLP | B9 + XR + NEW + XRN + SIG（106） | 同上 | 同上 | 2 | 10 |
| lgb_res | 選股 | LightGBM | B（67） | 2020-03-02 起，100 支 | min(\|e\|, 5%) | 1 | 5 |
| lgb_res_xrN（X1） | 選股 | LightGBM | B9 + XR + NEW + XRN + SIG（106） | 同上 | 同上 | 1 | 5 |

- 模型檔：
  - 前 9 個成員在 `v9\models_v9\<成員>\`，由 v9 的 evening / daily 訓練。
  - X1 在 `v11\models_v11\lgb_res_xrN\`，由 v11 的 evening / daily 訓練。
- 使用：v9 用前 9 個；v11 和 W0 再加 X1。W0 的 w_T = 0，所以 4 個大盤觀點成員雖然照樣計算，但不影響結果。

## 5. 模型架構與訓練

### 5.1 MLP（7 個成員）

```
x（當天那一 row，n 個 feature）
→ Linear(n, 64) → GELU → Dropout(0.2)
→ Linear(64, 32) → GELU → Dropout(0.2)
→ Linear(32, 1) → sigmoid = P(label = 1)
```

參數量 = 64n + 2,177：

| n | 成員 | 參數量 |
|---|---|---|
| 66 | mlp_res_B9 | 6,401 |
| 68 | mlp_raw_B9、mlp_raw_w4_9 | 6,529 |
| 71 | mlp_res_BN9 | 6,721 |
| 96 | mlp_res_xr9 | 8,321 |
| 98 | mlp_raw_xr9 | 8,449 |
| 106 | mlp_res_xrN9 | 8,961 |

### 5.2 Transformer（tf_raw_B）

```
X（同一支股票最近 20 個交易日的 B 組，20 × 67）
→ Linear(67, 32) + 可學的 positional embedding（20 × 32）
→ TransformerEncoder × 2（d_model 32、4 個 head、d_ff 64、dropout 0.2、pre-LayerNorm、GELU）
→ LayerNorm
→ 串接「最後一天的向量」和「20 天的平均」（64 維）
→ Dropout(0.2) → Linear(64, 1) → sigmoid
```

- 參數量 20,033。
- 要有至少 20 天歷史的 row 才能進模型。

### 5.3 LightGBM（lgb_res、lgb_res_xrN）

| 參數 | 值 |
|---|---|
| objective | binary |
| learning_rate | 0.02 |
| num_leaves | 15 |
| min_data_in_leaf | 500 |
| feature_fraction | 0.8 |
| bagging_fraction / bagging_freq | 0.8 / 1（每輪都抽） |
| lambda_l2 | 1.0 |
| 最多輪數 | 1000 |
| early stopping | 100 輪 |

### 5.4 前處理

- MLP、Transformer：
  - 用訓練窗的 row 算每個 feature 的 1% / 99% 分位數，先截尾。
  - 再用截尾後的平均、標準差標準化；缺值補 0（= 訓練平均）。
  - 這組 scaler 存在模型檔裡，預測時用同一組。
- LightGBM：不標準化，缺值由 LightGBM 自己處理。

### 5.5 訓練流程（每個模型都一樣）

1. 訓練窗：從 2020-03-02（mlp_raw_w4_9 是 4 年前）到前一個交易日，只用有 label 的 row。
2. 把訓練窗最後 120 個交易日切出來當驗證：
   - MLP、Transformer：AdamW（lr 1e-3、weight decay 1e-2）、batch 1024、gradient clipping 1.0、加權 BCE。最多 10 個 epoch；驗證的加權 log-loss 連續 2 個 epoch 沒進步就停（patience 2），記下最好的 epoch 數。
   - LightGBM：最多 1000 輪；驗證的 log-loss 連續 100 輪沒進步就停，記下最好的輪數。
3. 用同一個 seed、整個訓練窗（包含最後 120 天），從頭重訓到剛才記下的 epoch 數或輪數，存成模型檔。

目前模型池（最近 5 個訓練點，10/28 到 11/09）實際停在：

| 成員 | 停在 |
|---|---|
| mlp_raw_B9 | 1–5 個 epoch，平均 1.7 |
| mlp_raw_xr9 | 1–4 個 epoch，平均 1.6 |
| mlp_raw_w4_9 | 全部 1 個 epoch |
| tf_raw_B | 1–3 個 epoch，平均 1.6 |
| mlp_res_B9 | 1–6 個 epoch，平均 3.2 |
| mlp_res_xr9、mlp_res_BN9、mlp_res_xrN9 | 全部 1 個 epoch |
| lgb_res | 72–135 棵，平均 98 |
| lgb_res_xrN | 115–176 棵，平均 139 |

### 5.6 Rolling pool（每天怎麼更新）

- 晚上的 evening（沒跑的話就早上的 daily）用到前一個交易日為止的資料訓練新模型：每個 MLP 成員 2 個 seed，Transformer、LightGBM 各 1 個。seed 編號是日期加序號，例如 s26110901。
- 預測 D 日時，每個成員平均「最近 5 個訓練點」的全部模型，這些模型的訓練資料都在 D 日以前：
  - 每個 MLP 成員 10 個，Transformer 5 個，每個 LightGBM 5 個。
  - v9 一天約用 80 個模型，v11 約 85 個；W0 真正影響結果的是 50 個（4 個選股 MLP × 10 + 2 個 LightGBM × 5）。
- 筆電上 evening 約 7–8 分鐘（Transformer 最久），daily 只做預測，約 2 分鐘。

## 6. 合成與決策規則

### 6.1 大盤觀點

- p_raw(i) = (mlp_raw_B9、mlp_raw_xr9、mlp_raw_w4_9 的平均 + tf_raw_B) / 2。先在同一種模型裡平均，再把 MLP、Transformer 兩類平均。
- pbar = 當天 100 支 p_raw 的平均。

### 6.2 選股觀點

- p_e(i) = (4 個選股 MLP 的平均 + LightGBM 的平均) / 2。v9 的 LightGBM 只有 lgb_res；v11、W0 是 lgb_res 和 lgb_res_xrN 的平均。
- p_e_c(i) = p_e(i) − 當天 100 支 p_e 的平均。

### 6.3 分數

- score(i) = w_T × (pbar − 0.5) + p_e_c(i)。
- T2（v11，只動大盤觀點）：
  - pbar 先換成 pbar − 0.5 × sd(前 250 天的 pbar) × clip(z, −3, 3)。
  - z = 當天的夜盤報酬 / 前 250 個夜盤報酬的 sd。夜盤跌得越多，pbar 往上調得越多（反轉）。
  - 前 250 天的 pbar 存在 `v11\data\pbar_history.csv`。晚上的 evening 還沒有夜盤，所以 T2 只在早上的 daily 作用。
- T3（v11）：每月前 3 個交易日，把所有 score 一起往上移，讓 Top50 裡不受除權息規則影響、原本猜 0 的股票，多 5 支改成猜 1。T3 在對沖之前做。

### 6.4 除權息規則

- 當天除權息、明天除權息（今天是前一天）、昨天除權息、前天除權息的股票，一律猜 0。對應的欄位是 ex_today、ex_next1、ex_prev1、ex_prev2。
- 「明天」是用官方交易日曆找下一個交易日，再去查除權息預告；預告由 `v5\fetch_exdiv.py` 每天抓。

### 6.5 beta 對沖

1. β(i) = 個股盤中報酬對「當天 100 支平均盤中報酬」的 120 天 rolling beta（至少 60 天，只用前一天以前的資料），限制在 [0.2, 2.5]，缺值補當天的中位數。
2. 先照 score 給每支 s(i)：猜 1 是 +1，猜 0 是 −1；除權息規則的股票也是 −1。
3. 組合的 beta 曝險 E = mean(s·β) = s̄·β̄ + B，其中 B = mean[(s − s̄)(β − β̄)] 是選股觀點的「低 beta 傾斜」。選股觀點偏好低 beta 的股票，等於暗中押大盤跌。
4. 目標設成 E* = s̄·β̄ + λ·B：λ = 1 不對沖；0.75 抵銷 1/4 的傾斜；0.5 抵銷一半。
5. 在 c ∈ [−0.3, 0.3]（每 0.0025 一格）裡找一個 c：用「score + c·β > 0」重新決定（除權息規則仍然是 0），讓這組決策的 E 最接近 E*；一樣接近時選 |c| 最小的。這組決策就是最後的 0/1。

### 6.6 輸出與檢查

- `submission_*_MMDD.csv`：第一個 column 是 `id`（sample_submission.csv 的 50 支，順序相同），之後每個要預測的日期一個 column，值是 0 或 1。
- 同時會寫三個檔：
  - `probabilities_*`：每支的 p_raw、pbar、p_e、score、beta、對沖前後的決策。
  - `hedge_*`：每天的 c、對沖前後的傾斜。
  - `checks_*`：檢查結果。
- 備援：某支股票沒有 score 時，改用「過去 250 天平均盤中報酬 > 0 就猜 1」。正常不會用到，因為缺 score 會先出 FATAL。
- 每天的檢查（`v5\guards.py`，門檻在 `v9\train_and_predict_v9.py` 設定，v11 共用）：
  - FATAL（不寫上傳檔）：
    - 要預測的日子已經過了（用到舊 zip），或不是交易日。
    - 那一 row 還沒有前一晚的美股（00:00 打包的 zip）。
    - 最新有 label 的日子不是前一個交易日，或那天不是 100 支都有結果。
    - 夜盤以外的 109 個 feature 裡，有 feature 在超過 20% 的股票缺值。
    - 某個成員的模型池是空的。
    - 除權息預告超過 3 個交易日沒抓成功。
    - 輸出不合理：猜 1 的比例不在 0.10–1.00、pbar 不在 0.20–0.80，或 Top50 不是 50 支都有預測。
  - WARN（照常寫檔）：
    - 沒有夜盤（夜盤 feature 當缺值）。
    - 某個成員最新的模型沒有訓練到前一個交易日（用比較舊的模型池預測）。
    - zip 是預測日 05:00 以前打包的（美股可能不是最新的）。
    - 除權息預告今天沒抓成功，但 3 個交易日內成功過。

## 7. Feature 組和成員的對照

| 成員 | B9（64） | Nasdaq 3 個 | NIGHT（2） | SIG（2） | XR（30） | NEW（5） | XRN（5） | 合計 |
|---|---|---|---|---|---|---|---|---|
| mlp_raw_B9 | x | | x | x | | | | 68 |
| mlp_raw_xr9 | x | | x | x | x | | | 98 |
| mlp_raw_w4_9 | x | | x | x | | | | 68 |
| tf_raw_B | x | x | | | | | | 67 |
| mlp_res_B9 | x | | | x | | | | 66 |
| mlp_res_xr9 | x | | | x | x | | | 96 |
| mlp_res_BN9 | x | | | x | | x | | 71 |
| mlp_res_xrN9 | x | | | x | x | x | x | 106 |
| lgb_res | x | x | | | | | | 67 |
| lgb_res_xrN（X1） | x | | | x | x | x | x | 106 |

- B = B9 + 3 個 Nasdaq 報酬（us_ixic_ret1、us_ixic_ret5、us_ixic_ret1_lag1）。
- 全部成員合起來 111 個 feature；daily 檢查寫的「109 input features」是扣掉 NIGHT 的 2 個。
- 除權息旗標、日曆 feature（cal_*）、mkt_intra_20、mkt_absr_20 雖然在 panel 裡，但不是任何成員的輸入。

## 8. Feature 定義（111 個）

符號：
- O、H、L、C：還原開、高、低、收。
- V、AMT、N：成交股數、成交金額、成交筆數。
- r = (C − O) / O；gap = ln(O_d / C_{d−1})；y = 1{C > O}；e = r − 當天 100 支 r 的平均。
- D−1：預測日的前一個交易日。

B 組的個股 feature 都是用到 D−1 為止的資料算好，再放到 row D；下表的 C、O、H、L、V 等，沒寫日期的都是 D−1 的值。

### 8.1 B 組：67 個，只用競賽資料（`transformer\comp_features.py`）

**股票偏向（5）**

| feature | 定義 |
|---|---|
| intra_bias_20 | 過去 20 天 r 的平均（至少 10 天） |
| intra_bias_60 | 過去 60 天 r 的平均（至少 30 天） |
| intra_bias_250 | 過去 250 天 r 的平均（至少 120 天） |
| intra_bias_exp | 全部歷史 r 的平均（至少 60 天） |
| gap_bias_20 | 過去 20 天 gap 的平均（至少 10 天） |

**前一天的價量與技術指標（24）**

| feature | 定義 |
|---|---|
| ret1 | ln(C_{D−1} / C_{D−2}) |
| ret5 | ln(C_{D−1} / C_{D−6}) |
| ret20 | ln(C_{D−1} / C_{D−21}) |
| intraday_ret | ln(C_{D−1} / O_{D−1}) |
| gap | ln(O_{D−1} / C_{D−2}) |
| y_mean5 | D−5 到 D−1 的 y 平均（最近 5 天收紅的比例） |
| range | (H − L) / C |
| clv | (C − L) / (H − L)；H = L 時為 0.5 |
| upper_shadow | (H − max(O, C)) / C |
| lower_shadow | (min(O, C) − L) / C |
| vol_ratio | ln(V / V 的 20 日平均) |
| amt_ratio | ln(AMT / AMT 的 20 日平均) |
| trades_ratio | ln(N / N 的 20 日平均) |
| avg_trade_size | ln((V/N) / (V/N) 的 20 日平均) |
| RSI | 競賽的 RSI 欄位 |
| close_SMA30 | C / SMA − 1（SMA 是競賽的 30 日均線欄位） |
| MACD_hist_norm | MACD / C（競賽的 MACD 欄位） |
| ATR_pct | ATR / C |
| BB_pctb | (C − 下軌) / (上軌 − 下軌)，限制在 [−2, 3] |
| BB_width | (上軌 − 下軌) / 中軌 |
| vol_ratio_5_60 | 日報酬 ln(C_t / C_{t−1}) 的 5 日標準差 / 60 日標準差 |
| limit_up_prev | 1{ret1 > 9.3%}（前一天漲停） |
| limit_dn_prev | 1{ret1 < −9.3%}（前一天跌停） |
| streak_y | 到 D−1 為止 y 連續相同的天數，連紅為正、連黑為負 |

**三大法人（12）**：買賣超都除以 ADV（成交股數的 20 日平均）。

| feature | 定義 |
|---|---|
| foreign_hold | 全體外資及陸資持股比率 |
| foreign_hold_chg1 | 持股比率的 1 日變化 |
| foreign_hold_chg5 | 持股比率的 5 日變化 |
| foreign_net | 外陸資買賣超股數（不含外資自營商）/ ADV |
| foreign_net_5d | 外陸資 5 日買賣超合計 / ADV / 5 |
| trust_net | 投信買賣超股數 / ADV |
| trust_net_5d | 投信 5 日買賣超合計 / ADV / 5 |
| dealer_self_net | 自營商（自行買賣）買賣超股數 / ADV |
| dealer_self_net_5d | 自營商（自行買賣）5 日合計 / ADV / 5 |
| dealer_hedge_net | 自營商（避險）買賣超股數 / ADV |
| dealer_hedge_net_5d | 自營商（避險）5 日合計 / ADV / 5 |
| foreign_streak | 外陸資連續買超（正）或連續賣超（負）的天數 |

**產業（2）**

| feature | 定義 |
|---|---|
| group | 0 金融（28 開頭、5871、5876、5880）、1 半導體、2 電子（其餘的股票）、3 傳產、電信、航運與其他。Top100 裡分別是 17、19、35、29 支，Top50 裡是 13、7、16、14 支 |
| is_fin | 1{group = 0} |

**昨天 100 支的橫斷面（5）**

| feature | 定義 |
|---|---|
| mkt_intra_prev | D−1 的 100 支平均 r |
| breadth_prev | D−1 收紅（y = 1）的比例 |
| mkt_intra_5 | D−5 到 D−1，每天 100 支平均 r 再平均 |
| sector_intra_prev | D−1 同一個 group 的平均 r |
| rel_intra_prev | 自己 D−1 的 r − mkt_intra_prev |

**台股、韓股指數（4）**：用到 D−1 收盤。

| feature | 定義 |
|---|---|
| twii_ret1 | 加權指數 1 日 log 報酬 |
| twii_ret5 | 加權指數 5 日 log 報酬 |
| ks11_ret1 | KOSPI 1 日 log 報酬 |
| ks11_ret5 | KOSPI 5 日 log 報酬 |

**外資台指期淨未平倉（3）**：D−1 收盤後的值，單位萬口。

| feature | 定義 |
|---|---|
| txoi | TX_OI / 10,000 |
| txoi_chg | TX_OI 的 1 日變化 / 10,000 |
| txoi_chg5 | TX_OI 的 5 日變化 / 10,000 |

**美股（12）**：row D 的美股是前一晚的收盤；`_lag1` 是再往前一筆（前兩晚）。

| feature | 定義 |
|---|---|
| us_vix | VIX 收盤 |
| us_vix_chg | ln(VIX / 前一筆 VIX) |
| us_vix_chg_lag1 | 前兩晚的 us_vix_chg |
| us_ixic_ret1 | Nasdaq 1 日 log 報酬 |
| us_ixic_ret5 | Nasdaq 5 日 log 報酬 |
| us_ixic_ret1_lag1 | 前兩晚的 us_ixic_ret1 |
| us_gspc_ret1 | S&P 500 1 日 log 報酬 |
| us_gspc_ret5 | S&P 500 5 日 log 報酬 |
| us_gspc_ret1_lag1 | 前兩晚的 us_gspc_ret1 |
| beta_us | 個股開盤 gap 對 us_ixic_ret1 的 120 日 rolling beta（至少 60 天，只用 D−1 以前） |
| beta_x_us | beta_us × us_ixic_ret1，也就是昨晚 Nasdaq 對今天開盤的預期影響 |
| beta_x_us_lag1 | beta_us × us_ixic_ret1_lag1 |

### 8.2 NIGHT：2 個，台指期夜盤（`v9\rolling9.py` 的 night_table）

| feature | 定義 |
|---|---|
| night_ret | ln(夜盤收盤 / 同一個合約前一個日盤的收盤)。夜盤是 D−1 15:00 到 D 05:00；合約取夜盤成交量最大的那個 |
| night_range | (夜盤最高 − 最低) / 前一個日盤的收盤 |

- 日期落在臨時休市日的夜盤，改算到下一個交易日。
- FinMind 有時夜盤比前一天的日盤先出來，這時當缺值。
- 缺值時模型等於用訓練平均，也就是當作沒有夜盤。

### 8.3 SIG：2 個，波動大小（同時是樣本權重裡的 σ）

| feature | 定義 |
|---|---|
| absr_60 | 過去 60 天 min(\|r\|, 5%) 的平均（至少 20 天） |
| absres_60 | 過去 60 天 \|e\| 的平均（至少 20 天） |

### 8.4 XR：30 個，當天 100 支裡的排名（`v5\features_extra.py`）

xr_<f> = f 在當天 100 支裡的百分位排名 − 0.5，範圍大約是 −0.5 到 0.5。f 是下列 30 個 B 組 feature：

ret1、ret5、ret20、intraday_ret、gap、range、clv、upper_shadow、lower_shadow、vol_ratio、amt_ratio、trades_ratio、foreign_net、foreign_net_5d、trust_net、trust_net_5d、dealer_self_net、dealer_hedge_net、foreign_hold_chg5、intra_bias_20、intra_bias_60、intra_bias_250、gap_bias_20、RSI、close_SMA30、BB_pctb、ATR_pct、vol_ratio_5_60、beta_x_us、y_mean5

### 8.5 NEW：5 個，v3 加的選股 feature（`v9\features_v3.py`）

| feature | 定義 | 依據 |
|---|---|---|
| gap_bias_60 | 過去 60 天 gap 的平均（至少 30 天） | 漲幅來自隔夜的股票，盤中容易跌（Lou, Polk & Skouras 2019） |
| gap_bias_250 | 過去 250 天 gap 的平均（至少 120 天） | 同上 |
| resid_vol_60 | 過去 60 天 e 的標準差（至少 30 天） | 個股波動高的股票表現偏弱（Ang et al. 2006） |
| dow_bias_e | 同一個星期幾、過去 52 次的 e 平均（至少 20 次） | 報酬的季節性（Keloharju et al. 2016） |
| hi52 | C_{D−1} / 過去 250 天的最高收盤 − 1 | 52 週高點（George & Hwang 2004） |

### 8.6 XRN：5 個

xr_gap_bias_60、xr_gap_bias_250、xr_resid_vol_60、xr_dow_bias_e、xr_hi52：NEW 的 5 個在當天 100 支裡的百分位排名 − 0.5。

### 8.7 只用在規則、不是模型輸入的欄位

| 欄位 | 定義 | 用在 |
|---|---|---|
| ex_today、ex_next1、ex_prev1、ex_prev2 | 當天、下一個交易日、前一天、前兩天是除權息日 | 6.4 除權息規則 |
| beta | 6.5 的 120 天 beta | beta 對沖 |
| pbar 歷史、夜盤 z | 前 250 天的 pbar、夜盤報酬標準化 | 6.3 的 T2（v11） |

## 9. 程式位置

| 內容 | 檔案 |
|---|---|
| 合併每天的 zip | `transformer\update_data.py` |
| B 組 67 個 feature | `transformer\comp_features.py` |
| XR、除權息旗標、殘差 label、absres_60 | `v5\features_extra.py` |
| NEW、XRN | `v9\features_v3.py` |
| NIGHT、SIG、成員定義、MLP / LightGBM 訓練、模型池、合成、除權息規則 | `v9\rolling9.py` |
| MLP、Transformer、scaler、訓練迴圈 | `transformer\tfmodel.py` |
| panel、Transformer 的訓練和超參數 | `v5\ens_lib.py` |
| 模型池的訓練點、seed 編號 | `v5\rolling.py` |
| beta 和對沖 | `v8\hedge_beta.py` |
| 每天的 v9 | `v9\train_and_predict_v9.py`（`evening.bat`、`daily.bat`） |
| T2、T3、X1、自訂參數（W0） | `v11\train_and_predict_v11.py`（`evening_v11.bat`、`daily_v11.bat`、`custom_v11.bat`） |
| 檢查 | `v5\guards.py`、`v9\check_zip_fresh.py` |
| 夜盤下載 | `v7\fetch\fetch_night_session.py` |
| 除權息預告下載 | `v5\fetch_exdiv.py` |

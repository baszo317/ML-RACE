# RACE-STD v1：組內合併實驗的驗證標準（事先寫好的規則）

寫於 2026-11-15 21:10–21:32（台北），在這個實驗的任何成績（D 或 V）算出來之前。這份檔案的 SHA-256 會在建 panel 時寫進
`results\snapshot.json`，之後 `LOCK.json`、`SEALED_V.json` 都會核對它；寫完之後不再修改（要改只能另開 v2，並註明原因）。

使用者的要求（11/15 21:06）：「現在我和我的組員各用了自己的方式建構了自己的模型，但我們現在想合併出一個 race ver，我們有不同
feature 架構，我現在希望透過挑選出最好 feature 與架構模型，所以我們希望跑一個實驗，你可以立一個新標準驗證，validation 只做一次
避免 snooping，我有將其他組員使用的東西放在 race ver」；21:12：「實驗都在我本機跑就好」。

所有計算都在使用者的電腦上跑（Claude 桌面版的 Linux VM；資料不離開這台電腦）。`run_race_std.bat` 可以在 Windows 原生 Python 重跑。

## 0. 寫這份規則時已經知道的

| 系統 | 文件 | feature | 模型 | 決策 | 文件上的成績（各自的 protocol） |
|---|---|---|---|---|---|
| Baszo v9 / v11 / W0 | `FEATURES_AND_ARCHITECTURE.md` | 111 | 選股：MLP × 4 + LightGBM × 1–2；大盤：MLP × 3 + Transformer | w_T·(pbar − 0.5) + p_e_c，除權息猜 0，beta 對沖 | 2022-01..2026-10-22：v9 14.31、v11 15.02、W0 12.22 bp/日（W0 val 11.79 / hold 10.35 / post 21.71） |
| FinPred v2.3 | `race ver\framework_features_v2.3.md` | 61（46 個每天排名） | HistGBR 回歸（超額報酬） | 100 檔中位數以上猜 1 | 2022/1–2026/5 +9.9 bp（t 8.5）；2026/6–9 +31.1；10/27–11/9 +23.7 |
| 陳聖翰 V5 | `race ver\陳聖翰的V5架構.md`、`race ver\所有特徵by陳聖翰.md` | 169（40 個自己爬） | 大盤 Ridge + 選股 LightGBM Huber | λ·m̂ + ŝ > 0，λ 0.5 | CV 2023 / 2024 / 2025 / 2026H1：AR 16.1 bp，ex0 11.96 |
| NTU M2 | `race ver\NTU_M2_Final_System`（完整程式） | 39 | LightGBM 回歸（超額報酬） | Top25 | 11 個半年 2022H1–2026/09：9.32 bp；凍結 9/30 模型 10/1–11/13：28.84 |

- 四套都已經用 2022–2026 的資料挑過 feature 或模型（Baszo：val 2022–24 挑、hold / post 反覆看；FinPred：2022/1–2026/5 與 6–9 月；
  陳聖翰：2023–2026H1；M2：2022H1–2026/09 與 10–11 月）。**歷史上沒有一段資料對每個人都是全新的。** 這份標準能保證的是：
  這次合併實驗自己的選擇只碰 V 一次。各自先前的挑選造成的偏高無法消除，只能在結果旁邊註明。
- 陳聖翰的消融裡最重要的 4 組有 3 組是自己爬的資料（K_pre_stk、K_pre、T1_chip）；這次只有文件、沒有這些資料，所以 C 組只重建
  「競賽資料 + 夜盤 + 日曆」算得出來的 129 個，會低估他的 feature。FinPred、陳聖翰只有文件沒有程式：P、C 兩組是照文件重寫的
  （定義在附錄 A、B），不保證和原作者的程式逐位相同。U 用使用者自己的程式（`v9\rolling9.build_core`），M 直接呼叫 M2 的程式。
- 這次任何候選（包括四個 native）在這個標準下的 D、V 成績、IC、相關，寫這份規則之前都沒有算過。

## 1. 標準：所有候選都一樣

### 1.1 資料快照

- `D:\ML RACE\data`（競賽資料，Top100 共 100 支、global、info.json、`ref\` 的交易日曆和除權息）＋ `v7\data\tx_daily.csv`（夜盤）。
  有 label 的最後一天 2026-11-09。每個輸入檔的 SHA-256 寫進 `results\snapshot.json`；這個實驗不換快照（之後每天的資料更新不影響它）。
- 訓練用 100 支；計分用 Top50@Y。r = adj_close / adj_open − 1；e = r − 當天 100 支 r 的平均。
- 課程 zip 的 −1 佔位 row 一律當缺值。

### 1.2 資訊時點

- 第 t 天的 feature 只能用 t 日 08:50 以前公布的資料：個股資料到 t−1；^TWII、^KS11、TX_OI 到 t−1；美股 row t（前一晚收盤）可以用；
  夜盤（t−1 15:00 到 t 05:00）可以用；事先公布的交易日曆、除權息預告可以用。各組照原作者文件的時點（P、C 的美股比這個規則更保守，
  只用 row t−1，照原文件不改）。
- 檢查（`build_panel.py`，結果寫 `results\leak_check.txt`，在 Stage 1 之前）：
  (a) 截斷測試：隨機抽 9 天 t（D、V 各段都有），把資料截在 t、t 當天的價量換成缺值，重算 U、P、C、M，第 t 天每個 feature 必須相同
      （容許 1e−9）；不同就是 bug，修好才進 Stage 1。
  (b) 每個 feature 和當天 r 的 rank IC（**只用 D 的日子**）：|IC| > 0.30 從所有組剔除（陳聖翰的規則），> 0.08 列出。

### 1.3 訓練

- walk-forward、expanding window，和 v9–v16 相同的 6 個年度 fold：2022、2023、2024、2025、2026-01..05、2026-06..11-09。
  fold 從 s 開始：訓練用 2020-03-02 ≤ date < s、有 label 的 row；fold 內每一天都用同一組模型預測。
- early stopping 和任何調整只能用訓練窗裡的資料（各 learner 的做法在 2.2）。
- **D（開發）= 前 3 個 fold，2022-01-03 到 2024-12-31。V（驗證，只用一次）= 後 3 個 fold，2025-01-02 到 2026-11-09。**
  Stage 1 只訓練、只評分 D；程式在沒有 LOCK 的時候拒絕訓練 V 的 fold。

### 1.4 決策和分數

- 標準決策 SD：Top50 裡依分數由大到小，前 25 支猜 1（同分依股票代號），然後除權息當天、前一天、後一天、後兩天的股票改成 0
  （= v9 / W0 的除權息規則）。理由：Top25 讓 Σ(2p − 1) = 0，AR 只剩選股（命題 1），不同 learner 的分數尺度也不影響；
  除權息規則對每個候選相同，v2 已驗證（val +0.32，t 2.9）。
- 分數：AR_t = (1/n) Σ_{i ∈ Top50} (2p_i − 1)·r_i，n = 當天有 label 的 Top50 支數，單位 bp/日。
- 主要指標：期間平均 AR。另外報告：Newey–West t（5 lags，= v15lib.nw_t）、每日標準差、正的天數比例、Top50 的 Spearman rank IC
  （每天算再平均）、逐 fold、最差的 19 天平均（11/30–12/24 是 19 個交易日）。
- 成對比較一律用每天的差（同一天、同一批股票），t 用 Newey–West（5 lags）。

## 2. 候選

### 2.1 Feature 組

| 代號 | 來源 | 數量 | 做法 |
|---|---|---|---|
| U | Baszo：v9 / v11 全部成員 feature 的聯集（B9 64、Nasdaq 3、NIGHT 2、SIG 2、XR 30、NEW 5、XRN 5） | 111 | `v9\rolling9.build_core`（使用者的程式，原封不動） |
| P | FinPred v2.3 | 61 | 照文件重寫（附錄 A）；46 個 `z_` 每天在 100 支裡排名成 −0.5..0.5 |
| C | 陳聖翰 V5 能重建的部分：A–J 121、M_cal 5、夜盤 3 | 129 | 照文件重寫（附錄 B）；缺 K_pre 另外 10 個、K_pre_stk 7、T1_chip 17、L_macro 6 |
| M | NTU M2 | 39 | 直接呼叫 M2 的 `create_features`、`enrich`；市場 feature（breadth 等）用 Top50 算，和 M2 相同 |
| ALL | U ∪ P ∪ C ∪ M | 340 | 名稱加前綴 `U__`、`P__`、`C__`、`M__`，不去重 |

### 2.2 Learner（選股觀點；每個都照原作者文件的設定）

| 代號 | 來源 | target、權重 | 模型與參數 | early stopping | seed |
|---|---|---|---|---|---|
| L_U | Baszo lgb_res | y_resid = 1{e > 0}；權重 min(\|e\|, 5%)，平均化成 1 | LightGBM binary：lr 0.02、15 葉、min_data_in_leaf 500、feature / bagging fraction 0.8（每輪）、λ_l2 1 | 訓練窗最後 120 天當驗證（加權 log-loss），最多 1000 輪、100 輪沒進步停；整個訓練窗重訓到最佳輪數 | 1、2、3 |
| L_N | Baszo mlp_res | 同上；權重 min(\|e\|, 5%) / absres_60^0.5（缺值補訓練窗中位數） | MLP n → 64 → 32 → 1、GELU、dropout 0.2、AdamW lr 1e−3、wd 1e−2、batch 1024、gradient clipping 1.0；1% / 99% 截尾、標準化、缺值 0 | 最後 120 天、最多 10 epoch、patience 2；整個訓練窗重訓到最佳 epoch 數 | 1、2、3 |
| L_P | FinPred | clip(e, ±5%) | sklearn HistGradientBoostingRegressor（平方誤差）：800 輪、lr 0.0075、max_leaf_nodes 7、min_samples_leaf 2000、L2 5 | 無 | 0（沒有抽樣，結果確定） |
| L_C | 陳聖翰 | z = clip(e / σ20, ±5)，σ20 = 前 20 天盤中報酬的標準差；預測 × σ20 | LightGBM huber：lr 0.03、15 葉、max_depth 5、min_child_samples 300、L1 1、L2 10 | 訓練窗最後 15% 的日期當驗證，最多 2000 輪、100 輪；整個訓練窗重訓到最佳輪數 | 1 個（沒有抽樣，結果確定） |
| L_M | NTU M2 | 100 × clip(e, ±8%) | 中位數補值 + LightGBM regression：260 棵、lr 0.025、9 葉、min_child_samples 250、λ 60、colsample 0.85、deterministic | 無（固定 260 棵） | 2026、2027、2028 |

- MLP 用 numpy 寫（VM 沒有 torch）：架構、初始化（同 torch Linear 預設）、AdamW、clipping 和 torch 版相同，數值不會逐位相同。
- 有抽樣的 learner 都用 3 個 seed 平均（原作者 M2 是 1 個、Baszo 是 2 × 5 個訓練點）：標準統一，避免比到的是 seed 數。
- 和原作者的差別（為了同一個標準）：訓練起點統一 2020-03-02、訓練股票統一 100 支（M2 原本 50 支，mlp_res_B9 原本加擴充股票池）、
  e 統一用 100 支平均（M2 原本 50 支）、年度 refit（原作者每天或每半年）。

### 2.3 Stage 1 的候選（只在 D）

1. Grid：5 個 feature 組 × 5 個 learner = 25 個（P×L_P、C×L_C、M×L_M 就是 P、C、M 的 native）。
2. U-native = W0 / v11 選股觀點照原本的成員結構：(mlp_res_B9、mlp_res_xr9、mlp_res_BN9、mlp_res_xrN9 四個 L_N 的平均 +
   lgb_res（B 67）、lgb_res_xrN（106）兩個 L_U 的平均) / 2，機率直接平均（= v9 / v11 的 p_e）。
3. 合併：ENS4 = 四個 native 的當天 z 平均（等權）；ALL5 = ALL 上 5 個 learner 的當天 z 平均。
4. 消融：L* = grid 25 個裡 D 平均 AR 最高的那個的 learner；用 L* 跑 ALL−U、ALL−P、ALL−C、ALL−M。
- 當天 z：在當天 100 支裡標準化、缺值 0、截在 ±3（= v14lib.zday）。
- 合計 32 個候選。D 的結果全部可以看（這段本來就被大家用過）；D 上只做 2.4 的機械規則，不做其他挑選。

### 2.4 進 V 的候選（finalist，機械規則）

- 32 個候選依 D 的平均 AR（SD 決策）排序，前 3 名：F1、F2、F3（同分取 unit 少的）。
- ENS_TOP3 = F1、F2、F3 的當天 z 平均。
- 加上 4 個 native（已在前 3 名的不重複）。最多 8 個。
- `freeze.py` 把名單、每個候選的完整設定、程式、資料、這份 PREREG 的 SHA-256、D 的成績寫進 `results\LOCK.json`。
  有 LOCK 以後不能加、換候選或改程式；V 只算 LOCK 裡的候選。

## 3. V：只做一次

- `stage2_validate.py`：讀 LOCK，SHA-256 不符就停；訓練 V 的 3 個 fold、一次算完所有 finalist，寫 `results\SEALED_V.json` 和報告；
  已經有 SEALED_V.json 就拒絕再跑。`--audit` 只重算、比對是否相同，不產生新的判定。
- 若封存之後才發現 bug：照實寫在報告裡；修正後的數字標成「V 第二次，只報告」，第一次的判定不改，新的判定只能用之後的新資料（live）。

### 3.1 主要判定：race 版的選股觀點和 feature

1. 冠軍 w = V 平均 AR 最高的 finalist。
2. 1-SE 規則（Breiman et al. 1984；Hastie, Tibshirani & Friedman 2009 §7.10）：和 w 的差 ≤ 1 個標準誤（每天成對差的 Newey–West 標準誤）
   的 finalist 裡，選最簡單的：先比 unit 數（一個 feature 組 × 一個 learner 算 1 個 unit），再比 feature 數。選到的就是 race 版。
   理由：K 個裡取最大值會偏高（命題 3），比較複雜的合併在 v10–v16 幾乎都沒有通過；每天 09:00 前要跑完，unit 越多越容易出事。

### 3.2 第二個判定：要不要加大盤觀點（陳聖翰的架構）

- 對 3.1 選出的 S 比較 SD 和 SD_MV：
  - m̂_t = Ridge 大盤模型（陳聖翰）：C 組裡 45 個日層級 feature（G 28、H 的市場 4、I 5、M_cal 5、夜盤 3），中位數補值、標準化，
    RidgeCV α ∈ 10^0..10^5（16 個 log 格點），target = 當天 Top50 平均 r，一天一 row，同樣的 fold。
  - ŝ_i = IC_D × σ_e,t × z_i（Grinold–Kahn 的 α = IC·σ·z，命題 4）：IC_D = S 在 D 的平均 rank IC；σ_e,t = 前 60 天 e 的橫斷面標準差平均；
    z_i = S 的當天 z。
  - SD_MV：Top50 裡 m̂_t + ŝ_i > 0 猜 1，再套除權息規則（λ = 1：兩者都是期望報酬，相加就是 E[r_i | F] 的估計；v14 定理 1）。
- **規則：SD_MV − SD 在 D > 0 而且在 V > 0 → race 版加大盤觀點；否則不加。**（= v11 以來「val、hold 都 > 0」的規則）
- 只報告：λ = 0.5、Top50 猜 1 的比例、AR 拆成 timing 和選股。11/23 依排行榜決定風險時可以再調，但預設以這條規則為準。

### 3.3 只報告（不改判定）

- MCS 90%（Hansen, Lunde & Nason 2011；T_max、stationary bootstrap 平均 block 10 天、5000 次）。
- 冠軍對每個 native 的成對差、Newey–West t、SPA p 值（Hansen 2005，K = finalist 數）。
- 逐 fold、2026-10-01..11-09、19 天平均的分布（block bootstrap）、勝過每個 native 的機率。
- 收縮後的預期值：V 平均往 finalist 的總平均收縮（James–Stein，命題 3 的修正）。

## 4. 命題（證明附直觀和詳細兩種）

**命題 1（Top25 不含大盤）。** s_i = 2p_i − 1 ∈ {±1}。Top50 有 50 支、正好 25 支猜 1 時 Σ s_i = 0，所以
AR = (1/50) Σ s_i r_i = (1/50) Σ s_i (r_i − r̄)，和當天平均 r̄ 無關。除權息規則把 k 支從 1 改成 0 時 Σ s_i = −2k，
AR = (1/50) Σ s_i (r_i − r̄) − (2k/50)·r̄，大盤曝險只有 −2k/50。

- 直觀：一半看多、一半看空，大盤一起漲跌的部分互相抵消，剩下的只有「挑對誰比較強」。
- 證明：(1/50) Σ s_i r_i = (1/50) Σ s_i (r_i − r̄) + r̄·(1/50) Σ s_i；第二項在 Σ s_i = 0 時為 0，在 Σ s_i = −2k 時為 −(2k/50) r̄。∎

**命題 2（兩段式讓 V 無偏）。** 設 finalist 集合 F 只由 D 的資料決定，且 V 每天的分數和 D 的雜訊獨立。則對任何固定的 k，
E[V 平均_k | k ∈ F] = μ_k（候選 k 在 V 的真實平均）。

- 直觀：選人時沒看過 V 的考卷，V 的分數就不會因為「被選中」而偏高；偏高只會來自在 V 上再挑一次（命題 3）。
- 證明：令 A = {k ∈ F}，A 是 D 資料的函數。V 平均_k = μ_k + ε_k，ε_k 是 V 的雜訊，E[ε_k] = 0，且 ε_k 和 D 資料獨立，所以 ε_k 和 A 獨立。
  E[V 平均_k | A] = μ_k + E[ε_k | A] = μ_k + E[ε_k] = μ_k。∎
  （條件「V 和 D 的雜訊獨立」：V 的模型會用 D 的 label 訓練，但那是模型的輸入，不是 V 天的報酬；每日報酬的自相關很小，D 和 V 的交界只有一天。）

**命題 3（在 V 上取最大值的偏高）。** K 個 finalist 的估計誤差 ε_k ~ N(0, σ_k²)、σ_k ≤ σ（可以相關）。冠軍 w = argmax V 平均，則
E[V 平均_w − μ_w] ≤ σ·√(2 ln K)。K = 8 時是 2.04σ。

- 直觀：8 個一樣好的候選各自丟一次骰子，最大的那個平均會多出約 2 個標準差；所以冠軍的 V 分數要打折，差距在 1–2 個標準誤以內
  的冠軍不可靠（3.1 的 1-SE 規則就是這個用途）。
- 證明：V 平均_w − μ_w = ε_w ≤ max_k ε_k。對任何 t > 0，由 Jensen：exp(t·E[max ε_k]) ≤ E[exp(t·max ε_k)] ≤ Σ_k E[exp(t ε_k)]
  = Σ_k exp(t²σ_k²/2) ≤ K·exp(t²σ²/2)。取對數：E[max ε_k] ≤ (ln K)/t + tσ²/2；取 t = √(2 ln K)/σ 得 E[max ε_k] ≤ σ√(2 ln K)。∎

**命題 4（Grinold–Kahn：α = IC·σ·z）。** 某天的 (e_i, x_i) 在股票之間是二元常態，corr = ρ，z_i 是 x 的標準化。則
E[e_i | z_i] = ρ·σ_e·z_i。

- 直觀：分數高一個標準差，超額報酬平均高 ρ 個標準差；IC 越低，分數能代表的報酬越少。這樣換算後，選股分數和大盤預測 m̂（Ridge，
  單位是報酬）才能相加。
- 證明：二元常態的條件期望是線性的：E[e | z] = E[e] + Cov(e, z)/Var(z)·(z − E[z])。e 是超額報酬，E[e] = 0；z 標準化，Var(z) = 1、
  E[z] = 0；Cov(e, z) = ρ·σ_e·1。所以 E[e | z] = ρ σ_e z。∎（rank IC 和 Pearson 相關在常態下差不到 1%：ρ_s = (6/π) arcsin(ρ/2)。）

## 5. 計算量與執行

- 在 VM（2 個 worker、每個單執行緒）：Stage 1 約 255 個 seed-unit，預估 20–40 分鐘；Stage 2 視 finalist 約 10–25 分鐘。
  VM 每次指令最多 3 分鐘，所以 `train.py` 分段跑（`--budget` 秒），做完的 unit 存在 `cache\units\`，中斷後接著跑。
- 順序：`build_panel.py` → `train.py --stage 1` → `stage1_eval.py`（決定 L*、寫消融計畫）→ `train.py --stage 1b` →
  `stage1_eval.py --final` → `freeze.py` → `train.py --stage 2` → `stage2_validate.py`（只一次）。

## 附錄 A：P 組（FinPred v2.3）重寫的定義

記號：O、H、L、C 是還原價；V 成交股數、AMT 成交金額；r = C/O − 1；e = r − 當天 100 支 r 平均；gap = O/C₋₁ − 1；cc = C/C₋₁ − 1。
個股 feature 都在 t−1 收盤算好、往後移一天（第 t 天只用到 t−1）。`z_` = 當天 100 支裡的百分位排名 − 0.5（缺值保持缺值）。

| feature | 定義 |
|---|---|
| z_ew120_se、z_ew500_se | r 的指數加權平均的標準誤，半衰期 120 / 500 天：se = √(v / n_eff)，v = 加權變異數，n_eff = (Σw)² / Σw²；至少 60 天 |
| z_res_bias_all、z_res_bias_t | e 從頭到 t−1 的平均（至少 60 天）；平均 ÷ 標準誤 |
| z_res_ew500、z_res_ew500_shrunk | e 的指數加權平均（半衰期 500）；經驗貝氏收縮：m̄ + τ²/(τ² + se²)·(m − m̄)，τ² = max(橫斷面變異 − 平均 se², 0) |
| z_stock_dow_res_ew | 同一個星期幾、之前的 e，指數加權（半衰期 100 次） |
| z_res_intra_mean20 / 60、z_up_rate20 / 60、z_gap_mean20 / 60、z_intra_std20 / 60 | e 的平均、1{r > 0} 的平均、gap 的平均、r 的標準差 |
| z_mom5 / 20 / 60 / 250、z_hi_dist250 | ln(C / C₋ₖ)；ln(C / 250 天最高收盤) |
| z_macd_l1、z_sma_dev_l1、z_bb_pos_l1、z_bb_width_l1、z_atr_l1 | MACD/C；C/SMA − 1；(C − 中軌)/(上軌 − 下軌)；(上軌 − 下軌)/中軌；ATR/C |
| z_ushadow_l1、z_amount_z20 | (H − max(O, C))/C；ln AMT − 20 天平均 ln AMT |
| z_vol20 / 60 | cc 的標準差 |
| z_beta_intra、z_beta_gap_on_intra | r 對 100 支平均 r 的 250 天 beta（至少 120）；同一天 r 對 gap 的 250 天迴歸係數 |
| z_beta_gap_us、raw_beta_gap_us、z_exp_gap | gap_d 對 global row d 的 S&P 500 報酬的 250 天 beta（d ≤ t−1）；原始值；beta × row t−1 的 S&P 500 報酬 |
| z_foreign_flow_l1 / 20、z_trust_flow_l1 / 5 / 20、z_dealer_flow_l1 / 5 / 20、z_hedge_flow_l1 / 20 | 買賣超股數 / 成交股數，及其 5、20 天平均 |
| z_fhold_l1、z_fhold_chg1 / 5 | 外資持股比率（缺值沿用前一天）；1、5 天變化 |
| mkt_intra_l1、mkt_gap_l1、mkt_ret_l1、mkt_ret5、mkt_ret20 | t−1 的 100 支平均 r、gap、cc；cc 的 100 支平均加總 5 / 20 天 |
| spx_ret、ndx_ret、vix、vix_chg、vix_z20 | global row t−1（缺值沿用前一 row）：S&P 500、Nasdaq 漲跌幅、VIX、VIX 漲跌幅、(VIX − 20 row 平均)/20 row 標準差 |
| dow、post_holiday、tom_first2、settle_day | 星期幾；和前一交易日之間有平日休市；月初前兩個交易日；當月第三個星期三（休市就順延到下一個交易日） |

## 附錄 B：C 組（陳聖翰 V5）重寫的定義

文件只有名稱，下面的公式是照名稱和 Qlib 的慣例補的（kmid 等 = Qlib Alpha158 的 KBAR）。除了 I、M_cal、夜盤（第 t 天開盤前就知道），
全部用 t−1 收盤以前的值（= 他的「先算好再整體往後移一天」）。G 也照他的規則用 global row t−1。

| 組 | feature |
|---|---|
| A 6 | kmid = (C − O)/O、klen = (H − L)/O、kup = (H − max(O, C))/O、klow = (min(O, C) − L)/O、ksft = (2C − H − L)/O、kmid2 = (C − O)/(H − L) |
| B 13 | ret1 = cc；roc_k = C/C₋ₖ − 1；ma_bias_k = C/MA_k − 1；rsv_k = (C − k 天最低 L)/(k 天最高 H − 最低 L)；k = 5、10、20、60 |
| C 9 | std_5 / 20 / 60（cc）、range_5 / 20 / 60（(H − L)/C 的平均）、std_ratio_5_20、atr_rel = ATR/C、bb_width = (上軌 − 下軌)/中軌 |
| D 7 | vchg_1 = ln(V / V₋₁)、vrel_5 / 20 / 60 = V / k 天平均 V、amt_rel20、size_rel20 = (AMT/成交筆數) / 20 天平均、corr_rv_20 = corr(cc, ln V) |
| E 19 | f_net、trust_net、dself_net、dhedge_net = 買賣超 / V；_5、_20 = k 天買賣超合計 / k 天 V 合計；inst、inst_5 = 四者合計；f_streak；fhold；fhold_d1 / 5 / 20 |
| F 5 | rsi、rsi_chg5、macd_rel = MACD/C、sma_bias = C/SMA − 1、bb_pctb = (C − 下軌)/(上軌 − 下軌) |
| G 28 | ixic、gspc、twii、ks11 各 ret1、ret5、ret20、ma20_bias、std20；vix、vix_chg1、vix_rel20 = VIX / 20 row 平均 − 1；txoi_chg1、txoi_chg5（萬口）、txoi_z20；us_minus_tw_ret1 = gspc_ret1 − twii_ret1；staleness_days = t 和 global row 日期相差的天數 |
| H 19 | rank_（B 的 13 個在 100 支裡的百分位 − 0.5）；mkt_kmid、mkt_ret1、mkt_up_ratio、mkt_f_net（Top50 平均）；kmid_vs_mkt、ret1_vs_mkt |
| I 5 | dow、month、dom、gap_days（和前一交易日相差的日曆天數）、month_last_bd |
| J 10 | intra_mean_5 / 20 / 60、up_rate_20、intra_std_20、intra_sharpe_20 = 平均 / 標準差、gap（t−1 的 gap）、gap_mean_5 / 20、gap_intra_corr_60 |
| M_cal 5 | is_settle_day、bdays_to_settle（到下一個結算日的交易日數）、is_month_last_bd、is_month_first_bd、is_quarter_last_bd |
| 夜盤 3 | tx_night_ret、tx_night_absret、tx_night_range（= U 的 night_ret、\|night_ret\|、night_range） |

## 附錄 C：M 組（NTU M2）

`race ver\NTU_M2_Final_System\NTU_M2_Final_System\src\features\base_features.create_features` + `market.enrich`，取 `builder.KEEP`
的 39 個。輸入是競賽資料的同一批欄位（M2 原本就讀教師 zip，欄名相同）。100 支一起算個股 feature；breadth_p1、market_avg_ret_p1、
market_dispersion_p1 用 Top50 單獨算一次再對回每一 row，stock_minus_market_p1 重算，然後才 `enrich`（和 M2 只放 50 支時相同）。

## 6. 修正 1（使用者 21:39 的要求；21:45–22:00 寫，任何候選的成績都還沒算）

使用者 21:39：「陳聖翰最重要的 feature 我們這裡自己爬」。所以：

- C 組改成陳聖翰 V5 的全部 169 個：原本的 129 個 ＋ 自己爬的 40 個（K_pre 另外 10、K_pre_stk 7、T1_chip 17、L_macro 6），
  定義在附錄 D。ALL 變成 380 個（U 111、P 61、C 169、M 39）。
- 3.2 的 Ridge 大盤模型改用他的 61 個日層級 feature：G 28、H 的市場 4、I 5、K_pre 13、L_macro 6、M_cal 5。
- 資料：VM 沒有對外網路，所以 `research\fetch_ext.py` 在使用者的 Windows 上跑（雙擊 `run_fetch.bat`），存進 `race_std\ext\`；
  來源 Yahoo Finance（ADR、ETF、台幣抓不到時用 stooq）和 FinMind。`ext\` 每個檔的 SHA-256 也寫進 snapshot。
- 某個來源抓不到：那幾個 feature 整欄當缺值（樹模型自己處理、MLP 補 0），在報告列出，其他規則不變。
- 時點：美股、ADR、ETF、匯率、美債、美元指數用「美國日期 < t」的最後一個交易日（台北 t 日 05:00 前收盤）；日經、KOSPI 的開盤
  跳空用 t 當天的開盤（台北 08:00 開盤，09:00 前拿得到；live 要在 08:01 以後抓）；融資融券、借券、當沖在 t−1 收盤後公布，用到 t−1。
  截斷測試（1.2 a）也包含這 40 個。
- 修正之前已經算過、但不是任何候選成績的東西：21:40 的 same-day IC 檢查（當時 340 個 feature、只用 D 的日子，最大 |IC| 0.103，
  沒有 > 0.30）。重建 panel 後重算。
- 其他規則（fold、D / V、決策、候選、finalist 規則、V 只用一次、1-SE、大盤觀點的規則）都不變。

## 附錄 D：自己爬的 40 個（陳聖翰 V5 的 K_pre、K_pre_stk、T1_chip、L_macro）

美股類的 d = 美國日期 < t 的最後一個交易日；ret_1 = close_d / close_{d−1} − 1、ret_5 = close_d / close_{d−5} − 1。
ADR 比例：TSM 5、UMC 5、ASX 2、CHT 10。溢價 prem = ADR close_d × (USD/TWD)_d / (比例 × 台股 t−1 的未還原收盤) − 1。

| 組 | feature | 定義 |
|---|---|---|
| K_pre 另外 10（日層級） | n225_gap、ks11_gap、asia_gap_mean | 日經、KOSPI 在 t 的開盤 / 前一個交易日收盤 − 1；兩者平均（缺一個就用另一個） |
| | sox_ret_1、sox_ret_5、sox_rel_ixic_1 | 費半 ret_1、ret_5；sox_ret_1 − Nasdaq ret_1 |
| | tsm_ret_1、adr_prem、adr_prem_chg_1、adr_prem_z20 | TSM 的 ret_1；2330 的溢價、和前一天的差、用前 20 天標準化 |
| K_pre_stk 7（個股） | adr_ret_1、adr_prem、adr_prem_chg_1、adr_prem_z20 | 有 ADR 的 4 支（2330、2303、3711、2412），其他缺值 |
| | sector_ret_1、sector_ret_5、sector_rel_1 | 對應美股產業 ETF 的 ret_1、ret_5；ret_1 − SPY 的 ret_1（對應表在下面） |
| T1_chip 17（個股，用到 t−1） | margin_chg_vol、margin_util、margin_chg5 | 融資餘額變化 / 成交股數；融資餘額 / 融資限額；5 天變化 / 5 天成交股數 |
| | short_chg_vol、short_margin_ratio、short_chg5 | 融券餘額變化 / 成交股數；券資比；5 天變化 / 5 天成交股數 |
| | offset_vol | 資券互抵 / 成交股數 |
| | sbl_sell_vol、sbl_bal_v20、sbl_chg_vol、sbl_chg5_v20 | 借券賣出 / 成交股數；借券餘額 / 20 天平均成交股數；借券餘額變化 / 成交股數；5 天變化 / 20 天平均成交股數 |
| | daytrade_ratio、daytrade_ratio_rel20、daytrade_imb | 當沖成交股數 / 成交股數；再除以前 20 天平均；(當沖買進金額 − 賣出金額) / 兩者和 |
| | rank_margin_chg_vol、rank_sbl_sell_vol、rank_daytrade_ratio | 上面三個在 100 支裡的百分位 − 0.5 |
| L_macro 6（日層級） | twd_ret_1、twd_ret_5、tnx_chg_1、tnx_chg_5、dxy_ret_1、dxy_ret_5 | USD/TWD 的 ret；美債 10 年殖利率（^TNX）的差；美元指數的 ret |

- 產業 ETF 對應（FinMind 產業類別）：半導體 → SOXX；電腦及週邊、電子零組件、光電、其他電子、電子通路、資訊服務、通信網路 → XLK
  （但 2412、3045、4904 → XLC）；金融保險 → XLF；水泥、塑膠、鋼鐵、化學、玻璃陶瓷、造紙 → XLB；航運 → IYT；食品 → XLP；
  汽車、貿易百貨、紡織纖維、橡膠、居家生活、運動休閒、觀光 → XLY；電機機械、電器電纜、建材營造 → XLI；油電燃氣 → XLE；
  生技醫療 → XLV；其他或查不到 → SPY。
- 單位（張或股）下載後用資料本身核對（例如融券餘額和信用額度表的融券餘額應差 1000 倍），不一致就修正並記錄在報告裡。

# race_agg：用課程 topic 6、W12 的 aggregation 方法檢驗 race 版（事先寫好的規則）

寫於 2026-11-18 16:40–17:00（台北），在這個實驗的任何成績（D 或 V）算出來之前。寫完不再修改；要改只能在第 8 節加「修正」並註明時間和原因，
而且必須在受影響的候選算出任何 D 成績之前。第一部分（`..\race_std\research`）、第二部分（`..\race_std\part2`）、`race_live` 的檔案都不改，
這裡只讀。

## 0. 使用者的要求與範圍

- 使用者（11/18 16:30）：「可以以這張圖所有的東西來檢驗我比賽 ver 能否更好嗎」。圖是課程 topic 6 與 W12 的影片清單：
  - Blending and Bagging：Motivation of Aggregation、Uniform Blending、Linear and Any Blending、Bagging (Bootstrap Aggregation)
  - Adaptive Boosting：Motivation of Boosting、Diversity by Re-weighting、Adaptive Boosting Algorithm、Adaptive Boosting in Action
  - Decision Tree：Decision Tree Hypothesis、Decision Tree Algorithm、Decision Tree Heuristics in CART、Decision Tree in Action
  - Random Forest：Random Forest Algorithm、Out-of-bag Estimate、Feature Selection、Random Forest in Action
  - Gradient Boosted Decision Tree：Adaptive Boosted Decision Tree、Optimization of AdaBoost、Gradient Boosting、Summary of Aggregation
  - 延伸閱讀：Chen et al.（KDD Cup linear ensemble）、Breiman（Bagging predictors）、Freund & Schapire（boosting）、Loh 與 Breiman et al.（CART）、
    Breiman（Random forests）、Friedman（Greedy Function Approximation）
- 「比賽 ver」= race 版（`race_live`）：陳聖翰 V5 的 169 個 C 組 feature ＋ LU（Baszo lgb_res 的 LightGBM）＋ 每天重訓、最近 3 天的模型平均
  ＋ Top50 前 25 支猜 1 ＋ 除權息規則。
- 判定基準 **SD**：第一部分的 C.LU（年度 fold、seed 1–3 平均）＋ Top25 ＋ 除權息規則，D 15.10 bp/日。和第二部分、race_news 相同：
  所有候選都在同樣的年度 fold 上訓練，比「換掉或加上某個 aggregation 方法」前後的每日成對差。每天重訓的好處（第二部分 F5）假設可以和這裡的
  差相加；這個假設不在這裡檢驗。

## 1. 寫這份規則時已經知道的

- C.LU：D 15.10（2022 17.02、2023 10.69、2024 17.50），Top50 日 IC 0.1207。同樣 C 組 feature 的其他 learner（第一部分）：C.LN 14.62、
  C.LM 13.95、C.LC 13.16、C.LP 12.62。5 個 learner 在 ALL feature 上的 z 平均 ALL5 14.63（比 ALL.LN 15.32 低）；四個 native 平均 ENS4 11.49；
  ENS_TOP3 V +16.14（D 沒算，因為它是用 D 排名挑的）。
- 下面的 blending 成員是依「同一組 C feature 上的全部 learner」和「Baszo 自己的 MLP + LightGBM 結構」這兩個結構事先決定的，不是依 D 排名挑的；
  但各成員單獨的 D 分數我已經知道（上一點）。
- v15（U 組、v11 的選股觀點）：RF 葉子 250–4000 時 IC 和 lgb_res 相同（0.1058 對 0.1050，排序相關 0.90），sklearn 預設（葉子 1）只有 0.0698；
  RF 當 v11 額外成員 val（= 這裡的 D 年份）+0.19、hold −0.65、post +2.67。
- reg_svm：LightGBM 的 regularization 再加強（7 葉、min_data_in_leaf 2000、lambda_l2 10）沒有更好。
- race_live：每天重訓、3 個模型平均（WF3）比只用最新 1 個（WF1）高 +0.50（3 年同號）。第二部分：每月重訓比年度 +1.57。
- 以下任何候選在 C 組上的成績、成員之間的相關、AdaBoost / RF / CART 在這份資料上的任何數字、permutation importance，寫這份規則之前都沒有算過。

## 2. 共同設定

- 資料：`race_std\cache\panel.pkl`（第一部分凍結的 panel，166,346 row）、`features.json` 的 C 組 169 個 feature、`info.pkl`（Top50 與除權息旗標）。
- fold：D 的 3 個年度 fold（2022、2023、2024），訓練窗 2020-03-02 到 fold 開始的前一天，只用有 label 的 row（和第一部分 `train.py` 相同）。
- label 與權重（和 LU 相同）：e = resid_r（盤中報酬 − 當天 100 支平均），y = 1{e > 0}，ỹ = 2y − 1，w = min(|e|, 5%)，在訓練窗除以平均。
- 「最後 120 天」= 訓練窗最後 120 個交易日（= `learners._split_last_days`）；fit 部分 = 訓練窗扣掉最後 120 天。所有 early stopping 和參數選擇
  只用這兩部分，選完再用整個訓練窗重訓（和 LU 相同）。
- within-day z：`rslib.zday`（當天有值的股票標準化，截在 ±3，缺值 0）。zavg(a, b, …) = 各自 z 之後平均。
- 決策與分數：`rslib.decide_sd`（Top50 依分數取前 25 支猜 1，同分依股票代號；除權息當天、前一天、後一天、後兩天改 0），AR 單位 bp/日。
- 新訓練的單位存在 `race_agg\cache\units_agg\<候選>\<fold>_seed<s>.npz`（格式同第一部分）。LightGBM 4.6.0（deterministic、force_col_wise，
  和第一部分相同）；sklearn 依執行環境（版本記在結果檔）。

## 3. 候選（16 個判定候選 ＋ 1 個只報告的診斷）

### A. Blending（先有 g_t 再合成）

| 代號 | 投影片 | 做法 |
|---|---|---|
| A1 | Uniform Blending | zavg(C.LU, C.LN)：Baszo 自己的「LightGBM ＋ MLP 各半」結構放在 C 組上 |
| A2 | Uniform Blending | zavg(C.LU, C.LN, C.LP, C.LC, C.LM)：同一組 C feature 上的全部 5 個 learner |
| A3 | Linear Blending | 兩層學習：每個 fold，5 個成員先用 fit 部分重訓（g⁻，各自原本的程序和 seed），預測最後 120 天；在最後 120 天上用 5 個成員的 within-day z 做 w 加權 logistic regression（有截距、α_k ≥ 0、不加 regularization，L-BFGS-B）；測試年的分數 = Σ α_k z_k（z_k 來自用整個訓練窗訓練的 g_k，也就是第一部分存好的單位）。α 全為 0 時改用等權 |
| A4 | Any Blending（stacking） | 同 A3 的資料，但第二層是小的 LightGBM binary（w 加權）：輸入 = 5 個 z ＋ 當天 absres_60 的百分位排名 ＋ C__vix；num_leaves 4、max_depth 2、min_data_in_leaf 1000、learning_rate 0.05、固定 150 輪、lambda_l2 10、不抽樣、seed 1；分數 = 預測機率 |

### B. Bagging（uniform、邊學邊合成）

| 代號 | 投影片 | 做法 |
|---|---|---|
| B1 | Uniform aggregation of randomized g | LU seed 1–10 平均（seed 1–3 是存好的 C.LU，另訓練 seed 4–10），只增加 LightGBM 內部抽樣的平均次數 |
| B2 | Bagging (Bootstrap Aggregation) | 10 個 bag。bag b：rng = default_rng(10000 × fold 年 + b)，從訓練窗的交易日抽同樣天數、可重複（整天的 100 支一起抽，因為同一天的股票相關）；early stopping 用這個 bootstrap 樣本裡 fit 部分的日子訓練、原本的最後 120 天驗證；再用整個 bootstrap 樣本重訓到最佳輪數。LU 參數、seed = b。分數 = 10 個 bag 的機率平均 |

### C. AdaBoost（non-uniform、邊學邊合成）

| 代號 | 投影片 | 做法 |
|---|---|---|
| C1 | AdaBoost-Stump | 課程的演算法：u⁽¹⁾ = w（正規化）；每輪選加權錯誤率 ε_t 最小的 decision stump h(x) = s·sign(x_i − θ)，θ 是每個 feature 在訓練 row 上的 255 個分位數 bin 的邊界，缺值自成一個 bin、放在錯誤率較低的那一邊；◆_t = √((1 − ε_t)/ε_t)，α_t = ln ◆_t；答錯的乘 ◆_t、答對的除 ◆_t。T ≤ 1000，用 fit 部分訓練、在最後 120 天選加權指數損失 Σ w·exp(−ỹ G_T) 最小的 T，再用整個訓練窗跑 T 輪。分數 G_T = Σ α_t h_t。沒有隨機性，1 次 |
| C2 | AdaBoost-DTree | 同 C1 的迴圈，base learner 換成 C&RT（加權 Gini、max_depth 3、8 個葉子、min_data_in_leaf 500、255 bin、缺值走 LightGBM 預設方向；用 LightGBM 單棵樹實作：梯度 −u·ỹ、hessian u，葉子值的正負號 = 葉子裡加權多數）；課程的「依 u 抽樣」用權重代替（期望相同）。h_t = 葉子加權多數的 ±1。T ≤ 500，選法同 C1 |
| C3 | Optimization of AdaBoost | GBDT 但 loss 換成 AdaBoost 的指數損失：LU 的所有參數不變，custom objective Σ w·exp(−ỹF)（梯度 −wỹe^(−ỹF)、hessian we^(−ỹF)），early stopping 用最後 120 天的加權指數損失；seed 1–3；分數 = F |

### D. Decision Tree（conditional、邊學邊合成）

| 代號 | 投影片 | 做法 |
|---|---|---|
| D1 | Decision Tree Algorithm ＋ CART 的 pruning | 單一 C&RT：sklearn DecisionTreeClassifier（gini、min_samples_leaf 100、random_state 1、sample_weight w、缺值用 sklearn 內建的處理——CART surrogate branch 在 sklearn / LightGBM 都沒有，這是最接近的）；cost-complexity pruning：在 fit 部分算 pruning path，取 30 個 α（path 上不重複 α 依名次等距，含 0 和還留 ≥ 2 個葉子的最大 α），用最後 120 天的加權 log-loss（葉子機率截在 [1e−4, 1 − 1e−4]）選 α，再用整個訓練窗、同樣設定重訓。分數 = predict_proba |
| D2 | CART 的 categorical branching | LU，但 C__I_dow、C__I_month 宣告成 categorical（LightGBM 的 categorical split = C&RT 的 decision subset），其他參數不變；seed 1–3 |

### E. Random Forest

| 代號 | 投影片 | 做法 |
|---|---|---|
| E1 | Random Forest Algorithm | sklearn RandomForestClassifier：500 棵、max_features sqrt、max_samples 0.5、bootstrap、gini、sample_weight w、random_state 1；min_samples_leaf ∈ {250, 1000, 4000} 用 fit 部分訓練、最後 120 天的加權 log-loss 選，再用整個訓練窗重訓（= v15 A1 的設定）。取代 C.LU |
| E2 | Random Forest in Action（RF ＋ GBDT） | zavg(C.LU, E1) |
| E4 | Feature Selection（permutation test） | 每個 fold：g⁻ = LU 第一階段的模型（fit 部分訓練、最後 120 天 early stopping，seed 1）；importance_j = IC(最後 120 天) − 3 次「把第 j 個 feature 在最後 120 天的 row 之間隨機排列」後 IC 的平均，IC = 120 天每天 Spearman(預測, e) 的平均（當天所有有 label 的股票），排列的 rng = default_rng(1000 × fold 年 + j)；保留 importance > 0 的 feature（少於 20 個就取 importance 最高的 20 個），再用 C.LU 的程序（fit_LU、seed 1–3）在這些 feature 上訓練 |
| E3 | Out-of-bag Estimate（只報告，不判定） | 每個 fold、葉子 ∈ {50, 250, 1000, 4000}：(a) 一般 RF 的 row 層級 OOB（200 棵、max_samples 0.5）；(b) 整天抽樣的 RF（每棵樹抽一半天數、可重複，OOB = 沒抽到的日子；200 棵）；(c) fit 部分訓練、最後 120 天驗證；(d) 測試年（事後）。報告三種方法選到的葉子、OOB 估計和測試年的差 |

### F. Gradient Boosting（GBDT）

| 代號 | 投影片 | 做法 |
|---|---|---|
| F1 | Gradient Boosting for regression | 課程的 GBDT（squared error，對殘差做 regression tree）：LightGBM objective regression，目標 clip(e, ±5%)，不加權，LU 的樹參數，early stopping（L2）用最後 120 天，seed 1–3 |
| F2 | 由 loss 推出的分數換算 | 不訓練：分數 = (2p − 1) × absres_60，p = C.LU 的機率。依據（命題 A）：加權 logistic 的最佳解滿足 2p* − 1 = E[clip(e)│x] / E[min(\|e\|, 5%)│x]，乘回每支股票的 \|e\| 尺度得到 E[e│x]，這才是 Top25 期望 AR 的排序。absres_60 = 前 60 天 \|e\| 的平均（只用 D−1 以前；使用前先檢查），缺值補當天中位數 |

## 4. 判定（只用 D，727 天）

1. 每個候選和 SD 的每日成對差 d_t。**入選（三個都要）：D 平均差 > 0、Newey–West t（5 lags）≥ 2.0、3 年至少 2 年 > 0**（= 第二部分）。
2. 每個家族（A–F）最多一個：入選的取平均差最大的。E3 不參加。
3. 兩個以上家族有入選者：COMBO = 各家族入選者分數的 zavg，用同一條規則對 SD 再判一次；COMBO 通過就採用 COMBO，否則採用平均差最大的單一入選者。
4. 沒有入選者 → race 版不變。
5. 只報告：Holm 校正的 p（16 個候選，用 NW t 的常態近似單尾 p）、每年的差、每日 sd、19 天平均 < 0 的比例、Top50 日 IC、和 C.LU 分數的平均日排序相關、
   每天 Top25 和 SD 差幾支、A3 的 α、A4 的 importance、AdaBoost 每個 fold 的 T 與 ε_t、RF 選到的葉子、D1 的葉子數、E4 保留幾個 feature、
   每個成員的 best_iter；「uniform blending 由成員 IC 與相關預測的 IC」和實際值的對照（第 6 節命題 B）。

## 5. V（只在 D 有採用時才打開，只算一次）

- D 有採用者時：把採用的設定、這份檔案和程式的 SHA-256 寫進 `results\LOCK.json`，之後才訓練採用者在 V 的 3 個 fold（2025、2026-01..05、
  2026-06..）需要的單位，算一次 V（2025-01-02 到 2026-11-09）的成對差（SD 在 V = 第一部分存好的 C.LU V 單位），寫 `results\SEALED_V.json`，之後拒絕再算。
- 建議改 live 的條件：D 採用 **而且** V 平均差 > 0（= race_mkt 的做法）。V 只報告平均、t、每段，不再挑別的候選。
- D 沒有採用者：V 不打開，雲端和 VM 都不放 V 的預測檔。

## 6. 事先寫下的預測（之後對照，不影響判定）

| 候選 | 預測 | 理由（證明在 README） |
|---|---|---|
| A1 | +0 到 +1，t < 2 | 命題 B：等權平均的 IC = (a₁ + a₂)/√(2 + 2ρ)；兩個 IC 相近、ρ 0.7–0.85 時只多 2–8% |
| A2 | ≤ A1，可能 < 0 | 較弱的 3 個成員稀釋（ALL5 < ALL.LN 的先例） |
| A3 | 和 A1 / A2 差不多或更差 | 120 天估 5 個權重的誤差，和等權的差距相比不小（DeMiguel et al. 2009 的 1/N 現象） |
| A4 | < 0 | 第二層只有 120 天，any blending 容易 overfit（投影片的警告） |
| B1 | +0.1 到 +0.3，t < 2 | 只降低 seed 雜訊的變異數；WF1 → WF3 +0.50 的外推 |
| B2 | ≤ 0 | 每個 bag 只有約 63% 不重複的日子，bias 增加；LightGBM 本身已有 0.8 抽樣 |
| C1、C2 | 明顯 < 0 | 指數損失對答錯的點權重 e^(\|margin\|) 無上限，label 雜訊（SNR 約 0.01）會被放大；stump 只能做加法模型 |
| C3 | < 0，但比 C1、C2 小 | 命題 C：指數損失和 logistic 的母體最佳解排序相同（½ logit），差別只在有限樣本的雜訊敏感度 |
| D1 | 明顯 < 0（−5 以下） | 單一棵樹變異數大；pruning 後葉子很少，同葉子的股票同分 |
| D2 | 約 0 | 日曆 feature 的 gain 很小 |
| E1 | −1 到 0 | v15：RF 和 LightGBM 同 IC、排序相關 0.90 |
| E2 | 0 到 +0.5 | 命題 B，ρ 約 0.9 時只多約 2–3% IC |
| E4 | −0.5 到 +0.3 | 120 天估 169 個 importance，雜訊大；LightGBM 對雜訊 feature 本來就不敏感 |
| F1 | −1 到 −2 | C.LM、C.LP 的先例；squared loss 被大 \|e\| 主導 |
| F2 | −1 到 +1 | 命題 A 說方向對；但乘上 absres_60 也放大高波動股票的估計誤差 |
| E3 | row 層級 OOB 偏樂觀、選到較小的葉子 | 同一天的股票有共同的產業衝擊，row 層級 OOB 的樣本和訓練樣本不獨立 |

## 7. 命題（README 會給直觀與詳細證明）

- 命題 A（加權 logistic 估計什麼）：在 x 固定下最小化 E[w·BCE(y, p)]，p* = E[w·y│x]/E[w│x]；w = min(\|e\|, 5%) 時 2p* − 1 = E[clip(e, ±5%)│x]/E[min(\|e\|, 5%)│x]。
- 命題 B（uniform blending 的 IC）：成員 z 分數 s_k 和報酬的相關 a_k、彼此相關 ρ_kl 時，corr(Σ s_k, r) = Σ a_k / √(Σ_k Σ_l ρ_kl)；squared error 版本是投影片的
  avg E(g_t) = E(G) + avg E[(g_t − G)²]。
- 命題 C（AdaBoost = 指數損失的 stagewise 最小化；母體最佳解 F* = ½ ln(P(ỹ = 1│x)/P(ỹ = −1│x))，和 logistic 的排序相同）。
- 命題 D（相關資料下 row 層級 OOB 偏樂觀）：y_it = f(x_it) + δ_g(i),t + ε_it（同一天同產業的共同衝擊），OOB 的期望損失比真正的下一天損失少一項和 δ 的變異數成正比的量。

## 8. 修正

（無）

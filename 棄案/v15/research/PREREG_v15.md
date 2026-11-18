# v15 事先寫好的規則：de-noising 方法和 Random Forest（2026-11-07 21:10 寫，任何 v15 成績算出來之前）

使用者的要求（11/7 20:45）：「幫我嘗試任何你能想到的方法 如量化交易或任何 fin tech 可以做到去 noise 跟提升準確度的，還有我有讀到 random forest，
我不確定這在本次比賽會如何 work，反正你先大量 search 然後開始嘗試，訓練可以在我本機跑，可以弄成 v15」。

v9、v11、W0 的檔案和每天的流程都不改。程式：`v15\research\v15lib.py`、`v15_train.py`（在筆電訓練）、`v15_signals.py`（D、E 組，不用訓練）、
`v15_eval.py`（判定）；雙擊 `v15\run_v15.bat` 全部跑完。

## 0. 寫這份規則時已經知道的

- 文獻（README_v15 第 1 節）：Random Forest 在 S&P 500 每天的橫斷面預測裡是單一模型最好的（Krauss, Do & Huck 2017），在 Gu, Kelly & Xiu (2020) 的月資料則略輸 boosting 和神經網路；
  RF 的隨機性等於一種 regularization，最適合低訊雜比（Mentch & Zhou 2020）；López de Prado (2018) 認為金融資料比較適合 bagging 而不是 boosting；
  ExtraTrees 用隨機切點降低切點的變異（Geurts, Ernst & Wehenkel 2006）；DoubleEnsemble（Zhang et al. 2020，Qlib）在 Alpha158 上的 IC 比 LightGBM 高；
  Jane Street 2021 第一名用 supervised autoencoder + MLP 和多個 target；Kalman filter 的穩態就是 EWMA（Muth 1960）；
  wavelet 去噪如果用整段資料會偷看未來（v13 定理 6、Renaud, Starck & Murtagh 2005 的因果版本）；Vasicek (1973) 的 beta 收縮。
- v1–v14 已經測過、這次不重做的：xr_ 橫斷面排序 feature（X1 和兩個選股 MLP 已經有）、MLP 的 σ^0.5 權重（v9 第 5 項已採用，但 lgb_res 沒有）、
  十分位／報酬大小的 label（v12）、regularization 的強弱（reg_svm）、近期加權（變差）、beta 調整殘差 label（v2、v8，val < 0）、產業／因子中性（v14 E3，變差）、
  各種序列模型（v8、v13）、14 個模型 1/N（v14 E5）。
- 這次的任何候選在這份資料上的成績、IC、相關、IC 的持續性，寫這份規則之前都沒有算過。
- 重組的 v9、v11、W0 的公布數字（val / hold / post / 全期）：v11 14.60 / 14.34 / 20.35 / 15.02，W0 11.79 / 10.35 / 21.71 / 12.22。

## 1. 共同設定

- 期間：val 2022–2024、hold 2025-01 到 2026-05、post 2026-06 到 10-22。分數是 Top50 的 bp/日（競賽算法），括號是 Newey-West t（5 lags）。
- 基準：v11（w_T 0.5、λ 0.75、夜盤反轉、月初多 5 支）。W0（w_T 0、λ 0.5，目前上傳的）每一項都報告，但不拿來判定（和 v13、v14 相同）。
- 資料：`v9\cache\core9.pkl`（100 支，競賽資料）。label = 贏過當天 100 支平均（y_resid），超額報酬 resid_r。
- walk-forward：6 個年度 fold（2022、2023、2024、2025、2026-01..05、post），訓練窗 2020-03-02 到 fold 開始的前一天；
  early stopping 或選參數都只用訓練窗最後 120 天，選好之後用整個訓練窗重訓（和 lgb_res、MLP、SVM 相同）。
- 每個 unit 存成 `v15\cache\units\<名稱>\<fold>_seed<s>.npz`；做完的不重做，中斷後再雙擊會接著跑。

## 2. 候選（13 個）和對照

### A 組：Random Forest（新成員，放進 LightGBM 部分，和 SVM、F-LOAM 相同）

放法：候選先縮放成 lgb_res 的當天平均和標準差，LightGBM 部分 = (lgb_res + X1 + 候選) / 3。比較：v11 + 候選 − v11。

| 代號 | 方法 | 設定（事先固定） |
|---|---|---|
| A1 | Random Forest（Breiman 2001） | sklearn RandomForestClassifier；B 組 67 個 feature（缺值補訓練窗的中位數）；label y_resid；sample_weight 和 lgb_res 相同（min(|resid_r|, 5%)）；max_features = sqrt；每棵樹抽一半的 row（bootstrap、max_samples 0.5）；min_samples_leaf 從 {250, 1000, 4000} 選：訓練窗扣掉最後 120 天訓練 200 棵，比最後 120 天的加權 log-loss；選好後整個訓練窗 500 棵。樹分批（每批 50 棵）建，結果在分布上等於一個 500 棵的 forest。每個 fold 一個 unit |
| A2 | ExtraTrees（Geurts et al. 2006） | sklearn ExtraTreesClassifier（隨機切點、不 bootstrap），其他和 A1 相同 |
| A0 | 只報告：sklearn 預設的 RF | n_estimators 100、min_samples_leaf 1（長到底）、max_features sqrt、bootstrap；同樣的 feature 和權重。用來說明「照預設用 RF」在這裡會怎樣 |

### B 組：lgb_res 的 de-noising（取代 lgb_res）

放法：候選取代 lgb_res（縮放成 lgb_res 的當天平均和標準差），LightGBM 部分 = (候選 + X1) / 2。
**比較：v11[lgb_res := B_k] − v11[lgb_res := B0]**；B0 是同一台電腦、同一份程式、同樣 seed 重訓的 lgb_res（同樣縮放）。
這樣差異只來自 de-noising 本身，不來自機器或 LightGBM 版本（v10 看過：極小的數值差異就能讓 hold 差 0.4）。

| 代號 | 方法 | 設定（事先固定） |
|---|---|---|
| B0 | 對照：lgb_res 重訓 | = v10lib.train_lgb_unit：B 組 67、binary、權重 min(|r|, 5%)、lr 0.02、15 葉、min_data_in_leaf 500、feature/bagging 0.8、λ_l2 1、單執行緒；early stopping 100 輪（最多 1000）；seed 1–5 |
| B1 | 異質變異加權（GLS，Aitken 1935） | 權重 = min(|r|, 5%) / σ^0.5，σ = absres_60（前 60 天平均 |超額報酬|，到前一天為止；缺值補訓練窗中位數）。和 v9 四個選股 MLP 的權重相同 |
| B2 | Huber 權重（Huber 1964） | 權重 = min(|r|, 2 × 當天 100 支 |r| 的中位數)（常態下約 1.35σ，Huber 的建議值） |
| B3 | 3 天 target（多期平均去噪） | r̄ = 同一支股票 D、D+1、D+2 三天超額報酬的平均；label 1{r̄ > 0}、權重 min(|r̄|, 5%)。purging：訓練窗只用三天都在窗內的 row；early stopping 的訓練部分只用三天都在切點之前的 row |
| B4 | 5 天 target | 同 B3，5 天 |
| B5 | 拿掉除權息事件日 | 訓練時拿掉 ex_today、ex_next1、ex_prev1、ex_prev2 任一個是 1 的 row（這些日子規則本來就猜 0，模型不需要學；label 被事件主導，對模型是雜訊）|
| B6 | DoubleEnsemble（Zhang et al. 2020，Qlib） | Qlib Alpha158 的設定：3 個子模型、sample reweighting（α1 = α2 = 1、10 個 bin、decay 0.5）、feature selection（5 個 bin、比例 0.8 / 0.7 / 0.6 / 0.5 / 0.4）；子模型 = lgb_res 的設定和權重（再乘上 SR 權重）；每個子模型的輪數 = B0 的 early stopping 最佳輪數；loss curve 在 10 個 checkpoint、每個樣本的 log-loss；feature selection 用訓練窗隨機 20,000 row；預測 = 3 個子模型平均。seed 1–3（比較時 B0 也只用 seed 1–3）|

### C 組：MLP 的 de-noising（取代 mlp_res_xr9）

放法：候選取代四個選股 MLP 裡的 mlp_res_xr9（縮放成 mlp_res_xr9 的當天平均和標準差）。**比較：v11[xr9 := C_k] − v11[xr9 := C0]**。

| 代號 | 方法 | 設定（事先固定） |
|---|---|---|
| C0 | 對照：mlp_res_xr9 重訓 | = v9_train.train_custom（複製）：feature B9 + XR + SIG（96 個）、權重 min(|r|, 5%) / σ^0.5、64 → 32、GELU、dropout 0.2、AdamW lr 1e-3、wd 1e-2、batch 1024、最多 10 epoch、patience 2；seed 1–5 |
| C1 | 多任務 MLP（輔助 target） | 同樣的主幹；三個輸出：D（主要）、D+1、D+2 的 y_resid（同一支股票，權重同 C0 的算法）；loss = 主要 + 0.5 × (D+1 + D+2) / 2；輔助 label 在訓練窗外、或在 early stopping 切點之後的，權重設 0；early stopping 只看主要輸出；預測只用主要輸出 |
| C2 | supervised autoencoder + MLP（Jane Street 2021 第一名的架構，縮小） | 輸入加 N(0, 0.1²) 雜訊（只在訓練）→ encoder Linear(96, 32) + GELU → decoder Linear(32, 96)（對乾淨輸入的 MSE）＋ AE 輸出 Linear(32, 1)（主要 label 的 BCE）；主 MLP 吃 [輸入, encoder] → 64 → 32 → 1（dropout 0.2）；loss = MSE + BCE_AE + BCE_主；early stopping 只看主要輸出；預測用主 MLP |

### D 組：訊號層級的濾波去噪（不用訓練）

放法：和 v14 相同的 spread-matched new family（權重 1/3）：z 當天標準化（缺值 0、±3 截斷），p_raw = (2·p_e + sd(p_e)·z) / 3，再對齊 p_e 的當天標準差。
比較：v11 + 候選 − v11。

| 代號 | 方法 | 設定（事先固定） |
|---|---|---|
| D1 | Kalman filter（local level） | 每支股票的超額報酬 y_t = μ_t + ε_t、μ_t = μ_{t−1} + η_t；q = Var(η) / Var(ε) 用 fold 開始之前的資料做 concentrated MLE（10^−6 到 10^−1、26 個 log 格點）；訊號 = μ̂_{D | D−1} |
| D2 | 因果 wavelet 去噪（à trous Haar，Renaud, Starck & Murtagh 2005） | 每支股票的超額報酬；往回看的 Haar：c_j(t) = (c_{j−1}(t) + c_{j−1}(t − 2^{j−1})) / 2、w_j = c_{j−1} − c_j，J = 5；soft threshold √(2 ln 250) · σ_j(t)，σ_j(t) = w_j 前 250 天的 MAD / 0.6745；訊號 = 去噪後 D−1 的值 c_5 + Σ soft(w_j) |

截斷檢查：資料截斷在 D−1 重算，D 的訊號要完全相同。

### E 組：決策層級

| 代號 | 方法 | 設定（事先固定） |
|---|---|---|
| E1 | Vasicek (1973) beta 收縮（對沖用） | 和 hedge_beta 相同的 120 天回歸（盤中報酬對 Top100 平均，到前一天），另外算標準誤 se；每天 β̄ = 100 支的平均、先驗變異 s² = max(橫斷面變異 − 平均 se², 0.01)；β_V = (s²·β̂ + se²·β̄) / (s² + se²)；再照原本截斷到 [0.2, 2.5]。v11 和 W0 的對沖都改用 β_V |

## 3. 判定（和 v11b、v12、v13、v14 同一種規則）

1. 每個候選單獨（第 2 節各組的比較）：**val > 0 → 入選**。
2. 入選的合起來成 v15：
   - LightGBM 部分 = [B 部分, X1, 入選的 A（縮放）]；B 部分 = lgb_res（沒有 B 入選時），否則 = 入選的 B 各自縮放後平均。
   - mlp_res_xr9 := 入選的 C 各自縮放後平均（沒有就不變）。
   - 入選的 D 一起當 family：p_raw = (2·p_e + Σ_j sd(p_e)·z_j) / (2 + J)，再對齊標準差。
   - E1 入選就用 β_V。
   - **v15 − v11：hold > 0 而且 post > 0 → 通過**。
3. 通過也只是研究結果：要不要改每天的程式，由使用者在 11/23 定版前決定；通過才寫每天的程式（v15 現在不寫）。
4. 沒有效果時，13 個候選「至少一個入選、合起來兩段都 > 0」的機率 ≤ 1/4（README_v14 定理 6）。這個規則主要是擋掉會變差的改法。

## 4. 只報告（不影響判定）

- 每個候選和 v15 的 W0 版本。
- 每個成員、訊號自己的 rank IC（val / hold / post / 全期）；成對的 IC 差（B_k − B0、C_k − C0、A_k − lgb_res、A0 − A1）；和 v11 選股觀點的相關；v14 定理 2 的診斷（w*、w₀、最好的混合 IC）。
- RF：每個 fold 選到的葉子大小和三個值的 log-loss、A0（預設）對 A1、feature importance（MDI 前 10）；RF 取代 lgb_res；RF 當 new family。
- B3、B4：對 D 天和對多天 target 的 IC。B6：每個子模型保留的 feature 數、SR 權重的分布。C0 和存好的 mlp_res_xr9（seed 1–10）的相關；B0 和存好的 lgb_res 的相關。
- 逐年、10/1–10/22、規則的檢定力（v13 定理 3）。

## 5. 計算量（筆電 i5-10210U，2 個 worker 的估計）

- LightGBM：B0–B5 共 180 個 unit，每個約 0.3–1 分鐘；B6 18 個 unit，每個約 3–6 分鐘。
- RF：A1、A2 各 6 個 unit，每個約 3–8 分鐘；A0 6 個 unit。
- MLP：C0–C2 共 90 個 unit，每個約 0.2–1 分鐘。
- 合計約 2–4 小時。`run_v15.bat test` 是幾分鐘的小測試（小模型、少量資料，輸出在 `cache\units_test\`，不拿來判定）。

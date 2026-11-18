# v15：de-noising 方法和 Random Forest（2026-11-07，只有研究，沒改任何每天的程式）

使用者的問題（11/7 20:45）：「幫我嘗試任何你能想到的方法 如量化交易或任何 fin tech 可以做到去 noise 跟提升準確度的，還有我有讀到 random forest，
我不確定這在本次比賽會如何 work，反正你先大量 search 然後開始嘗試，訓練可以在我本機跑，可以弄成 v15」。

所以 v15 做三件事：

1. **文獻搜尋**（第 1 節）：去噪和 Random Forest 相關的方法、主張、在這個比賽能不能用、v1–v14 有沒有做過。
2. **13 個候選**，規則在任何成績之前寫好（`research\PREREG_v15.md`，11/7 21:10）：
   A 組 Random Forest、ExtraTrees；B 組 lgb_res 的 6 種去噪；C 組 MLP 的 2 種去噪；D 組 2 種濾波（Kalman、因果 wavelet）；E 組 beta 收縮。
   A、B、C 在筆電訓練（11/7 22:07–23:33，306 個 unit，85.9 分鐘）；D、E 不用訓練。結果在第 0、5、6 節。
3. **理論**（第 4 節，每個都有直觀和證明）：為什麼「去噪」在這裡大多只能降低變異、不能改變方向；RF 的葉子為什麼一定要大；多天 target 什麼時候划算；Kalman 為什麼等於 EWMA。

val 2022–2024、hold 2025-01–2026-05、post 2026-06–10-22。分數是 Top50 的 bp/日，括號是 Newey-West t。

## 0. 結論

**照事先寫好的規則，v15 沒有通過；v9、v11、W0 都不變。** 入選 7 個（A1、B2、B3、B4、B6、D2、E1），合起來 v15 − v11：
val +0.24（0.30）、hold −0.79（−0.85）、post −1.63（−0.46）、全期 −0.22（−0.34）；W0 版本 −0.08 / −2.14 / −0.34 / −0.70。

| 候選 | 方法 | 和基準比：val | hold | post | 全期 | 判定 |
|---|---|---|---|---|---|---|
| A1 | Random Forest | +0.19（0.67） | −0.65（−1.70） | +2.67（2.20） | +0.16（0.67） | 入選 |
| A2 | ExtraTrees | −0.03（−0.09） | −0.86（−2.05） | +2.03（1.11） | −0.09（−0.33） | 不入選 |
| B1 | 權重 ÷ σ^0.5 | −0.12（−0.51） | +0.09（0.29） | −0.46（−0.94） | −0.09（−0.49） | 不入選 |
| B2 | Huber 權重 | +0.29（1.10） | +0.28（0.70） | +0.57（0.59） | +0.31（1.42） | 入選 |
| B3 | 3 天 target | +0.42（1.28） | −0.09（−0.17） | +2.75（1.35） | +0.47（1.50） | 入選 |
| B4 | 5 天 target | +0.10（0.27） | +0.03（0.04） | +1.47（0.54） | +0.20（0.52） | 入選 |
| B5 | 拿掉除權息日 | −0.03（−0.12） | −0.16（−0.42） | +0.07（0.09） | −0.06（−0.30） | 不入選 |
| B6 | DoubleEnsemble | +0.09（0.28） | −0.18（−0.36） | −1.72（−1.10） | −0.14（−0.50） | 入選 |
| C1 | 多任務 MLP | −0.10（−0.63） | −0.27（−0.95） | −0.19（−0.32） | −0.16（−1.14） | 不入選 |
| C2 | supervised autoencoder + MLP | −0.21（−0.98） | −0.72（−1.84） | +0.08（0.21） | −0.33（−1.86） | 不入選 |
| D1 | Kalman filter | −0.53（−0.88） | −0.42（−0.51） | −0.09（−0.03） | −0.46（−0.92） | 不入選 |
| D2 | 因果 wavelet | +0.34（0.44） | −0.99（−1.12） | −0.72（−0.25） | −0.14（−0.23） | 入選 |
| E1 | Vasicek beta 收縮 | +0.30（1.97） | +0.14（0.69） | −0.23（−0.55） | +0.21（1.78） | 入選 |
| v15 | 入選的合起來 | +0.24（0.30） | −0.79（−0.85） | −1.63（−0.46） | −0.22（−0.34） | **沒通過** |

A 組、D 組、E 組和 v11 比；B 組和同一台電腦重訓的 lgb_res（B0）比，B6 和 B0 的 seed 1–3 比；C 組和重訓的 mlp_res_xr9（C0）比。

五個發現：

- **Random Forest 調對了就和 lgb_res 一樣好，但沒有更好；用 sklearn 的預設會差很多。** A1 的 IC 0.1058，lgb_res 0.1050（差 +0.0009，t 0.34），兩者每天的排序相關 0.90。
  sklearn 預設（葉子長到 1 個樣本）的 IC 只有 0.0698，比 A1 低 0.036（t −7.8），這就是定理 6 說的「葉子太小只是在記雜訊」。
  RF 加進 v11 是 post 好（+2.67）、hold 差（−0.65），和 v13 的 LSTM、CNN、F-LOAM 同一個型態：post 期間 lgb_res 和選股 MLP 的 IC 掉到 0.08，任何不一樣的模型都會把它拉高，是同一件事，不是獨立的證據（第 5 節）。
- **IC 變高不等於分數變高。** 多任務（C1）、autoencoder（C2）讓這個 MLP 的 IC 明顯變高（全期 +0.0018、+0.0020，t 2.75、2.03；C1 在 hold +0.0047，t 3.36），放進 v11 卻是負的（全期 −0.16、−0.33）。
  一個 MLP 在選股觀點只佔 1/8，IC 多 0.002 換算成分數不到 0.05 bp/日（v14 定理 7），被決策的雜訊蓋過。反過來，B3、B4 的 IC 比 B0 低（−0.0033、−0.0091），分數卻是正的。
- **label 和樣本的去噪大多讓 IC 變低。** σ 權重（B1）只有 val 的 IC 高一點（+0.0023，t 2.24），hold、post 都低（t −2.6、−2.7）；拿掉除權息日（B5）−0.0015（t −2.5）；DoubleEnsemble（B6）−0.0072（t −2.4），post 掉最多（0.0553 對 0.0797）。
  多天 target 預測「3 天、5 天平均」比 B0 準（0.1411 對 0.1288、0.1590 對 0.1426），但對 D 天本身沒有變好：持續的那部分 B0 本來就抓得到（定理 5）。
- **單獨三段都 > 0 的只有 B2（Huber）和 B4（5 天），但 13 個候選裡本來就很容易出現。** 沒有效果時每個的機率是 1/8，13 個裡至少一個的機率是 1 − (7/8)^13 ≈ 0.82。
  而且 B2 的 IC 其實比 B0 低（post −0.0071），W0 版本全期 −0.01、post −1.72；B4 三段的 t 都 < 0.6。
- **最穩的是 Vasicek beta 收縮，但很小。** E1 逐年 6 段有 5 段為正，全期 +0.21（t 1.78），W0 +0.29（t 1.89），只有 post −0.23。它只改對沖，每天的差 sd 只有 3–4 bp，所以 t 相對高；但 +0.2–0.3 bp/日在 19 天裡看不出來。

所以：去噪方法和 Random Forest 在這份資料上都沒有比現在的模型好，瓶頸仍然是資訊（和 v14 相同），不是雜訊處理的方式或模型的種類。
- 用更嚴的標準（全期 t ≥ 2）：13 個沒有一個通過，最高的是 E1 的 1.78。
- 如果只看 post > 0：會採用 A1、A2、B2、B3、B4、B5、C2 共 7 個，其中 5 個 hold 是負的（A1 −0.65、A2 −0.86、B3 −0.09、B5 −0.16、C2 −0.72）。
- Kalman 的 MLE 讓 47–73% 的股票 q 落在格點下限（半衰期 500–700 天）：最好的濾波就是很長的平均，B 組本來就有 intra_bias_exp。
- 因果 wavelet（D2）自己的 IC 只有 0.050。

## 1. 搜尋到的方法和在這個比賽的狀態

| 方法 | 文獻 | 文獻的主張 | 在這個比賽 | v15 |
|---|---|---|---|---|
| Random Forest | Breiman (2001)；Krauss, Do & Huck (2017)；Gu, Kelly & Xiu (2020) | S&P 500 每天預測「明天贏大盤的機率」：RF 0.43%/日、gradient-boosted trees 0.37%、DNN 0.33%（各取前後 10 支、未扣成本），三個平均超過 0.45%，但近年獲利下降；GKX 的月資料裡 trees 和神經網路最好 | 題目幾乎一樣（我們的 lgb_res 也是「贏過當天平均」的機率） | A1 |
| RF 的隨機性 = regularization | Mentch & Zhou (2020) | 每次分裂只看部分 feature（mtry）的作用和 ridge、lasso 的收縮相同，所以 RF 最適合低訊雜比 | 這裡訊雜比極低（IC 約 0.1） | 定理 6 |
| 金融資料用 bagging 不用 boosting | López de Prado (2018) 第 6 章 | boosting 會去擬合雜訊；樣本不獨立時 RF 也會 overfit，要用 max_samples = 平均 uniqueness、早停或少一點 feature | label 只有 1 天、沒有重疊；同產業的相關很弱（ICC 0.079，reg_svm） | A1 每棵樹抽一半 row |
| ExtraTrees | Geurts, Ernst & Wehenkel (2006) | 切點也隨機，降低「切點的變異」，模型更平滑 | 可以 | A2 |
| DoubleEnsemble | Zhang et al. (2020)；Qlib benchmark | 依每個樣本的 learning curve 調權重（壓低太容易和太難＝雜訊的樣本）＋ shuffle 的 feature selection；Qlib 的 benchmark（Alpha158、CSI300）IC 約 0.052–0.054，LightGBM 約 0.045–0.048（不同版本的表） | 可以，基底換成 lgb_res | B6 |
| supervised autoencoder + MLP、多個 target | Jane Street 2021 第一名（Yirun Zhang）；Bieganowski & Ślepaczuk (2024) | bottleneck 和 Gaussian noise 去噪；多個 target 一起學；作者說 sample weight 幫助最大；雜訊太大、bottleneck 太大會變差 | 可以 | C2（AE）、C1（多 target） |
| label 去噪（自監督、迭代修正） | Ma, Ventre & Polukarov (2021)；Zeng et al. (2024, LARA) | 用 autoencoder 或多個模型修正「雜訊 label」 | 這裡 label 是報酬的正負號，雜訊不是「標錯」，而是報酬本身大部分不可預測；對稱的雜訊不改變排序（定理 1） | 不做；DoubleEnsemble 的 sample reweighting 是同一類想法 |
| 異質變異加權（GLS） | Aitken (1935)；v9 第 5 項 | σ 大的股票 label 比較吵，權重除以 σ | 四個選股 MLP 已經用 σ^0.5；lgb_res 沒有 | B1（定理 4） |
| robust 權重 | Huber (1964) | 限制極端報酬（新聞、跳空）的影響 | 可以 | B2（定理 2、3） |
| 多期 target | Heston, Korajczyk & Sadka (2010) | 盤中報酬在整數天的間隔有延續，可以持續 40 天 | 持續的部分可以多天平均來去噪 | B3、B4（定理 5） |
| 除權息日的 label | v7、v9 的規則 | 這些日子規則本來就猜 0 | label 被事件主導，模型看不到 | B5（定理 4 (i)） |
| Kalman filter | Muth (1960)；Harvey (1989) | local level 模型的最佳濾波就是 EWMA | B 組有 intra_bias_20/60/250/exp | D1（定理 7） |
| wavelet 去噪 | Renaud, Starck & Murtagh (2005)；Johnstone & Silverman (1997) | 只往回看的 à trous Haar 可以拿來預測；每一層分開估雜訊 | 很多論文用整段資料做 wavelet，等於偷看未來（v13 定理 6） | D2（定理 8） |
| 資訊洩漏 | Algorithmic Finance (2019) "Information leakage in financial machine learning research" | 超過一半的投稿有 leakage，去噪常常用整段資料做 | 所有 D 組訊號都做截斷檢查 | — |
| beta 收縮 | Vasicek (1973)；Blume (1975)；Lally (1998)；Levi & Welch (2017) | 收縮的 beta 預測誤差比較小 | 對沖用 120 天 beta | E1（定理 9） |
| volatility scaling | Lim, Zohren & Roberts (2019) | 時間序列動能策略裡波動度縮放很關鍵 | 部位大小沒得調（v14 定理 1）；只剩樣本權重的意義，就是 B1 | B1 |
| 橫斷面排序 feature | Gu, Kelly & Xiu (2020) | 每期把 feature 換成橫斷面排序 | xr_ feature 已經有（X1、兩個選股 MLP） | 不重做 |
| Kaggle 金融比賽的經驗 | JPX Tokyo (2022) 第 4 名 flaty | 很難贏過隨機，名次受運氣影響很大 | 和 overfit 檢查的結論一樣 | — |

## 2. 在筆電上跑

- 雙擊 `v15\run_v15.bat`：先做 2–3 分鐘的小測試（每種模型都跑一遍小的），通過才開始正式訓練（約 1 小時），最後算 D、E 訊號（約 1 分鐘）和判定（約 1–3 分鐘）。
  結果在 `results\v15_eval.txt`。
  - `run_v15.bat test`：只做小測試。`run_v15.bat eval`：只算訊號和判定（做完多少 unit 就判多少）。
  - `run_v15.bat all 4`：4 個 worker，比較快，需要約 6 GB 空的記憶體；`all 1` 只用 1 個。
- 做完的 unit 存在 `cache\units\`，隨時可以關掉視窗，再雙擊會接著做。進度在 `results\v15_train.log`。
- 需要 lightgbm、scikit-learn、torch（bat 會檢查；之前的版本都裝過）。
- 雲端測過（lightgbm 4.7.0、scikit-learn 1.9.1、torch 2.15.0）：
  - B0 和存好的 lgb_res 在 2022 fold 最大差 5 × 10⁻⁸（存檔的小數位數），也就是完全重現。
  - C0（2 個 seed）和存好的 mlp_res_xr9（10 個 seed）每天排序相關 0.989。
  - 最大的 fold（post，15.2 萬 row）：LightGBM 一個 unit 約 20–30 秒、DoubleEnsemble 約 1 分鐘、RF 約 2 分鐘、ExtraTrees 約 50 秒、預設 RF 約 45 秒、MLP 約 5–7 秒；每個 process 的記憶體約 1.2–1.9 GB。全部加起來估計 40–60 分鐘（2 個 worker）。
- 你電腦的 VM 測過判定程式（numpy、pandas）：重組的 v9、v11、W0 和公布的數字逐位相同；E1 的原始 beta 和 hedge_beta 完全相同（最大差 0）。

## 3. 候選（細節在 `research\PREREG_v15.md`）

| 組 | 代號 | 方法 | 放法和比較 |
|---|---|---|---|
| A | A1 | Random Forest：500 棵、每棵抽一半 row、max_features sqrt、葉子大小從 {250, 1000, 4000} 用訓練窗最後 120 天選 | 縮放後放進 LightGBM 部分：(lgb_res + X1 + A1) / 3；v11 + A1 − v11 |
| | A2 | ExtraTrees：隨機切點、不抽樣，其他同 A1 | 同上 |
| | A0 | 只報告：sklearn 預設的 RF（100 棵、葉子 1） | 同上，只報告 |
| B | B0 | 對照：同一台電腦重訓的 lgb_res | — |
| | B1 | 權重 ÷ σ^0.5（GLS） | 取代 lgb_res；**和 B0 比**（v11[lgb_res := B_k] − v11[lgb_res := B0]） |
| | B2 | Huber 權重：上限 2 × 當天 |r| 中位數 | 同上 |
| | B3、B4 | 3 天、5 天的平均超額報酬當 target（purging） | 同上 |
| | B5 | 拿掉除權息事件日的 row | 同上 |
| | B6 | DoubleEnsemble（3 個子模型） | 同上（seed 1–3，和 B0 的 seed 1–3 比） |
| C | C0 | 對照：重訓的 mlp_res_xr9 | — |
| | C1 | 多任務 MLP：D、D+1、D+2 三個輸出，只用 D | 取代 mlp_res_xr9；**和 C0 比** |
| | C2 | supervised autoencoder + MLP | 同上 |
| D | D1 | Kalman local level | v14 的 new family（權重 1/3）；v11 + D1 − v11 |
| | D2 | 因果 à trous Haar 去噪 | 同上 |
| E | E1 | Vasicek beta 收縮 | 對沖改用 β_V；v11 + E1 − v11 |

規則和 v11b、v12、v13、v14 相同：單獨 val > 0 入選；入選的合起來成 v15，hold 和 post 都 > 0 才通過。W0 版本全部報告。

為什麼 B、C 組要和重訓的對照比：v10 看過，同樣的程式換一台電腦或換 LightGBM 版本，極小的數值差異就能讓 hold 差 0.4 bp。
和同一台電腦、同一份程式、同樣 seed 的對照比，差異才只來自去噪本身。

## 4. 理論（直觀＋證明）

記號：x 是 D 天 09:00 前知道的 feature，r 是 D 天的超額盤中報酬，y = 1{r > 0}，s(x) ∈ {−1, +1} 是決策（猜 1 或 0）。

### 4.1 對稱的 label 雜訊不改變排序（定理 1）

直觀：如果每個 label 都以同樣的機率被隨機翻掉，每支股票「贏」的機率都被往 1/2 拉近同樣的比例，誰比誰高不會變。
所以「把 label 的雜訊去掉」本身不會讓排序變對，只會讓估計的變異變小；真正能幫忙的，只有和 x 有關的雜訊（例如波動大的股票比較吵）或估計變異。

**定理 1**：η(x) = P(y = 1 | x)。觀察到的 label ỹ 以機率 ρ < 1/2 翻轉，翻不翻和 x、y 無關。則 P(ỹ = 1 | x) = ρ + (1 − 2ρ)η(x)，是 η 的嚴格遞增函數；
所以依 P(ỹ = 1 | x) 排序和依 η(x) 排序相同，而且 P(ỹ = 1 | x) > 1/2 若且唯若 η(x) > 1/2。

證明：P(ỹ = 1 | x) = P(y = 1 | x)(1 − ρ) + P(y = 0 | x)ρ = η(1 − ρ) + (1 − η)ρ = ρ + (1 − 2ρ)η。因為 1 − 2ρ > 0，這是 η 的嚴格遞增函數。
另外 P(ỹ = 1 | x) − 1/2 = (1 − 2ρ)(η − 1/2)，兩邊同號。∎

### 4.2 加權分類在學什麼：|r| 權重、Huber 權重（定理 2、3）

直觀：lgb_res 用 |r| 當權重，等於直接最大化比賽的分數；把權重的上限壓低（Huber），等於改成最大化「被截斷的報酬」。
如果報酬的雜訊對稱，截斷不會改變正負號的答案，只會讓極端日子（新聞、跳空）少一點影響力。

**定理 2**：g(r) ≥ 0 是偶函數（g(r) = g(−r)）。最小化 E[g(r)·1{s(x) ≠ sign(r)}] 的決策是 s*(x) = sign(E[g(r)·sign(r) | x])。
特別是 g(r) = |r| 時 s* = sign(E[r | x])（就是讓 AR 最大）；g(r) = min(|r|, c) 時 s* = sign(E[ψ_c(r) | x])，ψ_c(r) = sign(r)·min(|r|, c) 是 Huber 的 ψ 函數。

證明：給定 x，1{s ≠ sign(r)} = (1 − s·sign(r)) / 2，所以 E[g·1{s ≠ sign r} | x] = (E[g | x] − s·E[g·sign(r) | x]) / 2。
s 只能是 ±1，讓這個式子最小的是 s = sign(E[g·sign(r) | x])。g = |r| 時 g·sign(r) = r；g = min(|r|, c) 時 g·sign(r) = ψ_c(r)。∎

**定理 3**：給定 x，r = μ + ε，ε 對 0 對稱；ψ 是奇函數而且不遞減。則 μ > 0 時 E[ψ(r) | x] ≥ 0，μ < 0 時 ≤ 0。所以在對稱雜訊下，B2 的目標和真正的平均同號。

證明：ε 和 −ε 同分布，所以 E[ψ(μ + ε)] = E[ψ(μ − ε)]，因此 E[ψ(μ + ε)] = ½E[ψ(μ + ε) + ψ(μ − ε)]。ψ 是奇函數：ψ(μ − ε) = −ψ(ε − μ)。
μ > 0 時 μ + ε > ε − μ，ψ 不遞減，所以 ψ(μ + ε) ≥ ψ(ε − μ)，也就是 ψ(μ + ε) + ψ(μ − ε) ≥ 0，取期望得 ≥ 0。μ < 0 時同理。∎

如果雜訊不對稱（例如壞消息比好消息大），截斷就可能改變答案，這是 B2 的風險。

### 4.3 只依 feature 的權重不改變答案，但改變估計的變異（定理 4）

直觀：如果權重只和事先知道的東西有關（例如這支股票最近的波動 σ，或「明天是除權息日」），它不會改變每一點上的最佳答案，只改變「哪些樣本被學得比較仔細」。
在雜訊大小不同的時候，給雜訊小的樣本較大權重，估計的變異最小（GLS）。

**定理 4**：
(i) a(x) > 0 只依 x。最小化 E[a(x)·g(r)·1{s(x) ≠ sign(r)}] 的決策和沒有 a(x) 時相同。a(x) = 0 的區域（B5 拿掉的 row）則完全不影響其他區域的答案。
(ii)（Aitken）線性模型 y = Xβ + ε，ε 的共變異矩陣 Σ = diag(σ_i²)。所有線性不偏估計量裡，權重 W = Σ⁻¹ 的 WLS β̂_W = (XᵀWX)⁻¹XᵀWy 的變異最小：任何 CX = I 的 β̃ = Cy 都有 Var(β̃) − Var(β̂_W) 半正定。

證明：(i) 給定 x，目標是 a(x)·(E[g | x] − s·E[g·sign r | x]) / 2，a(x) > 0 是常數倍，最小的 s 不變。
(ii) 令 C = (XᵀWX)⁻¹XᵀW + Dm。CX = I 而且 (XᵀWX)⁻¹XᵀWX = I，所以 DmX = 0。
Var(Cy) = CΣCᵀ = (XᵀWX)⁻¹XᵀWΣWX(XᵀWX)⁻¹ + (XᵀWX)⁻¹XᵀWΣDmᵀ + DmΣWX(XᵀWX)⁻¹ + DmΣDmᵀ。
WΣ = I，所以第一項 = (XᵀWX)⁻¹；第二項 = (XᵀWX)⁻¹(DmX)ᵀ = 0，第三項同理是 0。所以 Var(Cy) = Var(β̂_W) + DmΣDmᵀ，後者半正定。∎

B1 用 σ^0.5 而不是 1/σ²：v5 在 MLP 上測過，α = 0.5 的 val t（0.94）比 α = 1（0.54）高；σ 本身也是估出來的，用到 1/σ² 會放大 σ 的估計誤差。

### 4.4 多天 target：雜訊變小，但「只活一天」的訊號被稀釋（定理 5）

直觀：把 D、D+1、D+2 三天的超額報酬平均起來當 label，不可預測的雜訊變成原來的 1/3；但只在 D 天有效的訊號（例如對昨天新聞的反應）也只剩 1/3。
所以只有在可預測的部分大多是「持續的」（例如股票固定的盤中習性）時才划算。另外 label 用到未來 k − 1 天，訓練窗最後 k − 1 天的 label 會碰到測試期，一定要拿掉（purging）。

**定理 5**：r_{t+j} = a(x_t) + b(x_t)·1{j = 0} + e_{t+j}（j = 0, …, k − 1），e 彼此獨立、平均 0、變異 σ²。k 天 target y⁽ᵏ⁾ = (1/k)Σ_j r_{t+j}。則
(i) E[y⁽ᵏ⁾ | x] = a + b/k，Var(y⁽ᵏ⁾ | x) = σ²/k；
(ii) 用 n 個樣本、p 個參數、模型設定正確的最小平方法估計時，對「D 天的平均」m(x) = a + b 的平均平方誤差約為 (1 − 1/k)²E[b²] + σ²p/(kn)（k = 1 時是 σ²p/n）。
所以 k 天 target 比較好若且唯若 (1 − 1/k)·E[b²] < σ²p/n。

證明：(i) 平均和變異直接計算：E[y⁽ᵏ⁾ | x] = (1/k)(k·a + b) = a + b/k；e 獨立，Var = (1/k²)·k·σ² = σ²/k。
(ii) 估計量 m̂_k 對自己的目標 m_k = a + b/k 不偏，變異約 (σ²/k)·p/n（最小平方法在隨機 x 上的標準結果 E[xᵀ(XᵀX)⁻¹x] ≈ p/n）。
m̂_k − m = (m̂_k − m_k) + (m_k − m) = 估計誤差 − b(1 − 1/k)，兩項不相關，平均平方誤差 = σ²p/(kn) + (1 − 1/k)²E[b²]。
和 σ²p/n 比：σ²p/(kn) + (1 − 1/k)²E[b²] < σ²p/n ⟺ (1 − 1/k)²E[b²] < σ²p(1 − 1/k)/n ⟺ (1 − 1/k)E[b²] < σ²p/n。∎

### 4.5 Random Forest：葉子要夠大，樹再多也有下限（定理 6）

直觀：這份資料每個 label 大約是 49 比 51 的硬幣。一片葉子只有幾個樣本時，它的平均幾乎全是雜訊；要好幾百個樣本，平均才開始有訊號。
所以 sklearn 預設的 RF（葉子長到 1 個樣本）在這裡幾乎是在記雜訊。另外多種幾棵樹只能平均掉「樹和樹之間不同」的那部分雜訊，大家共同的部分（相關 ρ）怎麼種都去不掉。

**定理 6**：
(i) P(y = 1 | x) = 1/2 + δ(x)，E[δ] = 0、E[δ²] = τ²。則任何只依 x 的分數和 y 的相關最多 2τ（在 δ 本身達到）；
一片有 n 個樣本的葉子，估計值的雜訊變異 ≤ 1/(4n)，而葉子之間訊號的變異 ≤ τ²，所以要讓訊號不小於雜訊，需要 n ≥ 1/(4τ²) = 1/(2τ)²。
(ii)（Breiman 2001；Hastie et al. 2009 式 15.1）B 棵樹的預測同分布、變異 σ²、兩兩相關 ρ，則平均的變異 = ρσ² + (1 − ρ)σ²/B，B → ∞ 時趨近 ρσ²。

證明：(i) E[y | x] = 1/2 + δ，所以 Cov(f(x), y) = Cov(f(x), δ(x))，相關 = Cov(f, δ)/(sd(f)·sd(y)) ≤ sd(δ)/sd(y) = τ/(1/2) = 2τ（Cauchy–Schwarz，sd(y) = 1/2 因為 E[y] = 1/2）。
葉子估計值 p̂ 是 n 個 Bernoulli(p) 的平均，Var(p̂) = p(1 − p)/n ≤ 1/(4n)；葉子的真值 1/2 + δ 在葉子之間的變異不超過 δ 的變異 τ²（條件期望的變異不超過原變數的變異）。
訊號 ≥ 雜訊需要 τ² ≥ 1/(4n)，即 n ≥ 1/(4τ²)。
(ii) Var((1/B)Σ_b T_b) = (1/B²)(Σ_b Var(T_b) + Σ_{b≠b'} Cov(T_b, T_b')) = (1/B²)(Bσ² + B(B − 1)ρσ²) = ρσ² + (1 − ρ)σ²/B。∎

數字：如果最好的分數和「贏不贏」的相關約 0.05–0.1（2τ），n 至少 100–400。這就是 A1 的葉子格點從 250 開始、lgb_res 的 min_data_in_leaf 是 500 的原因。

### 4.6 Kalman filter 的穩態就是 EWMA（定理 7，Muth 1960）

直觀：把「一支股票的超額報酬」看成「慢慢變的真值 + 每天的雜訊」，Kalman filter 每天把預測往今天的值拉一個固定比例 K。
這就是指數加權平均；K 由「真值變動 / 雜訊」的比例 q 決定。q 越小，越接近長期平均。

**定理 7**：y_t = μ_t + ε_t、μ_t = μ_{t−1} + η_t，q = Var(η)/Var(ε)。以 Var(ε) 為單位，預測變異 P_t 滿足 P_{t+1} = P_t/(P_t + 1) + q，
從任何 P_1 ≥ 0 都收斂到 P* = (q + √(q² + 4q))/2，增益收斂到 K* = (−q + √(q² + 4q))/2；穩態的一步預測是
μ̂_{t+1|t} = (1 − K*)μ̂_{t|t−1} + K*y_t = K*Σ_{j≥0}(1 − K*)^j y_{t−j}。

證明：Kalman 的更新：增益 K_t = P_t/(P_t + 1)，濾波後變異 P_t(1 − K_t) = P_t/(P_t + 1)，再加上狀態雜訊 q 得 P_{t+1} = P_t/(P_t + 1) + q。
f(P) = P/(P + 1) + q 在 [0, ∞) 上遞增、f'(P) = 1/(P + 1)² < 1，而且 f([0, ∞)) ⊂ [q, q + 1)，是 [q, q + 1] 上的壓縮映射，所以收斂到唯一不動點。
不動點：P = P/(P + 1) + q ⟺ P(P + 1) = P + q(P + 1) ⟺ P² − qP − q = 0，正根 P* = (q + √(q² + 4q))/2。
K* = P*/(P* + 1)；由 P*² = q(P* + 1) 得 P*/(P* + 1) = q/P* = 2q/(q + √(q² + 4q)) = (√(q² + 4q) − q)/2（分子分母同乘 √(q² + 4q) − q）。
預測式 μ̂_{t+1|t} = μ̂_{t|t−1} + K*(y_t − μ̂_{t|t−1})，反覆代入得到 EWMA 的形式。∎

資料：MLE 讓 47–73% 的股票 q 落在 10⁻⁶（格點下限），K 約 0.001，半衰期 500–700 天。所以最好的「去噪」就是很長的平均，B 組已經有了。

### 4.7 因果的 à trous Haar wavelet（定理 8）

直觀：每一層把序列和它往前 2^{j−1} 天的值平均，差就是這一層的細節。全部是「往回看」，所以去噪後 t 的值只用到 t 以前的資料；
很多論文用的是置中的（雙邊）wavelet 或整段做轉換，t 的值會用到 t 之後的資料，回測自然很好看（v13 定理 6、Algorithmic Finance 2019）。

**定理 8**：c_0 = x，c_j(t) = (c_{j−1}(t) + c_{j−1}(t − 2^{j−1}))/2，w_j = c_{j−1} − c_j。則
(i) x(t) = c_J(t) + Σ_{j=1}^{J} w_j(t)；
(ii) c_j(t) = 2^{−j}Σ_{i=0}^{2^j − 1} x(t − i)，所以 c_J(t) 和每個 w_j(t) 都只用到 x(t − 2^J + 1), …, x(t)。

證明：(i) Σ_j w_j = Σ_j (c_{j−1} − c_j) = c_0 − c_J（伸縮和）。
(ii) 對 j 歸納：j = 1 時 c_1(t) = (x(t) + x(t − 1))/2。假設對 j − 1 成立，c_j(t) = ½[2^{−(j−1)}Σ_{i=0}^{2^{j−1}−1} x(t − i) + 2^{−(j−1)}Σ_{i=0}^{2^{j−1}−1} x(t − 2^{j−1} − i)]
= 2^{−j}Σ_{i=0}^{2^j − 1} x(t − i)。w_j = c_{j−1} − c_j 也只用到這些值。∎
soft threshold 是對每個 w_j(t) 逐點做，門檻用 t 以前 250 天的 MAD，所以 D2 是因果的；`v15_signals.py` 的截斷檢查也確認了（差 0）。

### 4.8 Vasicek beta 收縮是後驗平均（定理 9）

直觀：120 天估出來的 beta 有誤差，誤差大的（標準誤大）應該往大家的平均拉得多一點。
這樣做的平均平方誤差一定比原本的估計小。

**定理 9**：先驗 β ~ N(β̄, s²)，觀察 β̂ | β ~ N(β, se²)。則 E[β | β̂] = (s²β̂ + se²β̄)/(s² + se²)；
它在所有 β̂ 的函數裡平均平方誤差最小，誤差是 s²se²/(s² + se²) < se²（原本 β̂ 的誤差）。

證明：常態先驗配常態觀察，後驗精確度 = 1/s² + 1/se²，後驗平均 = (β̄/s² + β̂/se²)/(1/s² + 1/se²) = (se²β̄ + s²β̂)/(s² + se²)。
對任何函數 f：E[(β − f(β̂))²] = E[(β − E[β | β̂])²] + E[(E[β | β̂] − f(β̂))²]（交叉項的條件期望為 0），所以後驗平均最小。
它的誤差 = 後驗變異 = 1/(1/s² + 1/se²) = s²se²/(s² + se²) < se²。∎
s² 用「橫斷面變異 − 平均 se²」估：β̂ 的橫斷面變異 = 真 beta 的變異 + 估計誤差的變異（Lally 1998 指出直接用 β̂ 的橫斷面變異會高估 s²）。

## 5. Random Forest 在這個比賽會怎樣（結果）

| 成員 | IC val | hold | post | 全期 | 和 lgb_res 的排序相關 |
|---|---|---|---|---|---|
| lgb_res（存好的） | 0.1108 | 0.0996 | 0.0806 | 0.1050 | — |
| A1 Random Forest（葉子 250 或 1000） | 0.1082 | 0.1034 | 0.0970 | 0.1058 | 0.903 |
| A2 ExtraTrees（葉子 250） | 0.0982 | 0.0991 | 0.1046 | 0.0990 | 0.778 |
| A0 sklearn 預設的 RF（葉子 1） | 0.0773 | 0.0658 | 0.0289 | 0.0698 | 0.489 |

- 題目本身很適合 RF：Krauss, Do & Huck (2017) 的題目幾乎一樣，RF 是單一模型裡最好的。在這裡，葉子夠大的 RF 和 lgb_res 的 IC 幾乎相同（差 +0.0009，t 0.34）。
- **葉子大小是關鍵**：A1 在 6 個 fold 裡 4 個選 250、2 個選 1000（三個值的 log-loss 只差 0.001–0.004），ExtraTrees 全部選 250。
  預設的葉子 1 讓 IC 掉 0.036（t −7.8），post 只剩 0.029。這裡每個 label 大約是 49 比 51 的硬幣，一片葉子要幾百個樣本才有訊號（定理 6）。
- **和 lgb_res 太像**：每天排序相關 0.90，用的是同樣的 67 個 feature，資訊一樣。定理 2 的診斷：最好的混合 IC 只從 0.1140 到 0.1143（val）。
- **ExtraTrees 比 RF 平滑**：IC 低 0.0068（t −3.7），但在 post 是所有成員裡最高的（0.1046）。這比較像 post 這段的 regime 偏好平滑的模型，不能拿來挑。
- **放進 v11 的各種放法都是「post 好、hold 差」**：加進 LightGBM 部分 post +2.67、hold −0.65；取代 lgb_res post +4.01、hold −0.65；當新 family post +2.99、hold −1.63。原因和 v13 相同：post 期間 lgb_res 和選股 MLP 變弱，任何不一樣的成員都會拉高 post。
- **feature importance（MDI，6 個 fold 平均）**：A1 前幾名是 ATR_pct、gap_bias_20、intra_bias_250、beta_us、intra_bias_exp，也就是股票固定的盤中習性和波動，和 lgb_res 一致。
  ExtraTrees 的前兩名是 is_fin、group，很可能是 MDI 的偏誤：隨機切點對 0/1 變數總是切在對的地方，連續變數則常切在差的位置，所以低基數的 feature 比較常被選到。
- **要用 RF 的話**：葉子至少幾百個樣本、max_features sqrt、每棵抽一半 row。預期和 LightGBM 差不多好，但不會更好。

## 6. 結果（`results\v15_eval.txt`，11/7 23:33）

- PART 0：重組的 v9、v11、W0 和公布的數字逐位相同；E1 的原始 beta 和 hedge_beta 完全相同。
- 對照的重現：
  - B0（你的電腦）和存好的 lgb_res：val 期間的決策完全相同；hold、post 的 fold 有極小的差異（機率最大差 0.012、排序相關 0.9998）。光是這個差異就讓 v11 的 hold 差 −0.15（t −1.61）。這就是 B 組要和同機器重訓的 B0 比、不能直接和存好的 lgb_res 比的原因（雲端重訓時 2022 fold 最大差只有 5 × 10⁻⁸，是 LightGBM 在不同機器上的數值差異）。
  - C0（seed 1–5）和存好的 mlp_res_xr9（seed 1–10）排序相關 0.994。
- 成員自己的 IC（全期）：B0 0.1051、B1 0.1049、B2 0.1043、B3 0.1017、B4 0.0959、B5 0.1035、B6 0.0976；C0 0.1041、C1 0.1059、C2 0.1061；A1 0.1058、A2 0.0990、A0 0.0698；D1 0.0800、D2 0.0495。
- W0 版本（各自和 W0 基準比，全期）：A1 −0.08、A2 −0.24、B1 −0.03、B2 −0.01、B3 −0.46、B4 +0.15、B5 −0.19、B6 +0.20、C1 +0.12、C2 +0.05、D1 −0.71、D2 −1.24、E1 +0.29（t 1.89）；v15 −0.70。
- 逐年：多數候選在 2022–2023 是正的、2025–2026 上半年是負的，post 又轉正（A1、A2、B3、B4），和 v13、v14 相同的「早期一點點正、最近亂跳」。只有 E1 在 6 段裡 5 段為正（2022 +0.23、2023 +0.19、2024 +0.49、2025 +0.01、2026 上半年 +0.46、post −0.23）。
- 10/1–10/22（15 天，只報告）：D1 +12.98（t 2.29）、D2 +10.60、B3 +6.46、B4 +6.30，都是雜訊的大小；D1 在 val、hold 都是負的，D2 的 hold、post 是負的。
- 規則的檢定力（真實效果 +0.5 bp/日時通過的機率）：E1 0.90、C1 0.78、C2 0.80、B1 0.70、B2 0.63、A1 0.56、B6 0.50、A2 0.48、B3 0.45、B4 0.40、D1 0.33、D2 0.29。
- 訓練時間（你的筆電，2 個 worker）：85.9 分鐘；LightGBM 每個 unit 約 0.3–0.9 分鐘、DoubleEnsemble 約 2 分鐘、RF 1–7 分鐘、ExtraTrees 1–5 分鐘、MLP 約 0.2 分鐘。

## 7. 檔案

| 檔案 | 內容 |
|---|---|
| `run_v15.bat` | 雙擊：小測試 → 訓練 → 訊號 → 判定 |
| `research\PREREG_v15.md` | 事先寫好的規則（11/7 21:10） |
| `research\v15lib.py` | 共用：路徑、panel、feature、unit 檔、成員表；判定用的 Data（= v14lib.Data，複製過來） |
| `research\v15_train.py` | A、B、C 組的訓練（可中斷續跑；Windows 用 spawn 的 worker） |
| `research\v15_signals.py` | D、E 組（Kalman、wavelet、Vasicek）和截斷檢查 → `cache\signals15.pkl` |
| `research\v15_eval.py` | 判定和報告 → `results\v15_eval.txt`、`v15_daily.csv`、`v15_selection.json` |
| `results\v15_signals.txt` | D、E 組的訊號報告 |
| `cache\units\` | 訓練好的 walk-forward 預測（每個 unit 一個 npz） |
| `cache\_stage\` | 傳檔到雲端用的暫存（core9 的分割檔約 173 MB、雲端小測試的輸出），**可以刪** |

整個 `v15\` 只有研究，可以整個刪掉；每天的流程不讀它。

## 參考文獻

- Aitken, A. C. (1935). On least squares and linear combination of observations. *Proceedings of the Royal Society of Edinburgh*, 55, 42–48.
- Bieganowski, B., & Ślepaczuk, R. (2024). Supervised autoencoder MLP for financial time series forecasting. arXiv:2404.01866.
- Blume, M. E. (1975). Betas and their regression tendencies. *Journal of Finance*, 30(3), 785–795.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. *Machine Learning*, 63(1), 3–42.
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223–2273.
- Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Heston, S. L., Korajczyk, R. A., & Sadka, R. (2010). Intraday patterns in the cross-section of stock returns. *Journal of Finance*, 65(4), 1369–1407.
- Huber, P. J. (1964). Robust estimation of a location parameter. *Annals of Mathematical Statistics*, 35(1), 73–101.
- Johnstone, I. M., & Silverman, B. W. (1997). Wavelet threshold estimators for data with correlated noise. *Journal of the Royal Statistical Society B*, 59(2), 319–351.
- Krauss, C., Do, X. A., & Huck, N. (2017). Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500. *European Journal of Operational Research*, 259(2), 689–702.
- Lally, M. (1998). An examination of Blume and Vasicek betas. *Financial Review*, 33(3), 183–197.
- Levi, Y., & Welch, I. (2017). Best practice for cost-of-capital estimates. *Journal of Financial and Quantitative Analysis*, 52(2).
- Lim, B., Zohren, S., & Roberts, S. (2019). Enhancing time series momentum strategies using deep neural networks. *Journal of Financial Data Science*, 1(4), 19–38.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Ma, Y., Ventre, C., & Polukarov, M. (2021). Denoised labels for financial time-series data via self-supervised learning. arXiv:2112.10139.
- Mentch, L., & Zhou, S. (2020). Randomization as regularization: A degrees of freedom explanation for random forest success. *Journal of Machine Learning Research*, 21(171), 1–36.
- Muth, J. F. (1960). Optimal properties of exponentially weighted forecasts. *Journal of the American Statistical Association*, 55(290), 299–306.
- Renaud, O., Starck, J.-L., & Murtagh, F. (2005). Wavelet-based combined signal filtering and prediction. *IEEE Transactions on Systems, Man, and Cybernetics B*, 35(6), 1241–1251.
- Vasicek, O. A. (1973). A note on using cross-sectional information in Bayesian estimation of security betas. *Journal of Finance*, 28(5), 1233–1239.
- Zeng, L., Wang, L., Niu, H., Zhang, R., Wang, L., & Li, J. (2024). Trade when opportunity comes: Price movement forecasting via locality-aware attention and iterative refinement labeling. *IJCAI 2024*, 6134–6142.
- Zhang, C., Li, Y., Chen, X., Jin, Y., Tang, P., & Li, J. (2020). DoubleEnsemble: A new ensemble method based on sample reweighting and feature selection for financial data analysis. *ICDM 2020*.
- Information leakage in financial machine learning research (2019). *Algorithmic Finance*, 8(1–2). https://journals.sagepub.com/doi/10.3233/AF-190900
- Jane Street Market Prediction 第一名的說明（Yirun Zhang，Kaggle discussion 224348）；Numerai 論壇的重現：https://forum.numer.ai/t/autoencoder-and-multitask-mlp-on-new-dataset-from-kaggle-jane-street/4338
- Qlib benchmarks（DoubleEnsemble 和 LightGBM 在 Alpha158 的 IC）：https://github.com/microsoft/qlib/tree/main/examples/benchmarks

# v14：還有哪些量化方法能用（2026-11-07，只有研究，沒改任何每天的程式）

使用者的問題（11/7 01:18）：「還有甚麼量化交易 shit 是我能參考或嘗試做的嗎，你可以直接做沒關係，但都建在 V14 之類的，盡量都讓我本地跑實驗」。

所以 v14 有三件事：

1. **對照表**（第 1 節）：常見的量化方法，在這個比賽能不能用、v1–v13 做過沒有、在哪裡。這是「能參考的」。
2. **五組沒做過的實驗**（E1–E5，第 2–4 節）：規則在任何成績出來之前寫好（`research\PREREG_v14.md`）。
   不用重訓任何模型：用 v8–v13 存好的 walk-forward 預測重組 v11、W0，再把新的訊號放進去。只要 numpy、pandas，雙擊 `run_v14.bat` 約 2 分鐘。
3. **沙盒** `research\v14_try.py`（第 5.2 節）：自己寫一個訊號函式，約 15 秒就得到 leak check、IC、和現在模型的相關、加進 v11 / W0 的分數。

val 2022–2024、hold 2025-01–2026-05、post 2026-06–10-22。分數是 Top50 的 bp/日，括號是 Newey-West t。

- 第 0 節：結論。
- 第 1 節：量化方法對照表；為什麼 Kelly、risk parity 這類部位大小的工具在這裡沒用（定理 1）。
- 第 2 節：v14 測的五組方法。
- 第 3 節：理論（直觀 + 證明，定理 2–7）。
- 第 4 節：結果（`results\v14_eval.txt`）。
- 第 5 節：在筆電上跑、沙盒。
- 第 6 節：檔案。
- 第 7 節：接下來值得做的。

## 0. 結論

**照事先寫好的規則，沒有一個通過；v9、v11、W0 都不變。** 入選的 E4a（季底作帳）和 E5（14 個模型 1/N）合起來 hold −0.58、post +3.07，hold ≤ 0。

| 候選 | 方法 | v11 + 候選 − v11：val | hold | post | 全期 | 判定 |
|---|---|---|---|---|---|---|
| E2a | PCA 統計套利 s-score（Avellaneda & Lee 2010） | −1.16（−1.66） | −1.67（−1.76） | −2.03（−0.84） | −1.38（−2.50） | 不入選 |
| E2b | 殘差 5 日反轉（Blitz et al. 2013） | −1.01（−1.43） | −2.99（−3.09） | −0.11（−0.05） | −1.51（−2.68） | 不入選 |
| E3a | 產業中性 | −1.47（−1.45） | −3.66（−2.57） | −3.90（−0.87） | −2.31（−2.71） | 不入選 |
| E3b | 統計因子中性（5 個 PCA beta） | −3.60（−2.99） | −3.11（−1.84） | −6.80（−1.54） | −3.73（−3.82） | 不入選 |
| E4a | 季底作帳（每季最後 20 天加投信累積買超） | +0.26（0.95） | −0.83（−1.74） | −3.20（−1.83） | −0.35（−1.29） | 入選 |
| E4b | 季初反轉（每季前 3 天反過來） | −0.05（−0.31） | −0.22（−1.00） | +0.69（1.08） | −0.04（−0.28） | 不入選 |
| E5 | 14 個存好的選股模型 1/N | +0.29（0.65） | −0.38（−0.47） | +2.80（1.26） | +0.31（0.74） | 入選 |
| E4a + E5 | 入選的合起來 | +0.35（0.76） | −0.58（−0.72） | +3.07（1.32） | +0.31（0.74） | **沒通過** |

五個發現：

- **就算 09:00 前就知道開盤價，也只多約 1 bp/日（E1，只報告）。** 開盤跳空是這次最強的新資訊：反轉的 rank IC 0.079（val、hold、post 都在 0.06–0.09），
  和 v11 選股觀點的相關只有 0.154。但加進 v11 全期 +1.08（t 1.88）、W0 只 +0.24（t 0.38）；開盤價有 0.1–0.5% 的誤差也差不多。
  定理 7 給的上限約 +2.2 bp/日。要拿到它，live 得在 08:59 抓證交所盤前試撮的模擬成交價、09:00 前上傳完，static 則根本不能用。**不值得做。**
  （phase1_4 檢查時看到的「9 月光看開盤跳空 43.78 bp」只有 20 天：同一類規則全期 7.24、2026-06 以後 28.18、2022–2024 只有 4.58。）
- **PCA 統計套利、殘差反轉的資訊，模型裡大多已經有了。** s-score 自己的 IC 0.0074、殘差 5 日反轉 0.0170。
  定理 2 的診斷：就算用最好的權重，選股觀點的 IC 也只從 0.1140 到 0.1141–0.1144（val）。v14 的放法給新訊號 1/3 的權重，
  但會變好的權重範圍只有 0 到 0.04（s-score）、0 到 0.15（殘差反轉），所以放進去反而 −1.4、−1.5（t −2.5、−2.7）。
- **中性化傷害最大，這本身是有用的資訊。** 選股觀點在 Top50 裡的差異，平均 40.5% 在 4 個產業 group 之間；把這部分拿掉，全期 −2.31（產業）、−3.73（PCA 因子）。
  照定理 3，這代表**選股觀點的產業／因子押注本身是賺錢的**（每天約 +2.3 到 +3.7 bp），不要做 sector-neutral。
- **季底作帳沒有出現。** 投信 60 天累積買超的 IC：季底 20 天 0.0334、其他日子 0.0250；季底減非季底 +0.0085（t 0.97）。
  val 剛好 > 0 入選，但 hold −0.83、post −3.20；把每天都加進去（安慰劑）全期 −0.18。私有測試期（11/30–12/24）剛好在 Q4 最後一個月，但沒有證據可以用。
- **再多的模型也合不出更高的 IC：瓶頸是資訊，這一點又確認一次。** 14 個存好的選股模型（MLP、LightGBM、SVM、LSTM、CNN、F-LOAM、v12 的 6 個 LightGBM、X2、X5），
  自己的 IC 在 0.085–0.114。1/N 的 IC 全期 0.1099，v11 自己是 0.1105。在 val 上找最好的權重，val 本身有 0.1188，
  拿到 hold 只剩 0.1046，比 v11 的 0.1088 還低（定理 4 (ii)：樣本內最佳權重平均一定高估）。

所以：

- **v1–v14 下來，適用這個比賽結構的常見量化方法幾乎都測過了**（第 1 節），剩下的都在 1 bp/日以內；19 天平均的運氣是 9–11 bp（overfit 檢查）。
- **11/23 前真正要決定的仍是 W0 還是 v11（風險設定）**，這需要排行榜上各隊每天的分數（第 7 節）；`leaderboard\` 目前只有 10/22、10/27 兩天的截圖，`daily_board.csv` 是空的。
- 不要追 10 月：E5 在 10/1–10/22 的 15 天 +10.08（t 1.23），E4b +4.58，都是雜訊的大小。

## 1. 量化方法對照表

### 1.1 每一種方法在這裡的狀態

| 方法 | 代表文獻 | 在這個比賽 | v1–v13 做過了嗎 | v14 |
|---|---|---|---|---|
| 短期反轉（1 天、1 週） | Jegadeesh (1990)；Lehmann (1990)；Lo & MacKinlay (1990) | 可以 | B 組 ret1、ret5、intraday_ret、rel_intra_prev | 沙盒範例 rev1 |
| PCA 統計套利、殘差反轉 | Avellaneda & Lee (2010)；Blitz, Huij, Lansdorp & Verbeek (2013)；Da, Liu & Schaumburg (2014) | 可以 | 沒有 | E2：變差 |
| 配對交易、cointegration | Gatev, Goetzmann & Rouwenhorst (2006) | 可以 | 沒有 | E2 是它的多股票版本 |
| 隔夜／盤中拉鋸、盤中習性 | Lou, Polk & Skouras (2019)；Bogousslavsky (2021)；Akbas, Boehmer, Jiang & Koch (2022) | 可以 | B 組 intra_bias_20/60/250/exp、gap_bias_20；v3 gap_bias_60/250 | — |
| 開盤價、盤前試撮 | 證交所開盤前資訊揭露 | 只有 live 能用，要 08:59 抓 | 沒有 | E1：上限約 +1 bp |
| 盤中動能（前半小時預測最後半小時） | Gao, Han, Li & Zhou (2018) | 不能：09:00 前就要交 | — | — |
| 季節性（星期幾、月份） | Keloharju, Linnainmaa & Nyberg (2016)；Heston & Sadka (2008) | 可以 | v3 dow_bias_e；core9 cal_dow0、cal_dow4 | — |
| 日曆（月初、結算日、長假前後、MSCI 調整日） | Lakonishok & Smidt (1988) | 可以 | v11 T3；core9 cal_*；v8 MSCI 檢查（27 次平均 −36 bp，t −1.7） | — |
| 季底作帳 | Carhart, Kaniel, Musto & Reed (2002) | 可以 | 沒有 | E4：沒通過 |
| 法人籌碼 | — | 可以 | B 組（外資、投信、自營商、外資持股） | E4 用投信 |
| 融資融券、營收、本益比、費半、匯率 | — | 可以 | 棄案 `best_features` 的 F 組（沒有比 B 組好） | — |
| 產業 lead-lag、供應鏈 | Moskowitz & Grinblatt (1999)；Hou (2007)；Cohen & Frazzini (2008) | 可以 | B 組 sector_intra_prev、beta_x_us | — |
| 夜盤、美股、韓股 | — | 可以 | v7、v11 T2、B 組 | — |
| 低 beta、beta 對沖 | Frazzini & Pedersen (2014) | 可以 | v8、v9 的 λ | — |
| 風險模型中性化 | Grinold & Kahn (2000) | 可以 | 只有 beta 對沖 | E3：明顯變差 |
| 公式化 alpha（Alpha158、101 alphas） | Yang et al. (2020)（Qlib）；Kakushadze (2016) | 可以 | Alpha158（v10、v11，沒有群組通過） | 101 沒做：一樣只用價量，可以用沙盒自己試 |
| 多模型合併、1/N | DeMiguel, Garlappi & Uppal (2009)；Grinold (1989) | 可以 | 每個候選都單獨加過 | E5：沒通過 |
| 自適應係數、regime | Hamilton (1989) | 可以 | v9 第十一節、v11d、overfit | — |
| 檢驗：deflated Sharpe、PBO、reality check | Bailey & López de Prado (2014)；Bailey, Borwein, López de Prado & Zhu (2017)；White (2000) | 用來檢驗 | overfit | 定理 6 |
| Purged / embargoed CV | López de Prado (2018) | 不需要：label 只有 1 天，walk-forward 本來就沒有重疊 | — | — |
| Kelly、risk parity、HRP、波動度調整、meta-labeling 的部位大小 | Kelly (1956)；Moreira & Muir (2017)；López de Prado (2018) | 不適用（定理 1） | — | — |
| 交易成本、turnover 最佳化 | Gârleanu & Pedersen (2013) | 不適用：分數沒有成本 | — | — |
| Triple-barrier label | López de Prado (2018) | 不適用：持有期固定（開盤到收盤） | — | — |
| 比賽策略（落後時加大變異數） | Brown, Harlow & Starks (1996) | 適用：11/23 選 W0 或 v11 | v7 第三節的計畫 | 第 7 節 |

### 1.2 為什麼部位大小的工具在這裡沒用

直觀：每支股票只能猜 1 或 0（等於做多或做空同樣大小），分數是報酬的線性平均，沒有成本。
這種情況下每支股票的最佳決策只看「期望報酬的正負號」，跟要押多大無關；Kelly、risk parity、HRP、波動度調整都是在決定大小，這裡沒有大小可以調。
meta-labeling 不能「不下注」（每格都要 0 或 1），所以它最多只是另一個預測正負號的模型。

**定理 1**：AR_t = (1/50) Σ_i s_i r_i，s_i ∈ {−1, +1} 在 09:00 前用資訊 F 決定。讓 E[AR_t | F] 最大的決策是 s_i* = sign(E[r_i | F])（等於 0 時隨便）。

證明：s_i 由 F 決定，所以 E[AR_t | F] = (1/50) Σ_i s_i μ_i，μ_i = E[r_i | F]。每一項可以分開最大化：s_i μ_i ≤ |μ_i|，等號若且唯若 s_i = sign(μ_i)。∎

所以任何方法只能透過「改變哪些股票的正負號」影響平均分數。beta 對沖、balanced 這類做法是用一點平均換變異數：
它們翻的是 |μ_i| 最小、離門檻最近的那幾支（v8、v11 第 10 節）。在比賽裡目標是「最後拿第一」而不是平均最大，變異數才有意義（第 7 節）。

## 2. v14 測的五組方法（設定在 `research\PREREG_v14.md`，01:40 寫好）

| 代號 | 方法 | 設定 | 為什麼值得測 |
|---|---|---|---|
| E1 | 開盤價（只報告） | 跳空 gap = 開_D / 收_{D−1} − 1，用第 D 天真實的開盤價；選股（−gap）、大盤（夜盤反轉改用真實的平均跳空）、加上誤差 | 唯一真正「新」的資訊；問的是上限 |
| E2a | PCA 統計套利 s-score | 100 支收盤對收盤報酬，252 天相關矩陣取前 5 個 eigenportfolio；60 天 OLS 殘差累加成 X；X 做 AR(1) 得到 OU 參數；κ > 252/30 才算；s = −(m − 平均 m) / σ_eq；訊號 −s | Avellaneda & Lee (2010) 的原始做法 |
| E2b | 殘差 5 日反轉 | 同樣的殘差；−（最近 5 天殘差和）/ 殘差標準差 | 殘差反轉比原始報酬反轉穩（Blitz et al. 2013） |
| E3a | 產業中性 | 當天 Top50 的 p_e 在 4 個 group 裡各自減平均 | 標準的 risk-model 做法，這裡只有 beta 對沖 |
| E3b | 統計因子中性 | 當天 Top50 的 p_e 對 [1、5 個 PCA beta] 取殘差 | 同上 |
| E4a | 季底作帳 | 每季最後 20 個交易日：投信最近 60 天買賣超股數的和 / 60 天平均成交股數 | Carhart et al. (2002)：基金在季底最後幾天拉抬持股、收盤特別明顯；台股俗稱「投信作帳」；私有測試期就在 Q4 最後一個月 |
| E4b | 季初反轉 | 每季前 3 個交易日：反過來 | 同一篇：季底的拉抬隔天會回吐 |
| E5 | 14 個模型 1/N | 選股 MLP、LightGBM 部分、SVM、LSTM-R、CNN-R、F-LOAM、v12 的 6 個 LightGBM、X2、X5，各自當天標準化後平均 | DeMiguel et al. (2009)：估計誤差大時 1/N 常常贏最佳化的權重；這些模型都已經存好 |

共同的放法（PREREG 第 1 節，spread-matched new family）：訊號 z 當天標準化，m = sd(p_e)·z，p_raw = (2·p_e + m) / 3，
再縮放到和 p_e 一樣的當天標準差。和 v13 不同的地方：v13 的放法在新訊號和現在的觀點相關低時，會讓 p_e 的標準差變小（相關 0.15 時約 0.8 倍），
等於偷偷把 w_T 從 0.5 加到約 0.62。固定標準差之後，只有選股的排序改變。v13 的放法也算了（第 4.3 節），結論相同。

判定和 v11b、v12、v13 相同：單獨 val > 0 入選；入選的合起來 hold、post 都 > 0 才通過。

## 3. 理論（直觀 + 證明）

### 3.1 加一個訊號：什麼時候會變好、權重可以給多少（定理 2）

直觀：新訊號 z 和現在的選股觀點 x 相關 ρ 的那一部分，x 已經有了；z 真正的貢獻是剩下的部分。
就算剩下的部分有用，給太多權重也會把 x 稀釋掉，所以「有幫助的權重」有一個上限 w₀，而且訊號越弱、w₀ 越小。

設定：同一天內 x、z、y（超額報酬）都標準化；IC_x = corr(x, y) = a > 0、IC_z = corr(z, y) = b ≥ 0（z 的方向取成自己的 IC 不是負的）、
corr(x, z) = ρ，|ρ| < 1。混合 s_w = (1 − w)x + wz，0 ≤ w ≤ 1。記 D = 2a²(1 − ρ) − (b − a)²。

**定理 2**：
(i) d corr(s_w, y) / dw 在 w = 0 等於 b − ρa（v13 定理 2 (i)）。
(ii) 所有線性組合裡最高的 IC 是 √((a² + b² − 2ρab) / (1 − ρ²))，z 的最佳權重 w* = (b − ρa) / ((a − ρb) + (b − ρa))（v13 定理 2 (ii)）。
(iii) 若 b > ρa 且 D > 0：對 0 < w ≤ 1，corr(s_w, y) ≥ a 若且唯若 w ≤ w₀ = 2a(b − ρa) / D。
  若 b > ρa 且 D ≤ 0：每個 w ∈ (0, 1] 都讓 IC 變高。若 b ≤ ρa：沒有任何 w ∈ (0, 1] 會讓 IC 變高。

證明：(i)(ii) 見 README_v13 第 3.2 節。(iii)：C(w) = Cov(s_w, y) = (1 − w)a + wb，V(w) = Var(s_w) = 1 − 2w(1 − ρ) + 2w²(1 − ρ)。
因為 a > 0、b ≥ 0，C(w) > 0，所以 corr(s_w, y) = C / √V ≥ a 等價於 q(w) := C² − a²V ≥ 0。展開：

q(w) = 2aw(b − a) + w²(b − a)² + 2a²w(1 − ρ) − 2a²w²(1 − ρ) = w[2a(b − ρa) − wD]。

- b > ρa、D > 0：w > 0 時 q ≥ 0 等價於 w ≤ 2a(b − ρa) / D = w₀。
- b > ρa、D ≤ 0：中括號裡兩項都 ≥ 0、第一項 > 0，所以 q > 0。
- b ≤ ρa、D ≥ 0：中括號裡兩項都 ≤ 0，q ≤ 0。
- b ≤ ρa、D < 0：q 是開口向上的二次式，q(0) = 0，q(1) = C(1)² − a²V(1) = b² − a² < 0（0 ≤ b ≤ ρa < a），所以在 [0, 1] 上 q ≤ max(q(0), q(1)) = 0。∎

另外 w₀ 換成「x + t·z」的寫法就是 t₀ = 2a(b − ρa) / (a² − b²)、w₀ = t₀ / (1 + t₀)，兩種算法相同。

怎麼用（`results\v14_eval.txt` PART 1，全期，每天的排序相關平均，近似）：

| 訊號 | 自己的 IC（全期） | ρ（和 v11 選股觀點，全期） | IC_z − ρ·IC_x（全期） | w*（val） | w₀（val） | 最好的混合（val，現在 0.1140） |
|---|---|---|---|---|---|---|
| E1 −gap（第 D 天的開盤） | 0.0788 | 0.154 | +0.0617 | 0.41 | 0.75 | 0.1343 |
| E2a s-score | 0.0074 | 0.074 | −0.0008 | 0.02 | 0.04 | 0.1141 |
| E2b 殘差 5 日反轉 | 0.0170 | 0.155 | −0.0001 | 0.08 | 0.15 | 0.1144 |
| E4 投信 60 天（每天） | 0.0277 | 0.180 | +0.0079 | 0.08 | 0.15 | 0.1145 |

v14 給新 family 的權重是 w = 1/3：只有 E1 在 w₀ 以內，E2、E4 都遠超過，所以放進去只會變差；就算用最好的權重，也只多 0.0001–0.0005 的 IC。

### 3.2 中性化等於拿掉產業／因子那一部分的報酬（定理 3）

直觀：選股分數可以拆成「押產業（因子）」和「同產業裡挑股票」兩部分。中性化把第一部分丟掉。
如果押產業平均是賺的，丟掉就會少賺；只有押產業平均是賠的，中性化才會變好。

設定：n 支股票，線性的分數向量 s（部位）、報酬 r、曝險矩陣 B（n × k，含截距；產業 dummy 或 PCA beta），P = B(BᵀB)⁻¹Bᵀ、M = I − P。

**定理 3**：sᵀr = (Ps)ᵀr + (Ms)ᵀr，而且中性化後的部位 Ms 的期望報酬是 E[(Ms)ᵀr] = E[sᵀr] − E[sᵀPr]。
所以中性化讓平均變高，若且唯若 E[sᵀPr] < 0，也就是分數裡落在 B 的部分平均是賠的。另外 (Ps)ᵀr = (Ps)ᵀ(Pr)：產業那部分的分數只和產業平均報酬有關。

證明：P、M 都是對稱的投影矩陣，P + M = I、P² = P。所以 sᵀr = sᵀ(P + M)r = (Ps)ᵀr + (Ms)ᵀr，取期望得第一式。
(Ps)ᵀr = sᵀPr = sᵀP²r = (Ps)ᵀ(Pr)。∎

這裡的決策是分數的正負號，不是線性的，所以定理 3 對決策是近似。結果：產業中性全期 −2.31、因子中性 −3.73，而 v11 選股觀點在 Top50 裡的變異數平均 40.5% 在 4 個 group 之間。
代表 v11 對「今天哪個產業會比較好」的判斷本身就有約 +2 到 +4 bp/日的價值（這和 B 組有 sector_intra_prev、beta_x_us 這些產業層級的 feature 一致）。

### 3.3 很多個訊號的上限，以及樣本內最佳權重的高估（定理 4）

直觀：n 個訊號能合出的最高 IC，由它們各自的 IC 和彼此的相關決定；大家都高度相關時，再多也合不高。
而且用同一段資料找出來的「最佳權重」，在那一段一定看起來比實際好，換一段就會掉下來。

設定：標準化的訊號 u ∈ ℝⁿ，相關矩陣 Σ（正定），c = Cov(u, y)（各自的 IC）。

**定理 4**：
(i) max_β corr(βᵀu, y) = √(cᵀΣ⁻¹c)，在 β ∝ Σ⁻¹c 達到。
(ii) 如果 c、Σ 是從一段樣本估的，ÎC(β) 是那段樣本上權重 β 的 IC，那麼對任何固定的 β，E[ÎC(β)] ≤ E[max_β′ ÎC(β′)]；
  所以 max_β E[ÎC(β)] ≤ E[max_β ÎC(β)]：樣本內最佳權重的 IC，平均不低於任何一組事先固定的權重的真實期望（通常更高，就是高估）。

證明：(i) corr(βᵀu, y) = βᵀc / √(βᵀΣβ)。βᵀc = (Σ^{1/2}β)ᵀ(Σ^{−1/2}c) ≤ √(βᵀΣβ)·√(cᵀΣ⁻¹c)（Cauchy–Schwarz），等號在 Σ^{1/2}β ∝ Σ^{−1/2}c，即 β ∝ Σ⁻¹c。
(ii) 每一組樣本上都有 ÎC(β) ≤ max_β′ ÎC(β′)，兩邊取期望；左邊再對 β 取最大。∎

結果（PART 1）：

- 用每天排序相關平均的近似公式：val 上最好的單一 family 0.1144、1/N 0.1169、最佳線性組合 0.1212（樣本內）。
- 實際把訊號合起來算的 IC（val / hold / post）：val 的最佳權重 0.1188 / 0.1046 / 0.0991；1/N 0.1141 / 0.1052 / 0.0949；v11 自己 0.1140 / 0.1088 / 0.0900。
  val 的最佳權重在 val 本身最好，換到 hold 就比 v11 低，正是 (ii) 說的高估。
- 上限本身也很小：0.1212 − 0.1144 ≈ 0.007，照定理 7 約 +0.9 bp/日，而且是樣本內的。

### 3.4 OU 過程和 s-score（定理 5）

直觀：把股票相對因子的殘差累加起來，看成一條會被拉回平均的線（Ornstein–Uhlenbeck）。s-score 是「離平均有幾個標準差」，
離得越遠，接下來往回走的期望越大。

設定：dX = κ(m − X)dt + σ dW，每 Δ = 1/252 年取樣一次。

**定理 5**：X_{t+Δ} = a + bX_t + ζ，b = e^{−κΔ}、a = m(1 − b)、ζ ~ N(0, σ²(1 − b²) / (2κ)) 且和 X_t 獨立。
穩定狀態的變異數 σ_eq² = σ² / (2κ) = Var(ζ) / (1 − b²)；h 天後的期望變化 E[X_{t+hΔ} − X_t | X_t] = (m − X_t)(1 − b^h)。

證明：對 e^{κt}X_t 用 Itô：d(e^{κt}X_t) = κm e^{κt}dt + σe^{κt}dW。從 t 積到 t + Δ：
X_{t+Δ} = e^{−κΔ}X_t + m(1 − e^{−κΔ}) + σ∫_t^{t+Δ} e^{−κ(t+Δ−u)} dW_u。
最後一項和 F_t 獨立、平均 0，由 Itô isometry 變異數是 σ²∫_0^Δ e^{−2κv}dv = σ²(1 − e^{−2κΔ}) / (2κ)，得 b、a、Var(ζ)。
Var(ζ) / (1 − b²) = σ² / (2κ)，就是 Δ → ∞ 時 X 的變異數。重複 h 次：E[X_{t+hΔ} | X_t] = m + (X_t − m)b^h。∎

補充：回歸有截距，所以 60 天的殘差和是 0（截距那一 column 的 normal equation：1ᵀe = 0），視窗最後 X = 0，s = −m / σ_eq，和 Avellaneda & Lee 相同。
結果：s-score 對隔天盤中超額報酬的 IC 只有 0.0074，hold、post 是負的。有完整 312 天資料的 stock-day（2022 起 95.8%）裡，99.3% 都過了 κ > 252/30，
不是因為篩掉太多股票。

### 3.5 這個規則的假陽性（定理 6）

直觀：每個候選在 val 上都是擲一次硬幣，候選越多、總會有幾個「入選」；入選的合起來再擲兩次（hold、post）。

**定理 6**：沒有效果、每個候選 val 的差對 0 對稱且彼此獨立、三段的日子不重疊時，K 個候選裡「至少一個入選、入選的合起來 hold 和 post 都 > 0」的機率是 (1 − 2^{−K}) / 4。
K = 7 時是 0.248。如果各候選 val 的差是常態、彼此相關都 ≥ 0，至少一個入選的機率更小，所以 0.248 是上限。

證明：每個候選入選的機率是 1/2，彼此獨立，所以至少一個入選的機率是 1 − 2^{−K}。合起來的候選只用 val 選出來，
它在 hold、post 的差和 val 無關（日子不重疊、每天的差只有很弱的自我相關），而且沒有效果時對 0 對稱、兩段彼此獨立，所以兩段都 > 0 的機率是 1/4。相乘。
相關都 ≥ 0 的常態向量是 associated（Pitt 1982），所以遞減事件 {d_k ≤ 0} 正相關：P(全部沒入選) ≥ Π P(d_k ≤ 0) = 2^{−K}，
至少一個入選的機率 ≤ 1 − 2^{−K}。∎

所以這個規則仍然主要是擋掉變差的改法（和 v13 定理 3 相同）。這次 7 個候選有 2 個入選、合起來 hold 是負的，和「沒有效果」完全相容。

### 3.6 IC 換成分數：為什麼開盤價也只值約 1–2 bp（定理 7）

直觀：分數裡選股的部分約等於「IC × 當天股票之間報酬的差異」。Top50 盤中超額報酬的當天標準差約 1.70%，
所以 IC 每多 0.01，選股分數最多多約 1.4 bp/日。開盤價只能把 IC 從 0.1105 拉到 0.127。

**定理 7**：同一天內，分數 S 和超額報酬 X 是二元常態、相關 ρ、sd(X) = σ_x，那麼 E[sign(S)·X] = √(2/π)·ρ·σ_x。

證明：令 S₀ = 標準化的 S，可以寫 X / σ_x = ρS₀ + √(1 − ρ²)ε，ε ~ N(0, 1) 和 S₀ 獨立。
E[sign(S)X] = σ_x(ρE[sign(S₀)S₀] + √(1 − ρ²)E[sign(S₀)]E[ε]) = σ_x ρ E|S₀| = σ_x ρ √(2/π)，因為 E|N(0, 1)| = √(2/π)、E[ε] = 0。∎

數字：σ_x ≈ 1.70%（Top50 每天的超額報酬標準差平均）。v11 選股觀點 IC 0.1105 → 理論 15.0 bp，實際 xsec 12.25（beta 對沖、除權息規則、Top50 和 100 支的差拿走一部分）。
開盤價：定理 2 (ii) 的最好混合 0.1269，多 0.0164 → 最多 √(2/π) × 0.0164 × 170 ≈ 2.2 bp/日；實際 v11 +1.08（xsec +1.36）、W0 +0.24。

## 4. 結果（`results\v14_eval.txt`）

### 4.1 訊號自己的 rank IC（100 支、每天和超額報酬的 Spearman，平均）

| 訊號 | val | hold | post | 全期 |
|---|---|---|---|---|
| v11 選股觀點 p_e | 0.1140 | 0.1088 | 0.0900 | 0.1105 |
| E1 −gap（第 D 天的開盤，只是上限） | 0.0882 | 0.0599 | 0.0741 | 0.0788 |
| E2a s-score（k 5） | 0.0130 | −0.0016 | −0.0030 | 0.0074 |
| E2a k 1 / 3 / 10（全期） | | | | 0.0063 / 0.0069 / 0.0056 |
| E2b 殘差反轉 5 天 | 0.0260 | 0.0007 | 0.0063 | 0.0170 |
| E2b 1 / 10 / 20 天（全期） | | | | 0.0352 / 0.0089 / 0.0040 |
| E4 投信 60 天，每天 | 0.0359 | 0.0064 | 0.0404 | 0.0277 |
| E4 投信 60 天，季底 20 天 | 0.0472 | 0.0076 | 0.0156 | 0.0334 |
| E4 投信 60 天，季初 3 天 | 0.0592 | −0.0172 | −0.0568 | 0.0247 |
| E4 投信 60 天，其他日子 | 0.0280 | 0.0078 | 0.0700 | 0.0250 |

季底 20 天減其他日子 +0.0085（t 0.97，380 對 784 天）；季初 3 天減其他日子 −0.0032（t −0.16）。

E5 的 14 個 family（全期 IC；和 v11 選股觀點的相關）：選股 MLP 0.1071（0.965）、LightGBM 部分 0.1081（0.950）、SVM 0.1093（0.885）、
LSTM-R 0.0936（0.736）、CNN-R 0.0847（0.713）、F-LOAM 0.1054（0.862）、lgb_sign 0.1033、lgb_dec_l2 0.1086、lgb_dec_pw 0.1063、lgb_size_l2 0.1015、
lgb_size_pw 0.1017、lgb_size_rank 0.0879、X2 0.1015、X5 0.0998（v12、v11 的 8 個和 v11 的相關 0.78–0.92）。

### 4.2 判定（PART 2）

見第 0 節的表。入選 E4a、E5；合起來 val +0.35、hold −0.58、post +3.07 → 沒通過。

### 4.3 只報告

W0（現在上傳的）＋ 候選 − W0，全期：E2a −2.04（−3.20）、E2b −1.41（−2.30）、E3a −2.52（−3.02）、E3b −4.47（−4.08）、
E4a −0.41（−1.52）、E4b +0.02（0.17）、E5 −0.01（−0.02）、入選的合起來 −0.22（−0.42）。方向和 v11 相同，沒有一個對 W0 比較好。

v13 的放法（候選縮放到 lgb_res 再平均），v11，全期：E2a −0.97、E2b −1.10、E4a −0.17、E4b −0.02。結論相同。

變形（v11，全期）：

| 變形 | 全期（t） | 說明 |
|---|---|---|
| E2a k 1（只有大盤因子） | −2.03（−3.47） | |
| E2a k 3 / k 10 | −1.51（−2.66）/ −1.02（−1.77） | 因子越多、殘差越小，傷害越小，但都是負的 |
| E2b 1 天 / 10 天 / 20 天 | −0.56（−0.99）/ −1.37（−2.47）/ −1.38（−2.54） | 1 天的 IC 最高（0.035），但 B 組本來就有 1 天反轉 |
| E4a 最後 5 / 10 / 40 天 | −0.16 / −0.10 / −0.56 | 視窗越短越接近 0，沒有一個 > 0 |
| E4 每天都加（安慰劑） | −0.18（−0.38） | |
| E3a + E3b 一起 | −4.14（−3.92） | |
| E5，val 的最佳權重（不是 1/N） | −0.37（−0.72）：val −0.68、hold −1.52、post +5.75 | 定理 4 (ii) |

E1（第 D 天的開盤價，只報告）：

| 做法 | val | hold | post | 全期 |
|---|---|---|---|---|
| 不用模型：完全知道答案（上限） | 118.61 | 135.06 | 194.33 | 129.83 |
| 不用模型：每支猜自己跳空的反方向 | 4.58 | 6.83 | 28.18 | 7.24 |
| 不用模型：猜相對跳空（減當天平均）的反方向 | 8.65 | 4.52 | 20.36 | 8.45 |
| 不用模型：全部猜大盤平均跳空的反方向 | 1.28 | 1.48 | 21.59 | 3.06 |
| v11 + E1a（−gap 當選股 family）− v11 | +0.29（0.46） | +1.70（1.68） | +4.72（1.46） | +1.08（1.88） |
| W0 + E1a − W0 | +0.41（0.56） | −0.39（−0.31） | +1.16（0.40） | +0.24（0.38） |
| v11 + E1a，開盤價誤差 0.1% / 0.2% / 0.3% / 0.5%（5 個 seed） | | | | +0.91 / +1.08 / +0.90 / +0.81 |
| W0 + E1a，同上 | | | | +0.10 / +0.20 / +0.26 / +0.55 |
| v11，夜盤反轉改用真實的平均跳空（E1c） | +0.10（0.22） | −0.74（−1.10） | +1.36（0.88） | −0.04（−0.09） |
| v11 + E1a + E1c（E1d） | +0.49（0.68） | −0.02（−0.02） | +4.44（1.22） | +0.68（1.05） |

- post 的跳空反轉特別強（不用模型就有 20–28 bp），但 val、hold 只有 1–9 bp；2026 年是高波動的一年，不能當成會持續。
- 大盤層級幾乎沒有東西：夜盤報酬和 100 支的平均跳空相關 0.826（2022 起，R² 0.68），v11 的夜盤反轉本來就在押大盤跳空的反轉；換成真實的跳空全期 −0.04。
- 要在 live 用，得在 08:59 左右抓盤前試撮。證交所從 2015-06-29 起在開盤前 30 分鐘揭露模擬成交價、量和最佳五檔，約每 5 秒更新一次；
  08:59–09:00 取消或變更的數量達開盤前申報的 30% 以上（2023-03-20 起），或模擬價格跳動超過 3.5%，會暫緩 2 分鐘開盤。
  實際做法是 08:58 抓價、算、09:00 前上傳，中間只有 1–2 分鐘；而且前半段的試撮常有假單。最多約 +1–2 bp/日，不值得冒「沒交到」的風險。
  如果還是想做，先問助教這樣算不算合規（static 的規則是只能用前一個交易日為止的資料，這一定不行）。

逐年（v11 + 候選 − v11）：E4a 2022 +0.19、2023 +0.11、2024 +0.47、2025 −0.60、2026-01..05 −1.40、post −3.20；
E5 +0.29、+0.49、+0.11、−0.21、−0.82、+2.80。兩個都是「早期一點點正、最近負或亂跳」。E2、E3 大部分年份都是負的（`v14_eval.txt`）。

規則的檢定力（README_v13 定理 3，P(通過 | 真實效果 +0.5 bp/日)）：E2a 0.32、E2b 0.32、E3a 0.24、E3b 0.22、E4a 0.55、E4b 0.83、E5 0.36。
E3 每天的差異很大（val 每天 sd 27–35 bp），但 E3a、E3b 逐年 6 段全部是負的，全期 t −2.7、−3.8，不是運氣。

### 4.4 總表（Top50 bp/日）

| 設定 | val | hold | post | 全期 | 每日 sd | timing | xsec | 猜 1 比例 | P(19 天 > 0) |
|---|---|---|---|---|---|---|---|---|---|
| v11 | 14.60 | 14.34 | 20.35 | 15.02 | 48.00 | 2.76 | 12.25 | 0.59 | 0.94 |
| W0 | 11.79 | 10.35 | 21.71 | 12.22 | 38.40 | −0.89 | 13.10 | 0.62 | 0.91 |
| v11 + E5 | 14.90 | 13.96 | 23.15 | 15.33 | 48.19 | 2.76 | 12.56 | 0.60 | 0.94 |
| v11 + E4a + E5 | 14.96 | 13.76 | 23.42 | 15.33 | 48.35 | 2.79 | 12.54 | 0.60 | 0.94 |
| v11 + E3a | 13.14 | 10.68 | 16.45 | 12.71 | 42.25 | 2.95 | 9.76 | 0.58 | 0.93 |
| v11 + E3b | 11.00 | 11.23 | 13.55 | 11.29 | 39.76 | 3.05 | 8.24 | 0.58 | 0.93 |
| v11 + E1a（開盤價，上限） | 14.90 | 16.04 | 25.06 | 16.09 | 48.35 | 2.48 | 13.61 | 0.60 | 0.94 |

全部 55 個設定在 `results\v14_eval.txt` 最後，每天的分數（1,164 天）在 `results\v14_daily.csv`。

## 5. 在筆電上跑

### 5.1 重跑 v14

- 雙擊 `v14\run_v14.bat`：先建訊號（`v14_signals.py`，約 20 秒，最後做 leak check），再判定（`v14_eval.py`，約 1.5–3 分鐘）。只要 numpy、pandas，不用 torch、lightgbm。
  - `run_v14.bat eval`：只判定（`cache\signals.pkl` 要已經有）；`run_v14.bat signals`：只建訊號。
  - 在你電腦的 VM 實測：建訊號 16 秒、判定 78–89 秒。重跑兩次，55 個設定每天的分數完全相同。
- 會讀的東西：`data\market\`、`v8\cache\`、`v9\cache\`、`v11\cache\screen\x\`、`v8\hedge_beta.py`、`v9\research\v9lib.py`；
  E5 另外讀 `v12\cache\screen\`、`v13\cache\seq_raw\`（沒有時用 `seq_raw_cloud\`）、`reg_svm\cache\svm_exact\`。這三個資料夾刪掉的話，E5 會跳過，其他照跑。
- leak check：每個訊號在 6 個隨機日子用「資料截斷在 D−1」重算，和完整的結果比，差都是 0（`results\v14_signals.txt`）。
- 重組的 v9、v11、W0 和公布的數字逐位相同（PART 0）。

### 5.2 沙盒：自己試一個訊號（`research\v14_try.py`）

```
cd /d "D:\ML RACE\v14\research"
python v14_try.py                       （跑 MY_SIGNALS 裡的每一個，約 15 秒）
python v14_try.py --only rev1 --w 0.2   （只跑一個，權重 0.2）
python v14_try.py --list                （可以用的資料表）
```

在檔案上半部寫一個函式，拿到的是每個欄位一張表（row = 交易日 T、column = 100 支股票：close、open、high、low、volume、amount、trades、
foreign_hold、foreign_net、trust_net、dealer_self、dealer_hedge、RSI、SMA、MACD、ATR、bb_up、bb_mid、bb_low、y），
回傳同樣形狀的表：第 T 天收盤時知道的值，越高代表這支第二天（開盤到收盤）會比較好。程式自己把它移到下一個交易日。例如：

```python
def my_idea(M):
    """5 天的投信買超 / 20 天平均量"""
    return M['trust_net'].rolling(5).sum() / M['volume'].rolling(20).mean()

MY_SIGNALS = {'my_idea': my_idea}
```

輸出：leak check（只能用 T 以前的 row；寫了 shift(−1) 會被抓到）、自己的 IC、和 v11 選股觀點的相關、定理 2 的 w*、w₀ 和最好的 IC、
加進 v11 和 W0 之後每一段的分數差。範例三個（rev1、vol_shock、range_pos）的輸出在 `results\v14_try_example.txt`：三個加進去都變差，w₀ 都是 0。
注意：w* < 0、「最好的 IC」比現在高，代表反過來用會有一點點幫助（例如 range_pos 反過來是 20 天動能）。

這是探索用的。如果試到看起來好的東西，先照 `research\PREREG_v14.md` 的格式把規則寫下來，再拿 hold、post 判定；
在同一份資料上一直試到好為止，就是 overfit 檢查裡說的 selection bias（試 N 個、留最好的，平均會高估 σ·E[max of N 個標準常態]）。

## 6. 檔案

| 檔案 | 內容 |
|---|---|
| `run_v14.bat` | 雙擊重跑（建訊號 → 判定） |
| `research\PREREG_v14.md` | 事先寫好的規則（11/7 01:40；01:55 補充 E3 的細節，都在任何 E2–E5 成績之前） |
| `research\v14_signals.py` | 從 `data\market\*.csv` 建訊號（s-score、殘差反轉、PCA beta、投信 60 天、跳空）和季底日曆，leak check → `cache\signals.pkl`、`cache\calendar.pkl` |
| `research\v14lib.py` | 共用：重組 v9 / v11 / W0 的 Data（= v13_eval.Data，複製過來，v13 刪掉也能跑）、spread-matched 放法、Top50 中性化 |
| `research\v14_eval.py` | 判定和報告 → `results\v14_eval.txt`、`v14_daily.csv`（每個設定每天的分數）、`v14_selection.json` |
| `research\v14_try.py` | 沙盒 |
| `results\v14_signals.txt`、`v14_eval.txt`、`v14_daily.csv`、`v14_selection.json`、`v14_try_example.txt` | 結果 |
| `results\v14_eval.err`、`v14_signals.err` | 第一次執行時的警告輸出（空的），可以刪 |
| `cache\` | 訊號（約 20 MB），可以刪，重跑會重建 |

整個 `v14\` 只有研究，可以整個刪掉；每天的流程不讀它。

## 7. 接下來值得做的

1. **11/23 的風險設定（W0 還是 v11）**：這是唯一還會明顯影響「拿第一」的決定。v7 第三節、v8 第七節的計畫要用 10/12–11/20 各隊每天的分數
   （平均、波動、和大盤的相關）。`leaderboard\` 目前只有 10/22、10/27 的截圖，`daily_board.csv` 是空的；網站上看得到舊資料的話，現在補記最划算。
   道理是比賽策略：最後名次看的是 19 天的累積分數，領先的時候要選和對手相關高、變異數小的設定，落後的時候反過來（Brown, Harlow & Starks 1996 的基金錦標賽）。
2. **不要把 v14 的任何候選放進每天的流程。** 最接近的 E5 全期 +0.31（t 0.74），hold 是負的。
3. **還想試的話用沙盒**，例如 101 Formulaic Alphas（Kakushadze 2016）裡只用價量的那些。預期和 Alpha158 一樣：同樣的資訊，模型多半已經有了。
   真正可能有用的只有「模型沒有的新資訊」，而且要在 09:00 前拿得到、能 point-in-time 回測；E1 說明了連開盤價本身都只值約 1 bp。
4. 順帶：v13 的 F-LOAM 在筆電已經跑完（`v13\results\v13_local_eval.txt`）：v11 + F-LOAM − v11 val +0.46（1.81）入選、hold −1.41（−3.11）→ 沒通過；`README_v13.md` 第 7.5 節還沒填。

## 參考文獻

- Akbas, F., Boehmer, E., Jiang, C., & Koch, P. D. (2022). Overnight returns, daytime reversals, and future stock returns. *Journal of Financial Economics*, 145(3), 850–875.
- Avellaneda, M., & Lee, J.-H. (2010). Statistical arbitrage in the US equities market. *Quantitative Finance*, 10(7), 761–782.
- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017). The probability of backtest overfitting. *Journal of Computational Finance*, 20(4), 39–69.
- Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*, 40(5), 94–107.
- Blitz, D., Huij, J., Lansdorp, S., & Verbeek, M. (2013). Short-term residual reversal. *Journal of Financial Markets*, 16(3), 477–504.
- Bogousslavsky, V. (2021). The cross-section of intraday and overnight returns. *Journal of Financial Economics*, 141(1), 172–194.
- Brown, K. C., Harlow, W. V., & Starks, L. T. (1996). Of tournaments and temptations: An analysis of managerial incentives in the mutual fund industry. *Journal of Finance*, 51(1), 85–110.
- Carhart, M. M., Kaniel, R., Musto, D. K., & Reed, A. V. (2002). Leaning for the tape: Evidence of gaming behavior in equity mutual funds. *Journal of Finance*, 57(2), 661–693.
- Cohen, L., & Frazzini, A. (2008). Economic links and predictable returns. *Journal of Finance*, 63(4), 1977–2011.
- Da, Z., Liu, Q., & Schaumburg, E. (2014). A closer look at the short-term return reversal. *Management Science*, 60(3), 658–674.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy? *Review of Financial Studies*, 22(5), 1915–1953.
- Frazzini, A., & Pedersen, L. H. (2014). Betting against beta. *Journal of Financial Economics*, 111(1), 1–25.
- Gao, L., Han, Y., Li, S. Z., & Zhou, G. (2018). Market intraday momentum. *Journal of Financial Economics*, 129(2), 394–414.
- Gârleanu, N., & Pedersen, L. H. (2013). Dynamic trading with predictable returns and transaction costs. *Journal of Finance*, 68(6), 2309–2340.
- Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs trading: Performance of a relative-value arbitrage rule. *Review of Financial Studies*, 19(3), 797–827.
- Grinold, R. C. (1989). The fundamental law of active management. *Journal of Portfolio Management*, 15(3), 30–37.
- Grinold, R. C., & Kahn, R. N. (2000). *Active Portfolio Management* (2nd ed.). McGraw-Hill.
- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357–384.
- Heston, S. L., & Sadka, R. (2008). Seasonality in the cross-section of stock returns. *Journal of Financial Economics*, 87(2), 418–445.
- Hou, K. (2007). Industry information diffusion and the lead-lag effect in stock returns. *Review of Financial Studies*, 20(4), 1113–1138.
- Jegadeesh, N. (1990). Evidence of predictable behavior of security returns. *Journal of Finance*, 45(3), 881–898.
- Kakushadze, Z. (2016). 101 formulaic alphas. *Wilmott*, 2016(84), 72–81.
- Kelly, J. L. (1956). A new interpretation of information rate. *Bell System Technical Journal*, 35(4), 917–926.
- Keloharju, M., Linnainmaa, J. T., & Nyberg, P. (2016). Return seasonalities. *Journal of Finance*, 71(4), 1557–1590.
- Lakonishok, J., & Smidt, S. (1988). Are seasonal anomalies real? A ninety-year perspective. *Review of Financial Studies*, 1(4), 403–425.
- Lehmann, B. N. (1990). Fads, martingales, and market efficiency. *Quarterly Journal of Economics*, 105(1), 1–28.
- Lo, A. W., & MacKinlay, A. C. (1990). When are contrarian profits due to stock market overreaction? *Review of Financial Studies*, 3(2), 175–205.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Lou, D., Polk, C., & Skouras, S. (2019). A tug of war: Overnight versus intraday expected returns. *Journal of Financial Economics*, 134(1), 192–213.
- Moreira, A., & Muir, T. (2017). Volatility-managed portfolios. *Journal of Finance*, 72(4), 1611–1644.
- Moskowitz, T. J., & Grinblatt, M. (1999). Do industries explain momentum? *Journal of Finance*, 54(4), 1249–1290.
- Pitt, L. D. (1982). Positively correlated normal variables are associated. *Annals of Probability*, 10(2), 496–499.
- White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5), 1097–1126.
- Yang, X., Liu, W., Zhou, D., Bian, J., & Liu, T.-Y. (2020). Qlib: An AI-oriented quantitative investment platform. arXiv:2009.11189.

盤前試撮的規則：

- 今周刊 Smart 自學網〈試撮是什麼〉：證交所 2015-06-29 起在開盤前 30 分鐘、收盤前 5 分鐘揭露模擬成交價、量和最佳五檔。https://smart.businessweekly.com.tw/Magazine/detail.aspx?id=7004615
- 股感〈暫緩開盤〉：08:59–09:00 模擬價格跳動超過 3.5%，或（2023-03-20 起）取消、變更達開盤前申報數量 30% 以上，暫緩 2 分鐘開盤。https://www.stockfeel.com.tw/?p=134934
- uptogo〈盤前試撮時段〉：08:30–09:00 約每 5 秒模擬撮合一次。https://uptogo.com.tw/?p=331721
- 證交所基本市況報導網站的即時資料介面 `getStockInfo.jsp`（iThome 鐵人賽文章的說明）。https://ithelp.ithome.com.tw/articles/10258478

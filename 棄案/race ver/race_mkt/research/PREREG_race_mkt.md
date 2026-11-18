# race_mkt：讓猜 1 的數量可以變動的大盤觀點，以及期貨籌碼（大台／小台／微台、多空 OI）能不能改善盤前資訊（事先寫好的規則）

寫於 2026-11-17 09:50–10:05（台北），在這個實驗的任何成績（D 或 V）算出來之前；期貨與選擇權資料在寫這份規則時還沒下載
（`run_fetch_mkt.bat` 由使用者在 Windows 執行）。這份檔案的 SHA-256 會在第一次建資料時寫進 `results\snapshot.json`，之後
`LOCK.json`、`SEALED_V.json` 都會核對；寫完不再修改（要改只能另開新檔並註明原因）。

使用者的要求（11/17 09:43）：「理論上我們現有策略猜 25 個 1、25 個 0 是一個上限很低的事，我想要可以產出任何數量 0/1 但有效的方法；
既然盤前資訊是整件事的核心，可以透過優化盤前資訊來進步嗎？我想用微台指、中台指（= 小台指 MTX）、大台指跟買多 OI、買空 OI 間的
關係來推測大盤。幫我在 race ver 建立數個實驗，可以用有創意一點的方法。」

## 0. 寫這份規則時已經知道的

- race 版（`..\race_live`）= C 組 169 個 feature ＋ LU ＋ 每天重訓 3 個模型平均（WF3）＋ Top50 前 25 支猜 1 ＋除權息規則。
  D（2022–2024）WF3 16.35 bp/日（t 10.25）；年度重訓 C.LU：D 15.10、V 16.16。Top25 時 timing ≡ 0（除了除權息改 0 造成的少量偏離）。
- 以前的大盤觀點都沒過：v8（Baszo 的 pbar、ridge、各種架構，大盤相關約 0.0–0.15）；race_std 第一部分陳聖翰的 Ridge 大盤觀點在 C.LU 上
  D −2.07、V −0.76；第二部分 pbar w_T 0.5 D +1.28（t 1.36，2023 −0.37）、v11 整套決策層 +0.07。
- C 組已經有外資台指期淨 OI 的變化（`C__txoi_chg1`、`chg5`、`z20`，來自課程 global 的 TX_OI）、夜盤、日韓開盤、ADR、美股、VIX 等
  61 個 day-level feature（`race_std\cache\day_features.pkl`）。
- 寫這份規則前看過的數字：Top50 盤中平均報酬 m 的每年平均與標準差（2022–2026：sd 57–105 bp，2026 最大）、2026-09-23 到 11-09 每天的 m；
  沒有算過任何 feature 和 m 的關係，也沒有算過任何下面的模型或決策的成績。

## 1. 資料與時點

- 凍結資料：`race_std\cache\info.pkl`（Top50 旗標、target_r、除權息旗標）、`day_features.pkl`、有 label 的最後一天 2026-11-09。
  開盤跳空 g_t = Top50 平均 (adj_open_t / adj_close_{t−1} − 1)，從 `data\market\*.csv` 算，只用 ≤ 2026-11-09。
- 選股分數：D 用 WF3（`race_live\cache\wf_units`，第 t 天 = t 之前 3 個交易日的模型平均，= `race_live\walkforward.scores`）；
  V 用 C.LU 年度 fold（`race_std\cache\units\C.LU`，3 seed 平均）；V 的校準用 C.LU 2022–2024 fold 的預測接在前面。
- 新資料（`ext\`）：FinMind TaiwanFuturesInstitutionalInvestors（TX、MTX、TMF）、TaiwanOptionInstitutionalInvestors（TXO）、
  TaiwanFuturesDaily（TX、MTX、TMF）、TaiwanStockTotalInstitutionalInvestors、TaiwanStockTotalMarginPurchaseShortSale；
  期交所大額交易人未沖銷部位（TX 合計 = TX + MTX/4 + TMF/20）、TXO Put/Call 比、近 3 年三大法人各期貨契約（只用來核對 FinMind）。
- 時點：第 t 天的期貨／選擇權／大盤籌碼 feature 只用日期 < t 的最後一筆（t−1 收盤後 13:45–18:00 公布，09:00 前拿得到）；
  夜盤成交量用標 t 的夜盤（t 日 05:00 收盤）。最後一筆比 t 早超過 7 個日曆天就當缺值。TMF 2024-07-29 才上市，之前 TMF 的量當 0。
- 某個來源下載失敗：它的 feature 從所有模型拿掉，照常進行，在 README 註明。

## 2. 期貨籌碼 feature（37 個，全部只用過去資料）

記號：z250(x) = (x − 前 250 天平均) / 前 250 天 sd（至少 60 天）；d1n(x) = (x_t − x_{t−1}) / 前 60 天 Δx 的 sd（至少 20 天）；
d5n 同理用 5 天差、250 天 sd；z60 同理 60 天。全部截在 ±5。淨 OI = 多方未平倉口數 − 空方未平倉口數。TX 當量 = TX + MTX/4 + TMF/20。
全市場 OI = TaiwanFuturesDaily 日盤（position）所有單月契約 OI 的和（不含價差單）。散戶淨 OI = −（外資 ＋ 投信 ＋ 自營商淨 OI）。

| 組 | feature |
|---|---|
| 三大法人（8） | 外資 TX 當量淨 OI 的 z250、d1n、d5n；投信 z250、d1n；自營商 z250、d1n；外資 TX 當量淨交易口數（多方交易 − 空方交易）/ 前 60 天 sd |
| 散戶（7） | 小額散戶比 = (MTX 散戶淨 OI/4 + TMF 散戶淨 OI/20) / (MTX 全市場 OI/4 + TMF 全市場 OI/20) 的水準、z250、d1n；MTX 散戶比 z250；TMF 散戶比 z250；大台 TX 散戶比 z250；背離 = 小額散戶比 z250 − 外資 TX 當量淨 OI z250 |
| 大額交易人（7） | TX 合計、所有契約（999999）：前十大淨部位 / 全市場 OI（全部交易人）z250、d1n；特定法人 z250；非特定 = 全部 − 特定 z250、d1n；前五大全部 z250；近月前十大淨比 − 所有契約前十大淨比 的 z250 |
| 選擇權（7） | 外資 TXO 買權淨未平倉金額 z250；賣權 z250；偏多 = 買權 − 賣權 z250、d1n；P/C 未平倉比 z250、d1n；P/C 成交量比 z60 |
| 價量（6） | 基差 = TX 日盤成交量最大的單月契約收盤（13:45）/ ^TWII 收盤（13:30）− 1，減前 20 天中位數；價差結構 = (次近月 − 近月結算價) / 近月 − 前 60 天中位數；TX 當量全市場 OI d1n；OI×價格 = sign(TX 日盤報酬) × TX 當量 OI d1n；小台＋微台成交量占比 z60；夜盤 TX 成交量 / 前一日盤 TX 成交量 z60 |
| 全市場現貨（2） | 外資現貨淨買超金額 / 前 60 天 sd；融資餘額（金額）變化 / 前 60 天 sd |

## 3. 大盤模型（walk-forward，每月第一個交易日 refit）

- 目標 m_t = 當天有 label 的 Top50 平均 target_r（bp）。訓練窗 = 2020-03-02 到 refit 前一個交易日（有 label），expanding。
- 前處理（只用訓練窗）：每個 feature 截在 1% / 99% 分位、標準化、缺值補 0；m 截在 1% / 99% 分位。
- Ridge 的 λ ∈ {3, 10, 30, 100, 300, 1000, 3000, 10000}：訓練窗切 5 段連續的區塊，第 2–5 段各用前面的區塊訓練、算 MSE，取平均最小。

| 代號 | 模型 |
|---|---|
| MB | Ridge，C 的 61 個 day feature 去掉 I_dow、I_month、I_dom（58 個）＝ 現在就有的盤前資訊 |
| MF | Ridge，MB 的 58 個 ＋ 第 2 節 37 個 |
| MFO | Ridge，只有第 2 節 37 個 |
| MS | 結構式（跳空反轉）：第一段 OLS 用 tx_night_ret、asia_gap_mean、n225_gap、ks11_gap、gspc_ret1、ixic_ret1、sox_ret_1、tsm_ret_1、K_adr_prem_chg_1、us_minus_tw_ret1 預測開盤跳空 g；第二段 OLS m on ĝ |
| MV | 投票：MF 的 95 個 feature 各自和 m 的訓練窗相關，\|t\| ≥ 2 的取 sign(r)·z 平均，再 OLS m on 投票分數 |

OOS 預測從 2021-01 開始（2021 只用來校準，不計分）。

## 4. 校準與決策層（任何數量的 1）

- 大盤：μ̂_t = â + b̂·m̂_t，(a, b) 是 m 對 m̂ 的 OLS，用 t 之前所有 OOS 日（從 2021-01 起，expanding，至少 120 天，不足就 μ̂ = 0）；
  經驗 Bayes 收縮：b̂ = max(0, b)·max(0, 1 − 1/t_b²)，â = a·max(0, 1 − 1/t_a²)。
- 選股：q_i = 當天 Top50 依分數的常態分數 Φ⁻¹((rank − 0.5)/n)（高分 = 正）；ĉ_t = e_i = r_i − m 對 q_i 的過原點 OLS（t 之前所有有分數的日子，
  pooled，至少 60 天，不足那天就用 DL0）。
- 除權息規則（當天、前一天、後一天、後兩天改 0）在所有決策之後套用。

| 代號 | 決策 |
|---|---|
| DL0 | Top25（race 版基準） |
| DL1 | s_i = sign(μ̂_t + ĉ_t q_i)（μ̂ = 0 時就是 Top25） |
| DL2 | s_i = sign(β_i μ̂_t + ĉ_t q_i)，β_i = v8 `hedge_beta.betas`（截在 [0.2, 2.5]，缺值補當天中位數） |
| DL3 | 直接學猜幾支：k ∈ {10, 15, 20, 25, 30, 35, 40}；對 t 之前的 OOS 日算 G_s(k) = AR_s(Top-k) − AR_s(Top25)，每個 k 做 G on m̂ 的 OLS（同樣的經驗 Bayes 收縮），取預測 G 最大的 k（k = 25 的 G ≡ 0；至少 120 天） |
| DL4 | 只在極端訊號下注：z_t = m̂_t 在校準窗的標準化值，\|z_t\| ≥ 1 才用 μ̂（(a, b) 只用 \|z\| ≥ 1 的過去 OOS 日估），其他天 Top25 |

## 5. 候選與判定（只用 D）

| 候選 | 決策 × 大盤模型 | 回答的問題 |
|---|---|---|
| P1 | DL1 × MB | 不加新資料，「任何數量」本身有沒有用 |
| P2 | DL1 × MF | 加期貨籌碼 |
| P3 | DL1 × MFO | 只靠期貨籌碼 |
| P4 | DL2 × MF | beta 版 |
| P5 | DL3 × MF | 直接學 k |
| P6 | DL4 × MF | 只在極端訊號下注 |
| P7 | DL1 × MS | 跳空反轉的結構式模型 |
| P8 | DL1 × MV | 投票 |

1. 每個候選和 DL0 的每日成對差 d_t（D，2022-01-03 到 2024-12-31）。**入選（三個都要）：D 平均 > 0、Newey–West t（5 lags）≥ 2.0、
   3 年至少 2 年 > 0。** 另外報告 Holm 校正的單邊 p（8 個）。
2. 期貨資料有沒有用（H2）：P2 − P1 的成對差（同樣的 t），以及 MF 和 MB 的 OOS 相關 corr(m̂, m) 逐年比較。只報告，不另外判定。
3. 都沒入選 → race 版不變。入選的（最多 3 個，依 D 平均）進第 6 節。

## 6. V（只做一次）

- D 的結果寫進 `results\LOCK.json`（含這份檔案和程式的 SHA-256）之後，`mkt_eval.py --seal-v` 才會算 V，算完寫 `SEALED_V.json`，
  之後拒絕再算。V 一次算 DL0 和全部 8 個候選（只報告），但採用條件只看第 5 節入選的：**D 入選而且 V 平均差 > 0**。
- V 的選股分數是年度 C.LU（不是每天重訓），和 D 不同；比的是同一天、同一組分數下「加大盤觀點 − Top25」，所以仍可比。

## 7. 只報告（不參與判定）

- 天花板：D 上 (a) 知道 m 的正負、用我們的排序選最好的 k；(b) m > 0 全猜 1、否則全猜 0；(c) 知道真實排序的 Top25；(d) 全知道。
- 「大盤預測要多準才值得」：用 D 的真實 m 造 z = ρ·m/σ + √(1 − ρ²)·ε，ρ ∈ {0.05, 0.1, 0.15, 0.2, 0.3}，DL1 用已知校準
  （μ̂ = ρσ_m z），算 AR − DL0；和公式 ≈ ρ²σ_m² / (σ_α√(2π)) 對照（README 的命題）。
- 每個大盤模型的 OOS corr(m̂, m)（逐年、依波動分三組）、sign 命中率、全押時 E[sign(m̂)·m]；每個期貨 feature 和 m 的相關（D，描述）。
- 每個候選：猜 1 的支數分布、AR 拆成 timing 和選股、19 天平均 > 0 / 20 / 30 / 40 bp 的機率（stationary bootstrap，block 10）。
- live 的提醒：race_live 的 FATAL「猜 1 不在 15–25 支」會擋住任何數量的版本；入選的話要另外改 guard，不在這次範圍。

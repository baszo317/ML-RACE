# v11 事先寫好的規則（2026-10-26 14:50 寫，任何 v11 的新成績算出來之前）

使用者的要求：「先把雜訊降下來」，用成員層級的 rank IC、20 個 seed 的 baseline、100 支股票、更嚴的門檻加穩定性條件、
組合先在 val 確認（forward selection）、先剔除重複資訊、用 nested walk-forward 比較規則，挑出最好的 Alpha158 feature，弄成 v11。
v9、v10 的檔案都不改；v11 的程式、cache、結果都在 `v11\`。

## 0. 已經看過的東西（誠實說明）

- v10 的 AR 篩選表（每組在 val、hold、post 的成績）和 V10 的檢驗。
- 上一輪用 rank IC（5 個 seed 的 baseline）重算的觀察：val 的 IC 離散程度大於雜訊、hold 回到雜訊、val 和 hold 排名幾乎不相關、
  「IC val t > 2」會選到 M 的 STD、RSV、RANK、MAX、QTLU、SUM，L 沒有。
- 所以這份規則是看過這些之後寫的。nested walk-forward 每一段只用之前的資料挑，但規則本身的設計看過所有期間；
  完全乾淨的只有 10/27 以後的 live。

## 1. 候選、family、成員（和 v10 相同）

- 候選 = Alpha158 的 23 個資訊群組（`v10\alpha158.py` 的 groups()）。ALL（158 個）只當 R5 規則和參考。
- L：選股 LightGBM `lgb_res`（B 67 個 + 群組）。M：選股 MLP `mlp_res_xr9`、`mlp_res_BN9`、`mlp_res_xrN9`（各自 v9 的 feature + 群組，
  三個一起加）。其他成員、大盤觀點、規則與係數（w_T 0.5、除權息 4 天、對沖 λ 0.75）都不動。
- 訓練程式與 v10 相同（`v10\research\v10lib.py` 的 train_lgb_unit、train_mlp_unit，年度 walk-forward 6 個 fold）。
  已確認雲端重訓 v10 的檔：LightGBM 完全相同（差 0），MLP 差 1.2e-7。

## 2. Stage 0：先剔除重複資訊（不看 label）

- 資料：100 支股票，2020-03-02 到 2021-12-31（第一個 fold 的訓練期），不用任何 label 或成績。
- 每個 Alpha158 欄位 c：R²_c = max(c 對該 family 既有 feature 的 OLS R²，c 和任一個既有 feature 的 Spearman 相關係數平方)。
  既有 feature：L = B（67 個）；M = 三個 MLP 的 feature 聯集（B9 + XR + NEW + XRN + SIG）。先做 0.5%/99.5% winsorize 和標準化，缺值補 0。
  第一項抓「幾個既有 feature 的線性組合」，第二項抓「某一個既有 feature 的單調轉換」（例如 ROC5 = exp(−ret5)）。
- 一個群組的 R² 取它所有欄位的平均；**平均 ≥ 0.90 就從該 family 的候選拿掉**（約等於平均相關係數 ≥ 0.95）。
- K_f = 留下來的群組數，用在 Bonferroni。

## 3. Stage 1：baseline 和指標

- baseline 用 20 個 seed：L 的 seed 1–5（v10 雲端訓練的檔）加 6–20（這次雲端訓練）；M 的 seed 1–10（筆電）加 11–20（這次雲端訓練）。
- **baseline 分成 4 個 5-seed 區塊（1–5、6–10、11–15、16–20），每個區塊各自算指標再平均。** 候選只有 5 個 seed，
  直接和 20 個 seed 的平均比，baseline 會因為「平均的模型比較多」而占便宜（ensemble 大小造成的偏差），所以比的是
  「5 個 seed 的候選」對「5 個 seed 的 baseline 的期望值」：沒有偏差，baseline 的雜訊又降為 1/4。
- 主要指標：每天的 rank IC = 當天所有有 label 的股票（約 100 支）上，family 的預測和實際盤中報酬（target_r）的 Spearman 相關。
  L 的預測 = lgb_res 的 seed 平均；M 的預測 = 三個 MLP 各自 seed 平均後再平均。（同一天減掉平均不影響排名，所以對 target_r 和
  resid_r 算出來一樣。）
- ΔIC_g(d) = IC_g(d) − 4 個 baseline 區塊 IC 的平均。t = Newey–West（5 lags）。
- 次要指標（AR guard 和報告用）：只換該 family 的成員、其他成員用 v9 的 walk-forward 預測，照 v9 的規則算 Top50 bp/日；
  baseline 的 AR = 4 個區塊的 AR 平均。
- 期間：val 2022–2024、hold 2025-01 到 2026-05、post 2026-06-01 到 2026-10-22。「年」= 日曆年，2026 年在 nested 裡依 fold 拆成
  2026 年 1–5 月和 post。

## 4. Stage 2–3：v11 的挑選規則（R4）

在一個挑選視窗 W 裡，每個 family 分開做：

1. **篩選**：ΔIC 的 NW t > t_B = Φ⁻¹(1 − 0.05 / K_f)（單尾 Bonferroni，K = 23 時 2.85），而且 W 裡每一年的 ΔIC 平均都 > 0。
2. **組合（forward selection）**：通過的群組依 t 由大到小。S = {第一個}；之後每一個 g，訓練 S ∪ {g}（6 個 fold、seed 1–5），
   g 留下的條件是 IC(S ∪ {g}) − IC(S) 在 W 的 NW t > 0（「合起來的 val 有改善才留下」）。
3. **AR guard**：最後的 S 只換該 family 時，W 上的 ΔAR（對 4 個區塊的平均）NW t 要 > −1；否則 S = ∅。

## 5. Stage 4：用 nested walk-forward 比較規則（評估「方法」）

- 外層期間 Y：2023、2024、2025、2026 年 1–5 月、post。每個 Y 的挑選視窗 W(Y) = 2022-01-01 到 Y 的前一天。
- 規則（每個 family 分開）：
  - R0：什麼都不選（= v9）。
  - R1：v10 的規則 —— AR plain t > 0（baseline = seed 1–5），入選的全部合在一起。
  - R2：IC t > 0，入選的全部合在一起。
  - R3：IC t > 2 且 W 裡每年 > 0，入選的全部合在一起。
  - R4：v11 的規則（第 4 節）。
  - R5：ALL 158 個（不挑）。
- 每個規則在 W(Y) 挑出 S，再看 S 在 Y 的表現：ΔIC（family 層級）和 ΔAR（只換該 family）。S 有兩組以上時，用 seed 1–5 訓練
  S 在需要的 fold；S 是空的，差就是 0。
- 合併所有外層期間的天數（2023-01 到 2026-10-22）：平均 ΔIC 與 NW t、平均 ΔAR 與 NW t。

## 6. 最後的 feature（每個 family 分開決定）

1. 合格的規則：nested 合併的 ΔIC NW t > 2，而且合併的 ΔAR > 0。
2. 合格的規則裡，選 nested 合併 ΔIC 平均最高的那一個（R1–R5）。
3. 沒有合格的規則 → 這個 family 不加任何 Alpha158 群組。
4. 選出的規則套用在 **W_live = 2022-01-03 到 2026-10-22（全部 walk-forward 的天數）**，得到 v11 這個 family 的群組。
   理由：nested 每一段都用「之前全部的資料」挑，live 是下一段，所以最後一次也用之前全部的資料。
5. 只報告：同一個規則只用 val（2022–2024）挑出的名單和它在 hold、post 的表現（hold、post 已經看過，不是乾淨的檢驗）。

## 7. v11

- 兩個 family 都沒有群組 → v11 = v9（live 不改），只留研究結果。
- 有群組 → v11 = v9 的成員換成加了群組的版本，完整 walk-forward 重訓（seed 1–5 的區塊對 baseline 區塊）報告各期間；
  另外做 v11 的每天預測程式（`v11\`，自己的模型池），和 v9 並行輸出，上傳哪一個由使用者決定。
- 只報告：每組的 val、hold、post 的 ΔIC 與 ΔAR、重複資訊表、p_e 的離散程度（加 feature 會不會改變選股觀點和大盤觀點的相對權重）、
  baseline 區塊之間的差（純雜訊的參考）。

## 8. 誠實說明

- 最後一步用到 hold、post，所以 v11 的名單沒有乾淨的歷史檢驗；證據是 nested walk-forward（每一段都只用之前的資料挑）。
- 規則在看過 v10 的表和上一輪 IC 分析之後才寫，nested 的結果仍可能偏樂觀。乾淨的只有 10/27 以後的 live；以這種幅度，
  live 大約要一年才看得出差別。

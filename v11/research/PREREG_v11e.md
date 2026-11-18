# v11 加 ranking objective 的 LightGBM（X5）：事先寫好的規則（2026-10-26 18:00，任何 X5 的成績算出來之前）

使用者的要求：「選股成員更多樣」的方向裡還沒測的 ranking objective，照同一套規則測。底是現在的 v11（v9 + T2 + T3 + X1，
w_T 0.5、λ 0.75、T2 權重 0.5、T3 5 支）。v9、v11 現有的檔案都不改；程式 `research\v11e_*.py`，結果 `results\v11e_*.txt`。

## 1. 候選（只有一個）

- **X5 lgb_rank**：LightGBM，objective = lambdarank，每一天的約 100 支股票是一組（query）。
  - label：同一天內盤中超額報酬 resid_r 的十分位（0–9；同一天依排名切）。
  - label_gain 用線性 0, 1, …, 9（預設的 2^l − 1 只在乎最前面幾支；我們要的是整天的排序，因為猜 1、猜 0 的分界在中間）；
    lambdarank_truncation_level 100（整組都算）。
  - feature：B（67 個，和 lgb_res 相同，只換 objective，看 objective 本身的多樣性）。
  - 其他參數和 lgb_res 相同（learning rate 0.02、num_leaves 15、min_data_in_leaf 500、feature_fraction 0.8、lambda_l2 1）；
    bagging 0.8 改成以「天」為單位抽（bagging_by_query），才不會把一天拆開。沒有樣本權重（報酬大小已經在十分位 label 裡）。
  - early stopping：訓練期最後 120 天的 NDCG@100（最多 1000 輪、patience 100），再用全部訓練資料重訓到最佳輪數。
  - seed 1–5、6 個年度 fold（30 個模型），和 X1、X2 相同的 walk-forward。
- 放進 v11：X5 的輸出是分數不是機率，所以每天先減去當天平均，再縮放到和 lgb_res 當天的標準差相同（和 X2 相同的做法），
  然後選股觀點的 LightGBM 部分 = (lgb_res + lgb_res_xrN + 縮放後的 X5) / 3。其他都照 v11。

## 2. 規則（使用者的規則，和 PREREG_v11b 相同）

1. **入選：v11 + X5 對 v11 的 val（2022–2024）AR 差 > 0。**
2. **檢驗：hold（2025-01 到 2026-05）差 > 0 且 post（2026-06-01 到 2026-10-22）差 > 0** → X5 加進 v11 每天的流程
   （預設打開，加 `--no-x5` 可以關掉）；任一段 ≤ 0 → 不加，v11 不變。
3. 只報告：全期、6/1–9/30、10 月、每日 sd、P(19 天 > 0)、選股觀點的 rank IC、X5 單獨的 rank IC、v9 + X5 對 v9。

## 3. 誠實說明

- 這是事後再加的一個候選；val、hold、post 都看過很多次，乾淨的只有 live。單一候選、規則寬，沒有用的改法也有約四分之一的機率
  三段都 > 0（val > 0 約一半，再兩段同號）。
- 第 6 節（README_v11 A6）的換算：選股 family 的 IC 每多 10⁻³ 只換 0.03–0.06 bp/日，所以就算 X5 有用，期望效果也很小。

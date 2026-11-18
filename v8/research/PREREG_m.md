# v8 事先寫好的規則（2026-10-24 19:55，跑 m1 之前寫，跑完不改）

對照組：v4 規則（score = 0.5 × (pbar − 0.5) + p_e_c，除權息日猜 0），成員是現有 walk-forward 預測（x0 匯出）。

四個處理組（都不重訓神經網路，只改「大盤觀點怎麼算、怎麼用」）：

- T1 beta 一致規則：β_i = 個股盤中報酬對 Top100 平均盤中報酬的 120 日滾動 beta（只用 t 以前，夾在 [0.2, 2.5]，缺值用當天中位數）。
  p⊥_i = p_e_c,i − b_t (β_i − β̄_t)，b_t 是當天 100 支的 p_e_c 對 β 的橫斷面 OLS 斜率。
  score_i = 0.5 × (pbar − 0.5) × β_i / β̄_t + p⊥_i。
- T2 日頻大盤模型取代 pbar：ridge，target = Top50 平均盤中報酬（夾在 ±3%），輸入 = d1 第 (3) 部分的 32 個盤前大盤變數（含夜盤 3 個），
  訓練期 2020-03-02 到 t−1，每月第一個交易日重訓，λ 用訓練期最後 250 天做 time-series 驗證從 {1,3,10,...,3000} 選。
  市場項 = 0.5 × sd_past(pbar) × μ̂_t / sd_past(μ̂)（等風險：和 v4 的市場項同樣大小，只換方向訊號；不扣平均，保留截距）。
- T3 組合：市場項 = 0.5 × sd_past(pbar) × S_t / sd_past(S)，S_t = (pbar_t − 0.5)/sd_past(pbar) + μ̂_t/sd_past(μ̂)。
- T4 = T3 的市場項 + T1 的 beta 一致規則。

判斷：val（2022–24）成對差的 t 依 Holm 排序，最好的要 ≥ 2.24、第二 ≥ 2.13、第三 ≥ 1.96、第四 ≥ 1.645，且 hold（2025-01 到 2026-05）的差 > 0，才採用。
post（2026-06 以後）只報告：它在 v2 設計和這次 d1–d3 的診斷都看過，不乾淨。
誠實說明：d1–d3 已經看過 val/hold 的相關係數（例如夜盤報酬為負相關、選股觀點的 beta 傾斜），所以這四個假設不是完全事前；T2/T3 用全部 32 個變數、不挑變數，就是為了不把這個資訊放進模型。

## 第二輪（2026-10-24 20:30，m2 結果出來後、訓練之前寫）

m2 的四個都沒過。T1 顯示：事後把選股觀點對 beta 中性化，同時拿掉了「押盤中偏空」的部分和一部分真的 alpha（rest 10.6 → 9.1）。
所以只再測一個架構改法，訓練前寫好：

- T5 選股觀點改用 beta 調整後的殘差當 label：e_i = r_i − β_i,t × m_t（β 同 T1，m_t = 當天 Top100 平均盤中報酬），
  y = 1{e_i > 0}，權重 min(|e_i|, 5%)；mlp_res_B、mlp_res_xr 各 seed 1–5，年度 fold + post（和 wf_runner 'year' 相同程式與 HP）。
  決策：score_i = 0.5 × (pbar − 0.5) × β_i / β̄_t + (p_ε,i − 當天平均 p_ε)，大盤成員不變。
- 對照組：v4 規則，選股成員用現有快取的 seed 1–5（同一個 protocol；v7 已確認雲端重訓和快取一致到 1e-6）。
- 判斷：val 成對 t ≥ 2.0 且 hold 差 > 0 才採用；post 只報告。只測這一個；沒過就停，不再調參。

## 第三輪（2026-10-24 20:25，使用者問「低 beta 偏好怎麼解決」之後、跑之前寫）

目標：每天把整本的 beta 曝險 E = mean_i(s_i β_i)（Top50，除權息日固定猜 0）拉回 E* = s̄_v4 × β̄，
也就是只拿掉「選股觀點帶來的 beta 傾斜 B」，保留大盤觀點原本的淨曝險。β 同 T1。兩個做法：

- H6a 用多空家數對沖：選股排序（v4 score）完全不變，只改猜 1 的家數 k，選 |E(k) − E*| 最小的 k（同分取最接近 v4 的 k）。
- H6b 用 beta 比例對沖：score'_i = score_i + c_t × β_i，c_t 從 [−0.3, 0.3]（間隔 0.0025）中選 |E − E*| 最小的（同分取 |c| 最小）。
  這是「在 E = E* 的限制下讓 Σ s_i × score_i 最大」的 Lagrange 解的形式。

判斷：val 成對 t 依 Holm（2 個：1.96、1.645）且 hold 差 > 0 才採用；post 只報告。
事前預期：兩個都會讓平均變小（暗押注在 val、hold 是賺的），主要是量化「解決」要付多少代價、波動少多少。
另外只報告（不判斷）：E* = s̄_v4 β̄ + λ B_v4，λ = 0、0.25、0.5、0.75 的前緣，以及大漲日、大跌日的分數。

## 第四輪（2026-10-24 20:45，使用者問「Transformer + MLP 是否沒用、要不要換 CNN」之後、訓練之前寫）

大盤觀點三個處理組（選股觀點、除權息規則、w_T 0.5 都不變）：

- C0 只用 Transformer：p_raw = tf_raw_B（不用兩個 MLP）。不需訓練。
  （d1 已看過 Transformer 的 pbar 和大盤的相關略高於 MLP，所以這個假設不是完全事前。）
- C1 CNN 取代 Transformer：p_raw = (MLP 兩個成員平均 + cnn_raw_B) / 2。
- C2 CNN 當第三個 family：p_raw = (MLP 平均 + tf_raw_B + cnn_raw_B) / 3。

cnn_raw_B（訓練前固定，不調參）：每支股票最近 20 天的 B 組 67 個 feature；Conv1d(67→32, kernel 3, padding 1) → GELU → Dropout 0.2 →
Conv1d(32→32, kernel 3, padding 2, dilation 2) → GELU → Dropout 0.2；輸出用「最後一天」和「20 天平均」接起來 → Linear(64→1)。
label、loss（min(|r|, 5%) 加權 BCE）、Scaler、HP（lr 1e-3、wd 1e-2、batch 1024、最多 10 epoch、patience 2、最後 120 天 early stopping 後重訓）
都和 tf_raw_B 相同（tfmodel.train_model）。seed 1–5，年度 fold 2022–2026/05 + post（訓練到 2026-05-29）。

判斷：val 成對 t 依 Holm（3 個：2.13、1.96、1.645）且 hold 差 > 0 才採用；post 只報告。另外報告三種 pbar 和大盤的相關。

## 第五輪（2026-10-24 21:00，使用者問「用 RNN 換掉 Transformer」之後、訓練之前寫）

- R1 RNN 取代 Transformer：p_raw = (MLP 兩個成員平均 + rnn_raw_B) / 2。
- R2 RNN 當第三個 family：p_raw = (MLP 平均 + tf_raw_B + rnn_raw_B) / 3。

rnn_raw_B（訓練前固定，不調參）：GRU，2 層，hidden 32，層間 dropout 0.2；每支股票最近 20 天的 B 組 67 個 feature；
輸出用「最後一天的 hidden state」和「20 天平均」接起來 → Dropout 0.2 → Linear(64→1)。
label、loss、Scaler、HP 和 tf_raw_B 完全相同（tfmodel.train_model），seed 1–5，年度 fold 2022–2026/05 + post（訓練到 2026-05-29）。

判斷：val 成對 t 依 Holm（2 個：1.96、1.645）且 hold 差 > 0 才採用；post 只報告。另外報告 pbar 和大盤的相關。

## 第六輪（2026-10-24 21:20，使用者要求試 LightGBM 和 logistic regression，訓練之前寫）

兩種模型、兩種 label（raw = 收 > 開，給大盤觀點；resid = 贏過當天 100 支平均，給選股觀點），輸入都是 B 組 67 個：

- lgb：LightGBM binary，樣本權重 min(|報酬|, 5%)（正規化），learning_rate 0.02、num_leaves 15、min_data_in_leaf 500、
  feature_fraction 0.8、bagging_fraction 0.8（每輪）、lambda_l2 1.0、最多 1000 輪；用訓練期最後 120 天（加權 logloss）做 early stopping（100 輪），
  再用全部訓練資料重訓到最佳輪數。seed 1–5。缺值交給 LightGBM 處理。
- lr：LogisticRegression（L2，lbfgs），輸入用 tfmodel.Scaler（夾 1%/99%、標準化、缺值補 0），同樣的權重；
  C 從 {1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1} 中用訓練期最後 120 天的加權 logloss 選，再用全部訓練資料重訓。沒有隨機性，每個 fold 一個模型。
- fold：年度 2022–2026/05 + post（訓練到 2026-05-29），和其他成員相同。

大盤觀點（對照：(MLP + Transformer) / 2）：L1 (MLP + lgb) / 2、L2 (MLP + Transformer + lgb) / 3、G1 (MLP + lr) / 2、G2 (MLP + Transformer + lr) / 3。
Holm 4 個（2.24、2.13、1.96、1.645）且 hold 差 > 0。
選股觀點（對照：p_e = 兩個 MLP 殘差成員的平均）：S1 p_e = (MLP 平均 + lgb_res) / 2、S2 p_e = (MLP 平均 + lr_res) / 2。Holm 2 個（1.96、1.645）且 hold 差 > 0。
只報告：只用 lgb、只用 lr；pbar 和大盤的相關；選股觀點的每日 rank IC。

## 第七輪（2026-10-24 21:32，使用者要求 LSTM、GRU 在大盤和個股都試，訓練之前寫）

- lstm：LSTM，2 層，hidden 32，層間 dropout 0.2；每支股票最近 20 天的 B 組 67 個 feature；輸出用「最後一天」和「20 天平均」接起來 → Dropout 0.2 → Linear(64→1)。
- GRU 用第五輪同一個結構（rnn_lib）。訓練方式都和 tf_raw_B 相同（tfmodel.train_model），seed 1–5，年度 fold + post。
- 大盤觀點（raw label）新訓練 lstm_raw：K1 (MLP + LSTM) / 2、K2 (MLP + Transformer + LSTM) / 3。GRU 的大盤結果第五輪已經測過（R1、R2），這輪只並列報告。
  Holm 2 個（1.96、1.645）且 hold 差 > 0。
- 選股觀點（resid label，和 mlp_res 相同）新訓練 gru_res、lstm_res：S3 p_e = (MLP 平均 + GRU) / 2、S4 p_e = (MLP 平均 + LSTM) / 2。
  Holm 2 個（1.96、1.645）且 hold 差 > 0。
- 只報告：只用 LSTM、只用 GRU；pbar 和大盤的相關；選股的每日 rank IC。

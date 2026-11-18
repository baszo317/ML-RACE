# v13 事先寫好的規則：吃原始價量序列的 LSTM、CNN 當選股成員（2026-11-05 18:59 寫，任何樣本外成績算出來之前）

使用者的要求（11/5）：「評估量化交易 F-LOAM（LSTM）與卷積神經網路（CNN）在這比賽的可行性，訓練可以在我本地跑，寫在 V13」。
F-LOAM 是 Fast LiDAR Odometry and Mapping（機器人用的 LiDAR SLAM，Wang et al. 2021），不是交易模型；這裡照括號解讀成 LSTM。
v9、v11、W0 的檔案和每天的流程都不改。程式：`v13\research\seqlib.py`、`v13_train.py`（訓練）、`v13_eval.py`（判定）。

## 0. 寫這份規則時已經知道的

- v8（README_v8 第 12、14 節）已經用 **B 組 67 個 feature 的 20 天序列** 測過 LSTM、GRU、CNN、Transformer，全部沒採用：
  - 選股：S3 (MLP + GRU) / 2 val −1.07（t −2.02）；S4 (MLP + LSTM) / 2 val −1.36（t −2.42）。
    LSTM 自己的 rank IC 0.096 / 0.099 / 0.107（val / hold / post），選股 MLP 0.099 / 0.105 / 0.081。
  - 大盤：CNN 取代 Transformer val −0.50（t −1.06）；LSTM −1.22（t −2.38）；GRU −1.19（t −2.49）。
  - 早停多半在第 1–3 個 epoch。
- reg_svm（11/4）：exact RBF SVM 自己的 IC 0.1093（選股成員最高），和 lgb_res 相關 0.835；v11 + SVM：val +0.03、hold −0.69、post −0.30，沒通過。
- 文獻：
  - Qlib CSI300 benchmark：吃原始 60 天價量（Alpha360）時 GRU IC 0.049、LSTM 0.044，高於 LightGBM 0.040、MLP 0.029；
    吃人工 feature（Alpha158）時反過來，LightGBM 0.0475、LSTM（20 個 feature）0.031。
  - Jiang, Kelly & Xiu (2023)：K 線圖的 2D CNN 在美股週頻有效；1D CNN 直接吃用 image scaling（視窗內最低到最高縮成 0–1）的價格序列，表現和 2D 相當或更好
    （Table IX，五日報酬的 equal-weight Sharpe：5 天 7.20 對 7.15、20 天 7.88 對 6.75）。沒有 1 天或盤中的結果。
- 所以 v8 沒測過、而且文獻上有理由的是：**讓序列模型吃原始序列，自己學 feature**。這是 v13 唯一的假設。
- 檢查（都沒有看任何樣本外成績）：
  - 原始 channel 在同一 row 和 B 的對應 feature 完全相同（intra = intraday_ret、gap、foreign_net 等，最大差 0）。
    row D 放的是 D−1 的資料，所以視窗 D−60 .. D−1 全部是 09:00 前已知。
  - leak check：把 D 之後的資料換成雜訊、或整個截斷在 D，D 的預測完全相同（LSTM、CNN 都是差 0）。
- 時間：雲端 1 thread，最大 fold（149,447 row）每個 epoch LSTM 44 秒、CNN 35 秒。
  測試模式（1 epoch、2 萬 row）跑過；訓練程式不算任何樣本外指標，所以沒有看到成績。
- 正式訓練 18:57 用下面第 1 節的設定開始；這份規則在訓練進行中、任何樣本外數字算出來之前寫好，之後不改。

## 1. 成員（兩個候選）

| 項目 | 設定 | 和誰相同 |
|---|---|---|
| label | 有沒有贏過當天 100 支的平均（y_resid） | v8 的選股 LSTM / GRU、lgb_res、SVM |
| 樣本權重 | min(\|超額報酬\|, 5%)，平均 1 | 同上（tfmodel._weights） |
| 輸入 | 每支股票最近 60 個交易日（D−60 .. D−1），每天 19 個 channel，見下 | Qlib Alpha360 也是 60 天 |
| scaling | 14 個逐日 channel：訓練窗 fit 的 tfmodel.Scaler（1% / 99% winsorize、z-score、缺值補 0）；5 個視窗 channel 本來就在 −0.5 到 0.5 | |
| walk-forward | 6 個 fold（2022、2023、2024、2025、2026-01–05、post 2026-06 以後），訓練窗 2020-03-02 到 fold 前一天 | 所有成員 |
| 訓練 | AdamW lr 1e-3、weight decay 1e-2、batch 1024、dropout 0.2；最多 30 個 epoch、patience 3，用訓練窗最後 120 個交易日的加權 log-loss 早停，再用整個訓練窗重訓到最佳 epoch | 除了 epoch 上限和 patience（v8 是 10、2），其他都是 tfmodel.train_model |
| seed | 1–5，取平均 | v8 |

19 個 channel：
- 逐日 14 個：intra = log(收/開)、gap = log(開/前收)、upper = log(高/max(開,收))、lower = log(min(開,收)/低)、
  vol、trades、size = 成交量、筆數、每筆股數各自除以 20 日平均（log）、
  fnet、tnet、dsnet、dhnet = 外資、投信、自營商自行買賣、自營商避險的買賣超 / 20 日均量、fhchg = 外資持股比率的變化、
  mkt = 當天 100 支 intra 的平均、rel = intra − mkt。
- 視窗 5 個（Jiang-Kelly-Xiu 的 image scaling）：開、高、低、收用「視窗內最低的低、最高的高」縮成 0–1 再減 0.5；成交量除以視窗內最大量再減 0.5。
- 沒有 B 的多日摘要（ret20、RSI、intra_bias_250 等）、沒有美股、沒有產業代號：模型只能自己從序列學。

兩個架構（訓練前固定，不調參）：

- **LSTM-R**（kind lstm）：v8 的 LSTMClassifier 原封不動（2 層、hidden 32、層間 dropout 0.2、輸出用最後一天和 60 天平均接起來 → Linear），15,297 個參數。
- **CNN-R**（kind tcn）：1×1 conv 到 32 個 channel → 4 層 residual 1D convolution（kernel 3，dilation 1、2、4、8，涵蓋 31 天）→ GELU、dropout 0.2 →
  最後一天和 60 天平均接起來 → Linear，13,121 個參數。v8 的 CNN 只看 7 天（為 20 天視窗設計），60 天視窗改用 dilation。

不測的：
- 大盤觀點：v8 四種架構的 pbar 和大盤的相關都在 0.085–0.15，瓶頸是資訊不是架構；目前上傳的 W0 也不用大盤觀點（w_T 0）。
- 2D K 線圖 CNN：Jiang-Kelly-Xiu 自己的 Table IX 顯示 1D 加 image scaling 不輸 2D；2D 在筆電 CPU 上要貴 10–100 倍。

## 2. 判定（和 SVM、v12 同一種規則）

1. 每個候選：5 個 seed 平均 → 每天減平均 → 每天縮放成 lgb_res 當天的標準差（同 X5、v12、SVM）→
   當選股觀點的**一個新的 family**：p_e = c((MLP 部分 + LightGBM 部分 + 候選) / 3)。
   MLP 部分 = v9 的 4 個選股 MLP 平均，LightGBM 部分 = (lgb_res + X1) / 2，其他（w_T 0.5、夜盤 T2、月初 T3、λ 0.75、除權息規則）都是 v11。
   「每個 family 先內部平均、再 family 之間平均」是 ens_lib.family_mean 本來的寫法。
2. **v11 + 候選 減 v11：val > 0 → 入選。**
3. **入選的放在一起（兩個都入選時 p_e = c((MLP + LightGBM + LSTM-R + CNN-R) / 4)）：hold、post 都 > 0 → 通過。**
4. 通過才寫每天的程式（每天或每週重訓的時間另外估），上不上線由使用者決定。沒通過：v9、v11、W0 都不變。
5. 判定前先檢查重建的 v9、v11、W0 和公布的數字相同：13.99 / 14.33 / 16.59 / 14.31、14.60 / 14.34 / 20.35 / 15.02、11.79 / 10.35 / 21.71 / 12.22
   （val / hold / post / 全期）。不同就先查原因，不判定。

## 3. 只報告、不影響判定

- 每個成員自己的每日 rank IC（val / hold / post / 全期）：LSTM-R、CNN-R、v8 的 LSTM-B 和 GRU-B（B 組 20 天）、lgb_res、X1、選股 MLP、SVM。
- LSTM-R 減 LSTM-B 的每日 IC 差和 t：同一個架構、同一個 label 和訓練方式，只差輸入（還有 epoch 上限），這是「吃原始序列有沒有比較好」最乾淨的比較。
- 和現在的選股觀點、各成員的每天相關；README 第 3 節的加入條件 IC_z − ρ·IC_x 和兩個訊號的最佳權重。
- W0 + 候選、v9 + 候選；候選放進 LightGBM 部分（權重 1/6，同 SVM）；v8 的 LSTM-B、GRU-B 當 v11 的新 family；分年的差；早停 epoch 的分布；訓練時間。

## 4. 要知道的限制

- val（2022–2024）在 v5 到 v12 已經用來挑過很多改法（overfit 檢查：val 和 hold 的相關只有 0.16）。
- 沒有效果時，一個候選照第 2 節通過的機率約 1/8（三段的正負各一半）；兩個候選至少一個通過約 0.23。
- 有效果也不一定過：每日 AR sd 約 48 bp，加一個成員的每日差 sd 大約 5–10 bp，+0.3 bp/日的真實效果在 val 只有 t ≈ 1（README 第 3.4 節）。
- 只測一種輸入表示、一組架構、不調參。沒通過的意思是「這個設計在這個比賽沒有幫助」，不是「任何 LSTM、CNN 都不可能有用」。

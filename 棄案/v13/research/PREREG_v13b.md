# v13b 事先寫好的規則：F-LOAM（Liao & Lin 2025）的不偷看版本當選股成員（2026-11-05 21:34 寫，正式訓練之前；訓練由使用者在筆電跑）

使用者（11/5 21:03）：「我說的 F-LOAM 是 https://link.springer.com/article/10.1007/s44163-025-00622-0」；
21:19：「不用雲端跑，程式給我我在本地跑就好，然後要合併前面的 LSTM、CNN」。
程式：`v13\research\floamlib.py`（成員）、`v13_train.py --kinds floam`（訓練，和 LSTM-R、CNN-R 同一支）、`v13_eval.py`（三個候選一起判定）、
`floam_checks.py`（論文做法的問題，只報告）。v9、v11、W0 的檔案和每天的流程都不改。

## 0. 寫這份規則時已經知道的

- 論文：Liao, L., & Lin, J. (2025). F-LOAM: an efficient hybrid model for stock price prediction based on SVMD denoising.
  *Discover Artificial Intelligence*, 5, 347。F-LOAM = Feature-enhanced LSTM-Optuna-Augmented Model。
  - SVMD：VMD（K 7、alpha 2000）分解 8 個輸入（USD/CNY、市值、流通市值、PE、PE LYR、PB、PS、PCF），留下和原序列 |Spearman| > 0.5 的 IMF 加回去。
  - 2 層 LSTM、每層 50 個 unit，hidden state h_t 和標準化後的輸入接起來，給 LightGBM 預測。
  - Optuna 調 LightGBM（num_leaves {20, 30, 40, 50}、learning_rate [0.01, 0.1]、feature_fraction [0.8, 1]、bagging_fraction [0.8, 1]、
    bagging_freq {1, 5, 10}）和 LSTM（units、dropout、batch、epochs、learning_rate）。
  - 論文的做法：目標是收盤價本身（R² 0.98–0.999）；訓練、測試是**隨機** 80/20；Optuna 的目標是**測試集**的 MSE；
    SRCC 是對「原始序列」算的，沒有說只用訓練期；沒有寫預測幾天後、視窗多長。
- v13 第一輪（PREREG_v13.md，雲端 11/5）：LSTM-R、CNN-R 都沒入選（v11 + LSTM-R val −0.19、v11 + CNN-R val −0.02）。
- reg_svm（11/4）：exact SVM IC 0.1093，放進 LightGBM 部分 val +0.03、hold −0.69，沒通過。
- 為了確認程式能跑，雲端跑過測試模式（1 個 epoch、2 萬 row、3 次 Optuna trial、seed 1，數字沒有意義）：
  v11 + 測試版 F-LOAM val +0.56、hold −0.30、post +1.15。這是玩具模型，不代表正式結果；這份規則不因它改變。
- `floam_checks.py` 的測試（只跑 2026-10-01 到 10-22 的 15 天）：整段 SVMD 的特徵和當天盤中報酬的 rank IC +0.18（t 7.0），
  只用過去 250 天的 SVMD −0.04；收盤價水準「明天 = 今天」的 R² 中位數 0.997、隨機切分後用前後兩天內插 0.998。

## 1. 成員：F-LOAM（kind floam）

| 項目 | 設定 | 和論文的差別、理由 |
|---|---|---|
| label | 有沒有贏過當天 100 支的平均（y_resid），權重 min(\|超額報酬\|, 5%) | 論文是收盤價；比賽算的是盤中方向 |
| LSTM 輸入 | 每支股票最近 20 天的 B 組 67 個 feature（tfmodel.Scaler，訓練窗 fit） | 比賽資料沒有 PE、PB、市值、匯率 |
| SVMD | 不用 | 整段做會偷看未來（`floam_checks.py` A3）；只用過去的版本 67 個 feature × 16.5 萬 stock-day 約 1,100 萬次分解，筆電做不到 |
| LSTM | 2 層 × 50 unit（同論文）、dropout 0.2、輸出 = Linear(h_t)；AdamW lr 1e-3、weight decay 1e-2、batch 1024；最多 30 epoch、patience 3，用訓練窗最後 120 個交易日早停，**保留早停的模型，不重訓** | 論文的 batch 16–128、lr 0.01–0.1 在 CPU 上太慢；不重訓是為了讓 h 在那 120 天是樣本外 |
| LightGBM 輸入 | 第 D 天的 B 組 67 個（不標準化）+ h_t 50 個 | 同論文（原始輸入 + h_t） |
| Optuna | TPE、30 次 trial、seed = unit 的 seed；搜尋範圍 = 論文的 LightGBM 範圍；目標 = 訓練窗最後 120 個交易日的加權 log-loss | 論文用測試集的 MSE，這裡絕不看要預測的 fold |
| 其他 LightGBM 參數 | min_data_in_leaf 500、lambda_l2 1.0、最多 1000 輪、early stopping 100（= lgb_res） | 論文沒寫 |
| 最後的模型 | 用選到的參數和輪數，在整個訓練窗重訓，預測 fold | |
| walk-forward | 6 個 fold（2022、2023、2024、2025、2026-01–05、post），訓練窗 2020-03-02 到 fold 前一天，seed 1–5 平均 | 所有成員相同 |

同時存下（只報告）：p_lstm = LSTM 自己的輸出；p_fixed = 同樣的 B + h_t，用 lgb_res 的固定參數（不用 Optuna）。

## 2. 判定

1. F-LOAM（5 個 seed 平均）每天減平均、縮放成 lgb_res 當天的標準差（同 X5、v12、SVM），放進 v11 的 **LightGBM 部分**：
   (lgb_res + X1 + F-LOAM) / 3。F-LOAM 最後的模型是 LightGBM，所以和 SVM 放在同一個位置；其他都是 v11。
2. **v11 + F-LOAM 減 v11：val > 0 → 入選。**
3. 和前面的 LSTM、CNN 合併：LSTM-R、CNN-R 在 11/5 都沒入選，所以「入選的放在一起」只有 F-LOAM。
   **入選而且 hold、post 都 > 0 → 通過。**
4. 通過才寫每天的程式（每天或每週重訓的時間另外估），上不上線由使用者決定。沒通過：v9、v11、W0 都不變。
5. 判定前先檢查重建的 v9、v11、W0 和公布的數字相同（同 PREREG_v13 2.5）。
6. 使用者在筆電重跑的 LSTM-R、CNN-R 只報告；它們的判定仍是 11/5 雲端那一次。

## 3. 只報告、不影響判定

- 自己的 rank IC：F-LOAM、不用 Optuna 的 F-LOAM、F-LOAM 自己的 LSTM（2×50），和 lgb_res、X1、LSTM-B 等並列。
- 每日 IC 差：F-LOAM − lgb_res（論文說的「混合模型比單獨 LightGBM 好」）、不用 Optuna − lgb_res（h_t 的效果）、
  F-LOAM − 不用 Optuna（Optuna 的效果）、F-LOAM 的 LSTM − LSTM-B。
- 其他放法：W0 + F-LOAM、v9 + F-LOAM、F-LOAM 當新的 family（1/3）、F-LOAM 取代 lgb_res、不用 Optuna 的版本放 LightGBM 部分、
  三個候選全部一起、10/1–10/22、分年、規則的靈敏度。
- `floam_checks.py`：A1 收盤價水準的 R²、A2 隨機切分、A3 整段 SVMD 和只用過去的 SVMD（2025-01 到 2026-10）。

## 4. 要知道的限制

- Optuna 在 120 天上挑 30 組，會高估驗證期的表現（overfit 檢查定理 3）；最後的判定仍是 walk-forward 樣本外，不受影響。
- 訓練窗前段的 h_t 是 LSTM 的樣本內輸出（論文也是這樣用）；LightGBM 可能過度相信 h_t。這是照論文的設計，不另外修。
- 沒有 SVMD，也沒有論文的估值、匯率資料，所以這是「F-LOAM 的架構和調參方式」在這個比賽的版本，不是論文的完整重現。
- 規則的靈敏度和 LSTM-R 差不多：真實 +0.5 bp/日的改善也只有約 0.37 的機會通過；沒效果時約 1/8。
- 只跑這一組設定，看過結果後不換設定再判一次。

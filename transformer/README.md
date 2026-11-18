> 2026-10-23 整理：v1 的訓練與比較程式（`train_and_predict.py`、`walkforward_nn.py`、`train_best.py`）、模型（`models\`、`models_live\`）和 static 上傳檔已移到 `..\棄案\transformer\`，要用時移回來即可。這個資料夾現在保留 v2 用到的共用程式（`comp_features.py`、`tfmodel.py`、`update_data.py`、`final_feature_lists.json`）和工具（`check_submission.py`、`score_static.py`）。下面是 v1 當時的說明。

# Transformer 模型：預測收盤價 > 開盤價

資料夾放在 `D:\ML RACE\transformer\`。程式會讀取上一層的 `data\` 與 `sample_submission.csv`。

## 環境

```
pip install torch numpy pandas
```

只需要 CPU 版的 torch。在 Windows 上直接 `pip install torch` 就是 CPU 版。

## 1. Static leaderboard（2026-06 ~ 09）

```
python train_and_predict.py            # 訓練（約 25 分鐘）並輸出 CSV
python check_submission.py submission_ensemble.csv
```

- `submission_ensemble.csv`：建議上傳。它是 Transformer 與 MLP 兩者機率的平均，在 2022–2024 驗證期最好。
- `submission_transformer.csv`：只用 Transformer。
- 訓練資料只到 2026-05-29，6–9 月的資料完全沒有用來訓練或選模型。
- 只要前一個交易日有資料，就用模型預測：目前涵蓋 06-01 ~ 09-01。09-01 的美股資料少一晚，因為 global_data 只到 08-31。
- 前一天沒有資料的日期（目前是 09-02 之後）改用備援規則：該股近 250 個交易日的平均盤中報酬 > 0 就猜 1，否則猜 0。這是 2022–2024 驗證期表現最好的簡單規則。
- 之後把 9 月的資料補進 `data\` 再重跑，那些日期就會自動改用模型。只想重新預測、不重新訓練時，加上 `--use-saved`。
- 6–8 月的參考成績：用 `score_static.py` 在模型定案後算一次，沒有拿來調參。單位是每天 bp。
  - ensemble：2.72（6 月 -24.7、7 月 +30.6、8 月 +0.9）。
  - Transformer：3.41。
  - 全部猜 0：-5.48。

  2026 年 1–5 月盤中偏空，6 月起轉為偏多，模型因此在 6 月吃虧。這就是 concept drift，比賽期間要定期重新訓練。

## 2. Dynamic / live leaderboard（每天 9:00 前）

1. 把 `data\market\*.csv` 與 `data\global\global_data.csv` 更新到前一個交易日。
2. 每週重新訓練一次，把訓練期延到最新資料，例如：

   ```
   python train_and_predict.py --train-end 2026-10-23 --dates 2026-10-26 --tag 1026
   ```

3. 同一週的其他日子只做預測，不重新訓練：

   ```
   python train_and_predict.py --use-saved --train-end 2026-10-23 --dates 2026-10-27 --tag 1027
   ```

## 3. 重現模型比較

```
python walkforward_nn.py
```

約 30 分鐘。結果表（每天 bp，Top50）：

| 模型 | 驗證期 2022–24 | 保留期 2025~2026/05 |
|---|---|---|
| Transformer + MLP 平均 | 12.30 | 12.04 |
| Transformer（20 天序列） | 12.14 | 10.33 |
| MLP baseline | 11.82 | 12.62 |
| LightGBM（同 feature） | 7.22 | 10.90 |
| 全部猜 0 | 2.22 | 7.58 |

## 檔案

- `comp_features.py`：從競賽資料產生 67 個 feature；每個 feature 只用當天 09:00 前已知的資訊。
- `tfmodel.py`：Transformer、MLP、train-only 正規化、以 |盤中報酬| 加權的 BCE、early stopping。
- `train_and_predict.py`：訓練最終模型並輸出 CSV。
- `walkforward_nn.py`：重跑上面的模型比較。
- `check_submission.py`：上傳前的格式檢查。
- `score_static.py`：用資料裡已有的 label 計算 6–8 月成績，只當參考，不要拿來調參。
- `models\`：已訓練好的模型。

## 對照課程注意事項

1. **Future information in features**：
   - 個股、台股與台指期資料一律用前一交易日；美股用台股開盤前已收盤的那一晚。
   - 已和原始資料逐筆核對過。
2. **Survivor bias**：股票池用競賽給定的 Top100，不依未來表現挑股票。
3. **Look-ahead in normalization**：正規化的分位數、平均與標準差只用訓練期計算。
4. **Random split on time series**：
   - 用 walk-forward 依年份切分。
   - early stopping 用訓練期最後 120 個交易日，按時間順序切。
5. **In-sample upstream selection**：
   - feature 與模型只依 2022–2024 驗證期決定；2025~2026/05 只用來確認。
   - 6–9 月完全沒有參與選擇。
6. **Concept drift**：每週以 expanding window 重新訓練。2026 年盤中偏空特別明顯，要持續監控每日成績。

# v3：在 v2 的選股觀點加入 5 個新 feature（未採用）

> **狀態：未採用，已移到 `棄案\v3\`。** walk-forward 沒有比 v2 好（見第三節），正式版維持 v2。
> 程式用的是相對路徑：要重新使用，先把整個資料夾移回 `D:\ML RACE\v3\`。

v3 原本放在 `D:\ML RACE\v3\`，不修改 v2（`v2\`）和 v1（`transformer\`）的任何檔案。

程式沿用 v2 的 `ens_lib.py`，包括 members、訓練、兩觀點組合、除權息規則和評分，只多了三個檔案：

- `features_v3.py`：5 個新 feature 和它們的 rank。
- `ens_v3.py`：在 v2 的 panel 上加入新 feature，並新增 feature set `B+N` 與 `B+XR+N`。
- `train_and_predict_v3.py`：從 v2 複製，只改了選股觀點的兩個 members、輸出檔名和模型資料夾。

## 一、改了什麼

| | v2 | v3 |
|---|---|---|
| 大盤觀點（raw members） | MLP B、MLP B+XR、Transformer B | 相同 |
| 選股觀點（resid members） | MLP B、MLP B+XR | MLP B+N、MLP B+XR+N |
| 規則 | score = 0.5 × (pbar − 0.5) + (p_e − 當天平均)；除權息日猜 0 | 相同 |

N 是 5 個新 feature。它們都只用競賽資料，而且在 09:00 前已知：

| feature | 定義 | 依據 |
|---|---|---|
| `gap_bias_60` | 過去 60 天平均隔夜跳空 log(open / 前一天 close) | 隔夜報酬高的股票，盤中報酬偏低（Lou, Polk & Skouras 2019, tug of war） |
| `gap_bias_250` | 同上，250 天 | 同上 |
| `resid_vol_60` | 過去 60 天殘差盤中報酬（盤中報酬 − 當天 100 支平均）的標準差 | 特質波動高的股票偏弱（Ang, Hodrick, Xing & Zhang 2006） |
| `dow_bias_e` | 同一支股票過去 52 個「同星期幾」的平均殘差盤中報酬 | 同期間季節性（Keloharju, Linnainmaa & Nyberg 2016） |
| `hi52` | 前一天收盤 / 過去 250 天最高收盤 − 1 | 接近 52 週高點的股票偏強（George & Hwang 2004） |

每個 feature 都有一個 cross-sectional rank 版本（`xr_*`，當天 100 支中的百分位，中心為 0），只放進 `B+XR+N`。

## 二、篩選證據（`research\screen_candidates.py`）

IC 是每天的 Spearman 相關，對象是殘差盤中報酬，在 Top50 上計算。partial IC 是扣掉 v1 ensemble_B（walk-forward OOS）已有的選股訊號後剩下的部分，期間 2022 到 2026/05。

| feature | IC pre / val / hold / post | partial IC (t) | B 組最相關的 feature |
|---|---|---|---|
| `gap_bias_60` | −0.095 / −0.081 / −0.064 / −0.108 | −0.025 (−3.8) | gap_bias_20，0.72 |
| `gap_bias_250` | −0.106 / −0.081 / −0.077 / −0.119 | −0.025 (−3.6) | intra_bias_exp，−0.75 |
| `resid_vol_60` | −0.085 / −0.074 / −0.074 / −0.088 | −0.030 (−4.1) | ATR_pct，0.88 |
| `dow_bias_e` | +0.048 / +0.060 / +0.057 / +0.048 | +0.031 (+4.9) | intra_bias_250，0.56 |
| `hi52` | +0.044 / +0.055 / +0.055 / +0.054 | +0.033 (+4.6) | intra_bias_60，0.59 |

期間：pre = 2020/03–2021，val = 2022–24，hold = 2025–2026/05，post = 2026/06–10/22。

- 四段期間的方向都一致；只看歷年 12 月，t 值也有 3–4。
- 5 個 feature 的 rank 等權重合併（正負號只用 2020–21 決定）：
  - 單獨使用：IC val 0.096、hold 0.094。v1 的選股訊號是 0.092 / 0.088。
  - 加到 v1 的選股訊號上：IC val 0.092 → 0.099，hold 0.088 → 0.096。
- `dow_bias_e` 大部分是這支股票整體的盤中偏誤；扣掉 intra_bias_250 後，partial IC 仍有 0.021（t 3.5）。

## 三、walk-forward 驗證（`research\wf_check_v3.py`）

結論：沒有比 v2 好，未採用。完整表格在 `research\wf_check_v3.txt`。

設定：
- 和 v2 整夜實驗相同的年度 protocol：expanding window；val 2022–24、hold 2025–2026/05；post 2026/06–10/22 用凍結在 2026-05-29 的模型。
- MLP，5 seeds。大盤觀點相同，只換選股觀點。
- 單位 bp/day，Top50，含除權息規則。
- 對照：同設定下 v2 的數字（val 12.54、hold 13.74）和 `v2\README_v2.md` 的「v2 規則（MLP），w_T = 0.5」（12.52、13.96）幾乎相同，表示這次的重現沒有問題。

| 選股觀點 | w_T | val | hold | post | 全期 | t_val | t_hold | t_post |
|---|---|---|---|---|---|---|---|---|
| v2：res_B + res_xr | 0.5 | 12.54 | 13.74 | 12.13 | 12.86 | | | |
| v3：res_BN + res_xrN | 0.5 | 12.59 | 13.70 | 10.34 | 12.72 | +0.10 | −0.05 | −1.00 |
| v2 + v3 四個平均 | 0.5 | 12.70 | 13.49 | 12.61 | 12.92 | +0.55 | −0.48 | +0.37 |
| v2：res_B + res_xr | 0（只選股） | 11.36 | 11.41 | 13.52 | 11.56 | | | |
| v3：res_BN + res_xrN | 0 | 10.90 | 9.92 | 13.73 | 10.86 | −1.24 | −2.06 | +0.11 |
| v2 + v3 四個平均 | 0 | 11.16 | 10.92 | 14.88 | 11.41 | −0.80 | −0.87 | +0.94 |

t 是和同一個 w_T 的 v2 每日差的 paired t。

另外試過不放進 MLP、直接把 5 個 feature 的 rank 線性加到 v2 的選股分數上：w_T = 0.5 時 val 從 12.54 掉到 11.5–11.7（λ = 0.25 時 t = −2.1），只有 post 變好。

解讀：
- w_T = 0.5 時，差異都在雜訊內（|t| < 1）；只看選股（w_T = 0）時，v3 反而變差（hold t = −2.06）。
- 篩選時的 partial IC 是相對 v1 的選股訊號算的。v2 的殘差模型加上 30 個 rank feature，已經吸收了這些資訊：新 feature 和 ATR_pct、gap_bias_20、intra_bias_* 的相關係數有 0.56–0.88。
- 依原本的規則（val 決定、hold 確認），沒有理由換掉 v2。post 的改善在 val 沒有出現，而且篩選時看過 post，不能當證據。

## 四、怎麼跑

第一次使用，先訓練（MLP 4 組 × 10 seeds，加 Transformer 5 seeds）：

```
python train_and_predict_v3.py --train-only
```

每天早上（9:00 前）雙擊 `daily_v3.bat`，或依序執行：

```
python ..\transformer\update_data.py        把當天下載的 zip 併入 data\
python ..\v2\fetch_exdiv.py                 更新除權息預告（和 v2 共用）
python train_and_predict_v3.py --dates auto MLP 每天重訓，Transformer 每 5 個交易日重訓，然後預測
python monitor_live.py                      已實現成績，拆成 timing 與 xsec
```

上傳 `out\submission_v3_MMDD.csv`。選項和 v2 完全相同，例如 `--retrain none`、`--w-t`、`--no-exdiv`、`--static`、`--tf-resid`。

v3 的模型存在 `models_v3\`，不會覆蓋 v2 的 `models_v2\`，兩版可以同時跑、同時比較。

## 五、避免 leakage

- 第 D 列的新 feature 只用 D−1 以前的資料：
  - 同一支股票往前 shift 一列。
  - `dow_bias_e` 只用之前同星期幾的日子。
- 殘差報酬（減掉當天 100 支平均）只用在過去的日子；預測日本身沒有 label。
- 正規化沿用 v1，只用訓練期資料。
- `features_v3.py` 算出的值和篩選程式逐筆相同，最大差 < 1e-15。
- 誠實說明：這 5 個是我看過 2020–2026/10 全部資料後挑的。方向符合文獻、四段期間也一致，但 walk-forward 的 val、hold 已經不是完全乾淨的驗證；真正乾淨的是接下來的 live leaderboard。

## 六、檔案

| 檔案 | 用途 |
|---|---|
| `features_v3.py` | 5 個新 feature 與 rank |
| `ens_v3.py` | v2 的 `ens_lib` 加上新 feature 和新的 feature set |
| `train_and_predict_v3.py` | 正式訓練與預測（live、static） |
| `monitor_live.py` | 已實現成績（v2 同名檔的複本，讀 `v3\out\`） |
| `daily_v3.bat` | 每天早上一鍵執行 |
| `research\screen_candidates.py` | 候選 feature 篩選（第二節的數字），輸出在 `research\candidate_screen.txt` |
| `research\wf_check_v3.py` | v2 對 v3 的 walk-forward 比較（第三節的數字），輸出在 `research\wf_check_v3.txt` |

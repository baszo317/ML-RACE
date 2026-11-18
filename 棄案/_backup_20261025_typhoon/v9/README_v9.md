# v9：把所有「val t 值為正」的改法混成一個模型

整理日期：2026-10-24；2026-10-25 新增第八、九、十節。**v9 已上線**：每天晚上（課程 zip 約 17:00 釋出後）執行 `v9\evening.bat`，早上（05:00 以後、9:00 以前）執行 `v9\daily.bat`，上傳 `v9\out\submission_v9_MMDD.csv`（第九、十節）。v1–v8 的模型與結果都沒有動，`v5\daily.bat` 留作緊急備案；10/25 為了上線改過的共用程式見第十節。

## 〇、簡短說明

- 使用者的想法：與其什麼都不改，不如把所有測過、t 值為正的改法都疊在一起。
- 我翻過 v1–v8 和棄案的全部結果，依事先寫好的規則（`research\PREREG_v9.md`）納入 11 項，排除 15 項以上（第一節）。挑選**只看 val**，所以 hold、post 是檢驗。
- 結果（walk-forward，bp/日）：

| | val | hold | post | 全期 | 每日 sd | P(19 天 > 0) | static | 10 月 |
|---|---|---|---|---|---|---|---|---|
| 現在（v5） | 13.22 | 13.95 | 12.63 | 13.38 | 49.6 | 0.91 | 12.16 | 15.26 |
| **v9** | 13.99 | 14.33 | 16.59 | 14.31 | 48.3 | 0.93 | 15.46 | 22.91 |
| 差（t） | +0.77（1.17） | +0.39（0.38） | +3.96（1.50） | +0.93 | | | | |

- **事先寫好的檢驗通過**：hold、post 兩段都 > 0。這是這一路測下來第一個三段都贏現在設定的版本，而且波動沒有變大。
- 但每一段都還在雜訊內（hold t 0.38、post t 1.50），val 的 +0.77 因為是挑選用的資料，會高估。
- 拆開看（第四節）：好處主要來自選股觀點（LightGBM、v3 的 feature 成員）和除權息前後日規則；大盤觀點的 4 項改法幾乎沒有貢獻。
- 原本的建議是 11/23 前和 v5 並行、只上傳 v5（第六節）。使用者 2026-10-25 決定直接上線 v9，每天的程式見第九節。
- rolling static（每天只用前一個交易日以前的資料訓練，和 v4 上傳的 static 同一種作法）：v9 16.42 bp/日，v4 上傳的 15.36、同組 V3 的 20.31；10 月延伸 21.00。84 天裡這些差距都在雜訊內（v9 − v4 t 0.33，v9 − V3 t −0.55）（第八節）。

## 一、納入與排除

### 1.1 納入的 11 項（val t > 0、合規、live 做得到；互斥的只留 val t 最高的）

| # | 改法 | 來源 | val 差（t） | hold 差 | post 差 |
|---|---|---|---|---|---|
| 1 | 夜盤 feature（night_ret、night_range）加進兩個大盤 MLP | v7 H2 | +0.11（0.51） | +0.67 | +0.90 |
| 2 | 選股 mlp_res_B 用擴充股票池訓練 | v7 H1 | +0.07（0.16） | +0.53 | +0.86 |
| 3 | 選股加入 LightGBM family | v8 S1 | +0.36（0.69） | +0.61 | +2.00 |
| 4 | 大盤加 4 年 window 的 MLP | v5 | +0.05（0.82） | +0.08 | −0.60 |
| 5 | σ 當輸入、權重 ÷ σ^0.5 | v5 | +0.39（0.94） | −0.21 | −1.05 |
| 6 | 選股加 v3 的 5 個 feature 成員（四個平均） | v3 | +0.16（0.55） | −0.25 | +0.48 |
| 7 | 拿掉那斯達克 3 個 feature | v5 | +0.05（0.12） | −0.64 | −0.38 |
| 8 | 除權息前一天猜 0 | v7 | +0.14（1.19） | −0.04 | +1.17 |
| 9 | 除權息後一天猜 0 | v7 | +0.19（1.96） | −0.03 | −0.02 |
| 10 | 除權息後兩天猜 0 | v7 | +0.11（1.15） | −0.01 | +0.25 |
| 11 | beta 比例對沖 λ = 0.75 | v8 H6b | +0.28（0.71） | −0.60 | +0.73 |

### 1.2 排除

| 改法 | 來源 | val 差（t） | 為什麼排除 |
|---|---|---|---|
| σ 權重 α = 1 | v5 | +0.25（0.54） | 和 α = 0.5 互斥，t 較低 |
| v3 feature 成員「取代」原本成員 | v3 | +0.05（0.10） | 和「四個平均」互斥，t 較低 |
| beta 對沖 λ = 0.5 | v8 | +0.43（0.64） | 和 λ = 0.75 互斥，t 較低 |
| 每季、每月重訓 | v1、v5 | +0.50、+0.99（1.40） | 已包含（v4 起每天重訓） |
| Transformer 用 F 組（含外部資料） | v1 | +0.30 | 外部資料只到 2026/07，live 拿不到 |
| MSCI 調整日猜 0 | v4 | 平均 −36 bp（t −1.7） | 歷史日期是用成交量事後找出來的，回測有偷看 |
| 台灣 50 調整日 | v5 | +3.8 bp（t 0.24） | 沒有個股層級的歷年名單 |
| 真實開盤價、試撮價 | v6、v7 | +0.11 | 不合規（9:00 才知道） |
| 月初加大盤曝險 | v8 | 約 +0.2 | 沒有正式做 val t 檢定 |
| w_T 0.25 / 0.75 / 1.0 | v5 | −1.17 / −0.13 / −0.65 | val t < 0 |
| 日頻大盤模型、ridge、beta 一致規則、beta 調整殘差 | v2、v8 | −0.02 到 −1.68 | val t < 0 |
| CNN、GRU、LSTM、只用 Transformer、LightGBM／logistic 進大盤 | v8 | −0.10 到 −1.46 | val t < 0 |
| logistic、GRU、LSTM、Transformer 進選股 | v2、v8 | −1.07 到 −1.36 | val t < 0 |
| 殘差 Transformer、更多 seed、margin、breadth、F 組 MLP、recency、日曆 feature、產業中性、bp 規則、全部對沖 | v1–v8 | < 0 | val t < 0 |

## 二、v9 的組成

- 大盤觀點（label：收 > 開）：
  - mlp_raw_B9：B − 那斯達克 + 夜盤 + σ（absr_60、absres_60）。
  - mlp_raw_xr9：同上再加 XR。
  - mlp_raw_w4_9：同 mlp_raw_B9，只用最近 4 年訓練。
  - tf_raw_B：不變。
  - 三個 MLP 的權重都是 min(|r|, 5%) ÷ σ^0.5，seed 1–10。p_raw = (三個 MLP 平均 + Transformer) / 2。
- 選股觀點（label：贏過當天 100 支平均）：
  - mlp_res_B9：B − 那斯達克 + σ，用擴充股票池訓練。
  - mlp_res_xr9：B + XR − 那斯達克 + σ。
  - mlp_res_BN9、mlp_res_xrN9：再加 v3 的 5 個 feature（與 rank）。
  - lgb_res：LightGBM。
  - 四個 MLP 的權重都 ÷ σ^0.5，seed 1–10。p_e = (四個 MLP 平均 + LightGBM) / 2。
- 規則：0.5 × (pbar − 0.5) + (p_e − 當天平均) > 0 猜 1；除權息當天、前一天、後一天、後兩天猜 0；最後做 beta 對沖 λ = 0.75。
- 訓練：和 v4 相同（年度 fold，early stopping 在最後 120 天，ens_lib.HP）。新訓練 420 個 MLP（7 個成員 × 6 個 fold × 10 個 seed），雲端 2 核 19 分鐘。

## 三、結果（`results\v9_eval.txt`）

見第〇節的表。事先寫好的檢驗：挑選只用 val，hold +0.39、post +3.96，兩段都 > 0，通過。

- 其他：每日 sd 49.6 → 48.3；P(19 天 > 0) 0.91 → 0.93；P(19 天 > 20) 0.24 → 0.28。
- 雜訊：hold t 0.38、post t 1.50，單看任何一段都不顯著；三段同號是這個組合最主要的證據。

## 四、拆解（不重訓的 ablation，只報告，不拿來改 v9）

| 設定 | val | hold | post | 全期 | 每日 sd |
|---|---|---|---|---|---|
| 現在（v5） | 13.22 | 13.95 | 12.63 | 13.38 | 49.6 |
| v9 | 13.99 | 14.33 | 16.59 | 14.31 | 48.3 |
| v9 不對沖 | 14.32 | 14.59 | 16.50 | 14.58 | 51.1 |
| v9 不用除權息前後日規則 | 13.54 | 14.33 | 15.25 | 13.91 | 48.7 |
| v9 不用 LightGBM | 13.90 | 12.12 | 14.51 | 13.44 | 47.0 |
| v9 不用 4 年 window MLP | 14.20 | 14.16 | 17.64 | 14.48 | 48.2 |
| v9 不用 v3 成員 | 13.69 | 14.16 | 15.68 | 13.99 | 48.2 |
| 只換大盤觀點（其他照現在） | 12.96 | 13.98 | 12.72 | 13.24 | 50.1 |
| 只換選股觀點（其他照現在） | 13.72 | 14.57 | 15.45 | 14.11 | 51.8 |
| 只換規則（除權息前後日 + 對沖） | 13.92 | 13.49 | 14.61 | 13.85 | 46.7 |
| 換成員、規則照現在 | 13.85 | 14.58 | 14.76 | 14.14 | 51.6 |

- 選股觀點是主要來源（只換選股：三段 +0.50 / +0.62 / +2.81）。
- 大盤觀點的改法（夜盤、σ、拿掉那斯達克、4 年 window）幾乎沒有貢獻（只換大盤：−0.26 / +0.03 / +0.09）。
- LightGBM 最重要：拿掉後 hold 從 14.33 掉到 12.12，比現在還差 1.83（t −2.05）。也就是說，MLP 那一側的改動在 hold 期間其實是負的，靠 LightGBM 補回來。這是組合裡最大的風險：效果很依賴 LightGBM 這一個元件。
- 除權息前後日規則全期 +0.40；v3 成員 +0.32。
- 對沖 λ = 0.75 全期 −0.27，換到每日 sd −2.9（第十一節說的風險設定）。
- 4 年 window MLP 全期 −0.17。
- **不要照這張表再刪元件**：這是看過 hold、post 之後的結果，照它挑等於拿檢驗資料來選，會再高估一次。

## 五、誠實說明

- val 是挑選用的資料，+0.77 會高估。
- hold、post 在之前的版本裡被看過，例如對沖的 λ、LightGBM 的結論都看過 post。所以這兩段只是「比較乾淨」，不是完全乾淨。
- 唯一乾淨的資料是之後的 live（10/26 起）。

## 六、上線前要做的事與建議（2026-10-24 寫；10-25 已全部完成，見第九節）

上線需要的東西：

1. **夜盤資料要在 9:00 前拿得到**：用 v7 的 `fetch_night.bat`（FinMind）。上線前找一天 8:30 實際跑一次，確認有當天日期的 after_market 資料。拿不到時，夜盤 feature 是缺值，模型會用訓練平均（等於沒有夜盤），不會壞掉；而且大盤觀點的改法本來就幾乎沒貢獻。
2. `pip install lightgbm`。
3. 擴充股票池：只用來訓練 mlp_res_B9，用 `棄案\data` 到 2026/07 的歷史即可，不用每天抓。
4. 除權息前一天規則需要「明天的除權息」：`fetch_exdiv.py` 每天抓的預告已經有。
5. 每天的訓練多了 5 個 MLP 成員和 LightGBM：筆電上估計多 3–5 分鐘。

建議流程：

1. 寫 `v9\train_and_predict_v9.py` 和 `v9\daily.bat`（沿用 v5 的 rolling pool 與 guards）。
2. 10/26 到 11/20：每天照常上傳 v5，同時跑 v9 但不上傳，用 `v8\monitor_market.py --sub` 比較兩者的 live 成績。
3. 11/23：用 live 的比較加上 w_T × 對沖的模擬，決定私有測試期用 v5 還是 v9。

## 七、檔案

| 檔案 | 用途 |
|---|---|
| `research\PREREG_v9.md` | 納入規則、組成、檢驗方式（訓練前寫） |
| `research\v9lib.py` | panel（夜盤、σ、v3 feature、擴充股票池）、成員定義 |
| `research\v9_train.py` | 420 個 MLP 的 walk-forward 訓練（要 torch） |
| `research\v9_eval.py` | 評估與 ablation |
| `results\v9_train.log`、`results\v9_eval.txt` | 輸出 |
| `cache\wf9_preds.csv.gz` | 7 個新成員的 walk-forward 預測（seed 平均），重算不用重訓 |
| `daily.bat` | 每天早上（第九節）：合併 zip、除權息預告、夜盤、訓練、預測、檢查、成績 |
| `train_and_predict_v9.py` | 每天預測（`--dates auto`）與 rolling static（`--static`） |
| `rolling9.py` | 成員、panel、訓練、模型池、兩個觀點、除權息規則、對沖 |
| `monitor_v9.py` | live 成績（v8 monitor_market 讀 `out\history_predictions.csv`） |
| `run_static.bat` | 用 `out\members_v9_static.csv.gz` 重組 static（幾秒，不訓練） |
| `models_v9\` | live 模型池（雲端預先訓練到 10/22，之後每天加新的） |
| `out\submission_v9_static.csv`、`probabilities_v9_static.csv`、`static_report_v9.txt` | 第八節的 rolling static |
| `out\members_v9_static.csv.gz` | rolling static 每個成員的模型池平均機率（含只用 5/29 模型的 frozen 版） |
| `results\v9_static_run.log` | rolling static 的訓練紀錄 |

重算：`cd research` 後 `python v9_eval.py > ..\results\v9_eval.txt`（需要 `v8\cache\` 的 wf_members、tab_preds、beta_panel，第一次會自動建立 panel）。

## 八、static 測試（rolling，2026-10-25）

作法和 v4 上傳的 static 一樣：每一天只用「前一個交易日（含）以前」的資料訓練的模型預測，6/1 用到 5/29、6/2 用到 6/1，以此類推，一路跑到資料最新的一天 10/22。

- 為什麼不是「用到 10 月的資料訓練一個模型，再去預測 6–9 月」：那樣 6–9 月的答案就在訓練資料裡（課程規則：用 6/1 預測 6/2，不能反過來），分數會虛高，也不能拿來上傳。rolling 版本到最後一天自然用到 10/21 為止的全部資料。
- 模型池（同 v4）：7 個 MLP 成員每天各加 2 個 seed、Transformer 每 5 天加 1 個、LightGBM 每天加 1 個；每天平均最近 5 個訓練點的模型。第一個訓練點（5/29）MLP 10 個 seed、Transformer 和 LightGBM 各 5 個。
- 共 1569 個模型（812 個訓練點），雲端 2 核約 90 分鐘。Transformer 6–9 月沿用 v4 static 的模型檔（同一個成員、同一個時程），10 月新訓練 3 個。

結果（Top50，bp/日；`out\static_report_v9.txt`）：

| | static 6/1–9/30（84 天） | 10 月（15 天，延伸） |
|---|---|---|
| **v9 rolling（`out\submission_v9_static.csv`）** | **16.42** | **21.00** |
| v9 rolling，對沖前 | 15.69 | 22.88 |
| v9 frozen（全部模型只訓練到 5/29） | 16.54 | 20.84 |
| v4 上傳（rolling） | 15.36 | — |
| 同組 V3（submission-1-phase1_2.csv） | 20.31 | — |

| 月 | v9 | v4 | 同組 V3 |
|---|---|---|---|
| 6 月 | 9.08 | 7.71 | 19.44 |
| 7 月 | 30.58 | 29.02 | 22.10 |
| 8 月 | −1.80 | −4.44 | 10.96 |
| 9 月 | 27.69 | 29.15 | 29.09 |
| 10 月 | 21.00 | — | — |

解讀：
- v9 比 v4 多 1.06 bp/日（t 0.33），比同組 V3 少 3.89（t −0.55）。84 天裡兩個差距都在雜訊內（每日差的 sd 分別是 29 和 65）。
- 雜訊有多大：4 年 window 成員的起點差兩天（研究版從 2022-06-01、上線版從 2022-05-30），機率最多只差 0.036，就讓 frozen static 從 15.46（第〇節）變成 16.54，99 天裡有 20 天的決策不同。所以 static 差 1 bp 不能用來判斷好壞。
- 同組 V3 的每日 sd 是 50，比 v9 的 72.5、v4 的 81.8 低：V3 的大盤權重是 0.25（我們是 0.5），押大盤方向押得比較輕。static 期間 v9 的 timing −3.20（大盤方向押錯）、xsec 19.62；V3 主要贏在 6 月和 8 月。
- 10 月 v9 21.00，walk-forward 研究版是 22.91（第〇節）。

檢查：
- 第一個訓練點（5/29）的 6 個 MLP 成員、LightGBM，和 walk-forward 研究（第三節）的 post fold 預測相差 < 2e-7，表示上線程式和研究用的是同一套訓練。4 年 window 的成員最多差 0.036，因為研究版的 window 從 2022-06-01 算、上線版從 2022-05-30 算（訓練資料差兩天）；這個成員換成研究版的預測，frozen static 就回到 15.46，和第〇節完全一樣。
- 上線程式的兩個觀點、除權息規則和對沖，用研究版的成員預測重算 2025-01 以後 437 天，每天的分數和 `research\v9_eval.py` 完全相同（差 < 1e-13）。
- Transformer 6/1 的預測和 v4 上傳的 static 檔相同（差 5e-7）。

### 8.1 大盤權重 w = 0.25（同組 V3 的設定）會怎樣（2026-10-25）

| | w = 0.5（上線設定） | w = 0.25 |
|---|---|---|
| rolling static 6/1–9/30 | 16.42 | 18.84 |
| 10 月（15 天） | 21.00 | 26.24 |
| static，對沖前 | 15.69 | 15.18 |
| static，frozen（只用 5/29 的模型） | 16.54 | 15.56 |
| walk-forward val 2022–24 | 13.99 | 12.92 |
| walk-forward hold 2025-01–2026-05 | 14.33 | 12.89 |
| walk-forward post 2026-06– | 16.59 | 16.50 |
| walk-forward 全期 | 14.31 | 13.21 |
| 每日 sd（walk-forward） | 48.3 | 43.5 |
| P(19 天平均 > 0) / > 20 | 0.926 / 0.283 | 0.920 / 0.222 |

- rolling static 上 w = 0.25 多 2.41（t 0.95），但 95% 的預測和 w = 0.5 相同；對沖前、frozen 兩個版本反而是 w = 0.25 比較低。這 2.41 是對沖和 rolling 在這 84 天剛好碰在一起，不穩。
- 五年 walk-forward 每一段都是 w = 0.25 比較差：val −1.07（t −1.84）、hold −1.44（t −1.46）、post −0.09，全期 −1.10（t −2.19），和 v5 的結論一樣（README_v5：−1.17）。波動變小，但 19 天平均 > 0 的機率沒有變好，> 20 的機率從 0.28 掉到 0.22。
- static 看起來比較好，是因為 6–9 月押大盤是虧的（w = 0.5 的 timing −3.20），押得少就虧得少，和 V3 贏的原因相同。
- 結論：live 維持 w = 0.5。看 static 分數來挑 w，就是用 static 的答案挑設定。
- 使用者決定 static 上傳 w = 0.25 的版本：`out\submission_v9_static_w025.csv`（18.84，報告 `out\static_report_v9_static_w025.txt`）。w = 0.5 的 `out\submission_v9_static.csv` 保留。重做：`python train_and_predict_v9.py --static --from-members out\members_v9_static.csv.gz --w-t 0.25 --tag static_w025`。live（`daily.bat`）不受影響，仍是 w = 0.5。

## 九、每天怎麼跑（live）

每天早上 9:00 前執行 `v9\daily.bat`（取代 `v5\daily.bat`），上傳 `v9\out\submission_v9_MMDD.csv`。

daily.bat 依序做：

1. 缺 lightgbm 或 pyarrow 時自動 `pip install`（只有第一次）。
2. `..\transformer\update_data.py`：合併今天的 zip（和 v5 相同）。
3. `..\v5\fetch_exdiv.py`：除權息預告（除權息前一天規則需要知道明天誰除權息）。
4. `..\v7\fetch\fetch_night_session.py`：台指期夜盤（FinMind，約 1 分鐘）。失敗不會停。
5. `train_and_predict_v9.py --dates auto`：
   - 檢查（沿用 v5 guards）：最新資料是前一個交易日、除權息預告今天抓過、feature 沒缺；任何 FATAL 就不寫檔。
   - 訓練：7 個 MLP 成員各 2 個 seed、1 個 Transformer、1 個 LightGBM，都只用到前一個交易日。雲端（2 核）實測 4.5 分鐘：Transformer 2.3 分鐘最久，用擴充股票池的 MLP 36 秒，其他 MLP 每個成員約 10 秒，LightGBM 17 秒。筆電實測（10/25）約 7 分鐘，見 10.1。
   - 預測：每個成員平均最近 5 個訓練點的模型 → 兩個觀點 → 除權息當天、前一天、後一天、後兩天猜 0 → beta 對沖 λ 0.75。
   - 輸出：`submission_v9_MMDD.csv`（上傳）、`probabilities_v9_MMDD.csv`、`checks_v9_MMDD.txt`、`history_predictions.csv`。
6. `monitor_v9.py`：到目前為止每個 live 日的成績（AR、timing、beta_bet、rest）。

其他：
- 模型池已經在雲端預先訓練到 10/22，放在 `v9\models_v9\`，所以第一次跑不用從頭訓練。下一個交易日是 10/27（10/26 是光復節補假，依 `data\ref\calendar.parquet`）。
- 我在雲端用 10/23 早上拿得到的資料（到 10/22）完整模擬跑了一次 live：訓練 10/22 的新模型、預測 10/23，檢查全部通過（雲端沒有除權息抓取紀錄，所以加了 `--force`）。10/23 的大盤觀點 pbar 0.513，猜 1 的比例 0.76。`out\submission_v9_1023cloud.csv` 只是測試，不要上傳。
- 夜盤：如果 8:30 時 FinMind 還沒有當天的夜盤，checks 會出現 WARN，夜盤 feature 當缺值（等於訓練平均）處理，不會停；第四節顯示大盤觀點的改法本來就幾乎沒貢獻。
- 夜盤的另一個風險：FinMind 可能先有 D 日的夜盤、還沒有 D−1 日的日盤（10/24 抓的資料裡還沒有 10/23 的日盤）。這時 night_ret 會跨兩天，所以程式改成當缺值。2020–2026 的歷史資料沒有發生過這種情況，研究結果不受影響。
- 除權息前一天：panel 沒有明天那一列，所以用官方交易日曆找下一個交易日，再查除權息預告。
- 時間不夠：`python train_and_predict_v9.py --dates auto --retrain none`（不訓練，只用已有的模型，約 1 分鐘）。
- 緊急備案：`v5\daily.bat`（v5 完全沒動）。
- 看成績：`python monitor_v9.py`；和別的檔案比：`python monitor_v9.py --sub out\submission_v9_*.csv`。

## 十、上線前的修改（2026-10-25）

### 10.1 筆電實測
- 用 10/23 的資料，把模型池裡 10/22 的模型拿掉後重訓（`models_dry\`、`out_dryrun\`）：全部檢查 OK；預測和雲端模擬的 `out\submission_v9_1023cloud.csv` 完全相同，每個機率最多差 1e-6。
- 時間：建 panel 約 1 分鐘；3 個大盤 MLP 共 21 秒；Transformer 3 分 27 秒；擴充股票池的 MLP 43 秒；其他 3 個選股 MLP 共 29 秒；LightGBM 32 秒。整個 `train_and_predict_v9.py` 約 7 分鐘，加上兩個下載約 7.5 分鐘。
- 環境：Windows、Python 3.12、torch 2.14.1+cpu。之後每次執行都會把套件版本寫在 checks 檔第一行。
- `models_dry\`、`out_dryrun\` 只是測試，可以刪。

### 10.2 晚上、早上兩段（課程 zip 約 17:00 收盤後釋出，隔天 9:00 前上傳）
- `evening.bat`（新增）：update_data → fetch_exdiv → `train_and_predict_v9.py --dates auto --prelim` → `backup_v9.py`。訓練當天的新模型，寫出預備檔 `out\submission_v9_MMDD_prelim.csv`。夜盤 05:00 才收盤，所以預備檔把夜盤當缺值（checks 是 INFO，不是 WARN）。
- `daily.bat`（早上）：步驟和以前一樣。模型昨晚已經訓練好，`--retrain auto` 會自動跳過訓練、只預測，約 2 分鐘。寫完會列出和昨晚的預備檔差幾支（也記在 checks 檔最後一行）。晚上沒跑也沒關係，早上會自己訓練（約 7 分鐘）。
- `history_predictions.csv` 多一個 column `run`（prelim 或 final），早上的 final 會取代同一天的 prelim。
- 預備檔和正式檔只差夜盤；第四節顯示大盤觀點的改法（含夜盤）幾乎沒有貢獻，所以趕不及時上傳預備檔損失很小。網站若允許重新上傳，可以晚上先傳預備檔當保險。

### 10.3 新的防呆
- 要預測的日子已經過去（用到舊 zip）：FATAL，在訓練之前就停。用舊資料測試時要加 `--force --out 資料夾`。
- 除權息預告今天下載失敗：最近 3 個交易日內有成功過就只給 WARN（預告提前數週公告，`exdiv_upcoming.csv` 只增不減），超過 3 個交易日才 FATAL。
- `transformer\update_data.py`：
  - 會停（exit code 1，daily.bat、evening.bat 都會停下）：下載內容改動了 data\ 裡已有日子的還原價（還原基準變了，新舊 rows 接不起來）、新 rows 的 y 和 adj_close > adj_open 不一致、zip 少了 Top100 的檔。確認是誤報後，在 cmd 執行 `daily.bat accept`（或 `evening.bat accept`）。
  - 只提醒：最新一天 SMA 對不上（歷史資料裡有 23 支股票、共 971 rows 對不上，多半是連續約 28 天，像是中間少了一天）、global 少了某天的 row（2020 年以來有 112 天，comp_features 會沿用前一個 row）、可能的缺日。
  - 每個用過的 zip 都留在 `..\downloads\archive\data_<預測日>.zip`；同一天若有不同的 zip，存成 `_2`、`_3`。
- `..\v7\fetch\fetch_night_session.py`：新下載的筆數比舊檔少 1% 以上，或最新的日盤、夜盤日期比舊檔早，就不覆寫（exit code 1；daily.bat 會繼續，夜盤當缺值）。先寫暫存檔再取代，每天在 `..\v7\data\archive\` 留一份 gzip 快照。

### 10.4 依賴整理
- `棄案\v3\features_v3.py` 複製到 `v9\features_v3.py`，`v7\cache\panel_ext.parquet` 複製到 `v9\data\panel_ext.parquet`（md5 相同，原檔保留）。`rolling9.py` 先找 v9 自己的副本，找不到才用舊位置；每天的流程不再讀棄案。
- 研究程式 `research\` 仍會讀 `棄案\v3`、`v7\cache`、`v8\cache`，要重算研究結果時請保留。

### 10.5 備份與紀錄
- `backup_v9.py`：把 `D:\ML RACE`（不含棄案和研究用的 cache）複製到 `%USERPROFILE%\ML_RACE_backup`（C 槽，和 D 槽是不同的硬碟），只複製新的或改過的檔、不會刪檔。evening.bat 最後會跑。第一次約 0.3 GB，之後幾秒。改成隨身碟或私人雲端資料夾：`python backup_v9.py E:\ML_RACE_backup`。
- `..\leaderboard\`：每天存 daily、accumulated board 的截圖或表，11/23 定版要用。

### 10.6 驗證
- static 重組（`--static --from-members`）改完後的輸出，和原本的上傳檔逐 byte 相同（w 0.25、w 0.5 都是）。
- live 流程在 VM（pandas 2.3.3）用假的模型機率模擬：舊 zip 會被擋下；加 `--force` 後 prelim、final 都正確寫出，final 會和 prelim 比對。
- 除權息檢查：今天成功、昨晚成功今早失敗、3 個交易日前成功、4 個交易日前成功、從沒成功，五種情形結果都符合設計。
- update_data：用改過還原價、改過 y、少一支股票的三種 zip 測試，都正確停下，`--accept` 可以放行。
- 夜盤：用只下載到一部分的資料測試，舊檔保留不動。
- 修改前的原始檔在 `..\_backup_20261025\`，要還原就複製回原位置。

### 10.7 觀察：FinMind 的夜盤資料會晚到
10/25（週日）中午抓到的 `tx_daily.csv` 仍然只到 10/22 的日盤和 10/23 的夜盤，週五 10/23 的日盤還沒出來。所以早上 8:30 時夜盤很可能常常還拿不到；這時 v9 會把夜盤當缺值（WARN，不會停）。每天 checks 檔的「TX night session present／no TX night session」會記下實際情形。

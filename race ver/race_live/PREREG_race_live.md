# race 版：每天重訓的 walk-forward 與每天的程式（事先寫好的規則）

寫於 2026-11-16 01:40–02:00（台北），在 walk-forward 的任何成績之前。第一部分（`..\race_std\research\`）、第二部分
（`..\race_std\part2\`）的檔案都不改；這裡只 import 它們的函式。

使用者的要求（11/16 01:38）：「在上午 2 點時開始幫我測 walk-forward，跟將晚上抓 FinMind、早上抓 Yahoo、算 feature、訓練、輸出上傳檔
弄完」。

## 0. V 仍然不打開

walk-forward 只跑 D（2022-01-03 到 2024-12-31），和第二部分相同；V 在 11/15 23:55 已用過一次。真正的前瞻檢驗是 live（11/16 起）。

## 1. race 版每天怎麼跑（live 和 walk-forward 共用同一套程式 `livelib.py`）

- feature：陳聖翰 V5 的 169 個（C 組），用第一部分 `build_panel.py` 的函式重算；爬的資料放在 `race_live\ext\`（從 `race_std\ext\`
  複製後每天增量更新），不動第一部分的檔案。欄位順序必須和第一部分 `features.json` 的 C 完全相同（程式會檢查）。
- 模型：第一部分的 LU（LightGBM binary，target 1{e > 0}，權重 min(|e|, 5%)，lr 0.02、15 葉、min_data_in_leaf 500、feature / bagging
  0.8、λ_l2 1；最後 120 天 early stopping 再整個訓練窗重訓）。`livelib.train_model` 用同樣的訓練窗和 seed 時，預測必須和第一部分
  `C.LU` 的 unit 逐位相同（程式會檢查）。
- 每天重訓（rolling pool，= Baszo v9 的做法，pool 3）：每個交易日 d 收盤後用「d 以前（含 d）有 label 的 row」訓練一個模型，
  seed = d 的 YYYYMMDD；預測 t 時用 t 之前最近 3 個交易日訓練的 3 個模型的機率平均（3 個模型，和第一部分的 3 個 seed 一樣多）。
- 決策：Top50 依機率前 25 支猜 1（同分依代號），除權息當天、前一天、後一天、後兩天改 0。不加大盤觀點、不對沖（對沖留給 11/23）。

## 2. walk-forward（WF-daily）

- 每個交易日 d（2021-12-28 到 2024-12-30）訓練一個模型，預測它會用到的後 3 個交易日；每天的分數照第 1 節合成、決策、計分。
- 順序：先跑 2024，再 2023，再 2022（每一年需要前一年最後 3 個交易日的模型）。**08:00 前沒跑完的年份不納入判定**（停止規則和成績無關）。
- 比較（同樣的日子、每天的成對差、Newey–West t 5 lags）：對第一部分的 SD（年度重訓、3 個 seed）、第二部分的 QR（每季）、MR（每月）。
- **判定**：live 採用 WF-daily，除非它在已完成的年份比 MR 差超過 1 個標準誤（成對差 < −SE）；那樣就改用每月重訓、3 個 seed。
- 只報告：每年的差、每日 sd、19 天平均 < 0 的比例、每天換掉幾支、每個模型的最佳輪數。

## 3. 每天的程式（Windows，需要網路的步驟只能在 Windows 跑）

| 時間 | 檔案 | 做什麼 |
|---|---|---|
| 晚上（21:30 以後，融資券、借券、當沖公布之後） | `race_evening.bat` | 合併課程 zip（Baszo 的 `transformer\update_data.py`）→ FinMind 增量（100 支 × 3 個資料集）→ 建 panel → 訓練今天的模型 |
| 早上（08:01 以後、08:50 以前） | `race_daily.bat` | 合併早上的 zip → 除權息預告、夜盤（Baszo 的程式）→ Yahoo 增量（美股前一晚收盤、日經 / KOSPI 今天開盤）→ FinMind 增量（晚上沒跑時補）→ 建 panel → pool 不足 3 個就補訓練 → 預測、決策 → `out\submission_race_MMDD.csv`（id ＋ 預測日一個 column，順序同 sample_submission）與檢查 |

檢查（FATAL 不寫上傳檔；WARN 照常寫）：
- FATAL：要預測的日子不是最後一個有 label 的交易日的下一個交易日；沒有 −1 佔位 row；pool 是空的；有模型的訓練資料不在 t 之前；
  輸出不是 Top50 剛好 50 支；猜 1 不在 15–25 支；C 組非 ADR 的 feature 在 Top50 超過 20% 缺值。
- WARN：pool 少於 3 個；美股不是最新一晚；日經或 KOSPI 開盤還沒有（08:00 前跑、或當地休市）；FinMind 最新資料不是前一個交易日。

## 4. VM 的測試（沒有網路）

用 `race_std\ext\` 已有的資料，對 2026-11-09（最後一個有 label 的日子）做一次完整的「早上」流程（假裝 11/09 是要預測的日子：
資料截在 11/06、11/09 的價量換成 −1），檢查：feature 和第一部分 panel 的同一天逐位相同（float32 捨入內）、輸出格式、檢查訊息。
網路步驟（FinMind、Yahoo、課程 zip）只能由使用者在 Windows 跑。

# race_live：race 版每天的程式與每天重訓的 walk-forward

（2026-11-16 01:40 起；事先寫好的規則在 `PREREG_race_live.md`，SHA-256 前 16 碼 aca2bd4a88510976，01:42 寫入，在任何 walk-forward 成績之前）

race 版 = 陳聖翰 V5 的 169 個 feature（C 組）＋ LightGBM LU（Baszo lgb_res 的設定）＋ 每天重訓（最近 3 個有 label 的交易日各一個模型，
seed = 那一天，機率平均）＋ Top50 依機率前 25 支猜 1、除權息規則（當天、前一天、後一天、後兩天改 0）。不加大盤觀點、不對沖（對沖留給 11/23）。
第一部分（`..\race_std\research`）、第二部分（`..\race_std\part2`）的程式和資料都沒有改，這裡只 import。

**walk-forward 的判定（07:33）：每天重訓（`race_mode.txt` = daily）。** D 上每天重訓 16.35 bp/日、每月重訓 16.67，差 −0.32（SE 0.41），
沒有差到 1 個 SE，照規則用每天重訓；兩者其實分不出高下，都比年度重訓高（+1.25、+1.57）。見下面「walk-forward」。

## 每天怎麼用（Windows）

| 時間 | 檔案 | 做什麼 | 大約時間 |
|---|---|---|---|
| 晚上 21:30 以後（可省略） | `race_evening.bat` | 合併晚上的 zip、除權息表、夜盤、FinMind 增量（約 300 次請求）、訓練今天的模型 | FinMind 約 5 分鐘＋訓練 1 個模型 |
| 早上 08:01–08:50 | `race_daily.bat` | 合併早上的 zip、除權息表、夜盤、Yahoo（28 個序列）、FinMind 只補還沒到的、建 panel、pool 不足就補訓練、預測、決策 | 晚上有跑：建 panel ＋ 預測；沒跑：再加最多 3 個模型的訓練 |

上傳 `out\submission_race_MMDD.csv`（id ＋ 預測日一個 column，順序同 sample_submission，格式和 v9 的上傳檔相同）。同一個資料夾還有
`pred_race_MMDD.csv`（Top50 每支的機率、每個模型的機率、名次、除權息旗標）和 `report_race_MMDD.txt`（檢查結果與 Top50 名單）。
FinMind token：在 Windows 設定 `FINMIND_TOKEN`（或在 cmd 先 `set FINMIND_TOKEN=...`）。

FATAL（不寫上傳檔，exit code 1）：沒有 −1 佔位 row、要預測的日子不是最後一個有 label 的交易日的下一個交易日、pool 是空的、
模型的訓練資料不在預測日之前、Top50 不是 50 支、猜 1 不在 15–25 支、Top50 的非 ADR C feature 缺值超過 20%。
WARN（照常寫上傳檔）：pool 少於應有的數量、美股不是最新一晚、日經 / KOSPI 今天還沒開盤（或休市）、FinMind 還沒到前一個交易日、
某些 C column 今天全部缺值但昨天有。檢查過是誤報：`python race_live.py predict --force`。

模式（`race_mode.txt` 的第一個字；沒有這個檔 = daily）：`daily` = 上面的每天重訓；`monthly` = 第二部分的 MR（每月第一個交易日以前的資料、
seed 1–3）。目前是 daily（walk-forward 的判定）。

## walk-forward（PREREG 第 2 節，只用 D）

每個交易日 d（2021-12-28 到 2024-12-30）訓練一個模型（第一部分 `learners.fit_LU`，訓練資料 2020-03-02 到 d、seed = d），
預測後 3 個交易日；第 t 天的分數 = t 之前 3 個交易日的模型的機率平均，決策同上。共 729 個模型，02:00–07:31：
VM（Linux，2 vCPU）跑 2024 和 2022 的後段；05:00 起使用者在 Windows 開 `run_wf_windows.bat`（4 個 worker）跑 2023 和 2022 大部分。
三年都在 08:00 前完成，全部納入。

| bp/日（727 天） | 平均 | NW t | 每日 sd | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|
| **WF3：每天重訓、3 個模型平均（race 版）** | **16.35** | 10.25 | 42.5 | 18.13 | 11.83 | 19.00 |
| WF1：每天重訓、只用最新 1 個（描述） | 15.84 | 9.96 | 42.7 | 17.58 | 11.38 | 18.49 |
| MR：每月重訓、3 seed（第二部分） | 16.67 | 10.65 | 42.0 | 19.16 | 11.67 | 19.08 |
| QR：每季重訓（第二部分） | 16.14 | 10.26 | 42.4 | 19.12 | 10.75 | 18.43 |
| SD：年度重訓（第一部分 C.LU） | 15.10 | 9.80 | 42.7 | 17.02 | 10.69 | 17.50 |

| 成對差（同一天，NW 5 lags） | 平均 | t | SE | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|
| WF3 − MR（判定用） | −0.32 | −0.78 | 0.41 | −1.03 | +0.16 | −0.08 |
| WF3 − QR | +0.21 | +0.45 | 0.46 | −1.00 | +1.08 | +0.57 |
| WF3 − SD | +1.25 | +2.10 | 0.59 | +1.10 | +1.14 | +1.50 |
| WF1 − WF3（描述） | −0.50 | −1.60 | 0.31 | −0.55 | −0.45 | −0.52 |
| MR − SD（第二部分，對照） | +1.57 | +2.58 | 0.61 | +2.14 | +0.98 | +1.58 |

- 判定：WF3 − MR = −0.32 ≥ −SE = −0.41 → live 用每天重訓。每天和每月在 D 上分不出高下（每月點估計略高，在雜訊內）；
  兩者都穩定高於年度重訓。3 個模型平均比只用最新一個高 0.50（3 年都一樣方向）。
- 19 天平均 < 0 的比例：WF3 2.0%、MR 2.1%、SD 3.1%。WF3 每天 Top25 平均換 6.7 支。每天模型的 best_iter 中位數 354（10% 201、90% 578）。
- 檢查：為了排程，每個模型拆成兩半（early stopping 求輪數、再用全部資料重訓），和 `fit_LU` 一次做完的結果逐位相同（2024-06-07）；
  thread 數不影響結果（col-wise ＋ deterministic）；Windows 訓練的模型（2023-03-15）在 VM 重算，預測最大差 5.6e-17。
- V 沒有打開：沒有 2025 以後的訓練 row，也沒有預測 2025 以後的 row。
- 結果檔：`results\wf_daily.txt`、`.json`、`wf_daily_daily.csv`；每天模型的預測在 `cache\wf_units\`。

## 2026-11-16 的預測與資料問題

- 05:03 使用者要求先用目前的 race 版做 11/16 的預測（爬不到的先不管）。用 `downloads\data (14).zip`（04:00 打包）在 VM 做了
  `out\submission_race_1116_early.csv`：模型放在 `models_early\`（正式版不會拿去用），缺夜盤（`v7\data\tx_daily.csv` 只到 11/10）、
  日經 / KOSPI 開盤（08:00 才開）、當天的 ^VIX（zip 裡是 −1，沿用前一天）；除權息表是 11/10 抓的。
- **zip 的問題**：這個 zip 多了 4 天國定假日的 −1 列（2026-09-25 中秋、09-28 教師節、10-09 國慶補假、10-26 光復節補假；11/10 的 zip 沒有）。
  合併後 `update_data.py` 列出 5 個預測日，20、60 日的移動平均類 feature 最近兩個月變成缺值。處理：這 4 天加進 `data\ref\closures_manual.csv`
  （和 10/23 颱風假同一個機制；原檔在 `cache\closures_manual_before_20261116.csv`），重新合併後只剩 11/16 一個預測日。
  用壞資料做的那一版移到 `out\discarded_holiday_rows\`、模型移到 `models_early_discarded_holiday_rows\`（25 支有 24 支相同）。
- 正式版：08:01 以後下載早上的 zip、跑 `race_daily.bat`（重新抓上面缺的資料、在 `models\` 訓練 3 個模型）。

## 檢查（VM，2026-11-16 02:01–02:36，用第一部分凍結的資料）

| 檢查 | 結果 |
|---|---|
| `selftest lu`：`livelib.train_model` 用第一部分 C.LU 的訓練窗（到 2023-12-29）和 seed 1 | best_iter 290 = 290；2024 年 24196 個預測和 `C.LU/2024-01-01_seed1.npz` 的最大差 0（逐位相同） |
| `selftest panel`：live 的建 panel（只算 C 組，爬的資料讀 `race_live\ext`）對第一部分 `panel.pkl` | 166346 rows 都一樣；169 個 C column 的 NaN 位置差 0 格、最大差 0；label 與除權息旗標相同 |
| 早上流程演練（假裝要預測 11/10：資料到 11/09，11/10 是 −1 佔位） | 訓練 11/05、11/06、11/09 三個模型（每個約 29 秒，2 threads）、預測、決策、寫檔；猜 1 有 25 支，沒有 FATAL / WARN；再跑一次結果逐位相同 |
| 同上，monthly 模式 | 用 10/30 以前的資料、seed 1–3；和 daily 的 25 支有 23 支相同 |
| 晚上流程（`train`） | 下一個交易日 11/10 需要的 3 個模型都已存在，不重訓 |
| `fetch_live.py --check` | Yahoo 與 FinMind 都到 11/13（1 支股票的當沖資料到 11/06） |

網路的部分（Yahoo、FinMind、課程 zip）VM 不能測；下載函式是 11/15 在 Windows 上跑過的 `fetch_ext.py` 的函式。

## 檔案

| 檔案 | 用途 |
|---|---|
| `race_daily.bat`、`race_evening.bat` | 每天的流程（上表） |
| `race_mode.txt` | daily / monthly（目前 daily） |
| `livelib.py` | 共用函式：建 C 組 panel（呼叫第一部分 `build_panel.py` 的函式，爬的資料改讀 `race_live\ext`）、訓練（= 第一部分 `learners.fit_LU`）、預測、決策、上傳檔 |
| `race_live.py` | `train`（晚上）、`predict`（早上）、`selftest lu / panel`（VM 檢查）；`--mode`、`--models`、`--out`、`--tag`、`--no-train`、`--force` |
| `fetch_live.py` | `--finmind`（晚上）、`--finmind --stale`（早上只補缺的）、`--yahoo`（早上）、`--check`；下載函式是第一部分的 `fetch_ext.py` |
| `walkforward.py`、`run_wf_windows.bat` | 每天重訓的 walk-forward（D，2022–2024）；`--prep`、`--run2`、`--status`、`--eval`、`--check-split` |
| `allcand_reference.py` | 所有 candidate 的每年 walk-forward 參考表（第一部分 33 個、第二部分 12 個變體、第三部分 WF3 / WF1；不參與任何決定）→ `results\allcand_wf_reference.txt`、`.csv`、`allcand_wf_daily.csv` |
| `ext\` | 第一次執行時從 `race_std\ext` 複製，之後每天更新（`race_std\ext` 不動） |
| `models\` | 正式的每天模型 `race_YYYYMMDD_seedS.txt` ＋ `.json`（訓練到哪一天、seed、輪數） |
| `models_early\` | 11/16 提早版的模型（缺夜盤資料；正式版不用） |
| `cache\live_panel.pkl` | 建好的 panel（輸入檔沒變就直接用）；`cache\wf_units\`、`cache\wf_phase1\` = walk-forward 的預測與輪數 |
| `results\` | `live.log`、`selftest.log`、`walkforward.log`、`wf_daily.*`、`allcand_wf_*` |
| `models_selftest\`、`out_selftest\` | VM 演練的模型與輸出（11/10，只是測試） |

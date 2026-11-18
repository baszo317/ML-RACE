# ML RACE：預測台股「收盤價 > 開盤價」

最後整理：2026-10-26（沒在用的 v2、v4、測試輸出、舊備份移到 `棄案\`；各版本在做什麼見「版本」一節）。
2026-10-27：新增 `v6\phase2.bat`（收盤後自己下載價格、算當天 live 上傳檔的 AR，見下面早上的第 6 步）；10/27 AR 差的原因見 `v11\results\rerun_1027_us\README.txt`；
兩個改法（休市後關 T2、Top50 固定一半猜 1）測過都沒採用（`v11\README_v11.md` 第 10 節）；daily 最後會檢查 zip 有沒有前一晚的美股。
2026-10-28：overfit 檢查（`overfit\README_overfit.md`，只有研究，沒改任何每天的程式）：成員模型會 overfit 訓練期，但分數都是樣本外、early stopping 停在對的地方，核心的 edge 是真的；疊上去的改進（v5 → v9 → v11）和用 static、10 月挑的係數（W0）在挑選雜訊之內。
2026-11-01：10/30 早上漏跑 daily，補做了 10/30 的三個上傳檔（v9、v11、W0，在各自的 `out\`）和 AR（W0 +29.62、v11 +14.97、v9 −21.43 bp），只用 10/30 09:00 前拿得到的資料；做法和檢查見 `v11\results\rerun_1030\README.txt`。
2026-11-04：新增 `reg_svm\`（只有研究，沒改任何每天的程式）：要不要 regularization、換成 SVM 會不會比較好。regularization 全部拿掉時 mini v9 −1.80 bp/日（大盤 MLP 傷害最大），加強也沒有更好；exact RBF SVM 當額外選股成員（筆電 2.7 小時，`reg_svm\run_svm_exact.bat`）自己的 IC 0.109 是選股成員裡最高的，但照事先寫的規則沒通過（v11 + SVM：hold −0.69、post −0.30），v9、v11 不變（`reg_svm\README_reg_svm.md`）。
2026-11-05：新增 `v13\`（只有研究，沒改任何每天的程式）：LSTM、CNN、F-LOAM 在這個比賽可不可行。第一部分（雲端跑完）：LSTM、1D CNN 吃原始的每日價量序列（60 天），當選股觀點的新成員，規則事先寫好；IC LSTM-R 0.094、CNN-R 0.085，比 v8 吃 B 組的 LSTM（0.101）低；v11 + LSTM-R val −0.19、v11 + CNN-R val −0.02，都沒入選，v9、v11、W0 不變。第二部分：F-LOAM（Liao & Lin 2025）——論文的做法（收盤價水準的 R²、隨機切分、用測試集調參、整段 SVMD）在我們的資料上會偷看未來（`v13\research\floam_checks.py`）；不偷看的版本（2×50 LSTM 的 h_t + B 組 → LightGBM，Optuna 只在訓練窗最後 120 天調）規則已寫好（`v13\research\PREREG_v13b.md`），等筆電跑 `v13\run_v13.bat`（全部約 3–5 小時，只跑 F-LOAM 用 `run_v13.bat floam` 約 1–2 小時）（`v13\README_v13.md`）。
2026-11-07：新增 `v14\`（只有研究，沒改任何每天的程式）：還有哪些量化方法能用。第 1 節是常見量化方法的對照表（能不能用、v1–v13 在哪裡做過）；實驗五組，規則事先寫好（`v14\research\PREREG_v14.md`），不重訓、只用存好的 walk-forward 預測，雙擊 `v14\run_v14.bat` 約 2 分鐘：PCA 統計套利 s-score、殘差反轉、產業／統計因子中性、季底作帳、14 個模型 1/N，**沒有一個通過**，v9、v11、W0 不變；就算 09:00 前就知道開盤價，加進 v11 也只多約 +1 bp/日（只報告）；中性化明顯變差（−2.3 到 −3.7），代表選股觀點的產業押注本身是賺的。另有沙盒 `v14\research\v14_try.py`：自己寫一個訊號函式，約 15 秒得到 leak check、IC 和加進 v11 / W0 的分數（`v14\README_v14.md`）。
2026-11-07（晚上）：新增 `v15\`（只有研究，沒改任何每天的程式）：去噪方法和 Random Forest。文獻搜尋＋ 13 個候選，規則事先寫好（`v15\research\PREREG_v15.md`），筆電跑 `v15\run_v15.bat` 86 分鐘：入選 7 個（RF、Huber 權重、3／5 天 target、DoubleEnsemble、因果 wavelet、Vasicek beta），合起來 val +0.24、hold −0.79、post −1.63，**沒通過**，v9、v11、W0 不變。RF 葉子夠大時 IC 0.106 和 lgb_res 相同（相關 0.90），sklearn 預設（葉子 1）只有 0.070；多任務、autoencoder 讓 MLP 的 IC 變高但分數變低；Kalman 的最佳濾波就是長期平均（`v15\README_v15.md`）。
2026-11-08：新增 `v16\`（只有研究，沒改任何每天的程式）：Kalman filter 會不會 work。三個 v1–v15 沒做過的用法，規則事先寫好（`v16\research\PREREG_v16.md`），不訓練模型，雙擊 `v16\run_v16.bat` 約 1 分鐘：K1 Kalman beta（對沖）、K2 時變的大盤觀點係數、K3 每天在 v11 和 W0 之間切換。三個的 val 都 ≤ 0（−0.05、−0.72、−0.03），**沒有一個入選**，v9、v11、W0 不變。beta 確實會變，但 MLE 選到的記憶（52–58 天）就是 120 天 OLS 的記憶（約 60 天），beta 預測誤差沒有變小；大盤觀點的係數和 v11 / W0 的相對表現變得比雜訊能分辨的快（偵測要 110–260 天），MLE 大多選固定參數，強迫調快更差（`v16\README_v16.md`）。晚上加第二部分 Kalman smoother（`v16\research\PREREG_v16b.md`，`run_v16.bat smoother` 約 1 分鐘）：只用過去資料的 smoother 和 filter 逐位相同；偷看版（整段資料一起平滑）全期可以到 +8.6 bp/日（t 7.6），全部來自當天和之後的資料；唯一合法的用法（只在訓練窗平滑 label 的線性模型）照規則通過（和 v11 比 +0.38，t 1.00），但它自己的 IC 從 0.091 掉到 0.058、學到假的動能，W0 版本是負的，不建議用（`v16\README_v16.md` 第 9 節）。
2026-11-15（晚上）：新增 `race ver\race_std\`（只有研究，沒改任何每天的程式）：組內四套系統（你、FinPred、陳聖翰、NTU M2）合併的驗證標準 RACE-STD v1，規則事先寫好（`race ver\race_std\research\PREREG_race_std.md`）；D = 2022–2024 開發，V = 2025-01 到 2026-11-09 只用一次（11/15 23:55 封存）。陳聖翰自己爬的 40 個 feature 用 `race ver\race_std\run_fetch.bat` 從 Yahoo、FinMind 重爬。結果：race 版選股觀點 = C.LU（陳聖翰的 169 個 feature ＋ lgb_res 的 LightGBM），V +16.16 bp/日（Top25 ＋除權息規則，不加大盤觀點）；比 W0 的選股結構（U-native，+14.17）高 +1.99，但 t 1.07，不顯著（`race ver\race_std\README_race_std.md`）。
2026-11-16：`race ver\race_std\part2\`（只有研究）：補測第一部分沒測的東西，規則事先寫好、只用 D 判定（V 沒有再打開）。beta 對沖、T2、T3、pbar 大盤觀點、W0 / v11 決策層加在 C.LU 上都沒過門檻；每月重訓比年度重訓 +1.57 bp/日（t 2.58，3 年都 > 0）→ race 版 live 每天重訓（`race ver\race_std\part2\README_part2.md`）。
2026-11-16（凌晨）：新增 `race ver\race_live\`：race 版每天的程式（晚上 `race_evening.bat`、早上 `race_daily.bat`；陳聖翰的 169 個 feature ＋ lgb_res 的 LightGBM、每天重訓、最近 3 天的模型平均、Top25 ＋除權息規則），上傳 `race ver\race_live\out\submission_race_MMDD.csv`。每天重訓的 walk-forward（規則事先寫好，D 2022–2024，729 個模型）16.35 bp/日：和每月重訓（16.67）分不出高下、比年度重訓高 +1.25（t 2.10），照規則用每天重訓。04:00 的課程 zip 多了 4 天國定假日（9/25、9/28、10/9、10/26）的 -1 列，已加進 `data\ref\closures_manual.csv`（`race ver\race_live\README_race_live.md`）。
2026-11-17：新增 `race ver\race_mkt\`（只有研究，沒改任何每天的程式）：可以猜任何數量 0/1 的大盤觀點，以及期貨籌碼（大台／小台／微台三大法人多空 OI、散戶多空比、大額交易人、外資選擇權、P/C；`race ver\race_mkt\run_fetch_mkt.bat` 在 Windows 下載）能不能改善盤前資訊，規則事先寫好（`race ver\race_mkt\research\PREREG_race_mkt.md`）。8 個版本在 D 都沒過門檻（−0.36 到 +0.32 bp/日，t ≤ 0.35），V（只算一次）−1.21 到 +1.47；期貨籌碼對大盤預測的 OOS 相關只多 0.006（0.117 → 0.123），放進選股模型 −0.38（t −0.70）。25/25 的上限不低：完全資訊下 Top25 108.65、任何數量 118.61 bp/日，差距在資訊。race 版不變（`race ver\race_mkt\README_race_mkt.md`）。
2026-11-17（下午）：新增 `race ver\race_news\`（只有研究，沒改任何每天的程式）：鉅亨網台股新聞（2019-10 到 2026-10，14.8 萬篇，`race ver\race_news\run_fetch_news.bat` 在 Windows 下載）做成 12 個新聞 feature（隔夜篇數、公司自己的新聞、異常注意力、正負面詞 tone、營收／財報／目標價／股利關鍵字、全市場篇數與 tone），規則事先寫好（`race ver\race_news\research\PREREG_race_news.md`）。加進 C.LU：D −0.29 bp/日（t −0.86），沒過門檻；最強的是隔夜篇數，和盤中殘差報酬 IC −0.030。另外：GDELT、TWSE OpenAPI、data.raytsai.dev 沒有用，MIS 試撮上限約 +1 bp/日（`race ver\race_mkt\README_race_mkt.md` 第 8 節）。race 版不變（`race ver\race_news\README_race_news.md`）。
2026-11-18：新增 `race ver\race_agg\`（只有研究，沒改任何每天的程式）：用課程 topic 6 / W12 的 aggregation 方法（uniform / linear / any blending、bagging、AdaBoost-Stump / DTree、單一 C&RT ＋ pruning、Random Forest ＋ OOB ＋ permutation 選 feature、換 loss 的 GBDT）檢驗 race 版，16 個候選＋ 1 個 OOB 診斷，規則事先寫好（`race ver\race_agg\research\PREREG_race_agg.md`），只用 D 判定。**沒有一個過門檻**（最接近的 linear blending +0.77 bp/日，t 1.55），race 版不變，V 沒有打開。race 版用的 logistic GBDT 就是這些方法裡最好的：單一 C&RT −6.89、AdaBoost −4.42 到 −5.09、RF −2.58；LU ＋ MLP 各半的 uniform blending 讓 IC 高 5.6%（t 2.41，和公式預測相同），但 AR 沒變（−0.01）。事後發現 race 版 D 的 15.10 裡有約 +0.45 的 seed 運氣（seed 1–10 任取 3 個平均 14.66）；一般 RF 的 row 層級 OOB 把 IC 高估 1.7–2 倍（同一天同產業的共同衝擊）（`race ver\race_agg\README_race_agg.md`）。

## 現在用哪一版

- **live（每天上傳）：v9**。晚上跑 `v9\evening.bat`，早上跑 `v9\daily.bat`，上傳 `v9\out\submission_v9_MMDD.csv`（見下面「每天的流程」）。係數固定（w_T 0.5、λ 0.75）。
- **v11（10/26 新增，可選）**：改雙擊 `v11\evening_v11.bat`（晚上）、`v11\daily_v11.bat`（早上）。它們先跑 v9 的完整流程，再跑 v11（多約 1 分鐘），同時寫出 `v9\out\submission_v9_MMDD.csv` 和 `v11\out\submission_v11_MMDD.csv`，上傳其中一個。證據和風險見 `v11\README_v11.md` 第 0 節（hold 差 +0.01、10 月 −8.63）。
- **race 版（11/16 新增，可選）**：晚上 `race ver\race_live\race_evening.bat`、早上 07:00 以後下載 zip 再跑 `race ver\race_live\race_daily.bat`，上傳 `race ver\race_live\out\submission_race_MMDD.csv`（見 `race ver\race_live\README_race_live.md`）。
- **static leaderboard**：`v9\out\submission_v9_static_w025.csv`（10/25 決定；為什麼和風險見 `v9\README_v9.md` 8.1）。w = 0.5 的 `v9\out\submission_v9_static.csv` 保留。
- 緊急備案：`v5\daily.bat`。注意 v5 從沒在這台電腦跑過，第一次會訓練 Transformer（約 5–10 分鐘）；它的模型池起點是 `棄案\v2\models_v2\`（v2 移到棄案後，程式會自己去那裡找）。
- 10/26 測過「係數每天依最近 15 天自動調整」：做得到，但 9 種調法在五年 walk-forward 全部比固定係數差，沒有採用（`v9\README_v9.md` 第十一節）。

## 每天的流程（課程 zip 約 17:00 收盤後釋出，隔天 9:00 前上傳）

> **10/27–28 實測，和原本的設計不同**：課程 zip 一天打包好幾次。收盤後的（10/27 15:52、21:05）只有當天的答案，沒有隔天要預測的 -1 row；00:00 的（10/28）有隔天的 -1 row，但 ^VIX ^IXIC ^GSPC 也是 -1（美股 04:00 才收盤）；前一晚的美股要到早上 07:00 左右的 zip 才有（10/23 也是 07:00）。所以晚上做不出預備檔。
>
> **10/28 起的做法**：晚上下載 zip、雙擊 `evening`（v9 或 v11 的）：遇到上面兩種 zip 會**只訓練模型**（筆電約 7–8 分鐘），不寫預備檔，正常結束。**早上 07:00 以後**再下載最新的 zip、雙擊 `daily`：模型已經訓練好，只預測（約 2 分鐘）。晚上沒跑 evening 也可以，早上的 daily 會先訓練（多約 7 分鐘，8:30 前開始）。

### 晚上（17:00 以後）

1. 從課程網站下載當天的 zip，放進 `downloads\`（留在自己的「下載」資料夾也可以，程式會找最新的 `data*.zip`）。
2. 雙擊 `v9\evening.bat`（或 `v11\evening_v11.bat`）：合併 zip、抓除權息預告、訓練當天的新模型（筆電約 7 分鐘），最後備份（`v9\backup_v9.py`）。
   - zip 裡有可用的隔天 row（有美股）時，才會再寫**預備檔** `v9\out\submission_v9_MMDD_prelim.csv`；10/27 起的 zip 都沒有，所以通常只有訓練。

### 早上（05:00 以後、9:00 以前）

3. 課程網站早上也會打包 zip（10/23 看到 07:00 一份，當天那一 row 已經有前一晚的美股收盤）：有的話先下載到 `downloads\`。
   然後雙擊 `v9\daily.bat`：抓夜盤和除權息；模型昨晚已經訓練好，只做預測（約 2 分鐘），寫出 `v9\out\submission_v9_MMDD.csv`，並列出和預備檔差幾支。
   （10/27 沒有下載早上的 zip，美股少了兩晚；事後重算只差約 4 bp，見 `v11\results\rerun_1027_us\README.txt`。）
4. 上傳 `v9\out\submission_v9_MMDD.csv`。
5. 截 daily board 和 accumulated board 的圖，存進 `leaderboard\`（說明在 `leaderboard\README.md`）。
6. 收盤（13:30）後約 1–2 小時，雙擊 `v6\phase2.bat`：自己下載當天 50 支的開盤、收盤價，算每個上傳檔的 AR，和排行榜的兩個 baseline 列在一起，可以直接和 daily board 對照。在 cmd 執行 `"D:\ML RACE\v6\phase2.bat" --uploaded 上傳的檔.csv` 記下那天上傳的是哪個檔（10/27 已經記好），累積表就會多一行 uploaded，對照 accumulated board（`v6\README_v6.md` 第七節）。

### 出狀況時

| 狀況 | 怎麼做 |
|---|---|
| 時間不夠 | 晚上跑過 evening 的話 daily 只要約 2 分鐘；昨晚有預備檔 `v9\out\submission_v9_MMDD_prelim.csv` 的話也可以直接傳（10/27 起通常沒有） |
| daily 出現「no U.S. close for this day yet」 | 用到的是 00:00 打包的 zip（隔天的 row 還沒有美股）：07:00 以後下載新的 zip 再跑 |
| 只想用現有模型預測（約 1 分鐘） | 在 `v9\` 執行 `python train_and_predict_v9.py --dates auto --retrain none` |
| 出現「is in the past」 | 用到的是舊 zip：下載最新的 zip 再跑 |
| daily 最後出現 `[WARN] U.S. data ...` | 合併的 zip 是前一晚打包的，美股不是最新的：課程網站有新的 zip（早上 07:00 左右）就下載到 `downloads\` 再跑一次 daily；沒有就照常上傳（10/27 這種情形只差約 4 bp） |
| 晚上跑 daily 停在「no -1 placeholder day」 | 正常：收盤後的 zip 只有當天的答案，沒有隔天的 -1 row（10/27 的 15:52、21:05 都是）。晚上請跑 evening（10/28 起只訓練模型），早上 07:00 以後下載新的 zip 再跑 daily |
| 出現「is in the past」，但已經是課程網站最新的 zip，而且那天其實沒開盤（颱風假等臨時休市） | 在 `data\ref\closures_manual.csv` 加一行 `日期,原因`，再跑 `daily.bat`：那天會從資料和交易日曆拿掉，要預測的日子改成下一個交易日（`v9\README_v9.md` 10.8；10/23 薔琵颱風就是這樣處理） |
| update_data.py 顯示 CHECK THESE，看過後確認是誤報 | 在 cmd 執行 `"D:\ML RACE\v9\daily.bat" accept`（晚上用 `evening.bat accept`） |
| 其他 FATAL，看過訊息後確認是誤報 | 在 `v9\` 執行 `python train_and_predict_v9.py --dates auto --force` |
| 晚上忘了跑 | 早上照常跑 `daily.bat`，它會自己訓練（約 7 分鐘） |

## 版本：每一版在做什麼

分數是 Top50 的 bp/日（競賽算法）。walk-forward：val 2022–24、hold 2025-01 到 2026-05、post 2026-06 以後。

| 版本 | 現在在哪 | 在做什麼 | 主要結果 | 狀態 |
|---|---|---|---|---|
| v1 | `transformer\`（舊的訓練程式、模型、上傳檔在 `棄案\transformer\`） | 67 個 B 組 feature（只用競賽資料）；MLP 和 Transformer（20 天序列）的機率平均，> 0.5 猜 1 | val 12.30、hold 12.04，post 只剩 0.76（大盤方向押太重） | 停用。資料夾留著 v9 每天要用的共用程式（`comp_features.py`、`tfmodel.py`、`update_data.py`、`final_feature_lists.json`） |
| v2 | `棄案\v2\` | 拆成大盤觀點（pbar = 100 支「收 > 開」機率的平均）和選股觀點（殘差 label：贏過當天 100 支平均）；score = 0.5 × (pbar − 0.5) + (p_e − 當天平均)；除權息日猜 0 | val 13.27、hold 13.45、post 12.22 | 停用，之後每一版的骨架。10/26 移到棄案（`v5\daily.bat` 第一次執行會讀它的 `models_v2\`） |
| v3 | `棄案\v3\` | v2 的選股觀點加 5 個新 feature | 沒有比 v2 好 | 未採用；`features_v3.py` 已複製到 `v9\` |
| v4 | `棄案\v4\` | v2 + rolling pool：每天只用到前一個交易日的資料加訓新模型、平均最近 5 個訓練點；除權息表補到 10 月 | static 15.36（舊的 static 上傳檔） | 停用。10/26 移到棄案；它的 static 檔仍是 `v6` 工具和 v9 static 報告的對照（程式會到棄案找） |
| v5 | `v5\` | 預測和 v4 完全相同；加上每天早上的防呆（`guards.py`，FATAL 就不寫檔）和監控；feature 去留、ablation、三個風險（selection bias、drift、檢定力） | 全期 13.38；21 個改法都沒過門檻 | 緊急備案（`v5\daily.bat`）；v9 每天用它的 `ens_lib.py`、`rolling.py`、`guards.py`、`features_extra.py`、`fetch_exdiv.py` |
| v6 | `v6\` | static 專用工具：`compare_static.py`（組內比較，附雜訊和「挑過幾組」的修正）、`leak_check.py`（檢查上傳檔有沒有 leak）；`others\` 放組員的檔。10/27 加上 live（Phase 2）的本地檢查 `phase2.bat`：自己下載當天價格、算每個上傳檔的 AR 和累積 | phase1_4 查出 6–8 月等於答案（FATAL）；phase2 和排行榜的 baseline 逐位相同 | 工具，在用 |
| v7 | `v7\` | live 研究：擴充股票池、台指期夜盤 feature、開盤 gap 上限、除權息前後日規則、w_T 風險設定 | 都沒過當時的門檻（val t > 0 的幾項後來併進 v9） | 研究；v9 每天用它的 `fetch\fetch_night_session.py` 和 `data\tx_daily.csv` |
| v8 | `v8\` | 大盤觀點診斷（pbar 從哪來、選股觀點暗藏的低 beta 押注）、架構比較（CNN、GRU、LSTM、LightGBM、logistic）、beta 對沖 | 5 個大盤改法都沒過；LightGBM 進選股（val +0.36）、對沖 λ 0.75（val +0.28）後來進了 v9 | 研究；v9 每天用 `hedge_beta.py`、`monitor_market.py`，v9 的研究程式用 `research\mlib.py` 和 `cache\` |
| v9 | `v9\` | 把 v1–v8 裡 val t > 0 的 11 個改法混成一個：夜盤、擴充股票池、LightGBM、4 年 window、σ 權重、v3 成員、拿掉那斯達克、除權息前一天／後一天／後兩天猜 0、beta 對沖 λ 0.75 | walk-forward 14.31（val 13.99、hold 14.33、post 16.59；v5 13.38）；rolling static 16.42，w 0.25 版 18.84 | **live 正式版**（10/27 起）；static 上傳 `out\submission_v9_static_w025.csv` |
| v9 第十一節 | `v9\research\adapt_eval.py` | 係數（w_T、λ、LightGBM 權重、Transformer 權重、門檻）每天依最近 15 天的成績自動調整 | 9 種調法全部輸固定係數（主要假設 −1.67，t −1.94）；視窗 5–250 天都輸 | 不採用，live 沒改 |
| v10 | `v10\` | Qlib Alpha158（158 個、23 組）加進 v9 選股觀點的 ablation；val（AR）t > 0 的群組全部加進去 | L 入選 19 組、M 5 組；hold −0.45（t −0.75）沒過事先寫的檢驗 | 不採用，live 沒改（`v10\README_v10.md`） |
| v11 | `v11\` | 第一部分：用低雜訊的方法重挑 Alpha158（rank IC、20 個 seed、嚴格規則、nested walk-forward），沒有群組通過。第二部分：v9 + 夜盤反轉（T2）+ 每月前 3 天多 5 支猜 1（T3）+ 選股第二個 LightGBM（X1），照「val bp > 0」從 7 個候選入選 | walk-forward 15.02（v9 14.31）：val +0.62、hold +0.01、post +3.76，事先寫的檢驗通過；但 10 月 15 天 −8.63（t −3.3） | 和 v9 並行：`v11\daily_v11.bat`、`evening_v11.bat` 同時寫出 v9 和 v11 的上傳檔，上傳哪一個由你決定（`v11\README_v11.md`）。10/27 測的 F1、F2 沒採用，多了 `--balanced` 選項（第 10 節） |
| v12 | `v12\` | 用實驗檢驗 v11 第 9 節對 ranking objective 的解讀：把「label 的資訊」（正負、十分位、報酬大小）和「損失函數」（pointwise、同一天 pairwise、lambdarank）分開，新訓練 6 種 LightGBM；照「val bp > 0」入選的再加到 v11 上檢驗 | 解讀：lgb_res 本來就是排序學習（成立）；X5 差是因為 NDCG 的位置折扣（成立），不是十分位 label（十分位反而比報酬大小好）。v12 = v11 + 4 個入選：hold −1.05（t −2.4）沒過事先寫的檢驗 | 只有研究，沒有每天的程式，live 沒改（`v12\README_v12.md`） |
| overfit | `overfit\` | 檢查目前的模型（v9、v11、10/27 上傳的 W0）會不會 overfit：理論保證（walk-forward 無偏、Bernstein 下界、挑選的高估上界、hold 重複使用、漂移）附證明；實驗：成員重訓看訓練期 vs 樣本外與 learning curve、純雜訊 label、CSCV / PBO、reality check、用最近一段挑係數、72 個改法的 winner's curse、半年 random effects | 成員訓練期 IC / 相關是樣本外的 1.5–4 倍，純雜訊也做得到，但 early stopping 停在樣本外最好附近；edge 在保守假設下仍 > 0（DSR 1.00）；val 和 hold 的相關只有 0.16，v11 − v9 真正效果約 +0.04–0.2；用前 99 天挑係數，下 19 天平均輸 v9 2.73（t −2.9） | 只有研究，live 沒改 |
| reg_svm | `reg_svm\` | 要不要 regularization：三個成員把 regularization 拿掉或加強後重訓、訓練天數 1/8 到全部；換成 SVM：linear、Nystroem 近似、exact RBF SVM（當 v11 的額外成員，規則事先寫好） | 全部拿掉 −1.80 bp/日（只拿掉大盤 MLP 的就 −2.52），LightGBM 幾乎不受影響、選股 MLP 靠 early stopping，加強沒有更好；exact SVM IC 0.1093（lgb_res 0.1050），但 v11 + SVM：val +0.03、hold −0.69、post −0.30，沒通過 | 只有研究，live 沒改（`reg_svm\README_reg_svm.md`） |
| v13 | `v13\` | 第一部分：LSTM、1D CNN 吃原始每日價量序列（60 天、19 個 channel），當 v11 選股觀點的新 family（`research\PREREG_v13.md`），雲端訓練 60 個 unit。第二部分：F-LOAM（Liao & Lin 2025）的不偷看版本，放進 LightGBM 部分（`research\PREREG_v13b.md`）；論文做法的檢查（`research\floam_checks.py`）；三個一起由 `run_v13.bat` 在筆電訓練、判定 | LSTM-R IC 0.0936、CNN-R 0.0847（v8 的 LSTM-B 0.1013、v11 選股觀點 0.1105）；v11 + LSTM-R：val −0.19、hold −0.52、post +4.02；v11 + CNN-R：val −0.02、hold −1.36、post +0.92；都沒入選。F-LOAM：等筆電跑完 | 只有研究，live 沒改（`v13\README_v13.md`） |
| v14 | `v14\` | 還有哪些量化方法能用：對照表（第 1 節）＋五組沒做過的方法，規則事先寫好（`research\PREREG_v14.md`），用存好的 walk-forward 預測判定、不重訓：E2 PCA 統計套利（Avellaneda-Lee s-score、殘差反轉）、E3 產業／統計因子中性、E4 季底作帳（投信）、E5 14 個選股模型 1/N；E1 開盤價（盤前試撮）的上限只報告；沙盒 `research\v14_try.py` | 入選 E4a、E5，合起來 val +0.35、hold −0.58、post +3.07，沒通過；E2 −1.4 到 −1.5、E3 −2.3 到 −3.7（t 到 −3.8）；知道開盤價也只 +1.08（v11）、+0.24（W0） | 只有研究，live 沒改（`v14\README_v14.md`） |
| v15 | `v15\` | 去噪方法和 Random Forest：文獻搜尋＋ 13 個候選（規則事先寫好，`research\PREREG_v15.md`）：RF、ExtraTrees 當新成員；lgb_res 的 6 種去噪取代 lgb_res、和重訓的對照比；MLP 的 2 種去噪取代 mlp_res_xr9、和重訓的對照比；Kalman、因果 wavelet 訊號；Vasicek beta 收縮 | 入選 A1、B2、B3、B4、B6、D2、E1，合起來 val +0.24、hold −0.79、post −1.63，沒通過；RF IC 0.1058（lgb_res 0.1050），預設 RF 0.0698；最穩的 E1 全期 +0.21（t 1.78） | 只有研究，live 沒改（`v15\README_v15.md`） |
| v16 | `v16\` | Kalman filter 會不會 work：Kalman 在 pipeline 能估的量的地圖（第 1 節）＋三個沒做過的用法，規則事先寫好（`research\PREREG_v16.md`），不重訓：K1 Kalman beta（對沖；隨機漫步或均值回歸，參數用濾波器自己的 likelihood 選）、K2 時變的大盤觀點係數（TVP regression）、K3 每天在 v11 和 W0 之間切換（local level）；理論：q = 0 等於固定參數、120 天 OLS 等於記憶 60 天的 Kalman、任何濾波器偵測變化的天數下限 | K1 −0.05 / −0.11 / −0.17（val / hold / post）、K2 −0.72 / 0 / −0.16、K3 −0.03 / 0 / 0，都沒入選；K1 的記憶 52–58 天 ≈ 120 天 OLS，beta 預測沒有變好；K2、K3 的 MLE 大多選固定參數，強迫調快更差；第二部分 smoother：合法的和 filter 相同，偷看版的 +8.6 是未來的資訊，平滑 label（S3）照規則通過但不建議 | 只有研究，live 沒改（`v16\README_v16.md`） |
| race_std | `race ver\race_std\` | 合併四套系統的驗證標準 RACE-STD v1：U 111、P 61、C 169（含自己爬的 40 個）、M 39、ALL，各配 5 個 learner，共 32 個候選在 D 比；前 3 名＋它們的平均＋4 個 native 進 V，V 只用一次 | D 前 3：ALL.LN 15.32、ALLmU.LN 15.13、C.LU 15.10；V：C.LU 16.16、ENS_TOP3 16.14、C.LC 15.22、U-native 14.17、FinPred 11.94、M2 9.60；大盤觀點 D −2.07、V −0.76，不加 | 只有研究；race 版每天的程式在 `race ver\race_live\`（`race ver\race_std\README_race_std.md`） |
| race_live | `race ver\race_live\` | race 版每天的程式：C 組 169 個 feature（含爬的 40 個）＋ LU（lgb_res 的 LightGBM），每天重訓（最近 3 個交易日各一個模型、機率平均），Top25 ＋除權息規則；每天重訓的 walk-forward（規則事先寫好，只用 D） | 每天重訓 16.35（2022 18.13、2023 11.83、2024 19.00）；每月重訓 16.67、每季 16.14、年度 15.10；每天 − 每月 −0.32（SE 0.41）→ 用每天重訓 | 可用，11/16 起（`race ver\race_live\README_race_live.md`） |
| race_mkt | `race ver\race_mkt\` | 任何數量 0/1 的大盤觀點（DL1 = sign(μ̂ + ĉq) 等 5 種決策 × 5 種大盤模型）＋ 37 個期貨籌碼 feature（大台／小台／微台三大法人、散戶多空比、大額交易人、選擇權、P/C），規則事先寫好；第二部分把期貨籌碼放進選股模型 | D：8 個版本差 −0.36 到 +0.32（t ≤ 0.35），沒有入選；V −1.21 到 +1.47；大盤 OOS 相關 MB 0.117、MF 0.123；CF.LU − C.LU −0.38（t −0.70） | 只有研究，race 版不變（`race ver\race_mkt\README_race_mkt.md`） |
| race_news | `race ver\race_news\` | 鉅亨網台股新聞 12 個 feature（隔夜篇數、公司新聞、異常注意力、tone、關鍵字、全市場）放進選股模型，規則事先寫好 | CN.LU − C.LU −0.29（t −0.86），IC 0.1207 → 0.1209；新聞 feature 只占 1.9% gain | 只有研究，race 版不變（`race ver\race_news\README_race_news.md`） |
| race_agg | `race ver\race_agg\` | 課程 topic 6 / W12 的 aggregation 方法放在 race 版上：uniform / linear / any blending（C 組 5 個 learner）、bagging（10 seed、整天 bootstrap）、AdaBoost-Stump / DTree、指數損失 GBDT、單一 C&RT（cost-complexity pruning）、categorical 日曆、RF、RF ＋ LU、permutation 選 feature、squared-loss GBDT、分數 × absres_60；OOB 診斷；規則事先寫好、只用 D | 16 個都沒過：A3 linear blending +0.77（t 1.55）、A1 uniform −0.01（IC +0.0067，t 2.41）、B1 −0.46、C1 −4.42、C2 −5.09、D1 −6.89、E1 −2.58、F1 −0.97；seed 1–3 的 D 分數比 3-seed 期望高約 0.45；row 層級 OOB 的 IC 0.22 對測試年 0.11–0.14 | 只有研究，race 版不變（`race ver\race_agg\README_race_agg.md`） |

`v9\out\` 裡的 static 檔：`submission_v9_static_w025.csv`（上傳中，18.84）、`submission_v9_static.csv`（w 0.5，16.42）、`submission_v9_static_wt0_lam05.csv`（21.53，`v6\others\` 裡的檔名是 v9.2）、`submission_v9_static_max.csv`（32.18，組內檔名 v9-3；係數是在 static 上搜尋出來的，同一組係數在 walk-forward 反而比 live 設定低 3.7）。

各版本的細節、證據和數學證明在各自資料夾的 README（v1 是 `transformer\README.md`，v2、v3、v4 是 `棄案\v2\README_v2.md`、`棄案\v3\README_v3.md`、`棄案\v4\README_v4.md`）。舊 README 裡寫的 `v2\`、`v4\`、`_backup_20261025\`、`_backup_20261025_typhoon\`、`v9\models_dry\`、`v9\out_dryrun\`，現在都在 `棄案\` 底下同樣的相對位置（例如 `v4\out\` → `棄案\v4\out\`）。

## 資料夾

| 資料夾或檔案 | 內容 | 注意 |
|---|---|---|
| `data\` | 競賽資料（`market\`、`global\`、`info.json`）和 `ref\`（交易日曆、除權息） | 每天由 `update_data.py` 更新 |
| `data\ref\exdiv_fetch_log.csv` | 每次抓除權息預告的成功或失敗紀錄 | 防呆會讀 |
| `data\ref\closures_manual.csv` | 臨時休市日（日曆上有、實際沒開盤，例如 10/23 颱風；11/16 加了 9/25、9/28、10/9、10/26 四天國定假日，因為那天的課程 zip 多了它們的 -1 列）；`update_data.py` 和夜盤對齊會讀 | 手動維護；`calendar_official.parquet` 是拿掉休市日之前的官方日曆 |
| `data_backup_original\` | `update_data.py` 第一次執行時做的原始資料備份 | 不要移動（`v6\leak_check.py` 也用它找原始資料的最後一天） |
| `downloads\` | 每天的 zip；`archive\` 留著每一個用過的 zip（`data_<預測日>.zip`） | |
| `leaderboard\` | live 排行榜的截圖和表 | 每天記，11/23 定版要用 |
| `v9\` | live 正式版：程式、模型池 `models_v9\`、輸出 `out\`、`data\panel_ext.parquet`、`features_v3.py`、研究 `research\` | |
| `transformer\`、`v5\`、`v7\`、`v8\` | 舊版本，但裡面有 v9 每天要用的程式或資料（見下一節），所以整個資料夾留在原位 | 不要移動 |
| `v7\data\tx_daily.csv` | 夜盤資料，`daily.bat` 每天更新；`v7\data\archive\` 每天一份快照 | v9 在用，不要刪 |
| `v6\` | static 工具和組員的檔（`others\`）；live 的本地檢查 `phase2.bat`，輸出在 `phase2\`（`files\` 放要一起算的檔） | |
| `v10\`、`v11\`、`v12\` | v10：Alpha158 研究（沒通過）。v12：ranking objective 的實驗（只有研究）。v11：Alpha158 重挑（沒通過）＋ v11 模型（`train_and_predict_v11.py`、`models_v11\`、`data\pbar_history.csv`，用 v9 的模型池） | 用 v11 時不要刪 `v11\models_v11\`、`v11\data\`；`v11\cache\_stage\` 是傳檔到雲端用的暫存 tar，可以刪 |
| `overfit\` | overfit 檢查（README、`research\` 程式、`results\` 結果；`cache\o1\` 的重訓沒有複製過來，重算會自己重訓） | 只有研究，可以整個刪掉 |
| `reg_svm\` | regularization 和 SVM 的研究（README、`research\` 程式、`results\` 結果）；exact SVM 的 `run_svm_exact.bat`，預測在 `cache\svm_exact\` | 只有研究，可以整個刪掉（`cache\svm_exact_test\` 是測試輸出） |
| `v13\` | LSTM、CNN、F-LOAM 的研究（README、`research\` 程式、`results\` 結果、`run_v13.bat`）；雲端 60 個 unit 的預測在 `cache\seq_raw_cloud\`，筆電訓練的在 `cache\seq_raw\` | 只有研究，可以整個刪掉（`cache\_stage\` 是傳檔到雲端用的暫存，可以刪） |
| `v14\` | 量化方法對照表和五組實驗（README、`research\` 程式、`results\` 結果、`run_v14.bat`）、沙盒 `research\v14_try.py`；`cache\` 是訊號（約 20 MB，重跑會重建） | 只有研究，可以整個刪掉；E5 會讀 `v12\`、`v13\`、`reg_svm\` 的 cache（刪掉時 E5 跳過） |
| `v15\` | 去噪方法和 RF 的研究（README、`research\` 程式、`results\` 結果、`run_v15.bat`）；訓練好的預測在 `cache\units\` | 只有研究，可以整個刪掉（`cache\_stage\` 是傳檔到雲端用的暫存，約 180 MB，可以刪） |
| `v16\` | Kalman filter 的研究（README、`research\` 程式、`results\` 結果、`run_v16.bat`）；`cache\signals16.pkl` 是 K1 的 beta（約 10 MB，重跑會重建） | 只有研究，可以整個刪掉（`cache\README_top_backup_20261108.md` 是改最上層 README 之前的備份） |
| `race ver\race_std\` | 合併實驗：README、`research\`（程式、PREREG）、`results\`（結果；`SEALED_V.json` 是 V 唯一一次的結果）、`ext\`（爬的資料）、`cache\`（panel 和 units，約 300 MB）、`run_fetch.bat`、`run_race_std.bat` | 只有研究；不要再跑 `stage2_validate.py`（`--audit` 可以） |
| `race ver\race_live\` | race 版每天的程式（`race_daily.bat`、`race_evening.bat`、`race_mode.txt`）、輸出 `out\`、模型 `models\`、爬的資料 `ext\`（每天更新）、walk-forward 與參考表的結果 `results\` | `ext\`、`models\`、`race_mode.txt` 每天要用，不要移動 |
| `race ver\race_mkt\` | 大盤觀點與期貨籌碼的研究：README、`run_fetch_mkt.bat`（Windows 下載）、`ext\`（期貨與選擇權資料，約 13 MB）、`research\`（程式、PREREG）、`part2\`（期貨籌碼進選股）、`results\`（`SEALED_V.json` 存在就不再算 V）、`cache\`（第二部分的 units、改最上層 README 之前的備份） | 只有研究，可以整個刪掉 |
| `race ver\race_news\` | 中文新聞研究：README、`run_fetch_news.bat`（Windows 下載）、`ext\cnyes\`（每月一個 CSV，約 60 MB）、`research\`（程式、PREREG）、`results\`、`cache\`（units、改最上層 README 之前的備份） | 只有研究，可以整個刪掉 |
| `race ver\race_agg\` | aggregation 方法的研究：README、`research\`（程式、PREREG）、`results\`（結果）、`cache\units_agg\`（132 個 unit 與 E4 的 importance，約 20 MB）、`run_race_agg.bat`；`cache\_stage\` 是傳檔用的 tar（約 300 MB）、`cache\README_top_backup_20261118.md` 是改最上層 README 之前的備份 | 只有研究，可以整個刪掉（`cache\_stage\` 可以先刪） |
| `棄案\` | 沒在用的東西：v2、v3、v4、v1 的舊檔、v9 的測試輸出、舊備份（`_backup_20261025\`、`_backup_20261025_typhoon\`、`_backup_20261026\`）；每一項為什麼不用、還有誰會讀，寫在 `棄案\README.md` | v9 每天的流程不讀；`backup_v9.py` 不備份 |
| `docs\` | 課程投影片 | |
| `sample_submission.csv` | 上傳格式範本（v9 用它的 50 支股票代號和順序） | 不要移動 |

## v9 每天的流程用到的檔（不要刪除、移動或改名）

- `v9\`：`rolling9.py`、`train_and_predict_v9.py`、`monitor_v9.py`、`features_v3.py`、`check_zip_fresh.py`（10/27 加，只印警告）、`data\panel_ext.parquet`、`models_v9\`
- `v5\`：`ens_lib.py`、`rolling.py`、`guards.py`、`features_extra.py`、`fetch_exdiv.py`
- `transformer\`：`comp_features.py`、`tfmodel.py`、`update_data.py`、`final_feature_lists.json`
- `v7\`：`fetch\fetch_night_session.py`、`data\tx_daily.csv`
- `v8\`：`hedge_beta.py`、`monitor_market.py`
- `data\`、`sample_submission.csv`

## 備份

`evening.bat` 最後會執行 `v9\backup_v9.py`，把整個資料夾（不含 `棄案` 和研究用的 cache）複製到 `%USERPROFILE%\ML_RACE_backup`（C 槽）。只複製新的或改過的檔，不會刪檔（所以之前備份過的 v2、v4 還留在備份裡）。想備份到隨身碟或私人雲端：`python backup_v9.py E:\ML_RACE_backup`。競賽資料不能分享給沒修課的人，備份請放在私人的地方。

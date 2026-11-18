10/30 的上傳檔和 AR（2026-11-01 補做：10/30 早上沒有跑 daily）

結果
- 三個 10/30 的檔，就是那天早上跑 v11\daily_v11.bat 和 custom_v11.bat（w_T 0、λ 0.5）會寫出的檔：
    v9\out\submission_v9_1030.csv                 v9（最上層 README 的 live 版）
    v11\out\submission_v11_1030.csv               v11 預設
    v11\out\submission_v11_1030_wt0_lam0.5.csv    v11、w_T 0、λ 0.5（W0）
  旁邊有同名的 probabilities、checks、hedge；checks 最後一行 [INFO] 註明是 11/1 補做的。
- 10/30 的 AR（v6\phase2_check.py --date 2026-10-30 --source course，價格是 downloads\data (6).zip 的 adj_open、adj_close；
  每支的貢獻在 v6\phase2\report_20261030.txt）：

     AR score    AR bp   timing    xsec   猜 1   猜對   檔案
     0.002962    29.62    19.13   10.49   0.62  31/50   v11 W0
     0.001497    14.97    12.75    2.22   0.58  29/50   v11 預設
    -0.002143   -21.43    -9.57  -11.86   0.44  26/50   v9
    -0.001918   -19.18   -25.51    6.33   0.34  27/50   (baseline) Previous day
    -0.007971   -79.71   -79.71    0.00   0.00  18/50   (baseline) Always down

  另外寫的 scripts\ar_check.py 算出來每個數字都相同（ar_check_1030.txt）。
- 當天盤勢是開低走高：Top50 開盤比 10/29 收盤平均低 230.7 bp（39/50 開低），盤中平均 +79.7 bp（32/50 收高於開）。
  v9 的 pbar 0.465，猜 1 只有 0.44，timing 吃虧；v11 的 T2 夜盤反轉（夜盤 -1.41%，z -1.06）把 pbar 拉到 0.496，猜 1 0.58；
  W0 沒有大盤觀點，猜 1 0.62，選股（xsec +10.49）也最好。只有一天，不代表哪個設定比較好。
- 10/30 沒有上傳，所以 v6\phase2\uploaded.csv 沒有加 10/30；scores.csv 多了這三個檔 10/30 的分數。

怎麼做的（只用 10/30 09:00 以前拿得到的資料）
- 課程資料：downloads\data (6).zip（10/30 22:05 打包，已經有 10/30 的答案）。scripts\make_zip_asof.py 把每個檔 10/30 之前的
  row 逐字元保留，10/30 那一 row 改成早上 07:00 zip 的樣子：個股全部 -1；global 的 ^VIX ^IXIC ^GSPC 照 zip 裡 10/30 那一 row
  （= 10/29 美股收盤：VIX 17.01、Nasdaq 26184.72、S&P 7720.39），^TWII ^KS11 TX_OI = -1；10/30 之後沒有 row。
  再用 transformer\update_data.py 合併（checks OK；10/23 颱風那一 row 照 closures_manual.csv 拿掉）。
- 夜盤：v7\data\tx_daily.csv 加上 FinMind TaiwanFuturesDaily TX 的 10/29 日盤（position）和 10/30 夜盤（after_market，
  10/29 15:00 到 10/30 05:00）共 23 行（tx_rows_added_1030.csv），也就是 10/30 早上 fetch_night_session.py 會多抓到的部分；
  10/30 的日盤和之後的夜盤沒有加。近月 202611：10/29 日盤收 46577、夜盤收 45926，night_ret -1.41%。
  FinMind 這次回傳的 10/29 夜盤和筆電 10/29 早上抓的逐行相同。
- 除權息：用 10/29 早上抓的表（Top100 預告 0 筆）；10/30 沒有重抓（雲端連不到 TWSE），所以 checks 有一行 WARN
  「1 個交易日前的表」（程式允許 3 天）。
- 模型：在雲端的資料夾複本上跑原本的程式（python 3.13.18、pandas 3.0.5、numpy 2.5.3、torch 2.14.1、lightgbm 4.7.0）。
  scripts\run_asof.py 只把兩個檢查的「今天」設成 10/30（is in the past、除權息表的天數），其他完全是原程式：
    v9   --dates auto                                          9 個成員各訓練到 10/29（seed 26102901、26102902），再預測
    v11  --dates auto                                          lgb_res_xrN 訓練到 10/29；T2 z -1.06；T3 不作用（不是月初）
    v11  --dates auto --retrain none --w-t 0 --lam 0.5         和 custom_v11.bat 相同
  紀錄在 logs\r1030_*.log。
- 筆電上的模型池（v9\models_v9、v11\models_v11）、v9\out 和 v11\out 的 history_predictions.csv、v11\data\pbar_history.csv
  都沒有改。下一次 evening / daily 照常跑，會直接訓練到最新一天；池裡只是少了 10/29 這個點（MLP 和 LightGBM 的 5 個點
  變成 10/21、10/22、10/27、10/28、10/30），影響很小。

檢查
- 同一套雲端環境重跑 10/29（verify_1029.txt）：把複本池裡 10/28 的模型拿掉、用同樣的 seed 重訓，再預測 10/29。
  v9、v11、W0 三個檔和筆電 10/29 早上寫的檔 50 支全部相同；probabilities 最多差 1e-6（檔案只到小數 6 位）；
  MLP 權重差 1e-7 到 1e-5、Transformer 1.7e-4，best epoch 都相同；LightGBM 樹的數目和葉子相同。
- 10/30 每個決定離翻轉的距離（margins_1030.txt）：對沖選的 c 下 |score + c·beta| 最小 5.8e-4 到 7.4e-4；整個對沖 grid 上
  最小 5e-6，也是在不會被選到的 c；選到的 c 和下一個 grid point 有明顯差距。所以換成筆電跑，三個檔會一樣。
- 沒有 leak：10/30 的個股 row 全是 -1、美股是 10/29 晚上的收盤、夜盤只到 10/30 05:00、模型的 label 只到 10/29
  （log：labelled data up to 2026-10-29；每個成員 trained up to ..2026-10-29）。
- 沒辦法確認的一點：10/30 早上 07:00 那份 zip 裡 10/30 row 的美股數字，這裡用的是 22:05 那份的數字。
  對照：10/29 早上 07:00 的 zip 和 10/30 22:05 的 zip，10/29 那一 row 的美股數字逐字相同。

檔案（v11\results\rerun_1030\）
- README.txt（這個檔）、ar_check_1030.txt、verify_1029.txt、margins_1030.txt、tx_rows_added_1030.csv
- scripts\make_zip_asof.py、run_asof.py、ar_check.py；logs\r1030_*.log（10/30）、logs\r1029_*.log（10/29 重跑）
- 傳到雲端用的 v11\cache\_stage\rerun_1030_in.tar.gz（約 100 MB）可以刪。

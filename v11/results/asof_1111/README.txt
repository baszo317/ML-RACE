11/11 的預測和 AR（2026-11-11 晚上補做：11/11 早上沒有跑 daily）

結果
- 三個 11/11 的檔，就是那天早上跑 v11\daily_v11.bat 和 custom_v11.bat（w_T 0、λ 0.5）會寫出的檔：
    v9\out\submission_v9_1111.csv                 v9
    v11\out\submission_v11_1111.csv               v11 預設
    v11\out\submission_v11_1111_wt0_lam0.5.csv    v11、w_T 0、λ 0.5（W0，10/27 起上傳的設定）
  旁邊有同名的 probabilities、checks、hedge；checks 最後一行 [INFO] 註明是 11/11 晚上補做的。
- 11/11 的 AR（v6\phase2_check.py --date 2026-11-11，價格是 downloads\data (13).zip 的 adj_open、adj_close；
  和 FinMind 的原始開收盤比對，50 支都一致，最大差 6.7e-16；每支的貢獻在 v6\phase2\report_20261111.txt）：

     AR score    AR bp   timing     xsec   猜 1   猜對   檔案
     0.000137     1.37    34.78   -33.41   0.62  29/50   v11 W0
     0.000137     1.37    34.78   -33.41   0.62  29/50   v11 預設（0/1 和 W0 完全相同）
    -0.003188   -31.88     5.80   -37.68   0.52  26/50   v9
    -0.003701   -37.01    -5.80   -31.22   0.48  22/50   (baseline) Previous day
    -0.014492  -144.92  -144.92     0.00   0.00  10/50   (baseline) Always down

  另外寫的 scripts\ar_check_1111.py 不經過 phase2_check.py，算出來每個數字都相同（logs\ar_check_1111.txt）。
- 當天大漲：Top50 盤中平均 +144.92 bp，40/50 收高於開。W0 猜 1 的比例 0.62，timing +34.78；
  選股是負的（xsec -33.41）：猜 0 的 19 支平均 +188.9 bp，猜 1 的 31 支平均 +118.0 bp。
  對沖把 5 支從 0 改成 1（2379、2395、2885、3293、3653），合計 +0.36 bp；不對沖是 +1.01 bp。
- v11 預設：夜盤 -0.24%（z -0.17），T2 把 pbar 從 0.497 調到 0.502；對沖後的 0/1 和 W0 相同。
- uploaded.csv 沒有改（這個檔是收盤後才做的）。scores.csv 在最後加了這三個檔和兩個 baseline 的 11/11 分數
  （inputs\scores_rows_added_1111.csv 這 5 行；舊的 row 逐字元沒動）。

怎麼做的（只用 11/11 09:00 以前拿得到的資料）
- 課程資料：downloads\data (13).zip（11/11 18:18 打包，已經有 11/11 的答案）。scripts\make_zip_asof.py 把每個檔 11/11 之前的
  row 逐字元保留，11/11 那一 row 改成早上 07:00 zip 的樣子：個股全部 -1；global 的 ^VIX ^IXIC ^GSPC 照 zip 裡 11/11 那一 row
  （= 11/10 美股：VIX 19.83、Nasdaq 25065.875、S&P 7520.93），^TWII ^KS11 TX_OI = -1。
  做出來的 inputs\data_asof_1111.zip 和 03:09 下載的 data (12).zip（00:00 打包）只差 global 那一 row 的三個美股數字。
  同一個程式從 data (12).zip 做 11/10 的 zip，和真的 11/10 07:00 zip（data (11).zip）101 個檔逐 byte 相同。
  再用 transformer\update_data.py 合併（checks OK）。
- 為什麼要等 data (13).zip：data (12).zip 是 00:00 打包的，11/11 那一 row 的美股還是 -1（daily 會 FATAL）。
  S&P 可以從 Yahoo 補（11/04、11/05、11/06、11/09 的收盤和課程資料逐位相同），Nasdaq 差 0.006 以內，
  但課程資料平日的 VIX 是盤中的數字，不是收盤：11/11 那一 row 19.83（Yahoo 11/10 收盤 19.46），
  11/10 那一 row 23.97（Yahoo 11/09 收盤 19.92）；只有週一那一 row（週五的值）和收盤相同。
- 夜盤：v7\data\tx_daily.csv 加上 FinMind TaiwanFuturesDaily TX 的 11/10 日盤（position）和 11/11 夜盤（after_market，
  11/10 15:00 到 11/11 05:00）共 24 行（inputs\tx_rows_added_1111.csv），也就是 11/11 早上 fetch_night_session.py 會多抓到的部分；
  11/11 的日盤沒有加。近月 202611：11/10 日盤收 42623、夜盤收 42522，night_ret -0.24%。
  FinMind 這次回傳的 11/10 夜盤和筆電 11/10 抓的 14 行相同。只影響大盤觀點（v9、v11 預設），W0 不用。
- 除權息：用 11/10 抓的表（Top100 預告 1 筆：2344 華邦電 11/18 除息，和 11/11 無關）。沒有重抓，所以 checks 有一行 WARN
  「1 個交易日前的表」（程式允許 3 天）。
- 模型：在雲端的資料夾複本上跑原本的程式（python 3.13.18、pandas 3.0.5、numpy 2.5.3、torch 2.14.1、lightgbm 4.7.0）。
  scripts\run_asof.py 只把兩個檢查的「今天」設成 11/11（is in the past、除權息表的天數），其他完全是原程式（scripts\run_1111.sh）：
    v9   --dates auto                                    9 個成員各訓練到 11/10（seed 26111001、26111002），再預測
    v11  --dates auto                                    lgb_res_xrN 訓練到 11/10；T2 z -0.17；T3 不作用（不是月初）
    v11  --dates auto --retrain none --w-t 0 --lam 0.5   和 custom_v11.bat 相同
  池是 10/30、11/02、11/06、11/09、11/10 五個點：前四個是筆電的模型檔，11/10 是雲端訓練的。紀錄在 logs\r1111_*.log。
- 筆電上的模型池（v9\models_v9、v11\models_v11）、v9\out 和 v11\out 的 history_predictions.csv、v11\data\pbar_history.csv、
  v7\data\tx_daily.csv、data\ 都沒有改。明天照常下載 zip、跑 daily，會直接訓練到 11/11；池裡只是少了 11/10 這個點
  （5 個點變成 10/30、11/02、11/06、11/09、11/11），影響很小。

檢查
- 同一套雲端環境重跑 11/10、只預測（筆電的池，logs\ver1110_*.log）：v9、v11、W0 三個檔的 0/1 和 hedge 和筆電 11/10 的檔
  50 支全部相同；probabilities 最多差 1e-6（檔案只到小數 6 位）。
- 訓練的重現（logs\verify_train_1110.txt）：把複本池裡 11/09 的點拿掉，用同樣的 seed 在雲端重訓，再預測 11/10：
  三個檔的 0/1 仍然和筆電相同。MLP 權重差 1e-7 到 2e-6，Transformer 1e-4。LightGBM 不是逐位相同：3 個 feature 的值在
  Windows 和 Linux 差最後一位（相對 2.6e-16），lgb_res 的 early stopping 筆電停在 72 棵、這裡 71 棵。
  W0 的 score 因此差中位數 2.8e-5、90% 分位 5.2e-5、最大 1.3e-4（全部來自 LightGBM）。
- 11/11 每個決定離翻轉的距離（logs\margins_1111.txt）：W0 選到的 c = 0.0025，|score + c·beta| 最小是 2885 的 6.5e-5，
  第二小是 3293 的 7.3e-4。2885 的距離大於上面跨環境差異的 90% 分位、小於最大值，所以在筆電上跑，2885 有小機率變成 0：
  那樣 c 不變、猜 1 的比例 0.60、AR 是 -6.07 bp（2885 當天 +186 bp；logs\flip_2885.txt）。
  其他 49 支離翻轉都在 7e-4 以上，換成筆電跑也不會變。v11 預設最小 1.2e-3，v9 最小 9.0e-4。
- 沒有 leak：11/11 的個股 row 全是 -1，美股是 11/10 晚上的數字，夜盤只到 11/11 05:00，模型的 label 只到 11/10
  （log：labelled data up to 2026-11-10；每個成員 trained up to ..2026-11-10）。

檔案（v11\results\asof_1111\）
- README.txt（這個檔）
- inputs\：data_asof_1111.zip（重建的 11/11 07:00 zip，不要放進 downloads\，免得 update_data.py 把它當成最新的 zip）、
  tx_rows_added_1111.csv、prices_finmind_2026-11-11.csv（也複製成 v6\phase2\prices\prices_2026-11-11.csv）、scores_rows_added_1111.csv
- scripts\：make_zip_asof.py、run_asof.py、run_1111.sh、ar_check_1111.py、margins_1111.py、verify_train_1110.sh、verify_train_compare.py
- logs\：r1111_*.log（11/11）、ver1110_*.log（11/10 只預測）、vt1110_*.log（11/10 重訓）、phase2_1111.log、ar_check_1111.txt、
  margins_1111.txt、verify_train_1110.txt、flip_2885.txt
- 傳檔用的 v11\cache\_stage\asof_1111_in.tar.gz（108 MB）和 asof_1111_out.tar.gz 可以刪。

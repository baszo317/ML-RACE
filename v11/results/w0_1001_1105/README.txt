W0（v11、w_T 0、λ 0.5，10/27 上傳的 submission_v9.4_1027_wt0_lam0.5 那一組）2026-10-01 到 11-05 的 AR（2026-11-05 算）

結果（Top50 bp/日；AR score = bp / 10000；價格 = downloads\data (9).zip 的 adj_open、adj_close，50 支）

  期間                          天數   合計 AR score   平均 bp   每日 sd   > 0 的天數   Always down 平均   Previous day 平均
  10/01-10/22（as-of 重建）       15     +0.058997      +39.33     59.3      12            -18.79             -6.36
  10/27-11/05（W0 選定之後）       8     +0.006985       +8.73     66.4       4            +90.17             -5.55
    其中你自己的 6 個檔            6     +0.003998       +6.66     73.1       3            +57.75            -43.17
  10 月（10/01-10/30）            19     +0.071159      +37.45     61.7      15             -9.45            -11.02
  11 月（11/02-11/05）             4     -0.005176      -12.94     51.5       1           +154.79            +17.42
  全部（10/01-11/05）             23     +0.065982      +28.69     62.1      16            +19.11             -6.07

  每天的 AR、timing、xsec、猜 1 比例、猜對、兩個 baseline：w0_daily_ar.csv

每一天的檔從哪來（都只用那天 09:00 以前拿得到的資料）
- 10/27、10/28、10/29、10/30、11/02、11/03：你的 W0 檔（v11\out\，10/30 是 11/1 補做的那一個）。
  10/29、11/02 在 v6\phase2\scores.csv 只算了 48 支（TWSE / FinMind），這裡用課程 zip 的 50 支：
  10/29 +125.64（原本 136.71）、11/02 -61.11（原本 -55.93）；另外 4 天和 scores.csv 相同。
- 10/01-10/22：那時還沒有 W0 的檔，用和 live 相同的時程重建。
  v9 的 9 個成員：v9\out\members_v9_static.csv.gz（rolling：每天只用前一個交易日以前訓練的模型）。
  X1（lgb_res_xrN）：照同一個時程在雲端訓練 19 個點（9/22-10/21，每點 1 個 seed yymmdd01，平均最近 5 點），x1_oct_pool.csv。
  組合用上線程式本身：rolling9.combine（w_T 0）-> train_and_predict_v11.shift_month_start（T3：10/1、10/2、10/5）
  -> rolling9.decide（λ 0.5）。T2 在 w_T 0 時不影響 score。
- 11/04、11/05：那兩天沒有產生檔。在雲端的資料夾複本上照 daily_v11.bat + custom_v11.bat 的順序跑原程式
  （v11\results\rerun_1030\scripts\run_asof.py）：make_zip_asof.py 從 data (9).zip 做出那天早上 07:00 的 zip
  -> update_data.py -> v9（9 個成員訓練到前一天）-> v11（X1 訓練到前一天）-> --retrain none --w-t 0 --lam 0.5。
  11/04 用的池是 10/27-11/03 的點，11/05 是 10/28-11/04，就是筆電那兩天有跑的話會有的池。
  兩樣沒有重抓：除權息表用 11/03 抓的（Top100 0 筆）；夜盤（雲端連不到 FinMind）只影響大盤觀點，W0 不用大盤觀點。
- 筆電上的模型池、history_predictions.csv、pbar_history.csv、v9\out、v11\out 都沒有改。

檢查
- 重組：同一套程式算 v9（w_T 0.5、λ 0.75）和 v9 w_T 0、λ 0.5 的 rolling：99 天 17.11 / 23.94、10 月 21.00 / 37.43，
  和 v9\out\static_report_*.txt 相同。
- X1：在雲端重訓筆電池裡的 10/22 s1、10/27、10/28、10/30、11/02 五個點，預測和筆電的模型完全相同（最大差 0）。
- 整個 as-of 流程：同樣做法重跑 11/03，v9、v11、W0 三個檔和筆電 11/03 的檔 50 支全部相同，probabilities 差 0；
  從 data (9).zip 做出來的 11/03 zip 和真的 11/03 07:00 zip（data (8).zip）101 個檔逐 byte 相同。
- 11/04、11/05 W0 的決策離翻轉：選到的 c 下 |score + c x beta| 最小 1.2e-3、5.8e-4（數值誤差約 1e-6），換成筆電跑結果相同。

要注意的兩點
- W0 的係數是 10/26 看過 static（6-9 月）和 10 月的成績之後挑的（README_v11 第 6、7 節、overfit\README_overfit.md），
  所以 10/01-10/22 的 +39.33 是挑選時看過的資料，會偏高。挑完之後真正樣本外的只有 10/27 以後 8 天：+8.73 bp/日，
  但每日 sd 66，8 天的平均誤差約 +-23，看不出和 0 的差別。
- custom_v11.bat 印的研究版 walk-forward（Jun-Oct 的模型凍結在 5/29）W0 10 月是 28.57；這裡的 39.33 是每天重訓的版本
  （v9 wt0_lam05 也一樣：凍結 31.43、每天重訓 37.43）。
- 10/01-10/22 拿掉 T3 是 32.19、拿掉 X1 是 39.04（oct_w0.log）。

檔案
- README.txt（這個檔）、w0_daily_ar.csv
- 10 月重建：submission_w0_october_asof.csv（15 天，上傳格式）、probabilities_w0_october_asof.csv、hedge_w0_october_asof.csv、x1_oct_pool.csv
- 11/04、11/05：submission_v11_1104_wt0_lam0.5.csv、submission_v11_1105_wt0_lam0.5.csv 和同名的 probabilities、hedge、checks
  （放在這裡，不放 v11\out\，免得和真的上傳檔混在一起）
- scripts\：oct_x1.py、oct_w0.py、run_nov.sh、verify_x1.py、final_table.py
- logs\：oct_x1.log、oct_w0.log、verify_x1.log、nov_*.log（11/04、11/05 的 update / v9 / v11 / W0）、chk1103_*.log、final_table.log
- 傳檔用的 v11\cache\_stage\w0_asof_1105_in.tar.gz（約 111 MB）可以刪。

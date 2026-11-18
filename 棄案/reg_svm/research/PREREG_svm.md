# exact RBF SVM 當額外的選股成員：事先寫好的規則（2026-11-04 17:00，exact 版任何成績算出來之前）

使用者的要求（11/4）：先問「需不需要 regularization、換成 SVM 會不會比較好」，看過時間估計後決定「給我 exact 版，考慮把它弄成額外成員」。
v9、v11 的檔案都不改。程式：`reg_svm\research\svm_exact.py`（訓練）、`svm_exact_eval.py`（判定）；預測存在 `reg_svm\cache\svm_exact\`。
exact 版要在使用者的筆電上跑 3–12 小時，所以這份規則寫在任何 exact 全量成績出來之前。

## 0. 寫這份規則時已經知道的

- 成員層級（雲端，和 lgb_res 同樣的 feature、label、權重、fold）：
  - linear SVM（hinge loss + L2）全期 rank IC 0.0964（C 用最後 120 天的 hinge loss 選）；logistic regression 0.0947；lgb_res 0.1050。
  - Nystroem 近似的 RBF SVM（1000 維），2022-01 到 2026-05 的 5 個 fold：C 0.1 時 0.1064；同期間 lgb_res 0.1072、mlp_res_xr9 0.1079。
    和 lgb_res 每天的相關 0.78–0.87。
  - linear SVM「取代」v9 的 LightGBM：全期 −0.69 bp/日（t −1.36）。
- exact SVC 的小樣本格點（只看了 fold 2024、2026-01；訓練用訓練窗扣掉最後 120 天的部分）：1 萬 row 時 C 10 < 1 < 0.1（越小越好）；
  最後 120 天（val）偏好 C 0.01–0.03，樣本外偏好 C 0.03–0.1；4 萬 row 時 val 仍然偏好 0.01–0.03。這是第 2 節格點和「C 不隨 row 數縮放」的依據。
- 速度：使用者的 i5-10210U 上 2 萬 row（C 1）fit 66 秒；support vector 佔 92–94%。
- 還沒有算過任何 SVM 當「額外成員」的 v9 / v11 AR。Nystroem 版的預覽在這份規則寫完之後才算，而且不影響這裡的判定。

## 1. 成員

| 項目 | 設定 | 和誰相同 |
|---|---|---|
| label | 有沒有贏過當天 100 支的平均（y_resid） | lgb_res |
| feature | B 組 67 個 | lgb_res |
| scaling | 用訓練窗 fit：winsorize 1% / 99%、z-score、缺值補 0 | transformer\tfmodel.Scaler |
| 樣本權重 | min(\|超額報酬\|, 5%)，平均 1（SVC 把每筆的 C 乘上權重） | lgb_res |
| kernel | RBF exp(−gamma \|x − x'\|²)，gamma = 0.5 / 67 | |
| walk-forward | 6 個 fold（2022、2023、2024、2025、2026-01–05、post 2026-06 以後），訓練窗 2020-03-02 到 fold 前一天 | 所有成員 |

沒有 seed：同樣的資料和參數，結果固定。

## 2. C 怎麼選（每個 fold 只用自己的訓練窗）

1. 格點 C ∈ {0.01, 0.03, 0.1, 0.3}。
2. 從訓練窗扣掉最後 120 個交易日的部分，隨機抽 2 萬 row（每個 fold 固定 seed）；每個 C fit 一次，在最後 120 天算 SVM 自己的損失
   （加權 hinge loss，權重同訓練），最小的 C 入選。每日 rank IC 只記錄、不參與選擇。（17:20 更正，見第 5 節）
3. 用整個訓練窗（4.5 萬到 15.2 萬 row）、同一個 C 重 fit，預測 fold 的每一天。C 不依 row 數縮放：SVC 的 hinge loss 是對 row 加總，
   C 不變時每筆分到的 regularization 1/(2Cn) 會隨 n 變小，這和線性模型最佳 regularization ∝ 1/n 一致（README 第 2 節）；
   小樣本格點裡 1 萬、4 萬 row 的 val 最佳 C 也相同。
4. 其他都不調。

## 3. 判定（和 v12 同一個規則）

1. SVM 的 decision value 每天減平均，縮放成 lgb_res 的每日標準差（同 X5、v12），放進 v11 的 LightGBM 部分：(lgb_res + X1 + SVM) / 3。
   **v11 + SVM 減 v11：val > 0 → 入選；入選而且 hold、post 都 > 0 → 通過。**
2. 通過才寫每天的程式（evening 要多 2–4 小時重訓，或改成每週重訓），上不上線由使用者決定。沒通過：v9、v11 都不變。
3. 只報告、不影響判定：SVM 自己的 rank IC 和相關、v9 + SVM（LightGBM 部分 (lgb_res + SVM) / 2）、W0 + SVM、SVM 當選股觀點的第三份。
4. 判定前先檢查重建的 v9、v11、W0 和公布的數字相同：13.99 / 14.33 / 16.59 / 14.31、14.60 / 14.34 / 20.35 / 15.02、11.79 / 10.35 / 21.71 / 12.22
   （val / hold / post / 全期）。不同的話先查原因，不判定。

## 4. 要知道的限制

- val（2022–2024）在 v5 到 v12 已經被用來挑過很多改法（`overfit\README_overfit.md`：65 個篩選過的改法，val 和 hold 的相關只有 0.16）。
  這次只有一個候選、一個判定，挑選的高估小；但每日 AR 的 sd 約 48 bp，1 bp/日的差大約只有 t 1，hold、post 的正負也有不少運氣。
- C 只用 2 萬 row、120 天挑，噪音大（小樣本格點裡 val 和樣本外偏好的 C 不同）。
- exact 一次要 3–12 小時，所以只跑這一組設定，不比較多組，也不事後換設定再判一次。

## 5. 更正（2026-11-04 17:20，exact 全量的任何成績出來之前）

第 2.2 步原本寫「最後 120 天的每日 rank IC 最高的 C」，改成「最後 120 天的加權 hinge loss 最小的 C」。理由：其他成員都用自己的損失選停止點
（LightGBM、MLP 的 early stopping 用 log-loss；v8 的 logistic regression 用 log-loss 選 C），雲端的 linear、Nystroem SVM 也事先寫好用 hinge loss；
用 IC 是寫程式時的不一致。改的時候已經看過小樣本格點 4 萬 row 的 C 0.1、0.3（fold 2024、2026-01）：

| fold | C | val rank IC | val hinge | 樣本外 rank IC |
|---|---|---|---|---|
| 2024 | 0.01 | 0.1033 | 0.9397 | 0.1231 |
| 2024 | 0.03 | 0.1009 | 0.9374 | 0.1276 |
| 2024 | 0.1 | 0.1011 | 0.9381 | 0.1331 |
| 2026-01 | 0.01 | 0.0939 | 0.9605 | 0.0710 |
| 2026-01 | 0.03 | 0.0928 | 0.9604 | 0.0775 |
| 2026-01 | 0.1 | 0.0911 | 0.9608 | 0.0821 |
| 2026-01 | 0.3 | 0.0840 | 0.9646 | 0.0799 |

兩個 fold 的 val IC 都偏好最小的 C，樣本外都偏好 C 0.1；hinge loss 在兩個 fold 都選 0.03（也不是樣本外最好的）。差異都在雜訊之內
（120 天的平均 IC 標準誤約 0.009）。這兩個 fold 的樣本外已經看過，判定時要記得。

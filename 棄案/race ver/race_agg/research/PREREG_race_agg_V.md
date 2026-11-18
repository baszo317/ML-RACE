# race_agg V：16 個候選在 V 上的檢驗（使用者要求，事先寫好的規則）

寫於 2026-11-18 18:15–18:30（台北），在任何 race_agg 候選的 V 成績算出來之前。`PREREG_race_agg.md`（SHA-256 前 16 碼 ccbc3dd4cb3859ac）不改；
這份檔案寫完也不再修改。

## 0. 為什麼打開 V

- 使用者（11/18 18:12）：「可以用 validation 測嗎，重複 val 沒差」。原規則（PREREG_race_agg.md 第 5 節）是 D 沒有採用者就不打開 V；
  這次是使用者明確要求、明知是重複使用 V。對 race 版來說這是 V 的第三次使用（race_std 用 V 選出 C.LU；race_mkt 算過一次大盤觀點）。
- 這次 V 一次把全部候選算完，寫進 `results\SEALED_V_agg.json` 後封存，程式拒絕再算（`--audit` 只比對）。D 階段的判定（16 個都沒過）不變。

## 1. 寫這份規則時已經知道的

- race_std 的 V（2025-01-02 到 2026-11-09，447 天，年度 refit）：C.LU +16.16（2025 13.44、2026-01..05 18.07、2026-06.. 20.58）；
  ENS_TOP3（ALL.LN、ALLmU.LN、C.LU 的 z 平均）+16.14；C.LC +15.22；U-native +14.17；ALLmU.LN +14.08；ALL.LN +14.04；P.LP +11.94；M.LM +9.60。
  C.LU 是 8 個 finalist 裡 V 最高的，所以它在 V 也偏高（race_std 收縮後估計約 +14.7）。
- race_agg 的全部 D 結果、事後的 seed 抽籤（C.LU 的 seed 1–3 在 D 比 3-seed 期望高約 0.45）、用 10 seed 的 LU 重做 blending 的 D 結果
  （LU10 ＋ C.LN +0.54，t 1.01）、E3 的 OOB 結果。
- 沒有算過：任何 race_agg 候選在 V 的成績；C.LN、C.LP、C.LM 在 V 的成績；B1 的 seed 4–10 在 V 的成績。

## 2. 做法

- 16 個候選（A1–A4、B1–B2、C1–C3、D1–D2、E1、E2、E4、F1、F2）的定義、程式、seed、參數和 PREREG_race_agg.md 第 3 節完全相同，
  只是 fold 換成 V 的 3 個：2025-01-01、2026-01-01、2026-06-01（= 第一部分的 V fold），訓練窗 2020-03-02 到 fold 開始的前一天。
- 要新訓練的單位：C.LN（seed 1–3）、C.LP（seed 0）、C.LM（seed 2026–2028）在 V 的單位（第一部分只訓練了 finalist 的 V 單位；程式就是
  第一部分的 `learners.fit_predict`）；A3 / A4 的 g⁻（各 fold 最後 120 天，5 個成員原本的程序和 seed）；B1、B2、C1、C2、C3、D1、D2、E1、
  E4（含選 feature）、F1。C.LU、C.LC 用第一部分存好的 V 單位。
- 基準 SD = 第一部分 C.LU 的 V 單位（seed 1–3 平均）＋ Top25 ＋ 除權息規則；必須重現 V +16.16，否則停止。
- V 的期間 2025-01-02 到 2026-11-09（447 天），分 3 段：2025、2026-01..05、2026-06..（= race_std 的 V 表）。E3 不在 V 上做。

## 3. 兩種比較

- **S（主要，用來判定）**：基準換成 10 個 seed 的 LU（C.LU 的 seed 1–3 ＋ B1 的 seed 4–10，機率平均）。理由：live 的 race 版每天用 seed = 日期
  的 3 個模型，沒有 seed 1–3 的運氣；D 上 seed 1–3 偏高約 0.45，C.LU 又是 V 的冠軍，直接和 SD 比會讓所有重新長樹的候選先天吃虧。
  - 含有 C.LU 的候選用 10 seed 的 LU 重做：A1 = zavg(LU10, C.LN)、A2 = zavg(LU10, C.LN, C.LP, C.LC, C.LM)、E2 = zavg(LU10, E1)、
    F2 = (2·LU10 − 1) × absres_60；A3、A4 的權重和 stacker 照原本（在 seed 1–3 的 g⁻ 上學），測試年的 LU 分數換成 LU10。
  - 其他候選（B2、C1、C2、C3、D1、D2、E1、E4、F1）本身不含 C.LU，直接和 LU10 比。B1 在 S 裡就是基準，不判定。共 15 個判定候選。
  - S 的 D 也照同樣方式算（D 的單位都已存在）。
- **P（只報告）**：和原本的 SD（seed 1–3）比，定義同 PREREG_race_agg.md，16 個都報告。B1 − SD 就是 seed 1–3 的運氣在 V 的大小。

## 4. 判定（用 S；寫在任何 V 成績之前）

1. **V 確認（tier 1）**：V 平均差 > 0、V 的 NW t（5 lags）≥ 2、V 的 3 段至少 2 段 > 0，而且 D 平均差 > 0。
2. **多重比較後成立（tier 2）**：tier 1，而且 D＋V 合併（1,174 天）的每日成對差，NW t 換成的單尾 p 經 Holm 校正（15 個候選）< 0.05。
3. **建議改 live：只有 tier 2 才建議**；多個時取 D＋V 合併平均差最大的一個，不做新的組合。只到 tier 1：只報告，不建議在 11/30 前改
   （要先寫 live 程式並測試；期望差和 19 天的雜訊相比太小）。都沒有：race 版不變。
4. P 和 S 的結論方向不同時，README 要寫出來；P 不改變判定。

只報告：每個候選 V 的 IC 差（t）、每日 sd、19 天平均 < 0 的比例、A3 在 V fold 的權重、AdaBoost 的 T、RF 的葉子、E4 留幾個 feature、
S 與 P 的 D＋V 合併平均和 t。

## 5. 事先的預測（不影響判定）

- A1、A2（S）：V 平均差 > 0（IC 會變高，命題 B），但 t < 2；A3 和 A2 差不多；A4 < 0。
- C1、C2、D1、E1 在 V 也明顯 < 0；C3、F1、E4、D2、B2 在 −1.5 到 +0.5 之間。
- P 下的 B1（10 seed − seed 1–3）在 V 約 −0.5 到 0（C.LU 是 V 的冠軍，seed 1–3 在 V 也可能偏高）。

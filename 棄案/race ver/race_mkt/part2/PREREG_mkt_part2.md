# race_mkt 第二部分：期貨籌碼放進選股模型（事先寫好的規則）

寫於 2026-11-17 10:13–10:16（台北），在這一部分的任何成績算出來之前。第一部分（`..\research\PREREG_race_mkt.md`）的
LOCK、SEALED_V 不改；這部分的程式在 `part2\`，結果在 `part2\results\`。

## 0. 為什麼有這一部分、寫的時候已經知道的

- 使用者問「既然盤前資訊是整件事的核心，可以透過優化盤前資訊來進步嗎？」。第一部分把期貨籌碼拿去做大盤擇時（任何數量的 1），
  結果（10:11 D、10:13 V）：8 個候選在 D 都沒過門檻（差 −0.36 到 +0.32，t ≤ 0.35），期貨籌碼對大盤預測的 OOS 相關只多約 0.006
  （MB 0.117 → MF 0.123；V 0.112 → 0.118），P2 − P1 +0.06（t 0.10）。D 上單一 feature 和 m 相關最大的是散戶（小台）淨多比
  （約 −0.11，樣本內描述）。
- 盤前資訊的另一個用法是選股：同一天所有股票共用的 day-level feature 在 LightGBM 裡可以和個股 feature 交互（例如外資期貨偏空時，
  外資持股高的股票相對變差）。C 組已經有 61 個 day-level feature（含 txoi），這裡測再加 37 個期貨籌碼 feature。
- 寫這份規則前沒有用期貨籌碼訓練過任何選股模型，也沒有算過任何選股成績。

## 1. 做法（只用 D，V 不打開）

- CF.LU：feature = C 組 169 個（`race_std\cache\features.json` 的 C）＋ 第一部分 PREREG 第 2 節的 37 個（`mktlib.futures_features`，
  第一部分已通過截斷測試），learner = `race_std\research\learners.fit_LU`（不改），seed 1、2、3，D 的 3 個年度 fold（2022、2023、2024），
  訓練窗 2020-03-02 到 fold 開始前一天（= race_std 第一部分）。
- 基準：race_std 已存的 C.LU（同樣 fold、seed），D 15.10。
- 程式檢查（在任何 CF 成績之前）：用同一套程式重訓 C.LU 2024 fold seed 1，預測必須和 `race_std\cache\units\C.LU\2024-01-01_seed1.npz`
  逐位相同；不同就先修程式。
- 決策 SD（Top50 前 25 支猜 1 ＋除權息規則，`rslib.decide_sd`），分數 = 3 seed 平均。

## 2. 判定

- CF.LU − C.LU 的每日成對差：**D 平均 > 0、Newey–West t（5 lags）≥ 2、3 年至少 2 年 > 0** 才算期貨籌碼對選股有用（= race_std 第二部分的門檻）。
- 有用的話，下一步是用每天重訓（race_live 的 WF3）在 D 重做一次再決定 live，不直接改 race 版；沒用就 race 版不變。
- 只報告：IC、每年的差、37 個 feature 在 gain importance 的占比、前 10 名。

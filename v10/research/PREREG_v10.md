# v10 事先寫好的規則（2026-10-26 06:45，使用者說「把 Alpha158 加進 v9 的 ablation，t 值大於 0 的都可以當 V10」之後、任何 Alpha158 模型的成績出來之前寫）

## 候選

- Alpha158：microsoft/qlib 的 Alpha158 預設設定（`qlib/contrib/data/loader.py` Alpha158DL、`handler.py` Alpha158），158 個
  feature。用競賽資料算（`v10\alpha158.py`）；row D 只用到 D−1 收盤以前的資料（和 B 組 feature 同一個 shift(1) 規則）。
  公式與 Qlib 原始碼逐字比對、數值與照 `qlib/data/ops.py`、`rolling.pyx` 移植的版本比對：`research\check_alpha158.py`。
- 候選單位 = 23 個資訊群組。同一個 qlib operator 的 5 個 window（5、10、20、30、60）一組；彼此只是線性轉換的 operator
  合成一組（資訊相同，分開測只是把同一個檢定做兩次）：
  KBAR(9)、PRICE(4)、ROC、MA、STD、BETA、RSQR、RESI、MAX、MIN、QTLU、QTLD、RANK、RSV、IDX（IMAX+IMIN+IMXD，15）、CORR、
  CORD、CNT（CNTP+CNTN+CNTD，15）、SUM（SUMP+SUMN+SUMD，15）、VMA、VSTD、WVMA、VSUM（VSUMP+VSUMN+VSUMD，15）；其餘每組 5 個。
- 放在哪裡（2 個 family，各自一組候選，共 46 個）：
  - **L**：選股 LightGBM `lgb_res`（v9 用 B 的 67 個）→ B + 群組。seed 1–5（和 v9 相同）。
  - **M**：選股 MLP 裡用 100 支訓練的三個成員 `mlp_res_xr9`、`mlp_res_BN9`、`mlp_res_xrN9` → 各自 v9 的 feature + 群組
    （三個一起加）。篩選用 seed 1–5（v9 是 1–10，運算量的取捨）。
  - 不動：`mlp_res_B9`（用擴充股票池訓練，那 210 支沒有 OHLCV 檔）、大盤觀點（v8 五個大盤改法都沒過；pbar 只影響
    timing；運算量）、Transformer、所有規則與係數。
- 另外兩個只報告、不參加挑選的設定：ALL（158 個全加，L、M 各一）。
- 對照：同一台電腦、同一套程式重訓的 baseline（L：B，seed 1–5，應該和 v9 的 `lgb_res` 完全相同；M：v9 的 feature，seed 1–5）。

## 怎麼比

- 一個候選 = 只把該 family 的成員換成加了群組的版本，其餘成員用 v9 現有的 walk-forward 預測（`v9\cache\wf9_preds.csv.gz`、
  `v8\cache\wf_members.csv.gz`、`v8\cache\tab_preds.csv.gz`），用 v9 的規則算 Top50 bp/日（`v9\research\v9_eval.py` 的
  views + decide：w_T 0.5、除權息當天／前一天／後一天／後兩天猜 0、beta 對沖 λ 0.75）。
- 和同 family 的 baseline 算每日成對差，t = 平均 / (sd / √n)（同 v9 的 plain t）。
- 期間同 v9：val 2022–2024、hold 2025-01 到 2026-05、post 2026-06 以後（模型訓練到 2026-05-29 後凍結）。
- **入選規則（使用者的規則）：val 成對差 t > 0**。hold、post 不參與挑選。

## V10

- L 的入選群組全部加進 `lgb_res`；M 的入選群組全部加進那三個 MLP。某個 family 沒有群組入選就不動；兩邊都沒有就沒有 V10。
- 完整重訓：LightGBM seed 1–5、MLP seed 1–10（同 v9）；同一台電腦另外補訓 v9 replica（M baseline 的 seed 6–10），
  V10 和 v9 replica 成對比較。
- **檢驗（事先寫好）**：V10 對 v9 replica 的 **hold 差 > 0 且 post 差 > 0** →「Alpha158 加進 v9 有推廣出去」；任一段 ≤ 0 →
  沒有推廣，live 維持 v9。
- 只報告：val（挑選用，必然偏高）、全期、每日 sd、P(19 天 > 0)、static 期間（6/1–9/30）、10 月、只改 L 或只改 M 的
  ablation、ALL 版本、V10 對 v9 原檔（雲端訓練的 cache）的差。

## 誠實說明

- 46 個候選。若全部沒有用，val t > 0 的機率各約一半，預期約 23 個入選；t > 0 是很寬的門檻，入選名單一定混了很多雜訊。
  hold、post 的檢驗是唯一的保護，而且一次檢定也只有約一半的機率擋下沒用的組合。
- 每個群組只加在一個 family；同一群組在 L、M 的結果不一樣不代表矛盾（不同模型、雜訊不同）。
- hold、post 在 v1–v9 都看過，不是完全乾淨；乾淨的只有 10/27 以後的 live。

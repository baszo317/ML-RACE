import pandas as pd, os
from openpyxl.styles import Font, PatternFill, Alignment
HERE=os.path.dirname(os.path.abspath(__file__))
dic=[
('date','原始','交易日','-','排序/對齊用'),
('etl:adj_open/high/low/close','原始','還原開高低收價（除權息調整，數值為向上調整後的尺度）','close: 收盤後','close/open 決定 y；直接使用水準值無跨股可比性，應轉成報酬率或比值'),
('price:成交筆數','原始','成交筆數','收盤後 → lag 1','與 20 日均量比 (trades_ratio)；2025 後 IC 轉負'),
('price:成交金額','原始','成交金額 (NTD)','收盤後 → lag 1','amt_ratio；與成交股數高度共線'),
('price:成交股數','原始','成交股數','收盤後 → lag 1','vol_ratio；也作為法人買賣超的標準化分母 (20 日均量)'),
('全體外資及陸資持股比率','原始','外資持股比例 (%)','收盤後公布 → lag 1','水準值弱；日變化 foreign_hold_chg1 為弱反向訊號 (IC 約 -0.02)，2025 後消失'),
('外陸資買賣超股數(不含外資自營商)','原始','外資淨買超股數','收盤後公布 → lag 1','標準化後 foreign_net：弱反向 (外資大買隔日盤中偏弱)，2025 後轉正，不穩定'),
('投信買賣超股數','原始','投信淨買超','收盤後公布 → lag 1','trust_net：85% 股票 IC 為正，但 2025 後衰減至約 0.007'),
('自營商買賣超股數(自行買賣)','原始','自營商自營部位淨買超','收盤後公布 → lag 1','dealer_self_net：76% 正 IC，2025 後反而增強 (0.026)'),
('自營商買賣超股數(避險)','原始','自營商避險部位淨買超','收盤後公布 → lag 1','幾乎無訊號 (IC 約 0)'),
('RSI','原始','RSI(14), Wilder smoothing，用當日收盤','收盤後 → lag 1','弱反向 (超買隔日盤中偏弱)，IC 約 -0.011'),
('SMA','原始','SMA(30) of adj_close','收盤後 → lag 1','價格水準，需轉為 close/SMA30-1；弱反向'),
('MACD','原始','MACD histogram (12,26,9)，價格尺度','收盤後 → lag 1','需除以價格；幾乎無訊號'),
('ATR','原始','ATR(14), Wilder','收盤後 → lag 1','ATR/close：波動度，弱反向；可作為風險/部位大小特徵'),
('BBANDS_upper/mid/lower','原始','Bollinger Bands (5 日, 2σ)，mid = SMA(5)','收盤後 → lag 1','轉成 %b 與 width；%b 弱反向 (短期均值回歸)'),
('y','標籤','adj_close > adj_open 則為 1 (平盤算 0)','當日收盤後','與同 row 的 close>open 100% 一致'),
('^VIX','全球','VIX','row d = 前一美股交易日收盤，台股 d 日開盤前已知','vix_chg (前前晚) 為負向訊號；當晚值訊號弱'),
('^IXIC','全球','Nasdaq 收盤','同上','當晚報酬：盤中反向 (跳空後回吐)；前前晚報酬：正向，半導體最強'),
('^GSPC','全球','S&P 500 收盤','同上','同 ^IXIC，略弱'),
('^TWII','全球','台股加權指數','row d = 台股 d 日收盤 → 必須 lag 1','lag 1 後弱正向，2025 後轉負'),
('^KS11','全球','KOSPI','row d = 韓股 d 日收盤 → 必須 lag 1','幾乎無訊號，2025 後轉負'),
('TX_OI','全球','外資台指期淨未平倉口數','對齊有歧義，保守用 lag 1','txoi_chg (lag 1)：98% 股票 IC 為正，最一致的單一訊號'),
]
D=pd.DataFrame(dic,columns=['欄位','來源','意義/參數','可用時點','分析結論'])
S=pd.read_csv(os.path.join(HERE,'feature_summary.csv')); P=pd.read_csv(os.path.join(HERE,'per_stock_summary.csv'),dtype={'sid':str})
IC=pd.read_csv(os.path.join(HERE,'per_stock_feature_IC.csv'),dtype={'Unnamed: 0':str}).rename(columns={'Unnamed: 0':'sid'})
note=pd.DataFrame({'說明':[
 'IC = 每支股票的 Spearman rank correlation(特徵, 當日盤中報酬 (close-open)/open)，所有特徵只用開盤前可得資訊',
 'IC_train: 2020-03~2024；IC_test: 2025-01~2026-08',
 'day_level_t: 把 Top50 每日平均後做時間序列檢定，避免同一天 50 支股票重複計數造成 t 值灌水',
 'OOS_AR_bp: 以 train 期 IC 方向、每股 train 期 70/30 分位數為門檻，只在最強 30% 訊號日預測 1，其餘預測 0，於 2025~2026 Top50 計算競賽指標 (2p-1)(close-open)/open，單位 bp',
 'OOS_excess_vs_all0_bp: 減去「全部預測 0」的 5.49 bp 基準',
 '單一股票 IC 標準誤約 0.025；|IC|>0.05 的比例 16.4%，高於純雜訊下的約 4.6%，但個股 top3 中仍會混入偶然結果']})
out=os.path.join(HERE,'feature_analysis.xlsx')
with pd.ExcelWriter(out,engine='openpyxl') as w:
    D.to_excel(w,'欄位說明',index=False); S.round(4).to_excel(w,'特徵總表',index=False)
    P.round(4).to_excel(w,'個股摘要',index=False); IC.round(4).to_excel(w,'個股IC矩陣',index=False); note.to_excel(w,'方法說明',index=False)
    for ws in w.book.worksheets:
        ws.freeze_panes='B2'
        for c in ws[1]: c.font=Font(bold=True,color='FFFFFF'); c.fill=PatternFill('solid',fgColor='1F3A5F'); c.alignment=Alignment(wrap_text=True,vertical='center')
        for col in ws.columns:
            L=max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[col[0].column_letter].width=min(max(10,L*1.1),70)
    for r in w.book['欄位說明'].iter_rows(min_row=2):
        for c in r: c.alignment=Alignment(wrap_text=True,vertical='top')
print(out)

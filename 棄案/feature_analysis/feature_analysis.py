"""Feature analysis for ML RACE (predict close > open).
All features at row t use only information available before the open of day t.
Run: python feature_analysis.py  (outputs CSVs next to this script)"""
import pandas as pd, numpy as np, os, json
HERE=os.path.dirname(os.path.abspath(__file__)); base=os.path.join(HERE,'..','data')
info=json.load(open(f'{base}/info.json')); TOP50=set(info['Top50@Y'])
g=pd.read_csv(f'{base}/global/global_data.csv',parse_dates=['date']).set_index('date')
# --- global features ---
# US columns (^VIX ^IXIC ^GSPC): row d already holds previous US session close -> usable at d (no lag)
# ^TWII ^KS11: same-day close -> lag 1 ; TX_OI: lag 1 (conservative)
US=pd.DataFrame(index=g.index)
US['us_vix']=g['^VIX']; US['us_vix_chg']=np.log(g['^VIX']).diff()
for k,n in [('^IXIC','ixic'),('^GSPC','gspc')]:
    US[f'us_{n}_ret1']=np.log(g[k]).diff(); US[f'us_{n}_ret5']=np.log(g[k]).diff(5)
LAG=pd.DataFrame(index=g.index)
for k,n in [('^TWII','twii'),('^KS11','ks11')]:
    LAG[f'{n}_ret1']=np.log(g[k]).diff(); LAG[f'{n}_ret5']=np.log(g[k]).diff(5)
LAG['txoi']=g['TX_OI']/1e4; LAG['txoi_chg']=g['TX_OI'].diff()/1e4; LAG['txoi_chg5']=g['TX_OI'].diff(5)/1e4
US_l1=US.shift(1).add_suffix('_lag1')[['us_vix_chg_lag1','us_ixic_ret1_lag1','us_gspc_ret1_lag1']]
def stock_feats(df):
    o,h,l,c=[df[f'etl:adj_{k}'] for k in ['open','high','low','close']]
    v=df['price:成交股數'];amt=df['price:成交金額'];n=df['price:成交筆數'];cols=df.columns
    F=pd.DataFrame(index=df.index)
    F['ret1']=np.log(c).diff(); F['ret5']=np.log(c).diff(5); F['ret20']=np.log(c).diff(20)
    F['intraday_ret']=np.log(c/o); F['gap']=np.log(o/c.shift()); F['y_mean5']=df['y'].rolling(5).mean()
    F['range']=(h-l)/c; F['clv']=((c-l)/(h-l)).where(h>l,0.5)
    F['upper_shadow']=(h-np.maximum(o,c))/c; F['lower_shadow']=(np.minimum(o,c)-l)/c
    F['vol_ratio']=np.log(v/v.rolling(20).mean()); F['amt_ratio']=np.log(amt/amt.rolling(20).mean())
    F['trades_ratio']=np.log(n/n.rolling(20).mean()); F['avg_trade_size']=np.log((v/n)/(v/n).rolling(20).mean())
    fh=df[cols[8]]; F['foreign_hold']=fh; F['foreign_hold_chg1']=fh.diff(); F['foreign_hold_chg5']=fh.diff(5)
    adv=v.rolling(20).mean()
    for i,nm in [(9,'foreign_net'),(10,'trust_net'),(11,'dealer_self_net'),(12,'dealer_hedge_net')]:
        F[nm]=df[cols[i]]/adv; F[nm+'_5d']=df[cols[i]].rolling(5).sum()/adv/5
    F['RSI']=df.RSI; F['close_SMA30']=c/df.SMA-1; F['MACD_hist_norm']=df.MACD/c; F['ATR_pct']=df.ATR/c
    F['BB_pctb']=((c-df.BBANDS_lower)/(df.BBANDS_upper-df.BBANDS_lower)).clip(-2,3)
    F['BB_width']=(df.BBANDS_upper-df.BBANDS_lower)/df.BBANDS_mid
    F=F.join(LAG.reindex(df.date).set_index(df.index))   # TW-close info, lag together
    F=F.shift(1)                                           # everything above: info up to close t-1
    d=df.date
    F=F.join(US.reindex(d).set_index(df.index)).join(US_l1.reindex(d).set_index(df.index))  # US: no extra lag
    F['target_r']=(c-o)/o; F['y']=df['y']; F['date']=d
    return F
frames=[]
for sid in info['Top100@Y']:
    df=pd.read_csv(f'{base}/market/{sid}.csv',parse_dates=['date'])
    F=stock_feats(df); F['sid']=sid; frames.append(F)
A=pd.concat(frames,ignore_index=True); A=A[A.date>='2020-03-01'].reset_index(drop=True)
A['top50']=A.sid.isin(TOP50)
FC=[c for c in A.columns if c not in ('target_r','y','date','sid','top50')]
A.to_pickle(os.path.join(HERE,'features.pkl'))
# ---- per-stock rank IC (Spearman) between feature and intraday return ----
R=A.copy(); R[FC+['target_r']]=R.groupby('sid')[FC+['target_r']].rank(pct=True)
def ic_table(R):
    out={}
    for sid,d in R.groupby('sid'):
        out[sid]=d[FC].corrwith(d['target_r'])
    return pd.DataFrame(out).T
IC=ic_table(R)
IC_tr=ic_table(R[R.date<'2025-01-01']); IC_te=ic_table(R[R.date>='2025-01-01'])
IC.round(4).to_csv(os.path.join(HERE,'per_stock_feature_IC.csv'),encoding='utf-8-sig')
# ---- day-level significance (avoids overcounting correlated stocks) ----
T=A[A.top50]
def day_t(f):
    d=T.groupby('date')[[f,'target_r']].mean().dropna()
    # for stock features use cross-stock-averaged signal; good enough for significance screening
    ic=d[f].rank().corr(d.target_r.rank()); return ic*np.sqrt(len(d)-2)/np.sqrt(1-ic**2)
# ---- OOS rule: train 2020-03..2024, test 2025-01..2026-08, Top50 ----
tr=A[A.date<'2025-01-01']; te=A[(A.date>='2025-01-01')&A.top50]
base_ret=-te.target_r.mean()
rows=[]
for f in FC:
    s=np.sign(IC_tr[f].mean())
    q=tr.groupby('sid')[f].quantile(0.7 if s>0 else 0.3)   # predict 1 only on strongest 30% signal days
    sig=te[f]; thr=te.sid.map(q)
    p=((sig>thr) if s>0 else (sig<thr)).astype(int)
    oos=((2*p-1)*te.target_r)[sig.notna()].mean()
    rows.append(dict(feature=f,IC_mean=IC[f].mean(),IC_pos_frac=(IC[f]>0).mean(),IC_train=IC_tr[f].mean(),
        IC_test=IC_te[f].mean(),day_level_t=day_t(f),OOS_AR_bp=oos*1e4,OOS_excess_vs_all0_bp=(oos-base_ret)*1e4))
S=pd.DataFrame(rows).sort_values('IC_mean',key=abs,ascending=False)
S.round(4).to_csv(os.path.join(HERE,'feature_summary.csv'),index=False,encoding='utf-8-sig')
pd.set_option('display.width',250); print(S.round(4).to_string(index=False)); print('always-0 test AR bp',base_ret*1e4)

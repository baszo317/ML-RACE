"""Test extra candidate features derivable from the provided data. Uses features.pkl (already leak-free)."""
import pandas as pd, numpy as np, os
HERE=os.path.dirname(os.path.abspath(__file__))
A=pd.read_pickle(os.path.join(HERE,'features.pkl')).sort_values(['sid','date']).reset_index(drop=True)
P=pd.read_csv(os.path.join(HERE,'per_stock_summary.csv'),dtype={'sid':str}); A['group']=A.sid.map(P.set_index('sid').group)
g=A.groupby('sid')
r_prev=g.target_r.shift(1)                                    # yesterday's intraday return (known)
# 1) time-varying stock bias
for n in [20,60,250]: A[f'intra_bias_{n}']=r_prev.groupby(A.sid).transform(lambda s:s.rolling(n,min_periods=n//2).mean())
A['gap_bias_20']=A.groupby('sid').gap.transform(lambda s:s.rolling(20,min_periods=10).mean())
# 2) market / sector state (yesterday)
A['r_prev']=r_prev
mk=A.groupby('date').r_prev.mean().rename('mkt_intra_prev'); A=A.join(mk,on='date')
br=A.assign(yp=g.y.shift(1)).groupby('date').yp.mean().rename('breadth_prev'); A=A.join(br,on='date')
sec=A.groupby(['date','group']).r_prev.mean().rename('sector_intra_prev'); A=A.join(sec,on=['date','group'])
A['rel_intra_prev']=A.r_prev-A.mkt_intra_prev
A['mkt_intra_5']=A.groupby('sid').mkt_intra_prev.transform(lambda s:s.rolling(5).mean())
# 3) stock-specific US sensitivity x tonight's US move
def beta(d):
    cov=d.gap.rolling(120,min_periods=60).cov(d.us_ixic_ret1.shift(1)); var=d.us_ixic_ret1.shift(1).rolling(120,min_periods=60).var()
    return cov/var
A['beta_us']=A.groupby('sid',group_keys=False).apply(beta)
A['beta_x_us']=A.beta_us*A.us_ixic_ret1
A['beta_x_us_lag1']=A.beta_us*A.us_ixic_ret1_lag1
# 4) price-level / regime
A['streak_y']=g.y.transform(lambda s:s.shift(1).groupby((s.shift(1)!=s.shift(2)).cumsum()).cumcount()+1)*np.where(g.y.shift(1)==1,1,-1)
A['limit_up_prev']=(A.ret1>0.093).astype(int); A['limit_dn_prev']=(A.ret1<-0.093).astype(int)
A['vol_ratio_5_60']=g.ret1.transform(lambda s:s.rolling(5).std()/s.rolling(60).std())
A['foreign_streak']=g.foreign_net.transform(lambda s:np.sign(s).groupby((np.sign(s)!=np.sign(s).shift()).cumsum()).cumcount()+1)*np.sign(A.foreign_net)
# 5) calendar
dates=pd.Series(sorted(A.date.unique())); nxt=dates.shift(-1); prv=dates.shift(1)
cal=pd.DataFrame({'date':dates,'dow':dates.dt.dayofweek,
   'pre_holiday':((nxt-dates).dt.days>3).astype(int),'post_holiday':((dates-prv).dt.days>3).astype(int)})
cal['ym']=cal.date.dt.to_period('M'); cal['tdom']=cal.groupby('ym').cumcount()+1; cal['tdom_rev']=cal.groupby('ym').cumcount(ascending=False)+1
cal['turn_of_month']=((cal.tdom<=3)|(cal.tdom_rev<=1)).astype(int)
wed=cal.date.dt.dayofweek==2; cal['settle_monthly']=(wed&(cal.date.dt.day.between(15,21))).astype(int)   # TX/TXO monthly settlement (3rd Wed)
cal['settle_weekly']=wed.astype(int)
A=A.merge(cal.drop(columns='ym'),on='date')
new=['intra_bias_20','intra_bias_60','intra_bias_250','gap_bias_20','mkt_intra_prev','breadth_prev','sector_intra_prev','rel_intra_prev','mkt_intra_5',
     'beta_us','beta_x_us','beta_x_us_lag1','streak_y','vol_ratio_5_60','foreign_streak']
R=A.copy(); R[new+['target_r']]=R.groupby('sid')[new+['target_r']].rank(pct=True)
def ics(R): return pd.DataFrame({s:d[new].corrwith(d.target_r) for s,d in R.groupby('sid')}).T
IC=ics(R); IC_tr=ics(R[R.date<'2025-01-01']); IC_te=ics(R[R.date>='2025-01-01'])
tr=A[A.date<'2025-01-01']; te=A[(A.date>='2025-01-01')&A.top50]; base=-te.target_r.mean()
rows=[]
for f in new:
    s=np.sign(IC_tr[f].mean()); q=tr.groupby('sid')[f].quantile(0.7 if s>0 else 0.3)
    p=((te[f]>te.sid.map(q)) if s>0 else (te[f]<te.sid.map(q))).astype(int)
    oos=((2*p-1)*te.target_r)[te[f].notna()].mean()
    rows.append(dict(feature=f,IC_mean=IC[f].mean(),IC_pos_frac=(IC[f]>0).mean(),IC_train=IC_tr[f].mean(),IC_test=IC_te[f].mean(),OOS_excess_bp=(oos-base)*1e4))
# cross-sectional (pooled) IC for bias features - they matter across stocks, not within
for f in ['intra_bias_20','intra_bias_60','intra_bias_250']:
    d=A[A.date>='2025-01-01']; cs=d.groupby('date').apply(lambda x:x[f].rank().corr(x.target_r.rank())).mean()
    print(f'cross-sectional IC 2025+ {f}: {cs:.4f}')
S=pd.DataFrame(rows).sort_values('IC_mean',key=abs,ascending=False); pd.set_option('display.width',200)
S.round(4).to_csv(os.path.join(HERE,'extra_feature_summary.csv'),index=False,encoding='utf-8-sig')
print(S.round(4).to_string(index=False))
# bias rule OOS: sign of intra_bias_250
p=(te.intra_bias_250>0).astype(int); print('rule sign(intra_bias_250) OOS bp',(((2*p-1)*te.target_r).mean())*1e4,' all0',base*1e4)
p=(te.intra_bias_60>0).astype(int); print('rule sign(intra_bias_60) OOS bp',(((2*p-1)*te.target_r).mean())*1e4)
# calendar: day-level mean of Top50 intraday by category
T=A[A.top50].groupby('date').agg(r=('target_r','mean'),**{c:(c,'first') for c in ['dow','pre_holiday','post_holiday','turn_of_month','settle_monthly','settle_weekly','limit_up_prev']})
for c in ['dow','pre_holiday','post_holiday','turn_of_month','settle_monthly','settle_weekly']:
    s=T.groupby(c).r.agg(['mean','std','count']); s['t']=s['mean']/s['std']*np.sqrt(s['count']); s['mean_bp']=s['mean']*1e4
    print(c); print(s[['mean_bp','count','t']].round(2))
L=A[A.top50].groupby('limit_up_prev').target_r.agg(['mean','count']); L['mean']*=1e4; print('limit_up_prev',L.round(2))
L=A[A.top50].groupby('limit_dn_prev').target_r.agg(['mean','count']); L['mean']*=1e4; print('limit_dn_prev',L.round(2))
A[['date','sid']+new+['dow','pre_holiday','post_holiday','turn_of_month','settle_monthly','settle_weekly','limit_up_prev','limit_dn_prev']].to_pickle(os.path.join(HERE,'extra_features.pkl'))

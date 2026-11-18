"""
panel_ext.py -- the v4 panel (competition Top100, unchanged) plus rows of extra stocks for training only.
Extra stock rows: features from transformer/comp_features._stock on files rebuilt by ext_data.build; market aggregates
(mkt_intra_prev, breadth_prev, mkt_intra_5, sector_intra_prev) and the residual target use the competition 100 only,
exactly as in the core panel; global features merged like comp_features. Point-in-time universe: in year Y an extra
stock is in the training set if its market cap at the last trading day of Y-1 ranks <= TOPN among the public stocks.
"""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (HERE, os.path.join(ROOT, 'transformer'), os.path.join(ROOT, 'v5')):
    if p not in sys.path:
        sys.path.insert(0, p)
import ext_data as X
import comp_features as CF
import ens_lib as E

FIN = {'金融保險業'}; SEMI = {'半導體業'}
ELEC = {'電子零組件業', '電腦及週邊設備業', '光電業', '通信網路業', '電子通路業', '資訊服務業', '其他電子業'}

def group_of(cat):
    return 0 if cat in FIN else 1 if cat in SEMI else 2 if cat in ELEC else 3

def universe(top_n=300, years=range(2020, 2027)):
    """{year: set of extra stock ids} -- market-cap rank at the last trading day of year-1 among the public stocks"""
    ids = X.available_ids(); caps = {}
    for sid in ids:
        p = pd.read_parquet(os.path.join(X.RAW, 'TaiwanStockPrice', f'{sid}.parquet'))
        s = pd.read_parquet(os.path.join(X.RAW, 'TaiwanStockShareholding', f'{sid}.parquet'))
        if not len(p) or not len(s) or 'NumberOfSharesIssued' not in s or 'close' not in p:
            continue
        p, s = p[['date', 'close']], s[['date', 'NumberOfSharesIssued']]
        x = p.merge(s, on='date'); x['date'] = pd.to_datetime(x.date); x['cap'] = x.close * x.NumberOfSharesIssued
        caps[sid] = x.set_index('date').cap
    C = pd.DataFrame(caps)
    out = {}
    for y in years:
        row = C[C.index < pd.Timestamp(f'{y}-01-01')].ffill().iloc[-1]
        out[y] = set(row.rank(ascending=False).loc[lambda r: r <= top_n].index)
    return out

def build(core=None, top_n=300, start='2019-06-01'):
    info = json.load(open(os.path.join(E.DATA, 'info.json'))); core_ids = set(info['Top100@Y'])
    P = E.build_panel(E.DATA) if core is None else core
    P['date'] = P.date.astype('datetime64[ns]'); P['core'] = True
    U = universe(top_n); extra = sorted(set().union(*U.values()) - core_ids)
    ind = X.industry()
    agg = P.groupby('date')[['mkt_intra_prev', 'breadth_prev', 'mkt_intra_5']].first()
    sect = P.groupby(['group', 'date']).sector_intra_prev.first()
    mkt_r = P.groupby('date').target_r.mean().rename('mkt_r')
    g = CF.read_global(os.path.join(E.DATA, 'global', 'global_data.csv'))
    Ug = pd.DataFrame({'date': g.date}); Ug['us_vix'] = g['^VIX']; Ug['us_vix_chg'] = np.log(g['^VIX']).diff()
    for k, nm in [('^IXIC', 'ixic'), ('^GSPC', 'gspc')]:
        Ug[f'us_{nm}_ret1'] = np.log(g[k]).diff(); Ug[f'us_{nm}_ret5'] = np.log(g[k]).diff(5)
    for nm in ['us_vix_chg', 'us_ixic_ret1', 'us_gspc_ret1']:
        Ug[nm + '_lag1'] = Ug[nm].shift(1)
    W = pd.DataFrame({'date': g.date})
    for k, nm in [('^TWII', 'twii'), ('^KS11', 'ks11')]:
        W[f'{nm}_ret1'] = np.log(g[k]).diff(); W[f'{nm}_ret5'] = np.log(g[k]).diff(5)
    W['txoi'] = g['TX_OI'] / 1e4; W['txoi_chg'] = g['TX_OI'].diff() / 1e4; W['txoi_chg5'] = g['TX_OI'].diff(5) / 1e4
    B = E.feature_sets()['B']
    keep = ['date', 'stock_id', 'target_r', 'y', 'resid_r', 'y_resid', 'top50', 'core'] + B
    parts = []
    for sid in extra:
        df = X.build(sid).drop(columns='shares_issued')
        F = CF._stock(df); F['stock_id'] = sid
        F = F[F.date >= pd.Timestamp(start)].copy()
        cat = ind.industry_category.get(sid, '')
        F['group'] = float(group_of(cat)); F['is_fin'] = float(F.group.iloc[0] == 0) if len(F) else 0.0
        F = F.merge(agg, left_on='date', right_index=True, how='left')
        F = F.merge(sect.rename('sector_intra_prev').reset_index(), on=['group', 'date'], how='left')
        F = F.sort_values('date')
        F['rel_intra_prev'] = F.target_r.shift(1) - F.mkt_intra_prev
        F = pd.merge_asof(F, Ug, on='date', direction='backward')
        F = pd.merge_asof(F, W, on='date', direction='backward', allow_exact_matches=False)
        cv = F.gap_today_raw.rolling(120, min_periods=60).cov(F.us_ixic_ret1); vr = F.us_ixic_ret1.rolling(120, min_periods=60).var()
        F['beta_us'] = (cv / vr).shift(1); F['beta_x_us'] = F.beta_us * F.us_ixic_ret1; F['beta_x_us_lag1'] = F.beta_us * F.us_ixic_ret1_lag1
        F = F.merge(mkt_r, left_on='date', right_index=True, how='inner')          # competition trading days only
        F['resid_r'] = F.target_r - F.mkt_r; F['y_resid'] = (F.resid_r > 0).astype(float).where(F.resid_r.notna())
        F['top50'] = False; F['core'] = False
        inu = np.array([sid in U.get(int(y), set()) for y in F.date.dt.year], dtype=bool)
        F = F[inu]                                                                 # point-in-time universe
        if F.empty:
            continue
        num = [c for c in keep if c not in ('date', 'stock_id', 'top50', 'core')]
        F[num] = F[num].astype('float32')
        parts.append(F[keep])
    Xr = pd.concat(parts, ignore_index=True)
    Pc = P[keep].copy(); num = [c for c in keep if c not in ('date', 'stock_id', 'top50', 'core')]; Pc[num] = Pc[num].astype('float32')
    return pd.concat([Pc, Xr], ignore_index=True).sort_values(['date', 'stock_id']).reset_index(drop=True), U, extra

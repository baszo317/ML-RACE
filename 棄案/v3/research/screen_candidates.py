r"""research\screen_candidates.py -- quick univariate screen of candidate features (not in B / XR) for close > open.
xsec candidates: daily Spearman IC with residual intraday return e = r - mean_100(r), over Top50.
timing candidates: time-series corr with m50 = Top50 mean intraday return.
Everything known before 09:00 of day t (shifted by one day / PIT available_from).

usage (from the v3 folder; prints the tables, about 2 minutes):
    python research\screen_candidates.py > research\candidate_screen.txt
needs pandas + numpy (+ pyarrow for the external point-in-time tables, which now live in ..\棄案\data\pit\;
without them the SOX / USD-TWD / market-flow / revenue / valuation parts are skipped).
The 5 features adopted in v3 are implemented (identically) in features_v3.py."""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'transformer'))
from comp_features import build_comp_features, read_market, read_global, SEMI

D = os.path.join(ROOT, 'data')
PITDIR = next((p for p in (os.path.join(D, 'pit'), os.path.join(ROOT, '棄案', 'data', 'pit')) if os.path.isdir(p)), None)
info = json.load(open(os.path.join(D, 'info.json')))
sids = info['Top100@Y']; top50 = set(info['Top50@Y'])

IND = {'foundry_mem': '2330 2303 5347 2408 2344 6488', 'ic_design': '2454 3034 2379 3661 3443 3529 5274 6415 5269 8299',
       'osat_equip': '3711 2449 6223 2360 2404', 'server_odm': '2317 2382 6669 3231 2356 2376 2357 2324 4938 2377 2301 2395',
       'server_comp': '2308 3017 3653 2059 3533 3665 6409 2345', 'pcb_ccl': '2383 2368 3037 8046 3044 4958 6274',
       'components': '2327 3008 2474 8069 3702 3036', 'telecom': '2412 3045 4904',
       'fin': '2881 2882 2891 2886 2884 2885 2892 2880 5880 2890 2883 2887 5876 2801 2812 2834 5871',
       'shipping_air': '2603 2609 2615 2618 2610', 'materials': '1301 1303 1326 6505 1101 1102 2002',
       'consumer': '1216 2912 2207 2105 9910 1476 8464 1402 3293 6446', 'industrial': '1519 1504 1590 2633'}
IND = {s: k for k, v in IND.items() for s in v.split()}
assert set(IND) == set(sids), set(sids) ^ set(IND)

# ---------------- base panel: the 67 B features exactly as the pipeline builds them
P = build_comp_features(D)
META = ['date', 'stock_id', 'target_r', 'y', 'top50', 'has_label', 'stale_us']
BFEAT = [c for c in P.columns if c not in META]
print('B features:', len(BFEAT))

# ---------------- per-stock candidates
rows = []
for s in sids:
    df = read_market(os.path.join(D, 'market', f'{s}.csv'))
    cols = df.columns
    o, h, l, c = [df[f'etl:adj_{k}'] for k in ('open', 'high', 'low', 'close')]
    v, amt = df['price:成交股數'], df['price:成交金額']
    r1 = np.log(c).diff(); gap = np.log(o / c.shift()); intra = (c - o) / o
    adv = v.rolling(20).mean(); vwap = amt / v
    F = pd.DataFrame({'date': df.date, 'stock_id': s})
    F['gap_bias_60'] = gap.rolling(60, min_periods=30).mean().shift(1)
    F['gap_bias_250'] = gap.rolling(250, min_periods=120).mean().shift(1)
    F['max_20'] = r1.rolling(20).max().shift(1)
    F['min_20'] = r1.rolling(20).min().shift(1)
    F['skew_60'] = r1.rolling(60).skew().shift(1)
    F['hi52'] = (c / c.rolling(250, min_periods=120).max() - 1).shift(1)
    F['amihud_20'] = np.log((r1.abs() / amt).rolling(20).mean() + 1e-18).shift(1)
    F['size_amt_20'] = np.log(amt.rolling(20).mean()).shift(1)
    F['foreign_net_20'] = (df[cols[9]].rolling(20).sum() / adv / 20).shift(1)
    F['trust_net_20'] = (df[cols[10]].rolling(20).sum() / adv / 20).shift(1)
    F['dealer_hedge_20'] = (df[cols[12]].rolling(20).sum() / adv / 20).shift(1)
    F['foreign_hold_chg20'] = df[cols[8]].diff(20).shift(1)
    F['intra_vol_20'] = intra.rolling(20).std().shift(1)
    F['gap_vol_20'] = gap.rolling(20).std().shift(1)
    F['intra_ac_60'] = intra.rolling(60).corr(intra.shift(1)).shift(1)       # own intraday autocorrelation
    F['ret60'] = r1.rolling(60).sum().shift(1)
    F['ret250_skip20'] = (r1.rolling(250).sum() - r1.rolling(20).sum()).shift(1)
    F['r'] = intra; F['r1'] = r1; F['gap0'] = gap; F['amt'] = amt
    F['fn_val'] = df[cols[9]] * vwap; F['tn_val'] = df[cols[10]] * vwap
    F['dsn_val'] = df[cols[11]] * vwap; F['dhn_val'] = df[cols[12]] * vwap
    rows.append(F)
X = pd.concat(rows, ignore_index=True)
X['top50'] = X.stock_id.isin(top50)
X['m'] = X.groupby('date').r.transform('mean')
X['e'] = X.r - X.m
X['r1_res'] = X.r1 - X.groupby('date').r1.transform('mean')
X = X.sort_values(['stock_id', 'date']).reset_index(drop=True)

def per_stock(fn):
    return pd.concat([fn(g) for _, g in X.groupby('stock_id', sort=False)]).reindex(X.index)

X['ivol_60'] = per_stock(lambda g: g.r1_res.rolling(60, min_periods=30).std().shift(1))
X['beta_intra_120'] = per_stock(lambda g: (g.r.rolling(120, min_periods=60).cov(g.m) / g.m.rolling(120, min_periods=60).var()).shift(1))
X['resid_vol_60'] = per_stock(lambda g: g.e.rolling(60, min_periods=30).std().shift(1))
X['dow'] = X.date.dt.dayofweek
X['dow_bias_e'] = X.groupby(['stock_id', 'dow']).e.transform(lambda s: s.shift(1).rolling(52, min_periods=20).mean())
X['dom_bias_e'] = np.nan  # placeholder (not used)
# fine industry
X['ind'] = X.stock_id.map(IND)
im = X.groupby(['ind', 'date']).r.mean().rename('ind_r').reset_index().sort_values(['ind', 'date'])
im['ind_intra_prev'] = im.groupby('ind').ind_r.shift(1)
im['ind_intra_5'] = im.groupby('ind').ind_r.transform(lambda s: s.rolling(5).mean().shift(1))
X = X.merge(im[['ind', 'date', 'ind_intra_prev', 'ind_intra_5']], on=['ind', 'date'], how='left')
X = X.sort_values(['stock_id', 'date']).reset_index(drop=True)
X['r_prev'] = X.groupby('stock_id').r.shift(1)
X['ind_rel_intra_prev'] = X.r_prev - X.ind_intra_prev
X['ret5_'] = X.groupby('stock_id').r1.transform(lambda s: s.rolling(5).sum().shift(1))
X['ret20_'] = X.groupby('stock_id').r1.transform(lambda s: s.rolling(20).sum().shift(1))
for k in ('5', '20'):
    X[f'ind_ret{k}'] = X.groupby(['ind', 'date'])[f'ret{k}_'].transform('mean')
    X[f'ind_rel_ret{k}'] = X[f'ret{k}_'] - X[f'ind_ret{k}']
X['ind_rel_intra_bias_20'] = X.groupby('stock_id').r.transform(lambda s: s.rolling(20, min_periods=10).mean().shift(1))
X['ind_rel_intra_bias_20'] -= X.groupby(['ind', 'date']).ind_rel_intra_bias_20.transform('mean')
X['is_semi'] = X.stock_id.isin(SEMI).astype(float)
X['is_fin'] = (X.ind == 'fin').astype(float)
X['is_tech'] = X.ind.isin(['foundry_mem', 'ic_design', 'osat_equip', 'server_odm', 'server_comp', 'pcb_ccl', 'components']).astype(float)

# ---------------- date-level (timing) candidates
A = X.groupby('date').agg(fn=('fn_val', 'sum'), tn=('tn_val', 'sum'), dsn=('dsn_val', 'sum'), dhn=('dhn_val', 'sum'),
                          amt=('amt', 'sum'), disp=('r', 'std'), gapm=('gap0', 'mean'), m=('r', 'mean')).sort_index()
T = pd.DataFrame(index=A.index)
T['agg_foreign_1'] = (A.fn / A.amt).shift(1)
T['agg_foreign_5'] = (A.fn.rolling(5).sum() / A.amt.rolling(5).sum()).shift(1)
T['agg_trust_1'] = (A.tn / A.amt).shift(1)
T['agg_trust_5'] = (A.tn.rolling(5).sum() / A.amt.rolling(5).sum()).shift(1)
T['agg_dealer_self_1'] = (A.dsn / A.amt).shift(1)
T['agg_dealer_hedge_1'] = (A.dhn / A.amt).shift(1)
T['agg_amt_ratio'] = np.log(A.amt / A.amt.rolling(20).mean()).shift(1)
T['disp_prev'] = A.disp.shift(1)
T['mkt_gap_prev'] = A.gapm.shift(1)
T['mkt_gap_bias_20'] = A.gapm.rolling(20).mean().shift(1)
T['mkt_intra_20'] = A.m.rolling(20).mean().shift(1)
T['mkt_intra_60'] = A.m.rolling(60).mean().shift(1)
T['cal_settle'] = ((A.index.dayofweek == 2) & (A.index.day >= 15) & (A.index.day <= 21)).astype(float)
T['cal_wed'] = (A.index.dayofweek == 2).astype(float)
g = read_global(os.path.join(D, 'global', 'global_data.csv'))
W = pd.DataFrame({'date': g.date})
W['txoi_z60'] = (g.TX_OI - g.TX_OI.rolling(60).mean()) / g.TX_OI.rolling(60).std()
W['txoi_chg_z'] = g.TX_OI.diff() / g.TX_OI.diff().rolling(60).std()
W['twii_ret20'] = np.log(g['^TWII']).diff(20)
W['twii_vol20'] = np.log(g['^TWII']).diff().rolling(20).std()
W['twii_ma60_gap'] = g['^TWII'] / g['^TWII'].rolling(60).mean() - 1
U = pd.DataFrame({'date': g.date})
U['vix_ma20_gap'] = np.log(g['^VIX'] / g['^VIX'].rolling(20).mean())
U['ixic_minus_gspc'] = np.log(g['^IXIC']).diff() - np.log(g['^GSPC']).diff()
U['ixic_minus_gspc_lag1'] = U.ixic_minus_gspc.shift(1)
U['gspc_ret1'] = np.log(g['^GSPC']).diff(); U['ixic_ret1'] = np.log(g['^IXIC']).diff()
dates = pd.DataFrame({'date': A.index})
T = T.reset_index()
T = pd.merge_asof(T.sort_values('date'), W.sort_values('date'), on='date', direction='backward', allow_exact_matches=False)
T = pd.merge_asof(T, U.sort_values('date'), on='date', direction='backward')

def pit(f, cols):
    try:
        x = pd.read_parquet(os.path.join(PITDIR, f)).sort_values('available_from')
    except Exception as ex:                     # no pit folder / no pyarrow: these columns stay empty
        print(f'skipping {f}: {ex}'); return pd.DataFrame({'date': T.date[:0], **{c: pd.Series(dtype=float) for c in cols}})
    x = x[['available_from'] + cols].rename(columns={'available_from': 'date'})
    x['date'] = pd.to_datetime(x.date).astype(T.date.dtype)
    return x
for f, cols in [('mkt_sox.parquet', ['sox_ret_1', 'sox_ret_5']), ('mkt_fx.parquet', ['usdtwd_chg_1', 'usdtwd_chg_20']),
                ('mkt_flows.parquet', ['mkt_foreign_1', 'mkt_foreign_5', 'mkt_trust_1', 'mkt_dealer_1'])]:
    T = pd.merge_asof(T, pit(f, cols), on='date', direction='backward')
PITEND = pd.Timestamp('2026-07-31')
for c in ['sox_ret_1', 'sox_ret_5', 'usdtwd_chg_1', 'usdtwd_chg_20', 'mkt_foreign_1', 'mkt_foreign_5', 'mkt_trust_1', 'mkt_dealer_1']:
    T.loc[T.date > PITEND, c] = np.nan
T['sox_minus_gspc'] = T.sox_ret_1 - T.gspc_ret1
T['sox_minus_ixic'] = T.sox_ret_1 - T.ixic_ret1
T['sox_minus_gspc_lag1'] = T.sox_minus_gspc.shift(1)
X = X.merge(T, on='date', how='left')

# xsec interaction candidates built from date-level info
X['semi_x_sox_ex'] = (X.is_semi - 0.5) * X.sox_minus_gspc
X['semi_x_sox_ex_lag1'] = (X.is_semi - 0.5) * X.sox_minus_gspc_lag1
X['tech_x_ixic_ex'] = (X.is_tech - 0.5) * X.ixic_minus_gspc
X['tech_x_ixic_ex_lag1'] = (X.is_tech - 0.5) * X.ixic_minus_gspc_lag1
X['fin_x_usdtwd'] = (X.is_fin - 0.5) * X.usdtwd_chg_1

# ---------------- v1 OOS predictions (ensemble_B = mean of MLP_B and Transformer_B) as "what the model already knows"
pr = []
for f in ['wf_mlp_B.pkl', 'wf_transformer_B.pkl']:
    q = pd.read_pickle(os.path.join(ROOT, 'transformer', 'results', f))[['date', 'stock_id', 'p']]
    q['stock_id'] = q.stock_id.astype(str); q['date'] = q.date.astype(X.date.dtype); pr.append(q.set_index(['date', 'stock_id']).p)
ens = pd.concat(pr, axis=1).mean(axis=1).rename('p_ens').reset_index()
X = X.merge(ens, on=['date', 'stock_id'], how='left')
X['s_v1'] = X.p_ens - X.groupby('date').p_ens.transform('mean')
pbar = X.groupby('date').p_ens.mean().rename('pbar')

# merge B features for redundancy check
Bp = P[['date', 'stock_id'] + BFEAT].copy(); Bp['stock_id'] = Bp.stock_id.astype(str); Bp['date'] = Bp.date.astype(X.date.dtype)
X = X.merge(Bp, on=['date', 'stock_id'], how='left', suffixes=('', '_B'))

# ---------------- evaluation
S = X[X.top50 & X.e.notna() & (X.date >= '2020-03-02')].copy()
def period(d):
    return np.select([d < pd.Timestamp('2022-01-01'), d < pd.Timestamp('2025-01-01'), d < pd.Timestamp('2026-06-01')],
                     ['pre', 'val', 'hold'], 'post')
S['per'] = period(S.date)
print('panel dates', S.date.min().date(), '->', S.date.max().date(), ' days:', S.date.nunique())

def daily_ic(df, f, t):
    x = df[['date', f, t]].dropna()
    gg = x.groupby('date')
    fr = gg[f].rank(); tr = gg[t].rank()
    fr = fr - fr.groupby(x.date).transform('mean'); tr = tr - tr.groupby(x.date).transform('mean')
    num = (fr * tr).groupby(x.date).sum(); den = np.sqrt((fr ** 2).groupby(x.date).sum() * (tr ** 2).groupby(x.date).sum())
    return (num / den).replace([np.inf, -np.inf], np.nan).dropna()

def ls_bp(df, f):
    x = df[['date', f, 'e']].dropna()
    rk = x.groupby('date')[f].rank(pct=True)
    sgn = np.sign(rk - x.groupby('date')[f].transform(lambda s: s.rank(pct=True).median()))
    return (sgn * x.e).groupby(x.date).mean() * 1e4

ic_se = daily_ic(S, 's_v1', 'e')
XS = ['gap_bias_60', 'gap_bias_250', 'max_20', 'min_20', 'skew_60', 'hi52', 'amihud_20', 'size_amt_20', 'foreign_net_20',
      'trust_net_20', 'dealer_hedge_20', 'foreign_hold_chg20', 'intra_vol_20', 'gap_vol_20', 'intra_ac_60', 'ret60', 'ret250_skip20',
      'ivol_60', 'resid_vol_60', 'beta_intra_120', 'dow_bias_e', 'ind_intra_prev', 'ind_intra_5', 'ind_rel_intra_prev',
      'ind_ret5', 'ind_rel_ret5', 'ind_ret20', 'ind_rel_ret20', 'ind_rel_intra_bias_20',
      'semi_x_sox_ex', 'semi_x_sox_ex_lag1', 'tech_x_ixic_ex', 'tech_x_ixic_ex_lag1', 'fin_x_usdtwd']
REF = ['intra_bias_20', 'intra_bias_250', 'rel_intra_prev', 'sector_intra_prev', 'beta_x_us', 'gap_bias_20', 'ret5', 'foreign_net_5d', 's_v1']
BX = [c for c in BFEAT if S.groupby('date')[c].nunique().median() > 5]   # features that vary across stocks

out = []
for f in REF + XS:
    ic = daily_ic(S, f, 'e')
    row = {'feature': f, 'ref': f in REF}
    per = pd.Series(period(ic.index), index=ic.index)
    for p in ['pre', 'val', 'hold', 'post']:
        v = ic[per == p]; row[f'IC_{p}'] = v.mean(); row[f't_{p}'] = v.mean() / v.std() * np.sqrt(len(v)) if len(v) > 5 else np.nan
    allv = ic[ic.index >= '2022-01-01']; row['IC_22on'] = allv.mean(); row['t_22on'] = allv.mean() / allv.std() * np.sqrt(len(allv))
    lb = ls_bp(S[S.date >= '2022-01-01'], f); row['LS_bp_22on'] = lb.mean()
    if f != 's_v1':
        ic_fs = daily_ic(S, f, 's_v1'); j = pd.concat([ic.rename('fe'), ic_fs.rename('fs'), ic_se.rename('se')], axis=1).dropna()
        pc = (j.fe - j.fs * j.se) / np.sqrt((1 - j.fs ** 2) * (1 - j.se ** 2))
        row['pIC_22on'] = pc.mean(); row['pt_22on'] = pc.mean() / pc.std() * np.sqrt(len(pc))
        if f not in REF:
            best, bc = None, 0
            for b in BX:
                cc = daily_ic(S[S.date >= '2022-01-01'], f, b).mean()
                if abs(cc) > abs(bc): best, bc = b, cc
            row['max_corr_B'] = f'{bc:+.2f} {best}'
    out.append(row)
R = pd.DataFrame(out)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print('\n=== XSEC: daily Spearman IC with residual intraday return (Top50). pIC = partial IC given v1 ensemble_B xsec signal ===')
print(R.round(4).to_string(index=False))

# ---------------- timing
Dd = S.groupby('date').agg(m50=('r', 'mean')).join(pbar)
Tt = T.set_index('date')
Pd = P.groupby('date')[['txoi_chg', 'us_ixic_ret1', 'us_ixic_ret1_lag1', 'us_vix_chg_lag1', 'mkt_intra_prev', 'breadth_prev', 'mkt_intra_5']].first()
Pd.index = Pd.index.astype(Dd.index.dtype)
Dd = Dd.join(Tt).join(Pd)
Dd['per'] = period(Dd.index)
TF = ['txoi_chg', 'us_ixic_ret1', 'us_ixic_ret1_lag1', 'mkt_intra_5',
      'agg_foreign_1', 'agg_foreign_5', 'agg_trust_1', 'agg_trust_5', 'agg_dealer_self_1', 'agg_dealer_hedge_1', 'agg_amt_ratio',
      'disp_prev', 'mkt_gap_prev', 'mkt_gap_bias_20', 'mkt_intra_20', 'mkt_intra_60', 'txoi_z60', 'txoi_chg_z', 'twii_ret20',
      'twii_vol20', 'twii_ma60_gap', 'vix_ma20_gap', 'ixic_minus_gspc', 'sox_ret_1', 'sox_ret_5', 'sox_minus_gspc', 'usdtwd_chg_1',
      'usdtwd_chg_20', 'mkt_foreign_1', 'mkt_foreign_5', 'mkt_trust_1', 'mkt_dealer_1', 'cal_settle', 'cal_wed']
tout = []
for f in TF:
    x = Dd[[f, 'm50', 'pbar', 'per']].dropna(subset=[f, 'm50'])
    lo, hi = x[f].quantile([0.01, 0.99]); fx = x[f].clip(lo, hi)
    row = {'feature': f}
    for p in ['pre', 'val', 'hold', 'post']:
        k = x.per == p
        if k.sum() > 20:
            cc = np.corrcoef(fx[k], x.m50[k])[0, 1]; row[f'corr_{p}'] = cc; row[f'n_{p}'] = int(k.sum())
    k = x.index >= '2022-01-01'
    cc = np.corrcoef(fx[k], x.m50[k])[0, 1]; row['corr_22on'] = cc; row['t_22on'] = cc * np.sqrt(k.sum())
    xx = x[k].dropna(subset=['pbar'])
    if len(xx) > 50:
        a = np.corrcoef(fx[k].loc[xx.index], xx.m50)[0, 1]; b = np.corrcoef(fx[k].loc[xx.index], xx.pbar)[0, 1]; c = np.corrcoef(xx.pbar, xx.m50)[0, 1]
        row['pcorr_22on|pbar'] = (a - b * c) / np.sqrt((1 - b ** 2) * (1 - c ** 2))
    tout.append(row)
print('\n=== TIMING: corr(feature_t, Top50 mean intraday return_t); pcorr controls for v1 ensemble pbar ===')
print(pd.DataFrame(tout).round(3).to_string(index=False))

# settlement / weekday means (bp) for reference
print('\n=== m50 mean (bp) by calendar flag ===')
for f in ['cal_settle', 'cal_wed']:
    print(f, Dd.groupby([Dd.per, Dd[f]]).m50.mean().unstack().mul(1e4).round(1).to_string())

# ---------------- follow-ups
X = X.sort_values(['stock_id', 'date']).reset_index(drop=True)
X['ebias_260'] = X.groupby('stock_id').e.transform(lambda s: s.rolling(260, min_periods=100).mean().shift(1))
X['dow_dev'] = X.dow_bias_e - X.ebias_260
S = X[X.top50 & X.e.notna() & (X.date >= '2020-03-02')].copy()
def crank(df, f):
    r = df.groupby('date')[f].rank(pct=True); return r - r.groupby(df.date).transform('mean')
def tstat(s): return s.mean() / s.std() * np.sqrt(len(s))
def pic(df, f, c):
    j = pd.concat([daily_ic(df, f, 'e').rename('fe'), daily_ic(df, f, c).rename('fc'), daily_ic(df, c, 'e').rename('ce')], axis=1).dropna()
    return (j.fe - j.fc * j.ce) / np.sqrt((1 - j.fc ** 2) * (1 - j.ce ** 2))
print('\n=== same-weekday seasonality: is it more than the stock\'s overall intraday bias? ===')
for f in ['dow_bias_e', 'dow_dev', 'ebias_260']:
    s = daily_ic(S, f, 'e'); pp = pd.Series(period(s.index), index=s.index)
    print(f, ' '.join(f'{p}:{s[pp == p].mean():+.4f}({tstat(s[pp == p]):+.1f})' for p in ['pre', 'val', 'hold', 'post']))
    for c in ['intra_bias_250', 's_v1']:
        q = pic(S[S.date >= '2022-01-01'], f, c); print(f'   partial | {c}: {q.mean():+.4f} (t {tstat(q):+.1f})')

print('\n=== 5 new xsec features combined (signs fixed on 2020-03..2021) vs v1 ensemble_B xsec signal ===')
NEW = ['gap_bias_250', 'gap_bias_60', 'resid_vol_60', 'dow_bias_e', 'hi52']
pre = S[S.date < '2022-01-01']
sg = {f: float(np.sign(daily_ic(pre, f, 'e').mean())) for f in NEW}; print('signs:', sg)
S['new_combo'] = sum(sg[f] * crank(S, f).fillna(0) for f in NEW) / len(NEW)
S['s_rank'] = crank(S, 's_v1')
V = S[S.s_v1.notna()].copy(); V['per'] = period(V.date)
print('IC new_combo:', {p: round(daily_ic(V[V.per == p], 'new_combo', 'e').mean(), 4) for p in ['val', 'hold']},
      ' daily corr(new_combo, s_v1):', round(daily_ic(V, 'new_combo', 's_v1').mean(), 3))
def ls_sign(df, f):
    x = df[['date', f, 'e']].dropna(); r = crank(x, f)
    return (np.sign(r) * x.e).groupby(x.date).mean() * 1e4
rows = []
for lam in [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
    V['comb'] = V.s_rank + lam * V.new_combo; o = {'lambda': lam}
    for p in ['val', 'hold']:
        o[f'LS_bp_{p}'] = round(ls_sign(V[V.per == p], 'comb').mean(), 2); o[f'IC_{p}'] = round(daily_ic(V[V.per == p], 'comb', 'e').mean(), 4)
    rows.append(o)
print(pd.DataFrame(rows).to_string(index=False))
P2 = S[S.date >= '2026-06-01']
print('post (2026-06..) new_combo alone: IC', round(daily_ic(P2, 'new_combo', 'e').mean(), 4), ' LS_bp', round(ls_sign(P2, 'new_combo').mean(), 2))

print('\n=== December vs quarter-end vs other months (IC with e) ===')
for f in ['trust_net_20', 'trust_net_5d', 'trust_net', 'foreign_net_20', 'hi52', 'gap_bias_250', 'resid_vol_60', 'dow_bias_e']:
    s = daily_ic(S, f, 'e'); mm = pd.Series(s.index.month, index=s.index)
    dec = s[mm == 12]; qe = s[mm.isin([3, 6, 9])]; oth = s[~mm.isin([3, 6, 9, 12])]
    print(f'{f:16s} Dec {dec.mean():+.4f} (t {tstat(dec):+.1f}, n={len(dec)})  Mar/Jun/Sep {qe.mean():+.4f} (t {tstat(qe):+.1f})  other {oth.mean():+.4f} (t {tstat(oth):+.1f})')

print('\n=== monthly revenue & valuation (data/pit, PIT available_from; ends 2026-07) ===')
R2 = S[S.date <= '2026-07-31'][['date', 'stock_id', 'e', 's_v1']].copy()
def asof(df0, src, cols):
    src = src.copy(); src['stock_id'] = src.stock_id.astype(str); src['af'] = pd.to_datetime(src.available_from).astype(df0.date.dtype)
    return pd.merge_asof(df0.sort_values('date'), src.sort_values('af')[['stock_id', 'af'] + cols], left_on='date', right_on='af', by='stock_id', direction='backward')
try:
    R2 = asof(R2, pd.read_parquet(os.path.join(PITDIR, 'revenue.parquet')), ['rev_yoy', 'rev_mom', 'rev_surprise'])
    cal = pd.read_parquet(os.path.join(D, 'ref', 'calendar.parquet')); pos = dict(zip(cal.date.astype(R2.date.dtype), cal.pos))
    R2['days_since_rev'] = R2.date.map(pos) - R2.af.map(pos)
    R2['rev_surprise_fresh5'] = R2.rev_surprise.where(R2.days_since_rev <= 4)
    R2 = asof(R2.drop(columns='af'), pd.read_parquet(os.path.join(PITDIR, 'per.parquet')), ['earnings_yield', 'book_to_price', 'dividend_yield']).drop(columns='af')
    for f in ['rev_yoy', 'rev_mom', 'rev_surprise', 'rev_surprise_fresh5', 'earnings_yield', 'book_to_price', 'dividend_yield']:
        s = daily_ic(R2, f, 'e'); pp = pd.Series(period(s.index), index=s.index); q = pic(R2, f, 's_v1')
        print(f'{f:20s}', ' '.join(f'{p}:{s[pp == p].mean():+.4f}({tstat(s[pp == p]):+.1f})' for p in ['pre', 'val', 'hold', 'post']),
              f'| partial|s_v1 {q.mean():+.4f} (t {tstat(q):+.1f})')
except Exception as ex:
    print('skipped (needs data/pit/revenue.parquet, per.parquet + pyarrow):', ex)

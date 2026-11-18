"""
mlib.py -- shared helpers for the v8 market-view (大盤觀點) research.

Inputs (all produced by earlier versions, nothing new is trained here):
  * v8/cache/panel.pkl          : v4/v5 panel (comp_features + features_extra), built by ens_lib.build_panel
  * v8/cache/wf_members.csv.gz  : seed-averaged walk-forward predictions of the 5 v4 members (x0_export_wf.py):
                                  yearly folds 2022-01..2026-05, post (2026-06..) = models frozen at 2026-05-29
Periods are the same as v2/v4/v5: val = 2022-2024 (decides), hold = 2025-01..2026-05 (confirms), post = 2026-06.. (report only).
Score = competition AR on the Top50 in bp/day:  AR_t = mean_i (2 d_i - 1) r_i ;  timing_t = mean_i(2 d_i - 1) * mean_i(r_i).
"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); V8 = os.path.dirname(HERE); ROOT = os.path.dirname(V8)
CACHE = os.path.join(V8, 'cache')
CUT = pd.Timestamp('2025-01-01'); POST = pd.Timestamp('2026-06-01'); VAL0 = pd.Timestamp('2022-01-01')
RAW = ['mlp_raw_B', 'mlp_raw_xr', 'tf_raw_B']; RES = ['mlp_res_B', 'mlp_res_xr']

_P = None
def panel():
    global _P
    if _P is None:
        f = os.path.join(CACHE, 'panel.pkl')
        if not os.path.exists(f):
            sys.path.insert(0, os.path.join(ROOT, 'v5'))
            import ens_lib as E
            P = E.build_panel(E.DATA); P['date'] = P['date'].astype('datetime64[ns]'); P.to_pickle(f)
        _P = pd.read_pickle(f)
        _P['stock_id'] = _P.stock_id.astype(str)
    return _P

def members():
    X = pd.read_csv(os.path.join(CACHE, 'wf_members.csv.gz'), dtype={'stock_id': str}, parse_dates=['date'])
    X['date'] = X.date.astype('datetime64[ns]')
    return X.set_index(['date', 'stock_id']).sort_index()

def views(M):
    """v4 views: p_raw = mean(mlp family mean, transformer), pbar = day mean over the 100 stocks;
    p_e = mean of the two resid MLPs, p_e_c = centred per day"""
    out = pd.DataFrame(index=M.index)
    out['p_raw'] = (M[['mlp_raw_B', 'mlp_raw_xr']].mean(axis=1) + M['tf_raw_B']) / 2
    out['p_raw_mlp'] = M[['mlp_raw_B', 'mlp_raw_xr']].mean(axis=1); out['p_raw_tf'] = M['tf_raw_B']
    out['pbar'] = out.groupby(level='date').p_raw.transform('mean')
    out['p_e'] = M[RES].mean(axis=1)
    out['p_e_c'] = out.p_e - out.groupby(level='date').p_e.transform('mean')
    return out

def market_daily(P=None):
    """daily market series on labelled days: m = Top50 mean intraday return (the timing target), m100 = Top100 mean"""
    P = panel() if P is None else P
    q = P[P.target_r.notna()]
    d = pd.DataFrame({'m': q[q.top50].groupby('date').target_r.mean(), 'm100': q.groupby('date').target_r.mean()})
    return d.sort_index()

def daily(score, P=None, exdiv=True):
    """score: Series indexed by (date, stock_id), prediction 1 if score > 0 -> per-day AR, timing, xsec, share1, m (bp)"""
    P = panel() if P is None else P
    q = score.rename('score').reset_index().merge(P[['date', 'stock_id', 'top50', 'target_r', 'ex_today']], on=['date', 'stock_id'])
    q = q[q.target_r.notna() & q.top50]
    pred = (q.score > 0).astype(int).values
    if exdiv:
        pred = np.where(q.ex_today.values == 1, 0, pred)
    q = q.assign(s=2 * pred - 1)
    g = q.groupby('date')
    AR = (q.s * q.target_r).groupby(q.date).mean() * 1e4
    sbar = g.s.mean(); m = g.target_r.mean()
    T = sbar * m * 1e4
    return pd.DataFrame({'AR': AR, 'timing': T, 'xsec': AR - T, 'share1': (sbar + 1) / 2, 'sbar': sbar, 'm': m * 1e4})

def periods(idx):
    idx = pd.DatetimeIndex(idx)
    return {'pre': idx < VAL0, 'val': (idx >= VAL0) & (idx < CUT), 'hold': (idx >= CUT) & (idx < POST), 'post': idx >= POST,
            'all22': idx >= VAL0}

def plain_t(d):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    return d.mean() / d.std(ddof=1) * np.sqrt(len(d)) if len(d) > 2 and d.std() > 0 else np.nan

def nw_t(d, lags=5):
    d = np.asarray(d, float); d = d[~np.isnan(d)]; n = len(d)
    if n < 3:
        return np.nan
    e = d - d.mean(); s = e @ e / n
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return d.mean() / np.sqrt(s / n) if s > 0 else np.nan

def corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); k = ~(np.isnan(a) | np.isnan(b))
    return np.corrcoef(a[k], b[k])[0, 1] if k.sum() > 3 else np.nan

def table(D, base=None, label='', cols=('AR',), per=('val', 'hold', 'post', 'all22')):
    r = {'config': label}
    pr = periods(D.index)
    for k in per:
        r[k] = D['AR'][pr[k]].mean()
    r['timing'] = D.timing[pr['all22']].mean(); r['xsec'] = D.xsec[pr['all22']].mean(); r['sd'] = D.AR[pr['all22']].std()
    if base is not None:
        d = D['AR'] - base['AR'].reindex(D.index)
        for k in per:
            r['d_' + k] = d[pr[k]].mean(); r['t_' + k] = plain_t(d[pr[k]])
    return r

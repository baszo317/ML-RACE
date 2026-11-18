"""
v9lib.py -- shared code of v9 ("every change with a positive val t, mixed into one model", PREREG_v9.md).

Panels
  core : v4/v5 panel (ens_lib.build_panel, the 100 competition stocks) + TX night session (v7 e7: night_ret, night_range)
         + sigma columns (v5 wf_runner: absr_60 = trailing 60-day mean of min(|r|, 5%), absres_60 = trailing 60-day mean of
         |r - mean_100(r)|, both up to the day before) + the 5 v3 features and their ranks (棄案\\v3\\features_v3.py)
  ext  : v7's panel_ext.parquet (core + point-in-time Top300 extra stocks, B features, residual label from the 100 only)
         + the same sigma columns
Members (MEMBERS below): MLP training = v5 wf_runner.train_custom with the sigma weighting (alpha 0.5) plus an optional
training window; everything else as v4 (ens_lib.HP, early stopping on the last 120 training days, refit).
"""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); V9 = os.path.dirname(HERE); ROOT = os.path.dirname(V9)
for p in (os.path.join(ROOT, 'v8', 'research'), os.path.join(ROOT, 'v5'), os.path.join(ROOT, 'transformer'), os.path.join(ROOT, '棄案', 'v3')):
    if p not in sys.path:
        sys.path.insert(0, p)
import mlib as L                     # v8 helpers (panel, members, periods, t-stats)
CACHE = os.path.join(V9, 'cache')
NAS = ['us_ixic_ret1', 'us_ixic_ret5', 'us_ixic_ret1_lag1']
NIGHT = ['night_ret', 'night_range']
SIG = ['absr_60', 'absres_60']
FOLDS = [('2022-01-01', '2023-01-01'), ('2023-01-01', '2024-01-01'), ('2024-01-01', '2025-01-01'), ('2025-01-01', '2026-01-01'),
         ('2026-01-01', '2026-06-01'), ('2026-06-01', '2100-01-01')]

def feature_lists():
    import features_extra as fe
    from features_v3 import NEW, XRN
    B = json.load(open(os.path.join(ROOT, 'transformer', 'final_feature_lists.json')))['B_competition_only']
    B9 = [f for f in B if f not in NAS]
    return dict(B9=B9, XR=list(fe.XR), NEW=list(NEW), XRN=list(XRN))

def members():
    F = feature_lists(); B9, XR, NEW, XRN = F['B9'], F['XR'], F['NEW'], F['XRN']
    return {
        'mlp_raw_B9':   dict(target='raw',   feats=B9 + NIGHT + SIG,            window=None, panel='core'),
        'mlp_raw_xr9':  dict(target='raw',   feats=B9 + XR + NIGHT + SIG,       window=None, panel='core'),
        'mlp_raw_w4_9': dict(target='raw',   feats=B9 + NIGHT + SIG,            window=4,    panel='core'),
        'mlp_res_B9':   dict(target='resid', feats=B9 + SIG,                    window=None, panel='ext'),
        'mlp_res_xr9':  dict(target='resid', feats=B9 + XR + SIG,               window=None, panel='core'),
        'mlp_res_BN9':  dict(target='resid', feats=B9 + NEW + SIG,              window=None, panel='core'),
        'mlp_res_xrN9': dict(target='resid', feats=B9 + XR + NEW + XRN + SIG,   window=None, panel='core'),
    }

def night_table():
    """v7 e7 exactly: FinMind TX rows; the night ('after_market') row dated D is the session that ends at 05:00 of D"""
    tx = pd.read_csv(os.path.join(ROOT, 'v7', 'data', 'tx_daily.csv'), dtype={'contract_date': str})
    tx['date'] = pd.to_datetime(tx.date).astype('datetime64[ns]')
    tx = tx[(tx.futures_id == 'TX') & tx.contract_date.str.fullmatch(r'\d{6}')]
    day = tx[tx.trading_session == 'position'][['date', 'contract_date', 'open', 'close']]
    ngt = tx[(tx.trading_session == 'after_market') & (tx.volume > 0)]
    ngt = ngt.sort_values(['date', 'volume'], ascending=[True, False]).drop_duplicates('date')[['date', 'contract_date', 'open', 'max', 'min', 'close']]
    dd = np.sort(day.date.unique()); i = np.searchsorted(dd, ngt.date.values) - 1
    ngt = ngt[i >= 0].copy(); ngt['prev_day'] = dd[i[i >= 0]]
    N = ngt.merge(day.rename(columns={'date': 'prev_day', 'close': 'prev_close', 'open': 'prev_open'}), on=['prev_day', 'contract_date'], how='left')
    N['night_ret'] = np.log(N.close / N.prev_close); N['night_range'] = (N['max'] - N['min']) / N.prev_close
    return N[['date', 'night_ret', 'night_range']]

def add_sigma(P):
    P = P.sort_values(['stock_id', 'date']).copy()
    a = P.target_r.abs().clip(upper=0.05)
    P['absr_60'] = a.groupby(P.stock_id).transform(lambda s: s.rolling(60, min_periods=20).mean().shift(1))
    if 'absres_60' not in P:
        P['absres_60'] = P.resid_r.abs().groupby(P.stock_id).transform(lambda s: s.rolling(60, min_periods=20).mean().shift(1))
    return P.sort_values(['date', 'stock_id']).reset_index(drop=True)

def core_panel():
    f = os.path.join(CACHE, 'core9.pkl')
    if os.path.exists(f):
        return pd.read_pickle(f)
    from features_v3 import add_v3
    P = L.panel().copy()
    P = add_v3(P, os.path.join(ROOT, 'data'))
    P = P.merge(night_table(), on='date', how='left')
    P = add_sigma(P)
    os.makedirs(CACHE, exist_ok=True); P.to_pickle(f)
    return P

def ext_panel():
    f = os.path.join(CACHE, 'ext9.pkl')
    if os.path.exists(f):
        return pd.read_pickle(f)
    X = pd.read_parquet(os.path.join(ROOT, 'v7', 'cache', 'panel_ext.parquet'))
    X['date'] = pd.to_datetime(X.date).astype('datetime64[ns]'); X['stock_id'] = X.stock_id.astype(str)
    X = add_sigma(X)
    X.to_pickle(f)
    return X

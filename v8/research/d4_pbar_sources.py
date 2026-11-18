"""d4: how much of the market view pbar is explained by (a) yesterday's Taiwan market, (b) last night's U.S. / night session?
OLS R^2 on standardised inputs, 2022-01..2026-10 (descriptive). run: python d4_pbar_sources.py > ../results/d4_pbar_sources.txt"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mlib as L
G = pd.read_pickle(os.path.join(L.CACHE, 'daily_market.pkl')); X = pd.read_pickle(os.path.join(L.CACHE, 'daily_view.pkl'))
J = X[['pbar']].join(G, how='inner')
def r2(cols, y='pbar', J=J):
    k = J[cols + [y]].notna().all(axis=1); Z = J.loc[k, cols]; Z = (Z - Z.mean()) / Z.std()
    A = np.c_[np.ones(k.sum()), Z.values]; b, *_ = np.linalg.lstsq(A, J.loc[k, y].values, rcond=None)
    e = J.loc[k, y].values - A @ b
    return 1 - e.var() / J.loc[k, y].var(), dict(zip(cols, b[1:]))
sets = {'(a) yesterday TW: gap, intraday, TX OI change (3)': ['mkt_gap_prev', 'mkt_intra_prev', 'txoi_chg'],
        '(a+) yesterday TW, 8 inputs': ['mkt_gap_prev', 'mkt_intra_prev', 'txoi_chg', 'txoi_chg5', 'twii_ret1', 'ks11_ret1', 'breadth_prev', 'mkt_intra_5'],
        '(b) last night: S&P, Nasdaq, VIX change, TX night session (4)': ['us_gspc_ret1', 'us_ixic_ret1', 'us_vix_chg', 'night_ret']}
for lab, cols in sets.items():
    v, b = r2(cols)
    print(f'{lab:62s} R^2 = {v:.3f};  coefficients (pbar units per sd): ' + ', '.join(f'{c} {x:+.4f}' for c, x in b.items()))
print('\nsame sets explaining the realised Top50 intraday return m (in-sample R^2, 2022-01..2026-10, for scale):')
for lab, cols in sets.items():
    v, _ = r2(cols, 'm'); print(f'  {lab:60s} R^2 = {v:.4f}')

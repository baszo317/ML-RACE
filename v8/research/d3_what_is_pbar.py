"""d3: what is the market view made of, and how does m_t depend on the overnight move?  (descriptive)
(1) corr(pbar, each market-level input) by period -> which signals the raw members actually use for the market
(2) Top50 intraday return by quintile of the overnight move (night session, S&P), per period -> shape of the 'gap fade'
(3) the stock view's beta tilt B_t: what moves it
run: python d3_what_is_pbar.py > ../results/d3_what_is_pbar.txt"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mlib as L
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
G = pd.read_pickle(os.path.join(L.CACHE, 'daily_market.pkl')); X = pd.read_pickle(os.path.join(L.CACHE, 'daily_view.pkl'))
D4 = pd.read_pickle(os.path.join(L.CACHE, 'decomp_v4.pkl'))
J = X[['pbar', 'pbar_mlp', 'pbar_tf', 'sbar']].join(G.drop(columns=['m', 'm100']), how='inner').join(D4[['B']], how='inner')
J['m'] = G.m.reindex(J.index)
feats = ['us_gspc_ret1', 'us_ixic_ret1', 'night_ret', 'us_vix_chg', 'us_vix', 'twii_ret1', 'ks11_ret1', 'txoi', 'txoi_chg', 'txoi_chg5',
         'mkt_intra_prev', 'mkt_intra_5', 'mkt_intra_20', 'mkt_gap_prev', 'mkt_ret1_prev', 'breadth_prev', 'mkt_absr_20', 'cal_tom']
pr = L.periods(J.index)
rows = []
for f in feats:
    r = {'input': f}
    for k in ('val', 'hold', 'post'):
        r['pbar~ ' + k] = L.corr(J.pbar[pr[k]], J[f][pr[k]])
    for k in ('val', 'hold', 'post'):
        r['m~ ' + k] = L.corr(J.m[pr[k]], J[f][pr[k]])
    r['B~ all'] = L.corr(J.B, J[f])
    rows.append(r)
print('(1) correlation of pbar (market view) and of m (realised) with each input, 2022-2026; last column: the stock view beta tilt B')
print(pd.DataFrame(rows).round(3).to_string(index=False))
# linear decomposition of pbar on the inputs (standardised), val+hold
k = (J.index < L.POST) & J[feats].notna().all(axis=1)
Z = (J.loc[k, feats] - J.loc[k, feats].mean()) / J.loc[k, feats].std()
b, *_ = np.linalg.lstsq(np.c_[np.ones(k.sum()), Z.values], J.pbar[k].values - J.pbar[k].mean(), rcond=None)
fit = np.c_[np.ones(k.sum()), Z.values] @ b
print(f'\n    R^2 of pbar on these {len(feats)} market inputs (2022-01..2026-05): {1 - np.var(J.pbar[k].values - J.pbar[k].mean() - fit) / np.var(J.pbar[k]):.3f}')
print('    standardised loadings (x1000): ' + ', '.join(f'{f} {v * 1000:.1f}' for f, v in sorted(zip(feats, b[1:]), key=lambda t: -abs(t[1]))[:10]))

print('\n(2) mean Top50 intraday return (bp) by quintile of the overnight move (quintiles within each period)')
for f in ['night_ret', 'us_gspc_ret1']:
    rows = []
    for kname in ('pre', 'val', 'hold', 'post'):
        g = J if kname != 'pre' else None
        if kname == 'pre':
            q = G[L.periods(G.index)['pre']][[f, 'm']].dropna()
        else:
            q = J[pr[kname]][[f, 'm']].dropna()
        qq = pd.qcut(q[f], 5, labels=False)
        rows.append({'period': kname, **{f'Q{i + 1}': q.m[qq == i].mean() * 1e4 for i in range(5)}, 'n': len(q),
                     'Q5-Q1': (q.m[qq == 4].mean() - q.m[qq == 0].mean()) * 1e4})
    print(f'  {f}:'); print(pd.DataFrame(rows).round(1).to_string(index=False))

print('\n(3) beta tilt B_t of the stock view (v4 rule): mean by period and relation to the market inputs')
print('    mean B: ' + ', '.join(f'{k} {J.B[m_].mean():.3f}' for k, m_ in pr.items() if m_.sum()))
print(f'    corr(B, pbar) {L.corr(J.B, J.pbar):.3f}; corr(B, m) {L.corr(J.B, J.m):.3f}; mean m by period (bp): ' +
      ', '.join(f'{k} {J.m[m_].mean() * 1e4:.2f}' for k, m_ in pr.items() if m_.sum()))

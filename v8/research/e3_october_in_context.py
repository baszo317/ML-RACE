"""e3 (descriptive): is October 2026 (15 live days so far) unusual for the market-related parts?  Percentile of the 15-day
averages among all 15-day windows of the walk-forward history 2022-01..2026-09 (v4 rule).  Also the sampling noise of a
correlation over n days.  run: python e3_october_in_context.py > ../results/e3_october_in_context.txt"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mlib as L
D = pd.read_pickle(os.path.join(L.CACHE, 'decomp_v4.pkl'))
D['mkt_part'] = D.timing + D.beta_bet
oc = D[D.index >= '2026-10-01']; hist = D[D.index < '2026-10-01']; n = len(oc)
print(f'October 2026: {n} days')
for c in ['AR', 'timing', 'beta_bet', 'mkt_part', 'rest', 'm']:
    h = hist[c].rolling(n).mean().dropna(); v = oc[c].mean()
    print(f'  {c:9s} {v:8.2f} bp/day   percentile among {len(h)} historical {n}-day windows: {100 * (h <= v).mean():5.1f} %')
for nn in (15, 19, 99, 250):
    print(f'sampling sd of a correlation estimated on {nn} days (true value near 0): about {1 / np.sqrt(nn):.2f}')

# model-free view of the hidden market bet: regress the daily stock-picking part (xsec) on the market move m
print('\nmodel-free check (no beta estimate): slope of daily xsec on m = hidden market exposure of the stock view')
pr = L.periods(D.index)
for k, v in list(pr.items()) + [('Oct26', D.index >= '2026-10-01')]:
    x = D[v]
    if len(x) < 10:
        continue
    b = np.polyfit(x.m, x.xsec, 1)[0]
    print(f'  {k:6s} n {len(x):4d}: slope {b:+.3f}, corr(xsec, m) {L.corr(x.xsec, x.m):+.3f}, mean m {x.m.mean():+.2f} -> implied hidden-bet P&L {b * x.m.mean():+.2f} bp/day '
          f'(beta decomposition says {x.beta_bet.mean():+.2f})')
b_all = np.polyfit(D[pr['all22']].m, D[pr['all22']].xsec, 1)[0]
print(f'  with the 2022-26 slope {b_all:+.3f}, October implied hidden-bet P&L = {b_all * oc.m.mean():+.2f} bp/day; explicit timing {oc.timing.mean():+.2f}')
print('\ncorrelation of the daily total score AR (and its parts) with the market move m (v4 rule):')
for k, v in pr.items():
    if v.sum() > 10:
        x = D[v]
        print(f'  {k:6s}: corr(AR, m) {L.corr(x.AR, x.m):+.3f}, corr(timing, m) {L.corr(x.timing, x.m):+.3f}, corr(xsec, m) {L.corr(x.xsec, x.m):+.3f}')

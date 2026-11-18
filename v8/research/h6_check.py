"""h6_check: robustness of the report-only partial hedge (H6b, lambda 0.5): is the c grid wide enough, how many stocks flip,
and is the difference to v4 spread over the years or driven by one period?  run: python h6_check.py > ../results/h6_check.txt"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mlib as L
import h6_hedge as H          # reuses the data and functions (prints its own report first)
F, CGRID = H.F, H.CGRID
rows = []
for d, g in F.groupby('date'):
    sc, b, ex = g.score.values, g.beta.values, g.ex_today.values == 1
    n = len(sc); s0 = np.where(ex, -1, np.where(sc > 0, 1, -1)); sbar0 = s0.mean(); bbar = b.mean()
    B0 = ((s0 - sbar0) * (b - bbar)).mean()
    S = (sc[None, :] + CGRID[:, None] * b[None, :] > 0) & ~ex[None, :]
    Ec = ((2 * S - 1) @ b) / n
    for lam in (0.5, 0.0):
        Et = sbar0 * bbar + lam * B0
        dc = np.abs(Ec - Et); ca = np.flatnonzero(dc <= dc.min() + 1e-12); j = ca[np.argmin(np.abs(CGRID[ca]))]
        s1 = np.where(S[j], 1, -1)
        rows.append({'date': d, 'lam': lam, 'c': CGRID[j], 'at_edge': abs(CGRID[j]) >= 0.3 - 1e-9, 'flips': int((s1 != s0).sum()),
                     'gap_E': Ec[j] - Et, 'n1_v4': int((s0 > 0).sum()), 'n1_new': int((s1 > 0).sum())})
R = pd.DataFrame(rows)
print('\n=== h6_check ===')
for lam, g in R.groupby('lam'):
    print(f'lambda {lam}: c median {g.c.median():.4f} (p5 {g.c.quantile(.05):.4f}, p95 {g.c.quantile(.95):.4f}); days at the grid edge {g.at_edge.mean():.3f}; '
          f'stocks flipped per day: mean {g.flips.mean():.1f}, median {g.flips.median():.0f}; number of 1s v4 {g.n1_v4.mean():.1f} -> {g.n1_new.mean():.1f}; '
          f'|E - E*| median {g.gap_E.abs().median():.4f}')
D = pd.read_pickle(os.path.join(L.CACHE, 'h6_daily.pkl'))
base, h = D[('v4', 1.0)], D[('H6b', 0.5)]
y = base.index.year.astype(str).values.astype(object); y[base.index >= '2026-06-01'] = '2026-06..10'; y[(base.index >= '2026-01-01') & (base.index < '2026-06-01')] = '2026-01..05'
T = pd.DataFrame({'v4': base.AR, 'H6b_0.5': h.AR, 'per': y}).groupby('per').agg(days=('v4', 'size'), v4=('v4', 'mean'), hedge=('H6b_0.5', 'mean'))
T['diff'] = T.hedge - T.v4
T['sd_v4'] = base.AR.groupby(y).std().values; T['sd_hedge'] = h.AR.groupby(y).std().values
print('\nby year, bp/day (H6b lambda 0.5 vs v4):'); print(T.round(2).to_string())

"""e4 (EXPLORATORY, added after h6): the fully market-neutral book -- hedge the TOTAL beta exposure to E* = 0 (no explicit market
view, no hidden tilt) with the H6b method (score + c*beta), and the same with the explicit market view only (E* = sbar_v4*mean beta,
= H6b lambda 0) for comparison.  run: python e4_full_neutral.py > ../results/e4_full_neutral.txt"""
import os, sys, io, contextlib
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mlib as L
with contextlib.redirect_stdout(io.StringIO()):
    import h6_hedge as H                       # data + grid (its report is suppressed here)
F, CGRID = H.F, H.CGRID
rows = []
for d, g in F.groupby('date'):
    sc, b, ex, r = g.score.values, g.beta.values, g.ex_today.values == 1, g.target_r.values
    n = len(sc)
    S = (sc[None, :] + CGRID[:, None] * b[None, :] > 0) & ~ex[None, :]
    Ec = ((2 * S - 1) @ b) / n
    dc = np.abs(Ec - 0.0); ca = np.flatnonzero(dc <= dc.min() + 1e-12); j = ca[np.argmin(np.abs(CGRID[ca]))]
    s = np.where(S[j], 1, -1); m = r.mean()
    rows.append((d, (s * r).mean() * 1e4, s.mean() * m * 1e4, m * 1e4, (s * b).mean()))
N = pd.DataFrame(rows, columns=['date', 'AR', 'timing', 'm', 'E']).set_index('date')
D = pd.read_pickle(os.path.join(L.CACHE, 'h6_daily.pkl')); base = D[('v4', 1.0)]
pr = L.periods(N.index); r19 = N.AR.rolling(19).mean().dropna(); dd = N.AR - base.AR
print('fully market-neutral book (E* = 0, H6b method), Top50 bp/day, walk-forward 2022-01..2026-10')
print(f'  all {N.AR.mean():.2f} (val {N.AR[pr["val"]].mean():.2f}, hold {N.AR[pr["hold"]].mean():.2f}, post {N.AR[pr["post"]].mean():.2f}); sd {N.AR.std():.1f}; '
      f'corr(AR, m) {L.corr(N.AR, N.m):+.3f}; mean |E| {N.E.abs().mean():.3f}; P19>0 {(r19 > 0).mean():.3f}, P19>20 {(r19 > 20).mean():.3f}')
print(f'  vs v4: all {dd.mean():+.2f}, val {dd[pr["val"]].mean():+.2f} (t {L.plain_t(dd[pr["val"]]):.2f}), hold {dd[pr["hold"]].mean():+.2f}, post {dd[pr["post"]].mean():+.2f}')
q = pd.qcut(base.m.rank(method='first'), 10, labels=False)
print(f'  by market day: biggest-down 10% {N.AR[q == 0].mean():.2f} | middle 80% {N.AR[(q > 0) & (q < 9)].mean():.2f} | biggest-up 10% {N.AR[q == 9].mean():.2f}')

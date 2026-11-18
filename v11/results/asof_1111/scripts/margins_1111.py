"""How far each 11/11 decision is from flipping (from the probabilities file the run wrote; score and beta have 6 decimals).

For the chosen hedge shift c: the smallest |score + c * beta| over the Top50 stocks that are not forced to 0 by the
ex-dividend rule; and on the whole c grid, the distance |E(c) - E*| of the chosen c and of the runner-up, i.e. how clearly
the hedge picks its c. Recomputing the decision from the file must give the file's pred column.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/home/claude/w1111/base/v8')
import hedge_beta as H

for name, f, lam in [('W0 (w_T 0, lambda 0.5)', 'asof/v11/out/probabilities_v11_1111_wt0_lam0.5.csv', 0.5),
                     ('v11 default (w_T 0.5, lambda 0.75)', 'asof/v11/out/probabilities_v11_1111.csv', 0.75),
                     ('v9', 'asof/v9/out/probabilities_v9_1111.csv', 0.75)]:
    P = pd.read_csv(f, dtype={'stock_id': str})
    ex = ((P[['ex_today', 'ex_next1', 'ex_prev1', 'ex_prev2']] == 1).any(axis=1)).values
    sc, be = P.score.values, P.beta.values
    pred, info = H.hedge_day(sc, be, ex, lam)
    s0 = np.where(ex, -1, np.where(sc > 0, 1, -1)); target = s0.mean() * be.mean() + lam * ((s0 - s0.mean()) * (be - be.mean())).mean()
    S = (sc[None, :] + H.CGRID[:, None] * be[None, :] > 0) & ~ex[None, :]
    E = ((2 * S - 1) @ be) / len(sc); d = np.abs(E - target)
    j = int(np.flatnonzero(np.isclose(H.CGRID, info['c']))[0])
    other = np.flatnonzero(np.any(S != S[j], axis=1))
    k = other[np.argmin(d[other])]
    m = np.abs(sc + info['c'] * be)[~ex]
    print(f"{name}: c {info['c']:.4f}, recomputed decisions equal the file: {bool((pred == P.pred.values).all())}, "
          f"share of 1 {pred.mean():.2f}, ex-dividend zeros {int(ex.sum())}")
    print(f"   smallest |score + c*beta| = {m.min():.2e} (2nd {np.sort(m)[1]:.2e}); |E - E*| chosen {d[j]:.4f}, "
          f"best grid point with different decisions {d[k]:.4f} (c {H.CGRID[k]:+.4f})")

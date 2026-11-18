r"""
v11_sensitivity.py -- how much Top50 AR does one unit of family rank IC buy in v9's system? (report only)

Adds a small amount of the true answer to one family's prediction (baseline block 1): p' = p + eps * sd_d(p) * z_d, z_d = the
cross-sectional rank of the day's intraday return scaled to sd 1, so the family's rank IC rises by a controlled amount; then
measures the AR change with everything else of v9 fixed. The slope dAR / dIC is the size of the AR gain that a real feature
improving the family's IC by that much can be expected to give (first order), before any noise.
run (in v11\research): python v11_sensitivity.py > ..\results\v11_sensitivity.txt
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v11lib as X
import v11_metrics as MT

def main():
    c = MT.context(); tr = c['tr']; AR = c['AR']
    for fam in ('lgb', 'mlp'):
        b = X.base_blocks(fam)[0]
        p = X.family_pred(fam, b)
        z = tr.reindex(p.index).groupby(level='date').rank(pct=True)
        z = (z - z.groupby(level='date').transform('mean')) / z.groupby(level='date').transform('std')
        sdp = p.groupby(level='date').transform('std')
        ic0 = X.daily_ic(p, tr); ar0 = AR.ar(**({'L': b} if fam == 'lgb' else {'M': b}))
        print(f'{fam}: baseline block 1 rank IC {ic0.mean():.4f}, AR {ar0.mean():.2f}')
        for eps in (0.01, 0.02, 0.05, 0.1):
            q = p + eps * sdp * z.fillna(0)
            d = {m: (q if fam == 'lgb' else b[m] + eps * b[m].groupby(level='date').transform('std') * z.reindex(b[m].index).fillna(0))
                 for m in X.MEM[fam]}
            if fam == 'lgb':
                d = {'lgb_res': q}
            ic = X.daily_ic(X.family_pred(fam, d), tr); ar = AR.ar(**({'L': d} if fam == 'lgb' else {'M': d}))
            dI = (ic - ic0).mean(); dA = (ar - ar0).mean()
            print(f'   eps {eps:.2f}: family rank IC {dI * 1e3:+.2f}e-3 -> AR {dA:+.3f} bp/day (t {X.nw_t((ar - ar0).values):.1f}); '
                  f'AR per 1e-3 of IC {dA / (dI * 1e3):.3f} bp')

if __name__ == '__main__':
    main()

r"""
v12_posthoc.py -- analyses that were NOT in research\PREREG_v12.md (done after seeing v12_eval.txt; report only):
  1. the bp difference of each selected candidate (and v12) against v11 split into the stock-picking part (xsec) and the
     net-exposure part (timing)
  2. Top50-only rank IC and a Top50 long-short (sign of prediction vs the day's Top50 median x excess return, bp/day) of the
     single models, to see whether Y2's higher IC on all stocks also holds on the Top50
  3. rank IC of the whole stock view p_e (all stocks and Top50) of v11, v11 + Y2 and v12
run (in v12\research, after v12_eval.py): python v12_posthoc.py > ..\results\v12_posthoc.txt
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v12lib as L
import v11b_lib as B
import v11b_train as TB
V = L.V
PER = {'val': ('2022-01-01', '2025-01-01'), 'hold': ('2025-01-01', '2026-06-01'), 'post': ('2026-06-01', '2100-01-01'),
       'all': ('2022-01-01', '2100-01-01')}
sel = lambda s, a, b: s[(s.index >= a) & (s.index < b)]

def main():
    D = pd.read_pickle(os.path.join(L.V12, 'cache', 'v12_daily.pkl'))['daily']
    print('1. bp difference against v11 split into xsec (stock picking) and timing (net exposure), bp/day (NW t)')
    for k in ('v11 + Y2', 'v11 + Y3', 'v11 + Y4', 'v11 + Y5', 'v12'):
        for comp in ('AR', 'xsec', 'timing'):
            dd = (D[k][comp] - D['v11'][comp]).dropna()
            print(f'  {k:9s} {comp:6s} ' + '  '.join(f'{p} {sel(dd, a, b).mean():+.2f} ({V.nw_t(sel(dd, a, b).values):.2f})'
                                                     for p, (a, b) in PER.items()))
    d = B.Data(); c = lambda s: s - s.groupby(level='date').transform('mean')
    M = {'lgb_res': d.LGB, 'X1': d.x1, 'X5': TB.load_member('lgb_rank').reindex(d.idx)}
    for k, n in (('Y2', 'lgb_dec_l2'), ('Y4', 'lgb_dec_pw'), ('Y5', 'lgb_size_pw')):
        M[k] = L.load_member(n).reindex(d.idx)
    def ls(pred):
        X = pd.DataFrame({'p': pred, 'r': d.tr})[d.top50.values].dropna()
        g = X.groupby(level='date')
        s = np.sign(X.p - g.p.transform('median')); e = X.r - g.r.transform('mean')
        return (s * e).groupby(level='date').mean() * 1e4
    rows = []
    for k, v in M.items():
        i50 = d.ic(c(v), top50=True); l = ls(c(v)); r = {'model': k}
        for p, (a, b) in PER.items():
            r[f'IC50 {p}'] = sel(i50, a, b).mean()
        for p, (a, b) in PER.items():
            r[f'LS50 {p}'] = sel(l, a, b).mean()
        rows.append(r)
    print('\n2. single models on the Top50 only: rank IC and long-short (sign vs the Top50 median, bp/day)')
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    sdd = lambda s: s.groupby(level='date').transform('std')
    sc = lambda y: d.LGB.groupby(level='date').transform('mean') + c(y) * sdd(c(d.LGB)) / sdd(c(y))
    view = lambda lg: c((pd.concat(d.mlp, axis=1).mean(axis=1) + pd.concat(lg, axis=1).mean(axis=1)) / 2)
    Y3 = L.load_member('lgb_size_l2').reindex(d.idx)
    PE = {'v11': view([d.LGB, d.x1]), 'v11 + Y2': view([d.LGB, d.x1, sc(M['Y2'])]),
          'v12': view([d.LGB, d.x1] + [sc(y) for y in (M['Y2'], Y3, M['Y4'], M['Y5'])])}
    print('\n3. rank IC of the whole stock view p_e (val / hold / post / all)')
    for k, pe in PE.items():
        i, i50 = d.ic(pe), d.ic(pe, top50=True)
        print(f'  {k:9s} all stocks ' + ' / '.join(f'{sel(i, a, b).mean():.4f}' for a, b in PER.values())
              + '   Top50 ' + ' / '.join(f'{sel(i50, a, b).mean():.4f}' for a, b in PER.values()))

if __name__ == '__main__':
    main()

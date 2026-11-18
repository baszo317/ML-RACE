r"""
v11d_regime.py -- the static-optimised setting (w_T 0, lambda 0.5) also won October: is that a regime that lasts?
(report only; nothing here changes v11)

Question of 2026-10-26 17:04: the coefficients picked on the static period (6/1-9/30) were w_T 0 and lambda 0.4-0.5, and the
October-best setting of v11c_october.py is the same. Test on 2022-01..2026-10 whether "the setting that has been winning for
the last N days keeps winning for the next 19 days" (19 = the private test period), with the walk-forward predictions:
  base   : v9 (w_T 0.5, lambda 0.75) and v11 (+ T2, T3, X1)
  switch : the same with w_T 0, lambda 0.5 (market view off, half of the low-beta tilt hedged)
D_t = AR(switch) - AR(base). For every 19-day window (non-overlapping, 2022-2026), X = mean of D over the N days that end
`gap` days before the window (N = 60, 99, 120, 250; gap 0 and 5: the private test starts 11/30, 5 trading days after the
11/20 decision data), Y = mean of D in the window. A lasting regime means corr(X, Y) > 0 and Y > 0 when X > 0.
run (in v11\research): python v11d_regime.py > ..\results\v11d_regime.txt
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v11c_october as C

SETS = {'v9': (0.5, 0.0, 0, 0, 0.75), 'v9 wT0 lam0.5': (0.0, 0.0, 0, 0, 0.5),
        'v11': (0.5, 0.5, 1, 1, 0.75), 'v11 wT0 lam0.5': (0.0, 0.5, 1, 1, 0.5)}

def main():
    t0 = time.time()
    C.prep()
    AR = {k: C.run(v)[1] for k, v in SETS.items()}
    D0 = C._G['d'].decide(C._G['d'].score([])[0])                    # v9: market m, timing
    A = pd.DataFrame(AR)
    print(f'{len(A)} walk-forward days {A.index.min().date()}..{A.index.max().date()}; {time.time() - t0:.0f}s')
    yr = pd.DatetimeIndex(A.index)
    per = {'2022': yr.year == 2022, '2023': yr.year == 2023, '2024': yr.year == 2024, '2025': yr.year == 2025,
           '2026 Jan-May': (yr >= '2026-01-01') & (yr < '2026-06-01'), 'Jun-Sep': (yr >= '2026-06-01') & (yr < '2026-10-01'),
           'Oct': yr >= '2026-10-01', 'all': yr >= '2022-01-01'}
    rows = []
    for nm, m in per.items():
        r = {'period': nm, 'days': int(m.sum())}
        for k in SETS:
            r[k] = A[k][m].mean()
        r['m (Top50 intraday)'] = D0.m[m].mean(); r['corr(sbar, m) v9'] = np.corrcoef(D0.sbar[m], D0.m[m])[0, 1]
        rows.append(r)
    print('\nTop50 bp/day by period (walk-forward; Jun-Oct 2026 = models frozen at 5/29)')
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    # monthly persistence of the difference
    for base, sw in (('v9', 'v9 wT0 lam0.5'), ('v11', 'v11 wT0 lam0.5')):
        Dd = (A[sw] - A[base])
        M = Dd.groupby(yr.to_period('M')).mean()
        ac1 = M.autocorr(1); ac2 = M.autocorr(2)
        print(f'\n{sw} minus {base}: daily mean {Dd.mean():+.2f} (all), monthly means: {int((M > 0).sum())} of {len(M)} months > 0; '
              f'autocorrelation of the monthly means lag 1 {ac1:+.2f}, lag 2 {ac2:+.2f} (sampling sd about {1 / np.sqrt(len(M)):.2f})')
        print('  last 8 months: ' + ', '.join(f'{p} {v:+.1f}' for p, v in M.tail(8).items()))
        print('  N = lookback days, gap = days between the lookback and the 19-day window; X = lookback mean, Y = next-19-day mean')
        out = []
        n = len(Dd)
        for N in (60, 99, 120, 250):
            for gap in (0, 5):
                xs, ys = [], []
                for e in range(N + gap, n - 18, 19):
                    xs.append(Dd.iloc[e - gap - N:e - gap].mean()); ys.append(Dd.iloc[e:e + 19].mean())
                xs, ys = np.array(xs), np.array(ys)
                rule = np.where(xs > 0, ys, 0.0)                         # switch only after the switch setting has been winning
                out.append({'N': N, 'gap': gap, 'windows': len(xs), 'corr(X, Y)': np.corrcoef(xs, ys)[0, 1],
                            'Y | X>0': ys[xs > 0].mean() if (xs > 0).any() else np.nan, 'n X>0': int((xs > 0).sum()),
                            'Y | X<=0': ys[xs <= 0].mean() if (xs <= 0).any() else np.nan,
                            'rule gain': rule.mean(), 'rule t': rule.mean() / rule.std(ddof=1) * np.sqrt(len(rule)) if rule.std() > 0 else np.nan,
                            'always switch': ys.mean()})
        print(pd.DataFrame(out).round(2).to_string(index=False))
    A.to_pickle(os.path.join(C.L.V11, 'cache', 'v11d_daily.pkl'))
    print(f'\ndone in {time.time() - t0:.0f}s')

if __name__ == '__main__':
    main()

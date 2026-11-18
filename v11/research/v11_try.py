r"""
v11_try.py -- how would YOUR parameters have done? Walk-forward 2022-01..2026-10 (1,164 days) of v11 with the parameters you
give, next to v11's defaults and v9 (report only; uses the research caches, no training, about a minute on the laptop).

  python v11_try.py --w-t 0.25
  python v11_try.py --w-t 0 --lam 0.5 --night-w 0.5 --extra-ones 5 --no-x1
Parameters (same names and defaults as train_and_predict_v11.py):
  --w-t 0.5  --lam 0.75  --night-w 0.5  --extra-ones 5  --no-x1
Periods: val 2022-24, hold 2025-01..2026-05, Jun-Sep 2026 (static), Oct 2026 (15 days), all. Differences use Newey-West t.
Values the research did not try (README_v11 sections 6 and 8) are allowed; the numbers then are the only evidence you have,
and picking the best of many tries on these same days overstates it (README_v9 section 11, README_v11 section 6).
"""
import os, sys, argparse, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v11b_lib as L
V = L.V
PER = {'val 2022-24': ('2022-01-01', '2025-01-01'), 'hold 2025-26/05': ('2025-01-01', '2026-06-01'),
       'Jun-Sep 2026': ('2026-06-01', '2026-10-01'), 'Oct 2026': ('2026-10-01', '2100-01-01'), 'all': ('2022-01-01', '2100-01-01')}

def run(data, adj1, pe, w_t, night_w, extra, lam):
    pb = data.pbar() - night_w * adj1
    s = w_t * (pb.reindex(pe.index.get_level_values('date')).values - 0.5) + pe
    L.EXTRA_ONES = int(extra)
    return data.decide(s, month_start=extra > 0, lam=lam)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--w-t', type=float, default=0.5); ap.add_argument('--lam', type=float, default=0.75)
    ap.add_argument('--night-w', type=float, default=0.5); ap.add_argument('--extra-ones', type=int, default=5)
    ap.add_argument('--no-x1', action='store_true')
    a = ap.parse_args(); t0 = time.time()
    print('loading the walk-forward predictions ...', flush=True)
    data = L.Data()
    pb = data.pbar(); sd = pb.rolling(250, min_periods=20).std().shift(1)
    adj1 = (sd * data.z_night.reindex(pb.index)).fillna(0.0)
    pe = {False: data.stock_view(), True: data.stock_view(x1=True)}
    runs = {'v9': run(data, adj1, pe[False], 0.5, 0.0, 0, 0.75),
            'v11 (defaults)': run(data, adj1, pe[True], 0.5, 0.5, 5, 0.75),
            'yours': run(data, adj1, pe[not a.no_x1], a.w_t, a.night_w, a.extra_ones, a.lam)}
    print(f'\nyour parameters: w_T {a.w_t:g}, lambda {a.lam:g}, T2 night weight {a.night_w:g}, T3 extra 1s {a.extra_ones}, '
          f'X1 {"off" if a.no_x1 else "on"}   (v11 defaults: 0.5, 0.75, 0.5, 5, on)')
    rows = []
    for nm, D in runs.items():
        r = {'setting': nm}
        for k, (s, e) in PER.items():
            m = (D.index >= s) & (D.index < e); r[k] = D.AR[m].mean()
        r19 = D.AR.rolling(19).mean().dropna()
        r.update({'daily sd': D.AR.std(), 'P(19d>0)': (r19 > 0).mean(), 'P(19d>20)': (r19 > 20).mean(), 'timing': D.timing.mean(), 'xsec': D.xsec.mean()})
        rows.append(r)
    print('\nTop50 bp/day, walk-forward (models retrained yearly; Jun-Oct 2026 = models frozen at 5/29)')
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    for ref in ('v11 (defaults)', 'v9'):
        d = (runs['yours'].AR - runs[ref].AR).dropna()
        print(f'\nyours minus {ref} (bp/day, NW t): ' + '; '.join(
            f'{k} {d[(d.index >= s) & (d.index < e)].mean():+.2f} ({V.nw_t(d[(d.index >= s) & (d.index < e)].values):.2f})' for k, (s, e) in PER.items()))
    print(f'\ndone in {time.time() - t0:.0f}s. To write the file for the next day with these parameters: '
          f'python ..\\train_and_predict_v11.py --dates auto --retrain none --w-t {a.w_t:g} --lam {a.lam:g} --night-w {a.night_w:g} '
          f'--extra-ones {a.extra_ones}' + (' --no-x1' if a.no_x1 else ''))

if __name__ == '__main__':
    main()

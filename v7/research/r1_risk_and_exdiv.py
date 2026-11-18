"""r1 (v6): two cheap checks on the existing walk-forward predictions (v2/results_v2/wf, same members and seeds as
v5 s1: mlp_raw_B, mlp_raw_xr x10, tf_raw_B x5, mlp_res_B, mlp_res_xr x10; annual folds 2022..2026/05, post = frozen
at 2026-05-29). Nothing is retrained and no model file is written.
(1) w_T as a risk setting: daily AR 2022-01..2026-10 for w_T = 0..2 and the distribution of the mean over every
    19-day stretch (overlapping), i.e. the length of the private test.
(2) ex-dividend neighbours: Top50 intraday returns on the day before / 1 and 2 days after an ex-dividend day, and the
    pre-registered rule test (force a guess on those stock-days; val paired t >= Holm bar for 6 tests, hold same sign).
run from this folder: python r1_risk_and_exdiv.py > results/r1_risk_and_exdiv.txt"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'v5', 'research'))
import compat
import numpy as np, pandas as pd
from statistics import NormalDist
import rlib as L
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
L._P = L.E.build_panel(L.E.DATA); L._P['date'] = L._P['date'].astype('datetime64[ns]')     # no panel cache on disk
P = L._P
S10, S5 = range(1, 11), range(1, 6)
DEF = [('mlp_raw_B', S10), ('mlp_raw_xr', S10), ('tf_raw_B', S5), ('mlp_res_B', S10), ('mlp_res_xr', S10)]
pr = {n: L.load_wf(n, s) for n, s in DEF}
mem = [L.member_def(n) for n, _ in DEF]
ens = lambda w: L.E.combine({m['name']: pr[m['name']] for m in mem}, mem, w_t=w)

print('(1) w_T as a risk setting: daily AR 2022-01..2026-10 (bp/day) and the mean over every 19-day stretch')
rows, keep = [], {}
for w in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
    e = ens(w); D = L.daily(e[['score']]); keep[w] = (e, D)
    r19 = D.AR.rolling(19).mean().dropna()
    rows.append({'w_T': w, 'mean': D.AR.mean(), 'sd': D.AR.std(), 'timing': D.timing.mean(), 'xsec': D.xsec.mean(),
                 'sd_19d': r19.std(), 'P(19d>0)': (r19 > 0).mean(), 'P(19d>20)': (r19 > 20).mean(),
                 'P(19d>25)': (r19 > 25).mean(), 'P(19d>30)': (r19 > 30).mean(), 'q05_19d': r19.quantile(.05), 'q95_19d': r19.quantile(.95)})
print(pd.DataFrame(rows).round(3).to_string(index=False))
print(f'days: {len(keep[0.5][1])}; 19-day stretches: {len(keep[0.5][1]) - 18} (overlapping, so neighbouring stretches share days)')

print('\n(2a) Top50 stock-days around ex-dividend dates (r = open->close, resid = r - same-day mean of the 100 stocks, bp)')
Q = P[P.top50 & P.target_r.notna()].copy()
Q['per'] = pd.cut(Q.date, pd.to_datetime(['2000-01-01', '2022-01-01', '2025-01-01', '2026-06-01', '2100-01-01']), right=False,
                  labels=['pre 2020-21', 'val 2022-24', 'hold 2025-26/05', 'post 2026/06-'])
rows = []
for f in ['ex_next1', 'ex_today', 'ex_prev1', 'ex_prev2']:
    for p in Q.per.cat.categories:
        z = Q[(Q[f] == 1) & (Q.per == p)]
        rows.append({'flag': f, 'period': p, 'n': len(z), 'r': z.target_r.mean() * 1e4, 'resid': z.resid_r.mean() * 1e4,
                     'P(y=1)': (z.target_r > 0).mean(), 't_resid': z.resid_r.mean() / z.resid_r.std() * np.sqrt(len(z))})
print(pd.DataFrame(rows).round(2).to_string(index=False))
print('ex_next1 = the day before an ex-dividend day; ex_prev1 / ex_prev2 = 1 / 2 trading days after it')

print('\n(2b) force a guess on neighbour stock-days vs the v2/v4 rule (w_T 0.5 + ex-dividend day -> 0), paired, bp/day')
e = keep[0.5][0]
F = L.frame(e[['score']]); F = F[F.top50].merge(P[['date', 'stock_id', 'ex_next1', 'ex_prev1', 'ex_prev2']], on=['date', 'stock_id'], how='left')
F['s0'] = 2 * np.where(F.ex_today == 1, 0, (F.score > 0).astype(int)) - 1
D0 = (F.s0 * F.target_r).groupby(F.date).mean() * 1e4
prd = L.periods(D0.index); rows = []
for f in ['ex_next1', 'ex_prev1', 'ex_prev2']:
    for g in (0, 1):
        s = np.where(F[f] == 1, 2 * g - 1, F.s0)
        d = (s * F.target_r).groupby(F.date).mean() * 1e4 - D0
        r = {'rule': f'{f} -> {g}', 'stock-days': int((F[f] == 1).sum()), 'changed': int(((F[f] == 1) & (F.s0 != 2 * g - 1)).sum())}
        for k in ('val', 'hold', 'post'):
            r['d_' + k] = d[prd[k]].mean(); r['t_' + k] = L.plain_t(d[prd[k]])
        rows.append(r)
print(pd.DataFrame(rows).round(2).to_string(index=False))
print(f'Holm bar for the best of 6 rules: z(1 - 0.05/6) = {NormalDist().inv_cdf(1 - 0.05 / 6):.2f}; hold must have the same sign.')

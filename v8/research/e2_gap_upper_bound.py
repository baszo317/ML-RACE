"""e2 (EXPLORATORY, upper bound only): how much would knowing today's opening gap help the MARKET view?
The real gap is known only at 09:00 (after the upload deadline), so this is a ceiling, like README_v7 section 2 did for the
stock view.  A near-perfect proxy would be the TX day-session futures quote between 08:45 and 09:00 (the futures open 15 minutes
before the stocks), but its history is not freely available, so it cannot be backtested here.
run: python e2_gap_upper_bound.py > ../results/e2_gap_upper_bound.txt"""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mlib as L
pd.set_option('display.width', 220)
info = json.load(open(os.path.join(L.ROOT, 'data', 'info.json'))); top50 = info['Top50@Y']
R = []
for s in top50:
    d = pd.read_csv(os.path.join(L.ROOT, 'data', 'market', f'{s}.csv'), parse_dates=['date']).sort_values('date')
    d = d[(d['etl:adj_open'] > 0) & (d['etl:adj_close'] > 0)]
    R.append(pd.DataFrame({'date': d.date, 'gap': np.log(d['etl:adj_open'] / d['etl:adj_close'].shift()), 'r': d['etl:adj_close'] / d['etl:adj_open'] - 1}))
R = pd.concat(R)
D = R.groupby('date').agg(gap=('gap', 'mean'), m=('r', 'mean'))
G = pd.read_pickle(os.path.join(L.CACHE, 'daily_market.pkl'))
D = D.join(G[['night_ret', 'us_gspc_ret1']], how='left')
D = D[D.index >= '2020-03-02']
pr = L.periods(D.index)
print('corr with the Top50 intraday return m_t, by period:')
for c in ['gap', 'night_ret', 'us_gspc_ret1']:
    print(f'  {c:13s}: ' + ', '.join(f'{k} {L.corr(D[c][v], D.m[v]):+.3f}' for k, v in pr.items()))
print('how well last night predicts the opening gap: corr(night_ret, gap) ' + ', '.join(f'{k} {L.corr(D.night_ret[v], D.gap[v]):.3f}' for k, v in pr.items()))
# unexpected part of the gap (not explained by the night session) vs m
k = D[['gap', 'night_ret', 'm']].notna().all(axis=1)
b = np.polyfit(D.night_ret[k], D.gap[k], 1); D['gap_surprise'] = D.gap - np.polyval(b, D.night_ret)
print('  gap surprise (gap - fit on night_ret): ' + ', '.join(f'{k_} {L.corr(D.gap_surprise[v], D.m[v]):+.3f}' for k_, v in pr.items()))
X = pd.read_pickle(os.path.join(L.CACHE, 'daily_view.pkl'))
J = X[['pbar', 'sbar']].join(D, how='inner')
for lab, sig in (('pbar (now)', J.pbar), ('-gap (ceiling)', -J.gap), ('-night_ret', -J.night_ret)):
    print(f'  corr(signal, m) 2022-26: {lab:15s} {L.corr(sig, J.m):+.3f}')
sd_s = J.sbar.std(); sd_m = J.m.std() * 1e4
print(f'\nscale: at the current bet size (sd of net exposure {sd_s:.3f}) each +0.1 of corr(exposure, m) is worth about '
      f'{0.1 * sd_s * sd_m:.1f} bp/day (cov = corr x sd(sbar) x sd(m), sd(m) = {sd_m:.0f} bp)')

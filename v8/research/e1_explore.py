"""e1 (EXPLORATORY, not part of the pre-registered test; nothing here is adopted on its own):
(1) risk frontier of the two market knobs: w_T (explicit market view) x lambda_B (share of the stock view's beta tilt kept;
    1 = as now, 0 = beta-neutral stock view).  Reported as a risk setting, like README_v7 section 3.
(2) does the market view's recent performance predict its next month?  (would a drift-aware on/off switch help?)
(3) timing and the hidden beta bet by ex-ante volatility regime (would scaling the market bet by volatility help?)
(4) October 2026 (the live period so far), day by day
run: python e1_explore.py > ../results/e1_explore.txt"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mlib as L
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
P = L.panel(); V = L.views(L.members())
Q = pd.read_pickle(os.path.join(L.CACHE, 'beta_panel.pkl'))
F = V[['pbar', 'p_e_c']].reset_index().merge(Q[['date', 'stock_id', 'beta']], on=['date', 'stock_id'], how='left')
F['beta'] = F.beta.clip(0.2, 2.5); F['beta'] = F.beta.fillna(F.groupby('date').beta.transform('median')).fillna(1.0)
x = F.beta - F.groupby('date').beta.transform('mean')
F['b_t'] = (x * F.p_e_c).groupby(F.date).transform('sum') / (x * x).groupby(F.date).transform('sum')
F['tilt'] = F.b_t * x
F = F.set_index(['date', 'stock_id'])

def run(score):
    D = L.daily(score)
    q = score.rename('score').reset_index().merge(Q[['date', 'stock_id', 'top50', 'target_r', 'ex_today', 'beta']], on=['date', 'stock_id'])
    q = q[q.top50 & q.target_r.notna() & q.beta.notna()].copy()
    pred = np.where(q.ex_today == 1, 0, (q.score > 0).astype(int)); q['s'] = 2 * pred - 1
    g = q.groupby('date'); q['sb'] = g.s.transform('mean'); q['bb'] = g.beta.transform('mean')
    D['B'] = ((q.s - q.sb) * (q.beta - q.bb)).groupby(q.date).mean().reindex(D.index)
    D['beta_bet'] = D.B * D.m
    D['rest'] = D.xsec - D.beta_bet
    return D

print('(1) risk frontier, walk-forward 2022-01..2026-10 (1,164 days), bp/day; P19 = share of 19-day windows with mean > 0 / > 20 / > 30')
rows = []
for lam in (1.0, 0.5, 0.0):
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        D = run(w * (F.pbar - 0.5) + F.p_e_c - (1 - lam) * F.tilt)
        r19 = D.AR.rolling(19).mean().dropna(); pr = L.periods(D.index)
        rows.append({'lambda_B': lam, 'w_T': w, 'val': D.AR[pr['val']].mean(), 'hold': D.AR[pr['hold']].mean(), 'post': D.AR[pr['post']].mean(),
                     'all': D.AR.mean(), 'sd': D.AR.std(), 'timing': D.timing.mean(), 'beta_bet': D.beta_bet.mean(), 'rest': D.rest.mean(),
                     'mean B': D.B.mean(), 'P19>0': (r19 > 0).mean(), 'P19>20': (r19 > 20).mean(), 'P19>30': (r19 > 30).mean(),
                     'Oct26': D.AR[D.index >= '2026-10-01'].mean()})
print(pd.DataFrame(rows).round(3).to_string(index=False))

base = run(0.5 * (F.pbar - 0.5) + F.p_e_c)
mk = base.m
print('\n(2) persistence: does last month\'s market-view P&L predict this month\'s?  (v4 rule, monthly sums of timing + beta bet)')
mon = (base[['timing', 'beta_bet']]).groupby(base.index.to_period('M')).mean()
mon['mkt_part'] = mon.timing + mon.beta_bet
for c in ['timing', 'beta_bet', 'mkt_part']:
    print(f'   {c:9s}: lag-1 autocorrelation of monthly means {mon[c].autocorr(1):+.3f} (n {len(mon)}); months > 0: {(mon[c] > 0).mean():.2f}')
# switch rule: market view on only if the trailing 60-day timing is > 0
V0 = run(0.0 * (F.pbar - 0.5) + F.p_e_c)
tr60 = base.timing.rolling(60).sum().shift(1)
sw = np.where(tr60 > 0, base.AR, V0.AR.reindex(base.index))
sw = pd.Series(sw, index=base.index)
print(f'   switch "market view on only if the last 60 days of timing > 0": all {sw.mean():.2f} vs always on {base.AR.mean():.2f} vs off {V0.AR.mean():.2f} '
      f'(paired t vs always on {L.plain_t(sw - base.AR):.2f})')

print('\n(3) by ex-ante volatility (terciles of the trailing 20-day mean |m| = mkt_absr_20, known before 09:00), v4 rule, 2022-01..2026-10')
G = pd.read_pickle(os.path.join(L.CACHE, 'daily_market.pkl'))
J = base.join(G[['mkt_absr_20', 'us_vix', 'night_range']], how='left')
for c in ['mkt_absr_20', 'us_vix']:
    J['q'] = pd.qcut(J[c].rank(method='first'), 3, labels=['low', 'mid', 'high'])
    t = J.groupby('q', observed=True).agg(days=('AR', 'size'), AR=('AR', 'mean'), timing=('timing', 'mean'), beta_bet=('beta_bet', 'mean'),
                                          rest=('rest', 'mean'), sd_m=('m', 'std'), sd_AR=('AR', 'std'))
    t['corr(sbar,m)'] = J.groupby('q', observed=True)[['sbar', 'm']].apply(lambda g: L.corr(g.sbar, g.m))
    print(f'  by {c}:'); print(t.round(2).to_string())

print('\n(4) October 2026 so far (v4 rule with the frozen 5/29 models), bp:')
o = base[base.index >= '2026-10-01'][['m', 'sbar', 'B', 'AR', 'timing', 'beta_bet', 'rest']]
print(o.round(2).to_string())
print('  mean: ' + ', '.join(f'{c} {o[c].mean():.2f}' for c in o.columns))

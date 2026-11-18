"""s14: reference data for monitor_live.py and how often its flags fire by chance.
(1) ref_daily_components.csv: daily AR, timing, between, within, xsec of the v4 rule (walk-forward 2022-01..2026-10,
    wf_v4_frame.pkl from s4). monitor_live.py compares the live n-day average with all historical n-day averages
    (same n), i.e. it reports an empirical percentile.
(2) false alarms: replay the monitor on every 19-day stretch of the history (the length of the private test) and count
    how often a part is flagged at least once although nothing is broken.
      first version : from day 10, average of the last min(20, n) days below the 5 % quantile of 20-day averages
      v5 (adopted)  : from day 5, percentile of the n-day average among historical n-day averages below 1 %"""
import os, numpy as np, pandas as pd, rlib as L
X = pd.read_pickle(os.path.join(L.CACHE, 'wf_v4_frame.pkl'))
P = L.panel()
F = X.merge(P[['date', 'stock_id', 'group']], on=['date', 'stock_id'])
F = F[F.top50 & F.target_r.notna()].copy()
pred = np.where(F.ex_today == 1, 0, (F.score > 0).astype(int))
F['s'] = 2 * pred - 1
F['sb'] = F.groupby('date').s.transform('mean'); F['rb'] = F.groupby('date').target_r.transform('mean')
F['sg'] = F.groupby(['date', 'group']).s.transform('mean'); F['rg'] = F.groupby(['date', 'group']).target_r.transform('mean')
F['AR'] = F.s * F.target_r; F['timing'] = F.sb * F.rb
F['between'] = (F.sg - F.sb) * (F.rg - F.rb); F['within'] = (F.s - F.sg) * (F.target_r - F.rg)
D = F.groupby('date')[['AR', 'timing', 'between', 'within']].mean() * 1e4
D['xsec'] = D.AR - D.timing
K = ['AR', 'timing', 'between', 'within', 'xsec']
HERE = os.path.dirname(os.path.abspath(__file__))
D[K].round(4).to_csv(os.path.join(HERE, 'ref_daily_components.csv'))
print(f'{len(D)} days -> ref_daily_components.csv; average (bp/day):', ', '.join(f'{k} {D[k].mean():.2f}' for k in K))
R20 = D[K].rolling(20).mean().dropna()
Q = pd.DataFrame({f'q{q:02d}': R20.quantile(q / 100) for q in (1, 5, 50, 95)}); Q['share_negative'] = (R20 < 0).mean()
print('\nquantiles of 20-day averages (bp/day):'); print(Q.round(2).to_string())

V = D[K].values; n = len(V)
roll = {j: np.sort(pd.DataFrame(V).rolling(j).mean().dropna().values, axis=0) for j in range(1, 21)}
def pct(j, x):
    """empirical percentile of an n-day average x (vector over parts) among historical n-day averages"""
    R = roll[j]
    return np.array([np.searchsorted(R[:, i], x[i], side='right') / len(R) for i in range(len(x))])
q05_20 = R20.quantile(0.05).values
res = {'first version (day >= 10, 5 % of 20-day averages)': [], 'v5 (day >= 5, n-matched percentile < 1 %)': []}
for a in range(0, n - 19 + 1):
    W = V[a:a + 19]; h1 = np.zeros(len(K), bool); h2 = np.zeros(len(K), bool)
    for j in range(5, 20):
        m = W[:j].mean(axis=0)
        if j >= 10:
            h1 |= m < q05_20
        h2 |= pct(j, m) < 0.01
    res['first version (day >= 10, 5 % of 20-day averages)'].append(h1)
    res['v5 (day >= 5, n-matched percentile < 1 %)'].append(h2)
print(f'\nshare of 19-day stretches ({n - 18}, overlapping) in which the monitor flags a part at least once by chance:')
for lab, hits in res.items():
    H = np.array(hits)
    print(f'  {lab}: ' + ', '.join(f'{k} {H[:, i].mean():.3f}' for i, k in enumerate(K)) + f'; any part {H.any(axis=1).mean():.3f}')

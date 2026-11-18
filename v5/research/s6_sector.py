"""s6: does the stock view bet on sectors? xsec_t = mean_i (s_i - s_bar)(r_i - r_bar) over the Top50 splits exactly into
between-sector  sum_g w_g (s_g - s_bar)(r_g - r_bar)   and   within-sector  mean_i (s_i - s_g)(r_i - r_g)
(cross terms vanish inside each sector). Sector = comp_features.group: 0 fin, 1 semi, 2 electronics, 3 traditional."""
import os, numpy as np, pandas as pd, rlib as L
X = pd.read_pickle(os.path.join(L.CACHE, 'wf_v4_frame.pkl'))
P = L.panel()
X = X.merge(P[['date', 'stock_id', 'group']], on=['date', 'stock_id'])
def decompose(F, score_col='score'):
    F = F[F.top50].copy()
    pred = (F[score_col] > 0).astype(int); pred = np.where(F.ex_today == 1, 0, pred)
    F['s'] = 2 * pred - 1; r = F.target_r
    F['sb'] = F.groupby('date').s.transform('mean'); F['rb'] = F.groupby('date').target_r.transform('mean')
    F['sg'] = F.groupby(['date', 'group']).s.transform('mean'); F['rg'] = F.groupby(['date', 'group']).target_r.transform('mean')
    F['between'] = (F.sg - F.sb) * (F.rg - F.rb); F['within'] = (F.s - F.sg) * (r - F.rg)
    d = F.groupby('date')[['between', 'within']].mean() * 1e4
    tilt = F.assign(t=F.sg - F.sb).groupby(['date', 'group']).t.first().unstack('group')
    return d, tilt
d, tilt = decompose(X)
d['year'] = np.where(d.index >= L.POST_START, 'post', d.index.year.astype(str))
out = d.groupby('year').mean()
out.loc['all'] = d[['between', 'within']].mean()
t = lambda s: s.mean() / s.std() * np.sqrt(len(s))
print('xsec split (bp/day): between-sector vs within-sector')
print(out.round(2).to_string())
print('t(between, all) = %.2f, t(within, all) = %.2f' % (t(d.between), t(d.within)))
tilt['year'] = np.where(tilt.index >= L.POST_START, 'post', tilt.index.year.astype(str))
print('\naverage net tilt of the predictions by sector (sector mean of 2d-1 minus Top50 mean), 0 fin 1 semi 2 elec 3 trad:')
print(tilt.groupby('year').mean().round(3).to_string())
# sector-neutral stock view as a diagnostic: p_e demeaned within sector (instead of within the day)
X2 = X.copy()
X2['p_e_cs'] = X2.p_e - X2.groupby(['date', 'group']).p_e.transform('mean')
X2['score'] = 0.5 * (X2.pbar - 0.5) + X2.p_e_cs
base = L.daily(X.set_index(['date', 'stock_id'])[['score']])
D2 = L.daily(X2.set_index(['date', 'stock_id'])[['score']])
r = L.summary_row(D2, base, 'diag: sector-neutral stock view')
print('\n', pd.DataFrame([L.summary_row(base, None, 'v4'), r]).round(2).to_string(index=False))
# how often does each part lose over a 19-day stretch (the length of the private test)?
from scipy.stats import norm
print('\nrisk over 19 trading days (2022-01..2026-10):')
for k in ('between', 'within'):
    s = d[k]; roll = s.rolling(19).mean().dropna()
    print(f'  {k:8s} daily sd {s.std():5.1f}; share of consecutive 19-day windows with mean < 0: {(roll < 0).mean():.3f}; '
          f'normal approx. for 19 random days: {norm.cdf(-s.mean() * np.sqrt(19) / s.std()):.3f}')

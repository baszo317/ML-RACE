"""d2: is there a hidden market bet inside the stock view?
On a big market day, stocks with high beta move more than the average, so a long/short book that is long low-beta and short
high-beta stocks loses on up days even when the net exposure sbar is 0.  Decomposition per day (Top50, s_i = 2 d_i - 1):
    AR = sbar*m  +  mean_i (s_i - sbar)(r_i - m)
       = timing  +  [B_t * m]  +  rest,      B_t = mean_i (s_i - sbar)(beta_i - mean beta)   (beta tilt of the book)
beta_i = rolling 120-day beta of the stock's intraday return on the Top100 mean intraday return, using days < t only.
run: python d2_hidden_beta.py > ../results/d2_hidden_beta.txt"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mlib as L
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)

P = L.panel(); M = L.members(); V = L.views(M)
mk = L.market_daily(P)
Q = P[['date', 'stock_id', 'top50', 'target_r', 'ex_today', 'group']].merge(mk[['m100']], left_on='date', right_index=True)
Q = Q.sort_values(['stock_id', 'date'])
def _beta(g):
    x, y = g.m100, g.target_r
    cv = (x * y).rolling(120, min_periods=60).mean() - x.rolling(120, min_periods=60).mean() * y.rolling(120, min_periods=60).mean()
    vr = x.rolling(120, min_periods=60).var(ddof=0)
    return (cv / vr).shift(1)
Q['beta'] = Q.groupby('stock_id', group_keys=False)[['m100', 'target_r']].apply(_beta)
Q.to_pickle(os.path.join(L.CACHE, 'beta_panel.pkl'))

def decompose(score, label):
    q = score.rename('score').reset_index().merge(Q, on=['date', 'stock_id'])
    q = q[q.target_r.notna() & q.top50 & q.beta.notna()].copy()
    pred = np.where(q.ex_today == 1, 0, (q.score > 0).astype(int)); q['s'] = 2 * pred - 1
    g = q.groupby('date')
    q['sb'] = g.s.transform('mean'); q['mb'] = g.target_r.transform('mean'); q['bb'] = g.beta.transform('mean')
    q['AR'] = q.s * q.target_r; q['tim'] = q.sb * q.mb; q['B'] = (q.s - q.sb) * (q.beta - q.bb)
    d = q.groupby('date').agg(AR=('AR', 'mean'), timing=('tim', 'mean'), B=('B', 'mean'), m=('mb', 'first'), sbar=('sb', 'first'))
    d['beta_bet'] = d.B * d.m; d['xsec'] = d.AR - d.timing; d['rest'] = d.xsec - d.beta_bet
    for c in ['AR', 'timing', 'beta_bet', 'xsec', 'rest', 'm']:
        d[c] *= 1e4
    d['eff_exposure'] = d.sbar + d.B          # mean_i s_i * beta_i / mean beta  ~  sbar + B (for mean beta ~ 1)
    pr = L.periods(d.index)
    rows = []
    for k in ('val', 'hold', 'post', 'all22'):
        x = d[pr[k]]
        rows.append({'model': label, 'period': k, 'days': len(x), 'AR': x.AR.mean(), 'timing (sbar*m)': x.timing.mean(),
                     'beta bet (B*m)': x.beta_bet.mean(), 'rest of xsec': x.rest.mean(), 'mean B': x.B.mean(), 'sd B': x.B.std(),
                     'corr(B,m)': L.corr(x.B, x.m), 'corr(sbar,m)': L.corr(x.sbar, x.m), 'corr(eff,m)': L.corr(x.eff_exposure, x.m),
                     'corr(xsec,m)': L.corr(x.xsec, x.m), 'sd beta_bet': x.beta_bet.std(), 'sd timing': x.timing.std(), 'sd rest': x.rest.std()})
    return d, pd.DataFrame(rows)

d4, T4 = decompose(0.5 * (V.pbar - 0.5) + V.p_e_c, 'v4 rule')
d0, T0 = decompose(V.p_e_c, 'stock view only (w_T=0)')
print('beta = rolling 120-day intraday beta on the Top100 mean, days before t; bp/day')
print(pd.concat([T4, T0]).round(3).to_string(index=False))

# which stocks does the stock view like?  average beta of the long side minus the short side, and sector shares
q = V.p_e_c.rename('pec').reset_index().merge(Q, on=['date', 'stock_id'])
q = q[q.top50 & q.beta.notna() & q.target_r.notna()]
q['long'] = q.pec > 0
by = q.groupby([q.date.dt.year, 'long']).beta.mean().unstack()
print('\naverage beta of stocks the stock view puts on the long side (True) vs the short side (False), by year:')
print(by.round(3).to_string())
sec = q.groupby(['group', 'long']).size().unstack(); sec = sec.div(sec.sum(axis=1), axis=0)
print('\nshare of stock-days on the long side by sector group (0 fin, 1 semi, 2 electronics, 3 traditional):')
print(sec.round(3).to_string())
print(f'\ncorr(p_e_c, beta) over stock-days: {L.corr(q.pec, q.beta):.3f};  by period: ' +
      ', '.join(f'{k} {L.corr(q.pec[m], q.beta[m]):.3f}' for k, m in L.periods(pd.DatetimeIndex(q.date)).items() if m.sum() > 100))
# big days
x = d4[d4.index >= L.POST].copy(); x['abs'] = x.m.abs()
print('\n2026-06..10, 12 largest |m| days (bp): m, sbar, B (beta tilt), timing, beta_bet, rest')
print(x.sort_values('abs', ascending=False).head(12)[['m', 'sbar', 'B', 'timing', 'beta_bet', 'rest']].round(3).to_string())
d4.to_pickle(os.path.join(L.CACHE, 'decomp_v4.pkl'))

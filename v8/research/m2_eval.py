"""m2: the pre-registered test (PREREG_m.md): v4 rule vs T1 (beta-consistent), T2 (ridge market signal), T3 (pbar + ridge),
T4 (T3 + beta-consistent).  Walk-forward member predictions from x0, ridge forecasts from m1 ('all' version, as registered).
run: python m2_eval.py > ../results/m2_eval.txt"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mlib as L
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)

P = L.panel(); V = L.views(L.members())
Bp = pd.read_pickle(os.path.join(L.CACHE, 'beta_panel.pkl'))[['date', 'stock_id', 'beta']]
mu = pd.read_pickle(os.path.join(L.CACHE, 'mu_ridge.pkl'))

# ---------------------------------------------------------------- inputs per (date, stock)
F = V[['pbar', 'p_e', 'p_e_c']].reset_index().merge(Bp, on=['date', 'stock_id'], how='left')
F['beta'] = F.beta.clip(0.2, 2.5)
F['beta'] = F.beta.fillna(F.groupby('date').beta.transform('median')).fillna(1.0)
F['bbar'] = F.groupby('date').beta.transform('mean')
def _slope(g):
    x = g.beta - g.beta.mean(); return (x * g.p_e_c).sum() / (x * x).sum() if (x * x).sum() > 0 else 0.0
bt = F.groupby('date')[['beta', 'p_e_c']].apply(_slope).rename('b_t')
F = F.merge(bt, left_on='date', right_index=True)
F['p_perp'] = F.p_e_c - F.b_t * (F.beta - F.bbar)

# ---------------------------------------------------------------- daily market signals, past-only scaling
d = F.groupby('date').pbar.first().to_frame()
d = d.join(mu[['mu_all']], how='left')
def past_sd(s, min_n=20):
    """sd over strictly earlier days (expanding); the first min_n days use the sd of the first min_n days"""
    v = s.expanding(min_periods=min_n).std().shift(1)
    return v.fillna(s.iloc[:min_n].std())
d['sd_pbar'] = past_sd(d.pbar)
# mu's own scale: its OOS history starts 2021-03, so use all earlier OOS forecasts (from m1) for the expanding sd
mu_full = mu.mu_all
sd_mu_full = mu_full.expanding(min_periods=20).std().shift(1)
d['sd_mu'] = sd_mu_full.reindex(d.index)
d['z_pbar'] = (d.pbar - 0.5) / d.sd_pbar
d['z_mu'] = d.mu_all / d.sd_mu
S = d.z_pbar + d.z_mu
S_full = S                                                     # S exists only from 2022 (pbar OOS starts 2022-01)
d['sd_S'] = past_sd(S_full)
d['term_v4'] = 0.5 * (d.pbar - 0.5)
d['term_T2'] = 0.5 * d.sd_pbar * d.z_mu
d['term_T3'] = 0.5 * d.sd_pbar * S / d.sd_S
F = F.merge(d[['term_v4', 'term_T2', 'term_T3']], left_on='date', right_index=True)
F = F.set_index(['date', 'stock_id'])

configs = {
    'v4 rule (control)': F.term_v4 + F.p_e_c,
    'T1 beta-consistent (pbar)': F.term_v4 * F.beta / F.bbar + F.p_perp,
    'T2 ridge market signal': F.term_T2 + F.p_e_c,
    'T3 pbar + ridge': F.term_T3 + F.p_e_c,
    'T4 T3 + beta-consistent': F.term_T3 * F.beta / F.bbar + F.p_perp,
}
Q = pd.read_pickle(os.path.join(L.CACHE, 'beta_panel.pkl'))
def run(score):
    D = L.daily(score)
    q = score.rename('score').reset_index().merge(Q[['date', 'stock_id', 'top50', 'target_r', 'ex_today', 'beta']], on=['date', 'stock_id'])
    q = q[q.top50 & q.target_r.notna() & q.beta.notna()].copy()
    pred = np.where(q.ex_today == 1, 0, (q.score > 0).astype(int)); q['s'] = 2 * pred - 1
    g = q.groupby('date'); q['sb'] = g.s.transform('mean'); q['bb'] = g.beta.transform('mean')
    q['B'] = (q.s - q.sb) * (q.beta - q.bb)
    D['B'] = q.groupby('date').B.mean().reindex(D.index)
    D['beta_bet'] = D.B * D.m
    return D
R = {k: run(v) for k, v in configs.items()}
base = R['v4 rule (control)']
rows = []
for k, D in R.items():
    r = L.table(D, None if k.startswith('v4') else base, k)
    pr = L.periods(D.index)
    r['beta_bet'] = D.beta_bet[pr['all22']].mean(); r['mean B'] = D.B[pr['all22']].mean()
    r['corr(sbar,m)'] = L.corr(D.sbar[pr['all22']], D.m[pr['all22']])
    r['P19>0'] = (D.AR[pr['all22']].rolling(19).mean().dropna() > 0).mean()
    rows.append(r)
T = pd.DataFrame(rows)
cols = ['config', 'val', 'hold', 'post', 'all22', 'sd', 'timing', 'xsec', 'beta_bet', 'mean B', 'corr(sbar,m)', 'P19>0',
        'd_val', 't_val', 'd_hold', 't_hold', 'd_post', 't_post']
print('Top50, bp/day; walk-forward 2022-01..2026-10 (post = models frozen at 2026-05-29); d_/t_ = paired difference to the v4 rule')
print(T[cols].round(3).to_string(index=False))

# pre-registered decision with Holm
tr = T[~T.config.str.startswith('v4')].copy().sort_values('t_val', ascending=False).reset_index(drop=True)
thr = [2.24, 2.13, 1.96, 1.645]
print('\nHolm (val t in decreasing order; stop at the first failure) and hold > 0:')
alive = True
for i, r in tr.iterrows():
    ok = alive and r.t_val >= thr[i] and r.d_hold > 0
    alive = alive and r.t_val >= thr[i]
    print(f'  {r.config:28s} val t {r.t_val:+.2f} (needs {thr[i]:.2f}), hold diff {r.d_hold:+.2f} -> {"ADOPT" if ok else "do not adopt"}')

for lab, a, b in (('static 2026-06-01..09-30', '2026-06-01', '2026-09-30'), ('live 2026-10-01..10-22', '2026-10-01', '2026-10-22')):
    print(f'\nfor information, {lab}:')
    for k, D in R.items():
        m = (D.index >= a) & (D.index <= b)
        dd = (D.AR - base.AR.reindex(D.index))[m]
        print(f'  {k:28s} AR {D.AR[m].mean():7.2f}  timing {D.timing[m].mean():6.2f}  beta_bet {D.beta_bet[m].mean():6.2f}  diff {dd.mean():+6.2f} (t {L.plain_t(dd):.2f}, {m.sum()} days)')
pd.to_pickle(R, os.path.join(L.CACHE, 'm2_daily.pkl'))

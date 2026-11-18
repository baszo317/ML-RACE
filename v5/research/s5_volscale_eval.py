"""s5: v4 rule vs the bp rule (pre-registered: k = k0 = 0.5 * median(eps)/median(a) from 2020-03..2021-12 only,
so the average weight equals w_T = 0.5 and only the SHAPE changes). Diagnostics (not used for adoption): partial versions."""
import os, numpy as np, pandas as pd, rlib as L
X = pd.read_pickle(os.path.join(L.CACHE, 'wf_v4_frame.pkl'))
P = L.panel().sort_values(['stock_id', 'date']).copy()
cl = lambda s: s.abs().clip(upper=0.05)
P['a_hat'] = P.groupby('stock_id').target_r.transform(lambda s: cl(s).rolling(60, min_periods=20).mean().shift(1))
P['e_hat'] = P.groupby('stock_id').resid_r.transform(lambda s: cl(s).rolling(60, min_periods=20).mean().shift(1))
pre = P[(P.date >= '2020-03-02') & (P.date < '2022-01-01')]
med_a, med_e = pre.a_hat.median(), pre.e_hat.median()
k0 = 0.5 * med_e / med_a
print(f'median a_hat {med_a*1e4:.1f} bp, median eps_hat {med_e*1e4:.1f} bp (2020-03..2021-12) -> k0 = {k0:.3f}')
for y in range(2022, 2027):
    q = P[P.date.dt.year == y]
    print(f'  {y}: median a {q.a_hat.median()*1e4:.1f}, eps {q.e_hat.median()*1e4:.1f}, ratio {q.e_hat.median()/q.a_hat.median():.3f}')

def bp_rule(F, k, mode='full', gamma=1.0):
    F = F.copy()
    a = F.a_hat.fillna(F.groupby('date').a_hat.transform('median')).fillna(med_a)
    e = F.e_hat.fillna(F.groupby('date').e_hat.transform('median')).fillna(med_e)
    if mode in ('full', 'xsec_only'):
        xv = 2 * F.p_e_c * e ** gamma
    else:
        xv = 2 * F.p_e_c * med_e ** gamma
    F['X'] = xv - xv.groupby(F.date).transform('mean')
    if mode in ('full', 'timing_only'):
        tv = (2 * F.p_raw - 1) * a ** gamma
    else:
        tv = (2 * F.p_raw - 1) * med_a ** gamma
    F['T'] = tv.groupby(F.date).transform('mean')
    F['score'] = k * F['T'] + F['X']
    return F.set_index(['date', 'stock_id'])[['score']]

def evaluate(F, label_prefix, static=False):
    base = L.daily(F.set_index(['date', 'stock_id'])[['score']])
    rows = [L.summary_row(base, None, label_prefix + 'v4 rule (w_T = 0.5)')]
    for lab, kw in [('bp rule, k = k0 (pre-registered)', dict(k=k0)),
                    ('  diag: bp rule, stock view only', dict(k=k0, mode='xsec_only')),
                    ('  diag: bp rule, market view only', dict(k=k0, mode='timing_only')),
                    ('  diag: bp rule, gamma = 0.63 (fitted on val+hold)', dict(k=0.5 * med_e ** .63 / med_a ** .63, gamma=0.63))]:
        D = L.daily(bp_rule(F, **kw))
        rows.append(L.summary_row(D, base, label_prefix + lab))
    return pd.DataFrame(rows)

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
TA = evaluate(X, '')
print('\n(A) walk-forward 2022-01..2026-10 (yearly retrain, post frozen)')
print(TA.round(2).to_string(index=False)); TA.to_csv('out_s5_volscale_wf.csv', index=False)
# static months with the v4 rolling pool
pp = pd.read_pickle(os.path.join(L.CACHE, 'v4_static_pool_probs.pkl'))['roll']
DEF = ['mlp_raw_B', 'mlp_raw_xr', 'tf_raw_B', 'mlp_res_B', 'mlp_res_xr']
ens = L.combine(pp, DEF, 0.5)
S = ens.reset_index().merge(P[['date', 'stock_id', 'a_hat', 'e_hat']], on=['date', 'stock_id'])
TB = evaluate(S, '')
print('\n(B) static 2026-06..09, v4 rolling pool')
print(TB[['config', 'post', 'sd', 'timing', 'xsec', 'd_post', 't_post']].round(2).to_string(index=False))
TB.to_csv('out_s5_volscale_static.csv', index=False)

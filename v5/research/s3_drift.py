"""s3: concept-drift diagnostics -- where does AR come from, and how stable is each part?
timing_t = s_t * m_t with s_t = net exposure (2*share1 - 1), m_t = Top50 mean intraday return.
mean(s*m) = cov(s, m) [skill: exposure moves with the market] + mean(s)*mean(m) [drift: average tilt x average drift]"""
import numpy as np, pandas as pd, rlib as L
S10, S5 = range(1, 11), range(1, 6)
pr = {n: L.load_wf(n, s) for n, s in [('mlp_raw_B', S10), ('mlp_raw_xr', S10), ('tf_raw_B', S5), ('mlp_res_B', S10), ('mlp_res_xr', S10)]}
DEF = ['mlp_raw_B', 'mlp_raw_xr', 'tf_raw_B', 'mlp_res_B', 'mlp_res_xr']
ens = L.combine(pr, DEF, 0.5)
D = L.daily(ens[['score']])
D['s'] = 2 * D.share1 - 1
v1 = L.daily(L.combine(pr, ['mlp_raw_B', 'mlp_raw_xr', 'tf_raw_B'], None)[['score']] if False else
             L.E.combine({n: pr[n].reindex(ens.index) for n in ['mlp_raw_B', 'mlp_raw_xr', 'tf_raw_B']},
                         [L.member_def(n) for n in ['mlp_raw_B', 'mlp_raw_xr', 'tf_raw_B']])[['score']], exdiv=False)
v1['s'] = 2 * v1.share1 - 1
def decomp(X, lab):
    rows = []
    X = X.copy(); X['year'] = X.index.year.astype(str)
    X.loc[(X.index >= '2026-01-01') & (X.index < '2026-06-01'), 'year'] = '2026-01..05'
    X.loc[X.index >= '2026-06-01', 'year'] = '2026-06..10'
    for y, g in X.groupby('year'):
        cov = ((g.s - g.s.mean()) * (g.m - g.m.mean())).mean()
        rows.append({'model': lab, 'period': y, 'days': len(g), 'AR': g.AR.mean(), 'xsec': g.xsec.mean(), 'timing': g.timing.mean(),
                     'timing_skill(cov)': cov, 'timing_drift(mean s*mean m)': g.s.mean() * g.m.mean(), 'mean_s': g.s.mean(),
                     'mkt m (bp)': g.m.mean(), 'all-0 AR': -g.m.mean()})
    return rows
T = pd.DataFrame(decomp(D, 'v4 (WF)') + decomp(v1, 'v1 rule (same raw members)'))
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(T.round(2).to_string(index=False))
T.to_csv('out_s3_drift_decomp.csv', index=False)
# monthly regime persistence of the intraday market drift
mm = D.m.groupby(D.index.to_period('M')).mean()
print('\nTop50 mean intraday return by month (bp/day), 2022-01..2026-10:')
print(mm.round(1).to_string())
print('lag-1 autocorrelation of monthly means: %.2f (n=%d)' % (mm.autocorr(1), len(mm)))
# xsec stability by month
xm = D.xsec.groupby(D.index.to_period('M')).mean(); tm = D.timing.groupby(D.index.to_period('M')).mean()
print('months with xsec > 0: %d / %d; timing > 0: %d / %d' % ((xm > 0).sum(), len(xm), (tm > 0).sum(), len(tm)))
print('sd of monthly xsec %.1f, of monthly timing %.1f' % (xm.std(), tm.std()))
# how heavily does each rule bet on the market direction, and with how much skill? (2022-01..2026-05, before post)
print('\nmarket bet, 2022-01..2026-05 (s = net exposure 2*share1 - 1, m = Top50 mean intraday return):')
for lab, X in [('v4 rule', D), ('v1 rule', v1)]:
    k = X.index < L.POST_START
    print(f'  {lab}: mean |s| {X.s[k].abs().mean():.3f}, sd(s) {X.s[k].std():.3f}, sd of daily timing {X.timing[k].std():.1f} bp, '
          f'corr(s, m) {np.corrcoef(X.s[k], X.m[k])[0, 1]:.3f}')

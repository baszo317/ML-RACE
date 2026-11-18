"""s4: section 2 of the previous plan -- turn both views into bp before adding them.
Theory (README_v2 5.1): E[e_i|x] = (2 p_e,i - 1) * E[|e_i| | x]. If E[|e_i| | x] is proportional to eps_i (trailing 60-day mean
|e_i|), the OOS slope of e on 2(p_e - mean_day p_e) inside an eps-quintile must grow in proportion to eps.
Check first; only then compare the bp rule with v4 (paired, same days)."""
import numpy as np, pandas as pd, rlib as L
S10, S5 = range(1, 11), range(1, 6)
pr = {n: L.load_wf(n, s) for n, s in [('mlp_raw_B', S10), ('mlp_raw_xr', S10), ('tf_raw_B', S5), ('mlp_res_B', S10), ('mlp_res_xr', S10)]}
DEF = ['mlp_raw_B', 'mlp_raw_xr', 'tf_raw_B', 'mlp_res_B', 'mlp_res_xr']
ens = L.combine(pr, DEF, 0.5)
P = L.panel().sort_values(['stock_id', 'date']).copy()
cl = lambda s: s.abs().clip(upper=0.05)
P['a_hat'] = P.groupby('stock_id').target_r.transform(lambda s: cl(s).rolling(60, min_periods=20).mean().shift(1))
P['e_hat'] = P.groupby('stock_id').resid_r.transform(lambda s: cl(s).rolling(60, min_periods=20).mean().shift(1))
X = ens.reset_index().merge(P[['date', 'stock_id', 'a_hat', 'e_hat', 'resid_r', 'target_r', 'top50', 'ex_today', 'mkt_r']], on=['date', 'stock_id'])
X = X[X.target_r.notna()].copy()
X['xsig'] = 2 * X.p_e_c                                   # 2 (p_e - mean_day p_e)
X['tsig'] = 2 * X.p_raw - 1
vh = X.date < L.POST_START
# ---------- theory check, stock view: slope of e on xs inside eps quintiles (val+hold, OOS)
Z = X[vh & X.e_hat.notna()].copy()
Z['q'] = pd.qcut(Z.e_hat, 5, labels=False)
rows = []
for q, g in Z.groupby('q'):
    b = (g.xsig * g.resid_r).sum() / (g.xsig ** 2).sum()
    rows.append({'eps quintile': q + 1, 'mean eps (bp)': g.e_hat.mean() * 1e4, 'mean |e| (bp)': g.resid_r.abs().mean() * 1e4,
                 'slope (bp)': b * 1e4, 'slope / mean eps': b / g.e_hat.mean(), 'n': len(g)})
TS = pd.DataFrame(rows)
print('stock view: OOS slope of e on 2(p_e - mean) by eps quintile (2022-01..2026-05)')
print(TS.round(3).to_string(index=False))
# bootstrap-by-day SE of the slope ratio between top and bottom quintile
days = Z.date.unique(); rng = np.random.default_rng(0)
G = {d: g for d, g in Z.groupby('date')}
num = Z.assign(n=Z.xsig * Z.resid_r, d2=Z.xsig ** 2).groupby(['date', 'q'])[['n', 'd2']].sum().unstack('q')
br = []
for _ in range(300):
    idx = rng.choice(len(num), len(num))
    s = num.iloc[idx].sum()
    b = s['n'] / s['d2']
    br.append(b[4] / b[0])
print('slope(top quintile) / slope(bottom quintile) = %.2f (bootstrap by day: 5-95%% %.2f..%.2f); eps ratio %.2f'
      % (TS['slope (bp)'].iloc[4] / TS['slope (bp)'].iloc[0], np.quantile(br, .05), np.quantile(br, .95),
         TS['mean eps (bp)'].iloc[4] / TS['mean eps (bp)'].iloc[0]))
# ---------- market view: slope of m on (2 pbar - 1) inside quintiles of the day's mean a_hat (time series)
dd = X.groupby('date').agg(m=('target_r', 'mean'), pbar=('p_raw', 'mean'), a=('a_hat', 'mean'))
dd = dd[dd.index < L.POST_START].dropna()
dd['f'] = 2 * dd.pbar - 1; dd['q'] = pd.qcut(dd.a, 5, labels=False)
rows = []
for q, g in dd.groupby('q'):
    b = (g.f * g.m).sum() / (g.f ** 2).sum()
    rows.append({'a quintile': q + 1, 'mean a (bp)': g.a.mean() * 1e4, 'mean |m| (bp)': g.m.abs().mean() * 1e4, 'slope (bp)': b * 1e4,
                 'slope / mean a': b / g.a.mean(), 'days': len(g)})
print('\nmarket view: slope of m (100-stock mean) on 2 pbar - 1 by market-vol quintile (time series, 2022-01..2026-05)')
print(pd.DataFrame(rows).round(3).to_string(index=False))
X.to_pickle(L.os.path.join(L.CACHE, 'wf_v4_frame.pkl'))

"""s8: risk 3 -- how large must a true difference be before we can see it?
paired difference d_t of daily AR; MDE (two-sided 5%, power 80%) = (1.96 + 0.84) * sd(d) / sqrt(n)."""
import os, numpy as np, pandas as pd, rlib as L
from scipy.stats import norm
M = pd.read_pickle(os.path.join(L.CACHE, 'family_daily_AR.pkl'))
base = M['v2 default w0.5']
pairs = [('w_T 0.75 vs 0.5', 'v2 default w0.75'), ('w_T 0 vs 0.5', 'v2 default w0.0'), ('+tf_res', 'v2 +tf_res w0.5'),
         ('B only (no XR)', 'v2 B only'), ('no exdiv', 'v2 default no exdiv'), ('MLP only (no TF)', 'v2 MLP-only w0.5'),
         ('v1 rule (mlp10+tf5)', 'mlp10+tf5 v1 rule'), ('v1-era mlp_F (85 feat.) vs v1 mlp_B (67)', None)]
rows = []
for lab, c in pairs:
    if c is None:
        d = (M['v1-era mlp_F'] - M['v1-era mlp_B']).dropna()
    else:
        d = (M[c] - base).dropna()
    sd = d.std(); r = {'comparison': lab, 'mean diff (val+hold)': d.mean(), 'sd of daily diff': sd, 't': d.mean() / sd * np.sqrt(len(d))}
    for n in (19, 84, 740, 1065):
        r[f'MDE n={n}'] = 2.80 * sd / np.sqrt(n)
    r['P(better on 19 days | true diff = mean)'] = norm.cdf(d.mean() * np.sqrt(19) / sd)
    rows.append(r)
T = pd.DataFrame(rows); pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
print(T.round(2).to_string(index=False)); T.to_csv('out_s8_power.csv', index=False)
s = base.std(); m = base.mean()
print(f'\nv2 default alone: mean {m:.2f}, sd {s:.1f} bp/day -> SE of a 19-day mean {s/np.sqrt(19):.1f}; P(19-day mean > 0) = {norm.cdf(m*np.sqrt(19)/s):.3f}')

# variance: v2/v4 rule vs the v1-like baseline on the same days (val + hold)
v1 = M['v1-like mlp3+tf3']; x, y = base.align(v1, join='inner'); n = len(x)
r = np.corrcoef(x + y, x - y)[0, 1]; t_pm = r * np.sqrt(n - 2) / np.sqrt(1 - r * r)
print(f'\nPitman-Morgan (Var(x) = Var(y) <=> corr(x + y, x - y) = 0): sd v2 {x.std():.1f} vs v1-like {y.std():.1f} bp/day, '
      f'ratio {x.std() / y.std():.3f}, r = {r:.3f}, t = {t_pm:.1f} (n = {n})')
# robustness to fat tails and autocorrelation: stationary bootstrap of the sd ratio (mean block length 10 days)
rng = np.random.default_rng(0); X, Y = x.values, y.values; B = 2000; ratios = np.empty(B)
for b in range(B):
    idx = np.empty(n, int); i = rng.integers(n)
    for k in range(n):
        idx[k] = i; i = rng.integers(n) if rng.random() < 0.1 else (i + 1) % n
    ratios[b] = X[idx].std() / Y[idx].std()
print(f'  stationary bootstrap of the sd ratio: 2.5-97.5 % = {np.quantile(ratios, 0.025):.3f}..{np.quantile(ratios, 0.975):.3f}, '
      f'share of resamples with ratio >= 1: {np.mean(ratios >= 1):.4f}')
print(f'  P(19-day mean > 0): v1-like {norm.cdf(y.mean() * np.sqrt(19) / y.std()):.3f} -> v2 {norm.cdf(x.mean() * np.sqrt(19) / x.std()):.3f}')

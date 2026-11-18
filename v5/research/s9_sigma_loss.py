"""s9: section 4.1 of the previous plan -- loss weight min(|r|,5%) / sigma^alpha with sigma as an input (alpha 0, 0.5, 1).
All MLP members retrained in the same environment with seeds 1-5 (orig = the v2/v4 training, reproduced to 2e-6);
the Transformer (market view) is the overnight run (seeds 1-5) in every row. Pre-registered: adopt only if val paired
t >= 2 (Holm-adjusted over the 3 variants) and hold has the same sign."""
import os, numpy as np, pandas as pd, rlib as L
from scipy.stats import norm
W5 = os.path.join(L.CACHE, 'wf5')
MEM = ['mlp_raw_B', 'mlp_raw_xr', 'mlp_res_B', 'mlp_res_xr']
def load5(name, variant, scheme='year', seeds=range(1, 6)):
    starts = [a for a, _ in L.YEAR_FOLDS] + ['2026-06-01'] if scheme == 'year' else None
    d = os.path.join(W5, f'{name}__{variant}__{scheme}')
    files = sorted(os.listdir(d))
    parts = {}
    for f in files:
        a, s = f[:-4].split('_seed')
        if int(s) in seeds:
            parts.setdefault(a, []).append(L._read(os.path.join(d, f)))
    out = []
    for a, xs in sorted(parts.items()):
        if len(xs) != len(seeds):
            raise RuntimeError(f'{name} {variant} {a}: {len(xs)} seeds')
        out.append(pd.concat(xs, axis=1).mean(axis=1))
    return pd.concat(out)
tf = L.load_wf('tf_raw_B', range(1, 6))
def rule(variant, w=0.5, with_tf=True):
    pr = {n: load5(n, variant) for n in MEM}
    names = MEM + (['tf_raw_B'] if with_tf else [])
    if with_tf:
        pr['tf_raw_B'] = tf
    return L.daily(L.combine(pr, names, w)[['score']])
if __name__ == '__main__':
    base = rule('orig')
    rows = [L.summary_row(base, None, 'orig (v4 training), MLP seeds 1-5 + TF 1-5')]
    R = {}
    for v, lab in [('sig_a0', 'sigma as input, alpha = 0'), ('sig_a05', 'sigma as input, alpha = 0.5'), ('sig_a1', 'sigma as input, alpha = 1')]:
        R[v] = rule(v); rows.append(L.summary_row(R[v], base, lab))
    T = pd.DataFrame(rows)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print(T.round(2).to_string(index=False)); T.to_csv('out_s9_sigma_loss.csv', index=False)
    # Holm on val (one-sided improvement)
    tv = T.set_index('config').t_val.dropna().sort_values(ascending=False)
    print('\nHolm (val, one-sided):')
    for i, (k, t) in enumerate(tv.items()):
        thr = norm.ppf(1 - 0.05 / (len(tv) - i)); print(f'  {k:32s} t_val = {t:5.2f}  needs >= {thr:.2f}')
    for v in ('sig_a05', 'sig_a1'):
        r = L.summary_row(R[v], R['sig_a0'], v + ' vs sig_a0')
        print(f"  {v} vs alpha 0 (weighting only): val {r['d_val']:+.2f} (t {r['t_val']:.2f}), hold {r['d_hold']:+.2f} (t {r['t_hold']:.2f}), post {r['d_post']:+.2f}")

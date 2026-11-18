"""s11: input-feature ablation -- drop one group of the 67 B features (and its xr_ ranks) from all 4 MLP members,
retrain (yearly protocol, seeds 1-3), keep the Transformer (overnight run, seeds 1-5) unchanged in every row.
Pre-registered: drop a group only if removing it IMPROVES val with Holm-adjusted significance (8 groups -> the best
needs t >= z(1 - 0.05/8) = 2.50) and hold has the same sign. Otherwise the group stays (status quo)."""
import os, numpy as np, pandas as pd, rlib as L
from scipy.stats import norm
from s9_sigma_loss import load5, MEM
from wf_runner import GROUPS
tf = L.load_wf('tf_raw_B', range(1, 6))
S3 = range(1, 4)
def rule(variant, w=0.5):
    pr = {n: load5(n, variant, 'year', S3) for n in MEM}; pr['tf_raw_B'] = tf
    return L.daily(L.combine(pr, MEM + ['tf_raw_B'], w)[['score']])
base = rule('orig')
rows = [L.summary_row(base, None, 'all 67 B features (orig, MLP seeds 1-3)')]
for g, fl in GROUPS.items():
    D = rule('drop_' + g)
    r = L.summary_row(D, base, f'drop {g} ({len(fl)})')
    rows.append(r)
T = pd.DataFrame(rows); pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
cols = ['config', 'val', 'hold', 'post', 'all', 'd_val', 't_val', 'd_hold', 't_hold', 'd_post', 't_post', 'd_all', 't_all', 'timing', 'xsec']
print(T[cols].round(2).to_string(index=False)); T.to_csv('out_s11_feature_groups.csv', index=False)
tv = T.set_index('config').t_val.dropna().sort_values(ascending=False)
print('\nHolm on val (one-sided, improvement = dropping helps):')
for i, (k, t) in enumerate(tv.items()):
    print(f'  {k:28s} t_val = {t:5.2f}  needs >= {norm.ppf(1 - 0.05 / (len(tv) - i)):.2f}')

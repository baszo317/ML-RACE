"""
walkforward_eval.py -- reproduce the feature-set comparison (needs: pip install lightgbm pyarrow)

Expanding-window walk-forward, train from 2020-03; test folds 2022, 2023, 2024 (validation) and
2025, 2026-01..05 (holdout). LightGBM binary classifier weighted by |intraday return| (= competition score).
Prints the competition score AR (bp/day, Top50) per config and the soft (confidence-weighted) paired t vs A.
Runtime: about 3 minutes per config on a 2-core laptop.
"""
import os, json, time
import numpy as np, pandas as pd, lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
P = pd.read_parquet(os.path.join(HERE, 'panel_candidates.parquet')); P['date'] = pd.to_datetime(P.date)
cat = pd.read_csv(os.path.join(HERE, 'feature_catalog.csv')); GROUPS = cat.groupby('group').feature.apply(list).to_dict()
L = json.load(open(os.path.join(HERE, 'final_feature_lists.json')))
CONFIGS = {'A_competition_all': [f for g, fs in GROUPS.items() if g.startswith('C_') for f in fs],
           'B_competition_only': L['B_competition_only'],
           'F_best': L['F_best']}
FOLDS = [('2022-01-01', '2023-01-01'), ('2023-01-01', '2024-01-01'), ('2024-01-01', '2025-01-01'),
         ('2025-01-01', '2026-01-01'), ('2026-01-01', '2026-06-01')]
PARAMS = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.7,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=2)
SEEDS = (11, 12, 13, 14, 15); N_TREES = 300
W = P.target_r.abs().clip(upper=0.05).values

def run(features):
    out = []
    for a, b in FOLDS:
        a, b = pd.Timestamp(a), pd.Timestamp(b); tr = (P.date < a).values; te = ((P.date >= a) & (P.date < b)).values
        ds = lgb.Dataset(P.loc[tr, features].values, P.y.values[tr], weight=W[tr], free_raw_data=False)
        pr = np.zeros(te.sum())
        for s in SEEDS:
            m = lgb.train({**PARAMS, 'seed': s, 'bagging_seed': s, 'feature_fraction_seed': s}, ds, num_boost_round=N_TREES)
            pr += m.predict(P.loc[te, features].values) / len(SEEDS)
        o = P.loc[te, ['date', 'sid', 'top50', 'target_r', 'nd_exdiv_today']].copy(); o['p'] = pr; out.append(o)
    return pd.concat(out)

def series(o, exdiv_override=False):
    o = o[o.top50]; pred = (o.p > 0.5).astype(int)
    if exdiv_override:
        pred = np.where(o.nd_exdiv_today == 1, 0, pred)
    hard = ((2 * pred - 1) * o.target_r).groupby(o.date).mean() * 1e4
    pos = (o.p - 0.5) / (o.p - 0.5).std(); soft = (pos * o.target_r).groupby(o.date).mean() * 1e4
    return hard, soft

if __name__ == '__main__':
    cut = pd.Timestamp('2025-01-01'); tt = lambda x: x.mean() / x.std() * np.sqrt(len(x)); res = {}
    for name, fs in CONFIGS.items():
        t0 = time.time(); o = run(fs); res[name] = o
        h, s = series(o); hx, _ = series(o, exdiv_override=True)
        line = (f'{name:20s} n={len(fs):3d}  AR val {h[h.index < cut].mean():6.2f}  hold {h[h.index >= cut].mean():6.2f}  all {h.mean():6.2f}'
                f'  | with ex-div override: val {hx[hx.index < cut].mean():6.2f} hold {hx[hx.index >= cut].mean():6.2f}')
        if name != 'A_competition_all':
            ha, sa = series(res['A_competition_all']); d = (s - sa).dropna()
            line += f'  | soft t vs A: val {tt(d[d.index < cut]):+.2f} hold {tt(d[d.index >= cut]):+.2f}'
        print(line + f'  ({time.time() - t0:.0f}s)', flush=True)
    q = P[P.top50 & (P.date >= '2022-01-01')]
    z = (-q.target_r).groupby(q.date).mean() * 1e4
    print(f'baseline always-0: val {z[z.index < cut].mean():.2f} hold {z[z.index >= cut].mean():.2f} all {z.mean():.2f}')

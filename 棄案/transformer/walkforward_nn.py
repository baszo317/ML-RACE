"""
walkforward_nn.py -- reproduce the model comparison (MLP baseline vs Transformer vs their average)

Expanding-window walk-forward, training from 2020-03-02; test folds 2022, 2023, 2024 (validation: used to choose the
model) and 2025, 2026-01..05 (holdout: confirmation only). Nothing from Jun-Sep 2026 is used.
Score = competition AR, mean over days of the Top50 average of (2p-1)(close-open)/open, in bp/day.
Runtime on a 2-core laptop: MLP ~1 min, Transformer ~25 min (3 seeds x 5 folds).
  python walkforward_nn.py
"""
import os, sys, json, time
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from comp_features import build_comp_features
from tfmodel import train_model, predict
from train_and_predict import HP, MODELS, TRAIN_START

DATA = os.path.join(os.path.dirname(HERE), 'data')
FOLDS = [('2022-01-01', '2023-01-01'), ('2023-01-01', '2024-01-01'), ('2024-01-01', '2025-01-01'),
         ('2025-01-01', '2026-01-01'), ('2026-01-01', '2026-06-01')]
SEEDS = (1, 2, 3)

def run(name, P, feats):
    cfg = MODELS[name]; out = []
    for a, b in FOLDS:
        a, b = pd.Timestamp(a), pd.Timestamp(b); t0 = time.time(); ps = []
        for s in SEEDS:
            bd = train_model(cfg['kind'], P, feats, lambda d: (d.date >= TRAIN_START) & (d.date < a), cfg['seq_len'], seed=s, hp=HP)
            ps.append(predict(bd, P, lambda d: (d.date >= a) & (d.date < b) & d.target_r.notna()).set_index(['date', 'stock_id']).p)
        o = pd.concat(ps, axis=1).mean(axis=1).rename('p').reset_index()
        out.append(o.merge(P[['date', 'stock_id', 'top50', 'target_r']], on=['date', 'stock_id']))
        print(f'{name} test {a.year}: {time.time() - t0:.0f}s', flush=True)
    return pd.concat(out, ignore_index=True)

def daily(o):
    o = o[o.top50]; return ((2 * (o.p > 0.5) - 1) * o.target_r).groupby(o.date).mean() * 1e4

if __name__ == '__main__':
    torch.set_num_threads(2)
    feats = json.load(open(os.path.join(HERE, 'final_feature_lists.json')))['B_competition_only']
    P = build_comp_features(DATA); P = P[P.date < '2026-06-01'].reset_index(drop=True)
    O = {n: run(n, P, feats) for n in ['mlp', 'transformer']}
    e = O['transformer'].merge(O['mlp'][['date', 'stock_id', 'p']], on=['date', 'stock_id'], suffixes=('', '_m'))
    e['p'] = (e.p + e.p_m) / 2; O['ensemble'] = e
    q = O['mlp'][O['mlp'].top50]; S = {k: daily(v) for k, v in O.items()}; S['always_0'] = (-q.target_r).groupby(q.date).mean() * 1e4
    cut = pd.Timestamp('2025-01-01')
    print(pd.DataFrame({k: {'validation 2022-24': s[s.index < cut].mean(), 'holdout 2025-26/05': s[s.index >= cut].mean(),
                            **{str(y): s[s.index.year == y].mean() for y in range(2022, 2027)}} for k, s in S.items()}).T.round(2))

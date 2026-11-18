r"""
train_best.py -- run on your own computer:   python train_best.py        (pip install torch numpy pandas)

Trains with the best feature set and tests two ideas against drift / overfitting, with the same walk-forward protocol
as before: training from 2020-03-02, test years 2022, 2023, 2024 = validation (used to decide), 2025 and 2026-01..05 =
holdout (confirmation only). Nothing from Jun-Sep 2026 is used (the new-data features stop at 2026-05-31 anyway).

Steps (resumable: finished experiments in results\ are skipped when you run it again)
  0. smoke test (~1 min): tiny MLP + Transformer fits, to catch environment problems before the long part
  1. MLP, 4 feature sets: B (67 competition features) | B + margin | B + breadth & TAIEX | F (best, 85 features)
  2. MLP with B features: recency weighting (half-life 250 trading days) | retraining every quarter instead of yearly
  3. Transformer: B and F   (the slow part)
  4. results\summary.txt  (also printed at the end)
  --final   additionally trains final models with F (5 seeds each, data <= 2026-05-29) into models_best\
            (only useful if F wins AND you fetch the margin / breadth / TAIEX data every day -- the course server
            does not provide them)
Runtime: about 1 hour on a 4-core laptop. Progress: results\train_best.log
"""
import os, sys, json, time, argparse, traceback
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
from comp_features import build_comp_features

RES = os.path.join(HERE, 'results'); os.makedirs(RES, exist_ok=True)
LOG = os.path.join(RES, 'train_best.log')
YEAR_FOLDS = [('2022-01-01', '2023-01-01'), ('2023-01-01', '2024-01-01'), ('2024-01-01', '2025-01-01'),
              ('2025-01-01', '2026-01-01'), ('2026-01-01', '2026-06-01')]
_q = list(pd.date_range('2022-01-01', '2026-04-01', freq='QS')) + [pd.Timestamp('2026-06-01')]
QUARTER_FOLDS = [(str(a.date()), str(b.date())) for a, b in zip(_q[:-1], _q[1:])]
SEEDS = (1, 2, 3)
CUT = pd.Timestamp('2025-01-01')
ND_MARGIN = ['nd_margin_util', 'nd_short_to_margin', 'nd_margin_chg_5', 'nd_short_chg_5', 'nd_margin_flow', 'nd_short_flow',
             'nd_offset_ratio', 'nd_offset_ratio_20', 'nd_margin_buy_ratio', 'nd_short_sell_ratio']
ND_BREADTH_TAIEX = ['nd_all_intra', 'nd_all_breadth', 'nd_all_intra_vw', 'nd_small_minus_top', 'nd_all_intra_5',
                    'nd_mkt_ret_20', 'nd_mkt_vol_20', 'nd_mkt_value_ratio']

def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def load_panel():
    P = build_comp_features(os.path.join(ROOT, 'data'))
    P = P[P.date < '2026-06-01'].reset_index(drop=True)
    L = json.load(open(os.path.join(ROOT, 'best_features', 'final_feature_lists.json')))
    nd_cols = [c for c in L['F_best'] if c.startswith('nd_')]
    src = os.path.join(ROOT, 'best_features', 'train_best.csv')
    nd = pd.read_csv(src, usecols=['date', 'stock_id'] + nd_cols, dtype={'stock_id': str}, parse_dates=['date'], encoding='utf-8-sig')
    P = P.merge(nd, on=['date', 'stock_id'], how='left')
    B = L['B_competition_only']; F = L['F_best']
    assert set(ND_MARGIN + ND_BREADTH_TAIEX) == set(nd_cols), 'unexpected new-data feature list'
    sets = {'B': B, 'B+margin': B + ND_MARGIN, 'B+breadth_taiex': B + ND_BREADTH_TAIEX, 'F': F}
    return P, sets

def walk_forward(P, feats, kind, seq_len, hp, folds, seeds=SEEDS, half_life=None):
    from tfmodel import train_model, predict
    out = []
    for a, b in folds:
        a, b = pd.Timestamp(a), pd.Timestamp(b); t0 = time.time(); ps = []
        for s in seeds:
            bd = train_model(kind, P, feats, lambda d: (d.date >= '2020-03-02') & (d.date < a), seq_len, seed=s, hp=hp, half_life=half_life)
            ps.append(predict(bd, P, lambda d: (d.date >= a) & (d.date < b) & d.target_r.notna()).set_index(['date', 'stock_id']).p)
        o = pd.concat(ps, axis=1).mean(axis=1).rename('p').reset_index()
        out.append(o.merge(P[['date', 'stock_id', 'top50', 'target_r']], on=['date', 'stock_id']))
        log(f'    test {a.date()}..{b.date()}: {time.time() - t0:.0f}s')
    return pd.concat(out, ignore_index=True)

def daily(o, soft=False):
    o = o[o.top50]
    pos = ((o.p - 0.5) / (o.p - 0.5).std()) if soft else (2 * (o.p > 0.5) - 1)
    return (pos * o.target_r).groupby(o.date).mean() * 1e4

def tstat(x):
    x = x.dropna(); return x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 and x.std() > 0 else np.nan

def smoke(P, sets, hp):
    from tfmodel import train_model, predict
    t0 = time.time(); sub = P[P.stock_id.isin(sorted(P.stock_id.unique())[:12])]
    for kind, L in [('mlp', 1), ('transformer', 20)]:
        bd = train_model(kind, sub, sets['F'], lambda d: (d.date >= '2023-01-01') & (d.date < '2023-07-01'), L, seed=1,
                         hp={**hp, 'epochs': 1, 'patience': 1}, val_days=20, half_life=250)
        pr = predict(bd, sub, lambda d: (d.date >= '2023-07-01') & (d.date < '2023-08-01') & d.target_r.notna())
        assert len(pr) > 0 and pr.p.between(0, 1).all(), f'{kind}: bad predictions'
    log(f'smoke test OK ({time.time() - t0:.0f}s)')

def summarize(results):
    lines = []; S = {k: daily(v) for k, v in results.items()}; SS = {k: daily(v, soft=True) for k, v in results.items()}
    for a, b in [('transformer_B', 'mlp_B'), ('transformer_F', 'mlp_F')]:
        if a in results and b in results:
            e = results[a].merge(results[b][['date', 'stock_id', 'p']], on=['date', 'stock_id'], suffixes=('', '_2'))
            e['p'] = (e.p + e.p_2) / 2; name = 'ensemble_' + a.split('_')[1]
            S[name] = daily(e); SS[name] = daily(e, soft=True)
    o = next(iter(results.values())); o = o[o.top50]; S['always_0'] = (-o.target_r).groupby(o.date).mean() * 1e4
    rows = []
    base_of = lambda k: ('mlp_B' if k.startswith('mlp_') else 'transformer_B' if k.startswith('transformer_') else 'ensemble_B' if k.startswith('ensemble_') else None)
    for k, s in S.items():
        r = {'experiment': k, 'val_AR': s[s.index < CUT].mean(), 'hold_AR': s[s.index >= CUT].mean(), 'all_AR': s.mean()}
        r.update({str(y): s[s.index.year == y].mean() for y in range(2022, 2027)})
        b = base_of(k)
        if b and b != k and b in SS and k in SS:
            d = (SS[k] - SS[b]).dropna(); dh = (s - S[b]).dropna()
            r.update({'vs': b, 'soft_t_val': tstat(d[d.index < CUT]), 'soft_t_hold': tstat(d[d.index >= CUT]),
                      'AR_diff_val': dh[dh.index < CUT].mean(), 'AR_diff_hold': dh[dh.index >= CUT].mean()})
        rows.append(r)
    T = pd.DataFrame(rows); T.to_csv(os.path.join(RES, 'summary.csv'), index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    lines.append('AR = competition score in bp/day (Top50). val = 2022-2024 (decides), hold = 2025-2026/05 (confirms).')
    lines.append('soft_t_* = paired t-stat of the confidence-weighted score vs the baseline in column "vs" (|t| > 2 is a clear difference).')
    lines.append(T.round(2).to_string(index=False))
    txt = '\n'.join(lines)
    with open(os.path.join(RES, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    print(txt)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--final', action='store_true'); ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--skip-transformer', action='store_true')
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads if a.threads > 0 else max(1, os.cpu_count() or 1))
    from train_and_predict import HP
    log(f'start: torch {torch.__version__}, {torch.get_num_threads()} threads')
    P, sets = load_panel(); log(f'panel {P.shape}, feature sets ' + ', '.join(f'{k}={len(v)}' for k, v in sets.items()))
    smoke(P, sets, HP)
    EXP = [('mlp_B', 'mlp', 'B', {}), ('mlp_B+margin', 'mlp', 'B+margin', {}), ('mlp_B+breadth_taiex', 'mlp', 'B+breadth_taiex', {}),
           ('mlp_F', 'mlp', 'F', {}), ('mlp_B_recency', 'mlp', 'B', {'half_life': 250}), ('mlp_B_quarterly', 'mlp', 'B', {'folds': QUARTER_FOLDS})]
    if not a.skip_transformer:
        EXP += [('transformer_B', 'transformer', 'B', {}), ('transformer_F', 'transformer', 'F', {})]
    results = {}
    for name, kind, fs, opt in EXP:
        f = os.path.join(RES, f'wf_{name}.pkl')
        if os.path.exists(f):
            results[name] = pd.read_pickle(f); log(f'{name}: already done, loaded'); continue
        log(f'{name}: {kind}, {len(sets[fs])} features {opt if opt else ""}')
        t0 = time.time()
        try:
            o = walk_forward(P, sets[fs], kind, 20 if kind == 'transformer' else 1, HP, opt.get('folds', YEAR_FOLDS), half_life=opt.get('half_life'))
        except Exception:
            log(f'{name} FAILED:\n{traceback.format_exc()}'); continue
        o.to_pickle(f); results[name] = o
        s = daily(o); log(f'{name}: val {s[s.index < CUT].mean():.2f}  hold {s[s.index >= CUT].mean():.2f} bp/day  ({time.time() - t0:.0f}s)')
    summarize(results)
    if a.final:
        from tfmodel import train_model
        md = os.path.join(HERE, 'models_best'); os.makedirs(md, exist_ok=True)
        for kind, L in [('mlp', 1), ('transformer', 20)]:
            for s in range(1, 6):
                b = train_model(kind, P, sets['F'], lambda d: (d.date >= '2020-03-02') & (d.date <= '2026-05-29'), L, seed=s, hp=HP)
                torch.save(dict(state_dict=b['model'].state_dict(), scaler=b['scaler'].state(), kind=kind, seq_len=L, features=sets['F'],
                                hp=b['hp'], train_end='2026-05-29', best_epoch=b['best_epoch']), os.path.join(md, f'{kind}_seed{s}.pt'))
                log(f'final {kind} seed {s} saved (epochs {b["best_epoch"]})')
    log('done')

if __name__ == '__main__':
    main()

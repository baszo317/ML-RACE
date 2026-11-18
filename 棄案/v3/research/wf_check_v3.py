r"""
research\wf_check_v3.py -- walk-forward check: do the 5 new stock-picking features (v3) beat v2?  (resumable)

  python research\wf_check_v3.py              from the v3 folder; MLP members, seeds 1-5
  python research\wf_check_v3.py --seeds 10   more seeds (finished units are reused)
  python research\wf_check_v3.py --summary    only rebuild the table from the finished units

Protocol = v2's overnight_experiments.py: expanding window from 2020-03-02; yearly test folds 2022, 2023, 2024
(validation: decides) and 2025, 2026-01..05 (holdout: confirms); plus "post" = 2026-06-01 .. last labelled day,
predicted by models trained up to 2026-05-29.
Members (all MLP, v2 hyper-parameters):
  market view (same in v2 and v3) : mlp_raw_B (B), mlp_raw_xr (B+XR)
  stock view v2                   : mlp_res_B (B), mlp_res_xr (B+XR)
  stock view v3                   : mlp_res_BN (B+N), mlp_res_xrN (B+XR+N)     N = features_v3.NEW (+ ranks)
Score = competition AR in bp/day over the Top50, v2 rule (W_T * (pbar - 0.5) + centred p_e) + ex-dividend rule.
timing / xsec = the part of AR from the overall up/down call / from picking stocks. t_* = paired t of the daily
difference against the v2 row with the same W_T.
Each (member, fold, seed) is saved in research\wf_v3\ as soon as it finishes. Output: research\wf_check_v3.txt
"""
import os, sys, time, argparse, traceback
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); V3 = os.path.dirname(HERE)
if V3 not in sys.path:
    sys.path.insert(0, V3)
import ens_v3 as E

WF = os.path.join(HERE, 'wf_v3'); OUT = os.path.join(HERE, 'wf_check_v3.txt')
YEAR_FOLDS = [('2022-01-01', '2023-01-01'), ('2023-01-01', '2024-01-01'), ('2024-01-01', '2025-01-01'),
              ('2025-01-01', '2026-01-01'), ('2026-01-01', '2026-06-01')]
POST_START = '2026-06-01'
CUT = pd.Timestamp('2025-01-01')
SPEC = {'mlp_res_BN': ('B+N', 'resid'), 'mlp_res_xrN': ('B+XR+N', 'resid'), 'mlp_res_B': ('B', 'resid'),
        'mlp_res_xr': ('B+XR', 'resid'), 'mlp_raw_B': ('B', 'raw'), 'mlp_raw_xr': ('B+XR', 'raw')}   # run order
RAW = ['mlp_raw_B', 'mlp_raw_xr']
CONF = [('v2 stock view: res_B + res_xr', RAW + ['mlp_res_B', 'mlp_res_xr']),
        ('v3 stock view: res_BN + res_xrN', RAW + ['mlp_res_BN', 'mlp_res_xrN']),
        ('v2+v3: all four res members', RAW + ['mlp_res_B', 'mlp_res_xr', 'mlp_res_BN', 'mlp_res_xrN'])]
W_GRID = (0.0, 0.5, 0.75)

def unit_file(name, a, s):
    return os.path.join(WF, name, f'{a}_seed{s}.pkl')

def run_unit(P, fs, name, a, b, s):
    from tfmodel import predict
    feats, target = SPEC[name]
    m = E.member(name, 'mlp', feats=feats, target=target)
    a_, b_ = pd.Timestamp(a), pd.Timestamp(b); te = a_ - pd.Timedelta(days=1)
    bd = E.train_member(P, m, s, te, fs[feats])
    return predict(bd, P, lambda d: (d.date >= a_) & (d.date < b_) & d.target_r.notna())[['date', 'stock_id', 'p']]

def load(name, seeds):
    parts = []
    for a in [x for x, _ in YEAR_FOLDS] + [POST_START]:
        fs = [unit_file(name, a, s) for s in seeds]
        if not all(os.path.exists(f) for f in fs):
            return None
        parts.append(pd.concat([pd.read_pickle(f).set_index(['date', 'stock_id']).p for f in fs], axis=1).mean(axis=1))
    return pd.concat(parts)

def series(out, P):
    q = out.reset_index().merge(P[['date', 'stock_id', 'top50', 'target_r', 'ex_today']], on=['date', 'stock_id'])
    q = q[q.target_r.notna()]
    q['s'] = 2 * E.apply_exdiv((q.score > 0).astype(int), q.ex_today) - 1
    x = q[q.top50]
    S = (x.s * x.target_r).groupby(x.date).mean() * 1e4
    T = x.groupby('date').s.mean() * x.groupby('date').target_r.mean() * 1e4
    return S, T

def summarize(P, seeds):
    probs = {n: load(n, seeds) for n in SPEC}
    rows, base = [], {}
    for label, names in CONF:
        if any(probs[n] is None for n in names):
            continue
        idx = None
        for n in names:
            idx = probs[n].index if idx is None else idx.intersection(probs[n].index)
        members = [E.member(n, 'mlp', feats=SPEC[n][0], target=SPEC[n][1]) for n in names]
        for w in W_GRID:
            S, T = series(E.combine({n: probs[n].reindex(idx) for n in names}, members, w_t=w), P)
            base.setdefault(w, S)
            r = {'config': label, 'w_T': w, 'val': S[S.index < CUT].mean(), 'hold': S[(S.index >= CUT) & (S.index < POST_START)].mean(),
                 'post': S[S.index >= POST_START].mean(), 'all': S.mean(), 'sd_day': S.std(), 'timing': T.mean(), 'xsec': (S - T).mean()}
            for y in range(2022, 2027):
                r[str(y)] = S[(S.index.year == y) & (S.index < POST_START)].mean()
            d = (S - base[w]).dropna()
            for nm, sel in [('t_val', d.index < CUT), ('t_hold', (d.index >= CUT) & (d.index < POST_START)),
                            ('t_post', d.index >= POST_START), ('t_all', d.index == d.index)]:
                z = d[sel]; r[nm] = z.mean() / z.std() * np.sqrt(len(z)) if len(z) > 2 and z.std() > 0 else np.nan
            rows.append(r)
    if not rows:
        print('no complete configuration yet'); return
    Tb = pd.DataFrame(rows).sort_values(['w_T', 'config'], kind='stable')
    pd.set_option('display.width', 300); pd.set_option('display.max_columns', 40)
    txt = [f'AR = competition score, bp/day, Top50, v2 rule + ex-dividend rule, MLP members, seeds {list(seeds)}.',
           'val = 2022-2024 (decides), hold = 2025-2026/05 (confirms), post = 2026/06 .. last labelled day (models frozen at 2026-05-29).',
           'timing / xsec = part of AR from the overall up/down call / from picking stocks.',
           't_* = paired t of the daily difference vs the "v2 stock view" row with the same w_T.', '',
           Tb.round(2).to_string(index=False)]
    open(OUT, 'w', encoding='utf-8').write('\n'.join(txt) + '\n'); print('\n'.join(txt))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--summary', action='store_true')
    ap.add_argument('--threads', type=int, default=0)
    a = ap.parse_args(); seeds = range(1, a.seeds + 1)
    P = E.build_panel(E.DATA); last = P[P.target_r.notna()].date.max()
    folds = YEAR_FOLDS + [(POST_START, str((last + pd.Timedelta(days=1)).date()))]
    if not a.summary:
        import torch
        torch.set_num_threads(a.threads if a.threads > 0 else max(1, os.cpu_count() or 1))
        fs = E.feature_sets(); t_all = time.time()
        units = [(n, fa, fb, s) for n in SPEC for fa, fb in folds for s in seeds]
        todo = [u for u in units if not os.path.exists(unit_file(u[0], u[1], u[3]))]
        print(f'{len(units)} units, {len(todo)} to do; labelled data up to {last.date()}', flush=True)
        for name, fa, fb, s in todo:
            t0 = time.time()
            try:
                o = run_unit(P, fs, name, fa, fb, s)
            except Exception:
                print(f'{name} {fa} seed {s} FAILED:\n{traceback.format_exc()}', flush=True); continue
            f = unit_file(name, fa, s); os.makedirs(os.path.dirname(f), exist_ok=True)
            o.to_pickle(f + '.tmp'); os.replace(f + '.tmp', f)
            print(f'{name:12s} fold {fa} seed {s}: {time.time() - t0:4.0f}s (total {(time.time() - t_all) / 60:5.1f} min)', flush=True)
    summarize(P, seeds)

if __name__ == '__main__':
    main()

r"""
overnight_experiments.py -- walk-forward evidence for the v2 choices (run on your computer, resumable)

  python overnight_experiments.py               all jobs in priority order (about 5-6 h on an 8-thread CPU)
  python overnight_experiments.py --hours 6     do not start a new job after 6 hours (finished jobs are kept)
  python overnight_experiments.py --summary     only rebuild the summary from what has finished
  python overnight_experiments.py --list        show the jobs and which ones are done
  python overnight_experiments.py --only mlp_   run only the jobs whose name starts with mlp_

Protocol (same as v1 / train_best.py): expanding window from 2020-03-02 (or a 2/4-year window where stated),
yearly test folds 2022, 2023, 2024 (validation: decides), 2025 and 2026-01..05 (holdout: confirms), plus one NEW fold
"post" = 2026-06-01 .. last labelled day, predicted by models trained up to 2026-05-29 (never used for any v1 choice).
Score = competition AR, bp/day, mean over days of the Top50 average of (2p-1) * (close-open)/open.
Each (job, fold, seed) is saved in results_v2\wf\ as soon as it finishes: stop at any time (Ctrl+C, shutdown) and
run again to continue. The v1 Transformer predictions in ..\transformer\results\wf_transformer_B.pkl (seeds 1-3,
same code and settings) are reused for the 2022..2026/05 folds, which saves about 1.5 hours.
Output: results_v2\summary_v2.txt / .csv
"""
import os, sys, time, argparse, traceback
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ens_lib as E

RES = os.path.join(HERE, 'results_v2'); WF = os.path.join(RES, 'wf'); os.makedirs(WF, exist_ok=True)
LOG = os.path.join(RES, 'overnight.log')
YEAR_FOLDS = [('2022-01-01', '2023-01-01'), ('2023-01-01', '2024-01-01'), ('2024-01-01', '2025-01-01'),
              ('2025-01-01', '2026-01-01'), ('2026-01-01', '2026-06-01')]
POST_START = '2026-06-01'
CUT = pd.Timestamp('2025-01-01')
V1_TF = os.path.join(E.TF_DIR, 'results', 'wf_transformer_B.pkl')

# job name -> (kind, feature set, window years, target);  JOBS = (name, seeds, folds) in priority order
SPEC = {'mlp_raw_B': ('mlp', 'B', None, 'raw'), 'mlp_res_B': ('mlp', 'B', None, 'resid'),
        'mlp_raw_xr': ('mlp', 'B+XR', None, 'raw'), 'mlp_res_xr': ('mlp', 'B+XR', None, 'resid'),
        'tf_raw_B': ('transformer', 'B', None, 'raw'), 'tf_res_B': ('transformer', 'B', None, 'resid'),
        'mlp_raw_w4': ('mlp', 'B', 4, 'raw'), 'mlp_raw_w2': ('mlp', 'B', 2, 'raw'), 'tf_raw_w4': ('transformer', 'B', 4, 'raw')}
JOBS = [('mlp_raw_B', range(1, 11), 'year+post'), ('mlp_res_B', range(1, 11), 'year+post'),        # ~5 min each
        ('mlp_raw_xr', range(1, 11), 'year+post'), ('mlp_res_xr', range(1, 11), 'year+post'),
        ('mlp_raw_w4', range(1, 11), 'late'), ('mlp_raw_w2', range(1, 11), 'year+post'),
        ('tf_raw_B', range(1, 4), 'post'),                    # ~30 min; its 2022..2026/05 folds = v1 file
        ('tf_res_B', range(1, 4), 'year+post'),               # ~2.5 h
        ('tf_raw_B', range(4, 6), 'year+post'),               # ~1.5 h
        ('tf_raw_w4', range(1, 4), 'late')]                   # ~1.2 h
LATE_FILL = {'mlp_raw_w4': 'mlp_raw_B', 'tf_raw_w4': 'tf_raw_B'}   # before 2025 a 4-year window = all data = same model

def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'; print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def folds_of(spec, post_end):
    post = [(POST_START, post_end)]
    return {'year+post': YEAR_FOLDS + post, 'post': post, 'late': YEAR_FOLDS[3:] + post}[spec]

def unit_file(name, a, s):
    return os.path.join(WF, name, f'{a}_seed{s}.pkl')

def run_unit(P, fs, name, a, b, s):
    kind, feats, window, target = SPEC[name]
    m = E.member(name, kind, window=window, feats=feats, target=target)
    a_, b_ = pd.Timestamp(a), pd.Timestamp(b); te = a_ - pd.Timedelta(days=1)
    bd = E.train_member(P, m, s, te, fs[feats], start=E.window_start(te, window))
    from tfmodel import predict
    return predict(bd, P, lambda d: (d.date >= a_) & (d.date < b_) & d.target_r.notna())[['date', 'stock_id', 'p']]

def load_member(name, seeds):
    """seed-average of the saved units per fold (Series indexed by date, stock_id) or None if a fold is missing.
    tf_raw_B seeds 1-3 on the yearly folds come from the v1 file when their units do not exist."""
    d = os.path.join(WF, name); by_fold = {}
    for f in (sorted(os.listdir(d)) if os.path.isdir(d) else []):
        if not f.endswith('.pkl'):
            continue
        a, s = f[:-4].split('_seed')
        if int(s) in seeds:
            by_fold.setdefault(a, []).append((1, os.path.join(d, f)))
    if name == 'tf_raw_B' and os.path.exists(V1_TF) and set(range(1, 4)) <= set(seeds):
        v1 = pd.read_pickle(V1_TF)[['date', 'stock_id', 'p']]
        for a, b in YEAR_FOLDS:
            units = [u for u in by_fold.get(a, []) if not any(f'_seed{k}.pkl' in u[1] for k in (1, 2, 3))]
            if len(units) == len(by_fold.get(a, [])):          # seeds 1-3 not run as units: use the v1 mean (weight 3)
                by_fold[a] = units + [(3, v1[(v1.date >= a) & (v1.date < b)])]
    if name in LATE_FILL:
        f = load_member(LATE_FILL[name], seeds if name.startswith('mlp') else range(1, 4))
        if f is None:
            return None
    expected = [a for a, _ in (YEAR_FOLDS[3:] if name in LATE_FILL else YEAR_FOLDS)] + [POST_START]
    parts = []
    for a in expected:                                        # every fold must be complete, else the member is not ready
        units = by_fold.get(a, [])
        if sum(w for w, _ in units) < len(seeds):
            return None
        fr = [(w, (u if isinstance(u, pd.DataFrame) else pd.read_pickle(u)).set_index(['date', 'stock_id']).p) for w, u in units]
        parts.append(sum(w * x for w, x in fr) / sum(w for w, _ in fr))
    s = pd.concat(parts)
    if name in LATE_FILL:
        s = pd.concat([f[f.index.get_level_values('date') < '2025-01-01'], s[s.index.get_level_values('date') >= '2025-01-01']])
    return s

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--hours', type=float, default=7.0); ap.add_argument('--summary', action='store_true')
    ap.add_argument('--list', action='store_true'); ap.add_argument('--only', default=''); ap.add_argument('--threads', type=int, default=0)
    a = ap.parse_args(); t_start = time.time()
    P = E.build_panel(E.DATA); last = P[P.target_r.notna()].date.max(); post_end = str((last + pd.Timedelta(days=1)).date())
    fs = E.feature_sets()
    units = [(name, fa, fb, s) for name, seeds, fspec in JOBS if name.startswith(a.only) for fa, fb in folds_of(fspec, post_end) for s in seeds]
    done = [u for u in units if os.path.exists(unit_file(u[0], u[1], u[3]))]
    if a.list:
        for u in units:
            print(('done ' if u in done else 'todo ') + f'{u[0]:11s} fold {u[1]}..{u[2]} seed {u[3]}')
        print(f'{len(done)}/{len(units)} units done'); return
    if not a.summary:
        import torch
        torch.set_num_threads(a.threads if a.threads > 0 else max(1, os.cpu_count() or 1))
        log(f'start: {len(units)} units, {len(done)} already done; labelled data up to {last.date()}; torch {torch.__version__}, {torch.get_num_threads()} threads')
        for name, fa, fb, s in units:
            f = unit_file(name, fa, s)
            if os.path.exists(f):
                continue
            if time.time() - t_start > a.hours * 3600:
                log(f'time budget of {a.hours} h reached: stopping before {name} {fa} seed {s} (run again later to continue)'); break
            t0 = time.time()
            try:
                o = run_unit(P, fs, name, fa, fb, s)
            except KeyboardInterrupt:
                log('interrupted'); raise
            except Exception:
                log(f'{name} {fa} seed {s} FAILED:\n{traceback.format_exc()}'); continue
            os.makedirs(os.path.dirname(f), exist_ok=True); o.to_pickle(f + '.tmp'); os.replace(f + '.tmp', f)
            log(f'{name:11s} fold {fa}..{fb} seed {s:2d}: {time.time() - t0:5.0f}s')
    summarize(P)

# ------------------------------------------------------------------ summary
def series(out, P, exdiv):
    q = out.reset_index().merge(P[['date', 'stock_id', 'top50', 'target_r', 'ex_today']], on=['date', 'stock_id'])
    q = q[q.target_r.notna()]
    pred = (q.score > 0).astype(int)
    if exdiv:
        pred = E.apply_exdiv(pred, q.ex_today)
    q['s'] = 2 * pred - 1; x = q[q.top50]
    S = (x.s * x.target_r).groupby(x.date).mean() * 1e4
    T = x.groupby('date').s.mean() * x.groupby('date').target_r.mean() * 1e4
    return S, T

def summarize(P):
    cache = {}
    def mem(name, seeds):
        k = (name, tuple(seeds))
        if k not in cache:
            cache[k] = load_member(name, list(seeds))
        return cache[k]
    S10, S5, S3 = range(1, 11), range(1, 6), range(1, 4)
    CONF = [  # label, [(job, seeds)], w_T grid (None = v1 rule on the raw members)
        ('v1-like: mlp_raw(3)+tf_raw(3), v1 rule', [('mlp_raw_B', S3), ('tf_raw_B', S3)], [None]),
        ('more seeds: mlp_raw(10)+tf_raw(5), v1 rule', [('mlp_raw_B', S10), ('tf_raw_B', S5)], [None]),
        ('windows: + mlp_w4 + tf_w4, v1 rule', [('mlp_raw_B', S10), ('mlp_raw_w4', S10), ('tf_raw_B', S5), ('tf_raw_w4', S3)], [None]),
        ('windows: + mlp_w4 + mlp_w2 + tf_w4, v1 rule', [('mlp_raw_B', S10), ('mlp_raw_w4', S10), ('mlp_raw_w2', S10), ('tf_raw_B', S5), ('tf_raw_w4', S3)], [None]),
        ('windows MLP: + mlp_w4, v1 rule', [('mlp_raw_B', S10), ('mlp_raw_w4', S10), ('tf_raw_B', S5)], [None]),
        ('windows MLP: + mlp_w4 + mlp_w2, v1 rule', [('mlp_raw_B', S10), ('mlp_raw_w4', S10), ('mlp_raw_w2', S10), ('tf_raw_B', S5)], [None]),
        ('mlp_raw_B(10) alone, v1 rule', [('mlp_raw_B', S10)], [None]),
        ('MLP only: mlp_raw_B + mlp_w4, v1 rule', [('mlp_raw_B', S10), ('mlp_raw_w4', S10)], [None]),
        ('MLP only: mlp_raw_B + mlp_w4 + mlp_w2, v1 rule', [('mlp_raw_B', S10), ('mlp_raw_w4', S10), ('mlp_raw_w2', S10)], [None]),
        ('mlp_raw_w4(10) alone, v1 rule', [('mlp_raw_w4', S10)], [None]),
        ('mlp_raw_w2(10) alone, v1 rule', [('mlp_raw_w2', S10)], [None]),
        ('v2 MLP only', [('mlp_raw_B', S10), ('mlp_raw_xr', S10), ('mlp_res_B', S10), ('mlp_res_xr', S10)], [0.0, 0.25, 0.5, 0.75, 1.0]),
        ('v2 default (= train_and_predict_v2)', [('mlp_raw_B', S10), ('mlp_raw_xr', S10), ('tf_raw_B', S5), ('mlp_res_B', S10), ('mlp_res_xr', S10)], [0.0, 0.25, 0.5, 0.75, 1.0]),
        ('v2 default + tf_res (--tf-resid)', [('mlp_raw_B', S10), ('mlp_raw_xr', S10), ('tf_raw_B', S5), ('mlp_res_B', S10), ('mlp_res_xr', S10), ('tf_res_B', S3)], [0.0, 0.25, 0.5, 0.75, 1.0]),
    ]
    rows, base, base_label = [], None, None
    for label, parts, grid in CONF:
        probs, members = {}, []
        for job, seeds in parts:
            p = mem(job, seeds)
            if p is None:
                break
            kind, feats, window, target = SPEC[job]
            probs[job] = p; members.append(E.member(job, kind, window=window, feats=feats, target=target))
        else:
            idx = None
            for p in probs.values():
                idx = p.index if idx is None else idx.intersection(p.index)
            probs = {k: v.reindex(idx) for k, v in probs.items()}
            for w in grid:
                if w is None:
                    raw_only = [m for m in members if m['target'] == 'raw']
                    out = E.combine(probs, raw_only)
                    variants = [('v1 rule', out, False), ('v1 rule + exdiv', out, True)]
                else:
                    out = E.combine(probs, members, w_t=w)
                    variants = [(f'w_T={w} + exdiv', out, True)] + ([(f'w_T={w} no exdiv', out, False)] if w == 0.5 else [])
                for vname, o, exd in variants:
                    S, T = series(o, P, exd)
                    if base is None:
                        base, base_label = S, f'{label} / {vname}'
                    r = {'config': label, 'rule': vname, 'val': S[S.index < CUT].mean(), 'hold': S[(S.index >= CUT) & (S.index < POST_START)].mean(),
                         'post': S[S.index >= POST_START].mean(), 'all': S.mean(), 'sd_day': S.std(), 'timing': T.mean(), 'xsec': (S - T).mean()}
                    for y in range(2022, 2027):
                        r[str(y)] = S[(S.index.year == y) & (S.index < POST_START)].mean()
                    d = (S - base).dropna()
                    for nm, sel in [('t_val', d.index < CUT), ('t_hold', (d.index >= CUT) & (d.index < POST_START)), ('t_post', d.index >= POST_START), ('t_all', d.index == d.index)]:
                        z = d[sel]; r[nm] = z.mean() / z.std() * np.sqrt(len(z)) if len(z) > 2 and z.std() > 0 else np.nan
                    rows.append(r)
    if not rows:
        log('summary: no complete configuration yet'); return
    T = pd.DataFrame(rows); T.to_csv(os.path.join(RES, 'summary_v2.csv'), index=False)
    pd.set_option('display.width', 300); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 500)
    txt = ['AR = competition score, bp/day, Top50. val = 2022-2024 (decides), hold = 2025-2026/05 (confirms),',
           'post = 2026/06 .. last labelled day (models frozen at 2026-05-29). timing / xsec = part of AR from the overall',
           f'up/down call / from picking stocks. t_* = paired t-stat of the daily difference vs the first row: {base_label}.',
           'Configurations whose jobs have not all finished are not shown yet.', '',
           T.round(2).to_string(index=False)]
    open(os.path.join(RES, 'summary_v2.txt'), 'w', encoding='utf-8').write('\n'.join(txt) + '\n')
    print('\n'.join(txt)); log('summary written: results_v2\\summary_v2.txt')

if __name__ == '__main__':
    main()

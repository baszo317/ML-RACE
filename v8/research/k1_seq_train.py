"""k1_seq_train: walk-forward training for the seventh round (PREREG_m.md): lstm_raw (market view), gru_res and lstm_res
(stock view).  B features (67), 20-day window, seeds 1-5, yearly folds 2022..2026-05 + post (trained to 2026-05-29), training
exactly as tf_raw_B (tfmodel.train_model with ens_lib.HP).  Writes ../cache/seq/<kind>_<target>/<fold>_seed<s>.pkl; resumable.
run: python k1_seq_train.py [--workers 2]   (needs torch; about 50 s per model on one core)"""
import os, sys, time, argparse, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); V8 = os.path.dirname(HERE); ROOT = os.path.dirname(V8)
for p in (HERE, os.path.join(ROOT, 'v5'), os.path.join(ROOT, 'transformer')):
    sys.path.insert(0, p)
import numpy as np, pandas as pd
import mlib as L
OUT = os.path.join(L.CACHE, 'seq')
FOLDS = [('2022-01-01', '2023-01-01'), ('2023-01-01', '2024-01-01'), ('2024-01-01', '2025-01-01'), ('2025-01-01', '2026-01-01'),
         ('2026-01-01', '2026-06-01'), ('2026-06-01', '2100-01-01')]
JOBS = [('lstm', 'raw'), ('rnn', 'resid'), ('lstm', 'resid')]
_G = {}

def path(kind, target, a, s):
    return os.path.join(OUT, f'{kind}_{target}', f'{a}_seed{s}.pkl')

def run_unit(u):
    kind, target, a, b, s = u
    f = path(kind, target, a, s)
    if os.path.exists(f):
        return u, 0.0, None, None
    t0 = time.time()
    try:
        import torch
        torch.set_num_threads(1)
        import rnn_lib, lstm_lib  # noqa: F401  (kind='rnn' = GRU, kind='lstm')
        import tfmodel as TM
        P, fs, E = _G['P'], _G['fs'], _G['E']
        te = pd.Timestamp(a) - pd.Timedelta(days=1)
        Q = P[P.date <= max(pd.Timestamp(b), te)]
        if target == 'resid':
            Q = Q.assign(target_r=Q.resid_r, y=Q.y_resid)
        bd = TM.train_model(kind, Q[Q.date <= te], list(fs['B']), lambda d: (d.date >= E.TRAIN_START) & (d.date <= te), 20, seed=s, hp=E.HP)
        a_, b_ = pd.Timestamp(a), pd.Timestamp(b)
        o = TM.predict(bd, Q, lambda d: (d.date >= a_) & (d.date < b_) & d.target_r.notna())[['date', 'stock_id', 'p']]
        os.makedirs(os.path.dirname(f), exist_ok=True); o.to_pickle(f + '.tmp'); os.replace(f + '.tmp', f)
        return u, time.time() - t0, None, bd['best_epoch']
    except Exception:
        return u, time.time() - t0, traceback.format_exc(), None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    import ens_lib as E
    _G['P'], _G['fs'], _G['E'] = L.panel(), E.feature_sets(), E
    units = [(k, t, fa, fb, s) for k, t in JOBS for fa, fb in FOLDS for s in range(1, 6)]
    todo = [u for u in units if not os.path.exists(path(u[0], u[1], u[2], u[4]))]
    print(f'{len(units)} units, {len(todo)} to do', flush=True)
    import multiprocessing as mp
    t0 = time.time(); eps = []
    if a.workers > 1 and 'fork' in mp.get_all_start_methods():
        pool = mp.get_context('fork').Pool(a.workers); it = pool.imap_unordered(run_unit, todo)
    else:
        pool = None; it = map(run_unit, todo)
    for i, (u, sec, err, ep) in enumerate(it, 1):
        if err:
            print('FAILED', u, err, flush=True)
        else:
            eps.append((f'{u[0]}_{u[1]}', ep))
        print(f'[{i}/{len(todo)}] {u} {sec:.0f}s, best epoch {ep}, elapsed {(time.time() - t0) / 60:.1f} min', flush=True)
    if pool is not None:
        pool.close(); pool.join()
    if eps:
        R = pd.DataFrame(eps, columns=['job', 'best_epoch'])
        print('best epoch counts:'); print(R.groupby('job').best_epoch.value_counts().sort_index().to_string())

if __name__ == '__main__':
    main()

"""c1_cnn_train: walk-forward training of the 1D-CNN market-view member cnn_raw_B (PREREG_m.md, fourth round).
B features (67), 20-day window, raw target (close > open), seeds 1-5, yearly folds 2022..2026-05 + post (trained to 2026-05-29),
training exactly as tf_raw_B (tfmodel.train_model with ens_lib.HP).  Writes ../cache/cnn/<fold>_seed<s>.pkl; resumable.
run: python c1_cnn_train.py [--workers 2]   (Windows: one process, roughly 30-60 minutes; needs torch)"""
import os, sys, time, argparse, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); V8 = os.path.dirname(HERE); ROOT = os.path.dirname(V8)
for p in (HERE, os.path.join(ROOT, 'v5'), os.path.join(ROOT, 'transformer')):
    sys.path.insert(0, p)
import numpy as np, pandas as pd
import mlib as L
OUT = os.path.join(L.CACHE, 'cnn')
FOLDS = [('2022-01-01', '2023-01-01'), ('2023-01-01', '2024-01-01'), ('2024-01-01', '2025-01-01'), ('2025-01-01', '2026-01-01'),
         ('2026-01-01', '2026-06-01'), ('2026-06-01', '2100-01-01')]
_G = {}

def path(a, s):
    return os.path.join(OUT, f'{a}_seed{s}.pkl')

def run_unit(u):
    a, b, s = u
    f = path(a, s)
    if os.path.exists(f):
        return u, 0.0, None, None
    t0 = time.time()
    try:
        import torch
        torch.set_num_threads(1)
        import cnn_lib  # noqa: F401  (adds kind='cnn' to tfmodel.make_model)
        import tfmodel as TM
        P, fs, E = _G['P'], _G['fs'], _G['E']
        te = pd.Timestamp(a) - pd.Timedelta(days=1)
        Q = P[P.date <= max(pd.Timestamp(b), te)]
        bd = TM.train_model('cnn', Q[Q.date <= te], list(fs['B']), lambda d: (d.date >= E.TRAIN_START) & (d.date <= te), 20, seed=s, hp=E.HP)
        a_, b_ = pd.Timestamp(a), pd.Timestamp(b)
        o = TM.predict(bd, Q, lambda d: (d.date >= a_) & (d.date < b_) & d.target_r.notna())[['date', 'stock_id', 'p']]
        os.makedirs(OUT, exist_ok=True); o.to_pickle(f + '.tmp'); os.replace(f + '.tmp', f)
        return u, time.time() - t0, None, bd['best_epoch']
    except Exception:
        return u, time.time() - t0, traceback.format_exc(), None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    import ens_lib as E
    _G['P'], _G['fs'], _G['E'] = L.panel(), E.feature_sets(), E
    units = [(fa, fb, s) for fa, fb in FOLDS for s in range(1, 6)]
    todo = [u for u in units if not os.path.exists(path(u[0], u[2]))]
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
            eps.append(ep)
        print(f'[{i}/{len(todo)}] {u} {sec:.0f}s, best epoch {ep}, elapsed {(time.time() - t0) / 60:.1f} min', flush=True)
    if pool is not None:
        pool.close(); pool.join()
    if eps:
        print('best epoch counts:', pd.Series(eps).value_counts().sort_index().to_dict())

if __name__ == '__main__':
    main()

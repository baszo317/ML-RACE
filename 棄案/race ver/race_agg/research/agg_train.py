r"""
agg_train.py -- race_agg training runner (research\PREREG_race_agg.md sections 2-3). D folds only.

  python agg_train.py --what all  [--workers 2] [--budget SECONDS]     every unit of B1 B2 C1 C2 C3 D1 D2 E1 E4 F1 and GM.*
  python agg_train.py --what B1,C3                                     only these
  python agg_train.py --status
  python agg_train.py --repro                                          retrain C.LU 2024 seed 1 and compare with the part-1 unit

A unit = (candidate, fold, seed) saved as cache\units_agg\<cand>\<fold>_seed<s>.npz (atomic, never redone). E4 needs its
selection file first (task E4sel, written by this runner). --budget stops handing out new units after that many seconds.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_k, '1')
import sys, json, time, argparse, multiprocessing as mp
import numpy as np, pandas as pd
import agglib as A
L = A.L

ORDER = ['GM.LU', 'GM.LN', 'GM.LP', 'GM.LC', 'GM.LM', 'E4sel', 'C1', 'C2', 'D1', 'E1', 'B1', 'B2', 'C3', 'D2', 'F1', 'E4']
ORDER_V = ['C.LN', 'C.LM', 'C.LP', 'D1', 'E1', 'C1'] + [c for c in ORDER if c not in ('D1', 'E1', 'C1')]   # slow units first
PERIOD = 'D'
_P = _X = _F = None


def _init():
    global _P, _X, _F
    if _P is None:
        _P = A.load_panel(); _F = A.feats_C(); _X = _P[_F].to_numpy(np.float32)


def tasks(what, period=None):
    period = period or PERIOD
    out = []
    for c in (ORDER_V if period == 'V' else ORDER):
        if what != 'all' and c not in what:
            continue
        for fa, fb in A.FOLDS_OF[period]:
            if c in A.MEMBER_V_TRAINED:
                for s in L.SEEDS[c.split('.')[1]]:
                    out.append((c, fa, fb, s))
            elif c.startswith('GM.'):
                for s in L.SEEDS[c.split('.')[1]]:
                    out.append((c, fa, fb, s))
            elif c == 'E4sel':
                out.append((c, fa, fb, 0))
            else:
                for s in A.SEEDS[c]:
                    out.append((c, fa, fb, s))
    return out


def done(t):
    c, fa, fb, s = t
    if c == 'E4sel':
        import agg_learners as G
        return os.path.exists(G.sel_file(fa))
    return os.path.exists(A.unit_file(c, fa, s))


def run_one(t):
    import agg_learners as G
    _init()
    c, fa, fb, s = t
    if done(t):
        return t, 'exists', 0.0, None
    t0 = time.time()
    tr, te = A.masks(_P, fa, fb)
    d = _P['date'].to_numpy(); sid = _P['stock_id'].to_numpy()
    if c == 'E4sel':
        sel = G.select_E4(_P, _X, _F, tr, fa)
        f = G.sel_file(fa); os.makedirs(os.path.dirname(f), exist_ok=True)
        with open(f + '.tmp', 'w', encoding='utf-8') as fh:
            json.dump(sel, fh, indent=1)
        os.replace(f + '.tmp', f)
        return t, 'done', time.time() - t0, dict(n_keep=sel['n_keep'], base_ic=round(sel['base_ic'], 4))
    if c.startswith('GM.'):
        pred, meta, rows = G.fit_GM(_P, _X, _F, tr, te, s, fa, c.split('.')[1])
    elif c in A.MEMBER_V_TRAINED:
        import learners as LR
        pred, meta = LR.fit_predict(c.split('.')[1], _P, _F, tr, te, s)
        rows = te
    else:
        pred, meta = G.FIT[c](_P, _X, _F, tr, te, s, fa)
        rows = te
    meta = dict(meta or {})
    meta.update(cand=c, fold=fa, seed=s, train_last=str(pd.Timestamp(d[tr].max()).date()), n_train=int(tr.sum()),
                n_pred=int(rows.sum()), seconds=round(time.time() - t0, 2))
    L.save_unit(A.unit_file(c, fa, s), d[rows], sid[rows], np.asarray(pred, np.float64), meta)
    short = {k: meta[k] for k in ('best_iter', 'T', 'leaf', 'leaves', 'n_keep', 'unique_days', 'best_epoch') if k in meta}
    return t, 'done', time.time() - t0, short


def _work(t):
    try:
        return run_one(t)
    except Exception as e:
        import traceback
        return t, 'ERROR ' + repr(e) + ' ' + traceback.format_exc()[-800:], 0.0, None


def run(what, workers, budget, reverse=False):
    T = [t for t in tasks(what) if not done(t)]
    if reverse:                                  # a second machine works from the other end of the list
        T = T[::-1]
    # E4 units need the selection of the same fold: put E4sel first, E4 last
    A.log(f'race_agg train [{PERIOD}]: {len(T)} units to do ({what}), workers {workers}, budget {budget}')
    t_start = time.time()
    if not T:
        return
    import agg_learners as G
    ctx = mp.get_context('spawn' if sys.platform == 'win32' else 'fork')     # Windows has no fork
    with ctx.Pool(workers, maxtasksperchild=4) as pool:
        pending = list(T); running = []
        while pending or running:
            while len(running) < workers and pending and (budget is None or time.time() - t_start < budget):
                ready = [i for i, t in enumerate(pending) if t[0] != 'E4' or os.path.exists(G.sel_file(t[1]))]
                if not ready:
                    break
                t = pending.pop(ready[0])
                running.append((t, pool.apply_async(_work, (t,))))
            if not running:
                if pending and (budget is None or time.time() - t_start < budget):
                    A.log(f'blocked: {len(pending)} units wait for a missing E4 selection')
                break
            time.sleep(0.5)
            still = []
            for t, r in running:
                if r.ready():
                    tt, status, sec, short = r.get()
                    A.log(f'{tt[0]} {tt[1]} seed{tt[3]}: {status} ({sec:.1f} s) {short if short else ""}')
                else:
                    still.append((t, r))
            running = still
    left = [t for t in tasks(what) if not done(t)]
    A.log(f'race_agg train [{PERIOD}]: {len(left)} units left ({what})')


def status():
    T = tasks('all')
    by = {}
    for t in T:
        by.setdefault(t[0], [0, 0])
        by[t[0]][0] += 1; by[t[0]][1] += int(done(t))
    for k in (ORDER_V if PERIOD == 'V' else ORDER):
        if k in by:
            print(f'{k:<6} {by[k][1]:>3} / {by[k][0]}')


def repro():
    import learners as LR
    _init()
    fa, fb = A.D_FOLDS[2]
    tr, te = A.masks(_P, fa, fb)
    t0 = time.time()
    pred, meta = LR.fit_LU(_P, _F, tr, te, 1)
    s0, m0 = L.load_npz(L.unit_file('C.LU', fa, 1))
    d = _P['date'].to_numpy(); sid = _P['stock_id'].to_numpy()
    s1 = pd.Series(pred, index=pd.MultiIndex.from_arrays([pd.to_datetime(d[te]), sid[te].astype(str)], names=['date', 'stock_id']))
    diff = (s1.sort_index() - s0.sort_index()).abs()
    A.log(f'repro C.LU {fa} seed 1: best_iter {meta["best_iter"]} (part 1: {m0["best_iter"]}), max |diff| {diff.max():.3e}, '
          f'n {len(s1)} / {len(s0)}, {time.time() - t0:.1f} s; lightgbm {__import__("lightgbm").__version__}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--what', default='all')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--budget', type=float, default=None)
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--repro', action='store_true')
    ap.add_argument('--period', default='D', choices=['D', 'V'])
    ap.add_argument('--reverse', action='store_true')
    a = ap.parse_args()
    PERIOD = a.period
    if PERIOD == 'V':
        assert os.path.exists(A.PREREG_V), 'the V folds need research\\PREREG_race_agg_V.md'
    if a.status:
        status()
    elif a.repro:
        repro()
    else:
        what = 'all' if a.what == 'all' else a.what.split(',')
        run(what, a.workers, a.budget, a.reverse)

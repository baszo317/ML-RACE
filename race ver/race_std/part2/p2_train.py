r"""
p2_train.py -- RACE-STD v1 part 2, F5 (part2\PREREG_part2.md): C.LU with quarterly (QR) or monthly (MR) refits on D.
Same learner as part 1 (LU, seeds 1-3, early stopping on the last 120 training days, then refit), only the refit days differ.

  python p2_train.py --freq Q [--budget 150] [--hard 170] [--workers 2]
  python p2_train.py --freq M ...
  python p2_train.py --freq M --status
Units: cache\units_p2\C.LU@<Q|M>\<refit day>_seed<s>.npz (atomic; finished ones are never redone).
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_k, '1')
import sys, time, argparse, multiprocessing as mp
import numpy as np, pandas as pd
import p2lib as Q
L = Q.L

_P = None
_F = None
SEEDS = (1, 2, 3)


def refit_days(freq):
    step = {'Q': '3MS', 'M': 'MS'}[freq]
    starts = list(pd.date_range(Q.D_A, '2024-12-31', freq=step))
    ends = starts[1:] + [pd.Timestamp(Q.D_B)]
    return [(a.strftime('%Y-%m-%d'), b.strftime('%Y-%m-%d')) for a, b in zip(starts, ends)]


def unit_file(freq, fa, seed):
    return os.path.join(Q.UNITS2, f'C.LU@{freq}', f'{fa}_seed{seed}.npz')


def _init():
    global _P, _F
    if _P is None:
        f = L.PANEL
        loc = os.environ.get('RACE_LOCAL_PANEL', '')
        if loc and os.path.exists(loc) and os.path.getsize(loc) == os.path.getsize(L.PANEL):
            f = loc
        _P = pd.read_pickle(f)
        _P['date'] = pd.to_datetime(_P.date).astype('datetime64[ns]'); _P['stock_id'] = _P.stock_id.astype(str)
    if _F is None:
        _F = L.features()


def run_one(t):
    import learners
    _init()
    freq, fa, fb, s = t
    f = unit_file(freq, fa, s)
    if os.path.exists(f):
        return t, 'exists', 0.0
    d = _P['date'].to_numpy()
    lab = _P['target_r'].notna().to_numpy() & _P['resid_r'].notna().to_numpy()
    tr = lab & (d >= np.datetime64(L.TRAIN_START)) & (d < np.datetime64(pd.Timestamp(fa)))
    te = (d >= np.datetime64(pd.Timestamp(fa))) & (d < np.datetime64(pd.Timestamp(fb)))
    assert d[tr].max() < np.datetime64(pd.Timestamp(fa))
    t0 = time.time()
    pred, meta = learners.fit_predict('LU', _P, L.feature_list('C', _F), tr, te, s)
    meta.update(unit=f'C.LU@{freq}', fold=fa, seed=s, train_last=str(pd.Timestamp(d[tr].max()).date()))
    L.save_unit(f, d[te], _P['stock_id'].to_numpy()[te], pred, meta)
    return t, 'done', time.time() - t0


def _work(t):
    try:
        return run_one(t)
    except Exception as e:
        import traceback
        return t, 'ERROR ' + repr(e) + ' ' + traceback.format_exc()[-1200:], 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--freq', required=True, choices=['Q', 'M'])
    ap.add_argument('--budget', type=float, default=1e9)
    ap.add_argument('--hard', type=float, default=1e9)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--status', action='store_true')
    a = ap.parse_args()
    os.makedirs(Q.RES2, exist_ok=True)
    log = L.log_to(os.path.join(Q.RES2, f'p2_train_{a.freq}.log'))
    T = [(a.freq, fa, fb, s) for fa, fb in refit_days(a.freq) for s in SEEDS]
    todo = [t for t in T if not os.path.exists(unit_file(*[t[0], t[1], t[3]]))]
    log(f'{a.freq}: {len(T)} units, {len(T) - len(todo)} done, {len(todo)} to do')
    if a.status or not todo:
        if not todo:
            log('ALL DONE')
        return
    # expected seconds ~ training rows (C.LU at the 2024 fold: about 94k rows, 50-70 s with 2 workers)
    _init()
    d = _P['date']
    lab = _P.target_r.notna() & (d >= L.TRAIN_START)
    est = {t: 0.0008 * int((lab & (d < pd.Timestamp(t[1]))).sum()) + 5 for t in todo}
    todo.sort(key=lambda t: -est[t])
    t0 = time.time()
    ctx = mp.get_context('fork' if sys.platform != 'win32' else 'spawn')
    with ctx.Pool(a.workers, initializer=_init) as pool:
        pending, running = list(todo), []
        while pending or running:
            while pending and len(running) < a.workers and time.time() - t0 < a.budget:
                el = time.time() - t0
                fit = [t for t in pending if el + est[t] < a.hard] or ([pending[0]] if el < 5 else [])
                if not fit:
                    break
                t = fit[0]; pending.remove(t)
                running.append(pool.apply_async(_work, (t,)))
            if not running:
                break
            time.sleep(0.2)
            still = []
            for r in running:
                if r.ready():
                    t, st, sec = r.get()
                    log(f'{t[0]} {t[1]} seed{t[3]}: {st[:300]} ({sec:.1f} s)')
                else:
                    still.append(r)
            running = still
            if time.time() - t0 >= a.budget and not running:
                break
    left = [t for t in T if not os.path.exists(unit_file(t[0], t[1], t[3]))]
    log(f'{a.freq}: {len(T) - len(left)} / {len(T)} units done' + ('  ALL DONE' if not left else ''))


if __name__ == '__main__':
    main()

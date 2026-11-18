r"""
train.py -- RACE-STD v1 training runner (research\PREREG_race_std.md sections 1.3, 2.3, 2.4).

  python train.py --stage 1   [--budget 120] [--workers 2] [--test]   grid + U-native units on the D folds
  python train.py --stage 1b  ...                                      ablation units ALLm*.L* on the D folds (needs
                                                                       results\stage1_grid.json)
  python train.py --stage 2   ...                                      the finalists' units on the V folds (needs
                                                                       results\LOCK.json with matching hashes)
  python train.py --stage 1 --status                                   only print what is done / missing

A unit = (feature set x learner) for one fold and one seed, saved as cache\units\<unit>\<fold>_seed<s>.npz (atomic).
Finished units are never redone: stop any time and run again. --budget: stop handing out new units after this many
seconds (the Linux VM of the Claude app ends a command after 3 minutes). Stage 1 / 1b refuse the V folds.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_k, '1')
import sys, json, time, argparse, multiprocessing as mp
import numpy as np, pandas as pd
import rslib as L

_P = None
_F = None


def load_panel():
    """cache/panel.pkl; RACE_LOCAL_PANEL may name a byte-identical local copy (faster in the VM)"""
    f = L.PANEL
    loc = os.environ.get('RACE_LOCAL_PANEL', '')
    if loc and os.path.exists(loc) and os.path.getsize(loc) == os.path.getsize(L.PANEL):
        f = loc
    P = pd.read_pickle(f)
    P['date'] = pd.to_datetime(P.date).astype('datetime64[ns]')
    P['stock_id'] = P.stock_id.astype(str)
    return P


def _init(panel_path=None):
    global _P, _F
    if _P is None:
        _P = load_panel()
    if _F is None:
        _F = L.features()


def tasks_for(stage, test=False):
    if stage == '1':
        units = sorted(set(L.grid_units() + L.U_NATIVE_MLP + L.U_NATIVE_LGB))
        folds = L.D_FOLDS
    elif stage == '1b':
        g = json.load(open(os.path.join(L.RESULTS, f'stage1_grid{"_test" if test else ""}.json'), encoding='utf-8'))
        units = [f'ALLm{x}.{g["lstar"]}' for x in L.GROUPS]
        folds = L.D_FOLDS
    elif stage == '2':
        lock = check_lock(test)
        units = list(lock['finalist_units'])
        folds = L.V_FOLDS
    else:
        raise ValueError(stage)
    out = []
    for u in units:
        fk, ln = L.unit_parts(u)
        for fa, fb in folds:
            for s in L.SEEDS[ln][:1] if test else L.SEEDS[ln]:
                out.append((u, fa, fb, s))
    return out


def check_lock(test=False):
    f = os.path.join(L.RESULTS, f'LOCK{"_test" if test else ""}.json')
    if not os.path.exists(f):
        sys.exit(f'no {f}: run freeze.py first (stage 2 trains only the locked finalists)')
    lock = json.load(open(f, encoding='utf-8'))
    now = L.code_hashes()
    bad = [k for k, h in lock['code'].items() if now.get(k) != h] + [k for k in now if k not in lock['code']]
    if bad:
        sys.exit(f'code changed after the LOCK: {bad} -- the protocol does not allow that (PREREG 2.4)')
    if L.sha256(L.PREREG) != lock['prereg_sha256']:
        sys.exit('PREREG_race_std.md changed after the LOCK')
    snap = json.load(open(os.path.join(L.RESULTS, 'snapshot.json'), encoding='utf-8'))
    if snap['data_sha256'] != lock['data_sha256']:
        sys.exit('data snapshot changed after the LOCK')
    return lock


def unit_done(t, test=False):
    u, fa, fb, s = t
    f = L.unit_file(u, fa, s, test)
    return os.path.exists(f), f


def run_one(t, test=False):
    import learners
    _init()
    u, fa, fb, s = t
    done, f = unit_done(t, test)
    if done:
        return t, 'exists', 0.0
    fk, ln = L.unit_parts(u)
    feats = L.feature_list(fk, _F)
    d = _P['date'].to_numpy()
    lab = _P['target_r'].notna().to_numpy() & _P['resid_r'].notna().to_numpy()
    tr = lab & (d >= np.datetime64(L.TRAIN_START)) & (d < np.datetime64(pd.Timestamp(fa)))
    te = (d >= np.datetime64(pd.Timestamp(fa))) & (d < np.datetime64(pd.Timestamp(fb)))
    if test:                                   # quick test: last 250 training days, first 20 test days
        days = np.unique(d[tr]); tr = tr & (d >= days[-250])
        tdays = np.unique(d[te]); te = te & (d <= tdays[min(19, len(tdays) - 1)])
    assert d[tr].max() < np.datetime64(pd.Timestamp(fa)), 'training rows reach the fold'
    t0 = time.time()
    pred, meta = learners.fit_predict(ln, _P, feats, tr, te, s, test)
    meta.update(unit=u, fold=fa, seed=s, train_last=str(pd.Timestamp(d[tr].max()).date()), test=bool(test))
    L.save_unit(f, d[te], _P['stock_id'].to_numpy()[te], pred, meta)
    return t, 'done', time.time() - t0


def _work(args):
    t, test = args
    try:
        return run_one(t, test)
    except Exception as e:
        import traceback
        return t, 'ERROR ' + repr(e) + ' ' + traceback.format_exc()[-1500:], 0.0


FOLD_SCALE = {'2022-01-01': 0.5, '2023-01-01': 0.75, '2024-01-01': 1.0, '2025-01-01': 1.3, '2026-01-01': 1.55, '2026-06-01': 1.65}
PER_FEAT = {'LP': 0.32, 'LU': 0.22, 'LC': 0.28, 'LN': 0.12, 'LM': 0.08}   # seconds per feature at the 2024 fold (VM)


def estimate(t, test=False):
    """expected seconds of a unit: the slowest finished seed / fold of the same unit (scaled), else a per-feature guess"""
    if test:
        return 5.0
    u, fa, fb, s = t
    fk, ln = L.unit_parts(u)
    best = None
    for fo, _ in L.FOLDS:
        for sd in L.SEEDS[ln]:
            f = L.unit_file(u, fo, sd)
            if os.path.exists(f):
                try:
                    sec = L.load_npz(f)[1].get('seconds')
                except Exception:
                    sec = None
                if sec:
                    v = float(sec) * FOLD_SCALE[fa] / FOLD_SCALE[fo]
                    best = v if best is None else max(best, v)
    if best is not None:
        return 1.2 * best
    return PER_FEAT[ln] * len(L.feature_list(fk)) * FOLD_SCALE[fa] + 5.0


def order_cost(t):
    """rough cost order (expensive first, so that the cheap ones fill the end of a budget)"""
    u, fa, fb, s = t
    fk, ln = L.unit_parts(u)
    nf = len(L.feature_list(fk))
    base = {'LP': 3.0, 'LU': 2.0, 'LC': 1.5, 'LN': 1.0, 'LM': 0.6}[ln]
    return -base * nf * (1 + L.FOLDS.index((fa, fb)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['1', '1b', '2'])
    ap.add_argument('--budget', type=float, default=1e9)
    ap.add_argument('--hard', type=float, default=1e9, help='do not start a unit expected to end after this many seconds')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--only', default='', help='comma-separated units (debugging)')
    ap.add_argument('--threads', type=int, default=1, help='threads per unit (LightGBM col-wise / HistGBR; same results)')
    a = ap.parse_args()
    os.environ['RACE_THREADS'] = str(a.threads)
    log = L.log_to(os.path.join(L.RESULTS, f'train_stage{a.stage}{"_test" if a.test else ""}.log'))
    T = tasks_for(a.stage, a.test)
    if a.only:
        keep = set(a.only.split(','))
        T = [t for t in T if t[0] in keep]
    todo = [t for t in T if not unit_done(t, a.test)[0]]
    log(f'stage {a.stage}{" TEST" if a.test else ""}: {len(T)} units, {len(T) - len(todo)} done, {len(todo)} to do')
    if a.status or not todo:
        if not todo:
            log('ALL DONE')
        return
    todo.sort(key=order_cost)
    t0 = time.time()
    if a.workers <= 1:
        for t in todo:
            if time.time() - t0 > a.budget:
                break
            r = _work((t, a.test)); log(f'{r[0][0]} {r[0][1]} seed{r[0][3]}: {r[1][:400]} ({r[2]:.1f} s)')
    else:
        ctx = mp.get_context('fork' if sys.platform != 'win32' else 'spawn')
        if sys.platform != 'win32':
            _init()                              # loaded once, shared with the forked workers
        with ctx.Pool(a.workers, initializer=_init) as pool:
            pending = list(todo)
            est = {t: estimate(t, a.test) for t in pending}
            meas = {}
            running = []
            while pending or running:
                while pending and len(running) < a.workers and time.time() - t0 < a.budget:
                    el = time.time() - t0
                    fit = [t for t in pending if el + est[t] < a.hard] or ([pending[0]] if el < 5 else [])
                    if not fit:
                        break
                    t = fit[0]; pending.remove(t)
                    running.append(pool.apply_async(_work, ((t, a.test),)))
                if not running:
                    break
                time.sleep(0.2)
                still = []
                for r in running:
                    if r.ready():
                        t, st, sec = r.get()
                        log(f'{t[0]} {t[1]} seed{t[3]}: {st[:400]} ({sec:.1f} s)')
                        if sec > 0:
                            meas[t[0]] = max(meas.get(t[0], 0.0), sec / FOLD_SCALE[t[1]])
                            for t2 in pending:
                                if t2[0] == t[0]:
                                    est[t2] = 1.2 * meas[t[0]] * FOLD_SCALE[t2[1]]
                    else:
                        still.append(r)
                running = still
                if time.time() - t0 >= a.budget and not running:
                    break
    left = [t for t in T if not unit_done(t, a.test)[0]]
    log(f'stage {a.stage}: {len(T) - len(left)} / {len(T)} units done' + ('  ALL DONE' if not left else ''))


if __name__ == '__main__':
    main()

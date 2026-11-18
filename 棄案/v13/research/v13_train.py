r"""
v13_train.py -- walk-forward training of the three v13 stock-view candidates (README_v13.md):
  lstm   LSTM-R   LSTM on 60 days of raw price / volume / flow channels            (seqlib.py, PREREG_v13.md)
  tcn    CNN-R    dilated 1D CNN on the same raw sequences                          (seqlib.py, PREREG_v13.md)
  floam  F-LOAM   2x50 LSTM on 20 days of the B features -> h_t + B -> LightGBM tuned by Optuna on the last 120 days
                  of the window (floamlib.py, PREREG_v13b.md)
label = beats the same-day average of the 100 stocks (y_resid, weight min(|excess return|, 5%)); 6 yearly folds
(2022, 2023, 2024, 2025, 2026-01..05, post 2026-06..), window 2020-03-02 .. the day before the fold, seeds 1-5
-> 90 units, each saved as ..\cache\seq_raw\<kind>\<fold>_seed<s>.npz (date, sid, p, meta; F-LOAM also p_fixed, p_lstm).
A finished unit is never redone: stop any time (Ctrl+C or close the window), start again, it continues.
Runs at below-normal priority and keeps Windows awake while it runs (closing the lid still sleeps the laptop).

usage (in v13\research):  python v13_train.py [--kinds lstm,tcn,floam] [--workers 2] [--threads N] [--seeds 1,2,3,4,5]
                                              [--folds 2022-01-01,...] [--test] [--rebuild-panel]
  --test   1 seed, 1 epoch, the last 20,000 rows of each window, 3 Optuna trials -> ..\cache\seq_raw_test\ (a few minutes)
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, time, json, argparse, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import seqlib as S

_G = {}
NAMES = {'lstm': 'LSTM-R', 'tcn': 'CNN-R', 'floam': 'F-LOAM'}


def log(msg, test=False):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)
    os.makedirs(S.RESULTS, exist_ok=True)
    with open(os.path.join(S.RESULTS, 'v13_train_test.log' if test else 'v13_train.log'), 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def low_priority():
    try:
        if os.name == 'nt':
            import ctypes
            k = ctypes.windll.kernel32
            k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)          # BELOW_NORMAL_PRIORITY_CLASS
        else:
            os.nice(10)
    except Exception:
        pass


def keep_awake(on):
    if os.name != 'nt':
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except Exception:
        pass


def _init(threads, kinds):
    low_priority()
    import torch
    torch.set_num_threads(max(1, threads))
    _G['threads'] = max(1, threads)
    if {'lstm', 'tcn'} & set(kinds):
        _G['P'] = S.load_panel()
        _G['T'] = S.torch_parts()
    if 'floam' in kinds:
        import floamlib
        _G['PB'] = floamlib.load_b_panel()


def run_unit(u):
    kind, fa, fb, seed, test = u
    f = S.unit_file(kind, fa, seed, test)
    if os.path.exists(f):
        return {'unit': u, 'skipped': True}
    t0 = time.time()
    try:
        if kind == 'floam':
            import floamlib
            out, meta = floamlib.run_unit(_G['PB'], fa, fb, seed, test=test, threads=_G['threads'])
            floamlib.save_unit(f, out, meta)
            return {'unit': u, 'meta': meta}
        P, T = _G['P'], _G['T']
        te = pd.Timestamp(fa) - pd.Timedelta(days=1); a_, b_ = pd.Timestamp(fa), pd.Timestamp(fb)
        Q = P[P.date <= max(b_, te)]
        Q = Q.assign(target_r=Q.resid_r, y=Q.y_resid)                       # 'resid' label (= ens_lib.target_panel)
        hp = dict(epochs=1, patience=1) if test else None
        bd = T['train'](kind, Q[Q.date <= te], lambda d: (d.date >= S.TRAIN_START) & (d.date <= te), seed, hp=hp,
                        max_rows=20000 if test else None)
        t1 = time.time()
        o = T['predict'](bd, Q, lambda d: (d.date >= a_) & (d.date < b_) & d.target_r.notna())
        meta = dict(kind=kind, fold=fa, seed=seed, best_epoch=bd['best_epoch'], rows=bd['rows'], val_rows=bd['val_rows'],
                    n_params=bd['n_params'], hp=bd['hp'], pred_rows=int(len(o)), train_sec=round(t1 - t0, 1),
                    pred_sec=round(time.time() - t1, 1), threads=int(T['torch'].get_num_threads()),
                    torch=T['torch'].__version__, pandas=pd.__version__, test=test, finished=time.strftime('%Y-%m-%d %H:%M:%S'))
        S.save_unit(f, o, meta)
        return {'unit': u, 'meta': meta}
    except Exception:
        return {'unit': u, 'error': traceback.format_exc()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kinds', default='lstm,tcn,floam')
    ap.add_argument('--workers', type=int, default=2, help='units trained at the same time')
    ap.add_argument('--threads', type=int, default=0, help='torch / LightGBM threads per worker (default: cores / workers)')
    ap.add_argument('--seeds', default='1,2,3,4,5')
    ap.add_argument('--folds', default='', help='comma separated fold starts, default all six')
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--rebuild-panel', action='store_true')
    a = ap.parse_args()
    low_priority(); keep_awake(True)
    kinds = [k for k in a.kinds.split(',') if k]
    bad_k = [k for k in kinds if k not in NAMES]
    if bad_k:
        sys.exit(f'unknown kind(s) {bad_k}; use lstm, tcn, floam')
    missing = []
    for mod in ['torch'] + (['lightgbm', 'optuna'] if 'floam' in kinds else []):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        sys.exit(f'missing: {" ".join(missing)}  ->  python -m pip install {" ".join(missing)}')
    import torch
    cores = os.cpu_count() or 2
    phys = max(1, cores // 2) if cores >= 4 else cores                     # hyper-threading laptops: physical cores
    threads = a.threads or max(1, phys // max(1, a.workers))
    t0 = time.time()
    if {'lstm', 'tcn'} & set(kinds):
        P = S.load_panel(rebuild=a.rebuild_panel)                           # builds ..\cache\raw_panel.pkl once
        msg = f'raw panel {len(P):,} rows, {P.date.min():%Y-%m-%d} .. {P.date.max():%Y-%m-%d}; '
    else:
        msg = ''
    log(f'v13_train.py start: kinds {",".join(kinds)}; {msg}torch {torch.__version__}, pandas {pd.__version__}, {cores} logical cpus, '
        f'workers {a.workers} x {threads} threads' + (' [TEST]' if a.test else ''), a.test)
    seeds = [1] if a.test else [int(s) for s in a.seeds.split(',') if s]
    folds = [fb for fb in S.FOLDS if not a.folds or fb[0] in a.folds.split(',')]
    units = [(k, fa, fb, s, a.test) for k in kinds for fa, fb in folds for s in seeds]
    todo = [u for u in units if not os.path.exists(S.unit_file(u[0], u[1], u[3], a.test))]
    todo = sorted(todo, key=lambda u: (u[1], u[0] != 'tcn'), reverse=True)  # big folds first: the run ends sooner
    log(f'{len(units)} units ({", ".join(f"{NAMES[k]} {sum(1 for u in units if u[0] == k)}" for k in kinds)}), {len(todo)} to do', a.test)
    if not todo:
        keep_awake(False); return 0
    bad = 0; done = 0; t0 = time.time()
    try:
        if a.workers <= 1:
            _init(threads, kinds)
            for r in (run_unit(u) for u in todo):
                done += 1; bad += _report(r, done, len(todo), t0, a.test)
        else:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'),
                                     initializer=_init, initargs=(threads, kinds)) as ex:
                futs = [ex.submit(run_unit, u) for u in todo]
                for fu in as_completed(futs):
                    done += 1; bad += _report(fu.result(), done, len(todo), t0, a.test)
    except KeyboardInterrupt:
        log('stopped by the user; finished units are kept, start again to continue', a.test); keep_awake(False); return 1
    keep_awake(False)
    left = [u for u in units if not os.path.exists(S.unit_file(u[0], u[1], u[3], a.test))]
    log(f'v13_train.py end: {len(units) - len(left)} of {len(units)} units done, {(time.time() - t0) / 60:.1f} min this run', a.test)
    return 0 if not left and not bad else 1


def _report(r, i, n, t0, test):
    u = r['unit']; name = f'{NAMES[u[0]]} {u[1]} seed {u[3]}'
    if r.get('error'):
        log(f'[{i}/{n}] {name} FAILED\n{r["error"]}', test); return 1
    if r.get('skipped'):
        return 0
    m = r['meta']
    extra = ''
    if u[0] == 'floam':
        bp = m['best_params']
        extra = (f'; Optuna: leaves {bp["num_leaves"]}, lr {bp["learning_rate"]:.3f}, {m["best_rounds"]} rounds '
                 f'(LSTM {m["lstm_sec"] / 60:.1f} min, LightGBM {m["lgb_sec"] / 60:.1f} min)')
    log(f'[{i}/{n}] {name}: best epoch {m["best_epoch"]}, {m["rows"]:,} training rows, train {m["train_sec"] / 60:.1f} min'
        f'{extra}; elapsed {(time.time() - t0) / 60:.1f} min', test)
    return 0


if __name__ == '__main__':
    sys.exit(main())

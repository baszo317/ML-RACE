r"""
svm_exact.py -- exact RBF SVM (scikit-learn SVC = libsvm) as an extra stock-view member, in the same walk-forward as
lgb_res (reg_svm\research\PREREG_svm.md; reg_svm\README_reg_svm.md section 1).  Research only: no daily file is touched.

Member
  label     beats the same-day average of the 100 stocks (y_resid)                              (= lgb_res)
  features  B, 67 columns (transformer\final_feature_lists.json, B_competition_only)            (= lgb_res)
  weights   min(|excess return|, 5%), mean 1 (SVC multiplies C by them)                          (= lgb_res)
  scaling   winsorize 1% / 99%, z-score, missing -> 0, fitted on the training window   (= transformer\tfmodel.Scaler)
  kernel    RBF exp(-gamma |x - x'|^2), gamma = g / 67
Folds: 2022, 2023, 2024, 2025, 2026-01..05, post 2026-06..; window 2020-03-02 .. the day before the fold (as every member).
Per fold
  1. tuning, inside the window only: 20,000 random rows of the window without its last 120 trading days; every C of the
     grid (gamma fixed) is fitted on them and scored on the last 120 days with the SVM's own loss (weighted hinge loss,
     as early stopping uses each member's own loss); the smallest wins (the daily rank IC is logged for information)
  2. full fit on ALL rows of the window with the chosen C (PREREG_svm.md 2.3) -- the slow step
  3. decision values for the fold's days -> ..\cache\svm_exact\<fold>.npz (numpy only: date, sid, p; any pandas reads it)
A finished fold is not redone: stop it (Ctrl+C or close the window), start it again, it continues with the missing folds.
Runs at below-normal priority and asks Windows not to go to sleep while it runs (closing the lid still sleeps).
usage (in reg_svm\research):  python svm_exact.py [--workers 2] [--cache-mb 800] [--folds 2022-01-01,...] [--test]
  --test: 3,000 rows per fold and 1,500 for tuning, writes ..\cache\svm_exact_test\ (a 1-2 minute check that everything runs)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, time, json, argparse, traceback
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); RS = os.path.dirname(HERE)
ROOT = os.environ.get('MLRACE_ROOT', os.path.dirname(RS))
CACHE = os.path.join(RS, 'cache'); RESULTS = os.path.join(RS, 'results')
FOLDS = [('2022-01-01', '2023-01-01'), ('2023-01-01', '2024-01-01'), ('2024-01-01', '2025-01-01'), ('2025-01-01', '2026-01-01'),
         ('2026-01-01', '2026-06-01'), ('2026-06-01', '2100-01-01')]
TRAIN_START = pd.Timestamp('2020-03-02')
GRID_C = [0.01, 0.03, 0.1, 0.3]
GRID_G = [0.5]                           # gamma = g / 67 (g = 1 would be scikit-learn's 'scale' for standardized features)
TUNE_N = 20000
C_FACTOR = 1.0                           # full-fit C = tuned C * C_FACTOR (PREREG_svm.md 2.3)
SEC_20K = 66.0                           # measured 2026-11-04 on the i5-10210U: 20,000 rows, C = 1, g = 1 -> 66 s


class Scaler:                            # = transformer\tfmodel.Scaler (copied: tfmodel imports torch)
    def fit(self, X):
        self.lo = np.nanquantile(X, 0.01, axis=0); self.hi = np.nanquantile(X, 0.99, axis=0)
        Xc = np.clip(X, self.lo, self.hi)
        self.mu = np.nanmean(Xc, axis=0); self.sd = np.nanstd(Xc, axis=0); self.sd[~(self.sd > 1e-12)] = 1.0
        self.mu = np.nan_to_num(self.mu); return self

    def transform(self, X):
        Z = (np.clip(X, self.lo, self.hi) - self.mu) / self.sd
        return np.nan_to_num(Z, nan=0.0).astype(np.float64)


def log(msg, test=False):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, 'svm_exact_test.log' if test else 'svm_exact.log'), 'a', encoding='utf-8') as fh:
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


def rank_ic(dates, f, r):
    """mean over days of the Spearman correlation of f with r inside the day"""
    X = pd.DataFrame({'d': pd.factorize(pd.Index(dates), sort=True)[0], 'f': np.asarray(f, float), 'r': np.asarray(r, float)}).dropna()
    g = X.groupby('d')
    X['rf'] = g.f.rank().values; X['rr'] = g.r.rank().values
    X['rf'] -= X.groupby('d').rf.transform('mean'); X['rr'] -= X.groupby('d').rr.transform('mean')
    s = X.assign(xy=X.rf * X.rr, xx=X.rf ** 2, yy=X.rr ** 2).groupby('d')[['xy', 'xx', 'yy']].sum()
    return float((s.xy / np.sqrt(s.xx * s.yy)).mean())


def out_file(a, test):
    return os.path.join(CACHE, 'svm_exact_test' if test else 'svm_exact', f'{a}.npz')


def fold_data(a, b, test):
    P = pd.read_pickle(os.path.join(ROOT, 'v9', 'cache', 'core9.pkl'))
    B = json.load(open(os.path.join(ROOT, 'transformer', 'final_feature_lists.json')))['B_competition_only']
    P = P[['date', 'stock_id', 'resid_r', 'y_resid'] + B].copy()
    P['date'] = pd.to_datetime(P.date).astype('datetime64[ns]'); P['stock_id'] = P.stock_id.astype(str)
    P = P.sort_values(['date', 'stock_id']).reset_index(drop=True)
    te = pd.Timestamp(a) - pd.Timedelta(days=1)
    Q = P[(P.date >= TRAIN_START) & (P.date <= te) & P.resid_r.notna()]
    R = P[(P.date >= pd.Timestamp(a)) & (P.date < pd.Timestamp(b)) & P.resid_r.notna()]
    del P
    ud = np.sort(Q.date.unique()); cut = ud[-120] if len(ud) > 240 else ud[-1]
    if test:                                                  # a small random part of the window, same rules otherwise
        keep = np.sort(np.random.default_rng(7).choice(len(Q), min(3000, len(Q)), replace=False))
        Q = Q.iloc[keep]
    X = Q[B].to_numpy(np.float64)
    sc = Scaler().fit(X)
    w = np.minimum(np.abs(Q.resid_r.values), 0.05); w = w / w.mean()
    return dict(Z=sc.transform(X), y=2 * Q.y_resid.values.astype(int) - 1, w=w, trm=Q.date.values < cut,
                qdate=Q.date.values, qr=Q.resid_r.values, ZR=sc.transform(R[B].to_numpy(np.float64)),
                rdate=R.date.values, rsid=np.asarray(R.stock_id.astype(str).tolist(), dtype='<U8'), rr=R.resid_r.values, ncol=len(B))


def expected_minutes(n):
    lo, hi = SEC_20K * (n / 20000) ** 2.3, SEC_20K * (n / 20000) ** 2.6
    return lo / 60, hi / 60


def run_fold(a, b, cfg):
    """tune, fit on the whole window, predict the fold; returns a summary dict (or the traceback)"""
    from sklearn.svm import SVC
    low_priority()
    test = cfg['test']
    f = out_file(a, test)
    if os.path.exists(f):
        return {'fold': a, 'skipped': True}
    t0 = time.time()
    try:
        D = fold_data(a, b, test)
        Z, y, w, trm, ncol = D['Z'], D['y'], D['w'], D['trm'], D['ncol']
        n = len(Z)
        tune_n = min(1500 if test else cfg['tune_n'], int(trm.sum()))
        rng = np.random.default_rng(100 + [x[0] for x in FOLDS].index(a))
        ix = np.sort(rng.choice(np.flatnonzero(trm), tune_n, replace=False))
        vdates, vr = D['qdate'][~trm], D['qr'][~trm]
        log(f'[{a}] window {n:,} rows ({int(trm.sum()):,} before the last 120 days); tuning {len(GRID_C) * len(GRID_G)} pairs on {tune_n:,} rows', test)
        rows = []
        for C in GRID_C:
            for g in GRID_G:
                t1 = time.time()
                svc = SVC(C=C, kernel='rbf', gamma=g / ncol, cache_size=cfg['cache_mb']).fit(Z[ix], y[ix], sample_weight=w[ix])
                fv = svc.decision_function(Z[~trm])
                vh = float((w[~trm] * np.maximum(0.0, 1.0 - y[~trm] * fv)).sum() / w[~trm].sum())
                rows.append({'fold': a, 'C': C, 'g': g, 'val_hinge': vh, 'val_ic': rank_ic(vdates, fv, vr), 'n_sv': int(svc.n_support_.sum()),
                             'sec': round(time.time() - t1, 1)})
                log(f'[{a}]   C {C:<5g} g {g:<4g}: val hinge {vh:.4f}, val rank IC {rows[-1]["val_ic"]:+.4f}  ({rows[-1]["sec"]:.0f} s)', test)
        T = pd.DataFrame(rows)
        j = T.val_hinge.idxmin(); C0, g0 = float(T.C[j]), float(T.g[j])
        C1 = C0 * cfg['c_factor']
        lo, hi = expected_minutes(n)
        log(f'[{a}] tuned C {C0:g}, g {g0:g} -> full fit on {n:,} rows with C {C1:g}, gamma {g0:g}/{ncol}; '
            f'expected roughly {lo:.0f}-{hi:.0f} min on an i5-10210U (started {time.strftime("%H:%M")})', test)
        t1 = time.time()
        svc = SVC(C=C1, kernel='rbf', gamma=g0 / ncol, cache_size=cfg['cache_mb']).fit(Z, y, sample_weight=w)
        fit_sec = time.time() - t1; nsv = int(svc.n_support_.sum())
        log(f'[{a}] full fit done in {fit_sec / 60:.1f} min, {nsv:,} support vectors ({nsv / n:.0%}); predicting {len(D["ZR"]):,} rows', test)
        t1 = time.time()
        p = np.concatenate([svc.decision_function(D['ZR'][s:s + 5000]) for s in range(0, len(D['ZR']), 5000)])
        pred_sec = time.time() - t1
        ic = rank_ic(D['rdate'], p, D['rr'])
        meta = dict(fold=a, rows=n, C_tuned=C0, g=g0, C=C1, gamma=g0 / ncol, n_sv=nsv, fit_sec=round(fit_sec, 1), pred_sec=round(pred_sec, 1),
                    total_sec=round(time.time() - t0, 1), oos_ic=ic, test=test, sklearn=__import__('sklearn').__version__,
                    finished=time.strftime('%Y-%m-%d %H:%M:%S'), tuning=rows)
        os.makedirs(os.path.dirname(f), exist_ok=True)
        tmp = f[:-4] + '.tmp'
        with open(tmp, 'wb') as fh:
            np.savez_compressed(fh, date=np.asarray(pd.to_datetime(D['rdate']), dtype='datetime64[ns]').astype(np.int64),
                                sid=D['rsid'], p=p.astype(np.float64), meta=np.array(json.dumps(meta)))
        os.replace(tmp, f)
        log(f'[{a}] saved {os.path.relpath(f, RS)}: OOS rank IC {ic:+.4f} (predict {pred_sec / 60:.1f} min, fold total {(time.time() - t0) / 60:.1f} min)', test)
        return meta
    except Exception:
        return {'fold': a, 'error': traceback.format_exc()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=2, help='folds trained at the same time (1 = least memory)')
    ap.add_argument('--cache-mb', type=int, default=800, help='libsvm kernel cache per fold, MB')
    ap.add_argument('--tune-n', type=int, default=TUNE_N)
    ap.add_argument('--c-factor', type=float, default=C_FACTOR)
    ap.add_argument('--folds', default='', help='comma separated fold starts, default all six')
    ap.add_argument('--test', action='store_true')
    a = ap.parse_args()
    low_priority(); keep_awake(True)
    try:
        import sklearn
    except ImportError:
        sys.exit('scikit-learn is missing: python -m pip install scikit-learn')
    cfg = {'test': a.test, 'cache_mb': a.cache_mb, 'tune_n': a.tune_n, 'c_factor': a.c_factor}
    folds = [fb for fb in FOLDS if not a.folds or fb[0] in a.folds.split(',')]
    todo = [fb for fb in folds if not os.path.exists(out_file(fb[0], a.test))]
    log(f'svm_exact.py start: {len(folds)} folds, {len(todo)} to do, workers {a.workers}, kernel cache {a.cache_mb} MB, '
        f'scikit-learn {sklearn.__version__}' + (' [TEST]' if a.test else ''), a.test)
    if not todo:
        log('all folds already done', a.test); keep_awake(False); return 0
    if a.workers > 1:                                     # largest folds first: the run ends sooner
        todo = todo[::-1]
    t0 = time.time(); bad = 0
    try:
        if a.workers <= 1:
            res = (run_fold(fa, fb, cfg) for fa, fb in todo)
            for r in res:
                bad += _report(r, a.test)
        else:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn')) as ex:
                futs = [ex.submit(run_fold, fa, fb, cfg) for fa, fb in todo]
                for fu in as_completed(futs):
                    bad += _report(fu.result(), a.test)
    except KeyboardInterrupt:
        log('stopped by the user; finished folds are kept, start again to continue', a.test); keep_awake(False); return 1
    keep_awake(False)
    done = [fb for fb in folds if os.path.exists(out_file(fb[0], a.test))]
    log(f'svm_exact.py end: {len(done)} of {len(folds)} folds done, {(time.time() - t0) / 3600:.2f} h this run', a.test)
    return 0 if len(done) == len(folds) and not bad else 1


def _report(r, test):
    if r.get('error'):
        log(f'[{r["fold"]}] FAILED\n{r["error"]}', test); return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

r"""
rs_svm.py -- SVM members in the same walk-forward as v9 (README_reg_svm.md section 4).
needs numpy, pandas, scikit-learn, lightgbm.

Features, labels, sample weights, folds and the ES split are those of lgb_res (stock view) and of v8's logistic market member:
B 67 features standardized with transformer\tfmodel.Scaler (winsorize 1 / 99%, z-score), weights min(|r|, 5%).
  lin   linear SVM = hinge loss + L2 penalty (LinearSVC(loss='hinge'), liblinear dual CD), C in CL.
  nys   RBF kernel SVM, kernel approximated by Nystroem (1000 landmarks from the training window, gamma = 1/67 = sklearn
        'scale' for standardized features) + the same linear SVM, C in CN.
  For both: every C is fitted on the ES part of the window (scored on the last 120 days: weighted hinge loss, rank IC /
  corr(pbar, m)) and refitted on the whole window (OOS decision values for every C).  The pre-registered choice of C is
  the smallest weighted hinge loss on the last 120 days (the model's own loss, as ES uses log-loss for LightGBM / MLP).
  Market view: decision values -> probabilities with Platt scaling (2-parameter logistic fit on the training window).
  rbf   exact RBF SVC (libsvm) on n random rows of the window (n = 2,500 .. 20,000): fit / predict seconds, number of
        support vectors, OOS rank IC; LightGBM (production parameters, ES) on the same rows for comparison.
  grid  exact RBF SVC, n = 10,000, C x gamma grid, fold 2024 (to pick C, gamma of rbf on the ES split).
run (in research): python rs_svm.py [--workers 2] [--parts lin,nys,grid,rbf]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, time, argparse, pickle, traceback, warnings, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rslib as L

OUT = os.path.join(L.CACHE, 'svm')
CL = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
CN = [1e-3, 1e-2, 1e-1]
NS = [2500, 5000, 10000, 20000]
GRID_C = [0.1, 1.0, 10.0]; GRID_G = [0.25, 1.0, 4.0]          # gamma = g / 67
LGB_PRM = dict(objective='binary', learning_rate=0.02, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8,
               bagging_freq=1, lambda_l2=1.0, num_threads=1, verbose=-1)
_G = {}


def unit_file(u):
    return os.path.join(OUT, '_'.join(str(x) for x in u) + '.pkl')


def lsvc(C, Z, y, w):
    from sklearn.svm import LinearSVC
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        clf = LinearSVC(C=C, loss='hinge', dual=True, max_iter=20000, tol=1e-4).fit(Z, 2 * y - 1, sample_weight=w)
    return clf


def platt(f, y, w):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(f.reshape(-1, 1), y, sample_weight=w)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def view_metric(target, D, f):
    return L.rank_ic(D.date.values, f, D.resid_r.values) if target == 'resid' else \
        L.pbar_corr(D.date.values, f, D.top50.values.astype(bool), D.target_r.values)


def run_unit(u):
    f = unit_file(u)
    if os.path.exists(f):
        return u, 0.0, None
    t0 = time.time()
    try:
        kind, view, a = u[0], u[1], u[2]
        b = dict(L.FOLDS)[a]
        m = L.members()['svm_res' if view == 'res' else 'svm_raw']
        P = _G['P']
        S = L.split(P, m, a, b)
        Q, R, yq, wq, yr, wr, trm = S['Q'], S['R'], S['yq'], S['wq'], S['yr'], S['wr'], S['trm']
        sc = L.Scaler().fit(Q[m['feats']].to_numpy(np.float64))
        Z = sc.transform(Q[m['feats']].to_numpy(np.float64)); ZR = sc.transform(R[m['feats']].to_numpy(np.float64))
        Dv = Q[~trm]
        res = dict(unit=u, key=pd.DataFrame({'date': R.date.values, 'stock_id': R.stock_id.values}), rows=[], oos={}, prob={})
        if kind in ('lin', 'nys'):
            if kind == 'nys':
                from sklearn.kernel_approximation import Nystroem
                ny = Nystroem(kernel='rbf', gamma=1.0 / Z.shape[1], n_components=1000, random_state=L.FOLDS.index((a, b)) + 1).fit(Z)
                Z = ny.transform(Z).astype(np.float32); ZR = ny.transform(ZR).astype(np.float32)
            for C in (CL if kind == 'lin' else CN):
                t1 = time.time()
                c1 = lsvc(C, Z[trm], yq[trm], wq[trm]); fv = c1.decision_function(Z[~trm])
                t_es = time.time() - t1; t1 = time.time()
                c2 = lsvc(C, Z, yq, wq); fq = c2.decision_function(Z); fr = c2.decision_function(ZR)
                t_full = time.time() - t1
                row = dict(C=C, val_hinge=L.whinge(fv, yq[~trm], wq[~trm]), val_metric=view_metric(m['target'], Dv, fv),
                           in_hinge=L.whinge(fq, yq, wq), in_metric=view_metric(m['target'], Q, fq),
                           oos_hinge=L.whinge(fr, yr, wr), oos_metric=view_metric(m['target'], R, fr),
                           iters=int(np.max(c2.n_iter_)), w_norm=float(np.linalg.norm(c2.coef_)), sec_es=t_es, sec_full=t_full,
                           frac_sv_proxy=float(np.mean((2 * yq - 1) * fq <= 1.0)))
                if m['target'] == 'raw':
                    A, B0 = platt(fq, yq, wq)
                    pr = 1 / (1 + np.exp(-(A * fr + B0)))
                    row.update(platt_A=A, platt_B=B0, oos_wll=L.wll(pr, yr, wr))
                    res['prob'][C] = pr.astype(np.float32)
                res['rows'].append(row); res['oos'][C] = fr.astype(np.float32)
        elif kind == 'rbf':
            from sklearn.svm import SVC
            import lightgbm as lgb
            n = u[3]; Cg = _G['rbf_cg']
            rng = np.random.default_rng(100 + L.FOLDS.index((a, b)))
            ix = np.sort(rng.choice(len(Z), n, replace=False))
            t1 = time.time()
            svc = SVC(C=Cg['C'], kernel='rbf', gamma=Cg['g'] / Z.shape[1], cache_size=1500).fit(Z[ix], 2 * yq[ix] - 1, sample_weight=wq[ix])
            t_fit = time.time() - t1; t1 = time.time()
            fr = svc.decision_function(ZR); t_pred = time.time() - t1
            nsv = int(svc.n_support_.sum())
            # LightGBM on the same rows (production parameters, ES on the rows that fall in the last 120 days)
            XQ = Q[m['feats']].to_numpy(np.float64)[ix]; XR = R[m['feats']].to_numpy(np.float64); tr_ix = trm[ix]
            prm = {**LGB_PRM, 'seed': 1, 'min_data_in_leaf': max(20, int(500 * n / len(Z)))}   # same leaf share of the data
            dtr = lgb.Dataset(XQ[tr_ix], yq[ix][tr_ix], weight=wq[ix][tr_ix])
            dva = lgb.Dataset(XQ[~tr_ix], yq[ix][~tr_ix], weight=wq[ix][~tr_ix], reference=dtr)
            mm = lgb.train(prm, dtr, 1000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
            best = max(int(mm.best_iteration), 1)
            gb = lgb.train(prm, lgb.Dataset(XQ, yq[ix], weight=wq[ix]), best); pl = gb.predict(XR)
            res['rows'].append(dict(n=n, C=Cg['C'], g=Cg['g'], sec_fit=t_fit, sec_pred=t_pred, n_sv=nsv, frac_sv=nsv / n,
                                    oos_metric=view_metric('resid', R, fr), lgb_oos_metric=view_metric('resid', R, pl), lgb_best=best,
                                    lgb_min_leaf=prm['min_data_in_leaf']))
            res['oos'][n] = fr.astype(np.float32)
        elif kind == 'grid':
            from sklearn.svm import SVC
            n = u[3]
            rng = np.random.default_rng(100 + L.FOLDS.index((a, b)))
            ix = np.sort(rng.choice(np.flatnonzero(trm), n, replace=False))            # ES part only -> scored on the last 120 days
            for C in GRID_C:
                for g in GRID_G:
                    t1 = time.time()
                    svc = SVC(C=C, kernel='rbf', gamma=g / Z.shape[1], cache_size=1500).fit(Z[ix], 2 * yq[ix] - 1, sample_weight=wq[ix])
                    fv = svc.decision_function(Z[~trm])
                    res['rows'].append(dict(n=n, C=C, g=g, val_metric=view_metric('resid', Dv, fv), val_hinge=L.whinge(fv, yq[~trm], wq[~trm]),
                                            n_sv=int(svc.n_support_.sum()), sec=time.time() - t1))
        res['seconds'] = time.time() - t0
        os.makedirs(OUT, exist_ok=True)
        with open(f + '.tmp', 'wb') as fh:
            pickle.dump(res, fh)
        os.replace(f + '.tmp', f)
        return u, time.time() - t0, None
    except Exception:
        return u, time.time() - t0, traceback.format_exc()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=2); ap.add_argument('--parts', default='lin,nys,grid,rbf')
    a = ap.parse_args()
    parts = a.parts.split(',')
    _G['P'] = L.panel()
    F = [fa for fa, _ in L.FOLDS]
    import multiprocessing as mp
    t0 = time.time()

    def run(units, tag):
        todo = [u for u in units if not os.path.exists(unit_file(u))]
        print(f'{tag}: {len(units)} units, {len(todo)} to do', flush=True)
        if not todo:
            return
        pool = mp.get_context('fork').Pool(a.workers)
        for i, (u, sec, err) in enumerate(pool.imap_unordered(run_unit, todo), 1):
            if err:
                print('FAILED', u, err, flush=True)
            print(f'[{tag} {i}/{len(todo)}] {u} {sec:.0f}s, elapsed {(time.time() - t0) / 60:.1f} min', flush=True)
        pool.close(); pool.join()

    if 'lin' in parts:
        run([('lin', v, fa) for v in ('res', 'raw') for fa in F], 'lin')
    if 'nys' in parts:
        run([('nys', v, fa) for v in ('res', 'raw') for fa in reversed(F)], 'nys')
    if 'grid' in parts:
        run([('grid', 'res', '2024-01-01', 10000)], 'grid')
    if 'rbf' in parts:
        g = pickle.load(open(unit_file(('grid', 'res', '2024-01-01', 10000)), 'rb'))
        G = pd.DataFrame(g['rows']); j = G.val_metric.idxmax()                         # pre-registered: best val rank IC
        _G['rbf_cg'] = {'C': float(G.C[j]), 'g': float(G.g[j])}
        print('rbf uses', _G['rbf_cg'], flush=True)
        run([('rbf', 'res', fa, n) for n in NS for fa in reversed(F)], 'rbf')


if __name__ == '__main__':
    main()

"""g1_tab_train: walk-forward training of LightGBM and logistic regression members (PREREG_m.md, sixth round).
B features (67), targets raw (market view) and resid (stock view), yearly folds 2022..2026-05 + post (trained to 2026-05-29).
Writes ../cache/tab/<model>_<target>/<fold>_seed<s>.pkl; resumable.  needs lightgbm and scikit-learn.
run: python g1_tab_train.py [--workers 2]"""
import os, sys, time, argparse, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); V8 = os.path.dirname(HERE); ROOT = os.path.dirname(V8)
for p in (HERE, os.path.join(ROOT, 'v5'), os.path.join(ROOT, 'transformer')):
    sys.path.insert(0, p)
import numpy as np, pandas as pd
import mlib as L
OUT = os.path.join(L.CACHE, 'tab')
FOLDS = [('2022-01-01', '2023-01-01'), ('2023-01-01', '2024-01-01'), ('2024-01-01', '2025-01-01'), ('2025-01-01', '2026-01-01'),
         ('2026-01-01', '2026-06-01'), ('2026-06-01', '2100-01-01')]
LGB = dict(objective='binary', learning_rate=0.02, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=1, verbose=-1)
CS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
_G = {}

def path(model, target, a, s):
    return os.path.join(OUT, f'{model}_{target}', f'{a}_seed{s}.pkl')

def wlogloss(y, p, w):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / w.sum()

def run_unit(u):
    model, target, a, b, s = u
    f = path(model, target, a, s)
    if os.path.exists(f):
        return u, 0.0, None, None
    t0 = time.time()
    try:
        P, feats, E = _G['P'], _G['feats'], _G['E']
        te = pd.Timestamp(a) - pd.Timedelta(days=1)
        tcol, ycol = ('target_r', 'y') if target == 'raw' else ('resid_r', 'y_resid')
        Q = P[(P.date >= E.TRAIN_START) & (P.date <= te) & P[tcol].notna()]
        X = Q[feats].to_numpy(np.float64); y = Q[ycol].values.astype(float)
        w = np.minimum(np.abs(Q[tcol].values), 0.05); w = w / w.mean()
        ud = np.sort(Q.date.unique()); cut = ud[-120] if len(ud) > 240 else ud[-1]; tr = Q.date.values < cut
        R = P[(P.date >= pd.Timestamp(a)) & (P.date < pd.Timestamp(b)) & P[tcol].notna()]
        XR = R[feats].to_numpy(np.float64)
        if model == 'lgb':
            import lightgbm as lgb
            prm = {**LGB, 'seed': s}
            dtr = lgb.Dataset(X[tr], y[tr], weight=w[tr]); dva = lgb.Dataset(X[~tr], y[~tr], weight=w[~tr], reference=dtr)
            m = lgb.train(prm, dtr, 1000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
            best = max(int(m.best_iteration), 1)
            m = lgb.train(prm, lgb.Dataset(X, y, weight=w), best)
            p = m.predict(XR)
        else:
            from sklearn.linear_model import LogisticRegression
            import tfmodel as TM
            sc = TM.Scaler().fit(X); Z = sc.transform(X); ZR = sc.transform(XR)
            ll = []
            for C in CS:
                r = LogisticRegression(C=C, max_iter=1000).fit(Z[tr], y[tr], sample_weight=w[tr])
                ll.append(wlogloss(y[~tr], r.predict_proba(Z[~tr])[:, 1], w[~tr]))
            best = CS[int(np.argmin(ll))]
            r = LogisticRegression(C=best, max_iter=1000).fit(Z, y, sample_weight=w)
            p = r.predict_proba(ZR)[:, 1]
        o = pd.DataFrame({'date': R.date.values, 'stock_id': R.stock_id.values, 'p': p})
        os.makedirs(os.path.dirname(f), exist_ok=True); o.to_pickle(f + '.tmp'); os.replace(f + '.tmp', f)
        return u, time.time() - t0, None, best
    except Exception:
        return u, time.time() - t0, traceback.format_exc(), None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    import ens_lib as E
    _G['P'], _G['feats'], _G['E'] = L.panel(), list(E.feature_sets()['B']), E
    units = [('lgb', t, fa, fb, s) for t in ('raw', 'resid') for fa, fb in FOLDS for s in range(1, 6)] + \
            [('lr', t, fa, fb, 1) for t in ('raw', 'resid') for fa, fb in FOLDS]
    todo = [u for u in units if not os.path.exists(path(u[0], u[1], u[2], u[4]))]
    print(f'{len(units)} units, {len(todo)} to do', flush=True)
    import multiprocessing as mp
    t0 = time.time(); res = []
    if a.workers > 1 and 'fork' in mp.get_all_start_methods():
        pool = mp.get_context('fork').Pool(a.workers); it = pool.imap_unordered(run_unit, todo)
    else:
        pool = None; it = map(run_unit, todo)
    for i, (u, sec, err, best) in enumerate(it, 1):
        if err:
            print('FAILED', u, err, flush=True)
        else:
            res.append((u[0], u[1], u[2], best))
        if i % 10 == 0 or i == len(todo):
            print(f'[{i}/{len(todo)}] {u} {sec:.0f}s, elapsed {(time.time() - t0) / 60:.1f} min', flush=True)
    if pool is not None:
        pool.close(); pool.join()
    if res:
        R = pd.DataFrame(res, columns=['model', 'target', 'fold', 'best'])
        print('LightGBM best number of rounds (early stopping):'); print(R[R.model == 'lgb'].groupby(['target', 'fold']).best.agg(['min', 'median', 'max']).to_string())
        print('logistic regression C chosen:'); print(R[R.model == 'lr'].pivot(index='fold', columns='target', values='best').to_string())

if __name__ == '__main__':
    main()

r"""
v12_train.py -- the six new LightGBM variants of research\PREREG_v12.md (label information x loss), B features (67),
seed 1..5, the 6 yearly walk-forward folds of v9 / v10 / v11 (X1, X5):
  Y1 lgb_sign      : binary, label 1{resid_r > 0}, no weights                                   (pointwise, sign)
  Y2 lgb_dec_l2    : L2 regression on the within-day decile 0..9                                 (pointwise, decile)
  Y3 lgb_size_l2   : L2 regression on clip(resid_r, -5%, 5%) in percent                          (pointwise, size)
  Y4 lgb_dec_pw    : within-day pairwise logistic, pair weight |decile_i - decile_j|             (pairwise, decile)
  Y5 lgb_size_pw   : within-day pairwise logistic, pair weight |r_i - r_j| (clipped)             (pairwise, size)
  Y6 lgb_size_rank : lambdarank as X5, label = 0.25% levels 0..40 of the clipped return, linear gains (lambdarank, size)
lgb_res's parameters; pointwise models bag rows (as lgb_res), pairwise / lambdarank models bag days (as X5); early stopping
(patience 100, at most 1000 rounds) on the last 120 training days with each model's own loss, then refit on all training days.
-> ..\cache\screen\<name>\<fold>_seed<s>.npz (resumable); best rounds -> ..\results\v12_train_units.csv
run (in v12\research): python v12_train.py --workers 2
"""
import os, sys, time, argparse, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v12lib as L
V = L.V
import v11e_train as E                         # X5's lambdarank parameters (read only)

MEMBERS = ['lgb_sign', 'lgb_dec_l2', 'lgb_size_l2', 'lgb_dec_pw', 'lgb_size_pw', 'lgb_size_rank']
KIND = {'lgb_sign': 'binary', 'lgb_dec_l2': 'l2', 'lgb_size_l2': 'l2', 'lgb_dec_pw': 'pairwise', 'lgb_size_pw': 'pairwise',
        'lgb_size_rank': 'lambdarank'}

def labels(name, Q):
    if name == 'lgb_sign':
        return Q.y_resid.values.astype(float)
    if name in ('lgb_dec_l2', 'lgb_dec_pw'):
        return L.decile_label(Q).astype(float)
    if name in ('lgb_size_l2', 'lgb_size_pw'):
        return L.size_label(Q)
    if name == 'lgb_size_rank':
        return L.size_level(Q)
    raise KeyError(name)

def params(name, seed, obj=None):
    k = KIND[name]
    if k == 'binary':
        return {**V.LGB_PRM, 'seed': int(seed)}
    if k == 'l2':
        return {**V.LGB_PRM, 'objective': 'regression', 'metric': 'l2', 'seed': int(seed)}
    if k == 'pairwise':
        return {**V.LGB_PRM, 'objective': L.lgb_objective(obj), 'metric': 'None', 'bagging_by_query': True, 'seed': int(seed)}
    return {**E.RANK_PRM, 'label_gain': list(range(41)), 'seed': int(seed)}

def train_unit(P, name, fold_a, fold_b, seed, feats=None, rounds=1000, patience=100):
    import lightgbm as lgb
    feats = feats or list(V.base_features()['lgb_res'])
    te = pd.Timestamp(fold_a) - pd.Timedelta(days=1)
    Q = P[(P.date >= V.TRAIN_START) & (P.date <= te) & P.resid_r.notna()].sort_values(['date', 'stock_id'])
    X = Q[feats].to_numpy(np.float64); y = labels(name, Q); dates = Q.date.values
    ud = np.sort(np.unique(dates)); cut = ud[-120] if len(ud) > 240 else ud[-1]; tr = dates < cut
    gsize = lambda m: pd.Series(dates[m]).groupby(dates[m]).size().values          # rows per day, in date order
    k = KIND[name]
    grouped = k in ('pairwise', 'lambdarank')
    ds = lambda m, ref=None: lgb.Dataset(X[m], y[m], group=gsize(m) if grouped else None, reference=ref)
    if k == 'pairwise':
        otr, ova = L.PairwiseObjective(y[tr], dates[tr]), L.PairwiseObjective(y[~tr], dates[~tr])
        dtr = ds(tr); dva = ds(~tr, dtr)
        m = lgb.train(params(name, seed, otr), dtr, rounds, valid_sets=[dva], feval=L.lgb_metric(ova),
                      callbacks=[lgb.early_stopping(patience, verbose=False)])
    else:
        dtr = ds(tr); dva = ds(~tr, dtr)
        m = lgb.train(params(name, seed), dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(patience, verbose=False)])
    best = max(int(m.best_iteration), 1)
    full = np.ones(len(X), bool)
    prm = params(name, seed, L.PairwiseObjective(y, dates)) if k == 'pairwise' else params(name, seed)
    m = lgb.train(prm, ds(full), best)
    R = P[(P.date >= pd.Timestamp(fold_a)) & (P.date < pd.Timestamp(fold_b)) & P.resid_r.notna()]
    p = m.predict(R[feats].to_numpy(np.float64))
    return pd.DataFrame({'date': R.date.values, 'stock_id': R.stock_id.values, 'p': p}), best

_G = {}
def run_unit(u):
    name, fa, fb, s = u
    f = L.unit_path(name, fa, s)
    if V.unit_exists(f):
        return u, 0.0, None, None
    t0 = time.time()
    try:
        o, best = train_unit(_G['P'], name, fa, fb, s)
        V.save_unit(f, o)
        return u, time.time() - t0, None, best
    except Exception:
        return u, time.time() - t0, traceback.format_exc(), None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--members', nargs='*', default=MEMBERS)
    a = ap.parse_args()
    os.makedirs(os.path.join(L.V12, 'results'), exist_ok=True)
    logf = open(os.path.join(L.V12, 'results', 'v12_train.log'), 'a', encoding='utf-8')
    def log(msg):
        line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'; print(line, flush=True); logf.write(line + '\n'); logf.flush()
    _G['P'] = V.panel(log=log)
    o = L.PairwiseObjective(np.array([0., 1., 2.]), np.array([1, 1, 1])); o.grad_hess(np.zeros(3)); o.loss(np.zeros(3))   # compile once
    U = [(n, fa, fb, s) for fa, fb in V.FOLDS for n in a.members for s in range(1, 6)]
    todo = [u for u in U if not V.unit_exists(L.unit_path(u[0], u[1], u[3]))]
    todo.sort(key=lambda u: (-V.FOLDS.index((u[1], u[2])), KIND[u[0]] != 'pairwise'))
    log(f'v12 members {a.members}: {len(U)} units, {len(todo)} to do, {a.workers} workers')
    import multiprocessing as mp
    t0 = time.time(); done = fails = 0
    bf = os.path.join(L.V12, 'results', 'v12_train_units.csv')
    if not os.path.exists(bf):
        open(bf, 'w').write('member,fold,seed,best,seconds\n')
    pool = mp.get_context('fork').Pool(a.workers) if a.workers > 1 else None
    it = pool.imap_unordered(run_unit, todo, chunksize=1) if pool else map(run_unit, todo)
    for u, sec, err, best in it:
        done += 1
        if err:
            fails += 1; log(f'FAILED {u}\n{err}')
        elif best is not None:
            open(bf, 'a').write(f'{u[0]},{u[1]},{u[3]},{best},{sec:.1f}\n')
        if done % 10 == 0 or done == len(todo) or done <= 2:
            log(f'[{done}/{len(todo)}] {u[0]} {u[1]} seed {u[3]}: {sec:.0f}s (best {best}); elapsed {(time.time() - t0) / 60:.1f} min')
    if pool:
        pool.close(); pool.join()
    log(f'finished {done} units in {(time.time() - t0) / 60:.1f} min, {fails} failed')

if __name__ == '__main__':
    main()

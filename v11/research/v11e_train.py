r"""
v11e_train.py -- X5 lgb_rank (research\PREREG_v11e.md): LightGBM lambdarank, one query = one day (~100 stocks),
label = within-day decile of resid_r (0..9), linear label_gain, truncation 100, query bagging 0.8, B features (67),
early stopping on NDCG@100 of the last 120 training days, refit; seed 1..5, the 6 yearly folds of v9 / v10 / v11.
-> ..\cache\screen\x\lgb_rank\lgb_rank\<fold>_seed<s>.npz (resumable)
run (in v11\research): python v11e_train.py --workers 2
"""
import os, sys, time, argparse, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v11b_train as TB
V = TB.V
NAME = 'lgb_rank'
RANK_PRM = dict(objective='lambdarank', label_gain=list(range(10)), lambdarank_truncation_level=100, metric='ndcg', eval_at=[100],
                learning_rate=0.02, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                bagging_by_query=True, lambda_l2=1.0, num_threads=1, verbose=-1)

def decile_label(Q):
    """within-day decile of resid_r, 0..9 (rows must be sorted by date)"""
    r = Q.groupby('date').resid_r.rank(method='first'); n = Q.groupby('date').resid_r.transform('size')
    return np.minimum(9, np.floor((r.values - 1) * 10 / n.values)).astype(int)

def train_unit(P, fold_a, fold_b, seed, feats=None):
    import lightgbm as lgb
    feats = feats or list(V.base_features()['lgb_res'])
    te = pd.Timestamp(fold_a) - pd.Timedelta(days=1)
    Q = P[(P.date >= V.TRAIN_START) & (P.date <= te) & P.resid_r.notna()].sort_values(['date', 'stock_id'])
    X = Q[feats].to_numpy(np.float64); y = decile_label(Q)
    dates = Q.date.values
    ud = np.sort(np.unique(dates)); cut = ud[-120] if len(ud) > 240 else ud[-1]; tr = dates < cut
    gsize = lambda m: pd.Series(dates[m]).groupby(dates[m]).size().values          # rows per day, in date order
    prm = {**RANK_PRM, 'seed': int(seed)}
    dtr = lgb.Dataset(X[tr], y[tr], group=gsize(tr)); dva = lgb.Dataset(X[~tr], y[~tr], group=gsize(~tr), reference=dtr)
    m = lgb.train(prm, dtr, 1000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    best = max(int(m.best_iteration), 1)
    m = lgb.train(prm, lgb.Dataset(X, y, group=gsize(np.ones(len(X), bool))), best)
    R = P[(P.date >= pd.Timestamp(fold_a)) & (P.date < pd.Timestamp(fold_b)) & P.resid_r.notna()]
    p = m.predict(R[feats].to_numpy(np.float64))
    return pd.DataFrame({'date': R.date.values, 'stock_id': R.stock_id.values, 'p': p}), best

_G = {}
def run_unit(u):
    fa, fb, s = u
    f = TB.unit_path(NAME, fa, s)
    if V.unit_exists(f):
        return u, 0.0, None, None
    t0 = time.time()
    try:
        o, best = train_unit(_G['P'], fa, fb, s)
        V.save_unit(f, o)
        return u, time.time() - t0, None, best
    except Exception:
        return u, time.time() - t0, traceback.format_exc(), None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    logf = open(os.path.join(TB.V11, 'results', 'v11e_train.log'), 'a', encoding='utf-8')
    def log(msg):
        line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'; print(line, flush=True); logf.write(line + '\n'); logf.flush()
    _G['P'] = V.panel(log=log)
    U = [(fa, fb, s) for fa, fb in V.FOLDS for s in range(1, 6)]
    todo = [u for u in U if not V.unit_exists(TB.unit_path(NAME, u[0], u[2]))]
    todo.sort(key=lambda u: -V.FOLDS.index((u[0], u[1])))
    log(f'X5 {NAME}: {len(U)} units, {len(todo)} to do, {a.workers} workers')
    import multiprocessing as mp
    t0 = time.time(); done = fails = 0; bests = []
    pool = mp.get_context('fork').Pool(a.workers) if a.workers > 1 else None
    it = pool.imap_unordered(run_unit, todo, chunksize=1) if pool else map(run_unit, todo)
    for u, sec, err, best in it:
        done += 1
        if err:
            fails += 1; log(f'FAILED {u}\n{err}')
        else:
            bests.append((u[0], best))
        if done % 5 == 0 or done == len(todo) or done <= 2:
            log(f'[{done}/{len(todo)}] {u[0]} seed {u[2]}: {sec:.0f}s (best {best}); elapsed {(time.time() - t0) / 60:.1f} min')
    if pool:
        pool.close(); pool.join()
    if bests:
        B = pd.DataFrame(bests, columns=['fold', 'best']).groupby('fold').best.agg(['min', 'median', 'max'])
        log('best rounds by fold:\n' + B.to_string())
    log(f'finished {done} units in {(time.time() - t0) / 60:.1f} min, {fails} failed')

if __name__ == '__main__':
    main()

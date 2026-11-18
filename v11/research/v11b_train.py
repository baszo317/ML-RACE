r"""
v11b_train.py -- the two new stock-view members of v11 part B (research\PREREG_v11b.md), yearly walk-forward as v9 / v10:
  X1 lgb_res_xrN  : LightGBM exactly as lgb_res (binary, weights min(|resid_r|, 5%), early stopping on the last 120 training days,
                    refit) with the features of mlp_res_xrN9 (B9 + XR + NEW + XRN + SIG, 106)
  X2 lgb_res_soft : LightGBM objective cross_entropy, label 0.5 + clip(resid_r, -5%, 5%) / 10% (return size in the label, no
                    weights), B features (67); same parameters, early stopping and refit
seed 1..5, 6 folds -> ..\cache\screen\x\<member>\<member>\<fold>_seed<s>.npz (resumable)
run (in v11\research): python v11b_train.py --workers 2
"""
import os, sys, time, argparse, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); V11 = os.path.dirname(HERE); ROOT = os.path.dirname(V11)
sys.path.insert(0, os.path.join(ROOT, 'v10', 'research')); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v10lib as V

SCREEN = os.path.join(V11, 'cache', 'screen')
MEMBERS = {'lgb_res_xrN': 'xrN', 'lgb_res_soft': 'soft'}

def member_features(name):
    B = V.base_features()
    return list(B['mlp_res_xrN9']) if name == 'lgb_res_xrN' else list(B['lgb_res'])

def soft_label(r):
    return 0.5 + np.clip(r, -0.05, 0.05) / 0.10

def train_unit(P, name, fold_a, fold_b, seed):
    """-> DataFrame date, stock_id, p (prediction for the fold's labelled rows), best rounds"""
    import lightgbm as lgb
    feats = member_features(name)
    te = pd.Timestamp(fold_a) - pd.Timedelta(days=1)
    Q = P[(P.date >= V.TRAIN_START) & (P.date <= te) & P.resid_r.notna()]
    X = Q[feats].to_numpy(np.float64)
    if name == 'lgb_res_soft':
        y = soft_label(Q.resid_r.values.astype(float)); w = None
        prm = {**V.LGB_PRM, 'objective': 'cross_entropy', 'seed': int(seed)}
    else:
        y = Q.y_resid.values.astype(float)
        w = np.minimum(np.abs(Q.resid_r.values), 0.05); w = w / w.mean()
        prm = {**V.LGB_PRM, 'seed': int(seed)}
    ud = np.sort(Q.date.unique()); cut = ud[-120] if len(ud) > 240 else ud[-1]; tr = Q.date.values < cut
    wt = (lambda m: None) if w is None else (lambda m: w[m])
    dtr = lgb.Dataset(X[tr], y[tr], weight=wt(tr)); dva = lgb.Dataset(X[~tr], y[~tr], weight=wt(~tr), reference=dtr)
    m = lgb.train(prm, dtr, 1000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    best = max(int(m.best_iteration), 1)
    m = lgb.train(prm, lgb.Dataset(X, y, weight=w), best)
    R = P[(P.date >= pd.Timestamp(fold_a)) & (P.date < pd.Timestamp(fold_b)) & P.resid_r.notna()]
    p = m.predict(R[feats].to_numpy(np.float64))
    return pd.DataFrame({'date': R.date.values, 'stock_id': R.stock_id.values, 'p': p}), best

def unit_path(name, fold_a, seed):
    return V.unit_path('x', name, name, fold_a, seed, SCREEN)

_G = {}
def run_unit(u):
    name, fa, fb, s = u
    f = unit_path(name, fa, s)
    if V.unit_exists(f):
        return u, 0.0, None, None
    t0 = time.time()
    try:
        o, best = train_unit(_G['P'], name, fa, fb, s)
        V.save_unit(f, o)
        return u, time.time() - t0, None, best
    except Exception:
        return u, time.time() - t0, traceback.format_exc(), None

def load_member(name, seeds=range(1, 6)):
    """seed mean over all folds (Series indexed by date, stock_id); None if a unit is missing"""
    parts = []
    for fa, _ in V.FOLDS:
        fs = [unit_path(name, fa, s) for s in seeds]
        if not all(V.unit_exists(f) for f in fs):
            return None
        parts.append(pd.concat([V.load_unit(f) for f in fs], axis=1).mean(axis=1))
    return pd.concat(parts).sort_index()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=2); ap.add_argument('--seeds', default='1-5')
    ap.add_argument('--members', nargs='*', default=list(MEMBERS))
    a = ap.parse_args(); lo, hi = map(int, a.seeds.split('-'))
    logf = open(os.path.join(V11, 'results', 'v11b_train.log'), 'a', encoding='utf-8')
    def log(msg):
        line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'; print(line, flush=True); logf.write(line + '\n'); logf.flush()
    _G['P'] = V.panel(log=log)
    U = [(n, fa, fb, s) for n in a.members for fa, fb in V.FOLDS for s in range(lo, hi + 1)]
    todo = [u for u in U if not V.unit_exists(unit_path(u[0], u[1], u[3]))]
    todo.sort(key=lambda u: -V.FOLDS.index((u[1], u[2])))
    log(f'v11b members {a.members}: {len(U)} units, {len(todo)} to do, {a.workers} workers')
    import multiprocessing as mp
    t0 = time.time(); done = fails = 0
    pool = mp.get_context('fork').Pool(a.workers) if a.workers > 1 else None
    it = pool.imap_unordered(run_unit, todo, chunksize=1) if pool else map(run_unit, todo)
    for u, sec, err, best in it:
        done += 1
        if err:
            fails += 1; log(f'FAILED {u}\n{err}')
        if done % 10 == 0 or done == len(todo) or done <= 2:
            log(f'[{done}/{len(todo)}] {u[0]} {u[1]} seed {u[3]}: {sec:.0f}s (best {best}); elapsed {(time.time() - t0) / 60:.1f} min')
    if pool:
        pool.close(); pool.join()
    log(f'finished {done} units in {(time.time() - t0) / 60:.1f} min, {fails} failed')

if __name__ == '__main__':
    main()

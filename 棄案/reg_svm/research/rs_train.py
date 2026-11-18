r"""
rs_train.py -- regularization ablation and data-size learning curve of v9's members (README_reg_svm.md sections 2-3).
needs numpy, pandas, lightgbm, scipy (erf for the numpy MLP).  Cloud, 2 cores.

Members (retrained exactly as the walk-forward: yearly folds 2022 .. 2026-05 + post, window 2020-03-02 .. day before the
fold, early stopping (ES) on the last 120 training days, refit on the whole window):
  lgb_res      stock view LightGBM (B 67, label beats the same-day average, weights min(|r|, 5%))
  mlp_res_xr9  stock view MLP (B9 + XR + sigma, 96 features), numpy copy of the torch procedure (npmlp)
  mlp_raw_B9   market view MLP (B9 + night + sigma, 68 features, label close > open)
Parameter sets (pset):
  LightGBM  cur      production: lr 0.02, 15 leaves, min_data_in_leaf 500, feature/bagging fraction 0.8, lambda_l2 1
            leaf20   cur but min_data_in_leaf 20 (LightGBM default)
            nol2bag  cur but lambda_l2 0, feature_fraction 1, no bagging
            none     min_data_in_leaf 20, lambda_l2 0, feature_fraction 1, no bagging (lr 0.02, 15 leaves kept)
            strong   min_data_in_leaf 2000, lambda_l2 10, 7 leaves
            default  LightGBM defaults: lr 0.1, 31 leaves, min_data_in_leaf 20, no lambda, no bagging, 100 trees
  MLP       cur      production: dropout 0.2, AdamW weight decay 1e-2
            nodw     dropout 0, weight decay 0
            strong   dropout 0.5, weight decay 1e-1
Every unit runs the ES fit (to pick the stopping point) and one refit on the whole window that is long enough to also give
the "no early stopping" predictions: LightGBM refit to 1000 trees (2000 for none / cur seed 1), MLP refit to 20 epochs.
Snapshots kept (OOS predictions): es (the ES stopping point = production rule), n1000 / n2000 / n100 (LightGBM fixed
trees), e10 / e20 (MLP fixed epochs).  frac < 1: a random fraction of the training DAYS (nested, same days for every
pset and seed of a fold), ES set = the last 120 days of the window at the same sampling rate.
run (in research): python rs_train.py [--workers 2] [--only part]   -> ../cache/units/...
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, time, argparse, pickle, traceback, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rslib as L
import npmlp_ref as N

OUT = os.path.join(L.CACHE, 'units')
LGB_BASE = dict(objective='binary', learning_rate=0.02, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8,
                bagging_freq=1, lambda_l2=1.0, num_threads=1, verbose=-1)          # = v8 g1_tab_train.LGB = rolling9.LGB_PRM
NOBAG = dict(feature_fraction=1.0, bagging_fraction=1.0, bagging_freq=0)
LGB_PSETS = {'cur': {}, 'leaf20': dict(min_data_in_leaf=20), 'nol2bag': dict(lambda_l2=0.0, **NOBAG),
             'none': dict(min_data_in_leaf=20, lambda_l2=0.0, **NOBAG), 'strong': dict(min_data_in_leaf=2000, lambda_l2=10.0, num_leaves=7),
             'default': dict(learning_rate=0.1, num_leaves=31, min_data_in_leaf=20, lambda_l2=0.0, **NOBAG)}
MLP_PSETS = {'cur': dict(dropout=0.2, wd=1e-2), 'nodw': dict(dropout=0.0, wd=0.0), 'strong': dict(dropout=0.5, wd=1e-1)}
LGB_CK = [10, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000]
_G = {}


# ------------------------------------------------------------------ MLP with dropout / weight decay as arguments
def mlp_fit(X, y, w, epochs, seed, dropout, wd, val=None, patience=2, bs=1024, on_epoch=None):
    """= npmlp.fit (numpy copy of the torch procedure) with dropout and weight decay exposed"""
    rng = np.random.default_rng(seed)
    model = N.MLP(X.shape[1], rng, dropout=dropout)
    order_rng = np.random.default_rng(seed + 7919)
    best, best_ep, bad = math.inf, 0, 0
    for ep in range(1, epochs + 1):
        order = order_rng.permutation(len(X))
        for s in range(0, len(X), bs):
            o = order[s:s + bs]
            model.step(X[o], y[o], w[o], rng, wd=wd)
        if on_epoch is not None:
            on_epoch(model, ep)
        if val is not None:
            vl = N.wlogloss_logits(model.logits(val[0]), val[1], val[2])
            if vl < best - 1e-5:
                best, best_ep, bad = vl, ep, 0
            else:
                bad += 1
                if bad >= patience:
                    break
    return model, best_ep


def sig(z):
    return 1.0 / (1.0 + np.exp(-z.astype(np.float64)))


# ------------------------------------------------------------------ one unit
def unit_file(u):
    member, pset, a, seed, frac = u
    return os.path.join(OUT, member, pset, f'f{frac:g}', f'{a}_s{seed}.pkl')


def run_unit(u):
    f = unit_file(u)
    if os.path.exists(f):
        return u, 0.0, None
    t0 = time.time()
    try:
        member, pset, a, seed, frac = u
        b = dict(L.FOLDS)[a]; m = L.members()[member]; P = _G['P']
        S = L.split(P, m, a, b, frac=frac, frac_seed=L.FOLDS.index((a, b)))
        Q, R, yq, yr, wq, wr, trm = S['Q'], S['R'], S['yq'], S['yr'], S['wq'], S['wr'], S['trm']
        curve, snap, ins = [], {}, {}

        def ev(step, pr, pq=None):
            curve.append({'step': step, 'split': 'oos', **L.member_metrics(m['target'], R, pr, yr, wr)})
            if pq is not None:
                curve.append({'step': step, 'split': 'in', **L.member_metrics(m['target'], Q, pq, yq, wq)})

        if m['kind'] == 'lgb':
            import lightgbm as lgb
            prm = {**LGB_BASE, **LGB_PSETS[pset], 'seed': int(seed)}
            XQ = Q[m['feats']].to_numpy(np.float64); XR = R[m['feats']].to_numpy(np.float64)
            dtr = lgb.Dataset(XQ[trm], yq[trm], weight=wq[trm]); dva = lgb.Dataset(XQ[~trm], yq[~trm], weight=wq[~trm], reference=dtr)
            mm = lgb.train(prm, dtr, 1000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
            best = max(int(mm.best_iteration), 1)
            nmax = 2000 if (pset == 'none' or (pset == 'cur' and seed == 1)) and frac == 1.0 else 1000
            if pset == 'default':
                nmax = max(100, best)
            nmax = max(nmax, best)
            full = lgb.train(prm, lgb.Dataset(XQ, yq, weight=wq), nmax)
            keep = {'es': best, 'n1000': 1000, 'n2000': 2000, 'n100': 100}
            for k in sorted(set([c for c in LGB_CK if c <= nmax] + [best])):
                pr = full.predict(XR, num_iteration=k)
                want_in = k in (best, 1000, 2000) or (pset == 'default' and k == 100)
                pq = full.predict(XQ, num_iteration=k) if want_in else None
                ev(k, pr, pq)
                for name, kk in keep.items():
                    if kk == k and kk <= nmax and (name != 'n100' or pset == 'default'):
                        snap[name] = pr.astype(np.float32)
                        if pq is not None:
                            ins[name] = L.member_metrics(m['target'], Q, pq, yq, wq)
        else:
            hp = MLP_PSETS[pset]
            sc = L.Scaler().fit(Q[m['feats']].to_numpy(np.float64))
            XQ = sc.transform(Q[m['feats']].to_numpy(np.float64)); XR = sc.transform(R[m['feats']].to_numpy(np.float64))
            _, best = mlp_fit(XQ[trm], yq[trm], wq[trm], 10, seed, hp['dropout'], hp['wd'], val=(XQ[~trm], yq[~trm], wq[~trm]))
            best = max(best, 1)
            keep = {'es': best, 'e10': 10, 'e20': 20}

            def on_epoch(model, ep):
                pr = sig(model.logits(XR))
                want_in = ep in (best, 10, 20)
                pq = sig(model.logits(XQ)) if want_in else None
                ev(ep, pr, pq)
                for name, kk in keep.items():
                    if kk == ep:
                        snap[name] = pr.astype(np.float32)
                        if pq is not None:
                            ins[name] = L.member_metrics(m['target'], Q, pq, yq, wq)
            mlp_fit(XQ, yq, wq, 20, seed, hp['dropout'], hp['wd'], on_epoch=on_epoch)
        res = dict(unit=u, best=best, curve=pd.DataFrame(curve), ins=ins, n_train=len(Q), days_train=Q.date.nunique(),
                   key=pd.DataFrame({'date': R.date.values, 'stock_id': R.stock_id.values}),
                   snap={k: v for k, v in snap.items()}, seconds=time.time() - t0)
        os.makedirs(os.path.dirname(f), exist_ok=True)
        with open(f + '.tmp', 'wb') as fh:
            pickle.dump(res, fh)
        os.replace(f + '.tmp', f)
        return u, time.time() - t0, None
    except Exception:
        return u, time.time() - t0, traceback.format_exc()


def all_units():
    """(tier, unit): tier 1 = production vs all off (full data), 2 = data size, 3 = one-knob / stronger"""
    F = [fa for fa, _ in L.FOLDS]
    U = []
    for a in F:
        for s in (1, 2):
            U.append((1, ('lgb_res', 'cur', a, s, 1.0)))
            for mem in ('mlp_res_xr9', 'mlp_raw_B9'):
                U += [(1, (mem, 'cur', a, s, 1.0)), (1, (mem, 'nodw', a, s, 1.0))]
        U.append((1, ('lgb_res', 'none', a, 1, 1.0)))                               # deterministic (no bagging)
        for ps in ('leaf20', 'strong', 'nol2bag', 'default'):
            U.append((3, ('lgb_res', ps, a, 1, 1.0)))
        for mem in ('mlp_res_xr9', 'mlp_raw_B9'):
            U.append((3, (mem, 'strong', a, 1, 1.0)))
    for frac in (0.5, 0.25, 0.125):
        for a in F:
            for s in (1, 2):
                U += [(2, ('lgb_res', 'cur', a, s, frac)), (2, ('mlp_res_xr9', 'cur', a, s, frac)), (2, ('mlp_res_xr9', 'nodw', a, s, frac))]
            U.append((2, ('lgb_res', 'none', a, 1, frac)))
    return U


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=2); ap.add_argument('--only', default='')
    a = ap.parse_args()
    _G['P'] = L.panel()
    TU = all_units()
    if a.only:
        TU = [x for x in TU if str(x[0]) in a.only.split(',')]
    U = [u for _, u in TU]
    fi = {fa: i for i, (fa, _) in enumerate(L.FOLDS)}
    tier = {u: t for t, u in TU}
    # by tier; inside a tier the slow ones first (LightGBM, late folds, bigger fractions)
    todo = sorted([u for u in U if not os.path.exists(unit_file(u))], key=lambda u: (tier[u], -u[4], u[0] != 'lgb_res', -fi[u[2]]))
    print(f'{len(U)} units, {len(todo)} to do', flush=True)
    import multiprocessing as mp
    t0 = time.time()
    pool = mp.get_context('fork').Pool(a.workers)
    for i, (u, sec, err) in enumerate(pool.imap_unordered(run_unit, todo), 1):
        if err:
            print('FAILED', u, err, flush=True)
        print(f'[{i}/{len(todo)}] {u} {sec:.0f}s, elapsed {(time.time() - t0) / 60:.1f} min', flush=True)
    pool.close(); pool.join()


if __name__ == '__main__':
    main()

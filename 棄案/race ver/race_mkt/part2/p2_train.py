r"""
p2_train.py -- race_mkt part 2 (part2\PREREG_mkt_part2.md): the C group + the 37 futures positioning features in the stock
model (LU, race_std learners.fit_LU unchanged), D folds 2022 / 2023 / 2024, seeds 1-3.

  python p2_train.py [--budget 170]    trains what is missing (first the harness check C.LU 2024 seed 1), then stops
  python p2_train.py --status
Units: ..\cache\units\<unit>\<fold>_seed<s>.npz (rslib format). Finished units are never redone.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '1')
import sys, json, time, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); MKT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(MKT, 'research'))
os.environ.setdefault('RACE_THREADS', '2')
import mktlib as K                                                    # noqa: E402
L = K.L
import learners                                                       # noqa: E402  (race_std, frozen)

UNITS = os.path.join(MKT, 'cache', 'units')
RES2 = os.path.join(HERE, 'results')
LOG = L.log_to(os.path.join(RES2, 'train.log'))
TASKS = [('C.LU', '2024-01-01', '2025-01-01', 1)] + [('CF.LU', fa, fb, s) for fa, fb in L.D_FOLDS for s in (1, 2, 3)]


def unit_file(u, fa, s):
    return os.path.join(UNITS, u, f'{fa}_seed{s}.npz')


def fit_LU_imp(P, feats, tr, te, seed):
    """= learners.fit_LU (same code path) + gain importance of the refit model"""
    import lightgbm as lgb
    e = P['resid_r'].to_numpy(float); d = P['date'].to_numpy()
    y = (e > 0).astype(float)
    w = np.minimum(np.abs(e), 0.05)
    w = w / np.nanmean(w[tr])
    prm = dict(objective='binary', learning_rate=0.02, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=learners.THREADS, verbose=-1, seed=seed,
               deterministic=True, force_col_wise=True)
    fit, val = learners._split_last_days(d, tr, learners.ES_DAYS)
    X = lambda m: P.loc[m, feats].to_numpy(np.float32)
    dtr = lgb.Dataset(X(fit), y[fit], weight=w[fit], free_raw_data=True)
    dva = lgb.Dataset(X(val), y[val], weight=w[val], reference=dtr)
    b = lgb.train(prm, dtr, num_boost_round=1000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    best = max(int(b.best_iteration or b.current_iteration()), 1)
    full = lgb.train(prm, lgb.Dataset(X(tr), y[tr], weight=w[tr]), num_boost_round=best)
    imp = dict(zip(feats, full.feature_importance('gain').astype(float)))
    return full.predict(X(te)), dict(best_iter=best, importance=imp)


def load():
    P = pd.read_pickle(L.PANEL)
    P['date'] = pd.to_datetime(P.date).astype('datetime64[ns]'); P['stock_id'] = P.stock_id.astype(str)
    C = L.features()['C']
    P = P[['date', 'stock_id', 'target_r', 'resid_r'] + C].copy()
    days = pd.DatetimeIndex(sorted(P.date.unique()))
    FUT, dropped = K.futures_features(days)
    FUT = FUT.add_prefix('FUT__')
    P = P.merge(FUT, left_on='date', right_index=True, how='left')
    return P, C, list(FUT.columns), dropped


def status():
    for t in TASKS:
        print(t, 'done' if os.path.exists(unit_file(t[0], t[1], t[3])) else 'missing')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--budget', type=float, default=1e9); ap.add_argument('--status', action='store_true')
    a = ap.parse_args()
    if a.status:
        return status()
    t_start = time.time()
    todo = [t for t in TASKS if not os.path.exists(unit_file(t[0], t[1], t[3]))]
    if not todo:
        LOG('all units exist'); return
    P, C, FUTC, dropped = load()
    LOG(f'panel {P.shape}, C {len(C)}, FUT {len(FUTC)} (dropped {dropped}), threads {learners.THREADS}')
    d = P['date'].to_numpy()
    lab = P['target_r'].notna().to_numpy() & P['resid_r'].notna().to_numpy()
    for (u, fa, fb, s) in TASKS:
        f = unit_file(u, fa, s)
        if os.path.exists(f):
            continue
        if u == 'CF.LU' and not os.path.exists(unit_file('C.LU', '2024-01-01', 1)):
            LOG('harness check missing: stop'); return
        if time.time() - t_start > a.budget:
            LOG('budget reached: run again to continue'); return
        feats = C if u == 'C.LU' else C + FUTC
        tr = lab & (d >= np.datetime64(L.TRAIN_START)) & (d < np.datetime64(pd.Timestamp(fa)))
        te = (d >= np.datetime64(pd.Timestamp(fa))) & (d < np.datetime64(pd.Timestamp(fb)))
        assert d[tr].max() < np.datetime64(pd.Timestamp(fa))
        t0 = time.time()
        pred, meta = fit_LU_imp(P, feats, tr, te, s)
        meta.update(unit=u, fold=fa, seed=s, n_feats=len(feats), train_last=str(pd.Timestamp(d[tr].max()).date()))
        L.save_unit(f, d[te], P['stock_id'].to_numpy()[te], pred, meta)
        LOG(f'{u} {fa} seed {s}: best_iter {meta["best_iter"]}, {time.time() - t0:.0f}s')
        if u == 'C.LU':
            ref, mref = L.load_npz(os.path.join(L.CACHE, 'units', 'C.LU', f'{fa}_seed{s}.npz'))
            new, _ = L.load_npz(f)
            j = ref.index.intersection(new.index)
            dmax = float((ref[j] - new[j]).abs().max())
            LOG(f'harness check vs race_std C.LU {fa} seed {s}: rows {len(j)}/{len(ref)}, max |diff| {dmax:.3g}, '
                f'best_iter {meta["best_iter"]} vs {mref.get("best_iter")}')
            if dmax != 0.0 or len(j) != len(ref):
                os.replace(f, f + '.FAILED'); LOG('HARNESS CHECK FAILED: stop'); return
    LOG('done')


if __name__ == '__main__':
    main()

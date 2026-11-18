r"""
agg_learners.py -- the trained candidates of race_agg (research\PREREG_race_agg.md section 3).

Every fit_* takes (P, X, feats, tr, te, seed, fa) and returns (pred for the te rows, meta). X = the C feature matrix (float32, all
panel rows). LU parameters = part-1 learners.fit_LU. Early stopping / selection only on the training window: fit part = window
minus its last 120 trading days, validation = those 120 days; then refit on the whole window.
"""
import os, json, time
import numpy as np, pandas as pd
import agglib as A
import learners as LR                                         # part-1 learners (read-only)

LU_PRM = dict(objective='binary', learning_rate=0.02, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=1, verbose=-1, deterministic=True,
              force_col_wise=True)


def _wlogloss(p, y, w):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(np.sum(w * -(y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w))


def _best(b):
    return max(int(b.best_iteration or b.current_iteration()), 1)


# ------------------------------------------------------------------ B1: more seeds of LU (= part-1 fit_LU)
def fit_B1(P, X, feats, tr, te, seed, fa):
    return LR.fit_LU(P, feats, tr, te, seed)


# ------------------------------------------------------------------ B2: day-bootstrap bagging of LU
def fit_B2(P, X, feats, tr, te, seed, fa):
    import lightgbm as lgb
    e, y, w = A.label_weight(P, tr)
    d = P['date'].to_numpy()
    days = np.unique(d[tr]); cut = days[-A.ES_DAYS]
    rng = np.random.default_rng(10000 * int(fa[:4]) + seed)
    pos = rng.integers(0, len(days), len(days))                 # n_days draws with replacement (whole days)
    idx = np.flatnonzero(tr)
    idx = idx[np.argsort(d[idx], kind='stable')]
    dd = d[idx]
    st = np.searchsorted(dd, days, 'left'); en = np.searchsorted(dd, days, 'right')
    rows = np.concatenate([idx[st[k]:en[k]] for k in pos])
    fit_rows = rows[d[rows] < cut]
    val = tr & (d >= cut)
    prm = dict(LU_PRM, seed=seed)
    dtr = lgb.Dataset(X[fit_rows], y[fit_rows], weight=w[fit_rows], free_raw_data=True)
    dva = lgb.Dataset(X[val], y[val], weight=w[val], reference=dtr)
    b = lgb.train(prm, dtr, num_boost_round=1000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    best = _best(b)
    full = lgb.train(prm, lgb.Dataset(X[rows], y[rows], weight=w[rows]), num_boost_round=best)
    return full.predict(X[te]), dict(best_iter=best, unique_days=int(len(np.unique(pos))), n_days=int(len(days)),
                                     n_rows=int(len(rows)))


# ------------------------------------------------------------------ C1: AdaBoost-Stump (exact weighted-error stumps on 255 bins)
NB = 255                                                      # bins 0..254 for values, 255 = missing


def make_edges(Xt):
    edges = []
    qs = np.linspace(0, 1, NB + 1)[1:-1]
    for j in range(Xt.shape[1]):
        x = Xt[:, j]; x = x[~np.isnan(x)]
        if len(x) == 0:
            edges.append(np.zeros(0, np.float32)); continue
        q = np.unique(np.quantile(x.astype(np.float64), qs).astype(np.float32))
        edges.append(q)
    return edges


def apply_bins(X, edges):
    B = np.empty(X.shape, np.uint8)
    for j, ed in enumerate(edges):
        x = X[:, j]
        b = np.searchsorted(ed, x, side='right')              # number of edges <= x: 0 .. len(ed)
        b[np.isnan(x)] = NB
        B[:, j] = b
    return B


def _bin_matrix(B):
    import scipy.sparse as sp
    n, p = B.shape
    cols = (np.arange(p, dtype=np.int64)[None, :] * (NB + 1) + B.astype(np.int64)).ravel()
    M = sp.csr_matrix((np.ones(n * p), cols, np.arange(0, n * p + 1, p)), shape=(n, p * (NB + 1)))
    return M.T.tocsr()                                         # (p*256, n)


def _stump_predict(B, j, k, s, nan_right):
    b = B[:, j].astype(np.int16)
    right = np.where(b == NB, nan_right, b > k)
    return np.where(right, s, -s).astype(np.float64)


def adaboost_stump(Xf, ytf, wf, T, Xv=None, ytv=None, wv=None):
    """discrete AdaBoost with weighted-error decision stumps; returns (stumps, edges, val exponential losses)"""
    edges = make_edges(Xf)
    M = np.array([len(e) for e in edges])
    Bf = apply_bins(Xf, edges)
    BT = _bin_matrix(Bf)
    Bv = apply_bins(Xv, edges) if Xv is not None else None
    u = wf / wf.sum()
    pos = (ytf > 0)
    valid = np.arange(NB)[None, :] < M[:, None]                # split after bin k needs k <= len(edges) - 1
    stumps, losses = [], []
    Gv = np.zeros(len(ytv)) if Xv is not None else None
    for t in range(T):
        H = BT @ np.column_stack([u * pos, u * ~pos])          # (p*256, 2) weighted counts of y = +1 / -1 per bin
        Hp = H[:, 0].reshape(-1, NB + 1); Hn = H[:, 1].reshape(-1, NB + 1)
        nanp, nann = Hp[:, NB], Hn[:, NB]
        Cp = np.cumsum(Hp[:, :NB], axis=1); Cn = np.cumsum(Hn[:, :NB], axis=1)
        Pt, Nt = Cp[:, -1:], Cn[:, -1:]
        e1 = Cp + (Nt - Cn)                                    # s = +1: left -> -1, right -> +1
        e0 = Cn + (Pt - Cp)                                    # s = -1: left -> +1, right -> -1
        E = np.stack([e1 + nanp[:, None], e1 + nann[:, None], e0 + nann[:, None], e0 + nanp[:, None]])
        E[:, ~valid] = np.inf
        o, j, k = np.unravel_index(int(np.argmin(E)), E.shape)
        eps = float(E[o, j, k]) / float(u.sum())
        if not (0 < eps < 0.5):
            break
        s = 1.0 if o in (0, 1) else -1.0
        nan_right = o in (1, 3)
        h = _stump_predict(Bf, j, k, s, nan_right)
        dia = np.sqrt((1 - eps) / eps); alpha = float(np.log(dia))
        wrong = h != ytf
        u = np.where(wrong, u * dia, u / dia); u = u / u.sum()
        stumps.append((int(j), int(k), s, bool(nan_right), alpha, float(eps)))
        if Xv is not None:
            Gv += alpha * _stump_predict(Bv, j, k, s, nan_right)
            losses.append(float(np.sum(wv * np.exp(np.clip(-ytv * Gv, -50, 50))) / np.sum(wv)))
    return stumps, edges, losses


def predict_stumps(X, edges, stumps):
    B = apply_bins(X, edges)
    G = np.zeros(len(X))
    for j, k, s, nr, a, _ in stumps:
        G += a * _stump_predict(B, j, k, s, nr)
    return G


def fit_C1(P, X, feats, tr, te, seed, fa, T_max=1000):
    e, y, w = A.label_weight(P, tr)
    yt = 2 * y - 1
    d = P['date'].to_numpy()
    fit, val = A.split_last(d, tr)
    st, ed, losses = adaboost_stump(X[fit], yt[fit], w[fit], T_max, X[val], yt[val], w[val])
    T = int(np.argmin(losses)) + 1
    st2, ed2, _ = adaboost_stump(X[tr], yt[tr], w[tr], T)
    pred = predict_stumps(X[te], ed2, st2)
    return pred, dict(T=T, T_ran=len(st), val_loss_T=losses[T - 1], val_loss_1=losses[0], val_loss_last=losses[-1],
                      eps_first=st[0][5], eps_T=st[T - 1][5], eps_refit_mean=float(np.mean([s[5] for s in st2])),
                      features=[feats[s[0]] for s in st2[:20]])


# ------------------------------------------------------------------ C2: AdaBoost-DTree (weighted-Gini C&RT, depth 3)
def adaboost_tree(Xf, ytf, wf, T, Xv=None, ytv=None, wv=None, seed=0):
    import lightgbm as lgb
    n = len(ytf)
    prm = dict(objective='none', learning_rate=1.0, num_leaves=8, max_depth=3, min_data_in_leaf=500, min_sum_hessian_in_leaf=0.0,
               lambda_l2=0.0, lambda_l1=0.0, min_gain_to_split=0.0, feature_fraction=1.0, bagging_fraction=1.0, bagging_freq=0,
               max_bin=255, num_threads=1, verbose=-1, deterministic=True, force_col_wise=True, seed=seed)
    ds = lgb.Dataset(Xf, label=(ytf > 0).astype(float), free_raw_data=False, params={'max_bin': 255, 'verbose': -1})
    bst = lgb.Booster(params=prm, train_set=ds)
    u = wf / wf.sum()
    state = {'u': u}

    def fobj(preds, data):
        uu = state['u'] * n                                    # mean 1
        return -uu * ytf, uu                                   # leaf value = sum(u y~) / sum(u): sign = weighted majority
    alphas, eps_list, losses = [], [], []
    Gv = np.zeros(len(ytv)) if Xv is not None else None
    for t in range(T):
        bst.update(fobj=fobj)
        if bst.current_iteration() <= t:
            break
        hf = np.where(bst.predict(Xf, start_iteration=t, num_iteration=1, raw_score=True) >= 0, 1.0, -1.0)
        u = state['u']
        eps = float(np.sum(u[hf != ytf]) / np.sum(u))
        if not (0 < eps < 0.5):
            break
        dia = np.sqrt((1 - eps) / eps); alpha = float(np.log(dia))
        u = np.where(hf != ytf, u * dia, u / dia); state['u'] = u / u.sum()
        alphas.append(alpha); eps_list.append(eps)
        if Xv is not None:
            hv = np.where(bst.predict(Xv, start_iteration=t, num_iteration=1, raw_score=True) >= 0, 1.0, -1.0)
            Gv += alpha * hv
            losses.append(float(np.sum(wv * np.exp(np.clip(-ytv * Gv, -50, 50))) / np.sum(wv)))
    return bst, alphas, eps_list, losses


def predict_adatree(bst, alphas, X):
    G = np.zeros(len(X))
    for t, a in enumerate(alphas):
        G += a * np.where(bst.predict(X, start_iteration=t, num_iteration=1, raw_score=True) >= 0, 1.0, -1.0)
    return G


def fit_C2(P, X, feats, tr, te, seed, fa, T_max=500):
    e, y, w = A.label_weight(P, tr)
    yt = 2 * y - 1
    d = P['date'].to_numpy()
    fit, val = A.split_last(d, tr)
    _, al, ep, losses = adaboost_tree(X[fit], yt[fit], w[fit], T_max, X[val], yt[val], w[val])
    T = int(np.argmin(losses)) + 1
    bst2, al2, ep2, _ = adaboost_tree(X[tr], yt[tr], w[tr], T)
    pred = predict_adatree(bst2, al2, X[te])
    return pred, dict(T=T, T_ran=len(al), val_loss_T=losses[T - 1], val_loss_1=losses[0], val_loss_last=losses[-1],
                      eps_first=ep[0], eps_T=ep[T - 1], eps_refit_mean=float(np.mean(ep2)), T_refit=len(al2))


# ------------------------------------------------------------------ C3: GBDT with the exponential (AdaBoost) loss
def fit_C3(P, X, feats, tr, te, seed, fa):
    import lightgbm as lgb
    e, y, w = A.label_weight(P, tr)
    yt = 2 * y - 1
    d = P['date'].to_numpy()
    fit, val = A.split_last(d, tr)

    def make_obj(ww, yy):
        def fobj(preds, data):
            ex = np.exp(np.clip(-yy * preds, -50, 50))
            return -ww * yy * ex, ww * ex
        return fobj
    wv, yv = w[val], yt[val]

    def feval(preds, data):
        return 'wexp', float(np.sum(wv * np.exp(np.clip(-yv * preds, -50, 50))) / np.sum(wv)), False
    prm = dict(LU_PRM, seed=seed, metric='None')
    prm['objective'] = make_obj(w[fit], yt[fit])
    dtr = lgb.Dataset(X[fit], y[fit], free_raw_data=True)
    dva = lgb.Dataset(X[val], y[val], reference=dtr)
    b = lgb.train(prm, dtr, num_boost_round=1000, valid_sets=[dva], feval=feval,
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    best = _best(b)
    prm2 = dict(prm); prm2['objective'] = make_obj(w[tr], yt[tr])
    full = lgb.train(prm2, lgb.Dataset(X[tr], y[tr]), num_boost_round=best)
    return full.predict(X[te]), dict(best_iter=best)


# ------------------------------------------------------------------ D1: single C&RT with cost-complexity pruning
def fit_D1(P, X, feats, tr, te, seed, fa, n_alpha=30, min_leaf=100):
    from sklearn.tree import DecisionTreeClassifier
    e, y, w = A.label_weight(P, tr)
    d = P['date'].to_numpy()
    fit, val = A.split_last(d, tr)
    base = DecisionTreeClassifier(criterion='gini', min_samples_leaf=min_leaf, random_state=seed)
    path = base.cost_complexity_pruning_path(X[fit], y[fit], sample_weight=w[fit])
    al = np.unique(path.ccp_alphas)
    al = al[:-1] if len(al) > 1 else al                        # the last alpha prunes to the root
    pick = np.unique(np.round(np.linspace(0, len(al) - 1, n_alpha)).astype(int))
    res = []
    for a in al[pick]:
        t = DecisionTreeClassifier(criterion='gini', min_samples_leaf=min_leaf, random_state=seed, ccp_alpha=float(a))
        t.fit(X[fit], y[fit], sample_weight=w[fit])
        res.append((float(a), _wlogloss(t.predict_proba(X[val])[:, 1], y[val], w[val]), int(t.get_n_leaves())))
    a_best = min(res, key=lambda r: r[1])[0]
    full = DecisionTreeClassifier(criterion='gini', min_samples_leaf=min_leaf, random_state=seed, ccp_alpha=a_best)
    full.fit(X[tr], y[tr], sample_weight=w[tr])
    return full.predict_proba(X[te])[:, 1], dict(alpha=a_best, leaves=int(full.get_n_leaves()), depth=int(full.get_depth()),
                                                  path=res, n_path_alphas=int(len(al)))


# ------------------------------------------------------------------ D2: LU with categorical dow / month
def fit_D2(P, X, feats, tr, te, seed, fa):
    import lightgbm as lgb
    e, y, w = A.label_weight(P, tr)
    d = P['date'].to_numpy()
    fit, val = A.split_last(d, tr)
    cat = [feats.index('C__I_dow'), feats.index('C__I_month')]
    prm = dict(LU_PRM, seed=seed)
    dtr = lgb.Dataset(X[fit], y[fit], weight=w[fit], categorical_feature=cat, free_raw_data=True)
    dva = lgb.Dataset(X[val], y[val], weight=w[val], reference=dtr, categorical_feature=cat)
    b = lgb.train(prm, dtr, num_boost_round=1000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    best = _best(b)
    full = lgb.train(prm, lgb.Dataset(X[tr], y[tr], weight=w[tr], categorical_feature=cat), num_boost_round=best)
    return full.predict(X[te]), dict(best_iter=best)


# ------------------------------------------------------------------ E1: random forest (v15 A1 setting)
def _rf(leaf, seed, n_est=500):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=n_est, max_features='sqrt', max_samples=0.5, bootstrap=True,
                                  min_samples_leaf=leaf, criterion='gini', random_state=seed, n_jobs=1)


def fit_E1(P, X, feats, tr, te, seed, fa, leaves=(250, 1000, 4000)):
    e, y, w = A.label_weight(P, tr)
    d = P['date'].to_numpy()
    fit, val = A.split_last(d, tr)
    res = []
    for lf in leaves:
        rf = _rf(lf, seed).fit(X[fit], y[fit], sample_weight=w[fit])
        res.append((lf, _wlogloss(rf.predict_proba(X[val])[:, 1], y[val], w[val])))
    lf = min(res, key=lambda r: r[1])[0]
    rf = _rf(lf, seed).fit(X[tr], y[tr], sample_weight=w[tr])
    return rf.predict_proba(X[te])[:, 1], dict(leaf=int(lf), val=res)


# ------------------------------------------------------------------ E4: permutation-importance feature selection, then LU
def _daily_ic(pred, e, day_codes, n_days):
    out = np.empty(n_days)
    for k in range(n_days):
        m = day_codes == k
        a = pd.Series(pred[m]).rank().to_numpy(); b = pd.Series(e[m]).rank().to_numpy()
        out[k] = np.corrcoef(a, b)[0, 1]
    return float(np.nanmean(out))


def select_E4(P, X, feats, tr, fa, reps=3, min_keep=20):
    import lightgbm as lgb
    e, y, w = A.label_weight(P, tr)
    d = P['date'].to_numpy()
    fit, val = A.split_last(d, tr)
    prm = dict(LU_PRM, seed=1)
    dtr = lgb.Dataset(X[fit], y[fit], weight=w[fit], free_raw_data=True)
    dva = lgb.Dataset(X[val], y[val], weight=w[val], reference=dtr)
    b = lgb.train(prm, dtr, num_boost_round=1000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    best = _best(b)
    Xv = X[val].copy(); ev = e[val]
    codes, uniq = pd.factorize(d[val], sort=True)
    nd = len(uniq)
    base = _daily_ic(b.predict(Xv, num_iteration=best), ev, codes, nd)
    imp = np.zeros(len(feats))
    for j in range(len(feats)):
        rng = np.random.default_rng(1000 * int(fa[:4]) + j)
        col = Xv[:, j].copy(); vals = []
        for r in range(reps):
            Xv[:, j] = col[rng.permutation(len(col))]
            vals.append(_daily_ic(b.predict(Xv, num_iteration=best), ev, codes, nd))
        Xv[:, j] = col
        imp[j] = base - float(np.mean(vals))
    keep = [feats[j] for j in range(len(feats)) if imp[j] > 0]
    if len(keep) < min_keep:
        keep = [feats[j] for j in np.argsort(-imp)[:min_keep]]
    keep = [f for f in feats if f in set(keep)]                  # original order
    return dict(fold=fa, best_iter=best, base_ic=base, importance={f: float(v) for f, v in zip(feats, imp)}, keep=keep,
                n_keep=len(keep))


def sel_file(fa):
    return os.path.join(A.UNITS, 'E4', f'{fa}_selection.json')


def fit_E4(P, X, feats, tr, te, seed, fa):
    f = sel_file(fa)
    if not os.path.exists(f):
        raise RuntimeError(f'missing {f}: run the selection task first')
    sel = json.load(open(f, encoding='utf-8'))
    pred, meta = LR.fit_LU(P, sel['keep'], tr, te, seed)
    meta.update(n_keep=sel['n_keep'])
    return pred, meta


# ------------------------------------------------------------------ F1: GBDT with squared loss on clip(e, 5%)
def fit_F1(P, X, feats, tr, te, seed, fa):
    import lightgbm as lgb
    e = P['resid_r'].to_numpy(float)
    z = np.clip(e, -0.05, 0.05)
    d = P['date'].to_numpy()
    fit, val = A.split_last(d, tr)
    prm = dict(LU_PRM, seed=seed, objective='regression', metric='l2')
    dtr = lgb.Dataset(X[fit], z[fit], free_raw_data=True)
    dva = lgb.Dataset(X[val], z[val], reference=dtr)
    b = lgb.train(prm, dtr, num_boost_round=1000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    best = _best(b)
    full = lgb.train(prm, lgb.Dataset(X[tr], z[tr]), num_boost_round=best)
    return full.predict(X[te]), dict(best_iter=best)


FIT = {'B1': fit_B1, 'B2': fit_B2, 'C1': fit_C1, 'C2': fit_C2, 'C3': fit_C3, 'D1': fit_D1, 'D2': fit_D2, 'E1': fit_E1,
       'E4': fit_E4, 'F1': fit_F1}


# ------------------------------------------------------------------ g^- of the 5 learners on the last 120 days (A3, A4)
def fit_GM(P, X, feats, tr, te, seed, fa, learner):
    d = P['date'].to_numpy()
    fit, val = A.split_last(d, tr)
    pred, meta = LR.fit_predict(learner, P, feats, fit, val, seed)
    return pred, meta, val

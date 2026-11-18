r"""
learners.py -- the five learners of RACE-STD v1 (research\PREREG_race_std.md section 2.2).

fit_predict(name, P, feats, tr, te, seed, test=False) -> (pred, meta)
  P      the panel (DataFrame with date, resid_r, absres_60 and the feature columns)
  tr, te boolean numpy masks of the training rows (labelled, TRAIN_START <= date < fold start) and the rows to predict
  pred   numpy array for the te rows, in the learner's own units (higher = more likely to beat the day)

  LU  Baszo lgb_res      LightGBM binary on 1{e > 0}, weight min(|e|, 5%); early stopping on the last 120 days, refit
  LN  Baszo mlp_res      MLP n-64-32-1 (GELU, dropout 0.2, AdamW, clipping), weight min(|e|, 5%) / absres_60^0.5; numpy
  LP  FinPred v2.3       sklearn HistGradientBoostingRegressor on clip(e, +-5%), 800 iterations, no early stopping
  LC  Chen V5            LightGBM huber on z = clip(e / sigma20, +-5), early stopping on the last 15% of dates, refit; x sigma20
  LM  NTU M2             median imputer + LightGBM regression on 100 clip(e, +-8%), 260 trees
"""
import os, time
import numpy as np, pandas as pd

ES_DAYS = 120
THREADS = int(os.environ.get("RACE_THREADS", "1"))      # col-wise LightGBM / HistGBR: results do not depend on it


def _dates(P):
    return P['date'].to_numpy()


def _split_last_days(d, tr, k):
    days = np.unique(d[tr])
    cut = days[-k] if len(days) > k else days[len(days) // 2]
    return tr & (d < cut), tr & (d >= cut)


def _X(P, feats, m):
    return P.loc[m, feats].to_numpy(np.float32)


# ------------------------------------------------------------------ LU
def fit_LU(P, feats, tr, te, seed, test=False):
    import lightgbm as lgb
    e = P['resid_r'].to_numpy(float); d = _dates(P)
    y = (e > 0).astype(float)
    w = np.minimum(np.abs(e), 0.05)
    w = w / np.nanmean(w[tr])
    prm = dict(objective='binary', learning_rate=0.02, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=THREADS, verbose=-1, seed=seed,
               deterministic=True, force_col_wise=True)
    if test:
        prm['min_data_in_leaf'] = 50
    fit, val = _split_last_days(d, tr, ES_DAYS)
    dtr = lgb.Dataset(_X(P, feats, fit), y[fit], weight=w[fit], free_raw_data=True)
    dva = lgb.Dataset(_X(P, feats, val), y[val], weight=w[val], reference=dtr)
    b = lgb.train(prm, dtr, num_boost_round=30 if test else 1000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(10 if test else 100, verbose=False)])
    best = max(int(b.best_iteration or b.current_iteration()), 1)
    full = lgb.train(prm, lgb.Dataset(_X(P, feats, tr), y[tr], weight=w[tr]), num_boost_round=best)
    return full.predict(_X(P, feats, te)), dict(best_iter=best)


# ------------------------------------------------------------------ LN (numpy MLP = torch Linear / GELU / Dropout / AdamW)
SQ2 = np.float32(np.sqrt(2.0)); ISQ2PI = np.float32(1.0 / np.sqrt(2 * np.pi))


def _erf(x):
    from scipy.special import erf
    return erf(x).astype(np.float32)


def _gelu(x):
    return np.float32(0.5) * x * (np.float32(1.0) + _erf(x / SQ2))


def _dgelu(x):
    return np.float32(0.5) * (np.float32(1.0) + _erf(x / SQ2)) + x * np.exp(np.float32(-0.5) * x * x) * ISQ2PI


class Scaler:
    """1% / 99% clipping, then mean / sd of the clipped training values; missing -> 0 (= transformer\\tfmodel.Scaler)"""
    def __init__(self, X):
        self.lo = np.nanquantile(X, 0.01, axis=0).astype(np.float32)
        self.hi = np.nanquantile(X, 0.99, axis=0).astype(np.float32)
        Xc = np.clip(X, self.lo, self.hi)
        self.mu = np.nanmean(Xc, axis=0).astype(np.float32)
        sd = np.nanstd(Xc, axis=0).astype(np.float32)
        self.sd = np.where((sd > 0) & np.isfinite(sd), sd, 1.0).astype(np.float32)
        self.lo = np.nan_to_num(self.lo); self.hi = np.nan_to_num(self.hi); self.mu = np.nan_to_num(self.mu)

    def __call__(self, X):
        Z = (np.clip(X, self.lo, self.hi) - self.mu) / self.sd
        return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class MLP:
    def __init__(self, n, seed, drop=0.2, lr=1e-3, wd=1e-2):
        rng = np.random.default_rng(seed)
        def lin(i, o):
            bd = 1.0 / np.sqrt(i)
            return [rng.uniform(-bd, bd, (i, o)).astype(np.float32), rng.uniform(-bd, bd, o).astype(np.float32)]
        self.p = lin(n, 64) + lin(64, 32) + lin(32, 1)
        self.m = [np.zeros_like(x) for x in self.p]; self.v = [np.zeros_like(x) for x in self.p]
        self.t = 0; self.drop, self.lr, self.wd = drop, lr, wd
        self.rng = np.random.default_rng(seed + 7919)

    def forward(self, X, train=False):
        W1, b1, W2, b2, W3, b3 = self.p
        h1 = X @ W1 + b1; a1 = _gelu(h1)
        m1 = m2 = None
        if train:
            keep = 1.0 - self.drop
            m1 = (self.rng.random(a1.shape) < keep).astype(np.float32) / np.float32(keep); a1 = a1 * m1
        h2 = a1 @ W2 + b2; a2 = _gelu(h2)
        if train:
            m2 = (self.rng.random(a2.shape) < keep).astype(np.float32) / np.float32(keep); a2 = a2 * m2
        z = (a2 @ W3 + b3)[:, 0]
        return z, (X, h1, a1, m1, h2, a2, m2)

    def step(self, X, y, w):
        z, (X, h1, a1, m1, h2, a2, m2) = self.forward(X, train=True)
        W1, b1, W2, b2, W3, b3 = self.p
        s = 1.0 / (1.0 + np.exp(-z))
        g = (w * (s - y) / len(y)).astype(np.float32)[:, None]          # d mean(w * BCE) / dz
        dW3 = a2.T @ g; db3 = g.sum(0)
        da2 = (g @ W3.T) * m2; dh2 = da2 * _dgelu(h2)
        dW2 = a1.T @ dh2; db2 = dh2.sum(0)
        da1 = (dh2 @ W2.T) * m1; dh1 = da1 * _dgelu(h1)
        dW1 = X.T @ dh1; db1 = dh1.sum(0)
        grads = [dW1, db1, dW2, db2, dW3, db3]
        norm = np.sqrt(sum(float((gg.astype(np.float64) ** 2).sum()) for gg in grads))
        if norm > 1.0:                                                  # clip_grad_norm_(1.0)
            grads = [gg * np.float32(1.0 / (norm + 1e-6)) for gg in grads]
        self.t += 1
        b1c, b2c = 1 - 0.9 ** self.t, 1 - 0.999 ** self.t
        for i, gg in enumerate(grads):                                  # torch.optim.AdamW
            self.p[i] *= np.float32(1 - self.lr * self.wd)
            self.m[i] = 0.9 * self.m[i] + 0.1 * gg
            self.v[i] = 0.999 * self.v[i] + 0.001 * gg * gg
            self.p[i] -= (self.lr * (self.m[i] / b1c) / (np.sqrt(self.v[i] / b2c) + 1e-8)).astype(np.float32)

    def predict(self, X, bs=8192):
        out = [self.forward(X[i:i + bs])[0] for i in range(0, len(X), bs)]
        return 1.0 / (1.0 + np.exp(-np.concatenate(out)))


def _wlogloss(p, y, w):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(np.sum(w * -(y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w))


def _train_mlp(Xtr, ytr, wtr, seed, epochs, Xva=None, yva=None, wva=None, patience=2, bs=1024):
    net = MLP(Xtr.shape[1], seed)
    rng = np.random.default_rng(seed + 104729)
    hist, best, best_ep, bad = [], np.inf, 1, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            net.step(Xtr[j], ytr[j], wtr[j])
        if Xva is not None:
            l = _wlogloss(net.predict(Xva), yva, wva); hist.append(l)
            if l < best - 1e-7:
                best, best_ep, bad = l, ep, 0
            else:
                bad += 1
                if bad >= patience:
                    break
    return net, best_ep, hist


def fit_LN(P, feats, tr, te, seed, test=False):
    e = P['resid_r'].to_numpy(float); d = _dates(P)
    y = (e > 0).astype(np.float32)
    sig = P['absres_60'].to_numpy(float)
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.nanmedian(sig[tr]))
    w = np.minimum(np.abs(e), 0.05) / np.sqrt(sig)
    fit, val = _split_last_days(d, tr, ES_DAYS)
    Xf = _X(P, feats, fit); sc = Scaler(Xf)
    wf = (w[fit] / w[fit].mean()).astype(np.float32); wv = (w[val] / w[fit].mean()).astype(np.float32)
    _, best_ep, hist = _train_mlp(sc(Xf), y[fit], wf, seed, 1 if test else 10, sc(_X(P, feats, val)), y[val], wv)
    Xa = _X(P, feats, tr); sca = Scaler(Xa)
    wa = (w[tr] / w[tr].mean()).astype(np.float32)
    net, _, _ = _train_mlp(sca(Xa), y[tr], wa, seed, best_ep)
    return net.predict(sca(_X(P, feats, te))), dict(best_epoch=int(best_ep), val_loss=[round(x, 6) for x in hist])


# ------------------------------------------------------------------ LP
def fit_LP(P, feats, tr, te, seed, test=False):
    from sklearn.ensemble import HistGradientBoostingRegressor
    e = P['resid_r'].to_numpy(float)
    m = HistGradientBoostingRegressor(loss='squared_error', max_iter=40 if test else 800, learning_rate=0.0075, max_leaf_nodes=7,
                                      min_samples_leaf=200 if test else 2000, l2_regularization=5.0, early_stopping=False,
                                      random_state=seed)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=THREADS, user_api='openmp'):
        m.fit(_X(P, feats, tr), np.clip(e[tr], -0.05, 0.05))
        pred = m.predict(_X(P, feats, te))
    return pred, dict(iters=int(m.n_iter_))


# ------------------------------------------------------------------ LC
SIGMA_COL = 'C__intra_std_20'


def fit_LC(P, feats, tr, te, seed, test=False):
    import lightgbm as lgb
    e = P['resid_r'].to_numpy(float); d = _dates(P)
    sig = P[SIGMA_COL].to_numpy(float)
    ok = tr & np.isfinite(sig) & (sig > 0)
    z = np.clip(e / np.where(sig > 0, sig, np.nan), -5, 5)
    prm = dict(objective='huber', learning_rate=0.03, num_leaves=15, max_depth=5, min_data_in_leaf=50 if test else 300,
               lambda_l1=1.0, lambda_l2=10.0, num_threads=THREADS, verbose=-1, seed=seed, deterministic=True, force_col_wise=True)
    days = np.unique(d[ok]); cut = days[int(np.floor(len(days) * 0.85))]
    fit, val = ok & (d < cut), ok & (d >= cut)
    dtr = lgb.Dataset(_X(P, feats, fit), z[fit], free_raw_data=True)
    dva = lgb.Dataset(_X(P, feats, val), z[val], reference=dtr)
    b = lgb.train(prm, dtr, num_boost_round=30 if test else 2000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(10 if test else 100, verbose=False)])
    best = max(int(b.best_iteration or b.current_iteration()), 1)
    full = lgb.train(prm, lgb.Dataset(_X(P, feats, ok), z[ok]), num_boost_round=best)
    st = sig[te].copy()
    dd = pd.Series(st).groupby(d[te]).transform('median').to_numpy()
    st = np.where(np.isfinite(st) & (st > 0), st, dd)
    st = np.where(np.isfinite(st) & (st > 0), st, np.nanmedian(sig[ok]))
    return full.predict(_X(P, feats, te)) * st, dict(best_iter=best)


# ------------------------------------------------------------------ LM
def fit_LM(P, feats, tr, te, seed, test=False):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from lightgbm import LGBMRegressor
    e = P['resid_r'].to_numpy(float)
    m = Pipeline([('impute', SimpleImputer(strategy='median')),
                  ('lgbm', LGBMRegressor(objective='regression', n_estimators=30 if test else 260, learning_rate=0.025,
                                         num_leaves=9, min_child_samples=250, reg_lambda=60.0, colsample_bytree=0.85,
                                         random_state=seed, deterministic=True, force_col_wise=True, n_jobs=THREADS, verbosity=-1))])
    Xtr = pd.DataFrame(_X(P, feats, tr), columns=feats)
    m.fit(Xtr, np.clip(e[tr], -0.08, 0.08) * 100)
    return m.predict(pd.DataFrame(_X(P, feats, te), columns=feats)), dict(trees=30 if test else 260)


FIT = {'LU': fit_LU, 'LN': fit_LN, 'LP': fit_LP, 'LC': fit_LC, 'LM': fit_LM}


def fit_predict(name, P, feats, tr, te, seed, test=False):
    t0 = time.time()
    pred, meta = FIT[name](P, feats, tr, te, seed, test)
    meta.update(seconds=round(time.time() - t0, 2), n_train=int(tr.sum()), n_pred=int(te.sum()), n_feat=len(feats))
    return np.asarray(pred, np.float64), meta

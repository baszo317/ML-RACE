r"""
v15_train.py -- walk-forward training of the v15 candidates that need a model (research\PREREG_v15.md section 2):

  A  RF / ExtraTrees (scikit-learn)          A1 RF, A2 ExtraTrees, A0 sklearn-default RF (report only)
  B  lgb_res de-noising (LightGBM)           B0 replica (control), B1 sigma weights, B2 Huber weights, B3 3-day target,
                                             B4 5-day target, B5 no ex-dividend rows, B6 DoubleEnsemble (Qlib)
  C  MLP de-noising (torch)                  C0 mlp_res_xr9 replica (control), C1 multi-task, C2 supervised autoencoder + MLP

6 yearly folds (2022, 2023, 2024, 2025, 2026-01..05, post), training window 2020-03-02 .. the day before the fold, early
stopping / parameter choice on the last 120 training days, then refit on the whole window. Every unit is saved as
..\cache\units\<name>\<fold>_seed<s>.npz and never redone: stop any time, start again, it continues.

usage (in v15\research):  python v15_train.py [--members B0,B1,...] [--workers 2] [--threads N] [--test]
  --test   tiny models on the last ~20,000 rows of each window -> ..\cache\units_test\ (a few minutes; checks that everything runs)
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, time, json, argparse, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v15lib as L

_G = {}
TEST_ROWS = 20000


def log(msg, test=False):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)
    os.makedirs(L.RESULTS, exist_ok=True)
    with open(os.path.join(L.RESULTS, 'v15_train_test.log' if test else 'v15_train.log'), 'a', encoding='utf-8') as fh:
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
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))
    except Exception:
        pass


# ------------------------------------------------------------------ common pieces
def window(P, fa, test):
    """training rows of the fold starting at fa: 2020-03-02 .. fa - 1 day, rows with a label (test: the last 20,000)"""
    te = pd.Timestamp(fa) - pd.Timedelta(days=1)
    Q = P[(P.date >= L.TRAIN_START) & (P.date <= te) & P.resid_r.notna()]
    if test:
        Q = Q.iloc[-TEST_ROWS:]
    return Q, te


def cut_date(dates):
    ud = np.sort(np.unique(dates))
    return ud[-L.VAL_DAYS] if len(ud) > 2 * L.VAL_DAYS else ud[-1]                  # = v10lib / tfmodel


def pred_rows(P, fa, fb):
    return P[(P.date >= pd.Timestamp(fa)) & (P.date < pd.Timestamp(fb)) & P.resid_r.notna()]


def logloss(y, p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def base_weight(r):
    w = np.minimum(np.abs(r), 0.05)                                                     # = lgb_res
    return w / w.mean()


# ------------------------------------------------------------------ B: lgb_res variants (B0..B5)
def lgb_design(P, Q, variant, te):
    """X, y, w, dates, last-label date of the training rows for a B variant (PREREG_v15 section 2, B)"""
    feats = L.b_features()
    last = Q.date.values
    if variant == 'noex':
        keep = np.ones(len(Q), bool)
        for k in L.EX4:
            keep &= ~(Q[k].values == 1)
        Q = Q[keep]; last = Q.date.values
    if variant in ('h3', 'h5'):
        k = 3 if variant == 'h3' else 5
        W = P[(P.date >= L.TRAIN_START) & (P.date <= te)][['date', 'stock_id', 'resid_r']].sort_values(['stock_id', 'date'])
        g = W.groupby('stock_id', sort=False)
        R = pd.concat([g.resid_r.shift(-j) for j in range(k)], axis=1)
        rbar = R.mean(axis=1, skipna=False)                                           # NaN unless all k days are in the window
        lastd = g.date.shift(-(k - 1))
        rbar = rbar.reindex(Q.index); lastd = lastd.reindex(Q.index)
        ok = rbar.notna().values
        Q = Q[ok]; r = rbar.values[ok]; last = lastd.values[ok]
        y = (r > 0).astype(float)
        w = np.minimum(np.abs(r), 0.05)
    else:
        r = Q.resid_r.values
        y = Q.y_resid.values.astype(float)
        if variant in ('base', 'noex'):
            w = np.minimum(np.abs(r), 0.05)
        elif variant == 'sigma':
            sig = Q.absres_60.values.astype(float)
            med = np.nanmedian(sig[np.isfinite(sig) & (sig > 0)])
            sig = np.where(np.isfinite(sig) & (sig > 0), sig, med)
            w = np.minimum(np.abs(r), 0.05) / np.sqrt(sig)
        elif variant == 'huber':
            W = P[(P.date >= L.TRAIN_START) & (P.date <= te) & P.resid_r.notna()]
            med = W.resid_r.abs().groupby(W.date).median()
            cap = 2.0 * med.reindex(Q.date.values).values
            w = np.minimum(np.minimum(np.abs(r), cap), 0.05)
        else:
            raise KeyError(variant)
    w = w / w.mean()
    X = Q[feats].to_numpy(np.float64)
    return X, y, w, Q.date.values, np.asarray(last, dtype='datetime64[ns]'), feats


def train_lgb_unit(P, variant, fa, fb, seed, test=False):
    import lightgbm as lgb
    Q, te = window(P, fa, test)
    X, y, wt, dates, last, feats = lgb_design(P, Q, variant, te)
    cut = cut_date(dates)
    tr = dates < cut
    if variant in ('h3', 'h5'):
        tr &= last < cut                                                                # purge: no label day on/after the cut
    va = dates >= cut
    rounds, patience = (60, 10) if test else (1000, 100)
    prm = {**L.LGB_PRM, 'seed': int(seed)}
    dtr = lgb.Dataset(X[tr], y[tr], weight=wt[tr]); dva = lgb.Dataset(X[va], y[va], weight=wt[va], reference=dtr)
    m = lgb.train(prm, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(patience, verbose=False)])
    best = max(int(m.best_iteration), 1)
    m = lgb.train(prm, lgb.Dataset(X, y, weight=wt), best)
    R = pred_rows(P, fa, fb)
    p = m.predict(R[feats].to_numpy(np.float64))
    out = pd.DataFrame({'date': R.date.values, 'stock_id': R.stock_id.values, 'p': p})
    meta = dict(best=best, rows=int(len(X)), inner_rows=int(tr.sum()), val_rows=int(va.sum()), cut=str(pd.Timestamp(cut).date()),
                w_cv=float(np.std(wt) / np.mean(wt)))
    return out, meta


# ------------------------------------------------------------------ B6: DoubleEnsemble (Zhang et al. 2020; qlib.contrib.model.double_ensemble)
DE_K, DE_ALPHA1, DE_ALPHA2, DE_BINS_SR, DE_BINS_FS, DE_DECAY = 3, 1.0, 1.0, 10, 5, 0.5
DE_RATIOS = (0.8, 0.7, 0.6, 0.5, 0.4)
DE_CK = 20                                                                              # loss-curve checkpoints
DE_FS_ROWS = 20000


def de_loss_curve(m, X, y, rounds, n_ck=DE_CK):
    """per-sample log-loss after each of n_ck checkpoints (raw scores are additive over trees)"""
    cks = sorted(set(int(np.ceil(rounds * i / n_ck)) for i in range(1, n_ck + 1)))
    raw = np.zeros(len(X)); prev = 0; C = np.empty((len(X), len(cks)), np.float32)
    for j, t in enumerate(cks):
        raw += m.predict(X, start_iteration=prev, num_iteration=t - prev, raw_score=True)
        prev = t
        C[:, j] = logloss(y, 1.0 / (1.0 + np.exp(-raw)))
    return C


def de_sample_weights(C, Lv, k_th):
    """= DEnsembleModel.sample_reweight: h = alpha1 * rank(-loss of the ensemble) + alpha2 * rank(l_end / l_start), 10 bins,
    weight = 1 / (decay^k * mean h of the bin + 0.1)"""
    Cn = pd.DataFrame(C).rank(axis=0, pct=True).values
    Ln = pd.Series(-Lv).rank(pct=True).values
    T = C.shape[1]; part = max(int(T * 0.1), 1)
    l_start = Cn[:, :part].mean(axis=1); l_end = Cn[:, -part:].mean(axis=1)
    h2 = pd.Series(l_end / l_start).rank(pct=True).values
    h = pd.Series(DE_ALPHA1 * Ln + DE_ALPHA2 * h2)
    b = pd.cut(h, DE_BINS_SR, labels=False)
    hbar = h.groupby(b).transform('mean').values
    return 1.0 / (DE_DECAY ** k_th * hbar + 0.1)


def de_feature_selection(models, Xs, ys, Ls, n_feat, rng):
    """= DEnsembleModel.feature_selection: g = mean(loss with feature f shuffled - loss) / std, 5 bins of g, keep a random
    80 / 70 / 60 / 50 / 40 % of the features of each bin, best bin first"""
    base = [m.predict(Xs[:, fk]) for m, fk in models]
    g = np.zeros(n_feat)
    for f in range(n_feat):
        perm = rng.permutation(len(Xs))
        pred = np.zeros(len(Xs))
        for (m, fk), bp in zip(models, base):
            pos = np.flatnonzero(fk == f)
            if len(pos):
                Xt = Xs[:, fk].copy(); Xt[:, pos[0]] = Xs[perm, f]
                pred += m.predict(Xt)
            else:
                pred += bp
        pred /= len(models)
        d = logloss(ys, pred) - Ls
        g[f] = d.mean() / (d.std() + 1e-7)
    g = np.nan_to_num(g)
    cats = pd.cut(pd.Series(g), DE_BINS_FS)
    keep = []
    for i, b in enumerate(sorted(cats.dropna().unique(), reverse=True)):
        fs = np.flatnonzero((cats == b).values)
        n = int(np.ceil(DE_RATIOS[min(i, len(DE_RATIOS) - 1)] * len(fs)))
        keep += list(rng.choice(fs, size=n, replace=False))
    return np.array(sorted(set(int(x) for x in keep))), g


def train_de_unit(P, fa, fb, seed, test=False):
    import lightgbm as lgb
    Q, te = window(P, fa, test)
    X, y, w0, dates, _, feats = lgb_design(P, Q, 'base', te)
    cut = cut_date(dates); tr = dates < cut
    rounds, patience = (60, 10) if test else (1000, 100)
    prm = {**L.LGB_PRM, 'seed': int(seed)}
    dtr = lgb.Dataset(X[tr], y[tr], weight=w0[tr]); dva = lgb.Dataset(X[~tr], y[~tr], weight=w0[~tr], reference=dtr)
    m = lgb.train(prm, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(patience, verbose=False)])
    R_ = max(int(m.best_iteration), 1)
    K = 2 if test else DE_K
    rng = np.random.default_rng(1000 + int(seed))
    sub = rng.choice(len(X), size=min(2000 if test else DE_FS_ROWS, len(X)), replace=False)
    fk = np.arange(X.shape[1]); wk = w0.copy()
    models, preds, nfeat, wq = [], [], [], []
    for k in range(K):
        mk = lgb.train(prm, lgb.Dataset(X[:, fk], y, weight=wk), R_)
        models.append((mk, fk.copy())); nfeat.append(int(len(fk)))
        if k + 1 == K:
            break
        C = de_loss_curve(mk, X[:, fk], y, R_)
        preds.append(mk.predict(X[:, fk]))
        Lv = logloss(y, np.mean(preds, axis=0))
        sr = de_sample_weights(C, Lv, k + 1)
        wk = w0 * sr; wk = wk / wk.mean()
        wq.append([float(np.quantile(sr, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)])
        fk, _ = de_feature_selection(models, X[sub], y[sub], Lv[sub], X.shape[1], rng)
    Rw = pred_rows(P, fa, fb); XR = Rw[feats].to_numpy(np.float64)
    p = np.mean([mk.predict(XR[:, f]) for mk, f in models], axis=0)
    out = pd.DataFrame({'date': Rw.date.values, 'stock_id': Rw.stock_id.values, 'p': p})
    meta = dict(best=R_, rows=int(len(X)), sub_models=K, n_features=nfeat, sr_weight_quantiles=wq,
                kept_last=[feats[i] for i in fk] if len(models) > 1 else feats)
    return out, meta


# ------------------------------------------------------------------ A: forests (A1 RF, A2 ExtraTrees, A0 default RF)
RF_GRID = (250, 1000, 4000)
RF_SEL_TREES, RF_TREES, RF_CHUNK = 200, 500, 50


def _impute(X, med):
    X = X.copy(); bad = np.isnan(X)
    if bad.any():
        X[bad] = np.take(med, np.nonzero(bad)[1])
    return X


def _forest(variant, n_trees, leaf, X, y, wt, preds_on, chunk, seed_base, threads):
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    if variant == 'et':
        cls, extra = ExtraTreesClassifier, dict(bootstrap=False)
    elif variant == 'rf':
        cls, extra = RandomForestClassifier, dict(bootstrap=True, max_samples=0.5)
    else:                                                                               # sklearn defaults
        cls, extra = RandomForestClassifier, dict(bootstrap=True)
    out = [np.zeros(len(Xp)) for Xp in preds_on]; imp = np.zeros(X.shape[1]); built = ci = 0
    while built < n_trees:
        nb = min(chunk, n_trees - built)
        est = cls(n_estimators=nb, min_samples_leaf=leaf, max_features='sqrt', n_jobs=threads, random_state=seed_base + ci, **extra)
        est.fit(X, y, sample_weight=wt)
        col = list(est.classes_).index(1)
        for i, Xp in enumerate(preds_on):
            out[i] += est.predict_proba(Xp)[:, col] * nb
        imp += est.feature_importances_ * nb
        built += nb; ci += 1
        del est
    return [o / n_trees for o in out], imp / n_trees


def train_rf_unit(P, variant, fa, fb, seed, test=False, threads=1):
    Q, te = window(P, fa, test)
    feats = L.b_features()
    X = Q[feats].to_numpy(np.float64)
    med = np.nanmedian(X, axis=0); med = np.where(np.isfinite(med), med, 0.0)
    X = _impute(X, med)
    y = Q.y_resid.values.astype(int); wt = base_weight(Q.resid_r.values)
    Rw = pred_rows(P, fa, fb); XR = _impute(Rw[feats].to_numpy(np.float64), med)
    dates = Q.date.values; cut = cut_date(dates); tr = dates < cut; va = ~tr
    meta = dict(rows=int(len(X)), variant=variant)
    if variant == 'default':
        n_tr, chunk = (10, 10) if test else (100, 10)
        (p,), imp = _forest(variant, n_tr, 1, X, y, wt, [XR], chunk, int(seed) * 1000 + 500, threads)
        meta.update(trees=n_tr, leaf=1)
    else:
        n_sel, n_fin, chunk = (20, 20, 10) if test else (RF_SEL_TREES, RF_TREES, RF_CHUNK)
        losses = {}
        for gi, leaf in enumerate(RF_GRID):
            (pv,), _ = _forest(variant, n_sel, leaf, X[tr], y[tr], wt[tr], [X[va]], chunk, int(seed) * 1000 + 100 * gi, threads)
            losses[leaf] = float(np.sum(wt[va] * logloss(y[va], pv, 1e-4)) / np.sum(wt[va]))
        best = min(losses, key=losses.get)
        (p,), imp = _forest(variant, n_fin, best, X, y, wt, [XR], chunk, int(seed) * 1000 + 900, threads)
        meta.update(trees=n_fin, leaf=int(best), val_logloss={str(k): v for k, v in losses.items()})
    order = np.argsort(-imp)[:10]
    meta['mdi_top10'] = [[feats[i], float(imp[i])] for i in order]
    out = pd.DataFrame({'date': Rw.date.values, 'stock_id': Rw.stock_id.values, 'p': p})
    return out, meta


# ------------------------------------------------------------------ C: MLPs (C0 replica, C1 multi-task, C2 supervised autoencoder)
E_HP = dict(d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.2, lr=1e-3, wd=1e-2, bs=1024, epochs=10, patience=2)   # = ens_lib.HP


def train_custom(Q, feats, start, train_end, seed, sig_col, test=False):
    """= v9\research\v9_train.train_custom (alpha 0.5), copied unchanged except the test switch (1 epoch)"""
    import torch, tfmodel as TM
    hp = {**dict(d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.2, lr=1e-3, wd=1e-2, bs=512, epochs=15, patience=3), **E_HP}
    if test:
        hp = {**hp, 'epochs': 1, 'patience': 1}
    torch.manual_seed(seed); np.random.seed(seed)
    D = TM.SeqData(Q, feats, 1)
    te = pd.Timestamp(train_end)
    tm = ((D.P.date >= start) & (D.P.date <= te)).values & D.P.target_r.notna().values
    scaler = TM.Scaler().fit(D.raw[tm]); D.scale(scaler)
    idx_all = D.valid(tm); dates = D.P.date.values[idx_all]
    ud = np.sort(np.unique(dates)); cut = ud[-120] if len(ud) > 240 else ud[-1]
    y = D.P.y.values.astype(np.float32)
    w = np.minimum(np.abs(D.P.target_r.fillna(0).values), 0.05)
    sig = D.P[sig_col].values.astype(float)
    med = np.nanmedian(sig[idx_all])
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, med)
    w = w / sig ** 0.5
    w = (w / w[idx_all].mean()).astype(np.float32)
    tr, va = idx_all[dates < cut], idx_all[dates >= cut]
    model = TM.make_model('mlp', len(feats), 1, dropout=hp['dropout'])
    model, best_ep = TM._fit(model, D, tr, y[tr], w[tr], hp['epochs'], hp['lr'], hp['wd'], hp['bs'], seed, val=(va, y[va], w[va]), patience=hp['patience'])
    best_ep = max(best_ep, 1)
    torch.manual_seed(seed)
    model = TM.make_model('mlp', len(feats), 1, dropout=hp['dropout'])
    model, _ = TM._fit(model, D, idx_all, y[idx_all], w[idx_all], best_ep, hp['lr'], hp['wd'], hp['bs'], seed)
    return dict(model=model.eval(), scaler=scaler, kind='mlp', seq_len=1, features=feats, hp=hp, best_epoch=best_ep)


def _mlp_classes():
    import torch, torch.nn as nn

    class MTMLP(nn.Module):
        """same trunk as tfmodel.MLPClassifier (64 -> 32, GELU, dropout); 3 outputs: day D (main), D+1, D+2"""
        def __init__(self, n, hidden=64, dropout=0.2, heads=3):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(n, hidden), nn.GELU(), nn.Dropout(dropout),
                                       nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout))
            self.head = nn.Linear(hidden // 2, heads)

        def forward(self, x):
            return self.head(self.trunk(x))

        def main(self, x):
            return self.forward(x)[:, 0]

    class AEMLP(nn.Module):
        """supervised autoencoder + MLP (Jane Street 2021, 1st place), sized like our MLPs"""
        def __init__(self, n, enc=32, hidden=64, dropout=0.2, noise=0.1):
            super().__init__()
            self.noise = noise
            self.enc = nn.Sequential(nn.Linear(n, enc), nn.GELU())
            self.dec = nn.Linear(enc, n)
            self.ae_head = nn.Linear(enc, 1)
            self.mlp = nn.Sequential(nn.Linear(n + enc, hidden), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

        def forward(self, x):
            xn = x + self.noise * torch.randn_like(x) if self.training else x
            e = self.enc(xn)
            return self.mlp(torch.cat([x, e], dim=1)).squeeze(-1), self.ae_head(e).squeeze(-1), self.dec(e)

        def main(self, x):
            return self.forward(x)[0]
    return MTMLP, AEMLP


def _fit_v15(variant, model, X, idx, T, epochs, hp, seed, val=None, patience=2):
    """the loop of tfmodel._fit (AdamW, shuffled mini-batches from a seeded generator, gradient clip 1.0, early stopping on the
    main output's weighted log-loss) with the multi-output losses of C1 / C2"""
    import torch, copy, math
    import torch.nn.functional as F
    bce = F.binary_cross_entropy_with_logits
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['wd'])
    tX = torch.from_numpy(X)

    def batch_loss(o):
        x = tX[o]; t = {k: torch.from_numpy(v[o]) for k, v in T.items()}
        if variant == 'mt':
            out = model(x)
            return bce(out[:, 0], t['y'], weight=t['w']) + 0.5 * (bce(out[:, 1], t['y1'], weight=t['w1']) + bce(out[:, 2], t['y2'], weight=t['w2'])) / 2
        main, ae, dec = model(x)
        return F.mse_loss(dec, x) + bce(ae, t['y'], weight=t['w']) + bce(main, t['y'], weight=t['w'])

    def val_loss(vidx):
        model.eval(); tot = 0.0
        with torch.no_grad():
            for s in range(0, len(vidx), 4096):
                o = vidx[s:s + 4096]
                lg = model.main(tX[o])
                tot += float(bce(lg, torch.from_numpy(T['y'][o]), weight=torch.from_numpy(T['w'][o]), reduction='sum'))
        return tot / float(T['w'][vidx].sum())

    best, best_ep, best_state, bad = math.inf, 0, None, 0
    for ep in range(1, epochs + 1):
        model.train(); order = torch.randperm(len(idx), generator=g).numpy()
        for s in range(0, len(idx), hp['bs']):
            o = idx[order[s:s + hp['bs']]]
            loss = batch_loss(o)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if val is not None:
            vl = val_loss(val)
            if vl < best - 1e-5:
                best, best_ep, best_state, bad = vl, ep, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if val is not None and best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep


def train_custom_v15(variant, Q, feats, start, train_end, seed, sig_col, test=False):
    """C1 / C2: the same data, scaler, weights, split and refit as train_custom; only the network and its loss differ"""
    import torch, tfmodel as TM
    MTMLP, AEMLP = _mlp_classes()
    hp = dict(L.MLP_HP)
    if test:
        hp = {**hp, 'epochs': 1, 'patience': 1}
    torch.manual_seed(seed); np.random.seed(seed)
    D = TM.SeqData(Q, feats, 1)
    te = pd.Timestamp(train_end)
    tm = ((D.P.date >= start) & (D.P.date <= te)).values & D.P.target_r.notna().values
    scaler = TM.Scaler().fit(D.raw[tm]); D.scale(scaler)
    idx_all = D.valid(tm); dates = D.P.date.values[idx_all]
    ud = np.sort(np.unique(dates)); cut = ud[-120] if len(ud) > 240 else ud[-1]
    sig = D.P[sig_col].values.astype(float)
    med = np.nanmedian(sig[idx_all])
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, med)
    raw_w = np.minimum(np.abs(D.P.target_r.fillna(0).values), 0.05) / sig ** 0.5
    norm = raw_w[idx_all].mean()
    T = dict(y=D.P.y.values.astype(np.float32), w=(raw_w / norm).astype(np.float32))
    if variant == 'mt':
        grp = D.P.groupby('stock_id', sort=False)
        for j in (1, 2):
            rj = grp.target_r.shift(-j).values.astype(float); dj = grp.date.shift(-j).values
            ok = np.isfinite(rj) & (pd.to_datetime(dj) <= te)
            ok = np.asarray(ok, bool)
            T[f'y{j}'] = (np.nan_to_num(rj) > 0).astype(np.float32)
            T[f'w{j}'] = np.where(ok, np.minimum(np.abs(np.nan_to_num(rj)), 0.05) / sig ** 0.5 / norm, 0.0).astype(np.float32)
            T[f'd{j}'] = dj
    tr, va = idx_all[dates < cut], idx_all[dates >= cut]
    T_in = dict(T)
    if variant == 'mt':                                                                 # embargo: no aux label on/after the cut
        for j in (1, 2):
            bad = np.asarray(pd.to_datetime(T[f'd{j}']) >= pd.Timestamp(cut), bool)
            T_in[f'w{j}'] = np.where(bad, 0.0, T[f'w{j}']).astype(np.float32)
    T_in = {k: v for k, v in T_in.items() if not k.startswith('d')}
    T_all = {k: v for k, v in T.items() if not k.startswith('d')}
    make = (lambda: MTMLP(len(feats), dropout=hp['dropout'])) if variant == 'mt' else (lambda: AEMLP(len(feats), dropout=hp['dropout']))
    X = D.X
    model = make()
    model, best_ep = _fit_v15(variant, model, X, tr, T_in, hp['epochs'], hp, seed, val=va, patience=hp['patience'])
    best_ep = max(best_ep, 1)
    torch.manual_seed(seed)
    model = make()
    model, _ = _fit_v15(variant, model, X, idx_all, T_all, best_ep, hp, seed)
    return dict(model=model.eval(), scaler=scaler, features=feats, best_epoch=best_ep)


def predict_v15(bd, Q, mask):
    import torch, tfmodel as TM
    D = TM.SeqData(Q, bd['features'], 1); D.scale(bd['scaler'])
    idx = D.valid(np.asarray(mask(D.P), dtype=bool))
    out = D.P.loc[idx, ['date', 'stock_id']].copy()
    with torch.no_grad():
        out['p'] = np.concatenate([torch.sigmoid(bd['model'].main(torch.from_numpy(D.X[idx[s:s + 4096]]))).numpy()
                                   for s in range(0, len(idx), 4096)])
    return out


def train_mlp_unit(P, variant, fa, fb, seed, test=False):
    """= v9_train.run_unit for mlp_res_xr9 (core panel, resid target, sigma column absres_60) with the C variant"""
    import tfmodel as TM
    feats = L.xr9_features()
    te = pd.Timestamp(fa) - pd.Timedelta(days=1); a_, b_ = pd.Timestamp(fa), pd.Timestamp(fb)
    Q = P.loc[P.date <= max(b_, te), ['date', 'stock_id', 'resid_r', 'y_resid'] + feats].copy()
    Q['target_r'] = Q.resid_r; Q['y'] = Q.y_resid
    start = L.TRAIN_START
    if test:
        ud = np.sort(Q.loc[(Q.date <= te) & Q.target_r.notna(), 'date'].unique())
        start = pd.Timestamp(ud[-200])
    mask = lambda d: (d.date >= a_) & (d.date < b_) & d.target_r.notna()
    if variant == 'st':
        bd = train_custom(Q[Q.date <= te], feats, start, te, seed, 'absres_60', test=test)
        o = TM.predict(bd, Q, mask)[['date', 'stock_id', 'p']]
    else:
        bd = train_custom_v15(variant, Q[Q.date <= te], feats, start, te, seed, 'absres_60', test=test)
        o = predict_v15(bd, Q, mask)
    return o.reset_index(drop=True), dict(best_epoch=int(bd['best_epoch']), start=str(start.date()))


# ------------------------------------------------------------------ orchestration
def _init(threads, need_torch):
    low_priority()
    _G['threads'] = max(1, threads)
    if need_torch:
        import torch
        torch.set_num_threads(max(1, threads))
    tf = os.path.join(L.ROOT, 'transformer')
    if tf not in sys.path:
        sys.path.insert(0, tf)
    _G['P'] = L.load_panel()


def run_unit(u):
    name, fa, fb, seed, test = u
    f = L.unit_file(name, fa, seed, test)
    if os.path.exists(f):
        return {'unit': u, 'skipped': True}
    t0 = time.time()
    try:
        spec = L.MEMBERS[name]; P = _G['P']
        if spec['kind'] == 'lgb':
            out, meta = train_lgb_unit(P, spec['variant'], fa, fb, seed, test)
        elif spec['kind'] == 'de':
            out, meta = train_de_unit(P, fa, fb, seed, test)
        elif spec['kind'] == 'rf':
            out, meta = train_rf_unit(P, spec['variant'], fa, fb, seed, test, threads=_G['threads'])
        elif spec['kind'] == 'mlp':
            out, meta = train_mlp_unit(P, spec['variant'], fa, fb, seed, test)
        else:
            raise KeyError(spec['kind'])
        meta.update(name=name, fold=fa, seed=int(seed), seconds=round(time.time() - t0, 1), test=bool(test), pred_rows=int(len(out)),
                    pandas=pd.__version__, finished=time.strftime('%Y-%m-%d %H:%M:%S'))
        L.save_unit(f, out, meta)
        return {'unit': u, 'meta': meta}
    except Exception:
        return {'unit': u, 'error': traceback.format_exc()}


def _report(r, i, n, t0, test):
    u = r['unit']; name = f'{u[0]} ({L.MEMBERS[u[0]]["label"]}) {u[1]} seed {u[3]}'
    if r.get('error'):
        log(f'[{i}/{n}] {name} FAILED\n{r["error"]}', test); return 1
    if r.get('skipped'):
        return 0
    m = r['meta']
    extra = ''
    if 'best' in m:
        extra = f', {m["best"]} rounds'
    if 'leaf' in m:
        extra = f', leaf {m["leaf"]}'
    if 'best_epoch' in m:
        extra = f', best epoch {m["best_epoch"]}'
    log(f'[{i}/{n}] {name}: {m["seconds"] / 60:.1f} min{extra}; elapsed {(time.time() - t0) / 60:.1f} min', test)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', default=','.join(L.ORDER))
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--threads', type=int, default=0, help='threads per worker (default: physical cores / workers)')
    ap.add_argument('--test', action='store_true')
    a = ap.parse_args()
    names = [n for n in a.members.split(',') if n]
    bad = [n for n in names if n not in L.MEMBERS]
    if bad:
        sys.exit(f'unknown member(s) {bad}; use {",".join(L.ORDER)}')
    need = {'lgb': 'lightgbm', 'de': 'lightgbm', 'rf': 'sklearn', 'mlp': 'torch'}
    missing = sorted({need[L.MEMBERS[n]['kind']] for n in names} - {m for m in need.values() if _has(m)})
    if missing:
        sys.exit(f'missing: {" ".join(missing)}  ->  python -m pip install {" ".join("scikit-learn" if m == "sklearn" else m for m in missing)}')
    low_priority(); keep_awake(True)
    cores = os.cpu_count() or 2
    phys = max(1, cores // 2) if cores >= 4 else cores
    threads = a.threads or max(1, phys // max(1, a.workers))
    units = []
    for n in sorted(names, key=L.ORDER.index):
        seeds = L.MEMBERS[n]['seeds'][:1] if a.test else L.MEMBERS[n]['seeds']
        for fa, fb in reversed(L.FOLDS):                                                # big windows first
            for s in seeds:
                units.append((n, fa, fb, s, a.test))
    todo = [u for u in units if not os.path.exists(L.unit_file(u[0], u[1], u[3], a.test))]
    vers = {}
    for mod in ('lightgbm', 'sklearn', 'torch'):
        try:
            vers[mod] = __import__(mod).__version__
        except ImportError:
            vers[mod] = '-'
    log(f'v15_train.py start: {len(units)} units ({", ".join(f"{n} {sum(1 for u in units if u[0] == n)}" for n in names)}), '
        f'{len(todo)} to do; lightgbm {vers["lightgbm"]}, scikit-learn {vers["sklearn"]}, torch {vers["torch"]}, pandas {pd.__version__}, '
        f'{cores} logical cpus, workers {a.workers} x {threads} threads' + (' [TEST]' if a.test else ''), a.test)
    if not todo:
        keep_awake(False); return 0
    need_torch = any(L.MEMBERS[u[0]]['kind'] == 'mlp' for u in todo)
    nbad = done = 0; t0 = time.time()
    try:
        if a.workers <= 1:
            _init(threads, need_torch)
            for u in todo:
                done += 1; nbad += _report(run_unit(u), done, len(todo), t0, a.test)
        else:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'),
                                     initializer=_init, initargs=(threads, need_torch)) as ex:
                futs = [ex.submit(run_unit, u) for u in todo]
                for fu in as_completed(futs):
                    done += 1; nbad += _report(fu.result(), done, len(todo), t0, a.test)
    except KeyboardInterrupt:
        log('stopped by the user; finished units are kept, start again to continue', a.test); keep_awake(False); return 1
    keep_awake(False)
    left = [u for u in units if not os.path.exists(L.unit_file(u[0], u[1], u[3], a.test))]
    log(f'v15_train.py end: {len(units) - len(left)} of {len(units)} units done, {(time.time() - t0) / 60:.1f} min this run'
        + (f', {nbad} FAILED (see above)' if nbad else ''), a.test)
    return 0 if not left and not nbad else 1


def _has(mod):
    try:
        __import__(mod); return True
    except ImportError:
        return False


if __name__ == '__main__':
    sys.exit(main())

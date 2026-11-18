r"""
o1_gap.py -- overfitting of the MODEL WEIGHTS: training-window fit vs out-of-sample, learning curves, random labels.
(needs numpy, pandas, lightgbm; scipy optional.  Cloud, 2 cores: about 40 minutes.  Writes ..\results\o1_gap.txt,
 ..\cache\o1\*.pkl (one per training unit, resumable))

Members (three of v9 / v11's members, trained exactly as in the walk-forward: yearly folds 2022 .. 2026-05 + post,
training window 2020-03-02 .. the day before the fold, early stopping on the last 120 training days, refit on the whole
window):
  mlp_raw_B9   market view MLP  (B - Nasdaq + night + sigma; label close > open; weights min(|r|, 5%) / sigma^0.5)
  mlp_res_xr9  stock view MLP   (B - Nasdaq + XR + sigma; label beats the same-day average)
  lgb_res      stock view LightGBM (B, label beats the same-day average; v8 g1_tab_train parameters)
The MLPs are trained with npmlp.py (numpy copy of the torch procedure; torch is not installed where this ran);
LightGBM is the real library with the same parameters.  Seeds 1-3 with the real labels, seed 1 with random labels.

For every unit:
  * production model (the one the walk-forward used): its fit on its own training window (in-sample) and on the fold
    (out-of-sample, OOS)
  * learning curve of the refit: after every epoch (MLP) / at fixed numbers of trees (LightGBM), in-sample and OOS
  * random labels: y replaced by coin flips (weights unchanged); how well the procedure fits pure noise in-sample, and
    what it scores on the true OOS labels (must be ~0)
Metrics: weighted log-loss (the training objective, lower = better); rank IC (stock view: daily Spearman of p with the
same-day excess return); corr(pbar, m) (market view: daily mean of p over the 100 stocks vs the Top50 intraday return);
edge = weighted share of correct signs - 0.5 (weights min(|r|, 5%)).
Mini pipeline: score = 0.5 (pbar - 0.5) + centred mean(mlp_res_xr9, lgb_res), 0 on the 4 ex-dividend flags, Top50 AR
(no hedge) in-sample (training window) and OOS.
run (in overfit\research): python o1_gap.py [--workers 2] > ..\results\o1_gap.txt
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, time, json, argparse, pickle, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import ofit_lib as L
import npmlp
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 300)

OUT = os.path.join(L.CACHE, 'o1')
TRAIN_START = pd.Timestamp('2020-03-02')
FOLDS = [('2022-01-01', '2023-01-01'), ('2023-01-01', '2024-01-01'), ('2024-01-01', '2025-01-01'), ('2025-01-01', '2026-01-01'),
         ('2026-01-01', '2026-06-01'), ('2026-06-01', '2100-01-01')]
NAS = ['us_ixic_ret1', 'us_ixic_ret5', 'us_ixic_ret1_lag1']
NIGHT = ['night_ret', 'night_range']; SIG = ['absr_60', 'absres_60']
XR_BASE = ['ret1', 'ret5', 'ret20', 'intraday_ret', 'gap', 'range', 'clv', 'upper_shadow', 'lower_shadow', 'vol_ratio', 'amt_ratio',
           'trades_ratio', 'foreign_net', 'foreign_net_5d', 'trust_net', 'trust_net_5d', 'dealer_self_net', 'dealer_hedge_net',
           'foreign_hold_chg5', 'intra_bias_20', 'intra_bias_60', 'intra_bias_250', 'gap_bias_20', 'RSI', 'close_SMA30', 'BB_pctb',
           'ATR_pct', 'vol_ratio_5_60', 'beta_x_us', 'y_mean5']                      # = v5\features_extra.XR_BASE
XR = ['xr_' + c for c in XR_BASE]
LGB_PRM = dict(objective='binary', learning_rate=0.02, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8,
               bagging_freq=1, lambda_l2=1.0, num_threads=1, verbose=-1)          # = v8 g1_tab_train.LGB
LGB_CHECK = [10, 25, 50, 100, 200, 300, 500, 750, 1000]
EX4 = ['ex_today', 'ex_next1', 'ex_prev1', 'ex_prev2']
_G = {}


def members():
    B = json.load(open(os.path.join(L.ROOT, 'transformer', 'final_feature_lists.json')))['B_competition_only']
    B9 = [f for f in B if f not in NAS]
    return {'mlp_raw_B9': dict(kind='mlp', target='raw', feats=B9 + NIGHT + SIG, sig='absr_60'),
            'mlp_res_xr9': dict(kind='mlp', target='resid', feats=B9 + XR + SIG, sig='absres_60'),
            'lgb_res': dict(kind='lgb', target='resid', feats=list(B), sig=None)}


def panel():
    P = pd.read_pickle(os.path.join(L.ROOT, 'v9', 'cache', 'core9.pkl'))
    P['date'] = pd.to_datetime(P.date).astype('datetime64[ns]'); P['stock_id'] = P.stock_id.astype(str)
    return P.sort_values(['date', 'stock_id']).reset_index(drop=True)


class Scaler:                                   # = transformer\tfmodel.Scaler
    def fit(self, X):
        self.lo = np.nanquantile(X, 0.01, axis=0); self.hi = np.nanquantile(X, 0.99, axis=0)
        Xc = np.clip(X, self.lo, self.hi)
        self.mu = np.nanmean(Xc, axis=0); self.sd = np.nanstd(Xc, axis=0); self.sd[~(self.sd > 1e-12)] = 1.0
        self.mu = np.nan_to_num(self.mu); return self

    def transform(self, X):
        Z = (np.clip(X, self.lo, self.hi) - self.mu) / self.sd
        return np.nan_to_num(Z, nan=0.0).astype(np.float32)


# ------------------------------------------------------------------ metrics
def rank_ic(dates, p, r):
    X = pd.DataFrame({'d': dates, 'p': p, 'r': r})
    g = X.groupby('d')
    R = pd.DataFrame({'d': X.d, 'p': g.p.rank(), 'r': g.r.rank()})
    c = R.groupby('d').apply(lambda x: np.corrcoef(x.p, x.r)[0, 1] if len(x) > 2 else np.nan, include_groups=False)
    return float(c.mean())


def pbar_corr(dates, p, top50, tr):
    X = pd.DataFrame({'d': dates, 'p': p, 'top': top50, 'r': tr})
    pb = X.groupby('d').p.mean(); m = X[X.top].groupby('d').r.mean()
    j = pb.index.intersection(m.index)
    return float(np.corrcoef(pb[j], m[j])[0, 1])


def wedge(p, y, w):
    return float((w * ((p > 0.5) == (y > 0.5))).sum() / w.sum() - 0.5)


def wll(p, y, w):
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / w.sum())


def metrics(kind, target, dates, p, y_fit, y_true, w, top50, tr, resid, tr_fit=None, resid_fit=None):
    """y_fit = labels the model was trained on (coin flips for the null runs); y_true = real labels.
    tr / resid = real returns; tr_fit / resid_fit = the returns implied by the training labels (null runs), so that
    'ic_fit' / 'corr_fit' measure how well the model fits what it was trained on in the same units as the real metric."""
    out = {'wll_fit': wll(p, y_fit, w), 'edge_fit': wedge(p, y_fit, w), 'wll_true': wll(p, y_true, w), 'edge_true': wedge(p, y_true, w)}
    if target == 'resid':
        out['ic'] = rank_ic(dates, p, resid)
        out['ic_fit'] = out['ic'] if resid_fit is None else rank_ic(dates, p, resid_fit)
    else:
        out['corr_pbar_m'] = pbar_corr(dates, p, top50, tr)
        out['corr_fit'] = out['corr_pbar_m'] if tr_fit is None else pbar_corr(dates, p, top50, tr_fit)
    return out


# ------------------------------------------------------------------ one training unit
def unit_file(u):
    name, a, seed, lab = u
    return os.path.join(OUT, f'{name}_{a}_s{seed}_{lab}.pkl')


def run_unit(u):
    f = unit_file(u)
    if os.path.exists(f):
        return u, 0.0, None
    t0 = time.time()
    try:
        name, a, seed, lab = u
        b = dict(FOLDS)[a]
        m = _G['M'][name]; P = _G['P']
        tcol, ycol = ('target_r', 'y') if m['target'] == 'raw' else ('resid_r', 'y_resid')
        te = pd.Timestamp(a) - pd.Timedelta(days=1)
        Q = P[(P.date >= TRAIN_START) & (P.date <= te) & P[tcol].notna()]
        R = P[(P.date >= pd.Timestamp(a)) & (P.date < pd.Timestamp(b)) & P[tcol].notna()]
        y_true = Q[ycol].values.astype(np.float32)
        tr_fit, resid_fit = Q.target_r.values, Q.resid_r.values
        if lab == 'random':                         # stock-day coin flips; implied returns = flipped sign, real size
            y_fit = np.random.default_rng(1000 + seed).integers(0, 2, len(Q)).astype(np.float32)
            sgn = 2 * y_fit - 1
            tr_fit, resid_fit = sgn * np.abs(Q.target_r.values), sgn * np.abs(Q.resid_r.values)
        elif lab == 'dayflip':                      # one coin per DAY: every label of the day flipped together
            ud0 = np.sort(Q.date.unique())
            c = pd.Series(np.random.default_rng(2000 + seed).integers(0, 2, len(ud0)), index=ud0)
            flip = 1 - 2 * c.reindex(Q.date.values).values
            tr_fit = Q.target_r.values * flip; resid_fit = Q.resid_r.values * flip
            y_fit = np.where(flip < 0, 1 - y_true, y_true).astype(np.float32)
        else:
            y_fit = y_true
        def weights(D):
            w = np.minimum(np.abs(D[tcol].values), 0.05)
            if m['sig']:
                s = D[m['sig']].values.astype(float); med = np.nanmedian(Q[m['sig']].values.astype(float))
                s = np.where(np.isfinite(s) & (s > 0), s, med); w = w / s ** 0.5
            return w
        wq = weights(Q); w_norm = wq.mean(); wq = (wq / w_norm).astype(np.float32); wr = (weights(R) / w_norm).astype(np.float32)
        ud = np.sort(Q.date.unique()); cut = ud[-120] if len(ud) > 240 else ud[-1]; trm = Q.date.values < cut
        info_q = (Q.date.values, Q.top50.values.astype(bool), Q.target_r.values, Q.resid_r.values)
        info_r = (R.date.values, R.top50.values.astype(bool), R.target_r.values, R.resid_r.values)
        yr = R[ycol].values.astype(np.float32)
        curve = []
        if m['kind'] == 'mlp':
            sc = Scaler().fit(Q[m['feats']].to_numpy(np.float64))
            XQ = sc.transform(Q[m['feats']].to_numpy(np.float64)); XR = sc.transform(R[m['feats']].to_numpy(np.float64))
            _, best = npmlp.fit(XQ[trm], y_fit[trm], wq[trm], 10, seed, val=(XQ[~trm], y_fit[~trm], wq[~trm]), patience=2)
            best = max(best, 1)
            snap = {}
            def on_epoch(model, ep):
                pq = 1 / (1 + np.exp(-model.logits(XQ).astype(np.float64))); pr = 1 / (1 + np.exp(-model.logits(XR).astype(np.float64)))
                curve.append({'step': ep, 'split': 'in', **metrics('mlp', m['target'], info_q[0], pq, y_fit, y_true, wq, info_q[1], info_q[2], info_q[3], tr_fit, resid_fit)})
                curve.append({'step': ep, 'split': 'oos', **metrics('mlp', m['target'], info_r[0], pr, yr, yr, wr, info_r[1], info_r[2], info_r[3])})
                if ep == best:
                    snap['q'], snap['r'] = pq.astype(np.float32), pr.astype(np.float32)
            npmlp.fit(XQ, y_fit, wq, 10, seed, on_epoch=on_epoch)
        else:
            import lightgbm as lgb
            XQ = Q[m['feats']].to_numpy(np.float64); XR = R[m['feats']].to_numpy(np.float64)
            prm = {**LGB_PRM, 'seed': int(seed)}
            dtr = lgb.Dataset(XQ[trm], y_fit[trm], weight=wq[trm]); dva = lgb.Dataset(XQ[~trm], y_fit[~trm], weight=wq[~trm], reference=dtr)
            mm = lgb.train(prm, dtr, 1000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
            best = max(int(mm.best_iteration), 1)
            full = lgb.train(prm, lgb.Dataset(XQ, y_fit, weight=wq), max(1000, best))
            snap = {}
            for k in sorted(set(LGB_CHECK + [best])):
                pq = full.predict(XQ, num_iteration=k); pr = full.predict(XR, num_iteration=k)
                curve.append({'step': k, 'split': 'in', **metrics('lgb', m['target'], info_q[0], pq, y_fit, y_true, wq, info_q[1], info_q[2], info_q[3], tr_fit, resid_fit)})
                curve.append({'step': k, 'split': 'oos', **metrics('lgb', m['target'], info_r[0], pr, yr, yr, wr, info_r[1], info_r[2], info_r[3])})
                if k == best:
                    snap['q'], snap['r'] = pq.astype(np.float32), pr.astype(np.float32)
        res = dict(unit=u, best=best, curve=pd.DataFrame(curve),
                   q=pd.DataFrame({'date': Q.date.values, 'stock_id': Q.stock_id.values, 'p': snap['q']}),
                   r=pd.DataFrame({'date': R.date.values, 'stock_id': R.stock_id.values, 'p': snap['r']}),
                   seconds=time.time() - t0)
        os.makedirs(OUT, exist_ok=True)
        with open(f + '.tmp', 'wb') as fh:
            pickle.dump(res, fh)
        os.replace(f + '.tmp', f)
        return u, time.time() - t0, None
    except Exception:
        return u, time.time() - t0, traceback.format_exc()


def load_unit(u):
    with open(unit_file(u), 'rb') as fh:
        r = pickle.load(fh)
    C = r['curve']                                  # units with the real labels: fit labels = real labels
    for k, kf in (('ic', 'ic_fit'), ('corr_pbar_m', 'corr_fit')):
        if k in C and kf not in C:
            assert u[3] == 'real', f'{u}: null-label unit written by an older version, delete it and re-run'
            C[kf] = C[k]
    return r


# ------------------------------------------------------------------ report
def pipeline_ar(P, pr_raw, pr_res_mlp, pr_res_lgb, exdiv=True):
    """mini pipeline on the rows of the three prediction frames (same rows); Top50 AR per day (bp).
    exdiv=False drops the ex-dividend rule (a fixed rule, not learned, so it earns something even with random members)"""
    X = pr_raw.rename(columns={'p': 'raw'}).merge(pr_res_mlp.rename(columns={'p': 'mlp'}), on=['date', 'stock_id']) \
        .merge(pr_res_lgb.rename(columns={'p': 'lgb'}), on=['date', 'stock_id'])
    X = X.merge(P[['date', 'stock_id', 'top50', 'target_r'] + EX4], on=['date', 'stock_id'])
    g = X.groupby('date')
    pbar = g.raw.transform('mean')
    pe = (X.mlp + X.lgb) / 2; pe = pe - pe.groupby(X.date).transform('mean')
    s = 0.5 * (pbar - 0.5) + pe
    pred = (s > 0).astype(int).values
    ex = np.zeros(len(X), bool)
    for c in EX4:
        ex |= X[c].values == 1
    if exdiv:
        pred = np.where(ex, 0, pred)
    X = X.assign(sgn=2 * pred - 1)
    q = X[X.top50.astype(bool)]
    return (q.sgn * q.target_r).groupby(q.date).mean() * 1e4


def report(units, P):
    M = members()
    R = {u: load_unit(u) for u in units if os.path.exists(unit_file(u))}
    print(f'o1_gap.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}   {len(R)} of {len(units)} units\n')
    # ---------------------------------------------------------- 0. replication check against the cached torch / lightgbm runs
    print('=' * 110)
    print('0. Is the re-run the same model as the walk-forward?  (OOS predictions, seed average, vs the cached walk-forward)')
    print('=' * 110)
    W9 = pd.read_csv(os.path.join(L.ROOT, 'v9', 'cache', 'wf9_preds.csv.gz'), dtype={'stock_id': str}, parse_dates=['date'])
    TB = pd.read_csv(os.path.join(L.ROOT, 'v8', 'cache', 'tab_preds.csv.gz'), dtype={'stock_id': str}, parse_dates=['date'])
    for name, ref, col in (('mlp_raw_B9', W9, 'mlp_raw_B9'), ('mlp_res_xr9', W9, 'mlp_res_xr9'), ('lgb_res', TB, 'lgb_resid')):
        oo = []
        for a, b in FOLDS:
            fr = [R[u]['r'] for u in R if u[0] == name and u[1] == a and u[3] == 'real']
            if fr:
                z = pd.concat(fr).groupby(['date', 'stock_id']).p.mean().reset_index(); oo.append(z)
        if not oo:
            continue
        O = pd.concat(oo); O['date'] = pd.to_datetime(O.date)
        J = O.merge(ref[['date', 'stock_id', col]], on=['date', 'stock_id']).merge(P[['date', 'stock_id', 'top50', 'target_r', 'resid_r']], on=['date', 'stock_id'])
        c = np.corrcoef(J.p, J[col])[0, 1]
        cdc = np.corrcoef(J.p - J.groupby('date').p.transform('mean'), J[col] - J.groupby('date')[col].transform('mean'))[0, 1]
        if M[name]['target'] == 'resid':
            m1 = rank_ic(J.date.values, J.p.values, J.resid_r.values); m2 = rank_ic(J.date.values, J[col].values, J.resid_r.values)
            lab = 'rank IC'
        else:
            m1 = pbar_corr(J.date.values, J.p.values, J.top50.values.astype(bool), J.target_r.values)
            m2 = pbar_corr(J.date.values, J[col].values, J.top50.values.astype(bool), J.target_r.values)
            lab = 'corr(pbar, m)'
        print(f'  {name}: {len(J)} rows; corr of predictions {c:.3f} (within-day {cdc:.3f}); OOS {lab}: re-run {m1:.4f}, cached walk-forward {m2:.4f}')
    print('  (cached MLPs = torch, seed 1-10 average; cached LightGBM = seed 1-5 average; the re-run = seed 1-3)\n')

    # ---------------------------------------------------------- 1. production model: in-sample vs OOS
    print('=' * 110)
    print('1. The production models: fit on their own training window (in) vs the next fold (OOS)')
    print('=' * 110)
    rows = []
    for u, r in R.items():
        name, a, seed, lab = u
        C = r['curve']; st = r['best']
        ci = C[(C.split == 'in') & (C.step == st)].iloc[0]; co = C[(C.split == 'oos') & (C.step == st)].iloc[0]
        key, kfit = ('ic', 'ic_fit') if M[name]['target'] == 'resid' else ('corr_pbar_m', 'corr_fit')
        rows.append({'member': name, 'fold': a[:7], 'seed': seed, 'labels': lab, 'best': st,
                     'in: wll(fit)': ci.wll_fit, 'in: edge(fit)': ci.edge_fit, 'in: IC/corr(fit)': ci[kfit], 'in: IC/corr(true)': ci[key],
                     'oos: wll': co.wll_true, 'oos: edge': co.edge_true, 'oos: IC/corr': co[key]})
    D = pd.DataFrame(rows)
    LABS = ('real', 'random', 'dayflip')
    for name in M:
        for lab in LABS:
            X = D[(D.member == name) & (D.labels == lab)]
            if X.empty:
                continue
            g = X.groupby('fold').agg({'best': 'mean', 'in: wll(fit)': 'mean', 'in: edge(fit)': 'mean', 'in: IC/corr(fit)': 'mean',
                                       'in: IC/corr(true)': 'mean', 'oos: wll': 'mean', 'oos: edge': 'mean', 'oos: IC/corr': 'mean'})
            g.loc['mean'] = g.mean()
            what = 'rank IC (100 stocks, daily mean)' if M[name]['target'] == 'resid' else 'corr(pbar_t, Top50 m_t)'
            print(f'\n  {name}, {lab} labels (IC/corr = {what}; (fit) = against the labels it was trained on, (true) = real labels; '
                  f'edge = weighted share of correct signs - 50%, percentage points):')
            if lab == 'real':
                g = g.drop(columns=['in: IC/corr(true)'])
            g2 = g.copy()
            for c in ('in: edge(fit)', 'oos: edge'):
                g2[c] = g2[c] * 100
            print(g2.round(4).to_string())
    print()
    D.to_csv(os.path.join(L.RESULTS, 'o1_units.csv'), index=False, float_format='%.5f')

    # ---------------------------------------------------------- 2. learning curves
    print('=' * 110)
    print('2. Learning curves of the refit (mean over folds and seeds): in-sample keeps improving, OOS does not')
    print('=' * 110)
    for name in M:
        for lab in ('real', 'random', 'dayflip'):
            Cs = [R[u]['curve'].assign(fold=u[1], seed=u[2]) for u in R if u[0] == name and u[3] == lab]
            if not Cs:
                continue
            C = pd.concat(Cs)
            key = 'ic_fit' if M[name]['target'] == 'resid' else 'corr_fit'          # OOS rows: fit = true labels
            piv = C.pivot_table(index='step', columns='split', values=['wll_fit', 'edge_fit', key], aggfunc='mean')
            piv.columns = [f'{s}: {v}' for v, s in piv.columns]
            for c in piv.columns:
                if 'edge' in c:
                    piv[c] = piv[c] * 100
            bests = [R[u]['best'] for u in R if u[0] == name and u[3] == lab]
            print(f'\n  {name}, {lab} labels; early stopping picked {np.mean(bests):.1f} on average (min {min(bests)}, max {max(bests)}); '
                  f'step = {"epoch" if M[name]["kind"] == "mlp" else "trees"}')
            cols = [c for c in piv.columns if c.startswith('in')] + [c for c in piv.columns if c.startswith('oos')]
            if M[name]['kind'] == 'lgb':
                piv = piv[piv.index.isin(LGB_CHECK)]
            print(piv[cols].round(4).to_string())
    print()

    # ---------------------------------------------------------- 3. mini pipeline AR
    print('=' * 110)
    print('3. Mini pipeline (market MLP + two stock-view members, ex-dividend rule, no hedge): Top50 AR bp/day')
    print('=' * 110)
    rows = []
    for lab in ('real', 'random'):
        for a, b in FOLDS:
            parts = {}
            for split in ('q', 'r'):
                fr = {}
                for name in M:
                    us = [u for u in R if u[0] == name and u[1] == a and u[3] == lab]
                    if not us:
                        break
                    z = pd.concat([R[u][split] for u in us]).groupby(['date', 'stock_id']).p.mean().reset_index()
                    z['date'] = pd.to_datetime(z.date); fr[name] = z
                if len(fr) == 3:
                    parts[split] = pipeline_ar(P, fr['mlp_raw_B9'], fr['mlp_res_xr9'], fr['lgb_res'])
                    parts[split + '_nx'] = pipeline_ar(P, fr['mlp_raw_B9'], fr['mlp_res_xr9'], fr['lgb_res'], exdiv=False)
            if len(parts) == 4:
                ai, ao = parts['q'], parts['r']
                rows.append({'labels': lab, 'fold': a[:7], 'in-sample days': len(ai), 'in-sample AR': ai.mean(), 'OOS days': len(ao),
                             'OOS AR': ao.mean(), 'OOS se': L.nw_se(ao.values), 'gap (in - OOS)': ai.mean() - ao.mean(),
                             'OOS AR without the ex-div rule': parts['r_nx'].mean(), 'in-sample AR without': parts['q_nx'].mean()})
    T = pd.DataFrame(rows)
    if not T.empty:
        for lab in ('real', 'random'):
            X = T[T.labels == lab]
            if X.empty:
                continue
            w = X['OOS days'] / X['OOS days'].sum()
            print(X.round(2).to_string(index=False))
            se = np.sqrt(np.sum(w ** 2 * X['OOS se'] ** 2))
            print(f'  {lab}: OOS AR pooled over folds {np.sum(w * X["OOS AR"]):.2f} bp/day (se {se:.2f}); without the ex-dividend rule '
                  f'{np.sum(w * X["OOS AR without the ex-div rule"]):.2f}; mean in-sample AR {X["in-sample AR"].mean():.2f}\n')
    T.to_csv(os.path.join(L.RESULTS, 'o1_pipeline.csv'), index=False, float_format='%.4f')

    # ---------------------------------------------------------- 4. the cached walk-forward members per fold (torch / lightgbm)
    print('=' * 110)
    print('4. OOS stability of the walk-forward members actually used (cached predictions, incl. the Transformer that was not re-run)')
    print('=' * 110)
    WM = pd.read_csv(os.path.join(L.ROOT, 'v8', 'cache', 'wf_members.csv.gz'), dtype={'stock_id': str}, parse_dates=['date'])
    J = WM[['date', 'stock_id', 'tf_raw_B']].merge(W9, on=['date', 'stock_id']).merge(TB[['date', 'stock_id', 'lgb_resid']], on=['date', 'stock_id'])
    J = J.merge(P[['date', 'stock_id', 'top50', 'target_r', 'resid_r']], on=['date', 'stock_id'])
    rows = []
    for a, b in FOLDS:
        X = J[(J.date >= pd.Timestamp(a)) & (J.date < pd.Timestamp(b))]
        r = {'fold': a[:7], 'days': X.date.nunique()}
        for col in ('tf_raw_B', 'mlp_raw_B9', 'mlp_raw_xr9', 'mlp_raw_w4_9'):
            r['corr(pbar,m) ' + col] = pbar_corr(X.date.values, X[col].values, X.top50.values.astype(bool), X.target_r.values)
        for col in ('mlp_res_xr9', 'mlp_res_B9', 'lgb_resid'):
            r['IC ' + col] = rank_ic(X.date.values, X[col].values, X.resid_r.values)
        rows.append(r)
    T4 = pd.DataFrame(rows).set_index('fold')
    print(T4.round(3).to_string())
    print('  (market-view members: correlation of the daily mean prediction with the Top50 intraday return; stock-view members: daily rank IC)')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=2); ap.add_argument('--report-only', action='store_true')
    ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()
    _G['M'] = members(); _G['P'] = panel()
    units = [(n, fa, s, 'real') for n in _G['M'] for fa, _ in FOLDS for s in range(1, a.seeds + 1)] + \
            [(n, fa, 1, 'random') for n in _G['M'] for fa, _ in FOLDS] + \
            [('mlp_raw_B9', fa, 1, 'dayflip') for fa, _ in FOLDS]
    # slowest first (late folds have the longest training windows)
    units = sorted(units, key=lambda u: (u[1], u[0].startswith('lgb')), reverse=True)
    todo = [u for u in units if not os.path.exists(unit_file(u))]
    if not a.report_only and todo:
        print(f'{len(units)} units, {len(todo)} to do', file=sys.stderr, flush=True)
        import multiprocessing as mp
        t0 = time.time()
        if a.workers > 1 and 'fork' in mp.get_all_start_methods():
            pool = mp.get_context('fork').Pool(a.workers); it = pool.imap_unordered(run_unit, todo)
        else:
            pool = None; it = map(run_unit, todo)
        for i, (u, sec, err) in enumerate(it, 1):
            if err:
                print('FAILED', u, err, file=sys.stderr, flush=True)
            print(f'[{i}/{len(todo)}] {u} {sec:.0f}s, elapsed {(time.time() - t0) / 60:.1f} min', file=sys.stderr, flush=True)
        if pool is not None:
            pool.close(); pool.join()
    report(units, _G['P'])


if __name__ == '__main__':
    main()

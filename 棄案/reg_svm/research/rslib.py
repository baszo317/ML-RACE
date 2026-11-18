r"""
rslib.py -- shared code of the regularization / SVM study (reg_svm\README_reg_svm.md).

Data (read only, copies of the user's files):
  v9\cache\core9.pkl            v9 research panel (100 stocks, 2020-01-02 .. 2026-10-23)
  v9\cache\wf9_preds.csv.gz     cached walk-forward predictions of v9's 7 MLP members (torch, seed 1-10)
  v8\cache\wf_members.csv.gz    tf_raw_B (Transformer market member)
  v8\cache\tab_preds.csv.gz     lgb_resid (= lgb_res, seed 1-5), lr_raw / lr_resid (v8 logistic regression)
  v8\cache\beta_panel.pkl       120-day beta used by the hedge
Members retrained here use exactly the walk-forward of v9 (yearly folds 2022 .. 2026-05 + post, training window
2020-03-02 .. the day before the fold, early stopping on the last 120 training days, refit on the whole window).
Pipeline replica = v9\research\v9_eval.py (views + decide with the 4 ex-dividend flags and the beta hedge lambda 0.75).
"""
import os, json, math
import numpy as np, pandas as pd

_CLOUD = '/mnt/user-data/uploads/ML RACE'                                 # the cloud copy of D:\ML RACE (read only)
SRC = os.environ.get('RS_SRC') or (_CLOUD if os.path.isdir(_CLOUD) else
                                   os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # else ..\..
RS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(RS, 'cache'); RESULTS = os.path.join(RS, 'results')
FOLDS = [('2022-01-01', '2023-01-01'), ('2023-01-01', '2024-01-01'), ('2024-01-01', '2025-01-01'), ('2025-01-01', '2026-01-01'),
         ('2026-01-01', '2026-06-01'), ('2026-06-01', '2100-01-01')]
TRAIN_START = pd.Timestamp('2020-03-02')
PER = {'val': ('2022-01-01', '2025-01-01'), 'hold': ('2025-01-01', '2026-06-01'), 'post': ('2026-06-01', '2100-01-01'),
       'all': ('2022-01-01', '2100-01-01')}
NAS = ['us_ixic_ret1', 'us_ixic_ret5', 'us_ixic_ret1_lag1']
NIGHT = ['night_ret', 'night_range']; SIG = ['absr_60', 'absres_60']
XR_BASE = ['ret1', 'ret5', 'ret20', 'intraday_ret', 'gap', 'range', 'clv', 'upper_shadow', 'lower_shadow', 'vol_ratio', 'amt_ratio',
           'trades_ratio', 'foreign_net', 'foreign_net_5d', 'trust_net', 'trust_net_5d', 'dealer_self_net', 'dealer_hedge_net',
           'foreign_hold_chg5', 'intra_bias_20', 'intra_bias_60', 'intra_bias_250', 'gap_bias_20', 'RSI', 'close_SMA30', 'BB_pctb',
           'ATR_pct', 'vol_ratio_5_60', 'beta_x_us', 'y_mean5']                     # = v5\features_extra.XR_BASE
XR = ['xr_' + c for c in XR_BASE]
EX4 = ('ex_today', 'ex_next1', 'ex_prev1', 'ex_prev2')
CGRID = np.round(np.arange(-0.3, 0.3 + 1e-9, 0.0025), 6)


def B_feats():
    return json.load(open(os.path.join(SRC, 'transformer', 'final_feature_lists.json')))['B_competition_only']


def members():
    """the three production members retrained here (= o1_gap.members) + the SVM members (B features, like v8's logistic)"""
    B = B_feats(); B9 = [f for f in B if f not in NAS]
    return {'mlp_raw_B9': dict(kind='mlp', target='raw', feats=B9 + NIGHT + SIG, sig='absr_60'),
            'mlp_res_xr9': dict(kind='mlp', target='resid', feats=B9 + XR + SIG, sig='absres_60'),
            'lgb_res': dict(kind='lgb', target='resid', feats=list(B), sig=None),
            'svm_res': dict(kind='svm', target='resid', feats=list(B), sig=None),
            'svm_raw': dict(kind='svm', target='raw', feats=list(B), sig=None)}


_P = None
def panel():
    global _P
    if _P is None:
        P = pd.read_pickle(os.path.join(SRC, 'v9', 'cache', 'core9.pkl'))
        P['date'] = pd.to_datetime(P.date).astype('datetime64[ns]'); P['stock_id'] = P.stock_id.astype(str)
        _P = P.sort_values(['date', 'stock_id']).reset_index(drop=True)
    return _P


class Scaler:                                   # = transformer\tfmodel.Scaler (winsorize 1% / 99%, z-score, nan -> 0)
    def fit(self, X):
        self.lo = np.nanquantile(X, 0.01, axis=0); self.hi = np.nanquantile(X, 0.99, axis=0)
        Xc = np.clip(X, self.lo, self.hi)
        self.mu = np.nanmean(Xc, axis=0); self.sd = np.nanstd(Xc, axis=0); self.sd[~(self.sd > 1e-12)] = 1.0
        self.mu = np.nan_to_num(self.mu); return self

    def transform(self, X):
        Z = (np.clip(X, self.lo, self.hi) - self.mu) / self.sd
        return np.nan_to_num(Z, nan=0.0).astype(np.float32)


def split(P, m, a, b, frac=1.0, frac_seed=0):
    """training frame Q (window, optionally a random fraction of its DAYS), OOS frame R, labels, weights, ES mask.
    The ES set is the last 120 trading days of the window (the production rule), subsampled at the same rate."""
    tcol, ycol = ('target_r', 'y') if m['target'] == 'raw' else ('resid_r', 'y_resid')
    te = pd.Timestamp(a) - pd.Timedelta(days=1)
    Q = P[(P.date >= TRAIN_START) & (P.date <= te) & P[tcol].notna()]
    ud_all = np.sort(Q.date.unique())
    cut = ud_all[-120] if len(ud_all) > 240 else ud_all[-1]
    if frac < 1.0:
        rng = np.random.default_rng(10_000 + frac_seed)
        perm = rng.permutation(len(ud_all))                   # nested subsets: the first k of one permutation per fold
        keep = np.sort(ud_all[perm[:max(30, int(round(frac * len(ud_all))))]])
        Q = Q[Q.date.isin(keep)]
    R = P[(P.date >= pd.Timestamp(a)) & (P.date < pd.Timestamp(b)) & P[tcol].notna()]

    def weights(D):
        w = np.minimum(np.abs(D[tcol].values), 0.05)
        if m['sig']:
            s = D[m['sig']].values.astype(float); med = np.nanmedian(Q[m['sig']].values.astype(float))
            s = np.where(np.isfinite(s) & (s > 0), s, med); w = w / s ** 0.5
        return w
    wq = weights(Q); nrm = wq.mean()
    return dict(Q=Q, R=R, yq=Q[ycol].values.astype(np.float32), yr=R[ycol].values.astype(np.float32),
                wq=(wq / nrm).astype(np.float32), wr=(weights(R) / nrm).astype(np.float32), trm=Q.date.values < cut,
                tcol=tcol, ycol=ycol)


# ------------------------------------------------------------------ fast metrics
def _day_codes(dates):
    d = pd.Index(dates)
    codes, uniq = pd.factorize(d, sort=True)
    return codes, uniq


def rank_ic_daily(dates, p, r):
    """daily Spearman of p with r (ranks within day, average ties), Series by date"""
    codes, uniq = _day_codes(dates)
    X = pd.DataFrame({'c': codes, 'p': np.asarray(p, float), 'r': np.asarray(r, float)}).dropna()
    g = X.groupby('c')
    rp = g.p.rank(); rr = g.r.rank()
    X = X.assign(rp=rp.values, rr=rr.values)
    X['rp'] -= X.groupby('c').rp.transform('mean'); X['rr'] -= X.groupby('c').rr.transform('mean')
    s = X.assign(xy=X.rp * X.rr, xx=X.rp ** 2, yy=X.rr ** 2).groupby('c')[['xy', 'xx', 'yy']].sum()
    ic = s.xy / np.sqrt(s.xx * s.yy)
    ic.index = uniq[ic.index]
    return ic


def rank_ic(dates, p, r):
    return float(rank_ic_daily(dates, p, r).mean())


def pbar_corr(dates, p, top50, tr):
    X = pd.DataFrame({'d': dates, 'p': p, 'top': top50, 'r': tr})
    pb = X.groupby('d').p.mean(); m = X[X.top].groupby('d').r.mean()
    j = pb.index.intersection(m.index)
    return float(np.corrcoef(pb[j], m[j])[0, 1])


def wll(p, y, w):
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    return float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / w.sum())


def whinge(f, y, w):
    s = 2 * np.asarray(y, float) - 1
    return float((w * np.maximum(0, 1 - s * f)).sum() / w.sum())


def member_metrics(target, R, p, yr, wr, prob=True):
    out = {'ic': rank_ic(R.date.values, p, R.resid_r.values)} if target == 'resid' else \
          {'corr': pbar_corr(R.date.values, p, R.top50.values.astype(bool), R.target_r.values)}
    if prob:
        out['wll'] = wll(p, yr, wr)
    return out


# ------------------------------------------------------------------ pipeline replica (v9_eval)
def read_csv(f):
    x = pd.read_csv(f, dtype={'stock_id': str}, parse_dates=['date']); x['date'] = x.date.astype('datetime64[ns]')
    return x.set_index(['date', 'stock_id']).sort_index()


class Pipe:
    def __init__(self):
        self.W9 = read_csv(os.path.join(SRC, 'v9', 'cache', 'wf9_preds.csv.gz'))
        self.M = read_csv(os.path.join(SRC, 'v8', 'cache', 'wf_members.csv.gz'))
        self.TB = read_csv(os.path.join(SRC, 'v8', 'cache', 'tab_preds.csv.gz'))
        idx = self.M.index.intersection(self.W9.dropna().index).intersection(self.TB.lgb_resid.index)
        self.idx = idx
        self.W9, self.M, self.TB = self.W9.loc[idx], self.M.loc[idx], self.TB.loc[idx]
        P = panel().set_index(['date', 'stock_id'])
        Bt = pd.read_pickle(os.path.join(SRC, 'v8', 'cache', 'beta_panel.pkl'))
        Bt['date'] = pd.to_datetime(Bt.date).astype('datetime64[ns]'); Bt['stock_id'] = Bt.stock_id.astype(str)
        Bt = Bt.set_index(['date', 'stock_id']).beta
        INFO = pd.DataFrame({'top50': P.top50, 'target_r': P.target_r, **{c: P[c] for c in EX4}}).reindex(idx)
        INFO['beta'] = Bt.reindex(idx).clip(0.2, 2.5)
        INFO['beta'] = INFO.beta.fillna(INFO.groupby(level='date').beta.transform('median')).fillna(1.0)
        self.INFO = INFO[INFO.top50.astype(bool) & INFO.target_r.notna()]
        self.raw9 = [self.W9.mlp_raw_B9, self.W9.mlp_raw_xr9, self.W9.mlp_raw_w4_9]
        self.tf = self.M.tf_raw_B
        self.res9 = [self.W9.mlp_res_B9, self.W9.mlp_res_xr9, self.W9.mlp_res_BN9, self.W9.mlp_res_xrN9]
        self.lgb = self.TB.lgb_resid
        self.tr = P.target_r.reindex(idx); self.resid = P.resid_r.reindex(idx); self.top50 = P.top50.reindex(idx).astype(bool)

    @staticmethod
    def centre(p):
        return p - p.groupby(level='date').transform('mean')

    def score(self, raw_parts, res_parts, w_t=0.5):
        p_raw = pd.concat([pd.concat(f, axis=1).mean(axis=1) for f in raw_parts], axis=1).mean(axis=1)
        p_e = pd.concat([pd.concat(f, axis=1).mean(axis=1) for f in res_parts], axis=1).mean(axis=1)
        return w_t * (p_raw.groupby(level='date').transform('mean') - 0.5) + self.centre(p_e)

    def decide(self, score, ex_cols=EX4, lam=0.75):
        X = self.INFO.join(score.rename('score'), how='inner')
        rows = []
        for d, g in X.groupby(level='date'):
            sc, b, r = g.score.values, g.beta.values, g.target_r.values
            ex = np.zeros(len(g), bool)
            for c in ex_cols:
                ex |= g[c].values == 1
            s = np.where(ex, -1, np.where(sc > 0, 1, -1))
            if lam < 1.0:
                sb, bb = s.mean(), b.mean(); B0 = ((s - sb) * (b - bb)).mean(); Et = sb * bb + lam * B0
                S = (sc[None, :] + CGRID[:, None] * b[None, :] > 0) & ~ex[None, :]
                E = ((2 * S - 1) @ b) / len(b)
                dd = np.abs(E - Et); cand = np.flatnonzero(dd <= dd.min() + 1e-12); j = cand[np.argmin(np.abs(CGRID[cand]))]
                s = np.where(S[j], 1, -1)
            m = r.mean()
            rows.append((d, (s * r).mean() * 1e4, s.mean() * m * 1e4, m * 1e4))
        D = pd.DataFrame(rows, columns=['date', 'AR', 'timing', 'm']).set_index('date'); D['xsec'] = D.AR - D.timing
        return D

    def v9(self):
        return self.decide(self.score([self.raw9, [self.tf]], [self.res9, [self.lgb]]))

    def rescale_like(self, s, ref):
        """centre s per day and give it ref's daily sd of the centred values (as X5 / v12), keep ref's day mean"""
        cs = self.centre(s); cr = self.centre(ref)
        k = cr.groupby(level='date').transform('std') / cs.groupby(level='date').transform('std')
        return ref.groupby(level='date').transform('mean') + cs * k


def periods(ix):
    ix = pd.DatetimeIndex(ix)
    return {k: (ix >= pd.Timestamp(a)) & (ix < pd.Timestamp(b)) for k, (a, b) in PER.items()}


def nw_t(d, lags=5):
    d = np.asarray(d, float); d = d[~np.isnan(d)]; n = len(d)
    if n < 3:
        return np.nan
    e = d - d.mean(); s = e @ e / n
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return d.mean() / np.sqrt(s / n) if s > 0 else np.nan


def ar_row(D, base=None):
    pr = periods(D.index)
    r = {p: D.AR[pr[p]].mean() for p in ('val', 'hold', 'post', 'all')}
    r['sd'] = D.AR.std(); r['timing'] = D.timing.mean(); r['xsec'] = D.xsec.mean()
    if base is not None:
        dd = (D.AR - base.AR.reindex(D.index))
        for p in ('val', 'hold', 'post', 'all'):
            r['d_' + p] = dd[pr[p]].mean(); r['t_' + p] = nw_t(dd[pr[p]].values)
    return r

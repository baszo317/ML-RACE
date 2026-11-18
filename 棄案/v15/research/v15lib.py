r"""
v15lib.py -- shared code of v15 (research\PREREG_v15.md): paths, the panel, feature lists, folds, unit files, the member
registry, and the evaluation helpers (Data = v14lib.Data, copied so that v15 runs even if v14 is deleted).

numpy + pandas only at import time; lightgbm / scikit-learn / torch are imported inside the training functions
(v15_train.py), so v15_signals.py and v15_eval.py run with numpy + pandas alone.
"""
import os, sys, json, math
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); V15 = os.path.dirname(HERE)
ROOT = os.environ.get('MLRACE_ROOT', os.path.dirname(V15))
CACHE = os.path.join(V15, 'cache'); RESULTS = os.path.join(V15, 'results')

FOLDS = [('2022-01-01', '2023-01-01'), ('2023-01-01', '2024-01-01'), ('2024-01-01', '2025-01-01'), ('2025-01-01', '2026-01-01'),
         ('2026-01-01', '2026-06-01'), ('2026-06-01', '2100-01-01')]                       # = v9 / v10 / v13 / v14 folds
TRAIN_START = pd.Timestamp('2020-03-02')                                                  # = ens_lib.TRAIN_START
VAL_DAYS = 120
LGB_PRM = dict(objective='binary', learning_rate=0.02, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8,
               bagging_freq=1, lambda_l2=1.0, num_threads=1, verbose=-1)                  # = v10lib.LGB_PRM = lgb_res
MLP_HP = dict(dropout=0.2, lr=1e-3, wd=1e-2, bs=1024, epochs=10, patience=2)              # = ens_lib.HP as used by v9_train
NAS = ['us_ixic_ret1', 'us_ixic_ret5', 'us_ixic_ret1_lag1']
SIG = ['absr_60', 'absres_60']
EX4 = ('ex_today', 'ex_next1', 'ex_prev1', 'ex_prev2')
XR_BASE = ['ret1', 'ret5', 'ret20', 'intraday_ret', 'gap', 'range', 'clv', 'upper_shadow', 'lower_shadow', 'vol_ratio', 'amt_ratio',
           'trades_ratio', 'foreign_net', 'foreign_net_5d', 'trust_net', 'trust_net_5d', 'dealer_self_net', 'dealer_hedge_net',
           'foreign_hold_chg5', 'intra_bias_20', 'intra_bias_60', 'intra_bias_250', 'gap_bias_20', 'RSI', 'close_SMA30', 'BB_pctb',
           'ATR_pct', 'vol_ratio_5_60', 'beta_x_us', 'y_mean5']                            # = v5 features_extra.XR_BASE
XR = ['xr_' + c for c in XR_BASE]

# ------------------------------------------------------------------ the candidates (PREREG_v15.md section 2)
#   kind     : lgb (lgb_res variants), de (DoubleEnsemble), rf (sklearn forests), mlp (torch)
#   variant  : what the unit function does
#   seeds    : seeds of the full run (the test run uses the first seed only)
#   group    : A / B / C (D and E are signals, v15_signals.py)
MEMBERS = {
    'B0': dict(kind='lgb', variant='base',   seeds=(1, 2, 3, 4, 5), group='B', label='lgb_res replica (control)'),
    'B1': dict(kind='lgb', variant='sigma',  seeds=(1, 2, 3, 4, 5), group='B', label='sigma^0.5 weights (GLS)'),
    'B2': dict(kind='lgb', variant='huber',  seeds=(1, 2, 3, 4, 5), group='B', label='Huber weights, cap 2 x day median |r|'),
    'B3': dict(kind='lgb', variant='h3',     seeds=(1, 2, 3, 4, 5), group='B', label='3-day target'),
    'B4': dict(kind='lgb', variant='h5',     seeds=(1, 2, 3, 4, 5), group='B', label='5-day target'),
    'B5': dict(kind='lgb', variant='noex',   seeds=(1, 2, 3, 4, 5), group='B', label='no ex-dividend event rows'),
    'B6': dict(kind='de',  variant='de',     seeds=(1, 2, 3),       group='B', label='DoubleEnsemble (Qlib)'),
    'A1': dict(kind='rf',  variant='rf',     seeds=(1,),            group='A', label='Random Forest'),
    'A2': dict(kind='rf',  variant='et',     seeds=(1,),            group='A', label='ExtraTrees'),
    'A0': dict(kind='rf',  variant='default', seeds=(1,),           group='A', label='sklearn default RF (report only)'),
    'C0': dict(kind='mlp', variant='st',     seeds=(1, 2, 3, 4, 5), group='C', label='mlp_res_xr9 replica (control)'),
    'C1': dict(kind='mlp', variant='mt',     seeds=(1, 2, 3, 4, 5), group='C', label='multi-task MLP (D, D+1, D+2)'),
    'C2': dict(kind='mlp', variant='ae',     seeds=(1, 2, 3, 4, 5), group='C', label='supervised autoencoder + MLP'),
}
ORDER = ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'C0', 'C1', 'C2', 'A1', 'A2', 'B6', 'A0']     # cheap first


def b_features():
    """the 67 B features (competition data only) used by lgb_res"""
    return list(json.load(open(os.path.join(ROOT, 'transformer', 'final_feature_lists.json')))['B_competition_only'])


def xr9_features():
    """features of mlp_res_xr9 (= v9lib.members()['mlp_res_xr9']: B9 + XR + SIG)"""
    return [f for f in b_features() if f not in NAS] + XR + SIG


NEEDED = ['date', 'stock_id', 'top50', 'target_r', 'resid_r', 'y_resid', 'group', 'absres_60'] + list(EX4)


def load_panel(log=print):
    r"""v9\cache\core9.pkl (the 100 competition stocks, built by v9 from the competition data only), the columns v15 needs,
    float64, sorted by (date, stock_id) exactly as v10lib.panel (so that B0 reproduces lgb_res row for row)"""
    P = pd.read_pickle(os.path.join(ROOT, 'v9', 'cache', 'core9.pkl'))
    feats = sorted(set(b_features()) | set(xr9_features()))
    cols = NEEDED + [c for c in feats if c not in NEEDED]
    P = P[cols].copy()
    P['date'] = pd.to_datetime(P.date).astype('datetime64[ns]'); P['stock_id'] = P.stock_id.astype(str)
    for k in cols:
        if k not in ('date', 'stock_id'):
            P[k] = P[k].astype(np.float64)
    P = P.sort_values(['date', 'stock_id']).reset_index(drop=True)
    return P


# ------------------------------------------------------------------ unit files
def unit_dir(name, test=False):
    return os.path.join(CACHE, 'units_test' if test else 'units', name)


def unit_file(name, fold, seed, test=False):
    return os.path.join(unit_dir(name, test), f'{fold}_seed{seed}.npz')


def save_unit(f, out, meta):
    """out: DataFrame date, stock_id, p (+ optional extra float columns); meta: dict (stored as a JSON string)"""
    os.makedirs(os.path.dirname(f), exist_ok=True)
    tmp = f[:-4] + '.tmp'
    arrs = dict(date=pd.to_datetime(out.date).values.astype('datetime64[ns]').astype(np.int64),
                sid=np.asarray(out.stock_id.astype(str).tolist(), dtype='<U8'),
                p=out.p.to_numpy(np.float64), meta=np.asarray(json.dumps(meta, default=float)))
    for k in out.columns:
        if k not in ('date', 'stock_id', 'p'):
            arrs[k] = out[k].to_numpy(np.float64)
    with open(tmp, 'wb') as fh:
        np.savez_compressed(fh, **arrs)
    os.replace(tmp, f)


def load_npz(f, col='p'):
    z = np.load(f, allow_pickle=False)
    idx = pd.MultiIndex.from_arrays([pd.to_datetime(z['date']).astype('datetime64[ns]'), np.asarray(z['sid']).astype(str)],
                                    names=['date', 'stock_id'])
    return pd.Series(np.asarray(z[col], dtype=np.float64), index=idx, name=col)


def load_meta(f):
    z = np.load(f, allow_pickle=False)
    return json.loads(str(z['meta']))


def load_member(name, seeds=None, test=False, col='p', need_all=True):
    """seed mean over the six folds; None if a fold has no unit (need_all) -- with need_all=False missing folds are skipped"""
    seeds = seeds or MEMBERS[name]['seeds']
    if test:
        seeds = seeds[:1]
    parts = []
    for fa, _ in FOLDS:
        fs = [unit_file(name, fa, s, test) for s in seeds]
        fs = [f for f in fs if os.path.exists(f)]
        if not fs or (need_all and len(fs) < len(seeds)):
            if need_all:
                return None
            continue
        parts.append(pd.concat([load_npz(f, col) for f in fs], axis=1).mean(axis=1))
    return pd.concat(parts).sort_index() if parts else None


def member_metas(name, test=False):
    out = []
    for fa, _ in FOLDS:
        for s in MEMBERS[name]['seeds'][:1] if test else MEMBERS[name]['seeds']:
            f = unit_file(name, fa, s, test)
            if os.path.exists(f):
                out.append(load_meta(f))
    return out


# ------------------------------------------------------------------ evaluation helpers (= v14lib)
for _p in (os.path.join(ROOT, 'v9', 'research'), os.path.join(ROOT, 'v8')):
    if _p not in sys.path:
        sys.path.append(_p)

PER = {'val': ('2022-01-01', '2025-01-01'), 'hold': ('2025-01-01', '2026-06-01'), 'post': ('2026-06-01', '2100-01-01'),
       'all': ('2022-01-01', '2100-01-01')}
YEARS = {'2022': ('2022-01-01', '2023-01-01'), '2023': ('2023-01-01', '2024-01-01'), '2024': ('2024-01-01', '2025-01-01'),
         '2025': ('2025-01-01', '2026-01-01'), '2026-01..05': ('2026-01-01', '2026-06-01'), 'post': ('2026-06-01', '2100-01-01')}
OCT = {'2026-10-01..10-22': ('2026-10-01', '2026-10-23')}
W_T, LAM, NIGHT_W, EXTRA_ONES = 0.5, 0.75, 0.5, 5                                          # = v11b_lib
CGRID = np.round(np.arange(-0.3, 0.3 + 1e-9, 0.0025), 6)                                   # = v10lib.CGRID
PUBLISHED = {'v9': (13.99, 14.33, 16.59, 14.31), 'v11': (14.60, 14.34, 20.35, 15.02), 'W0': (11.79, 10.35, 21.71, 12.22)}


def read_csv(f):
    x = pd.read_csv(f, dtype={'stock_id': str}, parse_dates=['date']); x['date'] = x.date.astype('datetime64[ns]')
    return x.set_index(['date', 'stock_id']).sort_index()


def w(ix, k, table=PER):
    a, b = table[k]; ix = pd.DatetimeIndex(ix)
    return (ix >= pd.Timestamp(a)) & (ix < pd.Timestamp(b))


def nw_t(d, lags=5):                                                                       # = v10lib.nw_t
    d = np.asarray(d, float); d = d[~np.isnan(d)]; n = len(d)
    if n < 3:
        return np.nan
    e = d - d.mean(); s = e @ e / n
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return d.mean() / np.sqrt(s / n) if s > 0 else np.nan


c = lambda s: s - s.groupby(level='date').transform('mean')
sdd = lambda s: s.groupby(level='date').transform('std')


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def per(s, table=PER):
    return {p: float(s[w(s.index, p, table)].mean()) for p in table}


def ic_daily(pred, r):
    """daily Spearman correlation of pred with r (two series on (date, stock_id))"""
    X = pd.DataFrame({'p': pred, 'r': r}).dropna()
    g = X.groupby(level='date')
    R = pd.DataFrame({'p': g.p.rank(), 'r': g.r.rank()})
    R = R - R.groupby(level='date').transform('mean')
    s = pd.DataFrame({'xy': R.p * R.r, 'xx': R.p ** 2, 'yy': R.r ** 2}).groupby(level='date').sum()
    return s.xy / np.sqrt(s.xx * s.yy)


def scale_to(y, ref):
    """y moved to ref's day mean and within-day sd (v13 'scale', used for every member that enters a model part)"""
    k = (sdd(c(ref)) / sdd(c(y))).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return ref.groupby(level='date').transform('mean') + c(y) * k


def zday(s, idx=None):
    """within-day z-score over the stocks that have a value; missing -> 0 (neutral); clipped to +-3 (= v14lib.zday)"""
    if idx is not None:
        s = s.reindex(idx)
    g = s.groupby(level='date')
    sd = g.transform('std')
    z = (s - g.transform('mean')) / sd.where(sd > 0)
    return z.clip(-3, 3).fillna(0.0)


def spread_match(p_raw, ref):
    pr = c(p_raw)
    k = (sdd(ref) / sdd(pr)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return pr * k


def with_families(base, zs, base_w=2):
    """= v14lib.with_families with every family active every day: p_raw = base_w * base + sum_j sd(base) z_j, spread-matched to base"""
    s = sdd(base)
    tot = base_w * base
    for z in zs:
        tot = tot + s * z.reindex(base.index).fillna(0.0)
    return spread_match(tot, base)


class Data:
    r"""= v14lib.Data (= v13_eval.Data = reg_svm svm_exact_eval.Data = v11b_lib.Data): v9 / v11 / W0 rebuilt from the cached
    walk-forward predictions. v15 additions: mlp members can be swapped (view(..., mlp=...)), and decide() takes an optional
    beta series (E1). Nothing that enters v9 / v11 / W0 is changed (PART 0 of v15_eval.py checks the published numbers)."""
    def __init__(self):
        W9 = read_csv(os.path.join(ROOT, 'v9', 'cache', 'wf9_preds.csv.gz'))
        M = read_csv(os.path.join(ROOT, 'v8', 'cache', 'wf_members.csv.gz'))
        LGB = read_csv(os.path.join(ROOT, 'v8', 'cache', 'tab_preds.csv.gz')).lgb_resid
        idx = M.index.intersection(W9.dropna().index).intersection(LGB.index)
        self.W9, self.TF, self.LGB, self.idx = W9.loc[idx], M.loc[idx, 'tf_raw_B'], LGB.loc[idx], idx
        P = pd.read_pickle(os.path.join(ROOT, 'v9', 'cache', 'core9.pkl'))
        P = P[['date', 'stock_id', 'top50', 'target_r', 'resid_r', 'group'] + list(EX4)].copy()
        P['date'] = pd.to_datetime(P.date).astype('datetime64[ns]'); P['stock_id'] = P.stock_id.astype(str)
        Q = P.set_index(['date', 'stock_id'])
        import hedge_beta as H
        Bt = H.betas(sorted(set(idx.get_level_values('date'))))
        Bt['date'] = pd.to_datetime(Bt.date).astype('datetime64[ns]'); Bt['stock_id'] = Bt.stock_id.astype(str)
        Bt = Bt.set_index(['date', 'stock_id']).beta
        INFO = pd.DataFrame({'top50': Q.top50, 'target_r': Q.target_r, **{k: Q[k] for k in EX4}}).reindex(idx)
        INFO['beta'] = Bt.reindex(idx).clip(0.2, 2.5)
        INFO['beta'] = INFO.beta.fillna(INFO.groupby(level='date').beta.transform('median')).fillna(1.0)
        self.INFO = INFO[INFO.top50.astype(bool) & INFO.target_r.notna()]
        self.resid = Q.resid_r.reindex(idx)
        self.group = Q.group.reindex(idx)
        self.raw = pd.concat([pd.concat([self.W9.mlp_raw_B9, self.W9.mlp_raw_xr9, self.W9.mlp_raw_w4_9], axis=1).mean(axis=1), self.TF],
                             axis=1).mean(axis=1)
        self.mlp = [self.W9.mlp_res_B9, self.W9.mlp_res_xr9, self.W9.mlp_res_BN9, self.W9.mlp_res_xrN9]
        xdir = os.path.join(ROOT, 'v11', 'cache', 'screen', 'x', 'lgb_res_xrN', 'lgb_res_xrN')
        self.x1 = self._x1(xdir).reindex(idx)
        import v9lib
        N = v9lib.night_table().set_index('date').night_ret.sort_index()
        sd = N.rolling(250, min_periods=60).std().shift(1)
        self.z_night = (N / sd).clip(-3, 3)
        days = pd.DatetimeIndex(sorted(P.date.unique()))
        first3 = pd.Series(days, index=days).groupby(days.to_period('M')).transform(lambda s: s.rank(method='first') <= 3)
        self.month_start = set(days[first3.values.astype(bool)])

    @staticmethod
    def _x1(folder):
        parts = []
        for fa, _ in FOLDS:
            fs = [os.path.join(folder, f'{fa}_seed{s}.npz') for s in (1, 2, 3, 4, 5)]
            fs = [f for f in fs if os.path.exists(f)]
            if not fs:
                raise FileNotFoundError(f'X1 units missing in {folder}')
            parts.append(pd.concat([load_npz(f) for f in fs], axis=1).mean(axis=1))
        return pd.concat(parts).sort_index()

    def pbar(self, night=False):
        pb = self.raw.groupby(level='date').mean()
        if night:
            sd = pb.rolling(250, min_periods=20).std().shift(1)
            z = self.z_night.reindex(pb.index)
            pb = pb - (NIGHT_W * sd * z).fillna(0.0)
        return pb

    def view(self, lgbs, extras=(), mlp=None):
        """centred p_e: (MLP mean + LightGBM-part mean + extra families) / (2 + number of extras); mlp: list replacing self.mlp"""
        m = pd.concat(mlp if mlp is not None else self.mlp, axis=1).mean(axis=1); lg = pd.concat(lgbs, axis=1).mean(axis=1)
        tot = m + lg
        for e in extras:
            tot = tot + e
        return c(tot / (2 + len(extras)))

    def run(self, pe, v11, w_t=W_T, lam=LAM, beta=None):
        pb = self.pbar(night=v11)
        s = w_t * (pb.reindex(pe.index.get_level_values('date')).values - 0.5) + pe
        return self.decide(s, month_start=v11, lam=lam, beta=beta)

    def decide(self, score, month_start=False, lam=LAM, beta=None):                         # = v11b_lib.Data.decide
        INFO = self.INFO
        if beta is not None:
            INFO = INFO.copy(); INFO['beta'] = beta.reindex(INFO.index).fillna(INFO.beta)
        X = INFO.join(score.rename('score'), how='inner')
        rows = []
        for d, g in X.groupby(level='date'):
            sc, b, r = g.score.values.copy(), g.beta.values, g.target_r.values
            ex = np.zeros(len(g), bool)
            for col in EX4:
                ex |= g[col].values == 1
            if month_start and d in self.month_start:
                cand = np.sort(sc[~ex & (sc <= 0)])[::-1]
                if len(cand) > EXTRA_ONES:
                    sc = sc - (cand[EXTRA_ONES - 1] + cand[EXTRA_ONES]) / 2
                elif len(cand):
                    sc = sc - cand[-1] + 1e-9
            s = np.where(ex, -1, np.where(sc > 0, 1, -1))
            if lam < 1.0:
                sb, bb = s.mean(), b.mean(); B0 = ((s - sb) * (b - bb)).mean(); Et = sb * bb + lam * B0
                Sg = (sc[None, :] + CGRID[:, None] * b[None, :] > 0) & ~ex[None, :]
                E = ((2 * Sg - 1) @ b) / len(b)
                dd = np.abs(E - Et); cnd = np.flatnonzero(dd <= dd.min() + 1e-12); j = cnd[np.argmin(np.abs(CGRID[cnd]))]
                s = np.where(Sg[j], 1, -1)
            m = r.mean()
            rows.append((d, (s * r).mean() * 1e4, s.mean() * m * 1e4, m * 1e4, s.mean()))
        D = pd.DataFrame(rows, columns=['date', 'AR', 'timing', 'm', 'sbar']).set_index('date'); D['xsec'] = D.AR - D.timing
        return D


def diff(D, x, y, table=PER):
    dd = (D[x].AR - D[y].AR).dropna()
    return {p: (float(dd[w(dd.index, p, table)].mean()), float(nw_t(dd[w(dd.index, p, table)].values))) for p in table}


def fmt(r):
    return '  '.join(f'{p} {r[p][0]:+.2f} ({r[p][1]:.2f})' for p in r)


def power(D, x, y, deltas=(0.0, 0.3, 0.5, 1.0)):
    dd = (D[x].AR - D[y].AR).dropna()
    sds = {p: (float(dd[w(dd.index, p)].std()), int(w(dd.index, p).sum())) for p in ('val', 'hold', 'post')}
    out = {}
    for delta in deltas:
        pr = 1.0
        for p, (sd_, n_) in sds.items():
            pr *= phi(delta / (sd_ / math.sqrt(n_))) if sd_ > 0 else 1.0
        out[delta] = pr
    return sds, out

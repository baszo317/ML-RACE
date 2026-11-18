r"""
v11lib.py -- shared code of the v11 research (research\PREREG_v11.md): Alpha158 groups for v9's stock-view members, selected
with a low-noise metric (daily rank IC of the family on all stocks), a 20-seed baseline used as four 5-seed blocks,
strict thresholds, forward selection and a nested walk-forward comparison of selection rules.

Builds on v10 (v10\research\v10lib.py: panel10, training units, v9's views / decide / periods / t-statistics).
Predictions:
  single-group configs (seed 1..5 means) : v10\cache\screen_lgb_preds.csv.gz, screen_mlp_preds.csv.gz
  baseline seeds                          : L 1-5 v10\cache\screen\lgb\base, 6-20 v11\cache\screen\lgb\base;
                                            M 1-10 v10\cache\screen\mlp\base, 11-20 v11\cache\screen\mlp\base
  union configs (forward selection, nested): v11\cache\screen\<family>\<A+B+...>\<member>\  (seed 1..5)
"""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); V11 = os.path.dirname(HERE); ROOT = os.path.dirname(V11)
sys.path.insert(0, os.path.join(ROOT, 'v10', 'research')); sys.path.insert(0, HERE)
import v10lib as V
import v11_train as T

CACHE = os.path.join(V11, 'cache'); RESULTS = os.path.join(V11, 'results')
GROUPS = V.GROUPS; ORDER = list(GROUPS)
MEM = T.MEMBERS
BLOCKS = [range(1, 6), range(6, 11), range(11, 16), range(16, 21)]
FOLD_A = [a for a, _ in V.FOLDS]
# periods used for "every year > 0" and for the nested walk-forward (2026 split at the hold / post boundary, as the folds)
PERIODS = [('2022', '2022-01-01', '2023-01-01'), ('2023', '2023-01-01', '2024-01-01'), ('2024', '2024-01-01', '2025-01-01'),
           ('2025', '2025-01-01', '2026-01-01'), ('2026a', '2026-01-01', '2026-06-01'), ('post', '2026-06-01', '2100-01-01')]

# ------------------------------------------------------------------ predictions
def _read(f):
    x = pd.read_csv(f, dtype={'stock_id': str}, parse_dates=['date']); x['date'] = x.date.astype('datetime64[ns]')
    return x.set_index(['date', 'stock_id']).sort_index()

_SCREEN = {}
def screen(family):
    """seed 1..5 means of the v10 screen: DataFrame with columns '<config>|<member>'"""
    if family not in _SCREEN:
        _SCREEN[family] = _read(os.path.join(ROOT, 'v10', 'cache', f'screen_{family}_preds.csv.gz'))
    return _SCREEN[family]

def unit_mean(family, config, member, seeds, root_family=None):
    """seed mean over all folds from the unit files (v11 first, then v10); None if a file is missing"""
    parts = []
    for fa in FOLD_A:
        fs = [T.unit_file(root_family or family, config, member, fa, s) for s in seeds]
        if not all(V.unit_exists(f) for f in fs):
            return None
        parts.append(pd.concat([V.load_unit(f) for f in fs], axis=1).mean(axis=1))
    return pd.concat(parts).sort_index()

def config_preds(family, groups, seeds=range(1, 6), folds_only=None):
    """member -> Series (date, stock_id) of the seed-mean prediction of base + groups; single groups and base seed 1-5 come from
    the v10 screen file, unions from v11 units (folds_only: only these folds are required, the rest is NaN)"""
    name = 'ALL' if list(groups) == ['ALL'] else T.config_name(groups)
    S = screen(family)
    out = {}
    for m in MEM[family]:
        col = f'{name}|{m}'
        if list(seeds) == list(range(1, 6)) and col in S and len(groups) <= 1:
            out[m] = S[col].dropna()
            continue
        if folds_only is None:
            x = unit_mean(family, name, m, seeds)
        else:
            parts = []
            for fa in folds_only:
                fs = [T.unit_file(family, name, m, fa, s) for s in seeds]
                if not all(V.unit_exists(f) for f in fs):
                    return None
                parts.append(pd.concat([V.load_unit(f) for f in fs], axis=1).mean(axis=1))
            x = pd.concat(parts).sort_index()
        if x is None:
            return None
        out[m] = x
    return out

def base_blocks(family):
    """the four 5-seed blocks of the baseline: list of {member: Series}"""
    out = []
    for b in BLOCKS:
        d = {}
        for m in MEM[family]:
            x = unit_mean(family, 'base', m, b)
            if x is None:
                raise RuntimeError(f'baseline {family} {m} seeds {b.start}-{b.stop - 1} not complete')
            d[m] = x
        out.append(d)
    return out

def family_pred(family, d, idx=None):
    """L: lgb_res; M: mean of the three MLP members"""
    s = pd.concat([d[m] for m in MEM[family]], axis=1).mean(axis=1) if family == 'mlp' else d['lgb_res']
    return s if idx is None else s.reindex(idx)

# ------------------------------------------------------------------ rank IC
def daily_ic(pred, target, method='spearman'):
    """daily cross-sectional correlation of pred and target over the rows where both exist (Series indexed by date)"""
    X = pd.DataFrame({'p': pred, 'r': target}).dropna()
    if method == 'spearman':
        g = X.groupby(level='date')
        X = pd.DataFrame({'p': g.p.rank(), 'r': g.r.rank()}, index=X.index)
    g = X.groupby(level='date')
    mp, mr = g.p.transform('mean'), g.r.transform('mean')
    dp, dr = X.p - mp, X.r - mr
    num = (dp * dr).groupby(level='date').sum()
    den = np.sqrt((dp * dp).groupby(level='date').sum() * (dr * dr).groupby(level='date').sum())
    ic = num / den
    return ic[g.size() >= 20]

def nw_t(d, lags=5):
    return V.nw_t(d, lags)

def period_masks(idx):
    idx = pd.DatetimeIndex(idx)
    return {p: (idx >= pd.Timestamp(a)) & (idx < pd.Timestamp(b)) for p, a, b in PERIODS}

def window(idx, start, end):
    idx = pd.DatetimeIndex(idx)
    return (idx >= pd.Timestamp(start)) & (idx < pd.Timestamp(end))

def stats(d, mask):
    """mean, NW t, plain t, n of a daily series on a boolean mask"""
    x = d[mask].dropna()
    return dict(mean=float(x.mean()) if len(x) else np.nan, t=nw_t(x.values), plain_t=V.plain_t(x.values), n=len(x))

def years_in(start, end):
    return [p for p, a, b in PERIODS if pd.Timestamp(a) < pd.Timestamp(end) and pd.Timestamp(b) > pd.Timestamp(start)]

def every_year_positive(d, start, end):
    """the mean of every calendar year inside [start, end) is > 0 (PREREG: 年 = 日曆年; in the nested windows 2026 only
    appears as January-May, in the final window as January-October)"""
    x = d[window(d.index, start, end)].dropna()
    if not len(x):
        return False
    return bool((x.groupby(pd.DatetimeIndex(x.index).year).mean() > 0).all())

def t_bonferroni(K, alpha=0.05):
    from statistics import NormalDist
    return NormalDist().inv_cdf(1 - alpha / K)

# ------------------------------------------------------------------ redundancy (Stage 0)
def base_feature_set(family):
    B = V.base_features()
    if family == 'lgb':
        return list(B['lgb_res'])
    u = []
    for m in V.MLP_MEMBERS:
        u += [c for c in B[m] if c not in u]
    return u

def redundancy(P, family, start='2020-03-02', end='2021-12-31'):
    """per Alpha158 column: OLS R^2 on the family's existing features, the max Spearman^2 with one existing feature (and which)"""
    import alpha158 as A
    Q = P[(P.date >= pd.Timestamp(start)) & (P.date <= pd.Timestamp(end))]
    base = [c for c in base_feature_set(family) if c in Q]
    def prep(M):
        M = M.astype(float)
        lo, hi = np.nanquantile(M, 0.005, axis=0), np.nanquantile(M, 0.995, axis=0)
        M = np.clip(M, lo, hi)
        mu, sd = np.nanmean(M, axis=0), np.nanstd(M, axis=0)
        sd[~(sd > 0)] = 1.0
        Z = (M - mu) / sd
        return np.where(np.isfinite(Z), Z, 0.0)
    X = prep(Q[base].to_numpy())
    keep = X.std(axis=0) > 0
    X = X[:, keep]; base = [b for b, k in zip(base, keep) if k]
    Xc = np.column_stack([np.ones(len(X)), X])
    XtX_inv = np.linalg.pinv(Xc.T @ Xc)
    Rb = pd.DataFrame(Q[base].to_numpy(), columns=base).rank().to_numpy()
    Rb = (Rb - np.nanmean(Rb, 0)) / np.nanstd(Rb, 0); Rb = np.where(np.isfinite(Rb), Rb, 0.0)
    rows = []
    for c in A.COLS:
        y = Q[c].to_numpy().astype(float)
        ok = np.isfinite(y)
        z = prep(y[:, None])[:, 0]
        beta = XtX_inv @ (Xc.T @ z)
        res = z - Xc @ beta
        r2 = 1 - (res @ res) / (((z - z.mean()) ** 2).sum()) if z.std() > 0 else np.nan
        ry = pd.Series(y).rank().to_numpy(); ry = (ry - np.nanmean(ry)) / np.nanstd(ry); ry = np.where(np.isfinite(ry), ry, 0.0)
        # Spearman with each base feature on the rows where c exists (missing ranks were set to the mean)
        rho = (Rb[ok].T @ ry[ok]) / ok.sum()
        j = int(np.nanargmax(np.abs(rho)))
        rows.append(dict(col=c, r2_ols=float(r2), rho_max=float(abs(rho[j])), rho_with=base[j]))
    R = pd.DataFrame(rows)
    R['r2'] = np.maximum(R.r2_ols, R.rho_max ** 2)
    R['group'] = R.col.map({c: g for g, cs in GROUPS.items() for c in cs})
    return R

# ------------------------------------------------------------------ AR in v9's system (family-only change)
class ARSystem:
    """v9's walk-forward predictions with one family replaced; AR = Top50 bp/day with v9's rule (v10lib.decide)"""
    def __init__(self, P=None):
        self.W9, self.TF, self.LGB, self.idx = V.load_v9()
        self.INFO = V.info_table(P if P is not None else V.panel(), self.idx)
        self.raw9 = [self.W9.mlp_raw_B9, self.W9.mlp_raw_xr9, self.W9.mlp_raw_w4_9]
        self.res9 = [self.W9.mlp_res_B9, self.W9.mlp_res_xr9, self.W9.mlp_res_BN9, self.W9.mlp_res_xrN9]
    def score(self, L=None, M=None):
        res = [self.W9.mlp_res_B9] + ([M[m].reindex(self.idx) for m in V.MLP_MEMBERS] if M is not None else self.res9[1:])
        lg = L['lgb_res'].reindex(self.idx) if L is not None else self.LGB
        for s in res + [lg]:
            if s.isna().any():
                raise RuntimeError(f'missing predictions: {int(s.isna().sum())}')
        return V.views([self.raw9, [self.TF]], [res, [lg]])
    def ar(self, L=None, M=None, ret_all=False):
        D = V.decide(self.INFO, self.score(L, M))
        return D if ret_all else D.AR
    def pe_sd(self, L=None, M=None):
        """daily cross-sectional sd of the centred stock view p_e_c on the Top50 (scale of the stock view vs the market view)"""
        res = [self.W9.mlp_res_B9] + ([M[m].reindex(self.idx) for m in V.MLP_MEMBERS] if M is not None else self.res9[1:])
        lg = L['lgb_res'].reindex(self.idx) if L is not None else self.LGB
        pe = pd.concat([pd.concat(res, axis=1).mean(axis=1), lg], axis=1).mean(axis=1)
        pe = pe.reindex(self.INFO.index)
        return pe.groupby(level='date').std()

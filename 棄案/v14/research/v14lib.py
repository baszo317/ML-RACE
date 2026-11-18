r"""
v14lib.py -- shared code of v14 (research\PREREG_v14.md). numpy and pandas only.

Data            = v13\research\v13_eval.Data (= reg_svm svm_exact_eval.Data = v11b_lib.Data): v9 / v11 / W0 rebuilt from the cached
                  walk-forward predictions; copied here so that v14 does not need the v13 folder (README says v13 can be deleted).
                  It reads v8\cache, v9\cache, v11\cache\screen\x\lgb_res_xrN, v8\hedge_beta.py and v9\research\v9lib.py.
zday, spread_match, with_families, neutralize_top50   the v14 placements (PREREG section 1)
load_member     seed-averaged walk-forward predictions in the common npz format (date, sid, p), for E5
"""
import os, sys, math
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); V14 = os.path.dirname(HERE)
ROOT = os.environ.get('MLRACE_ROOT', os.path.dirname(V14))
for p in (os.path.join(ROOT, 'v9', 'research'), os.path.join(ROOT, 'v8')):
    if p not in sys.path:
        sys.path.insert(0, p)

FOLDS = [('2022-01-01', '2023-01-01'), ('2023-01-01', '2024-01-01'), ('2024-01-01', '2025-01-01'), ('2025-01-01', '2026-01-01'),
         ('2026-01-01', '2026-06-01'), ('2026-06-01', '2100-01-01')]                       # = v13 seqlib.FOLDS
PER = {'val': ('2022-01-01', '2025-01-01'), 'hold': ('2025-01-01', '2026-06-01'), 'post': ('2026-06-01', '2100-01-01'),
       'all': ('2022-01-01', '2100-01-01')}
YEARS = {'2022': ('2022-01-01', '2023-01-01'), '2023': ('2023-01-01', '2024-01-01'), '2024': ('2024-01-01', '2025-01-01'),
         '2025': ('2025-01-01', '2026-01-01'), '2026-01..05': ('2026-01-01', '2026-06-01'), 'post': ('2026-06-01', '2100-01-01')}
OCT = {'2026-10-01..10-22': ('2026-10-01', '2026-10-23')}
W_T, LAM, NIGHT_W, EXTRA_ONES = 0.5, 0.75, 0.5, 5                                          # = v11b_lib
EX4 = ('ex_today', 'ex_next1', 'ex_prev1', 'ex_prev2')
CGRID = np.round(np.arange(-0.3, 0.3 + 1e-9, 0.0025), 6)                                   # = v10lib.CGRID
PUBLISHED = {'v9': (13.99, 14.33, 16.59, 14.31), 'v11': (14.60, 14.34, 20.35, 15.02), 'W0': (11.79, 10.35, 21.71, 12.22)}


def read_csv(f):
    x = pd.read_csv(f, dtype={'stock_id': str}, parse_dates=['date']); x['date'] = x.date.astype('datetime64[ns]')
    return x.set_index(['date', 'stock_id']).sort_index()


def load_npz(f, col='p'):                                                                  # = v13 seqlib.load_npz
    z = np.load(f, allow_pickle=True)
    idx = pd.MultiIndex.from_arrays([pd.to_datetime(z['date']).astype('datetime64[ns]'),
                                     np.asarray(z['sid']).astype(str)], names=['date', 'stock_id'])
    return pd.Series(np.asarray(z[col], dtype=np.float64), index=idx, name=col)


def load_member(folder, seeds=(1, 2, 3, 4, 5), col='p', no_seed=False):
    """seed mean of <folder>\<fold>_seed<s>.npz over the six folds (no_seed: <folder>\<fold>.npz); None if any fold is missing"""
    parts = []
    for fa, _ in FOLDS:
        fs = [os.path.join(folder, f'{fa}.npz')] if no_seed else [os.path.join(folder, f'{fa}_seed{s}.npz') for s in seeds]
        fs = [f for f in fs if os.path.exists(f)]
        if not fs:
            return None
        parts.append(pd.concat([load_npz(f, col) for f in fs], axis=1).mean(axis=1))
    return pd.concat(parts).sort_index()


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


class Data:
    r"""= v13\research\v13_eval.Data; changes: load_npz / FOLDS are the copies above, and the industry group (core9 'group') is kept
    as self.group for E3a. Nothing that enters v9 / v11 / W0 is changed (PART 0 of v14_eval.py checks the published numbers)."""
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
        import hedge_beta as H                                                              # = v10lib.info_table
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
        xdir = os.path.join(ROOT, 'v11', 'cache', 'screen', 'x', 'lgb_res_xrN', 'lgb_res_xrN')         # = v11b_train.load_member
        self.x1 = load_member(xdir).reindex(idx)
        import v9lib
        N = v9lib.night_table().set_index('date').night_ret.sort_index()
        sd = N.rolling(250, min_periods=60).std().shift(1)
        self.z_night = (N / sd).clip(-3, 3)
        days = pd.DatetimeIndex(sorted(P.date.unique()))
        first3 = pd.Series(days, index=days).groupby(days.to_period('M')).transform(lambda s: s.rank(method='first') <= 3)
        self.month_start = set(days[first3.values.astype(bool)])

    def pbar(self, night=False):
        pb = self.raw.groupby(level='date').mean()
        if night:
            sd = pb.rolling(250, min_periods=20).std().shift(1)
            z = self.z_night.reindex(pb.index)
            pb = pb - (NIGHT_W * sd * z).fillna(0.0)
        return pb

    def view(self, lgbs, extras=()):
        """centred p_e: (MLP mean + LightGBM-part mean + extra families) / (2 + number of extra families)"""
        m = pd.concat(self.mlp, axis=1).mean(axis=1); lg = pd.concat(lgbs, axis=1).mean(axis=1)
        tot = m + lg
        for e in extras:
            tot = tot + e
        return c(tot / (2 + len(extras)))

    def run(self, pe, v11, w_t=W_T, lam=LAM):
        pb = self.pbar(night=v11)
        s = w_t * (pb.reindex(pe.index.get_level_values('date')).values - 0.5) + pe
        return self.decide(s, month_start=v11, lam=lam)

    def decide(self, score, month_start=False, lam=LAM):                                   # = v11b_lib.Data.decide
        X = self.INFO.join(score.rename('score'), how='inner')
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


# ------------------------------------------------------------------ v14 placements (PREREG_v14.md section 1)
def zday(s, idx=None):
    """within-day z-score over the stocks that have a value; missing -> 0 (neutral); clipped to +-3"""
    if idx is not None:
        s = s.reindex(idx)
    g = s.groupby(level='date')
    sd = g.transform('std')
    z = (s - g.transform('mean')) / sd.where(sd > 0)
    return z.clip(-3, 3).fillna(0.0)


def spread_match(p_raw, ref):
    """centre p_raw within the day and scale it so that its within-day sd equals that of ref (ref is centred already)"""
    pr = c(p_raw)
    k = (sdd(ref) / sdd(pr)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return pr * k


def with_families(base, fams, base_w=2, ref=None):
    """base: centred stock view (v11 p_e, or the E5 view); fams: list of (z, active) with z a within-day z-score on the same index and
    active a boolean Series on dates (None = every day). Per day p_raw = base_w * base + sum of active sd(base) * z_j, then spread-matched
    to ref (default base). Days with no active family keep base exactly."""
    ref = base if ref is None else ref
    dates = base.index.get_level_values('date')
    s = sdd(base)
    tot = base_w * base
    any_on = np.zeros(len(base), bool)
    for z, active in fams:
        on = np.ones(len(base), bool) if active is None else active.reindex(dates, fill_value=False).values.astype(bool)
        tot = tot + pd.Series(np.where(on, (s * z.reindex(base.index).fillna(0.0)).values, 0.0), index=base.index)
        any_on |= on
    out = spread_match(tot, ref)
    return pd.Series(np.where(any_on, out.values, base.values), index=base.index)


def neutralize_top50(p, top50_mask, design):
    """within each day, on the Top50 rows only: residual of (p - Top50 mean) on the design columns, rescaled to the same sd, plus the
    Top50 mean back (PREREG_v14.md 1, note of 01:55). design: DataFrame on the same index (group dummies and / or betas); an intercept
    is always included. Other rows unchanged."""
    out = p.copy()
    P = pd.DataFrame({'p': p, 'top': top50_mask.reindex(p.index, fill_value=False).astype(bool)}).join(design)
    cols = list(design.columns)
    vals = out.values.copy()
    pos = pd.Series(np.arange(len(p)), index=p.index)
    for d, g in P[P.top].groupby(level='date'):
        y = g.p.values
        mu, u = y.mean(), y - y.mean()
        A = np.column_stack([np.ones(len(g))] + [g[k].fillna(g[k].mean() if g[k].notna().any() else 0.0).values for k in cols])
        coef, *_ = np.linalg.lstsq(A, u, rcond=None)
        e = u - A @ coef
        su, se = u.std(), e.std()
        e = e * (su / se) if se > 0 else e
        vals[pos.loc[g.index].values] = mu + e
    return pd.Series(vals, index=p.index)


def run_both(d, pe, tag, D, v11_only=False):
    """v11 and W0 runs of a stock view; stored in D under 'v11 + tag' / 'W0 + tag'"""
    D[f'v11 + {tag}'] = d.run(pe, True)
    if not v11_only:
        D[f'W0 + {tag}'] = d.run(pe, True, w_t=0.0, lam=0.5)


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

r"""
p2lib.py -- RACE-STD v1 part 2 (part2\PREREG_part2.md): the untested add-ons on D only. Imports the frozen part-1 code
(..\research\rslib.py) read-only; nothing in ..\research is changed.
"""
import os, sys, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); STD = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(STD, 'research'))
import rslib as L                                                     # noqa: E402

ROOT = L.ROOT
RES2 = os.path.join(HERE, 'results')
UNITS2 = os.path.join(L.CACHE, 'units_p2')
CGRID = np.round(np.arange(-0.3, 0.3 + 1e-9, 0.0025), 6)                # = v8 hedge_beta / v10lib.CGRID
EX4 = L.EX4
D_A, D_B = L.PER['D']


def info_D():
    """Top50 rows on D days with target_r, the ex-dividend flags and beta (v8 hedge_beta: 120 days, clipped, day median)"""
    X = L.load_info()
    X = X[X.top50.astype(bool) & (X.date >= D_A) & (X.date < D_B)].copy()
    v8 = os.path.join(ROOT, 'v8')
    if v8 not in sys.path:
        sys.path.insert(0, v8)
    import hedge_beta as H
    B = H.betas(sorted(X.date.unique()))
    B['date'] = pd.to_datetime(B.date).astype('datetime64[ns]'); B['stock_id'] = B.stock_id.astype(str)
    X = X.merge(B, on=['date', 'stock_id'], how='left')
    X['beta'] = X.beta.clip(0.2, 2.5)
    X['beta'] = X.beta.fillna(X.groupby('date').beta.transform('median')).fillna(1.0)
    return X.set_index(['date', 'stock_id']).sort_index()


def month_start_days(dates):
    d = pd.DatetimeIndex(sorted(set(dates)))
    s = pd.Series(d, index=d)
    first3 = s.groupby(d.to_period('M')).rank(method='first') <= 3
    return set(d[first3.values])


def pbar_user():
    """Baszo's market view from the cached walk-forward members (= v15lib.Data.raw / pbar): ((mean of the 3 MLP raw members)
    + tf_raw_B) / 2, averaged over the stocks of the day"""
    W9 = pd.read_csv(os.path.join(ROOT, 'v9', 'cache', 'wf9_preds.csv.gz'), dtype={'stock_id': str}, parse_dates=['date'])
    M = pd.read_csv(os.path.join(ROOT, 'v8', 'cache', 'wf_members.csv.gz'), dtype={'stock_id': str}, parse_dates=['date'])
    W9 = W9.set_index(['date', 'stock_id'])[['mlp_raw_B9', 'mlp_raw_xr9', 'mlp_raw_w4_9']].dropna()
    TF = M.set_index(['date', 'stock_id'])['tf_raw_B'].dropna()
    idx = W9.index.intersection(TF.index)
    raw = pd.concat([W9.loc[idx].mean(axis=1), TF.loc[idx]], axis=1).mean(axis=1)
    pb = raw.groupby(level='date').mean()
    pb.index = pd.DatetimeIndex(pb.index).astype('datetime64[ns]')
    return pb.sort_index()


def night_z():
    day = pd.read_pickle(os.path.join(L.CACHE, 'day_features.pkl'))
    N = day['C__tx_night_ret'].astype(float)
    N.index = pd.DatetimeIndex(N.index).astype('datetime64[ns]')
    N = N.sort_index()
    sd = N.rolling(250, min_periods=60).std().shift(1)
    return (N / sd).clip(-3, 3)                                       # = v15lib.Data.z_night


def pbar_t2(pb, z, w=0.5):
    sd = pb.rolling(250, min_periods=20).std().shift(1)
    return pb - (w * sd * z.reindex(pb.index)).fillna(0.0)            # = v15lib.Data.pbar(night=True)


def _ex(g):
    ex = np.zeros(len(g), bool)
    for c in EX4:
        ex |= g[c].to_numpy() == 1
    return ex


def _pick(E, Et):
    d = np.abs(E - Et)
    cand = np.flatnonzero(d <= d.min() + 1e-12)
    return cand[np.argmin(np.abs(CGRID[cand]))]


def decide_topk(score, INFO, k=25, k_days=None, lam=None):
    """Top-k of Top50 by score (ties: stock id), ex-dividend -> -1; optional beta hedge (Top-k version of v8 hedge_beta);
    k_days: dates with another k (T3: 30 on month-start days)"""
    X = INFO.join(score.rename('score'), how='left')
    rows = []
    for d, g in X.groupby(level='date', sort=True):
        sc = g.score.to_numpy(float); sc = np.where(np.isnan(sc), -np.inf, sc)
        b = g.beta.to_numpy(float); r = g.target_r.to_numpy(float); ex = _ex(g); n = len(g)
        kk = min((k_days or {}).get(d, k), n)
        order = np.argsort(-sc, kind='stable')                         # rows are sorted by stock id inside a date
        s = -np.ones(n); s[order[:kk]] = 1.0; s[ex] = -1.0
        changed = 0
        if lam is not None and lam < 1.0:
            sb, bb = s.mean(), b.mean()
            Et = sb * bb + lam * ((s - sb) * (b - bb)).mean()
            adj = np.where(np.isinf(sc), -1e9, sc)[None, :] + CGRID[:, None] * b[None, :]
            idx = np.argsort(-adj, axis=1, kind='stable')[:, :kk]
            S = np.zeros(adj.shape, bool); S[np.arange(len(CGRID))[:, None], idx] = True; S &= ~ex[None, :]
            E = ((2 * S - 1) @ b) / n
            s1 = np.where(S[_pick(E, Et)], 1.0, -1.0)
            changed = int((s1 != s).sum()); s = s1
        ok = ~np.isnan(r)
        if not ok.any():
            continue
        m = r[ok].mean()
        rows.append((d, float((s[ok] * r[ok]).mean() * 1e4), float(s[ok].mean() * m * 1e4), float(m * 1e4), float((s[ok] > 0).mean()), changed))
    D = pd.DataFrame(rows, columns=['date', 'AR', 'timing', 'm', 'ones', 'changed']).set_index('date')
    D['xsec'] = D.AR - D.timing
    return D


def decide_thresh(score, INFO, month_start=None, lam=1.0):
    """= v15lib.Data.decide (v11b_lib): +1 if score > 0, ex-dividend -1; month-start T3 (5 more 1s); beta hedge lambda"""
    X = INFO.join(score.rename('score'), how='left')
    rows = []
    for d, g in X.groupby(level='date', sort=True):
        sc = g.score.to_numpy(float).copy(); sc = np.where(np.isnan(sc), -1e9, sc)
        b = g.beta.to_numpy(float); r = g.target_r.to_numpy(float); ex = _ex(g); n = len(g)
        if month_start is not None and d in month_start:
            cand = np.sort(sc[~ex & (sc <= 0)])[::-1]
            if len(cand) > 5:
                sc = sc - (cand[4] + cand[5]) / 2
            elif len(cand):
                sc = sc - cand[-1] + 1e-9
        s = np.where(ex, -1.0, np.where(sc > 0, 1.0, -1.0))
        changed = 0
        if lam < 1.0:
            sb, bb = s.mean(), b.mean()
            Et = sb * bb + lam * ((s - sb) * (b - bb)).mean()
            S = (sc[None, :] + CGRID[:, None] * b[None, :] > 0) & ~ex[None, :]
            E = ((2 * S - 1) @ b) / n
            s1 = np.where(S[_pick(E, Et)], 1.0, -1.0)
            changed = int((s1 != s).sum()); s = s1
        ok = ~np.isnan(r)
        if not ok.any():
            continue
        m = r[ok].mean()
        rows.append((d, float((s[ok] * r[ok]).mean() * 1e4), float(s[ok].mean() * m * 1e4), float(m * 1e4), float((s[ok] > 0).mean()), changed))
    D = pd.DataFrame(rows, columns=['date', 'AR', 'timing', 'm', 'ones', 'changed']).set_index('date')
    D['xsec'] = D.AR - D.timing
    return D


def center(p):
    return p - p.groupby(level='date').transform('mean')

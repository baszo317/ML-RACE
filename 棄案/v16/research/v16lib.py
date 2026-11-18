r"""
v16lib.py -- shared code of v16 (research\PREREG_v16.md): paths, the price panel, the three Kalman filters and the
evaluation helpers. numpy + pandas only.

  K1  Kalman beta for the hedge: r_it = alpha_i + beta_it m_t + eps_it, beta_it - 1 = phi (beta_i,t-1 - 1) + eta_it
  K2  time-varying timing coefficients: m50_t = a_t + b_t x_t + eps_t, a and b random walks, x = pbar' - 0.5
  K3  local level on the daily AR difference v11 - W0, used to switch between the two each day

Data = v15lib.Data (= v14lib.Data = v11b_lib.Data) with one addition: run(..., timing=series) replaces the timing term
w_T (pbar - 0.5) by a given series (K2). Copied here so that v16 runs even if v14 / v15 are deleted.
"""
import os, sys, json, math
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); V16 = os.path.dirname(HERE)
ROOT = os.environ.get('MLRACE_ROOT', os.path.dirname(V16))
CACHE = os.path.join(V16, 'cache'); RESULTS = os.path.join(V16, 'results')

FOLDS = [('2022-01-01', '2023-01-01'), ('2023-01-01', '2024-01-01'), ('2024-01-01', '2025-01-01'), ('2025-01-01', '2026-01-01'),
         ('2026-01-01', '2026-06-01'), ('2026-06-01', '2100-01-01')]                       # = v9 / v10 / v13 / v14 / v15 folds
FOLD_STARTS = [pd.Timestamp(a) for a, _ in FOLDS]

# ------------------------------------------------------------------ K1 settings (PREREG_v16.md section 2)
K1_Q = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
K1_PHI = [1.0, 0.999, 0.997, 0.99]
K1_GRID = [(1.0, 0.0)] + [(p, q) for p in K1_PHI for q in K1_Q]                          # 37 points, index 0 = no time variation
K1_A_SD, K1_B_MEAN, K1_B_SD = 0.002, 1.0, 0.6
K1_RWIN, K1_RMIN, K1_BURN = 250, 60, 60
BETA_CLIP = (0.2, 2.5)

# ------------------------------------------------------------------ K2 settings
K2_QA = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
K2_QB = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
K2_GRID = [(a, b) for a in K2_QA for b in K2_QB]                                           # 30 points, index 0 = (0, 0)
K2_SX, K2_RWIN, K2_RMIN, K2_BURN = 0.03, 120, 60, 60
K2_A_SD, K2_B_MEAN, K2_B_SD = 0.002, 0.03, 0.03
K2_FLOOR, K2_MULT, K2_ICPT = 0.0075, (-1.0, 2.0), 0.1

# ------------------------------------------------------------------ K3 settings
K3_Q = np.concatenate([[0.0], 10.0 ** np.linspace(-6, -1, 26)])                           # 27 points, index 0 = q 0
K3_BURN = 20
K3_HALF_LIVES = (20, 60, 120, 250)


# ================================================================== price panel
def price_long(end=None):
    """the Top100 intraday returns exactly as v8 hedge_beta.betas reads them, long format, plus m100 = the day's Top100 mean"""
    info = json.load(open(os.path.join(ROOT, 'data', 'info.json')))
    R = []
    for sid in info['Top100@Y']:
        m = pd.read_csv(os.path.join(ROOT, 'data', 'market', f'{sid}.csv'), parse_dates=['date'])
        m = m[(m['etl:adj_open'] > 0) & (m['etl:adj_close'] > 0)]
        R.append(pd.DataFrame({'date': m.date, 'stock_id': sid, 'r': m['etl:adj_close'] / m['etl:adj_open'] - 1}))
    R = pd.concat(R, ignore_index=True)
    R['date'] = pd.to_datetime(R.date).astype('datetime64[ns]'); R['stock_id'] = R.stock_id.astype(str)
    if end is not None:
        R = R[R.date <= pd.Timestamp(end)]
    mk = R.groupby('date').r.mean().rename('m100')
    return R, mk, list(map(str, info['Top50@Y']))


def price_panel(end=None, extra_dates=()):
    """dates x stocks matrix of intraday returns (Top100), m100 and m50 (Top50 mean, stocks in sorted order like the judge);
    extra_dates are appended as rows without prices (the day to predict)"""
    R, mk, top50 = price_long(end)
    M = R.pivot(index='date', columns='stock_id', values='r').sort_index()
    add = sorted(set(pd.to_datetime(list(extra_dates))) - set(M.index))
    if add:
        M = pd.concat([M, pd.DataFrame(np.nan, index=pd.DatetimeIndex(add), columns=M.columns)]).sort_index()
    M.index.name = 'date'
    m = mk.reindex(M.index)
    m50 = M[sorted(top50)].mean(axis=1).where(M[sorted(top50)].notna().any(axis=1))
    return M, m, m50, top50


def resid_var(M, m, win=K1_RWIN, minp=K1_RMIN):
    """R_it = residual variance of the OLS of r_i on m over the previous `win` trading days (t-win .. t-1, >= minp days),
    population moments, floored at 1e-8"""
    X = pd.DataFrame(np.where(M.notna().values, m.values[:, None], np.nan), index=M.index, columns=M.columns)
    roll = lambda Z: Z.rolling(win, min_periods=minp).mean()
    my, mx = roll(M), roll(X)
    vy = roll(M * M) - my * my
    vx = roll(X * X) - mx * mx
    cv = roll(M * X) - my * mx
    v = (vy - cv * cv / vx).clip(lower=1e-8)
    return v.shift(1)


def ols_beta(M, m, win=120, minp=60):
    """= v8 hedge_beta.betas: 120-day rolling OLS slope of r_i on m100, days before D only"""
    X = pd.DataFrame(np.where(M.notna().values, m.values[:, None], np.nan), index=M.index, columns=M.columns)
    out = {}
    for sid in M.columns:
        x, y = X[sid], M[sid]
        ok = x.notna() & y.notna()
        x, y = x[ok], y[ok]
        cv = (x * y).rolling(win, min_periods=minp).mean() - x.rolling(win, min_periods=minp).mean() * y.rolling(win, min_periods=minp).mean()
        out[sid] = (cv / x.rolling(win, min_periods=minp).var(ddof=0)).shift(1)
    return pd.DataFrame(out).reindex(index=M.index, columns=M.columns)


def var_series(s, win, minp):
    """population variance of s over t-win .. t-1"""
    mu = s.rolling(win, min_periods=minp).mean()
    return (s.pow(2).rolling(win, min_periods=minp).mean() - mu * mu).shift(1)


# ================================================================== K1: Kalman beta
def k1_filter(M, m, V, grid=K1_GRID, starts=FOLD_STARTS, burn=K1_BURN):
    """run every grid point and every stock at once. returns pred[t, g, n] = beta_{t | t-1} (NaN before the stock starts),
    the pooled log-likelihood (sum over stocks, updates after the first `burn`) of the days before each fold start, and
    the number of likelihood terms"""
    y = M.values; mm = m.values; Vv = V.values
    T, N = y.shape; G = len(grid)
    phi = np.array([g[0] for g in grid])[:, None]; q = np.array([g[1] for g in grid])[:, None]
    A = np.zeros((G, N)); B = np.zeros((G, N)); P00 = np.zeros((G, N)); P01 = np.zeros((G, N)); P11 = np.zeros((G, N))
    started = np.zeros(N, bool); nupd = np.zeros(N, int)
    ll = np.zeros(G); nll = 0; snap = {}
    pred = np.full((T, G, N), np.nan)
    dates = M.index
    for t in range(T):
        d = dates[t]
        for f in starts:
            if f not in snap and d >= f:
                snap[f] = (ll.copy(), nll)
        obs = np.isfinite(y[t]) & np.isfinite(Vv[t]) & np.isfinite(mm[t])
        s = started
        if s.any():                                                                        # transition t-1 -> t
            B[:, s] = 1.0 + phi * (B[:, s] - 1.0)
            P11[:, s] = phi * phi * P11[:, s] + q
            P01[:, s] = phi * P01[:, s]
        new = obs & ~started
        if new.any():                                                                      # prior on the first day with R
            A[:, new] = 0.0; B[:, new] = K1_B_MEAN
            P00[:, new] = K1_A_SD ** 2; P01[:, new] = 0.0; P11[:, new] = K1_B_SD ** 2
            started = started | new
        pred[t][:, started] = B[:, started]
        if obs.any():
            o = obs; mt = mm[t]
            v = y[t, o][None, :] - A[:, o] - B[:, o] * mt
            F = P00[:, o] + 2 * mt * P01[:, o] + mt * mt * P11[:, o] + Vv[t, o][None, :]
            K0 = (P00[:, o] + mt * P01[:, o]) / F
            K1 = (P01[:, o] + mt * P11[:, o]) / F
            A[:, o] += K0 * v; B[:, o] += K1 * v
            P00[:, o] -= K0 * K0 * F; P01[:, o] -= K0 * K1 * F; P11[:, o] -= K1 * K1 * F
            nupd[o] += 1
            cnt = nupd[o] > burn
            if cnt.any():
                ll += (-0.5 * (np.log(F[:, cnt]) + v[:, cnt] ** 2 / F[:, cnt])).sum(axis=1)
                nll += int(cnt.sum())
    for f in starts:
        if f not in snap:
            snap[f] = (ll.copy(), nll)
    return pred, snap


def k1_select(pred, snap, dates, columns):
    """beta_{D | D-1} of the grid point with the highest pooled log-likelihood before each fold (fold days only)"""
    out = np.full((len(dates), pred.shape[2]), np.nan); sel = {}
    for fa, fb in FOLDS:
        f = pd.Timestamp(fa); llf, n = snap[f]
        g = int(np.argmax(llf)); sel[fa] = g
        rows = np.flatnonzero((dates >= f) & (dates < pd.Timestamp(fb)))
        out[rows] = pred[rows, g, :]
    return pd.DataFrame(out, index=dates, columns=columns), sel


def k1_memory(phi, q, m2, R):
    """steady state of the beta filter in the local-level approximation y/m = beta + eps/m (README_v16 theorem 3):
    signal-to-noise r = q m^2 / R, gain K = (-r + sqrt(r^2 + 4r)) / 2, memory 1/K days; for phi < 1 also the stationary sd"""
    r = q * m2 / R
    K = (-r + math.sqrt(r * r + 4 * r)) / 2 if r > 0 else 0.0
    mem = 1 / K if K > 0 else float('inf')
    sd = math.sqrt(q / (1 - phi * phi)) if phi < 1 else float('nan')
    hl = math.log(0.5) / math.log(phi) if phi < 1 else float('inf')
    return K, mem, sd, hl


# ================================================================== K2: time-varying timing coefficients
def k2_filter(y, x, Rv, dates, grid=K2_GRID, starts=FOLD_STARTS, burn=K2_BURN):
    """y = m50, x = pbar' - 0.5, Rv = R_t, all on the walk-forward days (the filter starts on the first one).
    returns the one-step predictions pa[t, g], pb[t, g], the per-day log-likelihood terms and the snapshots of the
    summed log-likelihood (days `burn` .. day before each fold start)"""
    y = np.asarray(y, float); x = np.asarray(x, float); Rv = np.asarray(Rv, float)
    T = len(y); G = len(grid)
    qa = np.array([g[0] for g in grid]); qb = np.array([g[1] for g in grid])
    a = np.zeros(G); b = np.full(G, K2_B_MEAN)
    P00 = np.full(G, K2_A_SD ** 2); P01 = np.zeros(G); P11 = np.full(G, K2_B_SD ** 2)
    pa = np.empty((T, G)); pb = np.empty((T, G)); llt = np.full((T, G), np.nan)
    ll = np.zeros(G); snap = {}
    for t in range(T):
        d = dates[t]
        for f in starts:
            if f not in snap and d >= f:
                snap[f] = ll.copy()
        if t > 0:                                                                          # transition t-1 -> t (R_t is known at t-1)
            P00 = P00 + qa * Rv[t]
            P11 = P11 + qb * Rv[t] / K2_SX ** 2
        pa[t] = a; pb[t] = b
        if np.isfinite(y[t]) and np.isfinite(x[t]) and np.isfinite(Rv[t]):
            xt = x[t]
            v = y[t] - a - b * xt
            F = P00 + 2 * xt * P01 + xt * xt * P11 + Rv[t]
            K0 = (P00 + xt * P01) / F; K1 = (P01 + xt * P11) / F
            a = a + K0 * v; b = b + K1 * v
            P00, P01, P11 = P00 - K0 * K0 * F, P01 - K0 * K1 * F, P11 - K1 * K1 * F
            l = -0.5 * (np.log(F) + v * v / F)
            llt[t] = l
            if t >= burn:
                ll = ll + l
    for f in starts:
        if f not in snap:
            snap[f] = ll.copy()
    return pa, pb, llt, snap


def k2_select(snap, dates):
    """grid index per day: the 2022 fold is fixed at (0, 0), later folds take the highest log-likelihood before the fold"""
    gsel = np.zeros(len(dates), int); sel = {}
    for i, (fa, fb) in enumerate(FOLDS):
        f = pd.Timestamp(fa)
        g = 0 if i == 0 else int(np.argmax(snap[f]))
        sel[fa] = g
        gsel[(dates >= f) & (dates < pd.Timestamp(fb))] = g
    return gsel, sel


def k2_timing(pa, pb, x, gsel, w_t=0.5):
    """w_T [x mult + icpt], mult = clip(1 + (b - b0) / max(b0, floor), -1, 2), icpt = clip((a - a0) / max(b0, floor), -0.1, 0.1);
    (a0, b0) = grid point 0 (fixed coefficients). gsel: grid index per day (array) or one index for every day"""
    t = np.arange(len(x)); g = np.broadcast_to(np.asarray(gsel), t.shape)
    a, b = pa[t, g], pb[t, g]; a0, b0 = pa[:, 0], pb[:, 0]
    den = np.maximum(b0, K2_FLOOR)
    mult = np.clip(1.0 + (b - b0) / den, *K2_MULT)
    icpt = np.clip((a - a0) / den, -K2_ICPT, K2_ICPT)
    return w_t * (np.asarray(x, float) * mult + icpt), mult, icpt


# ================================================================== K3: local level on v11 - W0
def local_level(y, qs, dates, starts=FOLD_STARTS, burn=K3_BURN):
    """y_t = mu_t + eps_t, mu random walk, q = Var(eta) / Var(eps), mu_0 ~ N(0, Var(eps)) (= v15 D1). returns
    pred[t, g] = mu_{t | t-1} and, for each fold start, (sum log F, sum v^2 / F, n) over days burn .. day before the start"""
    y = np.asarray(y, float); qs = np.asarray(qs, float)
    T = len(y); G = len(qs)
    a = np.zeros(G); P = np.ones(G)
    slogF = np.zeros(G); sv2F = np.zeros(G); n = 0; snap = {}
    pred = np.empty((T, G))
    for t in range(T):
        d = dates[t]
        for f in starts:
            if f not in snap and d >= f:
                snap[f] = (slogF.copy(), sv2F.copy(), n)
        pred[t] = a
        if np.isfinite(y[t]):
            F = P + 1.0; v = y[t] - a; K = P / F
            if t >= burn:
                slogF = slogF + np.log(F); sv2F = sv2F + v * v / F; n += 1
            a = a + K * v; P = P * (1 - K)
        P = P + qs
    for f in starts:
        if f not in snap:
            snap[f] = (slogF.copy(), sv2F.copy(), n)
    return pred, snap


def concentrated_ll(sn):
    slogF, sv2F, n = sn
    if n == 0:
        return np.zeros_like(slogF)
    return -0.5 * (n * np.log(np.maximum(sv2F / n, 1e-300)) + slogF)


def k3_select(snap, dates):
    gsel = np.zeros(len(dates), int); sel = {}
    for i, (fa, fb) in enumerate(FOLDS):
        f = pd.Timestamp(fa)
        g = 0 if i == 0 else int(np.argmax(concentrated_ll(snap[f])))
        sel[fa] = g
        gsel[(dates >= f) & (dates < pd.Timestamp(fb))] = g
    return gsel, sel


def q_from_half_life(h):
    """local level: steady-state gain K = 1 - 0.5^(1/h)  <=>  q = K^2 / (1 - K) (README_v15 theorem 7)"""
    K = 1 - 0.5 ** (1.0 / h)
    return K * K / (1 - K)


def switch_frame(Da, Db, use_a):
    """daily frame of a system that uses Da on the days where use_a is True and Db otherwise"""
    idx = Da.index.intersection(Db.index)
    u = pd.Series(use_a, index=idx) if not isinstance(use_a, pd.Series) else use_a.reindex(idx).fillna(True).astype(bool)
    out = Db.loc[idx].copy()
    for col in out.columns:
        out[col] = np.where(u.values, Da.loc[idx, col].values, Db.loc[idx, col].values)
    return out


def k3_run(Da, Db, qs=K3_Q, fixed=None):
    """K3 on two daily frames: Delta = AR_a - AR_b; MLE per fold (or a fixed grid index); returns (frame, choice, selection, pred)"""
    idx = Da.index.intersection(Db.index)
    dates = pd.DatetimeIndex(idx)
    delta = (Da.loc[idx].AR - Db.loc[idx].AR).values
    pred, snap = local_level(delta, qs, dates)
    if fixed is None:
        gsel, sel = k3_select(snap, dates)
    else:
        gsel = np.full(len(dates), fixed); sel = {fa: fixed for fa, _ in FOLDS}
    mu = pred[np.arange(len(dates)), gsel]
    use_a = mu >= 0
    return switch_frame(Da, Db, pd.Series(use_a, index=idx)), pd.Series(use_a, index=idx), sel, mu


# ================================================================== evaluation helpers (= v15lib / v14lib)
for _p in (os.path.join(ROOT, 'v9', 'research'), os.path.join(ROOT, 'v8')):
    if _p not in sys.path:
        sys.path.append(_p)

EX4 = ('ex_today', 'ex_next1', 'ex_prev1', 'ex_prev2')
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


def load_npz(f, col='p'):
    z = np.load(f, allow_pickle=False)
    idx = pd.MultiIndex.from_arrays([pd.to_datetime(z['date']).astype('datetime64[ns]'), np.asarray(z['sid']).astype(str)],
                                    names=['date', 'stock_id'])
    return pd.Series(np.asarray(z[col], dtype=np.float64), index=idx, name=col)


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


def phi_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def per(s, table=PER):
    return {p: float(s[w(s.index, p, table)].mean()) for p in table}


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
            pr *= phi_cdf(delta / (sd_ / math.sqrt(n_))) if sd_ > 0 else 1.0
        out[delta] = pr
    return sds, out


class Data:
    r"""= v15lib.Data: v9 / v11 / W0 rebuilt from the cached walk-forward predictions; decide() takes an optional beta
    series (K1), run() an optional timing series by date (K2). Nothing that enters v9 / v11 / W0 is changed."""
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
        INFO['beta'] = Bt.reindex(idx).clip(*BETA_CLIP)
        INFO['beta'] = INFO.beta.fillna(INFO.groupby(level='date').beta.transform('median')).fillna(1.0)
        self.INFO = INFO[INFO.top50.astype(bool) & INFO.target_r.notna()]
        self.resid = Q.resid_r.reindex(idx)
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
        self.dates = pd.DatetimeIndex(sorted(set(idx.get_level_values('date'))))

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

    def view(self, lgbs, extras=()):
        """centred p_e: (MLP mean + LightGBM-part mean + extra families) / (2 + number of extras)"""
        m = pd.concat(self.mlp, axis=1).mean(axis=1); lg = pd.concat(lgbs, axis=1).mean(axis=1)
        tot = m + lg
        for e in extras:
            tot = tot + e
        return c(tot / (2 + len(extras)))

    def run(self, pe, v11, w_t=W_T, lam=LAM, beta=None, timing=None):
        dts = pe.index.get_level_values('date')
        if timing is None:
            pb = self.pbar(night=v11)
            s = w_t * (pb.reindex(dts).values - 0.5) + pe
        else:
            s = timing.reindex(dts).values + pe
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


def long(Mx, name):
    """dates x stocks -> Series on (date, stock_id), NaN kept"""
    X = Mx.copy(); X.index.name = 'date'; X.columns.name = 'stock_id'
    out = X.reset_index().melt(id_vars='date', var_name='stock_id', value_name=name)
    out['date'] = pd.to_datetime(out.date).astype('datetime64[ns]'); out['stock_id'] = out.stock_id.astype(str)
    return out.set_index(['date', 'stock_id'])[name].sort_index()


def hedge_ready(beta, index):
    """clip to [0.2, 2.5]; missing -> the day's median over the rows given, then 1 (PREREG_v16 K1)"""
    b = beta.reindex(index).clip(*BETA_CLIP)
    return b.fillna(b.groupby(level='date').transform('median')).fillna(1.0)

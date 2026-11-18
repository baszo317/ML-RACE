r"""
v15_signals.py -- the v15 candidates that need no training (research\PREREG_v15.md section 2, D and E), numpy + pandas only:

  D1  Kalman filter, local level model on each stock's excess intraday return: y_t = mu_t + eps_t, mu_t = mu_{t-1} + eta_t;
      q = Var(eta) / Var(eps) per stock by concentrated maximum likelihood on a grid (10^-6 .. 10^-1, 26 points), using only
      the days before each fold start; signal for day D = mu_{D | D-1} (uses y up to D-1).
  D2  causal a trous Haar wavelet de-noising (Renaud, Starck & Murtagh 2005): c_j(t) = (c_{j-1}(t) + c_{j-1}(t - 2^(j-1))) / 2,
      w_j = c_{j-1} - c_j, J = 5; soft threshold sqrt(2 ln 250) sigma_j(t), sigma_j(t) = MAD of w_j over the last 250 days / 0.6745;
      signal for day D = de-noised value of D-1: c_5 + sum_j soft(w_j).
  E1  Vasicek (1973) shrinkage of the hedge betas: the 120-day regression of hedge_beta.py (intraday return on the Top100 mean,
      days before D) plus its standard error; per day beta_bar = mean of the 100, prior variance s2 = max(cross-sectional
      variance - mean se^2, 0.01); beta_V = (s2 beta + se^2 beta_bar) / (s2 + se^2).

Leak check: every signal is recomputed with the data cut at D-1 for 6 random days D and must be identical.
-> ..\cache\signals15.pkl  (columns D1, D2, beta_raw, beta_V on (date, stock_id)); report -> stdout
usage (in v15\research): python v15_signals.py
"""
import os, sys, json, time, math
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v15lib as L

Q_GRID = 10.0 ** np.linspace(-6, -1, 26)
J_LEVELS, MAD_WIN = 5, 250
K_THR = math.sqrt(2 * math.log(MAD_WIN))


def resid_matrix(end=None):
    P = pd.read_pickle(os.path.join(L.ROOT, 'v9', 'cache', 'core9.pkl'))[['date', 'stock_id', 'resid_r']]
    P['date'] = pd.to_datetime(P.date).astype('datetime64[ns]'); P['stock_id'] = P.stock_id.astype(str)
    Y = P.pivot(index='date', columns='stock_id', values='resid_r').sort_index()
    if end is not None:
        Y = Y[Y.index <= pd.Timestamp(end)]
    return Y


# ------------------------------------------------------------------ D1 Kalman local level
def kalman_grid(Y, fold_starts):
    """run the filter for every q on the grid and every stock at once.
    returns pred[t, g, n] = mu_{t | t-1} and, for each fold start, the concentrated log-likelihood of the days before it"""
    y = Y.values; T, N = y.shape; G = len(Q_GRID)
    q = Q_GRID[:, None] * np.ones((1, N))
    a = np.zeros((G, N)); Pv = np.ones((G, N))                                         # P in units of Var(eps); mu_0 ~ N(0, Var(eps))
    slogF = np.zeros((G, N)); sv2F = np.zeros((G, N)); cnt = np.zeros((1, N))
    pred = np.empty((T, G, N), np.float32)
    starts = {pd.Timestamp(f): None for f in fold_starts}
    dates = Y.index
    snap = {}
    for t in range(T):
        d = dates[t]
        for f in list(starts):
            if f not in snap and d >= f:                                               # sums over the days strictly before f
                snap[f] = (slogF.copy(), sv2F.copy(), cnt.copy())
        pred[t] = a
        yt = y[t]; ok = np.isfinite(yt)
        F = Pv + 1.0
        v = np.where(ok, yt - a, 0.0)
        K = Pv / F
        a = np.where(ok, a + K * v, a)
        Pv = np.where(ok, Pv * (1 - K), Pv) + q
        slogF += np.where(ok, np.log(F), 0.0); sv2F += np.where(ok, v * v / F, 0.0); cnt += ok
    for f in starts:
        if f not in snap:
            snap[f] = (slogF.copy(), sv2F.copy(), cnt.copy())
    return pred, snap


def d1_signal(Y):
    """mu_{D | D-1} with the fold's q (estimated on the days before the fold start)"""
    starts = [pd.Timestamp(fa) for fa, _ in L.FOLDS]
    pred, snap = kalman_grid(Y, starts)
    out = np.full(Y.shape, np.nan); qsel = {}
    dates = Y.index
    for (fa, fb) in L.FOLDS:
        f = pd.Timestamp(fa)
        slogF, sv2F, n = snap[f]
        n = np.maximum(n, 1)
        ll = -0.5 * (n * np.log(np.maximum(sv2F / n, 1e-300)) + slogF)                 # concentrated log-likelihood
        g = np.argmax(ll, axis=0)                                                       # best q per stock
        qsel[fa] = Q_GRID[g]
        m = (dates >= f) & (dates < pd.Timestamp(fb))
        rows = np.flatnonzero(m)
        out[rows] = pred[rows][:, g, np.arange(Y.shape[1])]
    return pd.DataFrame(out, index=Y.index, columns=Y.columns), qsel


# ------------------------------------------------------------------ D2 causal a trous Haar
def d2_signal(Y):
    X = Y.fillna(0.0)                                                                   # a missing day counts as 0 excess return
    c_prev = X; parts = []
    for j in range(1, J_LEVELS + 1):
        lag = 2 ** (j - 1)
        c_j = (c_prev + c_prev.shift(lag)) / 2.0
        w_j = c_prev - c_j
        sig = w_j.abs().rolling(MAD_WIN, min_periods=60).median() / 0.6745
        thr = K_THR * sig
        parts.append(np.sign(w_j) * (w_j.abs() - thr).clip(lower=0.0))
        c_prev = c_j
    den = c_prev + sum(parts)                                                           # de-noised value at t (uses x up to t)
    return den.shift(1)                                                                 # signal for day D = value at D-1


# ------------------------------------------------------------------ E1 Vasicek betas
def betas_with_se(dates_needed, end=None):
    """hedge_beta.betas (120-day rolling OLS of the stock's intraday return on the Top100 mean, min 60 days, shift 1) plus the
    OLS standard error se = sqrt(s_e^2 / ((n - 2) var(x))), s_e^2 = var(y) - beta^2 var(x) (population moments over the window)"""
    info = json.load(open(os.path.join(L.ROOT, 'data', 'info.json')))
    R = []
    for sid in info['Top100@Y']:
        m = pd.read_csv(os.path.join(L.ROOT, 'data', 'market', f'{sid}.csv'), parse_dates=['date'])
        m = m[(m['etl:adj_open'] > 0) & (m['etl:adj_close'] > 0)]
        R.append(pd.DataFrame({'date': m.date, 'stock_id': sid, 'r': m['etl:adj_close'] / m['etl:adj_open'] - 1}))
    R = pd.concat(R, ignore_index=True)
    if end is not None:
        R = R[R.date <= pd.Timestamp(end)]
    mk = R.groupby('date').r.mean().rename('m100')
    X = R.merge(mk, left_on='date', right_index=True)
    extra = pd.MultiIndex.from_product([sorted(set(pd.to_datetime(dates_needed)) - set(X.date)), info['Top100@Y']], names=['date', 'stock_id'])
    if len(extra):
        X = pd.concat([X, extra.to_frame(index=False)], ignore_index=True)
    X = X.sort_values(['stock_id', 'date'])
    out = []
    for sid, g in X.groupby('stock_id', sort=False):
        x, y = g.m100, g.r
        roll = lambda s: s.rolling(120, min_periods=60)
        mx, my = roll(x).mean(), roll(y).mean()
        cv = roll(x * y).mean() - mx * my
        vx = roll(x).var(ddof=0); vy = roll(y).var(ddof=0)
        n = roll(x.notna() & y.notna()).sum()
        b = cv / vx
        se2 = (vy - b * b * vx).clip(lower=0) / ((n - 2).clip(lower=1) * vx)
        out.append(pd.DataFrame({'date': g.date.values, 'stock_id': sid, 'beta': b.shift(1).values, 'se2': se2.shift(1).values}))
    B = pd.concat(out, ignore_index=True)
    B['date'] = pd.to_datetime(B.date).astype('datetime64[ns]'); B['stock_id'] = B.stock_id.astype(str)
    return B.set_index(['date', 'stock_id']).sort_index()


def vasicek(B):
    g = B.groupby(level='date')
    bbar = g.beta.transform('mean')
    s2 = (g.beta.transform('var') - g.se2.transform('mean')).clip(lower=0.01)
    bv = (s2 * B.beta + B.se2 * bbar) / (s2 + B.se2)
    return bv.where(B.beta.notna() & B.se2.notna())


# ------------------------------------------------------------------ main
def long(M, name):
    """dates x stocks -> Series on (date, stock_id), NaN kept (version-independent replacement of DataFrame.stack)"""
    X = M.copy(); X.index.name = 'date'; X.columns.name = 'stock_id'
    out = X.reset_index().melt(id_vars='date', var_name='stock_id', value_name=name)
    out['date'] = pd.to_datetime(out.date).astype('datetime64[ns]'); out['stock_id'] = out.stock_id.astype(str)
    return out.set_index(['date', 'stock_id'])[name].sort_index()


def build(end=None):
    Y = resid_matrix(end)
    D1, qsel = d1_signal(Y)
    D2 = d2_signal(Y)
    B = betas_with_se(sorted(Y.index), end)
    S = pd.concat([long(D1, 'D1'), long(D2, 'D2')], axis=1)
    BV = pd.DataFrame({'beta_raw': B.beta, 'beta_se2': B.se2, 'beta_V': vasicek(B)})
    return S, BV, qsel


def main():
    t0 = time.time()
    S, BV, qsel = build()
    print(f'v15_signals.py: D1, D2 on {S.index.get_level_values("date").nunique()} days x {S.index.get_level_values("stock_id").nunique()} stocks; '
          f'betas {len(BV):,} rows ({time.time() - t0:.0f}s)')
    print('\nD1 Kalman: chosen q per fold (median over stocks, and the share at the grid edges); steady-state gain K = (-q + sqrt(q^2 + 4q)) / 2')
    for fa, q in qsel.items():
        qm = float(np.median(q)); K = (-qm + math.sqrt(qm * qm + 4 * qm)) / 2
        print(f'  {fa}: median q {qm:.2e} (K {K:.3f}, half-life {math.log(0.5) / math.log(1 - K):.0f} days), '
              f'at the lower edge {np.mean(q <= Q_GRID[0]):.2f}, upper edge {np.mean(q >= Q_GRID[-1]):.2f}')
    # leak check: cut the data at D-1, recompute, compare day D
    rng = np.random.default_rng(15)
    days = sorted(S.dropna(subset=['D1']).index.get_level_values('date').unique())
    days = [d for d in days if d >= pd.Timestamp('2022-01-01')]
    pick = sorted(rng.choice(len(days), size=6, replace=False))
    worst = {'D1': 0.0, 'D2': 0.0, 'beta_V': 0.0}
    Y = resid_matrix()
    for i in pick:
        D = days[i]; prev = days[i - 1]
        a1 = S.xs(D, level='date')
        Y2 = pd.concat([Y[Y.index <= prev], pd.DataFrame(np.nan, index=pd.DatetimeIndex([D]), columns=Y.columns)])
        k1, _ = d1_signal(Y2)
        k2 = d2_signal(Y2)
        worst['D1'] = max(worst['D1'], float(np.nanmax(np.abs(k1.loc[D].values - a1.D1.reindex(k1.columns).values))))
        worst['D2'] = max(worst['D2'], float(np.nanmax(np.abs(k2.loc[D].values - a1.D2.reindex(k2.columns).values))))
        bvD = BV.xs(D, level='date').beta_V
        B3 = betas_with_se([D], end=prev); bv3 = vasicek(B3).xs(D, level='date')
        worst['beta_V'] = max(worst['beta_V'], float(np.nanmax(np.abs(bv3.reindex(bvD.index).values - bvD.values))))
    print('\nleak check (data cut at D-1, 6 random days from 2022 on; max |difference| at D, must be 0):  '
          + ', '.join(f'{k} {v:.1e}' for k, v in worst.items()))
    ok = all(v == 0.0 or v < 1e-12 for v in worst.values())
    print('  ' + ('OK' if ok else 'FAILED: a signal uses data of day D or later'))
    d = BV.dropna()
    print(f'\nE1 betas: mean |beta_V - beta| {float((d.beta_V - d.beta_raw).abs().mean()):.3f}, mean se {float(np.sqrt(d.beta_se2).mean()):.3f}, '
          f'cross-sectional sd of beta {float(d.beta_raw.groupby(level="date").std().mean()):.3f} -> {float(d.beta_V.groupby(level="date").std().mean()):.3f}')
    os.makedirs(L.CACHE, exist_ok=True)
    pd.to_pickle({'S': S, 'BV': BV, 'qsel': qsel, 'leak_ok': ok}, os.path.join(L.CACHE, 'signals15.pkl'))
    print(f'\nsaved {os.path.join(L.CACHE, "signals15.pkl")} ({time.time() - t0:.0f}s)')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

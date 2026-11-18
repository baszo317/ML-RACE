r"""
ofit_lib.py -- shared helpers of the overfitting check (overfit\README_overfit.md).

Nothing here trains a model or changes a live file.  Inputs are the cached walk-forward daily scores of earlier versions
(Top50 AR, bp/day, 2022-01-03 .. 2026-10-22, 1,164 days):
  v9\cache\adapt_grid.npz      4,375 coefficient settings of v9 (w_t, lam, lgb_w, tf_w, bias), walk-forward + rolling static
  v9\cache\v9_daily.pkl        v5 control, v9, v9 ablations
  v11\cache\v11b_daily.pkl     v9, T1..T3, X1..X4, v11
  v11\cache\v11e_daily.pkl     X5 on v9 / v11
  v11\cache\v11f_daily.pkl     v11 + F1, W0 (= the 10/27 upload: v11, w_T 0, lambda 0.5), W0 + F1, F2
  v12\cache\v12_daily.pkl      Y1..Y6 on v9 / v11, v12
  v10\cache\v10_daily.pkl      V10 and its ablations
  v10\cache\a158_daily.pkl     23 Alpha158 groups (+ ALL) added to the LightGBM (lgb|...) or the MLPs (mlp|...)
Periods are those of every earlier version: val 2022-01..2024-12 (decides), hold 2025-01..2026-05 (confirms), post 2026-06..
"""
import os, itertools
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); OFIT = os.path.dirname(HERE); ROOT = os.path.dirname(OFIT)
CACHE = os.path.join(OFIT, 'cache'); RESULTS = os.path.join(OFIT, 'results')
PER = {'val': ('2022-01-01', '2025-01-01'), 'hold': ('2025-01-01', '2026-06-01'), 'post': ('2026-06-01', '2100-01-01'),
       'static': ('2026-06-01', '2026-10-01'), 'Oct': ('2026-10-01', '2100-01-01'), 'all': ('2022-01-01', '2100-01-01')}
GRID = {'w_t': [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5], 'lam': [0.0, 0.25, 0.5, 0.75, 1.0], 'lgb_w': [0.0, 0.25, 0.5, 0.75, 1.0],
        'tf_w': [0.0, 0.25, 0.5, 0.75, 1.0], 'bias': [-0.02, -0.01, 0.0, 0.01, 0.02]}       # = v9\research\adapt_eval.py
NAMES = list(GRID)
SHAPE = tuple(len(GRID[k]) for k in NAMES)
V9DEF = {'w_t': 0.5, 'lam': 0.75, 'lgb_w': 0.5, 'tf_w': 0.5, 'bias': 0.0}


def mask(ix, k):
    a, b = PER[k]; ix = pd.DatetimeIndex(ix)
    return np.asarray((ix >= pd.Timestamp(a)) & (ix < pd.Timestamp(b)))


def grid_key(**kw):
    """column of adapt_grid for a setting (missing coefficients = v9's)"""
    c = {**V9DEF, **kw}
    return int(np.ravel_multi_index(tuple(GRID[k].index(c[k]) for k in NAMES), SHAPE))


def grid_settings():
    """DataFrame (K rows) with the coefficients of every column of adapt_grid"""
    idx = np.indices(SHAPE).reshape(len(SHAPE), -1).T
    return pd.DataFrame({k: np.array(GRID[k])[idx[:, i]] for i, k in enumerate(NAMES)})


def load_grid():
    """walk-forward grid (1164 x 4375) and rolling-static grid (99 x 4375), bp/day"""
    z = np.load(os.path.join(ROOT, 'v9', 'cache', 'adapt_grid.npz'), allow_pickle=False)
    return (pd.DatetimeIndex(z['dW']), z['AW']), (pd.DatetimeIndex(z['dS']), z['AS'])


def _read(rel):
    return pd.read_pickle(os.path.join(ROOT, *rel.split('/')))


def load_series():
    """every cached daily AR series of the development (name -> Series, bp/day on the same 1,164 days)"""
    S = {}
    d9 = _read('v9/cache/v9_daily.pkl')
    S['v5 (control)'] = d9['control: v4 rule, current members (= v5 live)'].AR
    S['v9'] = d9['v9: all 11 changes'].AR
    for k, v in d9.items():
        if k.startswith('ablation') or k.startswith('split'):
            S['v9 ' + k] = v.AR
    b = _read('v11/cache/v11b_daily.pkl')
    for k in ('T1', 'T2', 'T3', 'X1', 'X2', 'X3', 'X4'):
        S['v9 + ' + k] = b[k].AR
    S['v11'] = b['v11'].AR
    e = _read('v11/cache/v11e_daily.pkl')
    S['v9 + X5'] = e['v9 + X5'].AR; S['v11 + X5'] = e['v11 + X5'].AR
    f = _read('v11/cache/v11f_daily.pkl')
    S['v11 + F1'] = f['v11 + F1'].AR; S['W0'] = f['W0 (10/27 upload)'].AR; S['W0 + F1'] = f['W0 + F1'].AR
    S['F2'] = f['F2 (half and half)'].AR
    y = _read('v12/cache/v12_daily.pkl')['daily']
    for k, v in y.items():
        if k not in ('v9', 'v11'):
            S[k] = v.AR
    t = _read('v10/cache/v10_daily.pkl')
    S['v9 replica (v10)'] = t['v9 replica (same machine as V10)'].AR; S['V10'] = t['V10'].AR
    S['V10 only L'] = t['ablation: V10 with only the L change'].AR; S['V10 only M'] = t['ablation: V10 with only the M change'].AR
    S['V10 ref ALL'] = t['reference: ALL 158 in L and M (MLP seed 1-5)'].AR
    S['v9 replica MLP seed 1-5'] = t['reference: v9 replica with MLP seed 1-5'].AR
    a = _read('v10/cache/a158_daily.pkl')
    for k, v in a.items():
        if k != 'v9':
            S['a158 ' + k] = v.AR
    idx = S['v9'].index
    for k in S:
        S[k] = S[k].reindex(idx)
    return S


def candidates(S):
    """(name, base series, variant series, how it was selected) of every change that was screened during the development.
    'rule' says which earlier rule judged it (all of them looked at val first)."""
    C = []
    for k in ('T1', 'T2', 'T3', 'X1', 'X2', 'X3', 'X4'):
        C.append((f'v11b {k}', 'v9', f'v9 + {k}', 'val>0'))
    C.append(('v11e X5 on v11', 'v11', 'v11 + X5', 'val>0'))
    for j in range(1, 7):
        C.append((f'v12 Y{j} on v11', 'v11', f'v11 + Y{j}', 'val>0'))
    C.append(('v11f F1 on v11', 'v11', 'v11 + F1', '48 days'))
    C.append(('v11f F2 vs W0', 'W0', 'F2', 'all three >= W0'))
    for g in ('lgb', 'mlp'):
        for k in S:
            if k.startswith(f'a158 {g}|') and k != f'a158 {g}|base':
                C.append((f'v10 {k[5:]}', f'a158 {g}|base', k, 'val t>0'))
    C.append(('v10 V10', 'v9 replica (v10)', 'V10', 'hold, post > 0'))
    for k in S:
        if k.startswith('v9 ablation'):
            C.append(('v9 keep ' + k.split('without ')[-1], k, 'v9', 'component of v9'))
    C.append(('v9 vs v5', 'v5 (control)', 'v9', 'hold, post > 0'))
    C.append(('v11 vs v9', 'v9', 'v11', 'hold, post > 0'))
    return C


# ------------------------------------------------------------------ statistics
def nw_se(d, lags=5):
    """Newey-West standard error of the mean of d (Bartlett weights, as v8 mlib.nw_t)"""
    d = np.asarray(d, float); d = d[~np.isnan(d)]; n = len(d)
    e = d - d.mean(); s = e @ e / n
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return np.sqrt(max(s, 0.0) / n)


def nw_t(d, lags=5):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    se = nw_se(d, lags)
    return d.mean() / se if se > 0 else np.nan


def stationary_bootstrap_idx(n, B, mean_block, rng):
    """Politis-Romano stationary bootstrap: B x n index matrix, geometric block lengths with the given mean"""
    p = 1.0 / mean_block
    I = np.empty((B, n), dtype=np.int64)
    I[:, 0] = rng.integers(0, n, B)
    new = rng.random((B, n)) < p
    starts = rng.integers(0, n, (B, n))
    for t in range(1, n):
        I[:, t] = np.where(new[:, t], starts[:, t], (I[:, t - 1] + 1) % n)
    return I


def cscv(M, S=16, chunk=512):
    """Combinatorially symmetric cross-validation (Bailey, Borwein, Lopez de Prado & Zhu 2017).
    M: T x N daily performance.  Rows are cut into S contiguous blocks; for each of the C(S, S/2) ways of calling half
    of the blocks in-sample (IS) and the other half out-of-sample (OOS), the setting with the best IS mean is taken and
    its OOS relative rank w (0..1) recorded.  Returns dict with the per-split arrays and PBO = P(w <= 1/2)."""
    T, N = M.shape
    parts = np.array_split(np.arange(T), S)
    size = np.array([len(p) for p in parts], float)
    Bm = np.vstack([M[p].mean(axis=0) for p in parts])                 # S x N block means
    combs = np.array(list(itertools.combinations(range(S), S // 2)))
    out = {k: [] for k in ('is_best', 'oos_of_best', 'oos_median', 'rank', 'pick')}
    for s in range(0, len(combs), chunk):
        cb = combs[s:s + chunk]
        W = np.zeros((len(cb), S)); W[np.arange(len(cb))[:, None], cb] = 1.0
        Wi = W * size; Wo = (1 - W) * size
        IS = (Wi @ Bm) / Wi.sum(1, keepdims=True)
        OOS = (Wo @ Bm) / Wo.sum(1, keepdims=True)
        pick = IS.argmax(1)
        ob = OOS[np.arange(len(cb)), pick]
        # relative rank of the picked setting OOS (ties -> average rank), in (0, 1)
        less = (OOS < ob[:, None]).sum(1); eq = (OOS == ob[:, None]).sum(1)
        rank = (less + 0.5 * (eq + 1)) / (N + 1)
        out['is_best'].append(IS[np.arange(len(cb)), pick]); out['oos_of_best'].append(ob)
        out['oos_median'].append(np.median(OOS, axis=1)); out['rank'].append(rank); out['pick'].append(pick)
    out = {k: np.concatenate(v) for k, v in out.items()}
    out['pbo'] = float((out['rank'] <= 0.5).mean())
    lam = np.log(out['rank'] / (1 - out['rank']))
    out['logit'] = lam
    return out


def norm_cdf(x):
    from math import erf, sqrt
    return 0.5 * (1 + np.vectorize(erf)(np.asarray(x, float) / sqrt(2)))


def expected_max_normal(K):
    """E[max of K iid N(0,1)] by numerical integration (numpy only)"""
    x = np.linspace(-10, 10, 40001)
    pdf = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    F = norm_cdf(x)
    return float(np.trapezoid(x * K * pdf * F ** (K - 1), x)) if hasattr(np, 'trapezoid') else float(np.trapz(x * K * pdf * F ** (K - 1), x))


def fmt(x, d=2):
    return 'nan' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{d}f}'

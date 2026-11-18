r"""
v12lib.py -- shared pieces of v12 (research\PREREG_v12.md): paths, the label transforms, the within-day pairwise logistic
objective (numba), unit files under v12\cache\screen\<name>\ and loading the walk-forward predictions.
Reads v9 / v10 / v11 files, never writes them.
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); V12 = os.path.dirname(HERE); ROOT = os.path.dirname(V12)
for p in (os.path.join(ROOT, 'v10', 'research'), os.path.join(ROOT, 'v9', 'research'), os.path.join(ROOT, 'v11', 'research'), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import v10lib as V
try:
    from numba import njit
except ImportError:                 # only v12_train.py needs numba (pairwise models); evaluation runs without it
    def njit(*a, **k):
        return a[0] if a and callable(a[0]) else (lambda f: f)

SCREEN = os.path.join(V12, 'cache', 'screen')
CLIP = 0.05                                     # the same 5% cap as lgb_res's sample weights
MAG_STEP = 0.0025                               # lambdarank levels for the size label: 0.25% steps -> 0..40

# ------------------------------------------------------------------ labels (rows sorted by date)
def decile_label(Q):
    """within-day decile of resid_r, 0..9 (identical to v11e_train.decile_label, X5)"""
    r = Q.groupby('date').resid_r.rank(method='first'); n = Q.groupby('date').resid_r.transform('size')
    return np.minimum(9, np.floor((r.values - 1) * 10 / n.values)).astype(int)

def size_label(Q):
    """clip(resid_r, -5%, 5%) in percent"""
    return np.clip(Q.resid_r.values.astype(float), -CLIP, CLIP) * 100.0

def size_level(Q):
    """size label as lambdarank levels 0..40 (0.25% steps), gain linear in the level = linear in the clipped return"""
    return np.rint((np.clip(Q.resid_r.values.astype(float), -CLIP, CLIP) + CLIP) / MAG_STEP).astype(int)

# ------------------------------------------------------------------ within-day pairwise logistic loss
def day_starts(dates):
    """start offset of each day in rows sorted by date, plus the end"""
    d = np.asarray(dates)
    cut = np.flatnonzero(d[1:] != d[:-1]) + 1
    return np.concatenate([[0], cut, [len(d)]]).astype(np.int64)

def day_scale(y, starts):
    """c_d so that every day's pair weights c_d |y_i - y_j| sum to n_d / 2 (each stock's total pair weight averages 1)"""
    c = np.empty(len(starts) - 1)
    for k in range(len(starts) - 1):
        v = np.sort(y[starts[k]:starts[k + 1]]); n = len(v)
        tot = float(np.sum(v * (2 * np.arange(1, n + 1) - n - 1)))          # sum over pairs of |v_i - v_j|
        c[k] = (n / 2.0) / tot if tot > 0 else 0.0
    return c

@njit(cache=True)
def _pairwise_grad(s, y, starts, c, grad, hess):
    """pairs (i, j) of the same day with y_i > y_j, loss w log(1 + exp(-(s_i - s_j))), w = c_d (y_i - y_j):
    gradient and diagonal hessian (rho = 1 / (1 + exp(s_i - s_j)) = E_j / (E_i + E_j) with E = exp(s - day max))"""
    for d in range(len(starts) - 1):
        a = starts[d]; b = starts[d + 1]; cd = c[d]
        if b - a < 2 or cd == 0.0:
            continue
        mx = s[a]
        for i in range(a, b):
            mx = max(mx, s[i])
        E = np.exp(s[a:b] - mx)
        for i in range(a, b):
            for j in range(i + 1, b):
                dy = y[i] - y[j]
                if dy == 0.0:
                    continue
                if dy > 0:
                    hi = i; lo = j; w = dy * cd
                else:
                    hi = j; lo = i; w = -dy * cd
                den = E[hi - a] + E[lo - a]
                rho = E[lo - a] / den if den > 0 else 0.5
                g = w * rho; h = g * (1.0 - rho)
                grad[hi] -= g; grad[lo] += g; hess[hi] += h; hess[lo] += h

@njit(cache=True)
def _pairwise_loss(s, y, starts, c):
    """the loss sum (numerically stable form)"""
    loss = 0.0
    for d in range(len(starts) - 1):
        a = starts[d]; b = starts[d + 1]; cd = c[d]
        for i in range(a, b):
            for j in range(i + 1, b):
                dy = y[i] - y[j]
                if dy == 0.0:
                    continue
                if dy > 0:
                    w = dy * cd; ds = s[i] - s[j]
                else:
                    w = -dy * cd; ds = s[j] - s[i]
                if ds >= 0:
                    loss += w * np.log1p(np.exp(-ds))
                else:
                    loss += w * (np.log1p(np.exp(ds)) - ds)
    return loss

class PairwiseObjective:
    """LightGBM custom objective + metric for one dataset (rows sorted by date)"""
    def __init__(self, y, dates):
        self.y = np.ascontiguousarray(y, dtype=np.float64); self.starts = day_starts(dates)
        self.c = day_scale(self.y, self.starts); self.ndays = len(self.starts) - 1

    def grad_hess(self, s):
        g = np.zeros(len(self.y)); h = np.zeros(len(self.y))
        _pairwise_grad(np.ascontiguousarray(s, dtype=np.float64), self.y, self.starts, self.c, g, h)
        return g, h

    def loss(self, s):
        """mean over days of the day's loss sum"""
        return _pairwise_loss(np.ascontiguousarray(s, dtype=np.float64), self.y, self.starts, self.c) / self.ndays

def lgb_objective(obj):
    def f(preds, data):
        return obj.grad_hess(preds)
    return f

def lgb_metric(obj):
    def f(preds, data):
        return 'pairwise_logloss', obj.loss(preds), False
    return f

# ------------------------------------------------------------------ unit files
def unit_path(name, fold_a, seed):
    return os.path.join(SCREEN, name, f'{fold_a}_seed{seed}.npz')

def load_member(name, seeds=range(1, 6)):
    """seed mean over all folds (Series indexed by date, stock_id); None if a unit is missing"""
    parts = []
    for fa, _ in V.FOLDS:
        fs = [unit_path(name, fa, s) for s in seeds]
        if not all(V.unit_exists(f) for f in fs):
            return None
        parts.append(pd.concat([V.load_unit(f) for f in fs], axis=1).mean(axis=1))
    return pd.concat(parts).sort_index()

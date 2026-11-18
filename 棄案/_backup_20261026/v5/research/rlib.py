"""
rlib.py -- shared helpers for the v5 research scripts (walk-forward predictions, scoring, paired tests).

Periods (same as v2/v4): val = 2022-2024 (decides), hold = 2025-01..2026-05 (confirms), post = 2026-06.. (frozen
models at 2026-05-29 in the yearly protocol). AR = competition score in bp/day on the Top50:
    AR_t = mean_i (2 d_i - 1) * r_i,  timing_t = mean_i(2 d_i - 1) * mean_i(r_i),  xsec_t = AR_t - timing_t.
"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); V5 = os.path.dirname(HERE); ROOT = os.path.dirname(V5)
V4 = os.path.join(ROOT, 'v4')
if V4 not in sys.path:
    sys.path.insert(0, V4)
import ens_lib as E

WF = os.path.join(ROOT, 'v2', 'results_v2', 'wf')
V1_TF = os.path.join(ROOT, 'transformer', 'results', 'wf_transformer_B.pkl')
YEAR_FOLDS = [('2022-01-01', '2023-01-01'), ('2023-01-01', '2024-01-01'), ('2024-01-01', '2025-01-01'),
              ('2025-01-01', '2026-01-01'), ('2026-01-01', '2026-06-01')]
POST_START = pd.Timestamp('2026-06-01'); CUT = pd.Timestamp('2025-01-01')
SPEC = {'mlp_raw_B': ('mlp', 'B', None, 'raw'), 'mlp_res_B': ('mlp', 'B', None, 'resid'),
        'mlp_raw_xr': ('mlp', 'B+XR', None, 'raw'), 'mlp_res_xr': ('mlp', 'B+XR', None, 'resid'),
        'tf_raw_B': ('transformer', 'B', None, 'raw'), 'tf_res_B': ('transformer', 'B', None, 'resid'),
        'mlp_raw_w4': ('mlp', 'B', 4, 'raw'), 'tf_raw_w4': ('transformer', 'B', 4, 'raw')}
LATE_FILL = {'mlp_raw_w4': 'mlp_raw_B', 'tf_raw_w4': 'tf_raw_B'}
CACHE = os.environ.get('V5_CACHE', os.path.join(HERE, 'cache'))

_P = None
def panel():
    global _P
    if _P is None:
        f = os.path.join(CACHE, 'panel.pkl')
        if os.path.exists(f):
            _P = pd.read_pickle(f)
        else:
            _P = E.build_panel(E.DATA); _P['date'] = _P['date'].astype('datetime64[ns]')
            os.makedirs(CACHE, exist_ok=True); _P.to_pickle(f)
    return _P

def member_def(name):
    kind, feats, window, target = SPEC[name]
    return E.member(name, kind, window=window, feats=feats, target=target)

def _read(f):
    x = pd.read_pickle(f) if not isinstance(f, pd.DataFrame) else f
    x = x[['date', 'stock_id', 'p']].copy()
    x['date'] = pd.to_datetime(x.date).astype('datetime64[ns]'); x['stock_id'] = x.stock_id.astype(str)
    return x.set_index(['date', 'stock_id']).p

_V1 = None
def _v1_slice(a, b):
    global _V1
    if _V1 is None:
        _V1 = pd.read_pickle(V1_TF)
    return _read(_V1[(_V1.date >= a) & (_V1.date < b)])

def _load_folds(name, seeds, starts):
    """weighted seed mean per fold, exactly as overnight_experiments.load_member: tf_raw_B seeds 1-3 on the yearly
    folds exist only as the v1 seed mean (weight 3)"""
    parts = []
    for a in starts:
        b = dict(YEAR_FOLDS).get(a, '2100-01-01')
        units = []
        for s in seeds:
            f = os.path.join(WF, name, f'{a}_seed{s}.pkl')
            if os.path.exists(f):
                units.append((1, _read(f)))
        if name == 'tf_raw_B' and a != '2026-06-01' and {1, 2, 3} <= set(seeds):
            units.append((3, _v1_slice(a, b)))
        if sum(w for w, _ in units) < len(seeds):
            raise FileNotFoundError(f'{name} fold {a}: {sum(w for w, _ in units)} of {len(seeds)} seeds')
        M = pd.concat([x for _, x in units], axis=1); w = np.array([w for w, _ in units], float)
        parts.append(pd.Series(M.values @ w / w.sum(), index=M.index))
    return pd.concat(parts)

def load_wf(name, seeds):
    """seed average of member `name` over the yearly folds + post (Series indexed by date, stock_id)"""
    seeds = list(seeds)
    starts = [a for a, _ in YEAR_FOLDS] + ['2026-06-01']
    if name in LATE_FILL:
        base = load_wf(LATE_FILL[name], seeds if name.startswith('mlp') else [1, 2, 3])
        own = _load_folds(name, seeds, starts[3:])
        d = base.index.get_level_values('date')
        return pd.concat([base[d < CUT], own])
    return _load_folds(name, seeds, starts)

# ------------------------------------------------------------------ scoring
def frame(scores, P=None):
    """scores: DataFrame indexed by (date, stock_id) with a 'score' column (prediction 1 if > 0) or 'pred' -> merged with
    labels for the labelled Top50 rows"""
    P = panel() if P is None else P
    q = scores.reset_index().merge(P[['date', 'stock_id', 'top50', 'target_r', 'ex_today', 'resid_r', 'group']],
                                   on=['date', 'stock_id'])
    return q[q.target_r.notna()]

def daily(scores, exdiv=True, P=None, pred_col=None):
    """-> DataFrame per date: AR, timing, xsec, share1 (Top50, bp/day)"""
    q = frame(scores, P)
    pred = (q[pred_col] if pred_col else (q.score > 0)).astype(int).values
    if exdiv:
        pred = np.where(q.ex_today.values == 1, 0, pred)
    q = q.assign(s=2 * pred - 1)
    x = q[q.top50]
    g = x.groupby('date')
    AR = (x.s * x.target_r).groupby(x.date).mean() * 1e4
    T = g.s.mean() * g.target_r.mean() * 1e4
    return pd.DataFrame({'AR': AR, 'timing': T, 'xsec': AR - T, 'share1': (g.s.mean() + 1) / 2, 'm': g.target_r.mean() * 1e4})

def periods(idx):
    idx = pd.DatetimeIndex(idx)
    return {'val': idx < CUT, 'hold': (idx >= CUT) & (idx < POST_START), 'post': idx >= POST_START, 'all': np.ones(len(idx), bool)}

def nw_t(d, lags=5):
    """t-stat of mean(d) with Newey-West (Bartlett) standard error"""
    d = np.asarray(d, float); d = d[~np.isnan(d)]; n = len(d)
    if n < 3:
        return np.nan
    e = d - d.mean(); g0 = e @ e / n; s = g0
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return d.mean() / np.sqrt(s / n) if s > 0 else np.nan

def plain_t(d):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    return d.mean() / d.std(ddof=1) * np.sqrt(len(d)) if len(d) > 2 and d.std() > 0 else np.nan

def summary_row(D, base=None, label='', col='AR'):
    r = {'config': label}
    pr = periods(D.index)
    for k, m in pr.items():
        r[k] = D[col][m].mean()
    r['sd'] = D[col].std(); r['timing'] = D.timing.mean(); r['xsec'] = D.xsec.mean(); r['share1'] = D.share1.mean()
    if base is not None:
        d = (D[col] - base[col].reindex(D.index))
        for k, m in pr.items():
            r['d_' + k] = d[m].mean(); r['t_' + k] = plain_t(d[m])
        r['nwt_val'] = nw_t(d[pr['val']]); r['nwt_all'] = nw_t(d[pr['all']])
    return r

def combine(probs, names, w_t=0.5):
    """v2/v4 rule from member probability Series"""
    members = [member_def(n) if n in SPEC else n for n in names]
    pr = {m['name']: probs[m['name']] for m in members}
    idx = None
    for v in pr.values():
        idx = v.index if idx is None else idx.intersection(v.index)
    pr = {k: v.reindex(idx) for k, v in pr.items()}
    return E.combine(pr, members, w_t=w_t)

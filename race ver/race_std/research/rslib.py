r"""
rslib.py -- shared code of RACE-STD v1 (research\PREREG_race_std.md): paths, folds, periods, unit / candidate registry,
unit files, decisions and metrics. numpy + pandas only at import time (lightgbm / scikit-learn are imported in learners.py).

Environment variables:
  MLRACE_ROOT  the ML RACE folder (default: two levels above race_std)
  RACE_TAG     ''  = the official run (cache\, results\);  'win' etc. = a reproduction in cache_win\, results_win\
"""
import os, sys, json, math, hashlib, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STD = os.path.dirname(HERE)                                                   # ...\race ver\race_std
ROOT = os.environ.get('MLRACE_ROOT', os.path.dirname(os.path.dirname(STD)))    # ...\ML RACE
TAG = os.environ.get('RACE_TAG', '').strip()
CACHE = os.path.join(STD, 'cache' + ('_' + TAG if TAG else ''))
RESULTS = os.path.join(STD, 'results' + ('_' + TAG if TAG else ''))
DATA = os.path.join(ROOT, 'data')
TX_CSV = os.path.join(ROOT, 'v7', 'data', 'tx_daily.csv')
M2ROOT = os.path.join(ROOT, 'race ver', 'NTU_M2_Final_System', 'NTU_M2_Final_System')
PANEL = os.path.join(CACHE, 'panel.pkl')
FEATS = os.path.join(CACHE, 'features.json')
PREREG = os.path.join(HERE, 'PREREG_race_std.md')
LOCK = os.path.join(RESULTS, 'LOCK.json')
SEALED = os.path.join(RESULTS, 'SEALED_V.json')

FOLDS = [('2022-01-01', '2023-01-01'), ('2023-01-01', '2024-01-01'), ('2024-01-01', '2025-01-01'),
         ('2025-01-01', '2026-01-01'), ('2026-01-01', '2026-06-01'), ('2026-06-01', '2100-01-01')]   # = v9..v16 folds
D_FOLDS, V_FOLDS = FOLDS[:3], FOLDS[3:]
TRAIN_START = pd.Timestamp('2020-03-02')
PER = {'D': ('2022-01-01', '2025-01-01'), 'V': ('2025-01-01', '2100-01-01')}
FOLD_NAME = {'2022-01-01': '2022', '2023-01-01': '2023', '2024-01-01': '2024', '2025-01-01': '2025',
             '2026-01-01': '2026-01..05', '2026-06-01': '2026-06..'}
EX4 = ('ex_today', 'ex_next1', 'ex_prev1', 'ex_prev2')
GROUPS = ('U', 'P', 'C', 'M')
LEARNERS = ('LU', 'LN', 'LP', 'LC', 'LM')
SEEDS = {'LU': (1, 2, 3), 'LN': (1, 2, 3), 'LP': (0,), 'LC': (0,), 'LM': (2026, 2027, 2028)}
LEARNER_LABEL = {'LU': 'LightGBM binary (Baszo lgb_res)', 'LN': 'MLP (Baszo mlp_res)', 'LP': 'HistGBR (FinPred)',
                 'LC': 'LightGBM Huber z (Chen V5)', 'LM': 'LightGBM regression (NTU M2)'}
NATIVE = {'U-native': None, 'P-native': 'P.LP', 'C-native': 'C.LC', 'M-native': 'M.LM'}
U_NATIVE_MLP = ['UB9S.LN', 'UXR9.LN', 'UBN9.LN', 'UXRN9.LN']        # mlp_res_B9, mlp_res_xr9, mlp_res_BN9, mlp_res_xrN9
U_NATIVE_LGB = ['UB.LU', 'UXRN9.LU']                                  # lgb_res, lgb_res_xrN (X1)


def log_to(path):
    def f(*a):
        s = time.strftime('%H:%M:%S') + ' ' + ' '.join(str(x) for x in a)
        print(s, flush=True)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'a', encoding='utf-8') as fh:
                fh.write(s + '\n')
        except OSError:
            pass
    return f


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def code_hashes():
    return {f: sha256(os.path.join(HERE, f)) for f in sorted(os.listdir(HERE)) if f.endswith('.py')}


def dict_sha256(d):
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


# ------------------------------------------------------------------ feature registry (cache\features.json)
def features():
    with open(FEATS, encoding='utf-8') as fh:
        return json.load(fh)


def feature_list(key, F=None):
    """feature columns of a unit's feature key: U P C M ALL, ALLmU .. ALLmM (ablations), UB UB9S UXR9 UBN9 UXRN9 (U-native)"""
    F = F or features()
    if key in F:
        return list(F[key])
    if key == 'ALL':
        return [c for g in GROUPS for c in F[g]]
    if key.startswith('ALLm') and key[4:] in GROUPS:
        return [c for g in GROUPS if g != key[4:] for c in F[g]]
    raise KeyError(key)


def unit_parts(unit):
    fk, ln = unit.rsplit('.', 1)
    assert ln in LEARNERS, unit
    return fk, ln


# ------------------------------------------------------------------ candidates (PREREG 2.3)
def grid_units():
    return [f'{g}.{l}' for g in GROUPS + ('ALL',) for l in LEARNERS]


def stage1_candidates(lstar=None):
    """name -> dict(kind, units); kind: unit (one unit), unative (W0 p_e structure), zavg (mean of daily z of other candidates)"""
    C = {}
    for u in grid_units():
        C[u] = dict(kind='unit', units=[u])
    C['U-native'] = dict(kind='unative', units=U_NATIVE_MLP + U_NATIVE_LGB)
    C['ENS4'] = dict(kind='zavg', of=['U-native', 'P.LP', 'C.LC', 'M.LM'])
    C['ALL5'] = dict(kind='zavg', of=[f'ALL.{l}' for l in LEARNERS])
    if lstar is not None:
        for g in GROUPS:
            C[f'ALLm{g}.{lstar}'] = dict(kind='unit', units=[f'ALLm{g}.{lstar}'])
    for k, v in C.items():
        if v['kind'] == 'zavg':
            v['units'] = sorted({u for c in v['of'] for u in C[c]['units']})
    return C


def native_name(cand):
    for k, v in NATIVE.items():
        if v == cand or (k == 'U-native' and cand == 'U-native'):
            return k
    return None


def complexity(cdef, F=None):
    """(units, distinct features) -- the order of the 1-SE rule (PREREG 3.1)"""
    F = F or features()
    feats = set()
    for u in cdef['units']:
        feats |= set(feature_list(unit_parts(u)[0], F))
    return (len(cdef['units']), len(feats))


# ------------------------------------------------------------------ unit files
def unit_file(unit, fold_start, seed, test=False):
    return os.path.join(CACHE, 'units_test' if test else 'units', unit, f'{fold_start}_seed{seed}.npz')


def save_unit(f, dates, sids, p, meta):
    os.makedirs(os.path.dirname(f), exist_ok=True)
    tmp = f[:-4] + '.tmp.npz'
    np.savez_compressed(tmp, date=pd.to_datetime(dates).values.astype('datetime64[ns]').astype(np.int64),
                        sid=np.asarray([str(s) for s in sids], dtype='<U8'), p=np.asarray(p, np.float64),
                        meta=np.asarray(json.dumps(meta, default=float)))
    os.replace(tmp, f)


def load_npz(f):
    z = np.load(f, allow_pickle=False)
    idx = pd.MultiIndex.from_arrays([pd.to_datetime(z['date']).astype('datetime64[ns]'), np.asarray(z['sid']).astype(str)],
                                    names=['date', 'stock_id'])
    return pd.Series(np.asarray(z['p'], np.float64), index=idx, name='p'), json.loads(str(z['meta']))


def load_unit(unit, folds, test=False):
    """seed mean over the folds; None if any (fold, seed) file is missing"""
    parts = []
    for fa, _ in folds:
        ss = []
        for s in SEEDS[unit_parts(unit)[1]][:1] if test else SEEDS[unit_parts(unit)[1]]:
            f = unit_file(unit, fa, s, test)
            if not os.path.exists(f):
                return None
            ss.append(load_npz(f)[0])
        parts.append(pd.concat(ss, axis=1).mean(axis=1))
    return pd.concat(parts).sort_index()


def unit_metas(unit, folds, test=False):
    out = []
    for fa, _ in folds:
        for s in SEEDS[unit_parts(unit)[1]][:1] if test else SEEDS[unit_parts(unit)[1]]:
            f = unit_file(unit, fa, s, test)
            if os.path.exists(f):
                m = load_npz(f)[1]; m.update(fold=fa, seed=s); out.append(m)
    return out


# ------------------------------------------------------------------ scores of candidates
def zday(s):
    """within-day z over the stocks that have a value (100), missing -> 0, clipped to +-3 (= v14lib.zday)"""
    g = s.groupby(level='date')
    sd = g.transform('std')
    return ((s - g.transform('mean')) / sd.where(sd > 0)).clip(-3, 3).fillna(0.0)


def candidate_score(name, C, units):
    """units: dict unit -> Series (seed mean). Returns the candidate's score (higher = more likely to beat the day)"""
    d = C[name]
    if d['kind'] == 'unit':
        return units[d['units'][0]]
    if d['kind'] == 'unative':
        mlp = pd.concat([units[u] for u in U_NATIVE_MLP], axis=1).mean(axis=1)
        lgb = pd.concat([units[u] for u in U_NATIVE_LGB], axis=1).mean(axis=1)
        return (mlp + lgb) / 2
    if d['kind'] == 'zavg':
        zs = [zday(candidate_score(c, C, units)) for c in d['of']]
        return pd.concat(zs, axis=1).mean(axis=1)
    raise ValueError(d['kind'])


# ------------------------------------------------------------------ decision SD and metrics (PREREG 1.4)
def info_frame(P):
    """Top50 rows with what the decision and the score need, indexed (date, stock_id)"""
    X = P.loc[P.top50.astype(bool), ['date', 'stock_id', 'target_r', 'resid_r'] + list(EX4)].copy()
    return X.set_index(['date', 'stock_id']).sort_index()


def load_info():
    X = pd.read_pickle(os.path.join(CACHE, 'info.pkl'))
    X['date'] = pd.to_datetime(X.date).astype('datetime64[ns]'); X['stock_id'] = X.stock_id.astype(str)
    return X


def decide_sd(score, INFO, k=25, exdiv=True, mode='top'):
    """SD (mode 'top'): per day, Top50 by score (ties by stock id), top k -> +1 else -1; ex-dividend rows -> -1.
    mode 'zero': +1 if score > 0 (the market-view rule of PREREG 3.2), then the ex-dividend rule.
    Returns DataFrame indexed by date: AR (bp), n, ones, ic (Spearman of score and r over the scored Top50), m (Top50 mean r, bp)."""
    X = INFO.join(score.rename('score'), how='left')
    rows = []
    for d, g in X.groupby(level='date', sort=True):
        sid = g.index.get_level_values('stock_id').to_numpy()
        sc = g.score.to_numpy(float)
        sc = np.where(np.isnan(sc), -np.inf, sc)
        s = -np.ones(len(g))
        if mode == 'top':
            order = np.lexsort((sid, -sc))                # score desc, then stock id asc
            s[order[:min(k, len(g))]] = 1.0
        else:
            s[sc > 0] = 1.0
        if exdiv:
            ex = np.zeros(len(g), bool)
            for col in EX4:
                ex |= g[col].to_numpy() == 1
            s[ex] = -1.0
        r = g.target_r.to_numpy(float)
        ok = ~np.isnan(r)
        if ok.sum() == 0:
            continue
        ar = float((s[ok] * r[ok]).mean() * 1e4)
        scs = g.score.to_numpy(float)
        ok2 = ok & ~np.isnan(scs)
        ic = float(pd.Series(scs[ok2]).rank().corr(pd.Series(r[ok2]).rank())) if ok2.sum() > 5 else np.nan
        rows.append((d, ar, int(ok.sum()), int((s[ok] > 0).sum()), ic, float(r[ok].mean() * 1e4)))
    return pd.DataFrame(rows, columns=['date', 'AR', 'n', 'ones', 'ic', 'm']).set_index('date')


def nw_t(d, lags=5):                                                     # = v15lib.nw_t
    d = np.asarray(d, float); d = d[~np.isnan(d)]; n = len(d)
    if n < 3:
        return np.nan
    e = d - d.mean(); s = e @ e / n
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return d.mean() / np.sqrt(s / n) if s > 0 else np.nan


def nw_se(d, lags=5):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    t = nw_t(d, lags)
    return abs(d.mean() / t) if (t == t and t != 0) else np.nan


def in_per(idx, a, b):
    idx = pd.DatetimeIndex(idx)
    return (idx >= pd.Timestamp(a)) & (idx < pd.Timestamp(b))


def summary(D, a, b):
    x = D[in_per(D.index, a, b)]
    ar = x.AR.to_numpy()
    roll = pd.Series(ar).rolling(19).mean()
    return dict(days=int(len(x)), mean=float(ar.mean()), nw_t=float(nw_t(ar)), sd=float(ar.std(ddof=1)),
                pos=float((ar > 0).mean()), ic=float(x.ic.mean()), worst19=float(roll.min()) if len(ar) >= 19 else np.nan,
                ones=float(x.ones.mean()))


def by_fold(D, folds):
    return {FOLD_NAME[a]: float(D[in_per(D.index, a, b)].AR.mean()) for a, b in folds}


def stationary_boot_idx(n, B, mean_block, rng):
    """indices of B stationary-bootstrap resamples (Politis & Romano 1994) of length n"""
    p = 1.0 / mean_block
    out = np.empty((B, n), np.int64)
    for b in range(B):
        i = rng.integers(n)
        for t in range(n):
            if t > 0 and rng.random() < p:
                i = rng.integers(n)
            elif t > 0:
                i = (i + 1) % n
            out[b, t] = i
    return out

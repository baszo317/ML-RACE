r"""
v10lib.py -- shared code of the v10 research (Alpha158 groups added to v9, research\PREREG_v10.md).

Panel   : v9's research panel (v9\cache\core9.pkl = v4/v5 panel + v3 features + night session + sigma columns) with the 158
          Alpha158 columns of v10\alpha158.py merged on (date, stock_id) -> v10\cache\panel10.pkl
Members : the stock-view members of v9 that the screen changes
          L  lgb_res      LightGBM on B (67), v8 g1_tab_train.py / rolling9.train_lgb (seed 1..5)
          M  mlp_res_xr9, mlp_res_BN9, mlp_res_xrN9: v9_train.train_custom (sigma weights alpha 0.5), 100-stock panel
          a config adds the columns of one Alpha158 group to the member's features (appended at the end)
Score   : v9 (research\v9_eval.py): p_raw = (mean of the 3 MLP raw members + Transformer) / 2, pbar = day mean;
          p_e = (mean of the 4 MLP resid members + LightGBM) / 2, centred per day; score = 0.5 (pbar - 0.5) + p_e_c;
          0 on the ex-dividend day, the day before and the 2 days after; beta-proportional hedge lambda 0.75
Folds   : yearly walk-forward 2022..2026-05 + post (trained to 2026-05-31), as v9.
"""
import os, sys, json, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); V10 = os.path.dirname(HERE); ROOT = os.path.dirname(V10)
for p in (os.path.join(ROOT, 'v9', 'research'), os.path.join(ROOT, 'v8', 'research'), os.path.join(ROOT, 'v8'), os.path.join(ROOT, 'v5'),
          os.path.join(ROOT, 'transformer'), os.path.join(ROOT, 'v9'), V10):
    if p not in sys.path:
        sys.path.insert(0, p)
import alpha158 as A

CACHE = os.path.join(V10, 'cache'); RESULTS = os.path.join(V10, 'results')
V9CACHE = os.path.join(ROOT, 'v9', 'cache'); V8CACHE = os.path.join(ROOT, 'v8', 'cache')
DATA = os.path.join(ROOT, 'data')
FOLDS = [('2022-01-01', '2023-01-01'), ('2023-01-01', '2024-01-01'), ('2024-01-01', '2025-01-01'), ('2025-01-01', '2026-01-01'),
         ('2026-01-01', '2026-06-01'), ('2026-06-01', '2100-01-01')]
NAS = ['us_ixic_ret1', 'us_ixic_ret5', 'us_ixic_ret1_lag1']
SIG = ['absr_60', 'absres_60']
LGB_PRM = dict(objective='binary', learning_rate=0.02, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8,
               bagging_freq=1, lambda_l2=1.0, num_threads=1, verbose=-1)          # = v8 g1_tab_train.LGB = rolling9.LGB_PRM
TRAIN_START = pd.Timestamp('2020-03-02')                                         # = ens_lib.TRAIN_START
MLP_MEMBERS = ['mlp_res_xr9', 'mlp_res_BN9', 'mlp_res_xrN9']
GROUPS = A.groups()                                                               # 23 groups -> columns
CONFIGS = ['base'] + list(GROUPS) + ['ALL']

def base_features():
    """the v9 feature lists of the members the screen changes (identical to v9lib.members / rolling9.feature_sets)"""
    sys.path.insert(0, os.path.join(ROOT, 'v9'))
    import features_extra as fe
    from features_v3 import NEW, XRN
    B = json.load(open(os.path.join(ROOT, 'transformer', 'final_feature_lists.json')))['B_competition_only']
    B9 = [f for f in B if f not in NAS]; XR = list(fe.XR)
    return {'lgb_res': list(B), 'mlp_res_xr9': B9 + XR + SIG, 'mlp_res_BN9': B9 + list(NEW) + SIG,
            'mlp_res_xrN9': B9 + XR + list(NEW) + list(XRN) + SIG}

def config_features(member, config, extra=None):
    """features of `member` under `config` ('base', a group name, 'ALL', or 'v10' with extra = list of groups)"""
    f = base_features()[member]
    if config == 'base':
        add = []
    elif config == 'ALL':
        add = list(A.COLS)
    elif config == 'v10':
        add = [c for g in (extra or []) for c in GROUPS[g]]
    else:
        add = list(GROUPS[config])
    return f + [c for c in add if c not in f]

# ------------------------------------------------------------------ panel
NEEDED = ['date', 'stock_id', 'top50', 'target_r', 'y', 'resid_r', 'y_resid', 'ex_today', 'ex_next1', 'ex_prev1', 'ex_prev2']

def panel(rebuild=False, log=print):
    """v9 research panel + Alpha158 (cached as v10\\cache\\panel10.pkl; only the columns the screen uses)"""
    f = os.path.join(CACHE, 'panel10.pkl')
    if os.path.exists(f) and not rebuild:
        return pd.read_pickle(f)
    try:
        P = pd.read_pickle(os.path.join(V9CACHE, 'core9.pkl'))
    except Exception as e:                       # e.g. a pickle written by another pandas version
        log(f'cannot read v9\\cache\\core9.pkl ({e!r}); building the same panel with v9\\rolling9.build_core (slower)')
        import rolling9
        P = rolling9.build_core(DATA)
        P = P[P.date <= pd.Timestamp('2026-10-23')]
    P['date'] = pd.to_datetime(P.date).astype('datetime64[ns]'); P['stock_id'] = P.stock_id.astype(str)
    feats = sorted(set(sum(base_features().values(), [])))
    P = P[NEEDED + [c for c in feats if c not in NEEDED]].copy()
    X = A.build_alpha158(DATA, sids=sorted(P.stock_id.unique()))
    n0 = len(P)
    P = P.merge(X, on=['date', 'stock_id'], how='left')
    assert len(P) == n0
    P = P.sort_values(['date', 'stock_id']).reset_index(drop=True)
    miss = P.loc[P.target_r.notna(), A.COLS].isna().all(axis=1)
    log(f'panel10: {P.shape}, labelled rows without Alpha158 {int(miss.sum())}; VWAP factor restarts {sum(X.attrs.get("vwap_restarts", {}).values())}')
    os.makedirs(CACHE, exist_ok=True); P.to_pickle(f + '.tmp'); os.replace(f + '.tmp', f)
    return P

# ------------------------------------------------------------------ training units
def unit_path(family, config, member, fold_a, seed, root=None):
    """file of one unit: .npz (numpy only, readable by any pandas version); .pkl from the first cloud runs is also read"""
    root = root or os.path.join(CACHE, 'screen')
    return os.path.join(root, family, config, member, f'{fold_a}_seed{seed}.npz')

def unit_exists(f):
    return os.path.exists(f) or os.path.exists(f[:-4] + '.pkl')

def save_unit(f, o):
    os.makedirs(os.path.dirname(f), exist_ok=True)
    tmp = f[:-4] + '.tmp'
    with open(tmp, 'wb') as fh:
        np.savez_compressed(fh, date=pd.to_datetime(o.date).values.astype('datetime64[ns]').astype(np.int64),
                            sid=o.stock_id.astype(str).values.astype('U8'), p=o.p.values.astype(np.float64))
    os.replace(tmp, f)

def load_unit(f):
    if os.path.exists(f):
        z = np.load(f)
        idx = pd.MultiIndex.from_arrays([pd.to_datetime(z['date']).astype('datetime64[ns]'), z['sid'].astype(str)], names=['date', 'stock_id'])
        return pd.Series(z['p'], index=idx, name='p')
    x = pd.read_pickle(f[:-4] + '.pkl'); x['date'] = pd.to_datetime(x.date).astype('datetime64[ns]'); x['stock_id'] = x.stock_id.astype(str)
    return x.set_index(['date', 'stock_id']).p

def train_lgb_unit(P, feats, fold_a, fold_b, seed):
    """v8 g1_tab_train.run_unit (model lgb, target resid) with the given features -> DataFrame date, stock_id, p"""
    import lightgbm as lgb
    te = pd.Timestamp(fold_a) - pd.Timedelta(days=1)
    Q = P[(P.date >= TRAIN_START) & (P.date <= te) & P.resid_r.notna()]
    X = Q[feats].to_numpy(np.float64); y = Q.y_resid.values.astype(float)
    w = np.minimum(np.abs(Q.resid_r.values), 0.05); w = w / w.mean()
    ud = np.sort(Q.date.unique()); cut = ud[-120] if len(ud) > 240 else ud[-1]; tr = Q.date.values < cut
    R = P[(P.date >= pd.Timestamp(fold_a)) & (P.date < pd.Timestamp(fold_b)) & P.resid_r.notna()]
    prm = {**LGB_PRM, 'seed': int(seed)}
    dtr = lgb.Dataset(X[tr], y[tr], weight=w[tr]); dva = lgb.Dataset(X[~tr], y[~tr], weight=w[~tr], reference=dtr)
    m = lgb.train(prm, dtr, 1000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    best = max(int(m.best_iteration), 1)
    m = lgb.train(prm, lgb.Dataset(X, y, weight=w), best)
    p = m.predict(R[feats].to_numpy(np.float64))
    return pd.DataFrame({'date': R.date.values, 'stock_id': R.stock_id.values, 'p': p}), best

def train_mlp_unit(P, feats, fold_a, fold_b, seed):
    """v9_train.run_unit for a core-panel resid member with the given features -> DataFrame date, stock_id, p"""
    import torch, tfmodel as TM
    import v9_train as T9
    torch.set_num_threads(1)
    te = pd.Timestamp(fold_a) - pd.Timedelta(days=1)
    cols = ['date', 'stock_id', 'resid_r', 'y_resid', 'absres_60'] + [c for c in feats if c not in ('absres_60',)]
    Q = P.loc[P.date <= max(pd.Timestamp(fold_b), te), cols].copy()
    Q['target_r'] = Q.resid_r; Q['y'] = Q.y_resid
    bd = T9.train_custom(Q[Q.date <= te], feats, TRAIN_START, te, seed, 'absres_60')
    a_, b_ = pd.Timestamp(fold_a), pd.Timestamp(fold_b)
    o = TM.predict(bd, Q, lambda d: (d.date >= a_) & (d.date < b_) & d.target_r.notna())[['date', 'stock_id', 'p']]
    return o, bd['best_epoch']

# ------------------------------------------------------------------ evaluation (= research\v9_eval.py)
def read_csv(f):
    x = pd.read_csv(f, dtype={'stock_id': str}, parse_dates=['date']); x['date'] = x.date.astype('datetime64[ns]')
    return x.set_index(['date', 'stock_id']).sort_index()

def load_v9():
    """cached walk-forward predictions of v9's members: W9 (7 MLP members, seed 1..10), tf_raw_B, lgb_resid"""
    W9 = read_csv(os.path.join(V9CACHE, 'wf9_preds.csv.gz'))
    M = read_csv(os.path.join(V8CACHE, 'wf_members.csv.gz'))
    LGB = read_csv(os.path.join(V8CACHE, 'tab_preds.csv.gz')).lgb_resid
    idx = M.index.intersection(W9.dropna().index).intersection(LGB.index)
    return W9.loc[idx], M.loc[idx, 'tf_raw_B'], LGB.loc[idx], idx

CGRID = np.round(np.arange(-0.3, 0.3 + 1e-9, 0.0025), 6)
EX4 = ('ex_today', 'ex_next1', 'ex_prev1', 'ex_prev2')

def info_table(P, idx):
    """Top50 rows with the label, the ex-dividend flags and the 120-day beta of v8 (hedge_beta.betas reproduces
    v8\\cache\\beta_panel.pkl to 1e-14, so no pickle is needed)"""
    import hedge_beta as H
    B = H.betas(sorted(set(idx.get_level_values('date'))))
    B['date'] = pd.to_datetime(B.date).astype('datetime64[ns]'); B['stock_id'] = B.stock_id.astype(str)
    B = B.set_index(['date', 'stock_id']).beta
    Q = P.set_index(['date', 'stock_id'])
    INFO = pd.DataFrame({'top50': Q.top50, 'target_r': Q.target_r, **{c: Q[c] for c in EX4}}).reindex(idx)
    INFO['beta'] = B.reindex(idx).clip(0.2, 2.5)
    INFO['beta'] = INFO.beta.fillna(INFO.groupby(level='date').beta.transform('median')).fillna(1.0)
    return INFO[INFO.top50.astype(bool) & INFO.target_r.notna()]

def centre(p):
    return p - p.groupby(level='date').transform('mean')

def views(raw_parts, res_parts):
    p_raw = pd.concat([pd.concat(f, axis=1).mean(axis=1) for f in raw_parts], axis=1).mean(axis=1)
    p_e = pd.concat([pd.concat(f, axis=1).mean(axis=1) for f in res_parts], axis=1).mean(axis=1)
    return 0.5 * (p_raw.groupby(level='date').transform('mean') - 0.5) + centre(p_e)

def decide(INFO, score, ex_cols=EX4, lam=0.75):
    """v9_eval.decide: daily AR / timing / xsec (bp) on the Top50"""
    X = INFO.join(score.rename('score'), how='inner')
    rows = []
    for d, g in X.groupby(level='date'):
        sc, b, r = g.score.values, g.beta.values, g.target_r.values
        ex = np.zeros(len(g), bool)
        for c in ex_cols:
            ex |= g[c].values == 1
        s = np.where(ex, -1, np.where(sc > 0, 1, -1))
        if lam < 1.0:
            sb, bb = s.mean(), b.mean(); B0 = ((s - sb) * (b - bb)).mean(); Et = sb * bb + lam * B0
            S = (sc[None, :] + CGRID[:, None] * b[None, :] > 0) & ~ex[None, :]
            E = ((2 * S - 1) @ b) / len(b)
            dd = np.abs(E - Et); cand = np.flatnonzero(dd <= dd.min() + 1e-12); j = cand[np.argmin(np.abs(CGRID[cand]))]
            s = np.where(S[j], 1, -1)
        m = r.mean()
        rows.append((d, (s * r).mean() * 1e4, s.mean() * m * 1e4, m * 1e4))
    D = pd.DataFrame(rows, columns=['date', 'AR', 'timing', 'm']).set_index('date'); D['xsec'] = D.AR - D.timing
    return D

VAL0, CUT, POST = pd.Timestamp('2022-01-01'), pd.Timestamp('2025-01-01'), pd.Timestamp('2026-06-01')
def periods(idx):
    idx = pd.DatetimeIndex(idx)
    return {'val': (idx >= VAL0) & (idx < CUT), 'hold': (idx >= CUT) & (idx < POST), 'post': idx >= POST, 'all': idx >= VAL0}

def plain_t(d):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    return d.mean() / d.std(ddof=1) * np.sqrt(len(d)) if len(d) > 2 and d.std() > 0 else np.nan

def nw_t(d, lags=5):
    d = np.asarray(d, float); d = d[~np.isnan(d)]; n = len(d)
    if n < 3:
        return np.nan
    e = d - d.mean(); s = e @ e / n
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return d.mean() / np.sqrt(s / n) if s > 0 else np.nan

def load_units(family, config, members, seeds, root=None, need_all=True):
    """seed-mean prediction of each member of a config (Series indexed by date, stock_id); None if a unit is missing"""
    out = {}
    for m in members:
        parts = {}
        for fa, _ in FOLDS:
            fs = [unit_path(family, config, m, fa, s, root) for s in seeds]
            have = [f for f in fs if unit_exists(f)]
            if len(have) < len(fs):
                if need_all:
                    return None
                continue
            parts[fa] = pd.concat([load_unit(f) for f in have], axis=1).mean(axis=1)
        out[m] = pd.concat(parts.values()).sort_index()
    return out

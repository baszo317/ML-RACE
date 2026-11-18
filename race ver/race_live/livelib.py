r"""
livelib.py -- the race version (race ver\race_live\PREREG_race_live.md): Chen V5 features (C, 169) + LightGBM LU, a rolling pool
of the models of the last 3 labelled trading days, Top25 of Top50 + ex-dividend rule. Used by race_live.py (every day).

The frozen code of part 1 (race ver\race_std\research) is only imported. The crawled data used here is the live copy
race_live\ext (build_panel.EXT is pointed at it), so race_std\ext and its hashes never change.
"""
import os, sys, json, time, shutil, hashlib
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); RACEVER = os.path.dirname(HERE)
STD = os.path.join(RACEVER, 'race_std')
if os.path.join(STD, 'research') not in sys.path:
    sys.path.insert(0, os.path.join(STD, 'research'))
import rslib as L                                                    # noqa: E402  (frozen, read only)
import build_panel as B                                              # noqa: E402  (frozen, read only)

ROOT, DATA = L.ROOT, L.DATA
EXT = os.environ.get('RACE_LIVE_EXT', os.path.join(HERE, 'ext'))
B.EXT = EXT
CACHE, OUT, RESULTS = os.path.join(HERE, 'cache'), os.path.join(HERE, 'out'), os.path.join(HERE, 'results')
MODELS = os.environ.get('RACE_LIVE_MODELS', os.path.join(HERE, 'models'))
POOL, K_TOP, ES_DAYS = 3, 25, 120
EX4 = L.EX4
TRAIN_START = L.TRAIN_START
ADR_COLS_PREFIX = ('C__adr_',)
META_COLS = ['date', 'stock_id', 'top50', 'has_label', 'target_r', 'resid_r', 'y_resid', 'absres_60', 'group'] + list(EX4)


def r9():
    """the user's v9\\rolling9 (calendar, ex-dividend fix); same path handling as build_panel.user_panel"""
    v9 = os.path.join(ROOT, 'v9')
    if v9 not in sys.path:
        sys.path.insert(0, v9)
    import rolling9
    return rolling9


def threads():
    return int(os.environ.get('RACE_THREADS', '1'))


def lgb_params(seed):
    """= learners.fit_LU of part 1 (col-wise + deterministic: the result does not depend on the number of threads)"""
    return dict(objective='binary', learning_rate=0.02, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=threads(), verbose=-1, seed=int(seed),
                deterministic=True, force_col_wise=True)


def ensure_ext(log=print):
    """first run: copy part 1's crawled data (race_std\\ext) to race_live\\ext; afterwards only race_live\\ext is updated"""
    src = os.path.join(STD, 'ext')
    if not os.path.isdir(os.path.join(EXT, 'yahoo')):
        shutil.copytree(src, EXT, dirs_exist_ok=True)
        log(f'copied {src} -> {EXT}')


def c_features():
    """the 169 C columns in part 1's order (race_std\\cache\\features.json)"""
    return list(L.features()['C'])


def feat_hash(feats):
    return hashlib.sha256('\n'.join(feats).encode('utf-8')).hexdigest()[:16]


def seed_of(d):
    return int(pd.Timestamp(d).strftime('%Y%m%d'))


# ------------------------------------------------------------------ panel (C features + labels + rule columns), as part 1
def fingerprint(data_dir=DATA):
    """size + mtime of every input file (competition data, ex-dividend table, night session, race_live\\ext)"""
    h = hashlib.sha256()
    for top in (data_dir, EXT, os.path.join(ROOT, 'v7', 'data')):
        if not os.path.isdir(top):
            continue
        for dp, _, fs in os.walk(top):
            for f in sorted(fs):
                if f.endswith(('.csv', '.json', '.parquet')):
                    p = os.path.join(dp, f); st = os.stat(p)
                    h.update(f'{os.path.relpath(p, top)}|{st.st_size}|{int(st.st_mtime)}\n'.encode('utf-8'))
    return h.hexdigest()[:20]


def build_panel_C(data_dir=DATA, log=print, fix_next=True, use_cache=True):
    """the rows of build_panel.build_all / assemble restricted to the C columns; ex_next1 of the newest day from the calendar
    (= v9 live, rolling9.fix_ex_next). Cached in cache\\live_panel.pkl while no input file changes."""
    ensure_ext(log)
    fp = fingerprint(data_dir) + ('n' if fix_next else '')
    cf = os.path.join(CACHE, 'live_panel.pkl'); mf = os.path.join(CACHE, 'live_panel.json')
    if use_cache and os.path.exists(cf) and os.path.exists(mf):
        m = json.load(open(mf, encoding='utf-8'))
        if m.get('fingerprint') == fp and m.get('data_dir') == data_dir:
            log(f'panel: cached ({m["rows"]} rows, built {m["built"]})')
            return pd.read_pickle(cf), m['feats'], m.get('coverage', {})
    t0 = time.time()
    ids, top50 = B.read_info(data_dir)
    P, uorder, usub = B.user_panel(data_dir)
    P = P.copy()
    P['date'] = pd.to_datetime(P.date).astype('datetime64[ns]'); P['stock_id'] = P.stock_id.astype(str)
    if fix_next:
        P, nxt, ids_next = r9().fix_ex_next(P, data_dir)
        log(f'panel: ex_next1 of {P.date.max().date()} from the calendar: next trading day {nxt.date()}, ex-dividend {ids_next}')
    dates = pd.DatetimeIndex(sorted(P.date.unique())).astype('datetime64[ns]')
    long = B.read_market_long(data_dir, ids)
    gdf = B.read_global(data_dir)
    W = B.wide(long, dates, ids)
    G = gdf.set_index('date').reindex(dates).ffill()
    CAL = B.calendar_feats(dates, data_dir)
    NIGHT = B.tx_night(P).reindex(dates)
    FC, cday = B.feat_C(W, G, CAL, NIGHT, dates, ids, top50)
    grp = P.groupby('stock_id')['group'].last().to_dict()
    FK, kday, cov = B.feat_K(W, dates, ids, grp, None, log=log)
    FC = FC.join(FK, how='left')
    feats = list(FC.columns)
    ref = c_features()
    if feats != ref:
        raise RuntimeError(f'C columns differ from part 1 (features.json): {sorted(set(feats) ^ set(ref))[:10]}')
    meta = P[META_COLS].copy().reset_index(drop=True)
    idx = pd.MultiIndex.from_arrays([meta.date, meta.stock_id], names=['date', 'stock_id'])
    X = pd.DataFrame(FC.reindex(idx).to_numpy(np.float32), columns=feats)
    out = pd.concat([meta, X], axis=1).sort_values(['date', 'stock_id']).reset_index(drop=True)
    os.makedirs(CACHE, exist_ok=True)
    out.to_pickle(cf + '.tmp'); os.replace(cf + '.tmp', cf)
    with open(mf, 'w', encoding='utf-8') as fh:
        json.dump(dict(fingerprint=fp, data_dir=data_dir, rows=int(len(out)), feats=feats, coverage=cov,
                       built=time.strftime('%Y-%m-%d %H:%M:%S'), secs=round(time.time() - t0, 1)), fh)
    log(f'panel: built {len(out)} rows x {len(feats)} C columns in {time.time() - t0:.0f}s')
    return out, feats, cov


def labelled_days(P):
    """days on which the Top50 rows have the label"""
    m = P.top50.astype(bool) & P.target_r.notna() & P.resid_r.notna()
    return pd.DatetimeIndex(sorted(P.loc[m, 'date'].unique()))


def placeholder_days(P):
    """days after the last labelled day that have rows but no Top50 label (the -1 placeholder of the course data)"""
    lab = labelled_days(P)
    d = pd.DatetimeIndex(sorted(P.date.unique()))
    return d[d > lab.max()] if len(lab) else d


# ------------------------------------------------------------------ the model (= part 1 LU)
def train_model(P, feats, d_end, seed=None):
    """one LU model on the labelled rows TRAIN_START <= date <= d_end; the same computation as learners.fit_LU"""
    import lightgbm as lgb
    seed = seed_of(d_end) if seed is None else seed
    d = P['date'].to_numpy(); e = P['resid_r'].to_numpy(float)
    lab = P['target_r'].notna().to_numpy() & P['resid_r'].notna().to_numpy()
    tr = lab & (d >= np.datetime64(TRAIN_START)) & (d <= np.datetime64(pd.Timestamp(d_end)))
    y = (e > 0).astype(float)
    w = np.minimum(np.abs(e), 0.05); w = w / np.nanmean(w[tr])
    prm = lgb_params(seed)
    days = np.unique(d[tr])
    if not len(days) or pd.Timestamp(days[-1]) != pd.Timestamp(d_end):
        raise RuntimeError(f'no labelled rows on {pd.Timestamp(d_end).date()}')
    cut = days[-ES_DAYS] if len(days) > ES_DAYS else days[len(days) // 2]
    fit, val = tr & (d < cut), tr & (d >= cut)
    X = lambda m: P.loc[m, feats].to_numpy(np.float32)
    t0 = time.time()
    dtr = lgb.Dataset(X(fit), y[fit], weight=w[fit], free_raw_data=True)
    dva = lgb.Dataset(X(val), y[val], weight=w[val], reference=dtr)
    b = lgb.train(prm, dtr, num_boost_round=1000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    best = max(int(b.best_iteration or b.current_iteration()), 1)
    full = lgb.train(prm, lgb.Dataset(X(tr), y[tr], weight=w[tr]), num_boost_round=best)
    meta = dict(train_end=str(pd.Timestamp(days[-1]).date()), best_iter=best, n_train=int(tr.sum()), seed=int(seed),
                n_feat=len(feats), feat_hash=feat_hash(feats), secs=round(time.time() - t0, 1),
                written=time.strftime('%Y-%m-%d %H:%M:%S'))
    return full, meta


def model_path(d, seed, folder=None):
    return os.path.join(folder or MODELS, f'race_{pd.Timestamp(d).strftime("%Y%m%d")}_seed{int(seed)}')


def save_model(booster, meta, folder=None):
    f = model_path(meta['train_end'], meta['seed'], folder)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    booster.save_model(f + '.txt.tmp'); os.replace(f + '.txt.tmp', f + '.txt')
    with open(f + '.json.tmp', 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=1)
    os.replace(f + '.json.tmp', f + '.json')
    return f + '.txt'


def load_meta(d, seed, folder=None):
    f = model_path(d, seed, folder)
    if os.path.exists(f + '.txt') and os.path.exists(f + '.json'):
        m = json.load(open(f + '.json', encoding='utf-8')); m['file'] = f + '.txt'
        return m
    return None


def predict_pool(P, feats, t, metas):
    import lightgbm as lgb
    rows = P[P.date == pd.Timestamp(t)].copy()
    X = rows[feats].to_numpy(np.float32)
    ps = []
    for m in metas:
        b = lgb.Booster(model_file=m['file'])
        if b.num_feature() != len(feats) or m.get('feat_hash') != feat_hash(feats):
            raise RuntimeError(f'model {m["train_end"]} seed {m["seed"]}: feature columns differ')
        ps.append(b.predict(X, num_threads=threads()))
        rows[f'p_{m["train_end"]}_s{m["seed"]}'] = ps[-1]
    rows['p'] = np.mean(ps, axis=0)
    return rows


# ------------------------------------------------------------------ decision (= part 1 SD) and the upload file
def decide_top25(rows, k=K_TOP):
    """one day: Top50 by p (ties by stock id) -> the top k = 1; ex-dividend rows (today, next day, 1 or 2 days after) -> 0"""
    X = rows[rows.top50.astype(bool)].sort_values('stock_id').reset_index(drop=True)
    sc = X.p.to_numpy(float); sc = np.where(np.isnan(sc), -np.inf, sc)
    order = np.lexsort((X.stock_id.to_numpy(), -sc))                  # = rslib.decide_sd
    pred = np.zeros(len(X), int); pred[order[:min(k, len(X))]] = 1
    ex = np.zeros(len(X), bool)
    for c in EX4:
        ex |= X[c].to_numpy() == 1
    rank = np.empty(len(X), int); rank[order] = np.arange(1, len(X) + 1)
    X['rank'] = rank; X['top25'] = pred.copy(); X['ex_rule'] = ex.astype(int)
    pred[ex] = 0
    X['pred'] = pred
    return X.sort_values('rank').reset_index(drop=True)


def sample_ids():
    s = pd.read_csv(os.path.join(ROOT, 'sample_submission.csv'), dtype=str, encoding='utf-8-sig')
    return s.iloc[:, 0].astype(str).str.strip().tolist()


def write_submission(dec, t, folder=None, tag=''):
    ids = sample_ids()
    m = dec.set_index('stock_id').pred
    missing = [i for i in ids if i not in m.index]
    if missing:
        raise RuntimeError(f'no prediction for {missing}')
    col = pd.Timestamp(t).strftime('%Y-%m-%d')
    sub = pd.DataFrame({'id': ids, col: [int(m[i]) for i in ids]})
    folder = folder or OUT
    os.makedirs(folder, exist_ok=True)
    f = os.path.join(folder, f'submission_race_{pd.Timestamp(t).strftime("%m%d")}{tag}.csv')
    sub.to_csv(f + '.tmp', index=False, encoding='utf-8'); os.replace(f + '.tmp', f)    # = v9 files: no BOM, Windows line ends
    return f, sub

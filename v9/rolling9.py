r"""
rolling9.py -- v9 production code: members, panels, training, rolling model pool (static and live), decision rule.

v9 = every change with a positive val t-statistic, mixed into one model (README_v9.md, research\PREREG_v9.md):
  market view (label close > open)            : mlp_raw_B9, mlp_raw_xr9, mlp_raw_w4_9 (4-year window), tf_raw_B (unchanged)
  stock view  (label beat the 100-stock mean) : mlp_res_B9 (trained on the extended universe), mlp_res_xr9, mlp_res_BN9,
                                                mlp_res_xrN9, lgb_res (LightGBM)
  p_raw = (mean of the MLP raw members + Transformer) / 2,  pbar = mean of p_raw over the 100 stocks of the day
  p_e   = (mean of the MLP resid members + LightGBM) / 2,   p_e_c = p_e - mean of the day
  score = 0.5 * (pbar - 0.5) + p_e_c ;  1 if score > 0 ;  0 on the ex-dividend day, the day before and the two days after ;
  then the beta-proportional hedge with lambda = 0.75 (v8 hedge_beta.py: cancels a quarter of the low-beta tilt)
MLP training = research\v9_train.py (weights min(|r|, 5%) / sigma^0.5, sigma = trailing 60-day mean |return| known before
the day; early stopping on the last 120 training days, then refit); Transformer = v4/v5 unchanged (ens_lib.train_member);
LightGBM = v8 g1_tab_train.py (B features, early stopping 100 rounds on the last 120 days, then refit).

Rolling pool (as v4/v5, rolling.py): a model trained up to day t only reads rows dated <= t; day d is predicted with the
models of the last `pool` retrain points whose training ended on or before the previous trading day.
Model files (torch.save dicts) contain only tensors, lists, numbers and strings, so they load with any pandas / numpy
version; LightGBM models are stored as their text dump.
"""
import os, sys, json, time, glob, traceback
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
# search order (2026-10-25): v9 first (features_v3.py here is a copy of 棄案\v3\features_v3.py, byte-identical), then v8,
# transformer, v5 -- the daily run reads nothing in 棄案 any more
for _p in (os.path.join(ROOT, 'v5'), os.path.join(ROOT, 'transformer'), os.path.join(ROOT, 'v8'), HERE):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
import ens_lib as E          # v5 panel + Transformer training / bundles (unchanged)
import rolling as R          # v5 pool bookkeeping: retrain_points, seeds_at, pool_points, labelled_days, prev_day, file names

DATA = E.DATA
NAS = ['us_ixic_ret1', 'us_ixic_ret5', 'us_ixic_ret1_lag1']
NIGHT = ['night_ret', 'night_range']
SIG = ['absr_60', 'absres_60']
EX4 = ('ex_today', 'ex_next1', 'ex_prev1', 'ex_prev2')
W_T = 0.5
LAM = 0.75
ALPHA = 0.5
LGB_PRM = dict(objective='binary', learning_rate=0.02, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8,
               bagging_freq=1, lambda_l2=1.0, num_threads=1, verbose=-1)
SCHED = {'static': {'mlp': dict(every=1, new=2, init=10, pool=5), 'transformer': dict(every=5, new=1, init=5, pool=5),
                    'lgb': dict(every=1, new=1, init=5, pool=5)},
         'live':   {'mlp': dict(every=1, new=2, init=10, pool=5), 'transformer': dict(every=1, new=1, init=5, pool=5),
                    'lgb': dict(every=1, new=1, init=5, pool=5)}}
# extended-universe training panel: v9\data\panel_ext.parquet (copy of v7\cache\panel_ext.parquet, byte-identical, 2026-10-25);
# the old location is only a fallback
EXT_PARQUET = next((f for f in (os.path.join(HERE, 'data', 'panel_ext.parquet'), os.path.join(ROOT, 'v7', 'cache', 'panel_ext.parquet'))
                    if os.path.exists(f)), os.path.join(HERE, 'data', 'panel_ext.parquet'))
TX_CSV = os.path.join(ROOT, 'v7', 'data', 'tx_daily.csv')

# ------------------------------------------------------------------ members
def feature_sets():
    import features_extra as fe
    from features_v3 import NEW, XRN
    B = json.load(open(os.path.join(ROOT, 'transformer', 'final_feature_lists.json')))['B_competition_only']
    B9 = [f for f in B if f not in NAS]; XR = list(fe.XR)
    return {'B': list(B), 'B9+NIGHT+SIG': B9 + NIGHT + SIG, 'B9+XR+NIGHT+SIG': B9 + XR + NIGHT + SIG, 'B9+SIG': B9 + SIG,
            'B9+XR+SIG': B9 + XR + SIG, 'B9+NEW+SIG': B9 + list(NEW) + SIG, 'B9+XR+NEW+XRN+SIG': B9 + XR + list(NEW) + list(XRN) + SIG}

def mem(name, kind, feats, target, window=None, panel='core', train='sig05'):
    return dict(name=name, kind=kind, seq_len=20 if kind == 'transformer' else 1, window=window, feats=feats, target=target,
                panel=panel, train=train)

MEMBERS = [mem('mlp_raw_B9',   'mlp',         'B9+NIGHT+SIG',      'raw'),
           mem('mlp_raw_xr9',  'mlp',         'B9+XR+NIGHT+SIG',   'raw'),
           mem('mlp_raw_w4_9', 'mlp',         'B9+NIGHT+SIG',      'raw', window=4),
           mem('tf_raw_B',     'transformer', 'B',                 'raw', train='v4'),
           mem('mlp_res_B9',   'mlp',         'B9+SIG',            'resid', panel='ext'),
           mem('mlp_res_xr9',  'mlp',         'B9+XR+SIG',         'resid'),
           mem('mlp_res_BN9',  'mlp',         'B9+NEW+SIG',        'resid'),
           mem('mlp_res_xrN9', 'mlp',         'B9+XR+NEW+XRN+SIG', 'resid'),
           mem('lgb_res',      'lgb',         'B',                 'resid', train='lgb')]

def same_spec(saved, m):
    """v4/v5 keys for the Transformer (so v4/v5 Transformer files are accepted); v9 members also check panel and training"""
    saved = saved or {}
    if not all(saved.get(k) == m[k] for k in R.SPEC_KEYS):
        return False
    return m['train'] == 'v4' or (saved.get('panel') == m['panel'] and saved.get('train') == m['train'])

# ------------------------------------------------------------------ panels
def closed_days(data_dir=DATA):
    """days the market did not open although the calendar published in advance has them (2026-10-23 typhoon day): hand-made
    data\\ref\\closures_manual.csv, columns date,reason (2026-10-25). transformer\\update_data.py drops their rows from data\\,
    moves the day-to-predict placeholder to the next trading day and removes them from calendar.parquet"""
    f = os.path.join(data_dir, 'ref', 'closures_manual.csv')
    if not os.path.exists(f):
        return pd.DatetimeIndex([], dtype='datetime64[ns]')
    c = pd.read_csv(f, dtype=str, encoding='utf-8-sig', encoding_errors='replace')['date'].dropna().str.strip()
    return pd.DatetimeIndex(pd.to_datetime(c[c != ''].tolist())).astype('datetime64[ns]')

def next_trading_day(d, data_dir=DATA):
    """first trading day after d: data\\ref\\calendar.parquet without the closed days (Mon-Fri if the calendar is unreadable)"""
    closed = set(closed_days(data_dir).strftime('%Y-%m-%d')); d = pd.Timestamp(d)
    try:
        cal = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(os.path.join(data_dir, 'ref', 'calendar.parquet')).date)).sort_values()
        later = [x for x in cal[cal > d] if x.strftime('%Y-%m-%d') not in closed]
        if later:
            return pd.Timestamp(later[0])
    except Exception:
        pass
    x = d + pd.offsets.BDay(1)
    while x.strftime('%Y-%m-%d') in closed:
        x = x + pd.offsets.BDay(1)
    return x

def night_table(path=TX_CSV, data_dir=DATA):
    """TX night session (v7 e7): FinMind 'after_market' row dated D = the session 15:00 (D-1) .. 05:00 (D), front month =
    the contract with the largest night volume; night_ret = log(night close / previous day-session close of that contract).
    2026-10-25: a row dated on a closed day (closed_days) gets the next trading day. FinMind dates the session before an
    announced holiday with the trading day after it (the 10/08 15:00 .. 10/09 05:00 session is dated 10/12), but the typhoon
    closure of 10/23 was announced during the 10/22 session, which therefore kept the date 10/23 instead of 10/27"""
    if not os.path.exists(path):
        return pd.DataFrame({'date': pd.Series(dtype='datetime64[ns]'), 'prev_day': pd.Series(dtype='datetime64[ns]'),
                             'night_ret': pd.Series(dtype=float), 'night_range': pd.Series(dtype=float)})
    tx = pd.read_csv(path, dtype={'contract_date': str})
    tx['date'] = pd.to_datetime(tx.date).astype('datetime64[ns]')
    tx = tx[(tx.futures_id == 'TX') & tx.contract_date.str.fullmatch(r'\d{6}')]
    day = tx[tx.trading_session == 'position'][['date', 'contract_date', 'open', 'close']]
    ngt = tx[(tx.trading_session == 'after_market') & (tx.volume > 0)].copy()
    closed = closed_days(data_dir)
    ngt['moved'] = ngt.date.isin(closed)
    if ngt.moved.any():
        ngt.loc[ngt.moved, 'date'] = [next_trading_day(x, data_dir) for x in ngt.loc[ngt.moved, 'date']]
    # front month = largest night volume; a session FinMind itself dates on that day (a later one) wins over a moved one
    ngt = ngt.sort_values(['date', 'moved', 'volume'], ascending=[True, True, False]).drop_duplicates('date')[['date', 'contract_date', 'open', 'max', 'min', 'close']]
    dd = np.sort(day.date.unique()); i = np.searchsorted(dd, ngt.date.values) - 1
    ngt = ngt[i >= 0].copy(); ngt['prev_day'] = dd[i[i >= 0]]
    N = ngt.merge(day.rename(columns={'date': 'prev_day', 'close': 'prev_close', 'open': 'prev_open'}), on=['prev_day', 'contract_date'], how='left')
    N['night_ret'] = np.log(N.close / N.prev_close); N['night_range'] = (N['max'] - N['min']) / N.prev_close
    return N[['date', 'prev_day', 'night_ret', 'night_range']]

def add_sigma(P):
    P = P.sort_values(['stock_id', 'date']).copy()
    a = P.target_r.abs().clip(upper=0.05)
    P['absr_60'] = a.groupby(P.stock_id).transform(lambda s: s.rolling(60, min_periods=20).mean().shift(1))
    if 'absres_60' not in P:
        P['absres_60'] = P.resid_r.abs().groupby(P.stock_id).transform(lambda s: s.rolling(60, min_periods=20).mean().shift(1))
    return P.sort_values(['date', 'stock_id']).reset_index(drop=True)

def build_core(data_dir=DATA, pred_dates=()):
    """v4/v5 panel + the 5 v3 features (and ranks) + TX night session + sigma columns (= research\\v9lib.core_panel)"""
    from features_v3 import add_v3
    P = E.build_panel(data_dir, pred_dates)
    P['date'] = P['date'].astype('datetime64[ns]'); P['stock_id'] = P.stock_id.astype(str)
    P = add_v3(P, data_dir)
    N = night_table(data_dir=data_dir)
    # live guard: FinMind may publish the night session of day D before the day session of D-1; then the newest day
    # session it has is older than D-1 and night_ret would span two days -> missing instead (never happens in 2020-2026)
    dd = np.sort(P.date.unique()); prev = pd.Series(dd[:-1], index=dd[1:])
    bad = N.prev_day.values != N.date.map(prev).values
    N.loc[bad & N.date.isin(dd[1:]).values, ['night_ret', 'night_range']] = np.nan
    P = P.merge(N[['date', 'night_ret', 'night_range']], on='date', how='left')
    return add_sigma(P)

def build_ext(core, parquet=EXT_PARQUET):
    """training panel of mlp_res_B9: the 100 competition stocks (float32, as v7) + the extra point-in-time Top300 stocks
    of v9\\data\\panel_ext.parquet (copy of v7\\cache\\panel_ext.parquet; their rows end 2026-07; features and residual label
    use the 100 only)"""
    X = pd.read_parquet(parquet)
    X['date'] = pd.to_datetime(X.date).astype('datetime64[ns]'); X['stock_id'] = X.stock_id.astype(str)
    keep = [c for c in X.columns if c not in SIG]
    extra = X.loc[~X.core.astype(bool), keep]
    C = core[[c for c in keep if c != 'core']].copy(); C['core'] = True; C['top50'] = C.top50.astype(bool); C = C[keep]
    num = [c for c in keep if c not in ('date', 'stock_id', 'top50', 'core')]
    C[num] = C[num].astype('float32')
    extra = extra.assign(core=False, top50=False)
    return add_sigma(pd.concat([C, extra], ignore_index=True))

def fix_ex_next(P, data_dir=DATA):
    """ex_next1 of the newest panel day: the panel has no row for the following day, so look the next trading day up in
    the official calendar (data\\ref\\calendar.parquet; Mon-Fri if unreadable) and the ex-dividend table"""
    import features_extra as fe
    last = P.date.max()
    try:
        cal = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(os.path.join(data_dir, 'ref', 'calendar.parquet')).date)).sort_values()
        nxt = cal[cal > last][0]
    except Exception:
        nxt = last + pd.offsets.BDay(1)
    T = fe.exdiv_table(data_dir)
    ids = set(T.loc[T.date == pd.Timestamp(nxt), 'stock_id'])
    m = P.date == last
    P.loc[m, 'ex_next1'] = P.loc[m, 'stock_id'].isin(ids).astype(float).values
    return P, pd.Timestamp(nxt), sorted(ids)

# ------------------------------------------------------------------ training
def train_mlp_sig(P, m, te, seeds, feats):
    """research\\v9_train.train_custom for several seeds that share the data and the scaler (identical results)"""
    import torch, tfmodel as TM
    hp = {**dict(d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.2, lr=1e-3, wd=1e-2, bs=512, epochs=15, patience=3), **E.HP}
    te = pd.Timestamp(te)
    Q = E.target_panel(P[P.date <= te], m['target'])
    start = E.window_start(te, m['window'])
    D = TM.SeqData(Q, feats, 1)
    tm = ((D.P.date >= start) & (D.P.date <= te)).values & D.P.target_r.notna().values
    scaler = TM.Scaler().fit(D.raw[tm]); D.scale(scaler)
    idx_all = D.valid(tm); dates = D.P.date.values[idx_all]
    ud = np.sort(np.unique(dates)); cut = ud[-120] if len(ud) > 240 else ud[-1]
    y = D.P.y.values.astype(np.float32)
    w = np.minimum(np.abs(D.P.target_r.fillna(0).values), 0.05)
    sig = D.P['absr_60' if m['target'] == 'raw' else 'absres_60'].values.astype(float)
    med = np.nanmedian(sig[idx_all])
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, med)
    w = w / sig ** ALPHA
    w = (w / w[idx_all].mean()).astype(np.float32)
    tr, va = idx_all[dates < cut], idx_all[dates >= cut]
    out = {}
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        model = TM.make_model('mlp', len(feats), 1, dropout=hp['dropout'])
        model, best_ep = TM._fit(model, D, tr, y[tr], w[tr], hp['epochs'], hp['lr'], hp['wd'], hp['bs'], s, val=(va, y[va], w[va]), patience=hp['patience'])
        best_ep = max(best_ep, 1)
        torch.manual_seed(s)
        model = TM.make_model('mlp', len(feats), 1, dropout=hp['dropout'])
        model, _ = TM._fit(model, D, idx_all, y[idx_all], w[idx_all], best_ep, hp['lr'], hp['wd'], hp['bs'], s)
        out[s] = dict(model=model.eval(), scaler=scaler, kind='mlp', seq_len=1, features=list(feats), hp=hp, best_epoch=best_ep)
    return out

def train_lgb(P, m, te, seeds, feats):
    """v8 g1_tab_train.py: weights min(|r|, 5%), early stopping (100 rounds) on the last 120 training days, refit"""
    import lightgbm as lgb
    te = pd.Timestamp(te)
    tcol, ycol = ('target_r', 'y') if m['target'] == 'raw' else ('resid_r', 'y_resid')
    start = E.window_start(te, m['window'])
    Q = P[(P.date >= start) & (P.date <= te) & P[tcol].notna()]
    X = Q[feats].to_numpy(np.float64); y = Q[ycol].values.astype(float)
    w = np.minimum(np.abs(Q[tcol].values), 0.05); w = w / w.mean()
    ud = np.sort(Q.date.unique()); cut = ud[-120] if len(ud) > 240 else ud[-1]; tr = Q.date.values < cut
    out = {}
    for s in seeds:
        prm = {**LGB_PRM, 'seed': int(s)}
        dtr = lgb.Dataset(X[tr], y[tr], weight=w[tr]); dva = lgb.Dataset(X[~tr], y[~tr], weight=w[~tr], reference=dtr)
        mdl = lgb.train(prm, dtr, 1000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        best = max(int(mdl.best_iteration), 1)
        mdl = lgb.train(prm, lgb.Dataset(X, y, weight=w), best)
        out[s] = dict(model=mdl, kind='lgb', features=list(feats), best_epoch=best)
    return out

def train_seeds(P, m, te, seeds, fs):
    if m['kind'] == 'lgb':
        return train_lgb(P, m, te, seeds, fs[m['feats']])
    if m['kind'] == 'transformer':
        te = pd.Timestamp(te)
        return {s: E.train_member(P[P.date <= te], m, s, te, fs[m['feats']]) for s in seeds}
    return train_mlp_sig(P, m, te, seeds, fs[m['feats']])

# ------------------------------------------------------------------ model files
def save_bundle(b, m, seed, te, path):
    import torch
    d = dict(kind=b['kind'], features=list(b['features']), train_end=str(pd.Timestamp(te).date()), best_epoch=int(b['best_epoch']),
             member={k: v for k, v in m.items()}, seed=int(seed), fmt='v9')
    if b['kind'] == 'lgb':
        d['model_str'] = b['model'].model_to_string()
    else:
        d.update(state_dict=b['model'].state_dict(), seq_len=int(b['seq_len']), hp=dict(b['hp']),
                 scaler={k: np.asarray(v, dtype=float).tolist() for k, v in b['scaler'].state().items()})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(d, path + '.tmp'); os.replace(path + '.tmp', path)

def load_bundle(path):
    import torch
    try:
        ck = torch.load(path, weights_only=False)
    except TypeError:
        ck = torch.load(path)
    if ck.get('kind') == 'lgb':
        import lightgbm as lgb
        return dict(model=lgb.Booster(model_str=ck['model_str']), kind='lgb', features=ck['features'], member=ck.get('member'),
                    seed=ck.get('seed'), train_end=ck['train_end'], best_epoch=ck.get('best_epoch'))
    return E.load_bundle(path)

def load_checked(path, m):
    b = load_bundle(path)
    if not same_spec(b.get('member'), m):
        raise RuntimeError(f'{path} was trained with other settings than member {m["name"]}; move it away and rerun')
    return b

def predict(b, P, m, dates):
    """P(label = 1) of the 100 competition stocks on `dates` -> Series indexed by (date, stock_id)"""
    dates = pd.DatetimeIndex(sorted(set(pd.to_datetime(list(dates)))))
    if b['kind'] == 'lgb':
        Q = P[P.date.isin(dates)]
        if 'core' in Q:
            Q = Q[Q.core.astype(bool)]
        p = b['model'].predict(Q[b['features']].to_numpy(np.float64))
        return pd.Series(p, index=pd.MultiIndex.from_arrays([Q.date.values, Q.stock_id.values], names=['date', 'stock_id']), name='p')
    Q = P[(P.date >= dates.min() - pd.Timedelta(days=60)) & (P.date <= dates.max())]
    if m['panel'] == 'ext':
        return E.predict_rows(b, Q, lambda d: d.date.isin(dates) & d.core.astype(bool))
    return E.predict_rows(b, Q, lambda d: d.date.isin(dates))

def panel_of(m, core, ext):
    return ext if m['panel'] == 'ext' else core

# ------------------------------------------------------------------ static walk-forward
def static_tasks(core, members, dates, sched):
    """R.static_plan (v4/v5) grouped by (member, retrain point): one task trains / loads all seeds of that point"""
    units, use, prev = R.static_plan(core, members, dates, sched)
    tasks = {}
    for u in units:
        k = (u['m']['name'], u['te'])
        t = tasks.setdefault(k, dict(m=u['m'], te=u['te'], seeds=[], dates=u['dates'], first=u['first']))
        t['seeds'].append(u['seed'])
    return list(tasks.values()), units, use, prev

def do_task(t, core, ext, fs, root):
    """train the seeds of one retrain point whose model file is missing (saved models are only loaded) and write the
    predictions of every seed for the dates the point serves"""
    t0 = time.time(); m, te = t['m'], pd.Timestamp(t['te']); P = panel_of(m, core, ext)
    try:
        need = [s for s in t['seeds'] if not os.path.exists(R.pred_file(root, m, te, s))]
        bund, how = {}, 'loaded'
        for s in need:
            f = R.model_file(root, m, te, s)
            if os.path.exists(f):
                try:
                    bund[s] = load_checked(f, m)
                except Exception:
                    os.replace(f, f + '.bad')
        todo = [s for s in need if s not in bund]
        eps = []
        if todo:
            for s, b in train_seeds(P, m, te, todo, fs).items():
                save_bundle(b, m, s, te, R.model_file(root, m, te, s)); bund[s] = b; eps.append(b['best_epoch'])
            how = f'trained {len(todo)}'
        for s in need:
            p = predict(bund[s], P, m, t['dates']).rename('p').reset_index()
            p['date'] = pd.to_datetime(p.date).dt.strftime('%Y-%m-%d'); p['p'] = p.p.astype('float32')
            f = R.pred_file(root, m, te, s)
            p.to_csv(f + '.tmp', index=False, float_format='%.9g'); os.replace(f + '.tmp', f)
        return dict(name=m['name'], te=R.tag(te), n=len(need), how=how, sec=time.time() - t0, eps=eps, err=None)
    except Exception:
        return dict(name=m['name'], te=R.tag(te), n=0, how='FAILED', sec=time.time() - t0, eps=[], err=traceback.format_exc())

_G = {}
def _work(t):
    if _G.get('one_thread'):
        import torch
        torch.set_num_threads(1)
    return do_task(t, _G['core'], _G['ext'], _G['fs'], _G['root'])

def run_tasks(tasks, core, ext, fs, root, workers=1, log=print):
    todo = [t for t in tasks if any(not os.path.exists(R.pred_file(root, t['m'], t['te'], s)) for s in t['seeds'])]
    if not todo:
        return []
    cost = lambda t: (0 if t['m']['kind'] == 'transformer' and not os.path.exists(R.model_file(root, t['m'], t['te'], t['seeds'][0])) else
                      1 if t['m']['panel'] == 'ext' or t['first'] else 2)
    todo.sort(key=lambda t: (cost(t), t['te'], t['m']['name']))
    n, t0, res = len(todo), time.time(), []
    log(f'{n} retrain point(s) to train or load, {workers} worker(s)')
    import multiprocessing as mp
    par = workers > 1 and 'fork' in mp.get_all_start_methods()
    _G.update(core=core, ext=ext, fs=fs, root=root, one_thread=par)
    if par:
        pool = mp.get_context('fork').Pool(workers); it = pool.imap_unordered(_work, todo)
    else:
        if workers > 1:
            log('no fork on this system: running in one process')
        pool = None; it = map(_work, todo)
    for r in it:
        res.append(r)
        if r['err'] or len(res) % 25 == 0 or len(res) == n:
            log(f"[{len(res)}/{n}] {r['name']} {r['te']}: {r['how']} ({r['sec']:.0f}s); elapsed {(time.time() - t0) / 60:.1f} min")
        if r['err']:
            log(r['err'])
    if pool is not None:
        pool.close(); pool.join()
    return res

def static_probs(root, members, units, use, dates, frozen=False):
    return R.static_probs(root, members, units, use, dates, frozen=frozen)

# ------------------------------------------------------------------ live
def scan_pool(root, m):
    return R.scan_pool(root, m)

def live_update(core, ext, members, root, sched, fs, retrain='auto', log=print):
    """bring every member's pool up to date: `new` seeds trained up to the newest labelled day whenever the newest point
    is >= `every` trading days old (an empty pool starts with `init` seeds); unreadable model files are retrained"""
    lab = R.labelled_days(core); last = lab[-1]
    for m in members:
        sc = sched[m['kind']]; pts = scan_pool(root, m)
        if pts:
            newest = max(pts)
            age = int(((lab > newest) & (lab <= last)).sum())
            due = (retrain == 'now' and newest < last) or (retrain == 'auto' and age >= sc['every'])
            seeds = R.seeds_at(1, last, sc) if due else []
        else:
            if retrain == 'none':
                raise SystemExit(f"--retrain none but no models for {m['name']} in {root}")
            seeds = R.seeds_at(0, last, sc)
        todo = [s for s in seeds if not os.path.exists(R.model_file(root, m, last, s))]
        if todo:
            t0 = time.time()
            for s, b in train_seeds(panel_of(m, core, ext), m, last, todo, fs).items():
                save_bundle(b, m, s, last, R.model_file(root, m, last, s))
            log(f"trained {m['name']} seed(s) {todo} on data up to {last.date()} ({time.time() - t0:.0f}s)")
        elif not seeds:
            log(f"{m['name']}: no new model needed (newest trained up to {max(pts).date()})")

def live_probs(core, ext, members, root, sched, dates, fs=None, log=print):
    """member name -> probabilities for `dates` averaged over each date's pool (a model file that cannot be read is
    retrained in place); also returns a description of the pools"""
    lab = R.labelled_days(core); out, desc = {}, {}
    for m in members:
        sc = sched[m['kind']]; pts = scan_pool(root, m); parts = []; P = panel_of(m, core, ext)
        for d in dates:
            pv = R.prev_day(lab, d)
            pp = R.pool_points(pts, pv, sc['pool'])
            if not pp:
                raise SystemExit(f"no model of {m['name']} trained up to {pv.date()} or earlier in {root}")
            ps = []
            for t in pp:
                for s in pts[t]:
                    f = R.model_file(root, m, t, s)
                    try:
                        b = load_checked(f, m)
                    except Exception as e:
                        log(f'cannot read {f} ({e!r}); retraining it')
                        os.replace(f, f + '.bad')
                        b = train_seeds(P, m, t, [s], fs or feature_sets())[s]; save_bundle(b, m, s, t, f)
                    ps.append(predict(b, P, m, [d]))
            parts.append(pd.concat(ps, axis=1).mean(axis=1))
            desc[(m['name'], d)] = f"{len(ps)} models, trained up to {pp[0].date()}..{pp[-1].date()}"
        out[m['name']] = pd.concat(parts)
    return out, desc

# ------------------------------------------------------------------ ensemble and decision
def combine(probs, members, w_t=W_T):
    """market view: p_raw = mean(mean of MLP raw members, Transformer), pbar = mean over the 100 stocks of the day;
    stock view: p_e = mean(mean of MLP resid members, LightGBM), p_e_c = p_e - mean of the day;
    score = w_t * (pbar - 0.5) + p_e_c"""
    def fam(target):
        f = {}
        for m in members:
            if m['target'] == target and m['name'] in probs:
                f.setdefault(m['kind'], []).append(probs[m['name']])
        return {k: pd.concat(v, axis=1).mean(axis=1) for k, v in f.items()}
    raw, res = fam('raw'), fam('resid')
    out = pd.DataFrame({'p_raw': pd.concat(list(raw.values()), axis=1).mean(axis=1)})
    for k, v in raw.items():
        out[f'p_raw_{k}'] = v.reindex(out.index)
    out['pbar'] = out.groupby(level='date').p_raw.transform('mean')
    out['p_e'] = pd.concat(list(res.values()), axis=1).mean(axis=1).reindex(out.index)
    for k, v in res.items():
        out[f'p_e_{k}'] = v.reindex(out.index)
    out['p_e_c'] = out.p_e - out.groupby(level='date').p_e.transform('mean')
    out['score'] = w_t * (out.pbar - 0.5) + out.p_e_c
    return out

def decide(ens, panel, stocks, lam=LAM, ex_cols=EX4):
    """ens indexed by (date, stock_id) -> long table of the `stocks` (Top50) with the ex-dividend flags, beta, the
    prediction before the hedge (pred_nohedge) and the final prediction (pred); also a per-day hedge log"""
    import hedge_beta as H
    dates = sorted(ens.index.get_level_values('date').unique())
    flags = panel.set_index(['date', 'stock_id'])[list(ex_cols)]
    X = ens.join(flags, how='left').reset_index()
    X = X[X.stock_id.isin(stocks)].copy()
    for c in ex_cols:
        X[c] = X[c].fillna(0.0)
    X['ex_any'] = (X[list(ex_cols)] == 1).any(axis=1)
    B = H.betas(dates)
    B['date'] = pd.to_datetime(B.date).astype('datetime64[ns]'); B['stock_id'] = B.stock_id.astype(str)
    X = X.merge(B, on=['date', 'stock_id'], how='left')
    X['beta'] = X.beta.clip(0.2, 2.5); X['beta'] = X.beta.fillna(X.groupby('date').beta.transform('median')).fillna(1.0)
    X['pred_nohedge'] = ((X.score > 0) & ~X.ex_any).astype(int)
    X['pred'] = X.pred_nohedge
    logs = []
    for d, g in X.groupby('date'):
        if lam >= 1.0:
            continue
        pred, info = H.hedge_day(g.score.values, g.beta.values, g.ex_any.values, lam)
        X.loc[g.index, 'pred'] = pred
        logs.append({'date': pd.Timestamp(d).date(), **info})
    return X.sort_values(['date', 'stock_id']).reset_index(drop=True), pd.DataFrame(logs)

def daily_ar(X, panel, col='pred'):
    """bp/day on the Top50 for the dates with a known result: mean of (2 pred - 1) * (close - open) / open"""
    Y = X.merge(panel[['date', 'stock_id', 'target_r']], on=['date', 'stock_id'], how='left')
    Y = Y[Y.target_r.notna()]
    return ((2 * Y[col] - 1) * Y.target_r).groupby(Y.date).mean() * 1e4

r"""
walkforward.py -- PREREG_race_live.md section 2 (WF-daily).

One LU model (part 1 learners.fit_LU, unchanged) per trading day d on the 169 C features, trained on the labelled rows
TRAIN_START <= date <= d with seed = YYYYMMDD of d, predicting the next 3 trading days. The score of day t is the mean of the
models of the 3 trading days before t; decision SD (Top25 of Top50 + ex-dividend rule). D only (2022-2024); V stays sealed:
no training row and no predicted row is dated 2025 or later.

  python walkforward.py --prep                     compact copy of the frozen panel (C columns + labels) in the VM home
  python walkforward.py --run [--budget 160] [--workers 2] [--years 2024,2023,2022]
  python walkforward.py --status
  python walkforward.py --eval                     results\wf_daily.txt / .json / wf_daily_daily.csv
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); RACEVER = os.path.dirname(HERE)
STD = os.path.join(RACEVER, 'race_std')
sys.path.insert(0, os.path.join(STD, 'research'))
os.environ.setdefault('RACE_THREADS', '1')
import rslib as L                                                    # noqa: E402  (frozen, read only)
import learners                                                      # noqa: E402  (frozen, read only)

LOCAL = os.environ.get('WF_LOCAL', os.path.join(os.path.expanduser('~'), 'wf_cache'))
UNITS = os.path.join(HERE, 'cache', 'wf_units')
RES = os.path.join(HERE, 'results')
K = 3
V_START = np.datetime64('2025-01-01')
YEARS = ('2024', '2023', '2022')
LOG = L.log_to(os.path.join(RES, 'walkforward.log'))

_P = _F = _DAYS = None


# ------------------------------------------------------------------ data
def prep():
    f = L.PANEL
    loc = os.path.join(os.path.expanduser('~'), 'panel_local.pkl')
    if os.path.exists(loc) and os.path.getsize(loc) == os.path.getsize(L.PANEL):
        f = loc
    P = pd.read_pickle(f)
    feats = L.feature_list('C')
    os.makedirs(LOCAL, exist_ok=True)
    np.save(os.path.join(LOCAL, 'X.npy'), P[feats].to_numpy(np.float32))
    np.savez(os.path.join(LOCAL, 'meta.npz'),
             date=pd.to_datetime(P.date).values.astype('datetime64[ns]').astype(np.int64),
             sid=P.stock_id.astype(str).to_numpy().astype('<U8'),
             resid_r=P.resid_r.to_numpy(float), target_r=P.target_r.to_numpy(float))
    info = dict(feats=feats, n=int(len(P)), panel=f, panel_bytes=os.path.getsize(f), written=time.strftime('%Y-%m-%d %H:%M:%S'))
    with open(os.path.join(LOCAL, 'info.json'), 'w', encoding='utf-8') as fh:
        json.dump(info, fh)
    LOG(f'prep: {len(P)} rows x {len(feats)} C columns from {f} -> {LOCAL}')


def load():
    global _P, _F, _DAYS
    if _P is None:
        info = json.load(open(os.path.join(LOCAL, 'info.json'), encoding='utf-8'))
        X = np.load(os.path.join(LOCAL, 'X.npy'))
        z = np.load(os.path.join(LOCAL, 'meta.npz'))
        P = pd.DataFrame(X, columns=info['feats'], copy=False)
        P['date'] = pd.to_datetime(z['date']).values.astype('datetime64[ns]')
        P['stock_id'] = z['sid'].astype(str)
        P['resid_r'] = z['resid_r']; P['target_r'] = z['target_r']
        _P, _F = P, list(info['feats'])
        _DAYS = np.unique(P['date'].to_numpy())
    return _P, _F, _DAYS


def unit_path(d):
    return os.path.join(UNITS, pd.Timestamp(d).strftime('%Y%m%d') + '.npz')


def task_days(days, years=YEARS):
    """training days whose models are needed to score the prediction days of each year, in the given year order"""
    out, seen = [], set()
    for y in years:
        a, b = np.datetime64(f'{y}-01-01'), np.datetime64(f'{int(y) + 1}-01-01')
        idx = np.where((days >= a) & (days < b))[0]
        for j in range(idx[0] - K, idx[-1]):
            if j not in seen:
                seen.add(j); out.append(j)
    return out


def run_one(j, nthreads=1):
    learners.THREADS = int(nthreads)                               # col-wise + deterministic: same model for any thread count
    P, feats, days = load()
    d = days[j]
    f = unit_path(d)
    if os.path.exists(f):
        return j, 0.0
    t0 = time.time()
    dd = P['date'].to_numpy()
    lab = P['target_r'].notna().to_numpy() & P['resid_r'].notna().to_numpy()
    tr = lab & (dd >= np.datetime64(L.TRAIN_START)) & (dd <= d)
    te = (dd > d) & (dd <= days[min(j + K, len(days) - 1)]) & (dd < V_START)
    assert dd[tr].max() <= d and (not te.any() or dd[te].min() > d) and dd[tr].max() < V_START
    seed = int(pd.Timestamp(d).strftime('%Y%m%d'))
    pred, meta = learners.fit_LU(P, feats, tr, te, seed)
    meta.update(train_end=str(pd.Timestamp(d).date()), seed=seed, n_train=int(tr.sum()), secs=round(time.time() - t0, 2),
                threads=int(nthreads))
    L.save_unit(f, dd[te], P['stock_id'].to_numpy()[te], pred, meta)
    return j, time.time() - t0


PH1 = os.path.join(HERE, 'cache', 'wf_phase1')


def _lu_parts(j):
    """the data of learners.fit_LU for the model of training day j (same masks as run_one)"""
    P, feats, days = load()
    d = days[j]
    dd = P['date'].to_numpy()
    lab = P['target_r'].notna().to_numpy() & P['resid_r'].notna().to_numpy()
    tr = lab & (dd >= np.datetime64(L.TRAIN_START)) & (dd <= d)
    te = (dd > d) & (dd <= days[min(j + K, len(days) - 1)]) & (dd < V_START)
    assert dd[tr].max() <= d and (not te.any() or dd[te].min() > d) and dd[tr].max() < V_START
    seed = int(pd.Timestamp(d).strftime('%Y%m%d'))
    e = P['resid_r'].to_numpy(float)
    y = (e > 0).astype(float)
    w = np.minimum(np.abs(e), 0.05)
    w = w / np.nanmean(w[tr])
    return P, feats, d, dd, tr, te, seed, y, w


def _lu_prm(seed, nthreads):
    """= the parameters of learners.fit_LU"""
    return dict(objective='binary', learning_rate=0.02, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=int(nthreads), verbose=-1, seed=seed,
                deterministic=True, force_col_wise=True)


def ph1_path(d):
    return os.path.join(PH1, pd.Timestamp(d).strftime('%Y%m%d') + '.json')


def advance(j, nthreads=1, out_dir=None):
    """one half of learners.fit_LU for training day j: phase 1 = early stopping (-> best_iter, saved), phase 2 = refit on all
    training rows with best_iter rounds, predict, save the unit. Returns (j, phase, seconds)."""
    import lightgbm as lgb
    t0 = time.time()
    P, feats, d, dd, tr, te, seed, y, w = _lu_parts(j)
    f = unit_path(d) if out_dir is None else os.path.join(out_dir, pd.Timestamp(d).strftime('%Y%m%d') + '.npz')
    if os.path.exists(f):
        return j, 0, 0.0
    prm = _lu_prm(seed, nthreads)
    p1 = ph1_path(d) if out_dir is None else os.path.join(out_dir, pd.Timestamp(d).strftime('%Y%m%d') + '.json')
    if not os.path.exists(p1):
        fit, val = learners._split_last_days(dd, tr, learners.ES_DAYS)
        dtr = lgb.Dataset(learners._X(P, feats, fit), y[fit], weight=w[fit], free_raw_data=True)
        dva = lgb.Dataset(learners._X(P, feats, val), y[val], weight=w[val], reference=dtr)
        b = lgb.train(prm, dtr, num_boost_round=1000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        best = max(int(b.best_iteration or b.current_iteration()), 1)
        os.makedirs(os.path.dirname(p1), exist_ok=True)
        with open(p1 + '.tmp', 'w', encoding='utf-8') as fh:
            json.dump(dict(best_iter=best, secs1=round(time.time() - t0, 2), threads1=int(nthreads)), fh)
        os.replace(p1 + '.tmp', p1)
        return j, 1, time.time() - t0
    m1 = json.load(open(p1, encoding='utf-8'))
    full = lgb.train(prm, lgb.Dataset(learners._X(P, feats, tr), y[tr], weight=w[tr]), num_boost_round=int(m1['best_iter']))
    pred = full.predict(learners._X(P, feats, te))
    meta = dict(best_iter=int(m1['best_iter']), train_end=str(pd.Timestamp(d).date()), seed=seed, n_train=int(tr.sum()),
                secs=round(float(m1['secs1']) + time.time() - t0, 2), threads=int(nthreads), split=True)
    L.save_unit(f, dd[te], P['stock_id'].to_numpy()[te], pred, meta)
    return j, 2, time.time() - t0


def run2(budget, workers, years, order='forward', until=''):
    """scheduling only: like run, but each task is one half of a fit (see advance), so less time is left unused per call"""
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    import multiprocessing as mp
    t0 = time.time()
    P, feats, days = load()
    os.makedirs(UNITS, exist_ok=True)
    todo = [j for j in task_days(days, years) if not os.path.exists(unit_path(days[j]))]
    hist = []
    hf = os.path.join(PH1, '_halves.json')
    try:
        hist = json.load(open(hf, encoding='utf-8'))[-16:]
    except Exception:
        hist = [x[0] / 2 for x in recent_secs(8) if x[1] == 1]
    est = lambda: (float(np.median(hist[-8:])) * 1.25) if hist else 45.0
    if until:                                                       # local clock time HH:MM (Windows run: stop by then)
        hh, mm = (int(x) for x in until.split(':'))
        now = pd.Timestamp.now()
        stop = now.normalize() + pd.Timedelta(hours=hh, minutes=mm)
        budget = max(0.0, (stop - now).total_seconds())
    LOG(f'run2 ({sys.platform}): {len(todo)} models to finish, budget {budget:.0f}s, workers {workers}, estimate {est():.0f}s per half '
        f'(load {time.time() - t0:.1f}s)')
    if order == 'reverse':
        todo = todo[::-1]
    queue, running, done_n = list(todo), {}, 0
    spawn = os.environ.get('WF_SPAWN') == '1' or 'fork' not in mp.get_all_start_methods()
    ctx = mp.get_context('spawn' if spawn else 'fork')
    with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
        while True:
            while len(running) < workers and queue and budget - (time.time() - t0) >= est():
                j = queue.pop(0)
                running[ex.submit(advance, j, 1)] = j
            if not running:
                break
            fin, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fu in fin:
                j = running.pop(fu)
                _, ph, secs = fu.result()
                if ph == 1:
                    hist.append(secs); queue.insert(0, j)
                elif ph == 2:
                    hist.append(secs); done_n += 1
                    LOG(f'  {pd.Timestamp(days[j]).date()} done (second half {secs:.1f}s)')
    try:
        os.makedirs(PH1, exist_ok=True)
        with open(hf, 'w', encoding='utf-8') as fh:
            json.dump(hist[-32:], fh)
    except OSError:
        pass
    left = len([j for j in task_days(days, years) if not os.path.exists(unit_path(days[j]))])
    LOG(f'run2: {done_n} finished in {time.time() - t0:.0f}s; {left} left')


def check_split(day):
    """advance() twice into a scratch folder for an already finished day; the predictions must equal the unit (bitwise)"""
    P, feats, days = load()
    j = int(np.searchsorted(days, np.datetime64(pd.Timestamp(day), 'ns')))
    tmp = os.path.join(os.path.expanduser('~'), 'wf_splitcheck')
    os.makedirs(tmp, exist_ok=True)
    for x in os.listdir(tmp):
        os.remove(os.path.join(tmp, x))
    advance(j, 2, tmp); advance(j, 2, tmp)
    a, ma = L.load_npz(unit_path(days[j])); b, mb = L.load_npz(os.path.join(tmp, pd.Timestamp(days[j]).strftime('%Y%m%d') + '.npz'))
    diff = float(np.max(np.abs(a.to_numpy() - b.reindex(a.index).to_numpy())))
    LOG(f'check_split {pd.Timestamp(days[j]).date()}: best_iter {ma["best_iter"]} vs {mb["best_iter"]}; max |diff| {diff:.3g} -> '
        + ('IDENTICAL' if diff == 0 else 'DIFFERENT'))


def recent_secs(n=8):
    """(secs, threads) of the newest finished models"""
    if not os.path.isdir(UNITS):
        return []
    fs = sorted((x for x in os.listdir(UNITS) if x.endswith('.npz') and '.tmp' not in x),
                key=lambda x: os.path.getmtime(os.path.join(UNITS, x)))[-n:]
    out = []
    for x in fs:
        try:
            m = L.load_npz(os.path.join(UNITS, x))[1]
            out.append((float(m.get('secs', 0)), int(m.get('threads', 2))))
        except Exception:
            pass
    return out


def run(budget, workers, years):
    """scheduling only: two 1-thread fits at a time while one fits in the budget, then 2-thread fits in what is left"""
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    import multiprocessing as mp
    t0 = time.time()
    P, feats, days = load()
    os.makedirs(UNITS, exist_ok=True)
    todo = [j for j in task_days(days, years) if not os.path.exists(unit_path(days[j]))]
    hist = {1: [], 2: []}
    for sec, th in recent_secs(16):
        if sec > 0:
            hist[1 if th == 1 else 2].append(sec)

    def est(th):
        h = hist[th][-8:]
        if h:
            return float(np.median(h)) * 1.2
        o = hist[3 - th][-8:]
        return (float(np.median(o)) * (1.6 if th == 1 else 1 / 1.6) * 1.2) if o else 60.0

    LOG(f'run: {len(todo)} models to fit, budget {budget}s, workers {workers}, estimate {est(1):.0f}s (1 thread) / '
        f'{est(2):.0f}s (2 threads) per model (load {time.time() - t0:.1f}s)')
    done_n, it = 0, iter(todo)
    with ProcessPoolExecutor(max(workers, 1), mp_context=mp.get_context('fork')) as ex:
        running = {}
        while True:
            left = budget - (time.time() - t0)
            if workers >= 2:
                while len(running) < workers and left >= est(1):
                    j = next(it, None)
                    if j is None:
                        break
                    running[ex.submit(run_one, j, 1)] = j
            if not running and left >= est(2):
                j = next(it, None)
                if j is not None:
                    running[ex.submit(run_one, j, 2)] = j
            if not running:
                break
            fin, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fu in fin:
                j = running.pop(fu)
                _, secs = fu.result()
                if secs > 0:
                    done_n += 1
                    m = L.load_npz(unit_path(days[j]))[1]
                    hist[1 if int(m.get('threads', 2)) == 1 else 2].append(secs)
                    LOG(f'  {pd.Timestamp(days[j]).date()} done in {secs:.1f}s ({m.get("threads")} thread(s))')
    left = len([j for j in task_days(days, years) if not os.path.exists(unit_path(days[j]))])
    LOG(f'run: {done_n} fitted in {time.time() - t0:.0f}s; {left} left')


def status(years=YEARS):
    P, feats, days = load()
    for y in years:
        js = task_days(days, (y,))
        n = sum(os.path.exists(unit_path(days[j])) for j in js)
        print(f'{y}: {n}/{len(js)} models')
    rs = recent_secs(10)
    if rs:
        print('recent seconds per model:', ', '.join(f'{x:.0f}s/{t}t' for x, t in rs))


# ------------------------------------------------------------------ evaluation (section 2 of the PREREG)
def model_preds(days):
    out = {}
    for x in sorted(os.listdir(UNITS)):
        if x.endswith('.npz') and '.tmp' not in x:
            s, m = L.load_npz(os.path.join(UNITS, x))
            out[np.datetime64(pd.Timestamp(m['train_end']), 'ns')] = (s, m)
    return out


def scores(days, preds, k=K):
    """score of day t = mean of the k models trained on the k trading days before t; None for days missing a model"""
    rows = []
    pos = {d: i for i, d in enumerate(days)}
    complete = {}
    for t in days[(days >= np.datetime64('2022-01-01')) & (days < V_START)]:
        i = pos[t]
        ms = [days[i - h] for h in range(1, k + 1)]
        ok = all(m in preds for m in ms)
        complete[t] = ok
        if not ok:
            continue
        parts = [preds[m][0].xs(pd.Timestamp(t), level='date', drop_level=False) for m in ms]
        rows.append(pd.concat(parts, axis=1).mean(axis=1))
    s = pd.concat(rows).sort_index() if rows else pd.Series(dtype=float)
    return s, complete


def turnover(score, INFO):
    X = INFO.join(score.rename('score'), how='inner')
    prev, out = None, []
    for d, g in X.groupby(level='date', sort=True):
        top = set(g.sort_values('score', ascending=False, kind='stable').index.get_level_values('stock_id')[:25])
        if prev is not None:
            out.append(len(top - prev))
        prev = top
    return float(np.mean(out)) if out else np.nan


def evaluate():
    P, feats, days = load()
    preds = model_preds(days)
    INFO = L.load_info()
    INFO = L.info_frame(INFO[INFO.date < pd.Timestamp(V_START)])                     # D rows only (V stays sealed)
    s3, comp = scores(days, preds, 3)
    s1, _ = scores(days, preds, 1)
    years_done = [y for y in ('2022', '2023', '2024')
                  if all(v for t, v in comp.items() if str(pd.Timestamp(t).year) == y)]
    R = {'WF3': L.decide_sd(s3, INFO), 'WF1': L.decide_sd(s1, INFO)}
    p2 = pd.read_csv(os.path.join(STD, 'part2', 'results', 'part2_D_daily.csv'), index_col=0, parse_dates=True)
    keep = [d for d in R['WF3'].index if str(d.year) in years_done]
    A = pd.DataFrame({'WF3': R['WF3'].AR.reindex(keep), 'WF1': R['WF1'].AR.reindex(keep),
                      'SD': p2.SD.reindex(keep), 'QR': p2.QR.reindex(keep), 'MR': p2.MR.reindex(keep)})
    A.index.name = 'date'
    lines = [f'WF-daily (PREREG_race_live.md section 2), written {time.strftime("%Y-%m-%d %H:%M:%S")}',
             f'models fitted: {len(preds)}; complete years: {years_done or "none"}; days scored: {len(A)}', '']
    out = dict(years_done=years_done, n_days=len(A), rows={})
    if len(A):
        lines.append(f'{"":6s} {"mean":>7s} {"NW t":>6s} {"sd":>6s} ' + ' '.join(f'{y:>7s}' for y in years_done))
        for c in A.columns:
            x = A[c].dropna()
            by = [x[x.index.year == int(y)].mean() for y in years_done]
            lines.append(f'{c:6s} {x.mean():7.2f} {L.nw_t(x.values):6.2f} {x.std():6.1f} ' + ' '.join(f'{v:7.2f}' for v in by))
            out['rows'][c] = dict(mean=float(x.mean()), t=float(L.nw_t(x.values)), sd=float(x.std()), years=[float(v) for v in by])
        lines.append('')
        lines.append('paired differences (same days), NW 5 lags')
        for a, b in (('WF3', 'MR'), ('WF3', 'QR'), ('WF3', 'SD'), ('WF1', 'WF3'), ('MR', 'SD')):
            dlt = (A[a] - A[b]).dropna()
            se = L.nw_se(dlt.values)
            by = [dlt[dlt.index.year == int(y)].mean() for y in years_done]
            lines.append(f'{a}-{b:4s} {dlt.mean():+7.2f} t {L.nw_t(dlt.values):+6.2f} SE {se:5.2f}  by year ' +
                         ' '.join(f'{v:+7.2f}' for v in by))
            out[f'{a}-{b}'] = dict(mean=float(dlt.mean()), t=float(L.nw_t(dlt.values)), se=float(se), years=[float(v) for v in by])
        roll = A.rolling(19).mean().dropna()
        lines.append('')
        lines.append('share of 19-day means < 0: ' + ', '.join(f'{c} {(roll[c] < 0).mean():.3f}' for c in A.columns))
        s3k = s3[s3.index.get_level_values('date').isin(keep)]
        lines.append(f'names changed per day in the Top25 (WF3): {turnover(s3k, INFO):.1f}')
        bi = [m['best_iter'] for _, (s, m) in preds.items()]
        lines.append(f'best_iter of the daily models: median {np.median(bi):.0f}, 10% {np.percentile(bi, 10):.0f}, 90% {np.percentile(bi, 90):.0f}')
        dm = out['WF3-MR']
        verdict = 'WF-daily (pool 3)' if dm['mean'] >= -dm['se'] else 'monthly refit, 3 seeds'
        lines.append('')
        lines.append(f'decision rule (PREREG section 2): WF3 - MR = {dm["mean"]:+.2f}, -SE = {-dm["se"]:.2f} -> live uses {verdict}')
        lines.append('WF1 (latest model only) is descriptive; it is not part of the decision.')
        out['verdict'] = verdict
    os.makedirs(RES, exist_ok=True)
    txt = '\n'.join(lines)
    with open(os.path.join(RES, 'wf_daily.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    with open(os.path.join(RES, 'wf_daily.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    A.to_csv(os.path.join(RES, 'wf_daily_daily.csv'))
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prep', action='store_true'); ap.add_argument('--run', action='store_true')
    ap.add_argument('--status', action='store_true'); ap.add_argument('--eval', action='store_true')
    ap.add_argument('--budget', type=float, default=160); ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--years', default=','.join(YEARS))
    ap.add_argument('--run2', action='store_true'); ap.add_argument('--check-split', default='')
    ap.add_argument('--order', default='forward', choices=['forward', 'reverse']); ap.add_argument('--until', default='')
    a = ap.parse_args()
    years = tuple(a.years.split(','))
    if a.prep:
        prep()
    if a.run:
        run(a.budget, a.workers, years)
    if a.run2:
        run2(a.budget, a.workers, years, a.order, a.until)
    if a.check_split:
        check_split(a.check_split)
    if a.status:
        status(years)
    if a.eval:
        evaluate()


if __name__ == '__main__':
    main()

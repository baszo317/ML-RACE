r"""
rolling.py -- walk-forward "rolling pool" (copied unchanged from v4; used by train_and_predict_v5.py)

Course rule for the static leaderboard: day d may be predicted with everything known up to the previous trading day
(6/1 may be used to predict 6/2, never 6/2 to predict 6/1). v4 keeps, for every member, a pool of models trained on
labelled rows up to recent days:

  retrain points  the first point is given; afterwards a new point at trading day t whenever t is >= `every` trading
                  days after the newest point (every = 1 -> a new point every trading day)
  seeds           first point: seeds 1..init (v2's models of that day are copied when they exist);
                  later points: `new` seeds numbered from the date (yymmdd * 100 + k), so all models differ
  prediction      day d = equal-weight average of all models of the last `pool` points whose train_end <= d-1

Default (static): MLP every=1, new=2, pool=5 -> the MLP part of day d averages 10 models trained up to d-1 .. d-5;
Transformer every=5, new=1, pool=5. Models of neighbouring days differ only by a few days of data, so they are
(almost) exchangeable ensemble members and averaging them removes seed noise like 10 seeds of a single day
(README_v4.md, section 5).
Training a model only reads rows with date <= train_end, so no later information can enter it.
"""
import os, sys, time, glob, shutil, traceback
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import ens_lib as E

DEFAULT = {'static': {'mlp': dict(every=1, new=2, init=10, pool=5), 'transformer': dict(every=5, new=1, init=5, pool=5)},
           'live':   {'mlp': dict(every=1, new=2, init=10, pool=5), 'transformer': dict(every=1, new=1, init=5, pool=5)}}
SPEC_KEYS = ('kind', 'feats', 'window', 'target')

# ------------------------------------------------------------------ bookkeeping
def tag(te):
    return pd.Timestamp(te).strftime('%Y-%m-%d')

def seed_of(te, k):
    te = pd.Timestamp(te)
    return (te.year % 100) * 1_000_000 + te.month * 10_000 + te.day * 100 + k

def model_file(root, m, te, s):
    return os.path.join(root, m['name'], f'{tag(te)}_s{s}.pt')

def pred_file(root, m, te, s):
    return os.path.join(root, m['name'], f'{tag(te)}_s{s}.pred.csv')      # csv: readable by any pandas version

def labelled_days(P):
    return pd.DatetimeIndex(np.sort(P.loc[P.target_r.notna(), 'date'].unique()))

def prev_day(lab, d):
    x = lab[lab < pd.Timestamp(d)]
    return x[-1] if len(x) else None

def retrain_points(tdays, first, last, every):
    """[first, ...]: a new point at trading day t in (first, last] whenever t is >= `every` trading days after the
    previous point"""
    td = pd.DatetimeIndex(sorted(set(pd.to_datetime(list(tdays)))))
    first, last = pd.Timestamp(first), pd.Timestamp(last)
    pts = [first]
    for t in td[(td > first) & (td <= last)]:
        if int(((td > pts[-1]) & (td <= t)).sum()) >= every:
            pts.append(t)
    return pts

def seeds_at(i, te, sc):
    return list(range(1, sc['init'] + 1)) if i == 0 else [seed_of(te, k) for k in range(1, sc['new'] + 1)]

def pool_points(points, prev, pool):
    return [t for t in sorted(points) if t <= prev][-pool:]

def _load_ck(path):
    import torch
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)

def same_spec(saved, m):
    saved = saved or {}
    return all(saved.get(k) == m[k] for k in SPEC_KEYS)

def import_v2(m, te, s, v2_dir, dst):
    """copy v2's model <v2_dir>/<name>_seed<s>.pt if it has the same settings and was trained up to te"""
    if not v2_dir:
        return False
    f = os.path.join(v2_dir, f"{m['name']}_seed{s}.pt")
    if not os.path.exists(f):
        return False
    ck = _load_ck(f)
    if not same_spec(ck.get('member'), m) or pd.Timestamp(ck['train_end']) != pd.Timestamp(te):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(f, dst)
    return True

def v2_train_end(m, v2_dir, seeds):
    """common train_end of v2's seeds of member m, or None"""
    ends = set()
    for s in seeds:
        f = os.path.join(v2_dir or '', f"{m['name']}_seed{s}.pt")
        if not v2_dir or not os.path.exists(f):
            return None
        ck = _load_ck(f)
        if not same_spec(ck.get('member'), m):
            return None
        ends.add(pd.Timestamp(ck['train_end']))
    return ends.pop() if len(ends) == 1 else None

# ------------------------------------------------------------------ one model
def train_unit(P, m, te, s, fs, dst):
    te = pd.Timestamp(te)
    b = E.train_member(P[P.date <= te], m, s, te, fs[m['feats']])          # rows after te are never read
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    E.save_bundle(b, m, s, te, dst + '.tmp'); os.replace(dst + '.tmp', dst)

def load_checked(path, m):
    b = E.load_bundle(path)
    if not same_spec(b.get('member'), m):
        raise RuntimeError(f'{path} was trained with other settings than member {m["name"]}; move it away and rerun')
    return b

def predict_dates(b, P, dates):
    """P(close > open) of all stocks on `dates` (Series indexed by date, stock_id); only rows up to the last date and
    60 calendar days before the first one are read (the Transformer needs the 20 previous rows of each stock)"""
    dates = pd.DatetimeIndex(sorted(set(pd.to_datetime(list(dates)))))
    Q = P[(P.date >= dates.min() - pd.Timedelta(days=60)) & (P.date <= dates.max())]
    return E.predict_rows(b, Q, lambda d: d.date.isin(dates))

def do_unit(P, u, root, fs, v2_dir):
    """train (or copy from v2) one model and store its predictions for the dates it serves"""
    t0 = time.time(); m, te, s = u['m'], u['te'], u['seed']
    mf, pf = model_file(root, m, te, s), pred_file(root, m, te, s)
    try:
        how = 'saved model'
        if not os.path.exists(mf):
            if u.get('first') and import_v2(m, te, s, v2_dir, mf):
                how = 'copied from v2'
            else:
                train_unit(P, m, te, s, fs, mf); how = 'trained'
        b = load_checked(mf, m)
        p = predict_dates(b, P, u['dates']).rename('p').reset_index()
        p['date'] = pd.to_datetime(p.date).dt.strftime('%Y-%m-%d'); p['p'] = p.p.astype('float32')
        p.to_csv(pf + '.tmp', index=False, float_format='%.9g'); os.replace(pf + '.tmp', pf)
        return dict(name=m['name'], te=tag(te), seed=s, how=how, sec=time.time() - t0, epochs=b.get('best_epoch'), err=None)
    except Exception:
        return dict(name=m['name'], te=tag(te), seed=s, how='FAILED', sec=time.time() - t0, epochs=None, err=traceback.format_exc())

_G = {}
def _worker_init(panel_path, threads, root, v2_dir):
    import torch
    torch.set_num_threads(max(1, int(threads)))
    _G.update(P=pd.read_pickle(panel_path), fs=E.feature_sets(), root=root, v2=v2_dir)

def _worker_unit(u):
    return do_unit(_G['P'], u, _G['root'], _G['fs'], _G['v2'])

def run_units(P, units, root, fs, v2_dir, workers=1, threads=1, log=print):
    """run the units whose prediction file is missing; with workers > 1 they run in separate processes
    (each with `threads` torch threads); Transformer units start first because they take longest"""
    todo = [u for u in units if not os.path.exists(pred_file(root, u['m'], u['te'], u['seed']))]
    if not todo:
        return []
    todo.sort(key=lambda u: (u['m']['kind'] != 'transformer', u['te'], u['m']['name'], u['seed']))
    n, t0, res = len(todo), time.time(), []
    log(f'{n} model(s) to train or load ({sum(u["m"]["kind"] == "transformer" for u in todo)} Transformer), workers {workers}')
    def note(r):
        res.append(r)
        ep = f", epochs {r['epochs']}" if r['epochs'] else ''
        log(f"[{len(res)}/{n}] {r['name']} {r['te']} seed {r['seed']}: {r['how']} ({r['sec']:.0f}s{ep}); elapsed {(time.time() - t0) / 60:.1f} min")
        if r['err']:
            log(r['err'])
    if workers <= 1:
        for u in todo:
            note(do_unit(P, u, root, fs, v2_dir))
    else:
        import multiprocessing as mp
        os.makedirs(root, exist_ok=True); pp = os.path.join(root, '_panel_for_workers.pkl'); P.to_pickle(pp)
        try:
            with mp.get_context('spawn').Pool(workers, initializer=_worker_init, initargs=(pp, threads, root, v2_dir)) as pool:
                for r in pool.imap_unordered(_worker_unit, todo):
                    note(r)
        finally:
            if os.path.exists(pp):
                os.remove(pp)
    return res

# ------------------------------------------------------------------ static walk-forward
def static_plan(P, members, dates, sched):
    """units to run and, per (member, date), the pool used. The first point's models predict every date (so the
    frozen-at-first-point ensemble can be compared); later models only predict the dates they serve."""
    lab = labelled_days(P); dates = [pd.Timestamp(d) for d in dates]
    prev = {d: prev_day(lab, d) for d in dates}
    bad = [d for d in dates if prev[d] is None]
    if bad:
        raise SystemExit(f'no labelled day before {bad[0].date()}')
    first, last = prev[dates[0]], max(prev.values())
    units, use = [], {}
    for m in members:
        sc = sched[m['kind']]
        pts = retrain_points(lab, first, last, sc['every'])
        sd = {t: seeds_at(i, t, sc) for i, t in enumerate(pts)}
        serve = {t: [] for t in pts}
        for d in dates:
            pp = pool_points(pts, prev[d], sc['pool'])
            use[(m['name'], d)] = pp
            for t in pp:
                serve[t].append(d)
        for i, t in enumerate(pts):
            ds = dates if i == 0 else serve[t]
            if not ds:
                continue
            for s in sd[t]:
                units.append(dict(m=m, te=t, seed=s, dates=ds, first=(i == 0)))
    return units, use, prev

def load_preds(root, m, units):
    """long table date, stock_id, te, seed, p of member m"""
    parts = []
    for u in units:
        if u['m']['name'] != m['name']:
            continue
        x = pd.read_csv(pred_file(root, m, u['te'], u['seed']), dtype={'stock_id': str})
        parts.append(x.assign(te=tag(u['te']), seed=u['seed']))
    x = pd.concat(parts, ignore_index=True)
    return x.assign(date=pd.to_datetime(x.date).astype('datetime64[ns]'), te=pd.to_datetime(x.te).astype('datetime64[ns]'))

def pool_average(long, use, name, dates):
    """equal-weight average over the models of each date's pool -> Series indexed by date, stock_id"""
    U = pd.DataFrame([(d, t) for d in dates for t in use[(name, d)]], columns=['date', 'te']).astype('datetime64[ns]')
    x = long.merge(U, on=['date', 'te'], how='inner')
    return x.groupby(['date', 'stock_id']).p.mean()

def static_probs(root, members, units, use, dates, frozen=False):
    """member name -> pooled probabilities; frozen=True uses only the first point (= v2 static) for every date"""
    out = {}
    for m in members:
        long = load_preds(root, m, units)
        if frozen:
            t0 = pd.Timestamp(min(u['te'] for u in units if u['m']['name'] == m['name']))
            out[m['name']] = long[long.te == t0].groupby(['date', 'stock_id']).p.mean()
        else:
            out[m['name']] = pool_average(long, use, m['name'], dates)
    return out

# ------------------------------------------------------------------ live
def scan_pool(root, m):
    """{train_end: [seeds]} of the models saved for member m"""
    pts = {}
    for f in glob.glob(os.path.join(root, m['name'], '*_s*.pt')):
        b = os.path.basename(f)[:-3]
        te, s = b.rsplit('_s', 1)
        pts.setdefault(pd.Timestamp(te), []).append(int(s))
    return {t: sorted(v) for t, v in sorted(pts.items())}

def live_update(P, members, root, sched, fs, v2_dir, retrain='auto', log=print):
    """bring every member's pool up to date: start it with v2's models (or `init` new seeds), then add `new` seeds
    trained up to the newest labelled day whenever the newest point is >= `every` trading days old"""
    lab = labelled_days(P); last = lab[-1]
    for m in members:
        sc = sched[m['kind']]; pts = scan_pool(root, m)
        if not pts and v2_dir:
            te = v2_train_end(m, v2_dir, range(1, sc['init'] + 1))
            if te is not None and te <= last:
                for s in range(1, sc['init'] + 1):
                    import_v2(m, te, s, v2_dir, model_file(root, m, te, s))
                pts = scan_pool(root, m)
                log(f"{m['name']}: pool started with v2's {sc['init']} models trained up to {te.date()}")
        if pts:
            newest = max(pts)
            age = int(((lab > newest) & (lab <= last)).sum())
            due = (retrain == 'now' and newest < last) or (retrain == 'auto' and age >= sc['every'])
            seeds = seeds_at(1, last, sc) if due else []
        else:
            if retrain == 'none':
                raise SystemExit(f"--retrain none but no models for {m['name']} in {root}")
            seeds = seeds_at(0, last, sc)
        for s in seeds:
            t0 = time.time(); f = model_file(root, m, last, s)
            if not os.path.exists(f):
                train_unit(P, m, last, s, fs, f)
                log(f"trained {m['name']} seed {s} on data up to {last.date()} ({time.time() - t0:.0f}s)")
        if not seeds:
            log(f"{m['name']}: no new model needed (newest trained up to {max(pts).date()})")

def live_probs(P, members, root, sched, dates, log=print):
    """member name -> probabilities for `dates` averaged over each date's pool; also returns the pool description"""
    lab = labelled_days(P); out, desc = {}, {}
    for m in members:
        sc = sched[m['kind']]; pts = scan_pool(root, m); parts = []
        for d in dates:
            pv = prev_day(lab, d)
            pp = pool_points(pts, pv, sc['pool'])
            if not pp:
                raise SystemExit(f"no model of {m['name']} trained up to {pv.date()} or earlier")
            ps = [predict_dates(load_checked(model_file(root, m, t, s), m), P, [d]) for t in pp for s in pts[t]]
            parts.append(pd.concat(ps, axis=1).mean(axis=1))
            desc[(m['name'], d)] = f"{sum(len(pts[t]) for t in pp)} models, trained up to {pp[0].date()}..{pp[-1].date()}"
        out[m['name']] = pd.concat(parts)
    return out, desc

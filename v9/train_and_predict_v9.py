r"""
train_and_predict_v9.py -- v9 daily (live) prediction and static walk-forward (README_v9.md, sections 8-9)

Live / daily leaderboard (every morning, daily.bat):
  python train_and_predict_v9.py --dates auto
      input guards (v5 guards.py: newest zip merged, ex-dividend table fetched today, features present) ->
      2 new seeds per MLP member, 1 Transformer and 1 LightGBM trained up to the previous trading day ->
      prediction with the models of the last 5 retrain points -> ex-dividend rules -> beta hedge (lambda 0.75) ->
      output guards; any FATAL check stops before the file is written
  --retrain none     predict with the models already in the pool (about 1 minute), e.g. when time is short
  --force            write the file even if a FATAL check failed (only after you have read the message)
  --prelim           evening run (evening.bat, after the ~17:00 zip): trains the new models and writes the preliminary file
                     submission_v9_MMDD_prelim.csv; the TX night session of that day closes at 05:00, so it counts as missing.
                     Next morning daily.bat finds the models already trained and only predicts (about 2 minutes).
                     2026-10-28: the evening zips have no usable day to predict (15:52 / 21:05: no -1 row at all; 00:00: the
                     -1 row with ^VIX ^IXIC ^GSPC = -1, the U.S. close comes in the morning zip). Then --prelim only trains the
                     models (no preliminary file) and exits normally; daily.bat with the morning zip only predicts.
                     A final run whose day to predict has no U.S. close (-1) stops with a FATAL that says so.
Live checks added 2026-10-25: a day to predict in the past (old zip) is FATAL; a failed ex-dividend download is only a
WARN when the last success is <= 3 trading days old; the checks file records the package versions; the morning run
reports how many predictions differ from the evening's preliminary file.
Static leaderboard (2026-06-01..09-30) and its extension to the newest labelled day (October):
  python train_and_predict_v9.py --static [--workers 2]
      every day is predicted only with models trained up to the previous trading day (rolling pool as v4)
  python train_and_predict_v9.py --static --from-members out\members_v9_static.csv.gz [--lam 1.0] [--w-t 0.5] [--tag static_w025]
      recombine the stored member probabilities only (seconds; no training)
Outputs (v9\out\): submission_v9_<tag>.csv (upload), probabilities_v9_<tag>.csv, checks_v9_<tag>.txt,
  history_predictions.csv (live only), static: members_v9_static.csv.gz and static_report_v9.txt
"""
import os, sys, time, json, argparse
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rolling9 as V
E, R = V.E, V.R
import guards as G
G.SHARE1_RANGE = (0.10, 1.00)     # v9 walk-forward 2022-01..2026-10: min 0.16, 0.5 % quantile 0.24 (more 0s: ex-dividend rules)
G.PBAR_RANGE = (0.20, 0.80)       # v9 walk-forward: min 0.276, max 0.726

def log(t0, msg):
    print(f'[{time.time() - t0:5.0f}s] {msg}', flush=True)

def env_line():
    """package versions, written to the checks file (reproducibility)"""
    import platform
    v = {'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__}
    for mod in ('torch', 'lightgbm', 'pyarrow'):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = 'not installed'
    return ', '.join(f'{k} {x}' for k, x in v.items())

def check_not_past(dates, rep, today=None):
    """FATAL if a day to predict is already in the past: the zip in use is an old one (2026-10-25)"""
    today = pd.Timestamp.now().normalize() if today is None else pd.Timestamp(today).normalize()
    for d in dates:
        if pd.Timestamp(d) < today:
            rep.add('FATAL', f'{pd.Timestamp(d).date()} is in the past (today is {today.date()}): the zip in use is an old one -> download '
                             f'the newest zip (released about 17:00 after the close) and run again. If the newest zip still asks for this '
                             f'day and the market did not open on it (typhoon day), add the date to ..\\data\\ref\\closures_manual.csv '
                             f'and run daily.bat again (README_v9 10.8). Test runs on old data need --force --out <folder>')
    return rep

def check_exdiv_recent(data_dir, rep, max_days=3, today=None):
    """v5 guards.check_exdiv, relaxed (2026-10-25): OK if fetch_exdiv.py succeeded today; WARN if the newest success is at
    most `max_days` trading days old (ex-dates are announced weeks ahead and data\\ref\\exdiv_upcoming.csv only grows, so one
    failed download loses nothing); FATAL if it is older or there was never a success"""
    f = os.path.join(data_dir, 'ref', 'exdiv_fetch_log.csv')
    today = pd.Timestamp.now().normalize() if today is None else pd.Timestamp(today).normalize()
    if not os.path.exists(f):
        return G.check_exdiv(data_dir, rep, today=today)              # FATAL, v5 message
    L = pd.read_csv(f, parse_dates=['run_at'])
    ok = L.loc[L.status == 'ok', 'run_at']
    if ok.empty or (ok.dt.normalize() == today).any():
        return G.check_exdiv(data_dir, rep, today=today)              # OK (fetched today) or FATAL (never fetched), v5 messages
    last = ok.max().normalize()
    cal = G.trading_calendar(data_dir)
    age = int(((cal > last) & (cal <= today)).sum()) if cal is not None else int((today - last).days)
    if age <= max_days:
        rep.add('WARN', f'ex-dividend download did not succeed today; using the table fetched {last.date()} ({age} trading day(s) ago) - '
                        f'fine, ex-dates are announced weeks ahead (run ..\\v5\\fetch_exdiv.py again later if you can)')
    else:
        rep.add('FATAL', f'ex-dividend announcements last fetched {last.date()} ({age} trading days ago): run ..\\v5\\fetch_exdiv.py '
                         f'(or add known ex-dates to ..\\data\\ref\\exdiv_manual.csv and use --force)')
    return rep

def compare_with_prelim(out_dir, sub, sub_dates, stocks):
    """morning run: how many of the 50 predictions differ from last evening's preliminary file (if there is one)"""
    msgs = []
    S = sub.set_index('id')
    for d in sub_dates:
        col = pd.Timestamp(d).strftime('%Y-%m-%d')
        f = os.path.join(out_dir, f"submission_v9_{pd.Timestamp(d).strftime('%m%d')}_prelim.csv")
        if not os.path.exists(f):
            continue
        P0 = pd.read_csv(f, dtype={'id': str}).set_index('id')
        if col not in P0:
            continue
        n = int((P0[col].reindex(stocks).values != S[col].reindex(stocks).values).sum())
        msgs.append(f"{col}: {n} of {len(stocks)} predictions differ from last evening's {os.path.basename(f)} (night session added)")
    return '; '.join(msgs)

def load_panels(a, dates, t0):
    if a.panel_cache and os.path.exists(os.path.join(a.panel_cache, 'core9.pkl')):
        core = pd.read_pickle(os.path.join(a.panel_cache, 'core9.pkl')); ext = pd.read_pickle(os.path.join(a.panel_cache, 'ext9.pkl'))
        log(t0, f'panels read from {a.panel_cache}')
    else:
        core = V.build_core(a.data, dates)
        log(t0, f'core panel {core.shape}')
        ext = V.build_ext(core)
        log(t0, f'extended panel {ext.shape} ({int((~ext.core.astype(bool)).sum())} rows of extra stocks for mlp_res_B9)')
    for P in (core, ext):
        P['date'] = P['date'].astype('datetime64[ns]'); P['stock_id'] = P.stock_id.astype(str)
    return core, ext

def us_missing(data_dir, d):
    """True if global_data.csv has no U.S. close (^VIX ^IXIC ^GSPC) for day d: no row, -1 or empty (2026-10-28: the zip
    packed at midnight has the -1 row of the next day without the U.S. numbers; the morning zip fills them in)"""
    g = pd.read_csv(os.path.join(data_dir, 'global', 'global_data.csv'), parse_dates=['date'])
    row = g[g.date == pd.Timestamp(d)]
    if row.empty:
        return True
    v = row[['^VIX', '^IXIC', '^GSPC']].iloc[-1]
    return bool(((v == -1) | v.isna()).any())

def evening_no_prediction(a, dates):
    """evening run (--prelim, --dates auto): why no preliminary file can be made tonight, or None if it can"""
    if not getattr(a, 'prelim', False) or getattr(a, 'static', False):
        return None
    if not dates:
        return 'the zip has no day to predict yet (no -1 row; the evening zips only have the day\'s results)'
    if us_missing(a.data, dates[-1]):
        return (f'the -1 row of {pd.Timestamp(dates[-1]).date()} has no U.S. close yet (^VIX ^IXIC ^GSPC = -1; it comes in the '
                f'morning zip)')
    return None

def evening_train_only(a, members, root, sched, fs, t0, why, torch_ver=''):
    """evening without a usable day to predict: train the new models now, so that the morning run only predicts. The models
    are the same as the morning run would train (labelled days only; the seeds depend on the newest labelled day)."""
    log(t0, f'{why}: no preliminary file tonight; training the new models instead')
    core, ext = load_panels(a, [], t0)
    lab = R.labelled_days(core)
    log(t0, f'labelled data up to {lab[-1].date()}; torch {torch_ver}')
    if a.retrain != 'none':
        V.live_update(core, ext, members, root, sched, fs, retrain=a.retrain, log=lambda s: log(t0, s))
    log(t0, f'done: the models are trained on data up to {lab[-1].date()}. Next morning, after about 07:00, download the newest zip '
            f'(the day to predict with last night\'s U.S. close) and run daily.bat: it finds the models and only predicts (about 2 minutes)')

def report_static(X, core, frozen_X, extra, stocks, tpl_dates, out_dir, t0, members_df, fname='static_report_v9.txt'):
    lines = []
    def add(s=''):
        lines.append(s); print(s, flush=True)
    st = set(pd.to_datetime(tpl_dates)); s_end = max(st)
    def summary(name, ar):
        a1 = ar[ar.index.isin(st)]; a2 = ar[ar.index > s_end]
        add(f'{name:<58s} static {a1.mean():6.2f} ({len(a1)} d)   Oct {a2.mean():6.2f} ({len(a2)} d)   all {ar.mean():6.2f} ({len(ar)} d)')
    add('Top50 bp/day = mean of (2 pred - 1) * (close - open) / open; static = 2026-06-01..09-30, Oct = after 09-30 (extension)')
    summary('v9 rolling (each day: models trained up to the day before)', V.daily_ar(X, core))
    summary('  same, before the hedge', V.daily_ar(X, core, 'pred_nohedge'))
    summary('v9 frozen (all models trained up to 2026-05-29)', V.daily_ar(frozen_X, core))
    for name, f in extra:
        if f and os.path.exists(f):
            S = pd.read_csv(f, dtype={'id': str}); S = S.melt(id_vars='id', var_name='date', value_name='pred').rename(columns={'id': 'stock_id'})
            S['date'] = pd.to_datetime(S.date).astype('datetime64[ns]')
            summary(name, V.daily_ar(S, core))
    add()
    add('model pool used on the last day of the static period and on the newest day:')
    add(members_df)
    with open(os.path.join(out_dir, fname), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(str(x) for x in lines) + '\n')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=V.DATA)
    ap.add_argument('--template', default=os.path.join(E.ROOT, 'sample_submission.csv'))
    ap.add_argument('--dates', default=None, help="'auto' = the -1 placeholder day(s); or comma-separated dates")
    ap.add_argument('--static', action='store_true')
    ap.add_argument('--no-extend', action='store_true', help='static: only the template dates (no October extension)')
    ap.add_argument('--from-members', default=None, help='static: recombine stored member probabilities (no training)')
    ap.add_argument('--retrain', default='auto', choices=['auto', 'none', 'now'])
    ap.add_argument('--workers', type=int, default=1, help='static: parallel processes (Linux / WSL only)')
    ap.add_argument('--w-t', type=float, default=V.W_T)
    ap.add_argument('--lam', type=float, default=V.LAM, help='beta hedge: 1 = off, 0.75 = v9')
    ap.add_argument('--models-dir', default=None)
    ap.add_argument('--panel-cache', default=None, help='folder with core9.pkl / ext9.pkl (research cache) instead of building the panels')
    ap.add_argument('--limit', type=int, default=0, help='static: only the first N dates (smoke test)')
    ap.add_argument('--out', default=None, help='output folder (default v9\\out)')
    ap.add_argument('--tag', default='')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--prelim', action='store_true', help='evening run: tag MMDD_prelim; the night session is not out yet')
    a = ap.parse_args(); t0 = time.time()
    try:
        import torch
        torch_ver = torch.__version__
    except ImportError:                       # only --static --from-members works without torch
        torch_ver = 'not installed'
    mode = 'static' if a.static else 'live'
    sched = V.SCHED[mode]; members = [dict(m) for m in V.MEMBERS]; fs = V.feature_sets()
    root = a.models_dir if (a.models_dir and os.path.isabs(a.models_dir)) else os.path.join(HERE, a.models_dir or ('models_v9_static' if a.static else 'models_v9'))
    out_dir = a.out or os.path.join(HERE, 'out'); os.makedirs(out_dir, exist_ok=True)
    tpl = pd.read_csv(a.template, dtype={'id': str}); stocks = tpl['id'].tolist()
    tpl_dates = [pd.Timestamp(c) for c in tpl.columns if c != 'id']

    if a.static and a.from_members:
        M = pd.read_csv(a.from_members, dtype={'stock_id': str}, parse_dates=['date'])
        M['date'] = M.date.astype('datetime64[ns]'); M = M.set_index(['date', 'stock_id'])
        core = V.build_core(a.data, tpl_dates) if not a.panel_cache else load_panels(a, tpl_dates, t0)[0]
        probs = {m['name']: M[m['name']].dropna() for m in members}
        frozen = {m['name']: M[m['name'] + '_frozen'].dropna() for m in members}
        model_dates = sorted(M.index.get_level_values('date').unique())
        a.tag = a.tag or 'static'; ext = None
    else:
        if a.static:
            dates = list(tpl_dates); a.tag = a.tag or 'static'
        elif a.dates == 'auto':
            dates = E.placeholder_dates(a.data)
            why = evening_no_prediction(a, dates)
            if why:
                return evening_train_only(a, members, root, sched, fs, t0, why, torch_ver)
            if not dates:
                sys.exit("no -1 placeholder day in the data: run ..\\transformer\\update_data.py with today's zip first, or pass --dates")
            a.tag = a.tag or dates[-1].strftime('%m%d')
        else:
            dates = [pd.Timestamp(x) for x in a.dates.split(',')]; a.tag = a.tag or dates[-1].strftime('%m%d')
        if a.prelim and not a.static:
            a.tag += '_prelim'
        core, ext = load_panels(a, dates, t0)
        lab = R.labelled_days(core)
        if a.static and not a.no_extend:
            dates = sorted(set(dates) | set(d for d in lab if d > max(tpl_dates)))
        if a.static and a.limit:
            dates = sorted(dates)[:a.limit]
        log(t0, f'labelled data up to {lab[-1].date()}; torch {torch_ver}; schedule ' +
            '; '.join(f"{k}: every {v['every']}, {v['new']} new, pool {v['pool']}" for k, v in sched.items()))
        have = set(core.date.unique())
        model_dates = [d for d in dates if d in have and R.prev_day(lab, d) is not None]
        if not model_dates:
            sys.exit('none of the requested dates can be predicted (no labelled day before them)')

    rep = G.Report()
    if not a.static:
        core, nxt, ids = V.fix_ex_next(core, a.data)
        log(t0, f'next trading day {nxt.date()}: ex-dividend announced for {ids if ids else "none of the stocks"}')
        allf = sorted(set(sum((fs[m['feats']] for m in members), [])) - set(V.NIGHT))
        rep.add('INFO', 'environment: ' + env_line())
        check_not_past(model_dates, rep)
        G.check_inputs(core, model_dates, allf, a.data, rep)
        for d in model_dates:
            if not a.prelim and us_missing(a.data, d):
                rep.add('FATAL', f'{d.date()}: the zip has no U.S. close for this day yet (^VIX ^IXIC ^GSPC = -1; the zip packed at midnight '
                                 f'has the row without the numbers): download the morning zip (about 07:00) into ..\\downloads and run again')
        check_exdiv_recent(a.data, rep)
        for d in model_dates:
            nn = core.loc[core.date == d, V.NIGHT].isna().all(axis=1).mean()
            if nn > 0.5 and a.prelim:
                rep.add('INFO', f'{d.date()}: TX night session not out yet (it closes at 05:00) - expected in the evening run; '
                                f'daily.bat adds it next morning')
            elif nn > 0.5:
                rep.add('WARN', f'{d.date()}: no TX night session in ..\\v7\\data\\tx_daily.csv (run ..\\v7\\fetch\\fetch_night_session.py); '
                                f'night features count as missing (= training mean) today')
            else:
                rep.add('OK', f'{d.date()}: TX night session present')
        if rep.fatal() and not a.force:
            rep.show(lambda m: log(t0, m))
            sys.exit('stopped by the input checks above (nothing written). Fix the cause, or rerun with --force after checking.')

    if a.static and not a.from_members:
        tasks, units, use, prev = V.static_tasks(core, members, model_dates, sched)
        log(t0, f'static walk-forward: {len(model_dates)} dates {model_dates[0].date()}..{model_dates[-1].date()}, '
                f'{len(units)} models in {len(tasks)} retrain points (folder {os.path.basename(root)})')
        res = V.run_tasks(tasks, core, ext, fs, root, workers=a.workers, log=lambda s: log(t0, s))
        failed = [r for r in res if r['err']]
        if failed:
            sys.exit(f'{len(failed)} retrain point(s) failed (see above); rerun to continue (finished models are kept)')
        probs = V.static_probs(root, members, units, use, model_dates)
        frozen = V.static_probs(root, members, units, use, model_dates, frozen=True)
        M = pd.DataFrame({**{k: v for k, v in probs.items()}, **{k + '_frozen': v for k, v in frozen.items()}})
        M.reset_index().to_csv(os.path.join(out_dir, 'members_v9_static.csv.gz'), index=False, float_format='%.7f')
        pool_desc = []
        for d0 in sorted({max(d for d in model_dates if d <= max(tpl_dates)), model_dates[-1]}):
            for m in members:
                pp = use[(m['name'], d0)]
                pool_desc.append(f"  {d0.date()} {m['name']:<13s} {sum(len([u for u in units if u['m']['name'] == m['name'] and u['te'] == t]) for t in pp)} models, "
                                 f"trained up to {pp[0].date()}..{pp[-1].date()} (previous trading day {prev[d0].date()})")
    elif not a.static:
        if a.retrain != 'none':
            V.live_update(core, ext, members, root, sched, fs, retrain=a.retrain, log=lambda s: log(t0, s))
        G.check_pool(lambda m: V.scan_pool(root, m), members, R.prev_day(R.labelled_days(core), model_dates[-1]), rep)
        probs, desc = V.live_probs(core, ext, members, root, sched, model_dates, fs=fs, log=lambda s: log(t0, s))
        for (name, d), txt in desc.items():
            log(t0, f'{name} for {d.date()}: {txt}')

    ens = V.combine(probs, members, w_t=a.w_t)
    ens = ens[ens.index.get_level_values('date').isin(model_dates)]
    X, H = V.decide(ens, core, stocks, lam=a.lam)
    miss = X[X.score.isna()]
    if len(miss):
        rep.add('FATAL', f'{len(miss)} Top50 stock-days without a score, e.g. {miss[["date", "stock_id"]].head(3).values.tolist()}')
    if a.static:
        Xf, _ = V.decide(V.combine(frozen, members, w_t=a.w_t).loc[lambda d: d.index.get_level_values('date').isin(model_dates)], core, stocks, lam=a.lam)
        mate = [f for f in (os.path.join(E.ROOT, 'v6', 'submission-1-phase1_2.csv'), os.path.join(E.ROOT, 'v6', 'submission-1-phase1 (2).csv'),
                    os.path.join(E.ROOT, 'v6', 'others', 'submission-1-phase1_2.csv'), os.path.join(E.ROOT, 'v6', 'others', 'submission-1-phase1 (2).csv')) if os.path.exists(f)]
        report_static(X, core, Xf, [('v4 static upload (rolling, 15.36 on the static leaderboard)', next((f for f in (os.path.join(E.ROOT, 'v4', 'out', 'submission_v4_static.csv'), os.path.join(E.ROOT, '棄案', 'v4', 'out', 'submission_v4_static.csv')) if os.path.exists(f)), None)),
                                    ('groupmate V3 (submission-1-phase1_2.csv)', mate[0] if mate else None)],
                      stocks, tpl_dates, out_dir, t0, '\n'.join(pool_desc) if not a.from_members else '(recombined from stored member probabilities)',
                      fname='static_report_v9.txt' if a.tag == 'static' else f'static_report_v9_{a.tag}.txt')
    G.check_output(X[X.date.isin(model_dates)], set(stocks), rep)

    fb_rows = core[core.target_r.notna()].sort_values('date')
    fb = (fb_rows.groupby('stock_id').target_r.apply(lambda s: s.tail(250).mean()) > 0).astype(int).reindex(stocks).fillna(0).astype(int)
    suf = f'_{a.tag}' if a.tag else ''
    rep.show(lambda m: log(t0, m))
    with open(os.path.join(out_dir, f'checks_v9{suf}.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(f'[{l}] {m}' for l, m in rep.items) + '\n')
    if rep.fatal() and not a.force:
        sys.exit(f'stopped by {len(rep.fatal())} FATAL check(s) above: nothing written. Fix the cause, or rerun with --force after checking.')
    sub_dates = tpl_dates if a.static else [pd.Timestamp(d) for d in (E.placeholder_dates(a.data) if a.dates == 'auto' else model_dates)]
    sub = pd.DataFrame({'id': stocks})
    for d in sub_dates:
        x = X[X.date == d].set_index('stock_id')['pred'].reindex(stocks)
        sub[d.strftime('%Y-%m-%d')] = np.where(x.isna(), fb.values, x.fillna(0).values).astype(int)
    vals = sub.drop(columns='id')
    assert list(sub['id']) == stocks and vals.notna().all().all() and set(np.unique(vals.values)) <= {0, 1}, 'submission format check failed'
    if a.static:
        assert list(sub.columns) == list(tpl.columns), 'static submission columns differ from the template'
    sub.to_csv(os.path.join(out_dir, f'submission_v9{suf}.csv'), index=False)
    log(t0, f'wrote {os.path.join(out_dir, f"submission_v9{suf}.csv")}: {sub.shape[0]} stocks x {sub.shape[1] - 1} date(s), share of 1 = {vals.values.mean():.3f}')
    if not a.static and not a.prelim:
        msg = compare_with_prelim(out_dir, sub, sub_dates, stocks)
        if msg:
            log(t0, msg)
            with open(os.path.join(out_dir, f'checks_v9{suf}.txt'), 'a', encoding='utf-8') as fh:
                fh.write(f'[INFO] {msg}\n')
    cols = ['date', 'stock_id', 'p_raw', 'p_raw_mlp', 'p_raw_transformer', 'pbar', 'p_e', 'p_e_mlp', 'p_e_lgb', 'p_e_c', 'score'] + \
           list(V.EX4) + ['beta', 'pred_nohedge', 'pred']
    X[[c for c in cols if c in X]].to_csv(os.path.join(out_dir, f'probabilities_v9{suf}.csv'), index=False, float_format='%.6f')
    if len(H):
        H.to_csv(os.path.join(out_dir, f'hedge_v9{suf}.csv'), index=False, float_format='%.5f')
    if not a.static:
        hist = os.path.join(out_dir, 'history_predictions.csv')
        h = X[[c for c in cols if c in X]].assign(w_t=a.w_t, lam=a.lam, run='prelim' if a.prelim else 'final',
                                               made_at=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'))
        if os.path.exists(hist):
            old = pd.read_csv(hist, dtype={'stock_id': str}, parse_dates=['date'])
            h = pd.concat([old[~old.date.isin(h.date.unique())], h], ignore_index=True)
        h.sort_values(['date', 'stock_id']).to_csv(hist, index=False, float_format='%.6f')
    for d in model_dates[-3:]:
        x = X[X.date == d]
        exd = x[x.ex_any].stock_id.tolist()
        log(t0, f"{d.date()}: pbar {x.pbar.iloc[0]:.3f}; share of 1 among the 50: before hedge {x.pred_nohedge.mean():.2f}, final {x.pred.mean():.2f}; "
                f"changed by the hedge {int((x.pred != x.pred_nohedge).sum())}" + (f"; ex-dividend rules -> 0: {exd}" if exd else ''))

if __name__ == '__main__':
    main()

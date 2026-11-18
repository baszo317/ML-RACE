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
            if not dates:
                sys.exit("no -1 placeholder day in the data: run ..\\transformer\\update_data.py with today's zip first, or pass --dates")
            a.tag = a.tag or dates[-1].strftime('%m%d')
        else:
            dates = [pd.Timestamp(x) for x in a.dates.split(',')]; a.tag = a.tag or dates[-1].strftime('%m%d')
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
        G.check_inputs(core, model_dates, allf, a.data, rep)
        G.check_exdiv(a.data, rep)
        for d in model_dates:
            nn = core.loc[core.date == d, V.NIGHT].isna().all(axis=1).mean()
            if nn > 0.5:
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
        mate = [f for f in (os.path.join(E.ROOT, 'v6', 'submission-1-phase1_2.csv'), os.path.join(E.ROOT, 'v6', 'submission-1-phase1 (2).csv')) if os.path.exists(f)]
        report_static(X, core, Xf, [('v4 static upload (rolling, 15.36 on the static leaderboard)', os.path.join(E.ROOT, 'v4', 'out', 'submission_v4_static.csv')),
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
    cols = ['date', 'stock_id', 'p_raw', 'p_raw_mlp', 'p_raw_transformer', 'pbar', 'p_e', 'p_e_mlp', 'p_e_lgb', 'p_e_c', 'score'] + \
           list(V.EX4) + ['beta', 'pred_nohedge', 'pred']
    X[[c for c in cols if c in X]].to_csv(os.path.join(out_dir, f'probabilities_v9{suf}.csv'), index=False, float_format='%.6f')
    if len(H):
        H.to_csv(os.path.join(out_dir, f'hedge_v9{suf}.csv'), index=False, float_format='%.5f')
    if not a.static:
        hist = os.path.join(out_dir, 'history_predictions.csv')
        h = X[[c for c in cols if c in X]].assign(w_t=a.w_t, lam=a.lam, made_at=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'))
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

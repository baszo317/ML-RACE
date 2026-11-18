r"""
train_and_predict_v4.py -- v4 = v2 decision rule + walk-forward rolling pool + ex-dividend dates 2026/08-10

Static leaderboard (template dates 2026-06-01 .. 09-30):
  python train_and_predict_v4.py --static --workers 3
      day d is predicted ONLY with models trained on labelled data up to the previous trading day d-1
      (course rule: 6/1 may be used for 6/2, never 6/2 for 6/1):
        4 MLP members  : 2 new seeds every trading day, pool = the last 5 days (10 models per member)
        Transformer    : 1 new seed every 5 trading days, pool = the last 5 points (5 models)
      the 2026-05-29 models of v2 (..\v2\models_v2_static\) are the first pool point (copied, not retrained);
      every finished model and its predictions stay in models_v4_static\ -> stop at any time, rerun to continue
Live / daily leaderboard (every morning, after ..\transformer\update_data.py and fetch_exdiv.py; daily.bat):
  python train_and_predict_v4.py --dates auto
      adds 2 MLP seeds per member and 1 Transformer seed trained up to yesterday, predicts with the last 5 days;
      the first run starts the pool with v2's production models (..\v2\models_v2\)
  --retrain none     predict with the models already in the pool (seconds), e.g. when time is short
  --retrain now      add new seeds even if the schedule says it is not due
Decision rule (unchanged, README_v2.md):
  score = W_T * (pbar - 0.5) + (p_e - mean_day(p_e)), W_T = 0.5;  prediction = 1 if score > 0;  0 on ex-dividend days
Outputs (v4\out\): submission_v4_<tag>.csv (upload), probabilities_v4_<tag>.csv, history_predictions.csv (live only)
"""
import os, sys, time, argparse
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ens_lib as E
import rolling as R

MEMBERS = [E.member('mlp_raw_B',  'mlp',         feats='B',    target='raw'),
           E.member('mlp_raw_xr', 'mlp',         feats='B+XR', target='raw'),
           E.member('tf_raw_B',   'transformer', feats='B',    target='raw'),
           E.member('mlp_res_B',  'mlp',         feats='B',    target='resid'),
           E.member('mlp_res_xr', 'mlp',         feats='B+XR', target='resid')]
TF_RESID = E.member('tf_res_B', 'transformer', feats='B', target='resid')          # added with --tf-resid
V2_DIR = os.path.join(E.ROOT, 'v2')

def log(t0, msg):
    print(f'[{time.time() - t0:5.0f}s] {msg}', flush=True)

def schedules(a, mode):
    sc = {k: dict(v) for k, v in R.DEFAULT[mode].items()}
    for kind, pre in (('mlp', 'mlp'), ('transformer', 'tf')):
        for key in ('every', 'new', 'pool'):
            v = getattr(a, f'{pre}_{key}')
            if v is not None:
                sc[kind][key] = v
    return sc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=E.DATA)
    ap.add_argument('--template', default=os.path.join(E.ROOT, 'sample_submission.csv'))
    ap.add_argument('--dates', default=None, help="'auto' = the -1 placeholder day(s); or comma-separated dates; default: template columns")
    ap.add_argument('--static', action='store_true', help='static leaderboard: walk-forward over the template dates')
    ap.add_argument('--retrain', default='auto', choices=['auto', 'none', 'now'])
    ap.add_argument('--workers', type=int, default=1, help='static: parallel processes (3 is a good value on a 4-core laptop)')
    ap.add_argument('--threads', type=int, default=0, help="torch threads (per worker when --workers > 1; default: 1 per worker, otherwise torch's default = physical cores)")
    ap.add_argument('--tf-resid', action='store_true', help='also use a Transformer trained on the residual target (stock view)')
    ap.add_argument('--no-tf', action='store_true', help='leave the Transformer out (MLP members only)')
    ap.add_argument('--w-t', type=float, default=E.W_T, help='weight of the market view (0 = pure stock picking)')
    ap.add_argument('--no-exdiv', action='store_true')
    for pre in ('mlp', 'tf'):
        ap.add_argument(f'--{pre}-every', type=int, default=None, help=f'{pre}: new retrain point every N trading days')
        ap.add_argument(f'--{pre}-new', type=int, default=None, help=f'{pre}: seeds per new retrain point')
        ap.add_argument(f'--{pre}-pool', type=int, default=None, help=f'{pre}: number of most recent retrain points averaged')
    ap.add_argument('--models-dir', default=None)
    ap.add_argument('--v2-models', default=None, help="v2 model folder used for the first pool point ('' = none)")
    ap.add_argument('--tag', default='')
    a = ap.parse_args(); t0 = time.time()
    import torch
    if a.static and a.workers > 1:
        torch.set_num_threads(max(1, a.threads or 1))
    elif a.threads > 0:
        torch.set_num_threads(a.threads)           # otherwise torch's default (= physical cores; v2 used all logical threads)
    mode = 'static' if a.static else 'live'
    sched = schedules(a, mode)
    members = [dict(m) for m in MEMBERS + ([TF_RESID] if a.tf_resid else []) if not (a.no_tf and m['kind'] == 'transformer')]
    root = os.path.join(HERE, a.models_dir or ('models_v4_static' if a.static else 'models_v4'))
    v2_dir = a.v2_models if a.v2_models is not None else os.path.join(V2_DIR, 'models_v2_static' if a.static else 'models_v2')
    out_dir = os.path.join(HERE, 'out'); os.makedirs(out_dir, exist_ok=True)

    tpl = pd.read_csv(a.template, dtype={'id': str}); stocks = tpl['id'].tolist()
    if a.dates == 'auto':
        dates = E.placeholder_dates(a.data)
        if not dates:
            sys.exit("no -1 placeholder day in the data: run ..\\transformer\\update_data.py with today's zip first, or pass --dates")
        a.tag = a.tag or dates[-1].strftime('%m%d')
    else:
        dates = [pd.Timestamp(x) for x in (a.dates.split(',') if a.dates else [c for c in tpl.columns if c != 'id'])]
        a.tag = a.tag or ('static' if a.static else '')
    P = E.build_panel(a.data, dates)
    P['date'] = P['date'].astype('datetime64[ns]')            # one datetime resolution everywhere (pandas 2 and 3)
    lab = R.labelled_days(P); fs = E.feature_sets()
    log(t0, f'panel {P.shape}, labelled data up to {lab[-1].date()}; torch {torch.__version__}, {torch.get_num_threads()} thread(s); schedule ' +
        '; '.join(f"{k}: every {v['every']} day(s), {v['new']} new seed(s), pool {v['pool']}" for k, v in sched.items()))

    have = set(P.date.unique())
    model_dates = [d for d in dates if d in have and R.prev_day(lab, d) is not None]
    if not model_dates:
        sys.exit('none of the requested dates can be predicted (no labelled day before them)')
    if a.static:
        units, use, prev = R.static_plan(P, members, model_dates, sched)
        n_new = sum(not os.path.exists(R.pred_file(root, u['m'], u['te'], u['seed'])) for u in units)
        log(t0, f'static walk-forward: {len(model_dates)} dates {model_dates[0].date()}..{model_dates[-1].date()}, '
                f'{len(units)} models in total, {n_new} still to do (folder {os.path.basename(root)})')
        res = R.run_units(P, units, root, fs, v2_dir, workers=a.workers, threads=a.threads or 1, log=lambda s: log(t0, s))
        failed = [r for r in res if r['err']]
        if failed:
            sys.exit(f'{len(failed)} model(s) failed (see above); fix the problem and rerun -> finished models are kept')
        probs = R.static_probs(root, members, units, use, model_dates)
        for m in members:
            d0 = model_dates[-1]
            log(t0, f"{m['name']}: {model_dates[-1].date()} uses models trained up to "
                    f"{use[(m['name'], d0)][0].date()}..{use[(m['name'], d0)][-1].date()} (previous trading day {prev[d0].date()})")
    else:
        if a.retrain != 'none':
            R.live_update(P, members, root, sched, fs, v2_dir, retrain=a.retrain, log=lambda s: log(t0, s))
        probs, desc = R.live_probs(P, members, root, sched, model_dates)
        for (name, d), txt in desc.items():
            log(t0, f'{name} for {d.date()}: {txt}')

    ens = E.combine(probs, members, w_t=a.w_t)
    info = P.set_index(['date', 'stock_id'])[['ex_today']].reindex(ens.index)
    assert info.ex_today.notna().all(), 'predictions do not line up with the panel rows (date / stock_id mismatch)'
    ens['ex_today'] = info.ex_today.values
    ens['pred_plain'] = (ens.p_raw > 0.5).astype(int)                       # v1 rule, for comparison
    ens['pred'] = (ens.score > 0).astype(int)
    if not a.no_exdiv:
        ens['pred'] = E.apply_exdiv(ens.pred, ens.ex_today)
    ens = ens.reset_index()

    lab_rows = P[P.target_r.notna()].sort_values('date')                   # v1 fallback for dates without a previous day
    fb = (lab_rows.groupby('stock_id').target_r.apply(lambda s: s.tail(250).mean()) > 0).astype(int).reindex(stocks).fillna(0).astype(int)
    suf = f'_{a.tag}' if a.tag else ''
    sub = pd.DataFrame({'id': stocks})
    for d in dates:
        c = d.strftime('%Y-%m-%d')
        if d in model_dates:
            x = ens[ens.date == d].set_index('stock_id')['pred'].reindex(stocks)
            sub[c] = np.where(x.isna(), fb.values, x.fillna(0).values).astype(int)
        else:
            sub[c] = fb.values
    vals = sub.drop(columns='id')
    assert list(sub['id']) == stocks and vals.notna().all().all() and set(np.unique(vals.values)) <= {0, 1}, 'submission format check failed'
    if a.static and not a.dates:
        assert list(sub.columns) == list(tpl.columns), 'static submission columns differ from the template'
    sub.to_csv(os.path.join(out_dir, f'submission_v4{suf}.csv'), index=False)
    log(t0, f'wrote out\\submission_v4{suf}.csv: {sub.shape[0]} stocks (same ids and order as the template) x {sub.shape[1] - 1} dates, '
            f'values 0/1, share of 1 = {vals.values.mean():.3f}')
    ens.to_csv(os.path.join(out_dir, f'probabilities_v4{suf}.csv'), index=False, float_format='%.6f')
    if not a.static:            # live predictions only: monitor_live.py must not mix in the static backtest months
        hist = os.path.join(out_dir, 'history_predictions.csv')
        h = ens.assign(w_t=a.w_t, made_at=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'))
        if os.path.exists(hist):
            old = pd.read_csv(hist, dtype={'stock_id': str}, parse_dates=['date'])
            h = pd.concat([old[~old.date.isin(h.date.unique())], h], ignore_index=True)
        h.sort_values(['date', 'stock_id']).to_csv(hist, index=False, float_format='%.6f')
    for d in model_dates[-3:]:
        x = ens[(ens.date == d) & ens.stock_id.isin(stocks)]
        exd = x[x.ex_today == 1].stock_id.tolist()
        log(t0, f"{d.date()}: market view pbar {x.pbar.iloc[0]:.3f}; share of 1 among the 50: v1 rule {x.pred_plain.mean():.2f}, v4 {x.pred.mean():.2f}"
                + (f"; ex-dividend today -> 0: {exd}" if exd else ''))
    n_fb = len(dates) - len(model_dates)
    if n_fb:
        log(t0, f'{n_fb} requested date(s) without a labelled previous day -> v1 fallback rule')

if __name__ == '__main__':
    main()

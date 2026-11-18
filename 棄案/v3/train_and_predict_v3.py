r"""
train_and_predict_v3.py -- v3 ensemble: train (or reuse) the models and write the submission

v3 = v2 with the 5 new stock-picking features of features_v3.py (gap_bias_60, gap_bias_250, resid_vol_60, dow_bias_e,
hi52 + their ranks) added to the stock-view members. Everything else (market view, combine rule, W_T, ex-dividend
rule, retraining schedule) is v2 code, imported unchanged from ..\v2\ens_lib.py. Evidence: README_v3.md.

Live / daily leaderboard (every morning, after ..\transformer\update_data.py and fetch_exdiv.py):
  python train_and_predict_v3.py --dates auto
      retrains what is stale and predicts the -1 placeholder day. Default --retrain auto:
        MLP members  : retrained whenever newer labels exist (= every day, about 5 minutes)
        Transformer  : retrained when its training data is >= 5 trading days old (= weekly, about 30 minutes)
  python train_and_predict_v3.py --dates auto --retrain none     predict only with the saved models (seconds)
  python train_and_predict_v3.py --dates auto --retrain all      retrain everything
  python train_and_predict_v3.py --train-only                    (re)train on all labelled data, no prediction
Static leaderboard (models frozen at 2026-05-29 in models_v3_static\, template dates):
  python train_and_predict_v3.py --static

The v2 rule, unchanged in v3 (evidence and proofs in ..\v2\README_v2.md)
  market view : pbar = average P(close > open) of the 'raw' members (MLP on B, MLP on B+ranks, Transformer on B)
                over the 100 stocks of the day -> how much the models expect the whole market to rise intraday
  stock view  : 'resid' members (MLP on B+N, MLP on B+ranks+N; N = features_v3.py) are trained on "close/open beats the same-day average of
                the 100 stocks"; their probability minus its same-day mean = which stocks should beat the others
  score       : W_T * (pbar - 0.5) + (p_e - mean_day(p_e)),  W_T = 0.5;   prediction = 1 if score > 0
  ex-dividend : prediction = 0 on announced ex-dividend days (data\ref\corporate_actions.parquet,
                exdiv_upcoming.csv from fetch_exdiv.py, exdiv_manual.csv)
Outputs (folder v3\out\, tag = predicted date or --tag)
  submission_v3_<tag>.csv          upload this one
  submission_v3_plain_<tag>.csv    v1 rule on the raw members (P > 0.5, no ex-dividend rule), for comparison only
  probabilities_v3_<tag>.csv       all intermediate numbers per date x stock
  history_predictions.csv          every live prediction made so far (used by monitor_live.py; --static does not write it)
"""
import os, sys, time, argparse
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ens_v3 as E

MEMBERS = [E.member('mlp_raw_B',  'mlp',         feats='B',    target='raw',   seeds=10),
           E.member('mlp_raw_xr', 'mlp',         feats='B+XR', target='raw',   seeds=10),
           E.member('tf_raw_B',   'transformer', feats='B',    target='raw',   seeds=5),
           E.member('mlp_res_BN', 'mlp',         feats='B+N',    target='resid', seeds=10),     # v3: + 5 new features
           E.member('mlp_res_xrN', 'mlp',        feats='B+XR+N', target='resid', seeds=10)]     # v3: + 5 new + their ranks
TF_RESID = E.member('tf_res_B', 'transformer', feats='B', target='resid', seeds=5)     # added with --tf-resid
TF_MAX_AGE = 5          # trading days
STATIC_END = pd.Timestamp('2026-05-29')

def log(t0, msg):
    print(f'[{time.time() - t0:5.0f}s] {msg}', flush=True)

def model_path(mdir, m, s):
    return os.path.join(mdir, f"{m['name']}_seed{s}.pt")

def saved_train_end(mdir, m):
    """common train_end of the saved seeds of member m, or None if any seed is missing / was trained differently"""
    import torch
    ends = set()
    for s in range(1, m['seeds'] + 1):
        f = model_path(mdir, m, s)
        if not os.path.exists(f):
            return None
        try:
            ck = torch.load(f, weights_only=False)
        except TypeError:
            ck = torch.load(f)
        saved = ck.get('member') or {}
        if any(saved.get(k) != m[k] for k in ('kind', 'feats', 'window', 'target')):
            return None
        ends.add(ck['train_end'])
    return pd.Timestamp(ends.pop()) if len(ends) == 1 else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=E.DATA)
    ap.add_argument('--template', default=os.path.join(E.ROOT, 'sample_submission.csv'))
    ap.add_argument('--dates', default=None, help="'auto' = the -1 placeholder day(s) in the data; or comma-separated dates; default: template columns")
    ap.add_argument('--static', action='store_true', help='static leaderboard: models frozen at 2026-05-29, template dates')
    ap.add_argument('--retrain', default='auto', choices=['auto', 'all', 'none', 'mlp'])
    ap.add_argument('--train-only', action='store_true')
    ap.add_argument('--tf-resid', action='store_true', help='also use a Transformer trained on the residual target in the stock view')
    ap.add_argument('--models-dir', default=None)
    ap.add_argument('--w-t', type=float, default=E.W_T, help='weight of the market view (0 = pure stock picking)')
    ap.add_argument('--no-exdiv', action='store_true')
    ap.add_argument('--mlp-seeds', type=int, default=None); ap.add_argument('--tf-seeds', type=int, default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--threads', type=int, default=0)
    a = ap.parse_args(); t0 = time.time()
    import torch
    torch.set_num_threads(a.threads if a.threads > 0 else max(1, os.cpu_count() or 1))
    members = [dict(m) for m in MEMBERS + ([TF_RESID] if a.tf_resid else [])]
    for m in members:
        if m['kind'] == 'mlp' and a.mlp_seeds: m['seeds'] = a.mlp_seeds
        if m['kind'] == 'transformer' and a.tf_seeds: m['seeds'] = a.tf_seeds
    mdir = os.path.join(HERE, a.models_dir or ('models_v3_static' if a.static else 'models_v3')); os.makedirs(mdir, exist_ok=True)
    out_dir = os.path.join(HERE, 'out'); os.makedirs(out_dir, exist_ok=True)

    tpl = pd.read_csv(a.template, dtype={'id': str}); stocks = tpl['id'].tolist()
    if a.train_only:
        dates = []
    elif a.dates == 'auto':
        dates = E.placeholder_dates(a.data)
        if not dates:
            sys.exit("no -1 placeholder day in the data: run ..\\transformer\\update_data.py with today's zip first, or pass --dates")
        a.tag = a.tag or dates[-1].strftime('%m%d')
    else:
        dates = [pd.Timestamp(x) for x in (a.dates.split(',') if a.dates else [c for c in tpl.columns if c != 'id'])]
        a.tag = a.tag or ('static' if a.static else '')
    P = E.build_panel(a.data, dates)
    last_lab = P[P.target_r.notna()].date.max()
    train_end = STATIC_END if a.static else last_lab
    tdays = np.sort(P[P.target_r.notna()].date.unique())
    fs = E.feature_sets()
    log(t0, f'panel {P.shape}, labelled data up to {last_lab.date()}, new models use data up to {train_end.date()}')

    # ---------------------------------------------------------------- train what is needed
    bundles = {}
    for m in members:
        have = saved_train_end(mdir, m)
        if a.static:
            need = a.retrain == 'all' or have != STATIC_END          # static models must stop exactly at 2026-05-29
        elif a.retrain == 'none':
            need = have is None
            if need:
                sys.exit(f"--retrain none but no usable saved models for {m['name']} in {mdir}")
        elif a.retrain == 'all':
            need = True
        elif a.retrain == 'mlp':
            need = m['kind'] == 'mlp' or have is None
        else:   # auto
            if have is None:
                need = True
            elif m['kind'] == 'mlp':
                need = have < train_end
            else:
                need = int((tdays > np.datetime64(have)).sum()) >= TF_MAX_AGE
        bundles[m['name']] = []
        for s in range(1, m['seeds'] + 1):
            f = model_path(mdir, m, s)
            if need:
                b = E.train_member(P, m, s, train_end, fs[m['feats']]); E.save_bundle(b, m, s, train_end, f)
                b['train_end'] = str(train_end.date())
                log(t0, f"trained {m['name']} seed {s}/{m['seeds']} (data {E.window_start(train_end, m['window']).date()}..{train_end.date()}, epochs {b['best_epoch']})")
            else:
                b = E.load_bundle(f)
            bundles[m['name']].append(b)
        if not need:
            log(t0, f"{m['name']}: reusing {m['seeds']} saved models trained up to {have.date()}")
    if a.train_only:
        log(t0, 'training done'); return

    # ---------------------------------------------------------------- predict all 100 stocks (the views need the whole day)
    newest = max(pd.Timestamp(b[0]['train_end']) for b in bundles.values())
    have_dates = set(P.date.unique()); model_dates = [d for d in dates if d in have_dates and d > newest]
    if not model_dates:
        sys.exit('none of the requested dates can be predicted (no data on the previous trading day, or the dates are inside the training period)')
    sel = lambda d: d.date.isin(model_dates)
    probs = {m['name']: pd.concat([E.predict_rows(b, P, sel) for b in bundles[m['name']]], axis=1).mean(axis=1) for m in members}
    ens = E.combine(probs, members, w_t=a.w_t)
    info = P.set_index(['date', 'stock_id'])[['ex_today']].reindex(ens.index)
    ens['ex_today'] = info.ex_today.fillna(0).values
    ens['pred_plain'] = (ens.p_raw > 0.5).astype(int)                       # v1 rule, for comparison
    ens['pred'] = (ens.score > 0).astype(int)
    if not a.no_exdiv:
        ens['pred'] = E.apply_exdiv(ens.pred, ens.ex_today)
    ens = ens.reset_index()

    # requested dates without data on the previous day: v1 fallback rule
    lab = P[P.target_r.notna()].sort_values('date')
    fb = (lab.groupby('stock_id').target_r.apply(lambda s: s.tail(250).mean()) > 0).astype(int).reindex(stocks).fillna(0).astype(int)
    suf = f'_{a.tag}' if a.tag else ''
    for name, col in [('submission_v3', 'pred'), ('submission_v3_plain', 'pred_plain')]:
        sub = pd.DataFrame({'id': stocks})
        for d in dates:
            c = d.strftime('%Y-%m-%d')
            if d in model_dates:
                x = ens[ens.date == d].set_index('stock_id')[col].reindex(stocks)
                sub[c] = np.where(x.isna(), fb.values, x.fillna(0).values).astype(int)
            else:
                sub[c] = fb.values
        vals = sub.drop(columns='id')
        assert list(sub['id']) == stocks and vals.notna().all().all() and set(np.unique(vals.values)) <= {0, 1}, 'submission format check failed'
        sub.to_csv(os.path.join(out_dir, f'{name}{suf}.csv'), index=False)
        log(t0, f'wrote out\\{name}{suf}.csv: {sub.shape[0]} stocks (same ids and order as the template) x {sub.shape[1] - 1} dates, values 0/1, share of 1 = {vals.values.mean():.3f}')
    ens.to_csv(os.path.join(out_dir, f'probabilities_v3{suf}.csv'), index=False, float_format='%.6f')
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
        log(t0, f"{d.date()}: market view pbar {x.pbar.iloc[0]:.3f}; share of 1 among the 50: v1 rule {x.pred_plain.mean():.2f}, v3 {x.pred.mean():.2f}"
                + (f"; ex-dividend today -> 0: {exd}" if exd else ''))
    n_fb = len(dates) - len(model_dates)
    if n_fb:
        log(t0, f'{n_fb} requested date(s) without data on the previous day -> v1 fallback rule')

if __name__ == '__main__':
    main()

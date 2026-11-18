r"""
static_coef_v9.py -- v9 static file with combination coefficients that train_and_predict_v9.py does not expose (2026-10-25)

Recombines the stored rolling member probabilities (out\members_v9_static.csv.gz: every day is predicted only with models
trained up to the previous trading day; README_v9 section 8). No training; takes about half a minute (panel build).

  python static_coef_v9.py --w-t 0 --lam 0.4 --lgb-w 0.925 --bias -0.015 --centre top50med --tag static_max

score = w_t * (pbar - 0.5) + (p_e - centre of the day) - bias
  pbar   = day mean over the 100 stocks of p_raw = mean(mean of the 3 raw MLPs, Transformer)           (as v9)
  p_e    = (1 - lgb_w) * mean(kept resid MLPs) + lgb_w * LightGBM                                     (v9: lgb_w 0.5, all 4)
  centre = 'all': mean over the 100 stocks (v9) | 'top50': mean over the 50 | 'top50med': median over the 50
prediction 1 if score > 0; 0 on the chosen ex-dividend flags; then the beta hedge with lambda (rolling9.decide, unchanged).
With the defaults it reproduces out\submission_v9_static.csv (w_t 0.5, lam 0.75); --w-t 0 --lam 0.5 reproduces
out\submission_v9_static_wt0_lam05.csv.

Outputs (out\ or --out): submission_v9_<tag>.csv (upload), probabilities_v9_<tag>.csv, hedge_v9_<tag>.csv,
checks_v9_<tag>.txt, static_report_v9_<tag>.txt. The live run (daily.bat / train_and_predict_v9.py) is not affected.
"""
import os, sys, argparse
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rolling9 as V
import guards as G
G.SHARE1_RANGE = (0.10, 1.00)     # same output limits as train_and_predict_v9.py
G.PBAR_RANGE = (0.20, 0.80)
E = V.E
RAWM = ['mlp_raw_B9', 'mlp_raw_xr9', 'mlp_raw_w4_9']; TF = 'tf_raw_B'
RESM = ['mlp_res_B9', 'mlp_res_xr9', 'mlp_res_BN9', 'mlp_res_xrN9']; LGB = 'lgb_res'

def combine(M, stocks, w_t, lgb_w, keep, centre, bias, frozen=False):
    """same columns as rolling9.combine; equal to it for lgb_w 0.5, all 4 resid MLPs, centre 'all', bias 0"""
    s = '_frozen' if frozen else ''
    col = lambda c: M[c + s]
    out = pd.DataFrame(index=M.index)
    out['p_raw_mlp'] = pd.concat([col(c) for c in RAWM], axis=1).mean(axis=1)
    out['p_raw_transformer'] = col(TF)
    out['p_raw'] = pd.concat([out.p_raw_mlp, out.p_raw_transformer], axis=1).mean(axis=1)
    out['pbar'] = out.groupby(level='date').p_raw.transform('mean')
    out['p_e_lgb'] = col(LGB)
    if keep and lgb_w < 1:
        out['p_e_mlp'] = pd.concat([col(c) for c in keep], axis=1).mean(axis=1)
        if lgb_w == 0.5:
            out['p_e'] = pd.concat([out.p_e_mlp, out.p_e_lgb], axis=1).mean(axis=1)
        else:
            out['p_e'] = (1 - lgb_w) * out.p_e_mlp + lgb_w * out.p_e_lgb
    else:
        out['p_e_mlp'] = np.nan; out['p_e'] = out.p_e_lgb
    if centre == 'all':
        c = out.groupby(level='date').p_e.transform('mean')
    else:
        q = out.p_e[out.index.get_level_values('stock_id').isin(stocks)].groupby(level='date')
        c = (q.mean() if centre == 'top50' else q.median()).reindex(out.index.get_level_values('date')).values
    out['p_e_c'] = out.p_e - c
    out['score'] = w_t * (out.pbar - 0.5) + out.p_e_c - bias
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', default=os.path.join(HERE, 'out', 'members_v9_static.csv.gz'))
    ap.add_argument('--data', default=V.DATA)
    ap.add_argument('--template', default=os.path.join(E.ROOT, 'sample_submission.csv'))
    ap.add_argument('--w-t', type=float, default=V.W_T, help='market view weight (v9 0.5)')
    ap.add_argument('--lam', type=float, default=V.LAM, help='beta hedge: 1 = off (v9 0.75)')
    ap.add_argument('--lgb-w', type=float, default=0.5, help='LightGBM weight in the stock view (v9 0.5)')
    ap.add_argument('--drop', default='', help='resid MLP members to leave out, comma separated, e.g. mlp_res_B9')
    ap.add_argument('--bias', type=float, default=0.0, help='threshold: 1 if w_t*(pbar-0.5) + p_e_c > bias (v9 0; negative = more 1s)')
    ap.add_argument('--ex', default=','.join(V.EX4), help="ex-dividend flags that force 0, comma separated, or 'none' (v9: all 4)")
    ap.add_argument('--centre', default='all', choices=['all', 'top50', 'top50med'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--tag', default='static_coef')
    ap.add_argument('--note', default='', help='text appended to the report')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    keep = [m for m in RESM if m not in [x.strip() for x in a.drop.split(',') if x.strip()]]
    bad = [x for x in a.drop.split(',') if x.strip() and x.strip() not in RESM]
    if bad:
        sys.exit(f'--drop: unknown member(s) {bad}; choose from {RESM}')
    ex = () if a.ex.strip().lower() == 'none' else tuple(x.strip() for x in a.ex.split(',') if x.strip())
    if any(x not in V.EX4 for x in ex):
        sys.exit(f'--ex: choose from {V.EX4} or none')
    out_dir = a.out or os.path.join(HERE, 'out'); os.makedirs(out_dir, exist_ok=True)
    tpl = pd.read_csv(a.template, dtype={'id': str}); stocks = tpl['id'].tolist()
    tpl_dates = [pd.Timestamp(c) for c in tpl.columns if c != 'id']
    M = pd.read_csv(a.members, dtype={'stock_id': str}, parse_dates=['date'])
    M['date'] = M.date.astype('datetime64[ns]'); M = M.set_index(['date', 'stock_id']).sort_index()
    model_dates = sorted(M.index.get_level_values('date').unique())
    core = V.build_core(a.data, tpl_dates)
    core['date'] = core['date'].astype('datetime64[ns]'); core['stock_id'] = core.stock_id.astype(str)

    kw = dict(w_t=a.w_t, lgb_w=a.lgb_w, keep=keep, centre=a.centre, bias=a.bias)
    ens = combine(M, stocks, **kw); ens = ens[ens.index.get_level_values('date').isin(model_dates)]
    X, H = V.decide(ens, core, stocks, lam=a.lam, ex_cols=ex)
    Xf, _ = V.decide(combine(M, stocks, frozen=True, **kw).loc[lambda d: d.index.get_level_values('date').isin(model_dates)], core, stocks, lam=a.lam, ex_cols=ex)

    rep = G.Report()
    miss = X[X.score.isna()]
    if len(miss):
        rep.add('FATAL', f'{len(miss)} Top50 stock-days without a score')
    G.check_output(X[X.date.isin(model_dates)], set(stocks), rep)
    suf = f'_{a.tag}'
    with open(os.path.join(out_dir, f'checks_v9{suf}.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(f'[{l}] {m}' for l, m in rep.items) + '\n')
    if rep.fatal() and not a.force:
        rep.show(print)
        sys.exit('stopped by the FATAL check(s) above: nothing written (rerun with --force after checking)')

    sub = pd.DataFrame({'id': stocks})
    for d in tpl_dates:
        x = X[X.date == d].set_index('stock_id')['pred'].reindex(stocks)
        if x.isna().any():
            sys.exit(f'{d.date()}: some Top50 stocks have no prediction')
        sub[d.strftime('%Y-%m-%d')] = x.values.astype(int)
    vals = sub.drop(columns='id')
    assert list(sub.columns) == list(tpl.columns) and list(sub['id']) == stocks and set(np.unique(vals.values)) <= {0, 1}, 'format check failed'
    sub.to_csv(os.path.join(out_dir, f'submission_v9{suf}.csv'), index=False)
    cols = ['date', 'stock_id', 'p_raw', 'p_raw_mlp', 'p_raw_transformer', 'pbar', 'p_e', 'p_e_mlp', 'p_e_lgb', 'p_e_c', 'score'] + \
           list(V.EX4) + ['beta', 'pred_nohedge', 'pred']
    X[[c for c in cols if c in X]].to_csv(os.path.join(out_dir, f'probabilities_v9{suf}.csv'), index=False, float_format='%.6f')
    if len(H):
        H.to_csv(os.path.join(out_dir, f'hedge_v9{suf}.csv'), index=False, float_format='%.5f')

    st = set(tpl_dates); s_end = max(st); lines = []
    def add(s=''):
        lines.append(s); print(s)
    def summary(name, ar):
        a1 = ar[ar.index.isin(st)]; a2 = ar[ar.index > s_end]
        add(f'{name:<58s} static {a1.mean():6.2f} ({len(a1)} d)   Oct {a2.mean():6.2f} ({len(a2)} d)   all {ar.mean():6.2f} ({len(ar)} d)')
    add('Top50 bp/day = mean of (2 pred - 1) * (close - open) / open; static = 2026-06-01..09-30, Oct = after 09-30 (extension)')
    add(f'coefficients: w_t {a.w_t}, lam {a.lam}, lgb_w {a.lgb_w}, resid MLPs {"+".join(keep) if (keep and a.lgb_w < 1) else "(LightGBM only)"}, '
        f'bias {a.bias}, ex-dividend flags {"+".join(ex) if ex else "none"}, centre {a.centre}')
    add('  (v9 / live: w_t 0.5, lam 0.75, lgb_w 0.5, all 4 resid MLPs, bias 0, all 4 ex-dividend flags, centre all)')
    ar = V.daily_ar(X, core)
    summary('v9 rolling, these coefficients', ar)
    summary('  same, before the hedge', V.daily_ar(X, core, 'pred_nohedge'))
    summary('  same, frozen (all models trained up to 2026-05-29)', V.daily_ar(Xf, core))
    a1 = ar[ar.index.isin(st)]; mo = a1.groupby(a1.index.month).mean()
    add('by month (rolling): ' + '  '.join(f'{pd.Timestamp(2026, m, 1).strftime("%b")} {v:.2f}' for m, v in mo.items()) +
        f'  Oct {ar[ar.index > s_end].mean():.2f};  share of 1 in the file {vals.values.mean():.3f}')
    mate = [f for f in (os.path.join(E.ROOT, 'v6', 'submission-1-phase1_2.csv'), os.path.join(E.ROOT, 'v6', 'submission-1-phase1 (2).csv')) if os.path.exists(f)]
    for name, f in [('v9 static, w 0.5 (live setting)', os.path.join(HERE, 'out', 'submission_v9_static.csv')),
                    ('v9 static, w 0.25', os.path.join(HERE, 'out', 'submission_v9_static_w025.csv')),
                    ('v9 static, wT 0 lam 0.5', os.path.join(HERE, 'out', 'submission_v9_static_wt0_lam05.csv')),
                    ('v4 static upload (rolling, 15.36 on the static leaderboard)', os.path.join(E.ROOT, 'v4', 'out', 'submission_v4_static.csv')),
                    ('groupmate V3 (submission-1-phase1_2.csv)', mate[0] if mate else None)]:
        if f and os.path.exists(f):
            S = pd.read_csv(f, dtype={'id': str}).melt(id_vars='id', var_name='date', value_name='pred').rename(columns={'id': 'stock_id'})
            S['date'] = pd.to_datetime(S.date).astype('datetime64[ns]')
            summary(name, V.daily_ar(S, core))
    if a.note:
        add(); add(a.note)
    with open(os.path.join(out_dir, f'static_report_v9{suf}.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'wrote {os.path.join(out_dir, f"submission_v9{suf}.csv")}: {sub.shape[0]} stocks x {sub.shape[1] - 1} dates, share of 1 = {vals.values.mean():.3f}')

if __name__ == '__main__':
    main()

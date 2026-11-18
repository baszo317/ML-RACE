"""October 2026 (10/01..10/22) as-of predictions from the stored rolling member probabilities.

  python oct_w0.py BASE X1_POOL_CSV OUTDIR

v9 members: BASE/v9/out/members_v9_static.csv.gz (each day only uses models trained up to the previous trading day; README_v9 8)
X1        : X1_POOL_CSV from oct_x1.py (same schedule)
The combination is the live code: rolling9.combine -> (T3 of train_and_predict_v11) -> rolling9.decide -> rolling9.daily_ar.
Checks first: v9 (w_T 0.5, lambda 0.75) and v9 w_T 0 lambda 0.5 must give the published rolling numbers (17.11 / 21.00 and
23.94 / 37.43 for all 99 days / October).
"""
import os, sys
BASE, X1F, OUT = [os.path.abspath(x) for x in sys.argv[1:4]]
sys.path.insert(0, os.path.join(BASE, 'v11')); sys.path.insert(0, os.path.join(BASE, 'v9'))
import numpy as np, pandas as pd
import rolling9 as V
import train_and_predict_v11 as T11
os.makedirs(OUT, exist_ok=True)

M = pd.read_csv(os.path.join(BASE, 'v9', 'out', 'members_v9_static.csv.gz'), dtype={'stock_id': str}, parse_dates=['date'])
M['date'] = M.date.astype('datetime64[ns]'); M = M.set_index(['date', 'stock_id']).sort_index()
X1p = pd.read_csv(X1F, dtype={'stock_id': str}, parse_dates=['date'])
X1p['date'] = X1p.date.astype('datetime64[ns]'); X1p = X1p.set_index(['date', 'stock_id']).p.sort_index()
core = V.build_core(V.DATA, [])
core['date'] = core['date'].astype('datetime64[ns]'); core['stock_id'] = core.stock_id.astype(str)
stocks = pd.read_csv(os.path.join(BASE, 'sample_submission.csv'), dtype={'id': str})['id'].tolist()
all_dates = sorted(M.index.get_level_values('date').unique())
oct_dates = [d for d in all_dates if d >= pd.Timestamp('2026-10-01')]
X1 = dict(T11.X1)

def run(dates, w_t, lam, x1, extra):
    members = [dict(m) for m in V.MEMBERS] + ([X1] if x1 else [])
    probs = {m['name']: M[m['name']] for m in V.MEMBERS}
    if x1:
        probs[X1['name']] = X1p
    ens = V.combine(probs, members, w_t=w_t)
    ens = ens[ens.index.get_level_values('date').isin(dates)].copy()
    if x1:
        assert ens.p_e_lgb.notna().all()
    ms = T11.month_start_days(V.DATA, dates) if extra > 0 else set()
    ens = T11.shift_month_start(ens, core, stocks, ms, extra=extra)
    X, H = V.decide(ens, core, stocks, lam=lam)
    return X, H, V.daily_ar(X, core), sorted(ms)

def show(name, ar):
    o = ar[ar.index >= pd.Timestamp('2026-10-01')]
    print(f'{name:<44s} all {ar.mean():7.2f} ({len(ar)} d)   Oct {o.mean():7.2f} ({len(o)} d)')

print('checks (published: v9 rolling 17.11 / Oct 21.00; v9 wT0 lam0.5 rolling 23.94 / Oct 37.43):')
show('v9 (w_T 0.5, lambda 0.75), recombined', run(all_dates, 0.5, 0.75, False, 0)[2])
show('v9 w_T 0, lambda 0.5, recombined', run(all_dates, 0.0, 0.5, False, 0)[2])
# the stored static files themselves, for the same check
for nm in ('submission_v9_static.csv', 'submission_v9_static_wt0_lam05.csv'):
    S = pd.read_csv(os.path.join(BASE, 'v9', 'out', nm), dtype={'id': str}).melt(id_vars='id', var_name='date', value_name='pred').rename(columns={'id': 'stock_id'})
    S['date'] = pd.to_datetime(S.date).astype('datetime64[ns]')
    show(f'  stored {nm} (6/1-9/30 only)', V.daily_ar(S, core))

X, H, ar, ms = run(oct_dates, 0.0, 0.5, True, 5)
print(f'\nW0 = v11, w_T 0, lambda 0.5 (T3 5 extra 1s on {[str(d.date()) for d in ms]}, X1 on; T2 has no effect when w_T = 0)')
show('W0 October', ar)
for nm, args in (('v9 rolling (live setting)', (0.5, 0.75, False, 0)), ('v9 w_T 0 lambda 0.5', (0.0, 0.5, False, 0)),
                 ('W0 without X1', (0.0, 0.5, False, 5)), ('W0 without T3', (0.0, 0.5, True, 0))):
    show(nm, run(oct_dates, *args)[2])

# per-day table and files in the upload format
Y = X.merge(core[['date', 'stock_id', 'target_r']], on=['date', 'stock_id'], how='left')
rows = []
for d, g in Y.groupby('date'):
    s = 2 * g.pred.values - 1; r = g.target_r.values
    rows.append(dict(date=pd.Timestamp(d).date(), AR_bp=(s * r).mean() * 1e4, timing_bp=s.mean() * r.mean() * 1e4,
                     xsec_bp=((s * r).mean() - s.mean() * r.mean()) * 1e4, share1=g.pred.mean(), hits=int(((g.pred == 1) == (r > 0)).sum()),
                     n=len(g), mkt_bp=r.mean() * 1e4, month_start=int(g.month_start.iloc[0])))
T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, 'w0_october_daily.csv'), index=False, float_format='%.4f')
print('\n' + T.round(2).to_string(index=False))
sub = pd.DataFrame({'id': stocks})
for d in oct_dates:
    sub[pd.Timestamp(d).strftime('%Y-%m-%d')] = X[X.date == d].set_index('stock_id').pred.reindex(stocks).astype(int).values
sub.to_csv(os.path.join(OUT, 'submission_w0_october_asof.csv'), index=False)
cols = ['date', 'stock_id', 'pbar', 'p_e', 'p_e_mlp', 'p_e_lgb', 'p_e_c', 'month_start', 'ms_shift', 'score'] + list(V.EX4) + ['beta', 'pred_nohedge', 'pred']
X[[c for c in cols if c in X]].to_csv(os.path.join(OUT, 'probabilities_w0_october_asof.csv'), index=False, float_format='%.6f')
H.to_csv(os.path.join(OUT, 'hedge_w0_october_asof.csv'), index=False, float_format='%.5f')
print(f'\nwrote {OUT}')

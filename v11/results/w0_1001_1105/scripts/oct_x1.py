"""X1 (lgb_res_xrN) rolling predictions for 2026-10-01..10-22, same schedule as the v9 rolling static run
(rolling9.SCHED['static']['lgb']: a new point every trading day, 1 seed = yymmdd01, pool = last 5 points <= previous day).

  python oct_x1.py BASE OUTDIR [workers]

Only rows dated <= train_end are read by a model (rolling9.train_lgb), so building the panel from today's data is the same
as building it on that day (rows are only appended; checked in rerun_1030 / verify_1029).
"""
import os, sys, time, json
BASE, OUT = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
sys.path.insert(0, os.path.join(BASE, 'v9'))
import numpy as np, pandas as pd
import rolling9 as V
R = V.R

X1 = V.mem('lgb_res_xrN', 'lgb', 'B9+XR+NEW+XRN+SIG', 'resid', train='lgb')
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
core = V.build_core(V.DATA, [])
core['date'] = core['date'].astype('datetime64[ns]'); core['stock_id'] = core.stock_id.astype(str)
fs = V.feature_sets()
print(f'panel {core.shape}, dates {core.date.min().date()}..{core.date.max().date()} ({time.time() - t0:.0f}s)', flush=True)

M = pd.read_csv(os.path.join(BASE, 'v9', 'out', 'members_v9_static.csv.gz'), usecols=['date'])
all_dates = sorted(pd.to_datetime(M.date.unique()))                       # 2026-06-01..10-22 (99 days), as the static run
oct_dates = [d for d in all_dates if d >= pd.Timestamp('2026-10-01')]
units, use, prev = R.static_plan(core, [X1], all_dates, V.SCHED['static'])
need = sorted({t for d in oct_dates for t in use[(X1['name'], d)]})
todo = [u for u in units if u['te'] in need]
assert all(not u['first'] for u in todo), 'an October pool reaches the first point'
print(f'{len(oct_dates)} October days; {len(todo)} X1 models at points {need[0].date()}..{need[-1].date()}', flush=True)
json.dump({str(d.date()): [str(t.date()) for t in use[(X1['name'], d)]] for d in oct_dates}, open(os.path.join(OUT, 'pools.json'), 'w'), indent=1)

def work(u):
    t1 = time.time(); te, s = pd.Timestamp(u['te']), u['seed']
    f = os.path.join(OUT, f"{te.date()}_s{s}.pred.csv")
    if os.path.exists(f):
        return f'{te.date()} s{s}: exists'
    b = V.train_lgb(core, X1, te, [s], fs[X1['feats']])[s]
    ds = [d for d in oct_dates if te in use[(X1['name'], d)]]
    p = V.predict(b, core, X1, ds).rename('p').reset_index()
    p['date'] = pd.to_datetime(p.date).dt.strftime('%Y-%m-%d')
    p.to_csv(f + '.tmp', index=False, float_format='%.9g'); os.replace(f + '.tmp', f)
    with open(os.path.join(OUT, f"{te.date()}_s{s}.model.txt"), 'w') as fh:
        fh.write(b['model'].model_to_string())
    return f"{te.date()} s{s}: best_iter {b['best_epoch']}, {len(ds)} days, {time.time() - t1:.0f}s"

if __name__ == '__main__':
    import multiprocessing as mp
    with mp.get_context('fork').Pool(WORKERS) as pool:
        for i, msg in enumerate(pool.imap_unordered(work, todo), 1):
            print(f'[{i}/{len(todo)}] {msg}; elapsed {(time.time() - t0) / 60:.1f} min', flush=True)
    parts = []
    for u in todo:
        x = pd.read_csv(os.path.join(OUT, f"{pd.Timestamp(u['te']).date()}_s{u['seed']}.pred.csv"), dtype={'stock_id': str})
        parts.append(x.assign(te=pd.Timestamp(u['te'])))
    L = pd.concat(parts, ignore_index=True); L['date'] = pd.to_datetime(L.date)
    U = pd.DataFrame([(d, t) for d in oct_dates for t in use[(X1['name'], d)]], columns=['date', 'te'])
    avg = L.merge(U, on=['date', 'te']).groupby(['date', 'stock_id']).agg(p=('p', 'mean'), n=('p', 'size')).reset_index()
    assert (avg.n == 5).all(), avg.n.value_counts()
    avg['date'] = avg.date.dt.strftime('%Y-%m-%d')
    avg[['date', 'stock_id', 'p']].to_csv(os.path.join(OUT, 'x1_oct_pool.csv'), index=False, float_format='%.9g')
    print(f'wrote {OUT}/x1_oct_pool.csv: {avg.shape[0]} rows; total {(time.time() - t0) / 60:.1f} min', flush=True)

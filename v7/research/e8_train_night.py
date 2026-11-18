"""
e8_train_night.py -- H2 (README_v7 section 6): market-view MLP members with and without the night-session features.
  raw_B_ctrl   : B features                      (= mlp_raw_B, control retrained here)
  raw_xr_ctrl  : B+XR features                   (= mlp_raw_xr, control retrained here)
  raw_B_night  : B + night_ret, night_range
  raw_xr_night : B+XR + night_ret, night_range
raw target (close > open, BCE weighted by min(|r|, 5%)), the same training code as e2 (= wf_runner.train_custom 'orig'),
annual folds + post (frozen at 2026-05-31), seeds 1-5; predictions for the competition stocks.
  python e8_train_night.py --panel ../cache/panel_night.parquet --features ../cache/feature_sets.json --tfdir ../../transformer --out ../cache/wf_night
"""
import os, sys, time, json, argparse, traceback
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import e2_train_ext as E2
_G = {}
NIGHT = ['night_ret', 'night_range']
VARIANTS = {'raw_B_ctrl': ('B', []), 'raw_xr_ctrl': ('B+XR', []), 'raw_B_night': ('B', NIGHT), 'raw_xr_night': ('B+XR', NIGHT)}

def run_unit(u):
    variant, a, b, te, s = u
    f = os.path.join(_G['out'], variant, f'{a}_seed{s}.parquet')
    if os.path.exists(f):
        return u, 0.0, None, None
    t0 = time.time()
    try:
        import torch, tfmodel as TM
        torch.set_num_threads(1)
        base, extra = VARIANTS[variant]; feats = list(_G['fs'][base]) + extra
        Q = _G['P'][_G['P'].date <= max(pd.Timestamp(b), te)]                      # raw target: target_r, y as they are
        bd, n = E2.train_custom(Q[Q.date <= te], feats, te, s)
        a_, b_ = pd.Timestamp(a), pd.Timestamp(b)
        o = TM.predict(bd, Q, lambda d: (d.date >= a_) & (d.date < b_) & d.target_r.notna())[['date', 'stock_id', 'p']]
        os.makedirs(os.path.dirname(f), exist_ok=True); o.to_parquet(f + '.tmp.parquet', index=False); os.replace(f + '.tmp.parquet', f)
        return u, time.time() - t0, None, (n, bd['best_epoch'])
    except Exception:
        return u, time.time() - t0, traceback.format_exc(), None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', default=os.path.join(HERE, '..', 'cache', 'panel_night.parquet'))
    ap.add_argument('--features', default=os.path.join(HERE, '..', 'cache', 'feature_sets.json'))
    ap.add_argument('--tfdir', default=os.path.join(HERE, '..', '..', 'transformer'))
    ap.add_argument('--out', default=os.path.join(HERE, '..', 'cache', 'wf_night'))
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS)); ap.add_argument('--seeds', default='1-5'); ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    sys.path.insert(0, a.tfdir)
    P = pd.read_parquet(a.panel); P['date'] = pd.to_datetime(P.date).astype('datetime64[ns]'); P['stock_id'] = P.stock_id.astype(str)
    _G.update(P=P, out=a.out, fs=json.load(open(a.features)))
    lo, hi = map(int, a.seeds.split('-'))
    todo = [(v, fa, fb, te, s) for v in a.variants for fa, fb, te in E2.FOLDS for s in range(lo, hi + 1)
            if not os.path.exists(os.path.join(a.out, v, f'{fa}_seed{s}.parquet'))]
    print(f'{len(todo)} units, {a.workers} workers', flush=True); t0 = time.time()
    import multiprocessing as mp
    res = mp.get_context('fork').Pool(a.workers).imap_unordered(run_unit, todo) if a.workers > 1 and os.name != 'nt' else map(run_unit, todo)   # Windows: no fork
    for k, (u, dt, err, info) in enumerate(res, 1):
        print(f'[{k}/{len(todo)}] {u[0]} {u[1]} seed{u[4]}: {dt:.0f}s' + (f' rows {info[0]} best_ep {info[1]}' if info else '') + (f'\n{err}' if err else '') + f'  ({time.time() - t0:.0f}s)', flush=True)
    E2.write_means(a.out, a.variants)

if __name__ == '__main__':
    main()

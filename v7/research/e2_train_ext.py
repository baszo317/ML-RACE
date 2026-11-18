"""
e2_train_ext.py -- walk-forward training of the stock-view member mlp_res_B on
  core : the competition Top100 only (control; same as v2/v4/v5)
  ext  : core + point-in-time Top300 extra public stocks (panel_ext.py)
Training is v5/research/wf_runner.train_custom (variant orig) copied verbatim: BCE weighted by min(|resid|, 5%),
expanding window from 2020-03-02, early stopping on the last 120 trading days, refit on the whole window, ens_lib.HP.
Folds: 2022, 2023, 2024, 2025, 2026-01..05 (train up to the day before) + post 2026-06-01.. (frozen at 2026-05-31).
Predictions are written for the competition stocks only: <out>/<variant>/<fold start>_seed<s>.parquet
  python e2_train_ext.py --panel ../cache/panel_ext.parquet --tfdir ../../transformer --out ../cache/wf_ext --workers 2
"""
import os, sys, time, math, json, argparse, traceback
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
HP = dict(d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.2, lr=1e-3, wd=1e-2, bs=1024, epochs=10, patience=2)   # = ens_lib.HP
TRAIN_START = pd.Timestamp('2020-03-02')
FOLDS = [('2022-01-01', '2023-01-01'), ('2023-01-01', '2024-01-01'), ('2024-01-01', '2025-01-01'),
         ('2025-01-01', '2026-01-01'), ('2026-01-01', '2026-06-01')]
FOLDS = [(a, b, pd.Timestamp(a) - pd.Timedelta(days=1)) for a, b in FOLDS] + [('2026-06-01', '2100-01-01', pd.Timestamp('2026-05-31'))]
_G = {}

def train_custom(Q, feats, train_end, seed):
    import torch, tfmodel as TM
    hp = {**dict(d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.2, lr=1e-3, wd=1e-2, bs=512, epochs=15, patience=3), **HP}
    torch.manual_seed(seed); np.random.seed(seed)
    D = TM.SeqData(Q, feats, 1)
    te = pd.Timestamp(train_end)
    tm = ((D.P.date >= TRAIN_START) & (D.P.date <= te)).values & D.P.target_r.notna().values
    scaler = TM.Scaler().fit(D.raw[tm]); D.scale(scaler)
    idx_all = D.valid(tm); dates = D.P.date.values[idx_all]
    ud = np.sort(np.unique(dates)); cut = ud[-120] if len(ud) > 240 else ud[-1]
    y = D.P.y.values.astype(np.float32)
    w = TM._weights(D.P.target_r.fillna(0).values)
    tr, va = idx_all[dates < cut], idx_all[dates >= cut]
    model = TM.make_model('mlp', len(feats), 1, dropout=hp['dropout'])
    model, best_ep = TM._fit(model, D, tr, y[tr], w[tr], hp['epochs'], hp['lr'], hp['wd'], hp['bs'], seed, val=(va, y[va], w[va]), patience=hp['patience'])
    best_ep = max(best_ep, 1)
    torch.manual_seed(seed)
    model = TM.make_model('mlp', len(feats), 1, dropout=hp['dropout'])
    model, _ = TM._fit(model, D, idx_all, y[idx_all], w[idx_all], best_ep, hp['lr'], hp['wd'], hp['bs'], seed)
    return dict(model=model.eval(), scaler=scaler, kind='mlp', seq_len=1, features=feats, hp=hp, best_epoch=best_ep), len(idx_all)

def run_unit(u):
    variant, a, b, te, s = u
    f = os.path.join(_G['out'], variant, f'{a}_seed{s}.parquet')
    if os.path.exists(f):
        return u, 0.0, None, None
    t0 = time.time()
    try:
        import torch, tfmodel as TM
        torch.set_num_threads(1)
        P = _G['P']
        Q = P if variant == 'ext' else P[P.core]
        Q = Q[Q.date <= max(pd.Timestamp(b), te)].copy()
        Q['target_r'] = Q['resid_r']; Q['y'] = Q['y_resid']                              # residual target (ens_lib.target_panel)
        bd, n = train_custom(Q[Q.date <= te], _G['B'], te, s)
        a_, b_ = pd.Timestamp(a), pd.Timestamp(b)
        o = TM.predict(bd, Q, lambda d: (d.date >= a_) & (d.date < b_) & d.target_r.notna() & d.core)[['date', 'stock_id', 'p']]
        os.makedirs(os.path.dirname(f), exist_ok=True); o.to_parquet(f + '.tmp.parquet', index=False); os.replace(f + '.tmp.parquet', f)
        return u, time.time() - t0, None, (n, bd['best_epoch'])
    except Exception:
        return u, time.time() - t0, traceback.format_exc(), None

def init(panel, tfdir, out):
    sys.path.insert(0, tfdir)
    P = pd.read_parquet(panel); P['date'] = pd.to_datetime(P.date).astype('datetime64[ns]'); P['stock_id'] = P.stock_id.astype(str)
    _G['P'] = P; _G['out'] = out
    _G['B'] = json.load(open(os.path.join(tfdir, 'final_feature_lists.json')))['B_competition_only']

def write_means(out, variants):
    """seed average per variant -> <out>/<variant>_mean.parquet (what e3 / e6 read); per-seed columns p_seed<s>"""
    import glob
    for v in variants:
        parts = {}
        for f in sorted(glob.glob(os.path.join(out, v, '*_seed*.parquet'))):
            s = int(os.path.basename(f)[:-8].split('_seed')[1]); parts.setdefault(s, []).append(pd.read_parquet(f))
        if not parts:
            continue
        M = pd.DataFrame({s: pd.concat(x).set_index(['date', 'stock_id']).p for s, x in parts.items()})
        o = M.mean(axis=1).rename('p').reset_index()
        for s in sorted(M.columns):
            o[f'p_seed{s}'] = M[s].values
        o.to_parquet(os.path.join(out, f'{v}_mean.parquet'), index=False)
        print(f'written {v}_mean.parquet: {len(o)} rows, seeds {sorted(M.columns)}', flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', default=os.path.join(HERE, '..', 'cache', 'panel_ext.parquet'))
    ap.add_argument('--tfdir', default=os.path.join(HERE, '..', '..', 'transformer'))
    ap.add_argument('--out', default=os.path.join(HERE, '..', 'cache', 'wf_ext'))
    ap.add_argument('--variants', nargs='+', default=['core', 'ext']); ap.add_argument('--seeds', default='1-5')
    ap.add_argument('--folds', nargs='*', default=None); ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split('-'))
    units = [(v, fa, fb, te, s) for v in a.variants for fa, fb, te in FOLDS if (a.folds is None or fa in a.folds) for s in range(lo, hi + 1)]
    init(a.panel, a.tfdir, a.out)
    todo = [u for u in units if not os.path.exists(os.path.join(a.out, u[0], f'{u[1]}_seed{u[4]}.parquet'))]
    print(f'{len(units)} units, {len(todo)} to do, {a.workers} workers', flush=True)
    t0 = time.time()
    if a.workers <= 1 or os.name == 'nt':                       # Windows has no fork: run one at a time
        res = map(run_unit, todo)
    else:
        import multiprocessing as mp
        res = mp.get_context('fork').Pool(a.workers).imap_unordered(run_unit, todo)
    for k, (u, dt, err, info) in enumerate(res, 1):
        print(f'[{k}/{len(todo)}] {u[0]} {u[1]} seed{u[4]}: {dt:.0f}s' + (f' rows {info[0]} best_ep {info[1]}' if info else '') +
              (f'\n{err}' if err else '') + f'  (elapsed {time.time() - t0:.0f}s)', flush=True)
    write_means(a.out, a.variants)

if __name__ == '__main__':
    main()

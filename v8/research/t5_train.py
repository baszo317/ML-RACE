"""t5_train: walk-forward training for the T5 test (PREREG_m.md, second round).
Stock-view members mlp_res_B (B) and mlp_res_xr (B+XR), seeds 1-5, yearly folds 2022..2026-05 + post (trained to 2026-05-29),
exactly the training of v5/research/wf_runner.py (train_custom with the v2 weights), two targets:
  orig : resid_r = r - mean_100(r)                 (reproduces the cached predictions -> checks that this code = the cache)
  beta : e = r - beta_i * mean_100(r), beta = rolling 120-day intraday beta (days < t), clipped [0.2, 2.5], missing -> day median
Writes ../cache/t5/<member>__<target>/<fold>_seed<s>.pkl; resumable.  run: python t5_train.py [--workers 2]
(on Windows it runs in one process: 120 models, roughly 10-20 minutes; needs torch, as v5's daily training)"""
import os, sys, time, argparse, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); V8 = os.path.dirname(HERE); ROOT = os.path.dirname(V8)
for p in (HERE, os.path.join(ROOT, 'v5'), os.path.join(ROOT, 'transformer')):
    sys.path.insert(0, p)
import numpy as np, pandas as pd
import mlib as L
OUT = os.path.join(L.CACHE, 't5')
FOLDS = [('2022-01-01', '2023-01-01'), ('2023-01-01', '2024-01-01'), ('2024-01-01', '2025-01-01'), ('2025-01-01', '2026-01-01'),
         ('2026-01-01', '2026-06-01'), ('2026-06-01', '2100-01-01')]
MEMBERS = {'mlp_res_B': 'B', 'mlp_res_xr': 'B+XR'}
_G = {}

def build():
    import ens_lib as E
    P = L.panel().copy()
    B = pd.read_pickle(os.path.join(L.CACHE, 'beta_panel.pkl'))[['date', 'stock_id', 'beta']]
    P = P.merge(B, on=['date', 'stock_id'], how='left')
    P['beta'] = P.beta.clip(0.2, 2.5); P['beta'] = P.beta.fillna(P.groupby('date').beta.transform('median')).fillna(1.0)
    P['e_beta'] = P.target_r - P.beta * P.mkt_r
    return P.sort_values(['date', 'stock_id']).reset_index(drop=True), E.feature_sets(), E

def train_custom(Q, feats, train_end, seed, E):
    """= wf_runner.train_custom with alpha None (v2 weights min(|target|, 5%))"""
    import torch, tfmodel as TM
    hp = {**dict(d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.2, lr=1e-3, wd=1e-2, bs=512, epochs=15, patience=3), **E.HP}
    torch.manual_seed(seed); np.random.seed(seed)
    D = TM.SeqData(Q, feats, 1)
    te = pd.Timestamp(train_end)
    tm = ((D.P.date >= E.TRAIN_START) & (D.P.date <= te)).values & D.P.target_r.notna().values
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
    return dict(model=model.eval(), scaler=scaler, kind='mlp', seq_len=1, features=feats, hp=hp, best_epoch=best_ep)

def path(name, target, a, s):
    return os.path.join(OUT, f'{name}__{target}', f'{a}_seed{s}.pkl')

def run_unit(u):
    name, target, a, b, s = u
    f = path(name, target, a, s)
    if os.path.exists(f):
        return u, 0.0, None, None
    t0 = time.time()
    try:
        import torch, tfmodel as TM
        torch.set_num_threads(1)
        P, fs, E = _G['P'], _G['fs'], _G['E']
        te = pd.Timestamp(a) - pd.Timedelta(days=1)
        Q = P[P.date <= max(pd.Timestamp(b), te)].copy()
        if target == 'orig':
            Q['target_r'] = Q.resid_r; Q['y'] = Q.y_resid
        else:
            Q['target_r'] = Q.e_beta; Q['y'] = (Q.e_beta > 0).astype(float).where(Q.e_beta.notna())
        bd = train_custom(Q[Q.date <= te], list(fs[MEMBERS[name]]), te, s, E)
        a_, b_ = pd.Timestamp(a), pd.Timestamp(b)
        o = TM.predict(bd, Q, lambda d: (d.date >= a_) & (d.date < b_) & d.target_r.notna())[['date', 'stock_id', 'p']]
        os.makedirs(os.path.dirname(f), exist_ok=True); o.to_pickle(f + '.tmp'); os.replace(f + '.tmp', f)
        return u, time.time() - t0, None, bd['best_epoch']
    except Exception:
        return u, time.time() - t0, traceback.format_exc(), None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=2); ap.add_argument('--targets', nargs='+', default=['beta', 'orig'])
    a = ap.parse_args()
    _G['P'], _G['fs'], _G['E'] = build()
    units = [(n, t, fa, fb, s) for t in a.targets for n in MEMBERS for fa, fb in FOLDS for s in range(1, 6)]
    todo = [u for u in units if not os.path.exists(path(u[0], u[1], u[2], u[4]))]
    print(f'{len(units)} units, {len(todo)} to do', flush=True)
    import multiprocessing as mp
    t0 = time.time(); eps = []
    if a.workers > 1 and 'fork' in mp.get_all_start_methods():          # Linux / cloud: parallel processes
        pool = mp.get_context('fork').Pool(a.workers); it = pool.imap_unordered(run_unit, todo)
    else:                                                                 # Windows: one process (about 5-10 s per model)
        pool = None; it = map(run_unit, todo)
    for i, (u, sec, err, ep) in enumerate(it, 1):
        if err:
            print('FAILED', u, err, flush=True)
        else:
            eps.append((u[1], ep))
        if i % 10 == 0 or i == len(todo):
            print(f'[{i}/{len(todo)}] {u} {sec:.1f}s, elapsed {(time.time() - t0) / 60:.1f} min', flush=True)
    if pool is not None:
        pool.close(); pool.join()
    if eps:
        E_ = pd.DataFrame(eps, columns=['target', 'best_epoch'])
        print('best epoch (early stopping on the last 120 training days):'); print(E_.groupby('target').best_epoch.value_counts().sort_index().to_string())

if __name__ == '__main__':
    main()

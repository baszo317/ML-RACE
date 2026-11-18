"""v9_train: walk-forward training of the 7 new MLP members of v9 (PREREG_v9.md), seeds 1-10, yearly folds 2022..2026-05 +
post (trained to 2026-05-29).  Training = v5 wf_runner.train_custom with the sigma weighting (alpha 0.5) + an optional window.
Writes ../cache/wf9/<member>/<fold>_seed<s>.pkl; resumable.  run: python v9_train.py [--workers 2] [--seeds 1-10]
(needs torch; Windows runs it in one process, roughly 1-2 hours)"""
import os, sys, time, argparse, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v9lib as V
OUT = os.path.join(V.CACHE, 'wf9')
ALPHA = 0.5
_G = {}

def path(name, a, s):
    return os.path.join(OUT, name, f'{a}_seed{s}.pkl')

def train_custom(Q, feats, start, train_end, seed, sig_col):
    """v5 wf_runner.train_custom, alpha path (weights min(|target|, 5%) / sigma^alpha), with a training start date"""
    import torch, tfmodel as TM, ens_lib as E
    hp = {**dict(d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.2, lr=1e-3, wd=1e-2, bs=512, epochs=15, patience=3), **E.HP}
    torch.manual_seed(seed); np.random.seed(seed)
    D = TM.SeqData(Q, feats, 1)
    te = pd.Timestamp(train_end)
    tm = ((D.P.date >= start) & (D.P.date <= te)).values & D.P.target_r.notna().values
    scaler = TM.Scaler().fit(D.raw[tm]); D.scale(scaler)
    idx_all = D.valid(tm); dates = D.P.date.values[idx_all]
    ud = np.sort(np.unique(dates)); cut = ud[-120] if len(ud) > 240 else ud[-1]
    y = D.P.y.values.astype(np.float32)
    w = np.minimum(np.abs(D.P.target_r.fillna(0).values), 0.05)
    sig = D.P[sig_col].values.astype(float)
    med = np.nanmedian(sig[idx_all])
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, med)
    w = w / sig ** ALPHA
    w = (w / w[idx_all].mean()).astype(np.float32)
    tr, va = idx_all[dates < cut], idx_all[dates >= cut]
    model = TM.make_model('mlp', len(feats), 1, dropout=hp['dropout'])
    model, best_ep = TM._fit(model, D, tr, y[tr], w[tr], hp['epochs'], hp['lr'], hp['wd'], hp['bs'], seed, val=(va, y[va], w[va]), patience=hp['patience'])
    best_ep = max(best_ep, 1)
    torch.manual_seed(seed)
    model = TM.make_model('mlp', len(feats), 1, dropout=hp['dropout'])
    model, _ = TM._fit(model, D, idx_all, y[idx_all], w[idx_all], best_ep, hp['lr'], hp['wd'], hp['bs'], seed)
    return dict(model=model.eval(), scaler=scaler, kind='mlp', seq_len=1, features=feats, hp=hp, best_epoch=best_ep)

def run_unit(u):
    name, a, b, s = u
    f = path(name, a, s)
    if os.path.exists(f):
        return u, 0.0, None, None
    t0 = time.time()
    try:
        import torch, tfmodel as TM, ens_lib as E
        torch.set_num_threads(1)
        m = _G['M'][name]
        P = _G['ext'] if m['panel'] == 'ext' else _G['core']
        te = pd.Timestamp(a) - pd.Timedelta(days=1)
        Q = P[P.date <= max(pd.Timestamp(b), te)].copy()
        if m['target'] == 'resid':
            Q['target_r'] = Q.resid_r; Q['y'] = Q.y_resid
        start = E.window_start(te, m['window'])
        bd = train_custom(Q[Q.date <= te], m['feats'], start, te, s, 'absr_60' if m['target'] == 'raw' else 'absres_60')
        a_, b_ = pd.Timestamp(a), pd.Timestamp(b)
        if m['panel'] == 'ext':
            mask = lambda d: (d.date >= a_) & (d.date < b_) & d.target_r.notna() & d.core
        else:
            mask = lambda d: (d.date >= a_) & (d.date < b_) & d.target_r.notna()
        o = TM.predict(bd, Q, mask)[['date', 'stock_id', 'p']]
        os.makedirs(os.path.dirname(f), exist_ok=True); o.to_pickle(f + '.tmp'); os.replace(f + '.tmp', f)
        return u, time.time() - t0, None, bd['best_epoch']
    except Exception:
        return u, time.time() - t0, traceback.format_exc(), None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=2); ap.add_argument('--seeds', default='1-10')
    ap.add_argument('--members', nargs='*', default=None)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split('-'))
    _G['M'] = V.members(); _G['core'] = V.core_panel(); _G['ext'] = V.ext_panel()
    names = a.members or list(_G['M'])
    # the slow member (expanded universe) first, so that the two workers finish together
    names = sorted(names, key=lambda n: 0 if _G['M'][n]['panel'] == 'ext' else 1)
    units = [(n, fa, fb, s) for n in names for fa, fb in V.FOLDS for s in range(lo, hi + 1)]
    todo = [u for u in units if not os.path.exists(path(*u[:2], u[3]))]
    print(f'{len(units)} units, {len(todo)} to do; core panel {_G["core"].shape}, ext panel {_G["ext"].shape}', flush=True)
    import multiprocessing as mp
    t0 = time.time(); eps = []
    if a.workers > 1 and 'fork' in mp.get_all_start_methods():
        pool = mp.get_context('fork').Pool(a.workers); it = pool.imap_unordered(run_unit, todo)
    else:
        pool = None; it = map(run_unit, todo)
    for i, (u, sec, err, ep) in enumerate(it, 1):
        if err:
            print('FAILED', u, err, flush=True)
        else:
            eps.append((u[0], ep))
        if i % 20 == 0 or i == len(todo):
            print(f'[{i}/{len(todo)}] {u} {sec:.0f}s, elapsed {(time.time() - t0) / 60:.1f} min', flush=True)
    if pool is not None:
        pool.close(); pool.join()
    if eps:
        R = pd.DataFrame(eps, columns=['member', 'best_epoch'])
        print('best epoch (median, max) by member:'); print(R.groupby('member').best_epoch.agg(['median', 'max']).to_string())

if __name__ == '__main__':
    main()

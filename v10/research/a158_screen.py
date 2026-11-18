r"""
a158_screen.py -- training of the v10 experiment (research\PREREG_v10.md). Walk-forward units, resumable (finished units are
skipped, so it can be stopped with Ctrl+C and started again).

  --family lgb : L  = lgb_res with B + one Alpha158 group, configs base + 23 groups + ALL, seed 1..5
  --family mlp : M  = mlp_res_xr9, mlp_res_BN9, mlp_res_xrN9 with their v9 features + one group, same configs, seed 1..5
  --family v10 : stage 2, after a158_eval.py wrote ..\results\v10_selection.json: lgb_res with B + the selected L groups
                 (seed 1..5) and the three MLPs with + the selected M groups (seed 1..10), plus the v9 replica of the MLPs
                 (config base, seed 6..10 added to the screen's 1..5) so that V10 and v9 are compared on the same machine
Output: ..\cache\screen\<family>\<config>\<member>\<fold>_seed<s>.pkl (date, stock_id, p); log ..\results\a158_screen_<family>.log
run (in v10\research): python a158_screen.py --family lgb --workers 4
"""
import os, sys, time, json, argparse, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v10lib as V

_G = {}

def init_worker():
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    _G['P'] = V.panel(log=lambda *a: None)

def run_unit(u):
    family, config, member, fa, fb, seed, feats = u
    f = V.unit_path(family, config, member, fa, seed)
    if V.unit_exists(f):
        return u, 0.0, None, None
    t0 = time.time()
    try:
        P = _G['P']
        if member == 'lgb_res':
            o, best = V.train_lgb_unit(P, feats, fa, fb, seed)
        else:
            o, best = V.train_mlp_unit(P, feats, fa, fb, seed)
        V.save_unit(f, o)
        return u, time.time() - t0, None, best
    except Exception:
        return u, time.time() - t0, traceback.format_exc(), None

def units(family, configs=None, seeds=None):
    out = []
    if family in ('lgb', 'mlp'):
        cfgs = configs or V.CONFIGS
        mems = ['lgb_res'] if family == 'lgb' else V.MLP_MEMBERS
        ss = seeds or range(1, 6)
        for c in cfgs:
            for m in mems:
                feats = V.config_features(m, c)
                for fa, fb in V.FOLDS:
                    for s in ss:
                        out.append((family, c, m, fa, fb, s, feats))
    elif family == 'v10':
        sel = json.load(open(os.path.join(V.RESULTS, 'v10_selection.json')))
        if sel.get('lgb'):
            feats = V.config_features('lgb_res', 'v10', sel['lgb'])
            out += [('v10', 'v10', 'lgb_res', fa, fb, s, feats) for fa, fb in V.FOLDS for s in range(1, 6)]
        if not all(V.unit_exists(V.unit_path('lgb', 'base', 'lgb_res', fa, s)) for fa, _ in V.FOLDS for s in range(1, 6)):
            feats = V.config_features('lgb_res', 'base')                                            # v9 replica (L)
            out += [('lgb', 'base', 'lgb_res', fa, fb, s, feats) for fa, fb in V.FOLDS for s in range(1, 6)]
        for m in V.MLP_MEMBERS:
            if sel.get('mlp'):
                feats = V.config_features(m, 'v10', sel['mlp'])
                out += [('v10', 'v10', m, fa, fb, s, feats) for fa, fb in V.FOLDS for s in range(1, 11)]
            feats = V.config_features(m, 'base')
            out += [('mlp', 'base', m, fa, fb, s, feats) for fa, fb in V.FOLDS for s in range(6, 11)]   # v9 replica
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', choices=['lgb', 'mlp', 'v10', 'all'], default='all')
    ap.add_argument('--workers', type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    ap.add_argument('--configs', nargs='*', default=None)
    ap.add_argument('--seeds', default=None, help='e.g. 1-5')
    ap.add_argument('--limit', type=int, default=None, help='only the first N units still to do (test runs)')
    ap.add_argument('--members', nargs='*', default=None,
                    help='only these members, e.g. --family v10 --members lgb_res (V10 LightGBM trained on the machine of the L screen)')
    a = ap.parse_args()
    seeds = None
    if a.seeds:
        lo, hi = map(int, a.seeds.split('-')); seeds = range(lo, hi + 1)
    fams = ['lgb', 'mlp'] if a.family == 'all' else [a.family]
    os.makedirs(V.RESULTS, exist_ok=True)
    logf = open(os.path.join(V.RESULTS, f'a158_screen_{a.family}.log'), 'a', encoding='utf-8')
    def log(msg):
        line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
        print(line, flush=True); logf.write(line + '\n'); logf.flush()
    P = V.panel(log=log)                                   # builds v10\cache\panel10.pkl the first time
    U = [u for f in fams for u in units(f, a.configs, seeds)]
    if a.members:
        U = [u for u in U if u[2] in a.members]
    todo = [u for u in U if not V.unit_exists(V.unit_path(*u[:4], u[5]))]
    if a.limit:
        todo = todo[:a.limit]
    log(f'family {a.family}: {len(U)} units, {len(todo)} to do, {a.workers} worker(s); panel {P.shape}')
    if not todo:
        log('nothing to do'); return
    import multiprocessing as mp
    del P
    t0 = time.time(); done = 0; busy = 0.0; fails = 0
    if a.workers > 1:
        ctx = mp.get_context(os.environ.get('A158_START') or ('fork' if 'fork' in mp.get_all_start_methods() else 'spawn'))
        pool = ctx.Pool(a.workers, initializer=init_worker)
        it = pool.imap_unordered(run_unit, todo, chunksize=1)
    else:
        init_worker(); pool = None; it = map(run_unit, todo)
    try:
        for u, sec, err, best in it:
            done += 1; busy += sec
            if err:
                fails += 1; log(f'FAILED {u[:6]}\n{err}')
            if done % 10 == 0 or done == len(todo) or done <= 3:
                el = time.time() - t0; eta = el / done * (len(todo) - done)
                log(f'[{done}/{len(todo)}] {u[0]} {u[1]} {u[2]} {u[3]} seed {u[5]}: {sec:.0f}s (best {best}); '
                    f'elapsed {el / 60:.1f} min, about {eta / 60:.0f} min left')
    finally:
        if pool is not None:
            pool.close(); pool.join()
    log(f'finished: {done} units in {(time.time() - t0) / 60:.1f} min, {fails} failed; mean {busy / max(done, 1):.1f}s per unit')

if __name__ == '__main__':
    main()

r"""
v11_train.py -- training units of v11 (research\PREREG_v11.md), same training code as v10 (v10\research\v10lib.py).

  --job base               : the 20-seed baselines: L lgb_res (B) seed 6..20, M (mlp_res_xr9 / BN9 / xrN9, v9 features) seed 11..20,
                             all 6 folds (seed 1..5 of L and 1..10 of M come from v10\cache\screen)
  --job config --family lgb|mlp --groups G1 G2 ... [--folds 2025-01-01 ...] [--seeds 1-5]
                           : one union config (base features + the groups), seed 1..5 by default, all folds by default
  --job list --file jobs.json : several configs at once: [{"family": "lgb", "groups": [...], "folds": [...]}, ...]
Output: ..\cache\screen\<family>\<config>\<member>\<fold>_seed<s>.npz (date, stock_id, p as int64 / U8 / float64), resumable.
Config name: the groups joined by '+' in the order of v10's groups(); configs.json maps names to groups.
run (in v11\research): python v11_train.py --job base --workers 2
"""
import os, sys, time, json, argparse, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); V11 = os.path.dirname(HERE); ROOT = os.path.dirname(V11)
sys.path.insert(0, os.path.join(ROOT, 'v10', 'research'))
import numpy as np, pandas as pd
import v10lib as V

SCREEN11 = os.path.join(V11, 'cache', 'screen')
SCREEN10 = os.path.join(ROOT, 'v10', 'cache', 'screen')
MEMBERS = {'lgb': ['lgb_res'], 'mlp': list(V.MLP_MEMBERS)}
ORDER = list(V.GROUPS)

def config_name(groups):
    gs = [g for g in ORDER if g in set(groups)]
    assert len(gs) == len(set(groups)), groups
    return 'base' if not gs else '+'.join(gs)

def feats(member, groups):
    return V.config_features(member, 'v10', [g for g in ORDER if g in set(groups)]) if groups else V.config_features(member, 'base')

def unit_file(family, config, member, fa, seed):
    """v11 file if it exists, else v10's (base seeds and the single-group configs of the v10 screen)"""
    f11 = V.unit_path(family, config, member, fa, seed, SCREEN11)
    if V.unit_exists(f11):
        return f11
    f10 = V.unit_path(family, config, member, fa, seed, SCREEN10)
    if V.unit_exists(f10):
        return f10
    if family == 'mlp' and config == 'base':                     # v10 stage 2 wrote the replica seeds 6-10 under mlp/base too
        return f10
    return f11

_G = {}
def run_unit(u):
    family, config, member, fa, fb, seed, fl = u
    f = V.unit_path(family, config, member, fa, seed, SCREEN11)
    if V.unit_exists(f) or V.unit_exists(V.unit_path(family, config, member, fa, seed, SCREEN10)):
        return u, 0.0, None, None
    t0 = time.time()
    try:
        P = _G['P']
        if member == 'lgb_res':
            o, best = V.train_lgb_unit(P, fl, fa, fb, seed)
        else:
            o, best = V.train_mlp_unit(P, fl, fa, fb, seed)
        V.save_unit(f, o)
        return u, time.time() - t0, None, best
    except Exception:
        return u, time.time() - t0, traceback.format_exc(), None

def units_for(family, groups, folds=None, seeds=range(1, 6)):
    c = config_name(groups); out = []
    for m in MEMBERS[family]:
        fl = feats(m, groups)
        for fa, fb in V.FOLDS:
            if folds and fa not in folds:
                continue
            for s in seeds:
                out.append((family, c, m, fa, fb, s, fl))
    return out

def register(groups_list):
    f = os.path.join(V11, 'cache', 'configs.json')
    d = json.load(open(f)) if os.path.exists(f) else {}
    for gs in groups_list:
        d[config_name(gs)] = [g for g in ORDER if g in set(gs)]
    os.makedirs(os.path.dirname(f), exist_ok=True)
    json.dump(d, open(f + '.tmp', 'w'), indent=1); os.replace(f + '.tmp', f)

def run(U, workers, log):
    todo = [u for u in U if not (V.unit_exists(V.unit_path(*u[:4], u[5], SCREEN11)) or V.unit_exists(V.unit_path(*u[:4], u[5], SCREEN10)))]
    # slow units first (late folds, LightGBM) so that the workers finish together
    todo.sort(key=lambda u: (u[2] != 'lgb_res', -V.FOLDS.index((u[3], u[4]))))
    log(f'{len(U)} units, {len(todo)} to do, {workers} worker(s)')
    if not todo:
        return 0
    import multiprocessing as mp
    t0 = time.time(); done = fails = 0; busy = 0.0
    if workers > 1:
        pool = mp.get_context('fork').Pool(workers, initializer=_one_thread); it = pool.imap_unordered(run_unit, todo, chunksize=1)
    else:
        _one_thread(); pool = None; it = map(run_unit, todo)
    try:
        for u, sec, err, best in it:
            done += 1; busy += sec
            if err:
                fails += 1; log(f'FAILED {u[:6]}\n{err}')
            if done % 10 == 0 or done == len(todo) or done <= 2:
                el = time.time() - t0
                log(f'[{done}/{len(todo)}] {u[0]} {u[1][:40]} {u[2]} {u[3]} seed {u[5]}: {sec:.0f}s (best {best}); '
                    f'elapsed {el / 60:.1f} min, about {el / done * (len(todo) - done) / 60:.0f} min left')
    finally:
        if pool is not None:
            pool.close(); pool.join()
    log(f'finished {done} units in {(time.time() - t0) / 60:.1f} min, {fails} failed, mean {busy / max(done, 1):.1f}s per unit')
    return fails

def _one_thread():
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', choices=['base', 'config', 'list'], required=True)
    ap.add_argument('--family', choices=['lgb', 'mlp'])
    ap.add_argument('--groups', nargs='*', default=[])
    ap.add_argument('--folds', nargs='*', default=None)
    ap.add_argument('--seeds', default='1-5')
    ap.add_argument('--file', default=None)
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    os.makedirs(os.path.join(V11, 'results'), exist_ok=True)
    logf = open(os.path.join(V11, 'results', 'v11_train.log'), 'a', encoding='utf-8')
    def log(msg):
        line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
        print(line, flush=True); logf.write(line + '\n'); logf.flush()
    lo, hi = map(int, a.seeds.split('-'))
    if a.job == 'base':
        U = units_for('lgb', [], seeds=range(6, 21)) + units_for('mlp', [], seeds=range(11, 21))
        log('job base: L seed 6-20, M seed 11-20, all folds')
    elif a.job == 'config':
        register([a.groups]); U = units_for(a.family, a.groups, a.folds, range(lo, hi + 1))
        log(f'job config {a.family} {config_name(a.groups)} folds {a.folds or "all"} seeds {lo}-{hi}')
    else:
        J = json.load(open(a.file)); register([j['groups'] for j in J]); U = []
        for j in J:
            sl = j.get('seeds', [lo, hi])
            U += units_for(j['family'], j['groups'], j.get('folds'), range(sl[0], sl[1] + 1))
        seen = set(); U = [u for u in U if not ((u[:4] + (u[5],)) in seen or seen.add(u[:4] + (u[5],)))]
        log(f'job list {a.file}: {len(J)} configs')
    _G['P'] = V.panel(log=log)
    fails = run(U, a.workers, log)
    sys.exit(1 if fails else 0)

if __name__ == '__main__':
    main()

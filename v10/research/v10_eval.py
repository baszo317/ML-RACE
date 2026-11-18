r"""
v10_eval.py -- stage 2 of research\PREREG_v10.md: V10 (v9 + the Alpha158 groups selected in ..\results\v10_selection.json)
against the v9 replica trained on the same machine (L: lgb_res B, seed 1-5; M: the three MLPs, seed 1-10).

Pre-registered check: V10 - replica > 0 in hold (2025-01..2026-05) AND in post (2026-06..). val picked the groups (biased up).
Also reported: V10 vs the v9 cache, only-L and only-M ablations, ALL (158 features, screen seeds) for reference, sd, P(19 > 0),
the static months (6/1-9/30) and October.
run (in v10\research, after a158_screen.py --family v10): python v10_eval.py > ..\results\v10_eval.txt
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v10lib as V
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)

def get(family, config, members, seeds):
    x = V.load_units(family, config, members, seeds)
    if x is None:
        agg = os.path.join(V.CACHE, f'screen_{family}_preds.csv.gz')            # seed 1-5 means written by a158_eval.py
        if os.path.exists(agg) and list(seeds) == list(range(1, 6)):
            W = V.read_csv(agg)
            if all(f'{config}|{m}' in W for m in members):
                return {m: W[f'{config}|{m}'].dropna() for m in members}
    return x

def main():
    t0 = time.time()
    sel = json.load(open(os.path.join(V.RESULTS, 'v10_selection.json')))
    W9, TF, LGB, idx = V.load_v9()
    INFO = V.info_table(V.panel(), idx)
    raw9 = [W9.mlp_raw_B9, W9.mlp_raw_xr9, W9.mlp_raw_w4_9]; tf = [TF]
    MM = V.MLP_MEMBERS
    Lb = get('lgb', 'base', ['lgb_res'], range(1, 6))
    Mb = get('mlp', 'base', MM, range(1, 11))
    Lv = get('v10', 'v10', ['lgb_res'], range(1, 6)) if sel['lgb'] else Lb
    Mv = get('v10', 'v10', MM, range(1, 11)) if sel['mlp'] else Mb
    miss = [n for n, x in (('L base', Lb), ('M base seed 1-10', Mb), ('L v10', Lv), ('M v10', Mv)) if x is None]
    if miss:
        sys.exit(f'not finished yet: {miss} (run a158_screen.py --family v10)')
    print(f'V10 groups: L (lgb_res) {sel["lgb"] or "none"}; M (mlp_res_xr9 / BN9 / xrN9) {sel["mlp"] or "none"}')
    print(f'features: lgb_res {len(V.config_features("lgb_res", "v10", sel["lgb"]))} (v9 67); '
          + ', '.join(f'{m} {len(V.config_features(m, "v10", sel["mlp"]))} (v9 {len(V.config_features(m, "base"))})' for m in MM))
    def score(L, M):
        r = [W9.mlp_res_B9] + [M[m].reindex(idx) for m in MM]
        for s in r + [L['lgb_res'].reindex(idx)]:
            assert s.notna().all()
        return V.views([raw9, tf], [r, [L['lgb_res'].reindex(idx)]])
    res9 = [W9.mlp_res_B9, W9.mlp_res_xr9, W9.mlp_res_BN9, W9.mlp_res_xrN9]
    cfg = {'v9 (cache: trained in the cloud for v9)': V.views([raw9, tf], [res9, [LGB]]),
           'v9 replica (same machine as V10)': score(Lb, Mb),
           'V10': score(Lv, Mv),
           'ablation: V10 with only the L change': score(Lv, Mb),
           'ablation: V10 with only the M change': score(Lb, Mv)}
    La, Ma = get('lgb', 'ALL', ['lgb_res'], range(1, 6)), get('mlp', 'ALL', MM, range(1, 6))
    if La is not None and Ma is not None:
        Mb5 = get('mlp', 'base', MM, range(1, 6))
        cfg['reference: ALL 158 in L and M (MLP seed 1-5)'] = score(La, Ma)
        cfg['reference: v9 replica with MLP seed 1-5'] = score(Lb, Mb5)
    R = {k: V.decide(INFO, s) for k, s in cfg.items()}
    base = R['v9 replica (same machine as V10)']; cache = R['v9 (cache: trained in the cloud for v9)']
    rows = []
    for k, D in R.items():
        pr = V.periods(D.index); r19 = D.AR.rolling(19).mean().dropna()
        st = (D.index >= '2026-06-01') & (D.index <= '2026-09-30'); oc = D.index >= '2026-10-01'
        r = {'config': k, 'val': D.AR[pr['val']].mean(), 'hold': D.AR[pr['hold']].mean(), 'post': D.AR[pr['post']].mean(),
             'all': D.AR[pr['all']].mean(), 'sd': D.AR.std(), 'timing': D.timing.mean(), 'xsec': D.xsec.mean(),
             'P19>0': (r19 > 0).mean(), 'P19>20': (r19 > 20).mean(), 'static': D.AR[st].mean(), 'Oct': D.AR[oc].mean()}
        for ref, tag in ((base, ''), (cache, 'c_')):
            dd = D.AR - ref.AR
            for p in ('val', 'hold', 'post', 'all'):
                r[f'{tag}d_{p}'] = dd[pr[p]].mean(); r[f'{tag}t_{p}'] = V.plain_t(dd[pr[p]])
        rows.append(r)
    T = pd.DataFrame(rows)
    print('\nTop50 bp/day, walk-forward 2022-01..2026-10 (yearly retrain; post = models trained to 2026-05-29)')
    print(T[['config', 'val', 'hold', 'post', 'all', 'sd', 'timing', 'xsec', 'P19>0', 'P19>20', 'static', 'Oct']].round(2).to_string(index=False))
    print('\npaired difference to the v9 replica (d_, plain t_)')
    print(T[['config', 'd_val', 't_val', 'd_hold', 't_hold', 'd_post', 't_post', 'd_all', 't_all']].round(2).to_string(index=False))
    print('\npaired difference to the v9 cache (c_d_, c_t_)')
    print(T[['config', 'c_d_val', 'c_t_val', 'c_d_hold', 'c_t_hold', 'c_d_post', 'c_t_post', 'c_d_all', 'c_t_all']].round(2).to_string(index=False))
    r = T[T.config == 'V10'].iloc[0]
    ok = r.d_hold > 0 and r.d_post > 0
    print(f'\npre-registered check (selection used val only): hold diff {r.d_hold:+.2f} (t {r.t_hold:.2f}), post diff {r.d_post:+.2f} '
          f'(t {r.t_post:.2f}) -> ' + ('both > 0: adding the selected Alpha158 groups generalised to the periods not used for picking'
                                        if ok else 'not both > 0: the selected groups did not generalise; keep v9 live'))
    pd.to_pickle(R, os.path.join(V.CACHE, 'v10_daily.pkl'))
    out = pd.DataFrame({'lgb_res10': Lv['lgb_res'].reindex(idx), **{m + '10': Mv[m].reindex(idx) for m in MM},
                        'lgb_res_rep': Lb['lgb_res'].reindex(idx), **{m + '_rep': Mb[m].reindex(idx) for m in MM}})
    out.reset_index().to_csv(os.path.join(V.CACHE, 'wf10_preds.csv.gz'), index=False, float_format='%.7f')
    print(f'done in {time.time() - t0:.0f}s')

if __name__ == '__main__':
    main()

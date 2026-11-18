r"""
a158_eval.py -- evaluation of the v10 screen (research\PREREG_v10.md).

Every config of a family replaces only that family's members (L: lgb_res; M: mlp_res_xr9, mlp_res_BN9, mlp_res_xrN9) in v9's
walk-forward predictions; Top50 bp/day with v9's rule (w_T 0.5, ex-dividend day / day before / 2 days after -> 0, beta hedge
lambda 0.75); paired daily difference to the family's baseline (retrained on the same machine).
Selection (user's rule): val (2022-2024) t > 0; ALL is reported only. When a family is complete the selection is written to
..\results\v10_selection.json.
Works from the unit files (..\cache\screen\) and/or the per-family files ..\cache\screen_<family>_preds.csv.gz (seed means;
written here, so results trained on another machine can be added by copying one file).
run (in v10\research): python a158_eval.py > ..\results\a158_screen_eval.txt
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v10lib as V
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
FAM = {'lgb': (['lgb_res'], range(1, 6)), 'mlp': (V.MLP_MEMBERS, range(1, 6))}

def family_preds(family):
    members, seeds = FAM[family]
    agg = os.path.join(V.CACHE, f'screen_{family}_preds.csv.gz')
    old = V.read_csv(agg) if os.path.exists(agg) else None
    out = {}
    for c in V.CONFIGS:
        x = V.load_units(family, c, members, seeds)
        if x is not None:
            out[c] = x
        elif old is not None and all(f'{c}|{m}' in old for m in members):
            out[c] = {m: old[f'{c}|{m}'].dropna() for m in members}
    if out:
        W = pd.DataFrame({f'{c}|{m}': s for c, d in out.items() for m, s in d.items()})
        if old is None or set(W.columns) != set(old.columns) or len(W) != len(old):
            W.reset_index().to_csv(agg + '.tmp', index=False, float_format='%.7f', compression='gzip'); os.replace(agg + '.tmp', agg)
    return out

def row(name, D, base, nfeat):
    pr = V.periods(D.index); d = D.AR - base.AR.reindex(D.index)
    r = {'config': name, 'nfeat': nfeat}
    for k in ('val', 'hold', 'post', 'all'):
        r[k] = D.AR[pr[k]].mean()
    for k in ('val', 'hold', 'post', 'all'):
        r['d_' + k] = d[pr[k]].mean(); r['t_' + k] = V.plain_t(d[pr[k]])
    r['nw_t_val'] = V.nw_t(d[pr['val']])
    return r

def main():
    t0 = time.time()
    W9, TF, LGB, idx = V.load_v9()
    P = V.panel()
    INFO = V.info_table(P, idx)
    raw9 = [W9.mlp_raw_B9, W9.mlp_raw_xr9, W9.mlp_raw_w4_9]; tf = [TF]
    res9 = [W9.mlp_res_B9, W9.mlp_res_xr9, W9.mlp_res_BN9, W9.mlp_res_xrN9]
    D9 = V.decide(INFO, V.views([raw9, tf], [res9, [LGB]]))
    pr = V.periods(D9.index)
    print(f'v9 from the caches (check against v9_eval.txt 13.99 / 14.33 / 16.59 / 14.31): val {D9.AR[pr["val"]].mean():.2f}, '
          f'hold {D9.AR[pr["hold"]].mean():.2f}, post {D9.AR[pr["post"]].mean():.2f}, all {D9.AR[pr["all"]].mean():.2f}; {len(D9)} days')
    out, daily = {}, {'v9': D9}
    for fam in ('lgb', 'mlp'):
        preds = family_preds(fam)
        if 'base' not in preds:
            print(f'\n{fam}: baseline not finished yet ({len(preds)} configs done)'); continue
        members = FAM[fam][0]
        def score(c):
            ps = {m: preds[c][m].reindex(idx) for m in members}
            for m, s in ps.items():
                assert s.notna().all(), (fam, c, m, int(s.isna().sum()))
            if fam == 'lgb':
                return V.views([raw9, tf], [res9, [ps['lgb_res']]])
            return V.views([raw9, tf], [[W9.mlp_res_B9] + [ps[m] for m in members], [LGB]])
        Db = V.decide(INFO, score('base')); daily[f'{fam}|base'] = Db
        if fam == 'lgb':
            dl = float(np.abs(preds['base']['lgb_res'].reindex(idx) - LGB).max())
            print(f'\nL baseline vs v9 lgb_res (cache, 7 decimals): max abs diff of p {dl:.1e}; AR diff all {(Db.AR - D9.AR).abs().max():.2e} bp max per day')
        else:
            c = {m: float(np.corrcoef(preds['base'][m].reindex(idx), W9[m])[0, 1]) for m in members}
            print(f'\nM baseline (seed 1-5) vs v9 members (seed 1-10, cache): corr {", ".join(f"{m} {v:.4f}" for m, v in c.items())}; '
                  f'AR all {Db.AR[pr["all"]].mean():.2f} vs v9 {D9.AR[pr["all"]].mean():.2f}')
        rows = []
        for c in V.CONFIGS:
            if c not in preds:
                continue
            D = Db if c == 'base' else V.decide(INFO, score(c)); daily[f'{fam}|{c}'] = D
            rows.append(row(c, D, Db, len(V.config_features(members[0], c)) - len(V.config_features(members[0], 'base'))))
        T = pd.DataFrame(rows)
        T['selected'] = (T.config != 'base') & (T.config != 'ALL') & (T.t_val > 0)
        name = {'lgb': 'L = lgb_res (LightGBM, B + group), seed 1-5', 'mlp': 'M = mlp_res_xr9 / BN9 / xrN9 (+ group), seed 1-5'}[fam]
        print(f'\n{name}: Top50 bp/day; d_ / t_ = paired difference to the baseline of the family (plain t; nw_t_val = Newey-West 5 lags)')
        print(T[['config', 'nfeat', 'val', 'hold', 'post', 'all', 'd_val', 't_val', 'nw_t_val', 'd_hold', 't_hold', 'd_post', 't_post',
                 'd_all', 't_all', 'selected']].round(2).to_string(index=False))
        cand = T[(T.config != 'base') & (T.config != 'ALL')]
        sel = cand[cand.selected]
        print(f'{fam}: {len(cand)} of {len(V.GROUPS)} groups evaluated; val t > 0: {len(sel)} ({", ".join(sel.config)})')
        if len(sel):
            print(f'   the selected groups, mean of their single-group differences: val {sel.d_val.mean():+.2f}, hold {sel.d_hold.mean():+.2f}, '
                  f'post {sel.d_post.mean():+.2f}; not selected: val {cand[~cand.selected].d_val.mean():+.2f}, hold {cand[~cand.selected].d_hold.mean():+.2f}, '
                  f'post {cand[~cand.selected].d_post.mean():+.2f}')
            print(f'   corr(t_val, t_hold) over the groups {np.corrcoef(cand.t_val, cand.t_hold)[0, 1]:+.2f}, corr(t_val, t_post) {np.corrcoef(cand.t_val, cand.t_post)[0, 1]:+.2f} '
                  '(about 0 when the val ranking is noise)')
        out[fam] = dict(complete=len(cand) == len(V.GROUPS), groups=list(sel.config), t_val={r.config: round(float(r.t_val), 3) for r in cand.itertuples()})
    pd.to_pickle(daily, os.path.join(V.CACHE, 'a158_daily.pkl'))
    if out and all(v['complete'] for v in out.values()) and len(out) == 2:
        sel = {'rule': 'val (2022-2024) paired t > 0 vs the baseline of the family (PREREG_v10.md)', 'written': time.strftime('%Y-%m-%d %H:%M'),
               'lgb': out['lgb']['groups'], 'mlp': out['mlp']['groups'], 't_val': {k: v['t_val'] for k, v in out.items()}}
        f = os.path.join(V.RESULTS, 'v10_selection.json')
        json.dump(sel, open(f, 'w'), indent=1)
        print(f'\nselection written to {f}: L {sel["lgb"]}; M {sel["mlp"]}')
    else:
        print('\nselection not written yet (both families must be complete)')
    print(f'done in {time.time() - t0:.0f}s')

if __name__ == '__main__':
    main()

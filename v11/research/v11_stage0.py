r"""
v11_stage0.py -- Stage 0 of research\PREREG_v11.md: redundancy of every Alpha158 group with the features each family already
has (no labels, 2020-03-02..2021-12-31). A group is dropped from a family's candidates when the mean over its columns of
R^2_c = max(OLS R^2 on the existing features, max Spearman^2 with one existing feature) is >= 0.90.
run (in v11\research): python v11_stage0.py > ..\results\v11_stage0_redundancy.txt
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v11lib as X
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
THR = 0.90

def main():
    t0 = time.time()
    P = X.V.panel()
    out = {}
    for fam in ('lgb', 'mlp'):
        R = X.redundancy(P, fam)
        G = R.groupby('group', sort=False).agg(n=('col', 'size'), r2_mean=('r2', 'mean'), r2_ols_mean=('r2_ols', 'mean'),
                                                 rho2_mean=('rho_max', lambda s: float((s ** 2).mean())), r2_min=('r2', 'min'), r2_max=('r2', 'max'))
        top = R.loc[R.groupby('group', sort=False).rho_max.idxmax()].set_index('group')
        G['closest'] = top.col.str.replace('a158_', '') + ' ~ ' + top.rho_with + ' (' + top.rho_max.round(3).astype(str) + ')'
        G['dropped'] = G.r2_mean >= THR
        name = {'lgb': 'L (existing = B, 67 features)', 'mlp': f'M (existing = union of the three MLPs, {len(X.base_feature_set("mlp"))} features)'}[fam]
        print(f'\n{name}: R^2 per group (mean over its columns of max(OLS R^2, max Spearman^2)); dropped if mean >= {THR}')
        print(G.round(3).to_string())
        print(f'per column (highest 25 R^2):')
        print(R.sort_values('r2', ascending=False).head(25)[['col', 'group', 'r2', 'r2_ols', 'rho_max', 'rho_with']].round(3).to_string(index=False))
        keep = [g for g in X.ORDER if not G.loc[g, 'dropped']]
        out[fam] = dict(dropped=[g for g in X.ORDER if G.loc[g, 'dropped']], kept=keep, K=len(keep),
                        t_bonferroni=round(X.t_bonferroni(len(keep)), 4),
                        r2_mean={g: round(float(G.loc[g, 'r2_mean']), 4) for g in X.ORDER})
        print(f'{fam}: dropped {out[fam]["dropped"] or "none"}; K = {len(keep)}; Bonferroni one-sided 5%: t > {out[fam]["t_bonferroni"]:.3f}')
        R.to_csv(os.path.join(X.RESULTS, f'v11_stage0_columns_{fam}.csv'), index=False, float_format='%.5f')
    json.dump(out, open(os.path.join(X.CACHE, 'stage0.json'), 'w'), indent=1)
    print(f'\nwritten {os.path.join(X.CACHE, "stage0.json")}; {time.time() - t0:.0f}s')

if __name__ == '__main__':
    main()

r"""
v11_screen.py -- the single-group table of v11 (research\PREREG_v11.md): every Alpha158 group against the 20-seed baseline
(four 5-seed blocks), daily rank IC of the family on all stocks (primary) and Top50 AR in v9's system (secondary), by period.
Also: checks of the baseline blocks, the noise reference (each block against the other three), the scale of the stock view.
run (in v11\research, after v11_train.py --job base): python v11_screen.py > ..\results\v11_screen.txt
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v11lib as X
import v11_metrics as MT
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 200)
VAL = ('2022-01-01', '2025-01-01'); HOLD = ('2025-01-01', '2026-06-01'); POST = ('2026-06-01', '2100-01-01'); ALL = ('2022-01-01', '2100-01-01')

def main():
    t0 = time.time()
    S0 = json.load(open(os.path.join(X.CACHE, 'stage0.json')))
    # ---------------------------------------------------------------- checks of the baseline blocks
    print('baseline blocks (seed 1-5, 6-10, 11-15, 16-20); block 1 must equal the seed 1-5 mean of the v10 screen')
    for fam in ('lgb', 'mlp'):
        B = X.base_blocks(fam)
        for m in X.MEM[fam]:
            s = X.screen(fam)[f'base|{m}'].dropna()
            d = (B[0][m].reindex(s.index) - s).abs().max()
            print(f'  {fam} {m}: max |block1 - screen base| = {d:.1e}; corr of blocks 1..4 with block 1: '
                  + ', '.join(f'{np.corrcoef(B[0][m], B[k][m].reindex(B[0][m].index))[0, 1]:.4f}' for k in range(1, 4)))
        for i in range(4):
            MT.get(fam, 'base', f'b{i + 1}', d=B[i])
    MT.save()
    # ---------------------------------------------------------------- noise reference: block k against the mean of the other three
    print('\nnoise reference: each baseline block against the mean of the other three blocks (no feature changes; NW t)')
    rows = []
    for fam in ('lgb', 'mlp'):
        for k in range(1, 5):
            ic = MT.get(fam, 'base', f'b{k}')['ic']; ar = MT.get(fam, 'base', f'b{k}')['ar']
            oth_ic = pd.concat([MT.get(fam, 'base', f'b{j}')['ic'] for j in range(1, 5) if j != k], axis=1).mean(axis=1)
            oth_ar = pd.concat([MT.get(fam, 'base', f'b{j}')['ar'] for j in range(1, 5) if j != k], axis=1).mean(axis=1)
            dI, dA = (ic - oth_ic).dropna(), (ar - oth_ar).dropna()
            r = {'family': fam, 'block': k}
            for nm, (a, b) in (('val', VAL), ('hold', HOLD), ('post', POST)):
                r[f'dIC_{nm}'] = dI[X.window(dI.index, a, b)].mean(); r[f'tIC_{nm}'] = X.nw_t(dI[X.window(dI.index, a, b)].values)
                r[f'dAR_{nm}'] = dA[X.window(dA.index, a, b)].mean(); r[f'tAR_{nm}'] = X.nw_t(dA[X.window(dA.index, a, b)].values)
            rows.append(r)
    N = pd.DataFrame(rows)
    print(N.round(4).to_string(index=False))
    for fam in ('lgb', 'mlp'):
        n = N[N.family == fam]
        print(f'  {fam}: sd of these t (val / hold / post): IC {n.tIC_val.std():.2f} / {n.tIC_hold.std():.2f} / {n.tIC_post.std():.2f}; '
              f'AR {n.tAR_val.std():.2f} / {n.tAR_hold.std():.2f} / {n.tAR_post.std():.2f}')
    # ---------------------------------------------------------------- single groups
    out = {}
    for fam in ('lgb', 'mlp'):
        bIC = MT.base(fam, 'ic'); bAR = MT.base(fam, 'ar'); bSD = MT.base(fam, 'pe_sd')
        print(f'\n{"L = lgb_res" if fam == "lgb" else "M = mlp_res_xr9 / BN9 / xrN9"}: baseline (mean of 4 blocks) rank IC '
              + ', '.join(f'{p} {bIC[m].mean():.4f}' for p, m in X.period_masks(bIC.index).items())
              + f'; AR val {bAR[X.window(bAR.index, *VAL)].mean():.2f} hold {bAR[X.window(bAR.index, *HOLD)].mean():.2f} '
              f'post {bAR[X.window(bAR.index, *POST)].mean():.2f}')
        rows = []
        for g in X.ORDER + ['ALL']:
            m = MT.get(fam, g)
            dI = MT.delta(fam, g, 'ic'); dA = MT.delta(fam, g, 'ar'); dA1 = MT.delta(fam, g, 'ar', ref='b1')
            dP = MT.delta(fam, g, 'icp')
            r = {'group': g, 'red': '' if g == 'ALL' else ('drop' if g in S0[fam]['dropped'] else ''),
                 'R2': np.nan if g == 'ALL' else S0[fam]['r2_mean'][g]}
            for nm, (a, b) in (('val', VAL), ('hold', HOLD), ('post', POST), ('all', ALL)):
                w = X.window(dI.index, a, b)
                r[f'IC_{nm}'] = dI[w].mean() * 1e3; r[f't_{nm}'] = X.nw_t(dI[w].values)
            for p, mk in X.period_masks(dI.index).items():
                r[f'y{p}'] = dI[mk].mean() * 1e3
            for nm, (a, b) in (('val', VAL), ('hold', HOLD), ('post', POST)):
                w = X.window(dA.index, a, b)
                r[f'AR_{nm}'] = dA[w].mean(); r[f'tAR_{nm}'] = X.nw_t(dA[w].values)
            w = X.window(dA1.index, *VAL); r['tAR_val_v10'] = X.plain_t(dA1[w].values) if hasattr(X, 'plain_t') else X.V.plain_t(dA1[w].values)
            w = X.window(dP.index, *VAL); r['tICp_val'] = X.nw_t(dP[w].values)
            r['sd_ratio'] = (m['pe_sd'] / bSD.reindex(m['pe_sd'].index)).mean()
            rows.append(r)
        T = pd.DataFrame(rows)
        print('ΔIC in 1e-3 (rank IC of the family on all stocks minus the mean of the 4 baseline blocks), NW t; y2022..ypost = ΔIC by period;\n'
              'AR_ = Top50 bp/day difference in v9 (only this family replaced) to the mean of the 4 blocks, NW t; tAR_val_v10 = v10\'s '
              'statistic (plain t against seed 1-5); tICp_val = Pearson IC; sd_ratio = scale of the stock view / baseline; R2 = Stage 0')
        print(T[['group', 'red', 'R2', 'IC_val', 't_val', 'IC_hold', 't_hold', 'IC_post', 't_post', 'IC_all', 't_all',
                 'y2022', 'y2023', 'y2024', 'y2025', 'y2026a', 'ypost']].round(2).to_string(index=False))
        print(T[['group', 'AR_val', 'tAR_val', 'AR_hold', 'tAR_hold', 'AR_post', 'tAR_post', 'tAR_val_v10', 'tICp_val', 'sd_ratio']].round(2).to_string(index=False))
        c = T[T.group != 'ALL']
        print(f'  {fam}: sd of the group t (val / hold / post): {c.t_val.std():.2f} / {c.t_hold.std():.2f} / {c.t_post.std():.2f}; '
              f'mean t {c.t_val.mean():+.2f} / {c.t_hold.mean():+.2f} / {c.t_post.mean():+.2f}; '
              f'corr(t_val, t_hold) {np.corrcoef(c.t_val, c.t_hold)[0, 1]:+.2f}, corr(t_val, t_post) {np.corrcoef(c.t_val, c.t_post)[0, 1]:+.2f}, '
              f'corr(t_hold, t_post) {np.corrcoef(c.t_hold, c.t_post)[0, 1]:+.2f}; corr(IC t, AR t) on val {np.corrcoef(c.t_val, c.tAR_val)[0, 1]:+.2f}')
        out[fam] = T
    MT.save()
    pd.to_pickle(out, os.path.join(X.CACHE, 'screen_table.pkl'))
    print(f'\ndone in {time.time() - t0:.0f}s')

if __name__ == '__main__':
    main()

r"""
v11_final.py -- v11 against v9 in the walk-forward (research\PREREG_v11.md section 7): both families changed as
..\results\v11_selection.json says (a family without groups keeps v9's features), every period, against the baseline blocks.

Pairing: v11 = (L_v11 or L block k, M_v11 or M block k), baseline = (L block k, M block k), averaged over the 4 blocks k, so a
family that does not change cancels exactly and the changed family is compared at the same ensemble size (5 seeds).
The final groups were chosen on all walk-forward days, so every period here is in-sample for the choice (report only).
run (in v11\research, after v11_rules.py wrote the selection): python v11_final.py > ..\results\v11_final.txt
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v11lib as X
import v11_metrics as MT
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)

def main():
    t0 = time.time()
    sel = json.load(open(os.path.join(X.RESULTS, 'v11_selection.json')))
    G = {f: sel[f].get('groups') or [] for f in ('lgb', 'mlp')}
    print(f'v11 groups: L (lgb_res) {G["lgb"] or "none"}; M (mlp_res_xr9 / BN9 / xrN9) {G["mlp"] or "none"}')
    if not G['lgb'] and not G['mlp']:
        print('no Alpha158 group selected: v11 = v9, nothing to compare'); return
    ctx = MT.context(); AR = ctx['AR']
    BL, BM = X.base_blocks('lgb'), X.base_blocks('mlp')
    L11 = X.config_preds('lgb', G['lgb']) if G['lgb'] else None
    M11 = X.config_preds('mlp', G['mlp']) if G['mlp'] else None
    rows, d_v11, d_base = [], [], []
    for k in range(4):
        Dv = AR.ar(L=L11 or BL[k], M=M11 or BM[k], ret_all=True)
        Db = AR.ar(L=BL[k], M=BM[k], ret_all=True)
        d_v11.append(Dv); d_base.append(Db)
    V11 = sum(d_v11) / 4; B = sum(d_base) / 4                    # daily AR, timing, m, xsec averaged over the blocks
    W9 = AR.ar(ret_all=True)                                     # v9 as trained in the cloud for v9 (cache)
    dAR = V11.AR - B.AR
    per = {'val': ('2022-01-01', '2025-01-01'), 'hold': ('2025-01-01', '2026-06-01'), 'post': ('2026-06-01', '2100-01-01'),
           'static 6/1-9/30': ('2026-06-01', '2026-10-01'), 'Oct': ('2026-10-01', '2100-01-01'), 'all': ('2022-01-01', '2100-01-01')}
    for nm, (a, b) in per.items():
        w = X.window(dAR.index, a, b)
        rows.append({'period': nm, 'days': int(w.sum()), 'v9 (cache)': W9.AR[w].mean(), 'v9 replica (blocks)': B.AR[w].mean(),
                     'v11': V11.AR[w].mean(), 'diff': dAR[w].mean(), 'NW t': X.nw_t(dAR[w].values), 'plain t': X.V.plain_t(dAR[w].values)})
    print('\nTop50 bp/day in v9\'s system (w_T 0.5, ex-dividend 4 days, hedge 0.75); v11 and the replica are means over the 4 seed blocks')
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    for y in (2022, 2023, 2024, 2025, 2026):
        w = pd.DatetimeIndex(dAR.index).year == y
        print(f'  {y}: diff {dAR[w].mean():+.2f} (t {X.nw_t(dAR[w].values):.2f})')
    r19v, r19b = V11.AR.rolling(19).mean().dropna(), B.AR.rolling(19).mean().dropna()
    print(f'daily sd: v11 {V11.AR.std():.1f}, replica {B.AR.std():.1f}; P(19-day mean > 0): v11 {(r19v > 0).mean():.3f}, replica {(r19b > 0).mean():.3f}; '
          f'timing v11 {V11.timing.mean():.2f} / replica {B.timing.mean():.2f}; xsec v11 {V11.xsec.mean():.2f} / replica {B.xsec.mean():.2f}')
    for f, d in (('lgb', L11), ('mlp', M11)):
        if d is None:
            continue
        name = '+'.join(G[f])
        m = MT.get(f, name) if len(G[f]) == 1 else MT.get(f, name, 's1-5', d=d)
        dI = (m['ic'] - MT.base(f, 'ic').reindex(m['ic'].index)).dropna()
        print(f'{f} rank IC difference (1e-3): ' + ', '.join(f'{nm} {dI[X.window(dI.index, a, b)].mean() * 1e3:+.2f} (t {X.nw_t(dI[X.window(dI.index, a, b)].values):.2f})'
                                                       for nm, (a, b) in per.items() if nm in ('val', 'hold', 'post', 'all')))
    MT.save()
    pd.to_pickle({'v11': V11, 'replica': B, 'v9': W9}, os.path.join(X.CACHE, 'final_daily.pkl'))
    print(f'done in {time.time() - t0:.0f}s')

if __name__ == '__main__':
    main()

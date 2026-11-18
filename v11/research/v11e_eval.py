r"""
v11e_eval.py -- X5 lgb_rank on top of v11 (research\PREREG_v11e.md): the user's rule (val AR difference > 0) and the
pre-registered check (hold and post differences > 0), plus reports (ICs, diversity, sd, P(19 > 0), static, October, v9 + X5).
run (in v11\research, after v11e_train.py): python v11e_eval.py > ..\results\v11e_eval.txt
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v11b_lib as L
import v11b_train as TB
V = L.V
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
PER = {'val': ('2022-01-01', '2025-01-01'), 'hold': ('2025-01-01', '2026-06-01'), 'post': ('2026-06-01', '2100-01-01'),
       'all': ('2022-01-01', '2100-01-01'), 'static': ('2026-06-01', '2026-10-01'), 'Oct': ('2026-10-01', '2100-01-01')}

def w(ix, k):
    a, b = PER[k]; ix = pd.DatetimeIndex(ix)
    return (ix >= pd.Timestamp(a)) & (ix < pd.Timestamp(b))

def main():
    t0 = time.time()
    d = L.Data()
    x5 = TB.load_member('lgb_rank')
    if x5 is None:
        sys.exit('X5 predictions missing: run v11e_train.py first')
    x5 = x5.reindex(d.idx); assert x5.notna().all(), int(x5.isna().sum())
    dt = lambda s: s.index.get_level_values('date')
    c = lambda s: s - s.groupby(level='date').transform('mean')
    sdd = lambda s: s.groupby(level='date').transform('std')
    x5s = d.LGB.groupby(level='date').transform('mean') + c(x5) * sdd(c(d.LGB)) / sdd(c(x5))

    def view(lgbs):
        return c((pd.concat(d.mlp, axis=1).mean(axis=1) + pd.concat(lgbs, axis=1).mean(axis=1)) / 2)
    def run(pe, night, ms):
        pb = d.pbar(night=night)
        s = L.W_T * (pb.reindex(dt(pe)).values - 0.5) + pe
        return d.decide(s, month_start=ms)
    pe = {'v9': view([d.LGB]), 'v9 + X5': view([d.LGB, x5s]), 'v11': view([d.LGB, d.x1]), 'v11 + X5': view([d.LGB, d.x1, x5s])}
    D = {'v9': run(pe['v9'], False, False), 'v9 + X5': run(pe['v9 + X5'], False, False),
         'v11': run(pe['v11'], True, True), 'v11 + X5': run(pe['v11 + X5'], True, True)}
    chk = {k: D['v9'].AR[w(D['v9'].index, k)].mean() for k in ('val', 'hold', 'post', 'all')}
    chk11 = {k: D['v11'].AR[w(D['v11'].index, k)].mean() for k in ('val', 'hold', 'post', 'all')}
    print(f'rebuilt: v9 {chk["val"]:.2f} / {chk["hold"]:.2f} / {chk["post"]:.2f} / {chk["all"]:.2f} (13.99 / 14.33 / 16.59 / 14.31); '
          f'v11 {chk11["val"]:.2f} / {chk11["hold"]:.2f} / {chk11["post"]:.2f} / {chk11["all"]:.2f} (14.60 / 14.34 / 20.35 / 15.02)')
    # X5 itself: IC and diversity
    ic = lambda p: d.ic(c(p))
    I = {'lgb_res': ic(d.LGB), 'lgb_res_xrN (X1)': ic(d.x1), 'lgb_rank (X5)': ic(x5)}
    print('\nown rank IC on all stocks (val / hold / post):')
    for k, s in I.items():
        print(f'  {k:18s} ' + ' / '.join(f'{s[w(s.index, p)].mean():.4f}' for p in ('val', 'hold', 'post')))
    cc = lambda a, b: np.corrcoef(c(a).values, c(b).values)[0, 1]
    print(f'correlation of the day-centred predictions: X5 with lgb_res {cc(x5, d.LGB):.3f}, X5 with X1 {cc(x5, d.x1):.3f}, '
          f'X1 with lgb_res {cc(d.x1, d.LGB):.3f}, X5 with the MLP mean {cc(x5, pd.concat(d.mlp, axis=1).mean(axis=1)):.3f}')
    rows = []
    for k, Dk in D.items():
        r = {'config': k}
        for p in ('val', 'hold', 'post', 'all', 'static', 'Oct'):
            r[p] = Dk.AR[w(Dk.index, p)].mean()
        r19 = Dk.AR.rolling(19).mean().dropna(); r['sd'] = Dk.AR.std(); r['P19>0'] = (r19 > 0).mean(); r['P19>20'] = (r19 > 20).mean()
        icp = d.ic(pe[k])
        for p in ('val', 'hold', 'post'):
            r[f'IC_{p}'] = icp[w(icp.index, p)].mean()
        rows.append(r)
    T = pd.DataFrame(rows)
    print('\nTop50 bp/day (walk-forward) and the stock view\'s rank IC')
    print(T.round(4).to_string(index=False))
    out = {}
    for base, cand in (('v11', 'v11 + X5'), ('v9', 'v9 + X5')):
        dd = (D[cand].AR - D[base].AR).dropna()
        r = {}
        print(f'\n{cand} minus {base} (bp/day, NW t):')
        for p in ('val', 'hold', 'post', 'all', 'static', 'Oct'):
            x = dd[w(dd.index, p)]; r[p] = (float(x.mean()), float(V.nw_t(x.values)))
            print(f'  {p:6s} {x.mean():+.2f} ({V.nw_t(x.values):.2f})')
        print('  by year: ' + ', '.join(f'{y} {dd[pd.DatetimeIndex(dd.index).year == y].mean():+.2f}' for y in range(2022, 2027)))
        out[cand] = r
    v = out['v11 + X5']
    sel = v['val'][0] > 0
    ok = sel and v['hold'][0] > 0 and v['post'][0] > 0
    print(f'\nuser\'s rule: val {v["val"][0]:+.2f} -> ' + ('selected' if sel else 'not selected (val <= 0): v11 stays as it is'))
    if sel:
        print(f'pre-registered check: hold {v["hold"][0]:+.2f} (t {v["hold"][1]:.2f}), post {v["post"][0]:+.2f} (t {v["post"][1]:.2f}) -> '
              + ('both > 0: add X5 to v11\'s daily program' if ok else 'not both > 0: X5 is not added; v11 stays as it is'))
    json.dump({'selected': bool(sel), 'passed': bool(ok), 'v11 + X5 vs v11': out['v11 + X5'], 'v9 + X5 vs v9': out['v9 + X5']},
              open(os.path.join(L.V11, 'results', 'v11e_selection.json'), 'w'), indent=1)
    pd.to_pickle(D, os.path.join(L.V11, 'cache', 'v11e_daily.pkl'))
    print(f'\ndone in {time.time() - t0:.0f}s')

if __name__ == '__main__':
    main()

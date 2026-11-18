r"""
v11b_eval.py -- evaluation of v11 part B (research\PREREG_v11b.md): the 7 candidates one at a time on top of v9, the user's
rule (val AR difference > 0), v11 = v9 + every selected change, the pre-registered check (hold and post differences > 0),
and reports (timing correlation, stock-view IC, sd, P(19 days > 0), static period, October, ablation).
run (in v11\research, after v11b_train.py): python v11b_eval.py > ..\results\v11b_eval.txt
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v11b_lib as L
V = L.V
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 100)
PER = {'val': ('2022-01-01', '2025-01-01'), 'hold': ('2025-01-01', '2026-06-01'), 'post': ('2026-06-01', '2100-01-01'),
       'all': ('2022-01-01', '2100-01-01'), 'static': ('2026-06-01', '2026-10-01'), 'Oct': ('2026-10-01', '2100-01-01')}
CANDS = ['T1', 'T2', 'T3', 'X1', 'X2', 'X3', 'X4']
NAMES = {'T1': 'pbar over the Top50', 'T2': 'night-session reversal', 'T3': 'month start +5 ones', 'X1': 'second LightGBM (xrN9 features)',
         'X2': 'LightGBM with return-size label', 'X3': 'z-score combination', 'X4': 'smoothing across days'}

def w(ix, k):
    a, b = PER[k]; ix = pd.DatetimeIndex(ix)
    return (ix >= pd.Timestamp(a)) & (ix < pd.Timestamp(b))

def corr(a, b, mask):
    x = pd.concat([a, b], axis=1).dropna()
    x = x[w(x.index, mask)]
    return np.corrcoef(x.iloc[:, 0], x.iloc[:, 1])[0, 1]

def describe(name, D, D0, pb, pe, data, ics=True):
    d = (D.AR - D0.AR.reindex(D.index)).dropna()
    r = {'config': name}
    for k in ('val', 'hold', 'post', 'all'):
        r[k] = D.AR[w(D.index, k)].mean()
    for k in ('val', 'hold', 'post', 'all'):
        x = d[w(d.index, k)]
        r[f'd_{k}'] = x.mean(); r[f't_{k}'] = V.nw_t(x.values)
    r['d_static'] = d[w(d.index, 'static')].mean(); r['d_Oct'] = d[w(d.index, 'Oct')].mean()
    r['sd'] = D.AR.std(); r19 = D.AR.rolling(19).mean().dropna(); r['P19>0'] = (r19 > 0).mean(); r['P19>20'] = (r19 > 20).mean()
    r['timing'] = D.timing.mean(); r['xsec'] = D.xsec.mean()
    for k in ('val', 'hold', 'post'):
        r[f'corr_pbar_m_{k}'] = corr(pb, D.m, k); r[f'corr_sbar_m_{k}'] = corr(D.sbar, D.m, k)
    if ics:
        ic = data.ic(pe); ic50 = data.ic(pe, top50=True)
        for k in ('val', 'hold', 'post'):
            r[f'IC100_{k}'] = ic[w(ic.index, k)].mean(); r[f'IC50_{k}'] = ic50[w(ic50.index, k)].mean()
    return r

def main():
    t0 = time.time()
    data = L.Data()
    if data.x1 is None or data.x2 is None:
        sys.exit('X1 / X2 predictions missing: run v11b_train.py first')
    for nm, s in (('X1', data.x1), ('X2', data.x2)):
        assert s.notna().all(), (nm, int(s.isna().sum()))
    D0, pb0, pe0 = data.run([])
    pr = {k: D0.AR[w(D0.index, k)].mean() for k in ('val', 'hold', 'post', 'all')}
    print(f'v9 rebuilt: val {pr["val"]:.2f}, hold {pr["hold"]:.2f}, post {pr["post"]:.2f}, all {pr["all"]:.2f} '
          f'(v9_eval.txt: 13.99 / 14.33 / 16.59 / 14.31); {len(D0)} days')
    rows = [describe('v9', D0, D0, pb0, pe0, data)]
    daily = {'v9': D0}
    for c in CANDS:
        D, pb, pe = data.run([c]); daily[c] = D
        rows.append(describe(f'{c} {NAMES[c]}', D, D0, pb, pe, data, ics=c.startswith('X')))
        print(f'  {c} done ({time.time() - t0:.0f}s)', flush=True)
    T = pd.DataFrame(rows)
    print('\nsingle changes on top of v9 (Top50 bp/day; d_ = paired difference to v9, NW t)')
    print(T[['config', 'val', 'hold', 'post', 'all', 'd_val', 't_val', 'd_hold', 't_hold', 'd_post', 't_post', 'd_all', 't_all',
             'd_static', 'd_Oct']].round(2).to_string(index=False))
    print('\nrisk and decomposition')
    print(T[['config', 'sd', 'P19>0', 'P19>20', 'timing', 'xsec']].round(3).to_string(index=False))
    print('\nmarket view: corr(pbar, m) and corr(net exposure, m) by period (m = Top50 mean intraday return)')
    print(T[['config'] + [f'corr_pbar_m_{k}' for k in ('val', 'hold', 'post')] + [f'corr_sbar_m_{k}' for k in ('val', 'hold', 'post')]].round(3).to_string(index=False))
    print('\nstock view: mean daily rank IC of the centred p_e with the intraday return (all ~100 stocks / Top50)')
    Ti = T[T.config.str.startswith('X') | (T.config == 'v9')]
    print(Ti[['config'] + [f'IC100_{k}' for k in ('val', 'hold', 'post')] + [f'IC50_{k}' for k in ('val', 'hold', 'post')]].round(4).to_string(index=False))
    sel = [c for c, r in zip(CANDS, rows[1:]) if r['d_val'] > 0]
    print(f'\nuser\'s rule (val AR difference > 0): selected {sel or "none"}')
    res = {'selected': sel, 'single': {c: {k: round(float(r[k]), 4) for k in ('d_val', 't_val', 'd_hold', 't_hold', 'd_post', 't_post')}
                                      for c, r in zip(CANDS, rows[1:])}}
    if sel:
        Dv, pbv, pev = data.run(sel); daily['v11'] = Dv
        rv = describe('v11 = v9 + ' + ' + '.join(sel), Dv, D0, pbv, pev, data)
        abl = []
        for c in sel:
            rest = [x for x in sel if x != c]
            Da, pba, pea = data.run(rest)
            ra = describe(f'v11 without {c}', Da, D0, pba, pea, data)
            ra['contribution_all'] = rv['all'] - ra['all']
            abl.append(ra)
        A = pd.DataFrame([rv] + abl)
        print('\nv11 and the ablation (each row: difference to v9; only reported, not used to change v11)')
        print(A[['config', 'val', 'hold', 'post', 'all', 'd_val', 't_val', 'd_hold', 't_hold', 'd_post', 't_post', 'd_all', 't_all',
                 'd_static', 'd_Oct', 'sd', 'P19>0', 'P19>20']].round(2).to_string(index=False))
        print(A[['config', 'timing', 'xsec', 'corr_pbar_m_val', 'corr_pbar_m_hold', 'corr_pbar_m_post', 'IC100_val', 'IC100_hold', 'IC100_post']].round(3).to_string(index=False))
        ok = rv['d_hold'] > 0 and rv['d_post'] > 0
        print(f'\npre-registered check: hold {rv["d_hold"]:+.2f} (t {rv["t_hold"]:.2f}), post {rv["d_post"]:+.2f} (t {rv["t_post"]:.2f}) -> '
              + ('both > 0: v11 holds; build the daily program (parallel to v9)' if ok else 'not both > 0: v11 fails; live stays v9'))
        res.update(v11={k: round(float(rv[k]), 4) for k in ('val', 'hold', 'post', 'all', 'd_val', 't_val', 'd_hold', 't_hold', 'd_post',
                                                              't_post', 'd_all', 't_all', 'd_static', 'd_Oct', 'sd', 'P19>0')}, passed=bool(ok))
        d = (Dv.AR - D0.AR).dropna()
        print('by year (v11 - v9): ' + ', '.join(f'{y} {d[pd.DatetimeIndex(d.index).year == y].mean():+.2f}' for y in range(2022, 2027)))
    pd.to_pickle(daily, os.path.join(L.V11, 'cache', 'v11b_daily.pkl'))
    json.dump(res, open(os.path.join(L.V11, 'results', 'v11b_selection.json'), 'w'), indent=1)
    print(f'\ndone in {time.time() - t0:.0f}s')

if __name__ == '__main__':
    main()

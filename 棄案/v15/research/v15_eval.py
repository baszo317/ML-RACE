r"""
v15_eval.py -- judgment and reports of v15 (research\PREREG_v15.md sections 3 and 4). numpy + pandas only.

reads  ..\cache\units\<name>\*.npz (v15_train.py), ..\cache\signals15.pkl (v15_signals.py), and the cached walk-forward
       predictions of v8 / v9 / v11 (through v15lib.Data, = v14lib.Data)
writes ..\results\v15_daily.csv (every configuration, every day), ..\results\v15_selection.json; the report goes to stdout
usage (in v15\research):  python v15_eval.py [--test]  > ..\results\v15_eval.txt
  --test   use ..\cache\units_test (tiny models of run_v15.bat test): checks the program only, the numbers mean nothing
"""
import os, sys, json, math, argparse, time
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v15lib as L
from v15lib import c, sdd, per, w, nw_t, ic_daily, scale_to, zday, with_families, diff, fmt, power, PER, YEARS, OCT, PUBLISHED

A_C = ['A1', 'A2']
B_C = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']
C_C = ['C1', 'C2']
D_C = ['D1', 'D2']
E_C = ['E1']
CAND = A_C + B_C + C_C + D_C + E_C


def kday(resid, k):
    """mean excess return of the same stock over D .. D+k-1 (only for the report of B3 / B4)"""
    M = resid.unstack('stock_id').sort_index()
    R = sum(M.shift(-j) for j in range(k)) / k
    R.index.name = 'date'; R.columns.name = 'stock_id'
    out = R.reset_index().melt(id_vars='date', var_name='stock_id', value_name='r')
    out['stock_id'] = out.stock_id.astype(str)
    return out.set_index(['date', 'stock_id']).r.reindex(resid.index)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--test', action='store_true')
    a = ap.parse_args(); test = a.test
    t0 = time.time()
    print('v15_eval.py' + ('  [TEST: tiny models from run_v15.bat test -- the numbers below mean nothing]' if test else ''))
    print(f'  {time.strftime("%Y-%m-%d %H:%M")}, pandas {pd.__version__}, numpy {np.__version__}')
    d = L.Data(); idx = d.idx

    def get(name, seeds=None):
        s = L.load_member(name, seeds=seeds, test=test)
        if s is None:
            print(f'  {name}: units missing -> skipped (python v15_train.py --members {name})')
            return None
        x = s.reindex(idx); miss = int(x.isna().sum())
        if miss:
            x = x.fillna(x.groupby(level='date').transform('mean'))
            print(f'  {name}: {miss} of {len(idx):,} rows missing (filled with the day mean)')
        return x
    print('\nmembers:')
    M = {n: get(n) for n in L.ORDER}
    B0_3 = get('B0', seeds=(1, 2, 3))
    for n in L.ORDER:
        if M[n] is not None:
            ms = L.member_metas(n, test)
            print(f'  {n:3s} {L.MEMBERS[n]["label"]:42s} {len(ms):3d} units, {sum(m["seconds"] for m in ms) / 60:6.1f} unit-minutes')
    sf = os.path.join(L.CACHE, 'signals15.pkl')
    SG = pd.read_pickle(sf) if os.path.exists(sf) else None
    if SG is None:
        print('  signals15.pkl missing -> D1, D2, E1 skipped (python v15_signals.py)')
        sigs = {}; betaV = None
    else:
        if not SG.get('leak_ok', False):
            print('  WARNING: the leak check of v15_signals.py failed: D1 / D2 / E1 are not judged')
        sigs = {k: SG['S'][k].reindex(idx) for k in D_C}
        betaV = SG['BV'].beta_V.clip(0.2, 2.5)

    # ------------------------------------------------------------ part 0
    print('\nPART 0 -- rebuild check (val / hold / post / all, bp/day)')
    D = {'v9': d.run(d.view([d.LGB]), False), 'v11': d.run(d.view([d.LGB, d.x1]), True),
         'W0': d.run(d.view([d.LGB, d.x1]), True, w_t=0.0, lam=0.5)}
    ok = True
    for k, ref in PUBLISHED.items():
        got = tuple(round(per(D[k].AR)[p], 2) for p in PER)
        same = all(abs(x - y) <= 0.005 for x, y in zip(got, ref)); ok &= same
        print(f'  {k:4s} rebuilt {" / ".join(f"{x:.2f}" for x in got)}   published {" / ".join(f"{x:.2f}" for x in ref)}   '
              + ('same' if same else 'DIFFERENT'))
    if not ok:
        print('  WARNING: the rebuild differs from the published numbers: find the cause before deciding')
    print('  replicas (same program, this computer):')
    if M['B0'] is not None:
        dd = (M['B0'] - d.LGB).abs()
        print(f'    B0 vs cached lgb_res (seed 1-5): max |p diff| {float(dd.max()):.2e}, within-day rank corr {float(ic_daily(c(M["B0"]), c(d.LGB)).mean()):.4f}')
    if M['C0'] is not None:
        print(f'    C0 (seed 1-5) vs cached mlp_res_xr9 (seed 1-10): within-day rank corr {float(ic_daily(c(M["C0"]), c(d.mlp[1])).mean()):.4f}')
    if SG is not None:
        br = SG['BV'].beta_raw.clip(0.2, 2.5).reindex(d.INFO.index)
        m_ = br.notna()
        print(f'    E1 raw betas vs hedge_beta (Top50 rows): max |diff| {float((br[m_] - d.INFO.beta[m_]).abs().max()):.2e} '
              f'({int(m_.sum()):,} rows)')

    # ------------------------------------------------------------ part 1
    print('\nPART 1 -- own daily rank IC with the excess return (all 100 stocks; report only)')
    pe11 = d.view([d.LGB, d.x1])
    S = {'v11 stock view p_e': pe11, 'lgb_res (cached)': d.LGB, 'X1 (cached)': d.x1, 'mlp_res_xr9 (cached)': d.mlp[1],
         'stock MLPs (mean of 4, cached)': pd.concat(d.mlp, axis=1).mean(axis=1)}
    for n in L.ORDER:
        if M[n] is not None:
            S[f'{n} {L.MEMBERS[n]["label"]}'] = M[n]
    if B0_3 is not None:
        S['B0 seeds 1-3'] = B0_3
    for k, v in sigs.items():
        S[f'{k} {"Kalman local level" if k == "D1" else "causal a trous Haar"}'] = v
    ICd = {k: ic_daily(c(v), d.resid) for k, v in S.items()}
    print(pd.DataFrame([{'member / signal': k[:46], **per(v)} for k, v in ICd.items()]).round(4).to_string(index=False))
    key = {n: f'{n} {L.MEMBERS[n]["label"]}' for n in L.ORDER if M[n] is not None}
    key.update({k: f'{k} {"Kalman local level" if k == "D1" else "causal a trous Haar"}' for k in sigs})

    def icd(x, y):
        dd = (ICd[key[x] if x in key else x] - ICd[key[y] if y in key else y]).dropna()
        return '  '.join(f'{p} {dd[w(dd.index, p)].mean():+.4f} ({nw_t(dd[w(dd.index, p)].values):.2f})' for p in PER)
    print('\n  paired daily IC differences, mean (NW t):')
    pairs = [(f'B{i}', 'B0') for i in range(1, 6)] + [('B6', 'B0 seeds 1-3'), ('C1', 'C0'), ('C2', 'C0'),
             ('A1', 'lgb_res (cached)'), ('A2', 'lgb_res (cached)'), ('A2', 'A1'), ('A0', 'A1'), ('B0', 'lgb_res (cached)')]
    for x, y in pairs:
        xk = key.get(x, x); yk = key.get(y, y)
        if xk in ICd and yk in ICd:
            print(f'    {x:4s} - {y:20s}: {icd(xk, yk)}')
    for k, h in (('B3', 3), ('B4', 5)):
        if M[k] is not None:
            rk = kday(d.resid, h)
            a1, a0 = per(ic_daily(c(M[k]), rk)), per(ic_daily(c(M['B0']), rk))
            print(f'  {k} against its own {h}-day target: ' + ' / '.join(f'{a1[p]:.4f}' for p in PER)
                  + f'   (B0 against the same target: ' + ' / '.join(f'{a0[p]:.4f}' for p in PER) + ')')
    print('\n  within-day rank correlation (mean over days, all periods) and two-signal diagnostics against the v11 stock view')
    print('  (README_v14 theorem 2: adding a little of z raises the IC iff IC_z > rho * IC_x; w* = best weight; ranks, approximate)')
    diag = {}
    others = {'v11 p_e': pe11, 'lgb_res': d.LGB, 'X1': d.x1, 'MLPs': S['stock MLPs (mean of 4, cached)']}
    for name in ['A1', 'A2', 'A0', 'B6', 'C1', 'C2', 'D1', 'D2']:
        k = key.get(name)
        if k is None:
            continue
        z = S[k]
        print(f'    {name}: ' + ', '.join(f'{o} {ic_daily(c(z), c(v)).mean():.3f}' for o, v in others.items()))
        rho_d = ic_daily(c(z), c(pe11))
        for p in PER:
            ix, iz, rho = per(ICd['v11 stock view p_e'])[p], per(ICd[k])[p], float(rho_d[w(rho_d.index, p)].mean())
            a_, b_ = ix - rho * iz, iz - rho * ix
            wstar = b_ / (a_ + b_) if (a_ + b_) != 0 else np.nan
            icopt = math.sqrt(max(ix * ix + iz * iz - 2 * rho * ix * iz, 0) / (1 - rho * rho)) if abs(rho) < 1 else np.nan
            diag[(name, p)] = dict(IC_x=ix, IC_z=iz, rho=rho, gain_condition=b_, w_star=wstar, IC_best=icopt)
        print('       ' + '  '.join(f'{p}: IC_z-rho*IC_x {diag[(name, p)]["gain_condition"]:+.4f}, w* {diag[(name, p)]["w_star"]:.2f}, '
                                    f'best {diag[(name, p)]["IC_best"]:.4f} (now {diag[(name, p)]["IC_x"]:.4f})' for p in ('val', 'hold', 'post')))

    # ------------------------------------------------------------ part 2
    sc = scale_to
    def mlp_with(x):
        out = list(d.mlp); out[1] = x; return out
    res, views, base_of, beta_of = {}, {}, {}, {}
    for k in A_C:
        if M[k] is not None:
            views[f'v11 + {k}'] = d.view([d.LGB, d.x1, sc(M[k], d.LGB)]); base_of[k] = ('v11 + ' + k, 'v11')
    if M['B0'] is not None:
        views['v11 [lgb_res := B0]'] = d.view([sc(M['B0'], d.LGB), d.x1])
        for k in B_C[:-1]:
            if M[k] is not None:
                views[f'v11 [lgb_res := {k}]'] = d.view([sc(M[k], d.LGB), d.x1]); base_of[k] = (f'v11 [lgb_res := {k}]', 'v11 [lgb_res := B0]')
    if B0_3 is not None and M['B6'] is not None:
        views['v11 [lgb_res := B0 seeds 1-3]'] = d.view([sc(B0_3, d.LGB), d.x1])
        views['v11 [lgb_res := B6]'] = d.view([sc(M['B6'], d.LGB), d.x1]); base_of['B6'] = ('v11 [lgb_res := B6]', 'v11 [lgb_res := B0 seeds 1-3]')
    if M['C0'] is not None:
        views['v11 [xr9 := C0]'] = d.view([d.LGB, d.x1], mlp=mlp_with(sc(M['C0'], d.mlp[1])))
        for k in C_C:
            if M[k] is not None:
                views[f'v11 [xr9 := {k}]'] = d.view([d.LGB, d.x1], mlp=mlp_with(sc(M[k], d.mlp[1]))); base_of[k] = (f'v11 [xr9 := {k}]', 'v11 [xr9 := C0]')
    if SG is not None and SG.get('leak_ok', False):
        for k in D_C:
            views[f'v11 + {k}'] = with_families(pe11, [zday(sigs[k], idx)]); base_of[k] = (f'v11 + {k}', 'v11')
        views['v11 + E1'] = pe11; beta_of['v11 + E1'] = betaV; base_of['E1'] = ('v11 + E1', 'v11')
    for k, v in views.items():
        D[k] = d.run(v, True, beta=beta_of.get(k))
        D[k.replace('v11', 'W0', 1)] = d.run(v, True, w_t=0.0, lam=0.5, beta=beta_of.get(k))
    print('\nPART 2 -- the rules (PREREG_v15.md section 3), bp/day (NW t)')
    print('  each candidate alone, against its own base (A, D, E: v11; B: v11 with the B0 replica; B6: B0 seeds 1-3; C: v11 with the C0 replica):')
    for k in CAND:
        if k in base_of:
            x, y = base_of[k]; r = diff(D, x, y); res[k] = r
            print(f'    {k}  {x[:30]:30s} - {y[:30]:30s}: {fmt(r)}   -> ' + ('val > 0: selected' if r['val'][0] > 0 else 'not selected'))
        else:
            print(f'    {k}  missing (not trained / not built) -> not judged')
    sel = [k for k in CAND if k in res and res[k]['val'][0] > 0]
    print(f'  selected: {", ".join(sel) if sel else "none"}')
    passed = False; r15 = None
    if sel:
        selB = [k for k in sel if k in B_C]
        if selB:
            parts = [sc(M[k], d.LGB) for k in selB]
            lgbB = pd.concat(parts, axis=1).mean(axis=1)
        else:
            lgbB = d.LGB
        lgbs = [lgbB, d.x1] + [sc(M[k], d.LGB) for k in sel if k in A_C]
        selC = [k for k in sel if k in C_C]
        mlp = mlp_with(pd.concat([sc(M[k], d.mlp[1]) for k in selC], axis=1).mean(axis=1)) if selC else None
        pe15 = d.view(lgbs, mlp=mlp)
        selD = [k for k in sel if k in D_C]
        if selD:
            pe15 = with_families(pe15, [zday(sigs[k], idx) for k in selD])
        b15 = betaV if 'E1' in sel else None
        D['v15'] = d.run(pe15, True, beta=b15); D['W0 v15'] = d.run(pe15, True, w_t=0.0, lam=0.5, beta=b15)
        r15 = diff(D, 'v15', 'v11')
        passed = r15['hold'][0] > 0 and r15['post'][0] > 0
        print(f'  v15 = v11 + {" + ".join(sel)}:')
        print(f'    v15 - v11: {fmt(r15)}')
        print('    ' + ('hold and post both > 0 -> PASSED (research result; a daily program only if you decide so before 11/23)' if passed
                        else 'hold or post <= 0 -> NOT PASSED: v9, v11 and W0 stay as they are'))
        rw = diff(D, 'W0 v15', 'W0')
        print(f'    report only, W0 v15 - W0: {fmt(rw)}')
    else:
        print('  nothing selected -> NOT PASSED: v9, v11 and W0 stay as they are')

    # ------------------------------------------------------------ part 3
    print('\nPART 3 -- report only')
    print('  W0 versions (each against its own W0 base):')
    for k in CAND:
        if k in base_of:
            x, y = base_of[k]
            print(f'    {k}  {fmt(diff(D, x.replace("v11", "W0", 1), y.replace("v11", "W0", 1)))}')
    print('\n  by year, candidate - its base (bp/day, NW t):')
    for k in CAND:
        if k in base_of:
            x, y = base_of[k]
            print(f'    {k}  {fmt(diff(D, x, y, YEARS))}')
    print('\n  2026-10-01 .. 10-22 (15 days, information only):')
    for k in CAND:
        if k in base_of:
            x, y = base_of[k]
            print(f'    {k}  v11 {fmt(diff(D, x, y, OCT))}   W0 {fmt(diff(D, x.replace("v11", "W0", 1), y.replace("v11", "W0", 1), OCT))}')
    print('\n  power of the rule (normal approximation; P(selected and both later periods > 0 | true effect delta), alone):')
    pw = {}
    for k in CAND:
        if k in base_of:
            x, y = base_of[k]; sds, pr = power(D, x, y); pw[k] = pr
            print(f'    {k}: daily sd ' + ', '.join(f'{p} {sds[p][0]:.1f}' for p in sds) + ';  ' + ', '.join(f'{dl:+.1f} -> {pv:.2f}' for dl, pv in pr.items()))
    extra = {}
    if M['A1'] is not None:
        extra['v11, A1 (RF) replaces lgb_res'] = (d.view([sc(M['A1'], d.LGB), d.x1]), 'v11 [lgb_res := B0]' if M['B0'] is not None else 'v11')
        extra['v11, A1 (RF) as a new family (v14 placement)'] = (with_families(pe11, [zday(M['A1'], idx)]), 'v11')
        extra['v11, A1 + A2 in the LightGBM part'] = (d.view([d.LGB, d.x1] + [sc(M[k], d.LGB) for k in ('A1', 'A2') if M[k] is not None]), 'v11')
    if M['A0'] is not None:
        extra['v11 + A0 (sklearn default RF) in the LightGBM part'] = (d.view([d.LGB, d.x1, sc(M['A0'], d.LGB)]), 'v11')
    if M['B0'] is not None:
        extra['v11 [lgb_res := B0] (replica) vs v11'] = (views['v11 [lgb_res := B0]'], 'v11')
    if M['C0'] is not None:
        extra['v11 [xr9 := C0] (replica) vs v11'] = (views['v11 [xr9 := C0]'], 'v11')
    if extra:
        print('\n  other placements (report only):')
        for k, (v, base) in extra.items():
            if k not in D:
                D[k] = d.run(v, True)
            print(f'    {k[:52]:52s} - {base:22s}: {fmt(diff(D, k, base))}')
    print('\n  forests (A): chosen min_samples_leaf per fold and the weighted log-loss of the grid on the last 120 days')
    for n in ('A1', 'A2'):
        for m in L.member_metas(n, test):
            print(f'    {n} {m["fold"]}: leaf {m["leaf"]}, ' + ', '.join(f'{k}: {v:.5f}' for k, v in m['val_logloss'].items())
                  + f'; {m["rows"]:,} rows, {m["seconds"] / 60:.1f} min')
    for n in ('A1', 'A2', 'A0'):
        ms = L.member_metas(n, test)
        if ms:
            imp = {}
            for m in ms:
                for f, v in m['mdi_top10']:
                    imp[f] = imp.get(f, 0.0) + v / len(ms)
            top = sorted(imp.items(), key=lambda t: -t[1])[:10]
            print(f'    {n} MDI importance (mean over folds, top 10): ' + ', '.join(f'{f} {v:.3f}' for f, v in top))
    ms = L.member_metas('B6', test)
    if ms:
        print('\n  DoubleEnsemble (B6): rounds, features per sub-model, SR weight quantiles (5/25/50/75/95%) of sub-models 2..K')
        for m in ms:
            print(f'    {m["fold"]} seed {m["seed"]}: {m["best"]} rounds, features {m["n_features"]}, '
                  + ' | '.join('/'.join(f'{x:.2f}' for x in q) for q in m['sr_weight_quantiles']))
    for n in ('C0', 'C1', 'C2'):
        ms = L.member_metas(n, test)
        if ms:
            eps = [m['best_epoch'] for m in ms]
            print(f'  {n} best epochs: median {np.median(eps):.0f}, range {min(eps)}-{max(eps)}')
    for n in ('B0', 'B1', 'B2', 'B3', 'B4', 'B5'):
        ms = L.member_metas(n, test)
        if ms:
            bs = [m['best'] for m in ms]
            print(f'  {n} LightGBM rounds: median {np.median(bs):.0f}, range {min(bs)}-{max(bs)}; weight cv {np.median([m["w_cv"] for m in ms]):.2f}')
    rows = []
    for k, Dk in D.items():
        rr = per(Dk.AR); r19 = Dk.AR.rolling(19).mean().dropna()
        rows.append({'config': k[:44], **rr, 'sd': Dk.AR.std(), 'timing': Dk.timing.mean(), 'xsec': Dk.xsec.mean(),
                     'share1': (Dk.sbar.mean() + 1) / 2, 'P19>0': (r19 > 0).mean()})
    print('\nTop50 bp/day (walk-forward 2022-01 .. 2026-10)')
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    os.makedirs(L.RESULTS, exist_ok=True)
    tag = 'v15_test' if test else 'v15'
    pd.DataFrame({k: v.AR for k, v in D.items()}).to_csv(os.path.join(L.RESULTS, f'{tag}_daily.csv'), float_format='%.4f')
    json.dump({'rebuild_ok': bool(ok), 'test': bool(test), 'each_minus_base': res, 'selected': sel, 'v15_minus_v11': r15, 'passed': bool(passed),
               'diagnostics': {f'{k[0]} {k[1]}': v for k, v in diag.items()}, 'power': pw},
              open(os.path.join(L.RESULTS, f'{tag}_selection.json'), 'w'), indent=1, default=float)
    print(f'\nsaved results\\{tag}_daily.csv, results\\{tag}_selection.json ({time.time() - t0:.0f}s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())

r"""
v16_eval.py -- K2, K3 and the judgment of v16 (research\PREREG_v16.md sections 2-4). numpy + pandas only.

  K1  Kalman beta for the hedge (built by v16_signals.py)                         v11 + K1 - v11
  K2  time-varying timing coefficients: m50_t = a_t + b_t x_t + eps_t, x = pbar' - 0.5    v11 + K2 - v11
  K3  each day v11 or W0, by the local level of the daily difference AR(v11) - AR(W0)     K3 - v11

reads  ..\cache\signals16.pkl (v16_signals.py) and the cached walk-forward predictions (v16lib.Data)
writes ..\results\v16_daily.csv, ..\results\v16_selection.json; the report goes to stdout
usage (in v16\research):  python v16_eval.py > ..\results\v16_eval.txt
"""
import os, sys, json, math, time
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v16lib as L
from v16lib import per, w, nw_t, diff, fmt, power, PER, YEARS, OCT, PUBLISHED

CAND = ['K1', 'K2', 'K3']


def k2_build(d, x, m50, dates, cut=None):
    """K2 on the walk-forward days; cut = D: m50 is known only up to D-1 (leak check)"""
    y = m50.copy()
    if cut is not None:
        y = y[y.index < cut]
        dates = dates[dates <= cut]
    R2 = L.var_series(y, L.K2_RWIN, L.K2_RMIN)
    yy = y.reindex(dates); RR = R2.reindex(dates)
    if cut is not None:                                                                    # R_D uses m50 up to D-1 only
        full = pd.concat([y, pd.Series([np.nan], index=pd.DatetimeIndex([cut]))]) if cut not in y.index else y
        RR = L.var_series(full, L.K2_RWIN, L.K2_RMIN).reindex(dates)
    xx = x.reindex(dates)
    pa, pb, llt, snap = L.k2_filter(yy.values, xx.values, RR.values, dates)
    gsel, sel = L.k2_select(snap, dates)
    T, mult, icpt = L.k2_timing(pa, pb, xx.values, gsel)
    return dict(pa=pa, pb=pb, llt=llt, snap=snap, gsel=gsel, sel=sel, T=pd.Series(T, index=dates), mult=pd.Series(mult, index=dates),
                icpt=pd.Series(icpt, index=dates), x=xx, y=yy, R=RR, dates=dates)


def main():
    t0 = time.time()
    print('v16_eval.py -- Kalman filter: K1 hedge beta, K2 time-varying timing coefficients, K3 v11 / W0 switch (research\\PREREG_v16.md)')
    print(f'  {time.strftime("%Y-%m-%d %H:%M")}, pandas {pd.__version__}, numpy {np.__version__}')
    sf = os.path.join(L.CACHE, 'signals16.pkl')
    if not os.path.exists(sf):
        print('  cache\\signals16.pkl missing: run python v16_signals.py first'); return 1
    S = pd.read_pickle(sf)
    d = L.Data(); dates = d.dates
    pe11 = d.view([d.LGB, d.x1]); pe9 = d.view([d.LGB])
    D = {'v9': d.run(pe9, False), 'v11': d.run(pe11, True), 'W0': d.run(pe11, True, w_t=0.0, lam=0.5)}

    # ------------------------------------------------------------ part 0
    print('\nPART 0 -- checks')
    ok = True
    for k, ref in PUBLISHED.items():
        got = tuple(round(per(D[k].AR)[p], 2) for p in PER)
        same = all(abs(x - y) <= 0.005 for x, y in zip(got, ref)); ok &= same
        print(f'  {k:4s} rebuilt {" / ".join(f"{x:.2f}" for x in got)}   published {" / ".join(f"{x:.2f}" for x in ref)}   '
              + ('same' if same else 'DIFFERENT'))
    if not ok:
        print('  WARNING: the rebuild differs from the published numbers: find the cause before deciding')
    ols = L.hedge_ready(S['OLS120'], d.INFO.index)
    print(f'  120-day OLS beta of v16 (prices read like hedge_beta) vs the beta in the judge (Top50 rows): max |diff| '
          f'{float((ols - d.INFO.beta).abs().max()):.1e}')
    m50 = S['m50']
    dm = (m50.reindex(D['v11'].index) * 1e4 - D['v11'].m).abs()
    print(f'  m50 from the prices vs the judge\'s Top50 mean m: max |diff| {float(dm.max()):.1e} bp over {int(dm.notna().sum())} days')
    print(f'  K1 leak check (v16_signals.py): ' + ('OK' if S.get('leak_ok') else 'FAILED -> K1 is not judged') + f' (max {S.get("leak_worst", float("nan")):.1e})')

    # ------------------------------------------------------------ K1
    beta = {k: L.hedge_ready(S[k], d.INFO.index) for k in ('K1', 'K1q0', 'E1')}
    if S.get('leak_ok'):
        D['v11 + K1'] = d.run(pe11, True, beta=beta['K1']); D['W0 + K1'] = d.run(pe11, True, w_t=0.0, lam=0.5, beta=beta['K1'])
    for k, lab in (('K1q0', 'K1 q=0'), ('E1', 'E1 Vasicek')):
        D[f'v11 + {lab}'] = d.run(pe11, True, beta=beta[k]); D[f'W0 + {lab}'] = d.run(pe11, True, w_t=0.0, lam=0.5, beta=beta[k])

    # ------------------------------------------------------------ K2
    x11 = d.pbar(night=True).reindex(dates) - 0.5
    x9 = d.pbar(night=False).reindex(dates) - 0.5
    K2 = k2_build(d, x11, m50, dates)
    T0, _, _ = L.k2_timing(K2['pa'], K2['pb'], K2['x'].values, 0)
    nest = float(np.max(np.abs(T0 - L.W_T * K2['x'].values)))
    Dn = d.run(pe11, True, timing=pd.Series(T0, index=dates))
    nest_ar = float((Dn.AR - D['v11'].AR).abs().max())
    print(f'  K2 nesting: q = (0, 0) timing term vs w_T x: max |diff| {nest:.1e}; v11 rebuilt through it: max |AR diff| {nest_ar:.1e}')
    rng = np.random.default_rng(1602)
    pick = sorted(rng.choice(np.flatnonzero(dates >= pd.Timestamp('2023-01-01')), size=4, replace=False))
    worst2 = 0.0
    for i in pick:
        Dd = dates[i]
        Kc = k2_build(d, x11, m50, dates, cut=Dd)
        worst2 = max(worst2, abs(float(Kc['T'].loc[Dd]) - float(K2['T'].loc[Dd])))
    leak2 = worst2 < 1e-12
    print(f'  K2 leak check (m50 cut at D-1, 4 random days from 2023 on; max |diff| of the timing term at D): {worst2:.1e}  '
          + ('OK' if leak2 else 'FAILED -> K2 is not judged'))
    if leak2:
        D['v11 + K2'] = d.run(pe11, True, timing=K2['T'])
    K29 = k2_build(d, x9, m50, dates)
    D['v9 + K2'] = d.run(pe9, False, timing=K29['T'])
    forced2 = {}
    for qb in (1e-4, 1e-3, 1e-2):
        g = L.K2_GRID.index((0.0, qb))
        Tf, _, _ = L.k2_timing(K2['pa'], K2['pb'], K2['x'].values, g)
        name = f'v11 + K2 fixed q_b {qb:g}'
        D[name] = d.run(pe11, True, timing=pd.Series(Tf, index=dates)); forced2[qb] = name

    # ------------------------------------------------------------ K3
    D['K3'], use3, sel3, mu3 = L.k3_run(D['v11'], D['W0'])
    worst3 = 0.0
    for i in pick:
        Dd = dates[i]
        a = D['v11'][D['v11'].index <= Dd].copy(); b = D['W0'][D['W0'].index <= Dd].copy()
        a.loc[Dd, 'AR'] = np.nan; b.loc[Dd, 'AR'] = np.nan
        _, _, _, muc = L.k3_run(a, b)
        worst3 = max(worst3, abs(float(muc[-1]) - float(mu3[np.flatnonzero(dates == Dd)[0]])))
    print(f'  K3 leak check (AR of D removed, 4 random days; max |diff| of mu at D): {worst3:.1e}  ' + ('OK' if worst3 < 1e-9 else 'FAILED'))
    forced3 = {}
    for h in L.K3_HALF_LIVES:
        q = L.q_from_half_life(h)
        name = f'K3 half-life {h}'
        D[name], _, _, _ = L.k3_run(D['v11'], D['W0'], qs=[q], fixed=0); forced3[h] = (name, q)

    # ------------------------------------------------------------ part 1 (parameters)
    print('\nPART 1 -- what the filters chose (only data before each fold; never the AR)')
    print('  K1 (v16_signals.py): ' + '; '.join(f'{r["fold"][:7]} phi {r["phi"]:g} q {r["q"]:g} (memory {r["memory_days"]:.0f} d, '
                                              f'gain {r["gain_vs_q0"]:+.1f})' for r in S['k1_table']))
    print('  K2: (q_a, q_b) per fold and the log-likelihood gain over (0, 0) on the days before the fold:')
    for i, (fa, _) in enumerate(L.FOLDS):
        g = K2['sel'][fa]; sn = K2['snap'][pd.Timestamp(fa)]
        print(f'    {fa}: q_a {L.K2_GRID[g][0]:g}, q_b {L.K2_GRID[g][1]:g}' + ('  (fixed: no out-of-sample pbar before 2022)' if i == 0 else
              f', gain {sn[g] - sn[0]:+.2f}; best with q_b > 0: {max(sn[j] - sn[0] for j in range(len(L.K2_GRID)) if L.K2_GRID[j][1] > 0):+.2f}'))
    print('    v9 version: ' + ', '.join(f'{fa[:7]} ({L.K2_GRID[g][0]:g}, {L.K2_GRID[g][1]:g})' for fa, g in K29['sel'].items()))
    print('  K3: q per fold (concentrated likelihood of the daily v11 - W0 difference before the fold):')
    for i, (fa, _) in enumerate(L.FOLDS):
        g = sel3[fa]; q = L.K3_Q[g]
        Kq = (-q + math.sqrt(q * q + 4 * q)) / 2 if q > 0 else 0.0
        print(f'    {fa}: q {q:.2e}' + (f' (half-life {math.log(0.5) / math.log(1 - Kq):.0f} days)' if q > 0 else ' (cumulative mean)')
              + ('  (fixed: no difference before 2022)' if i == 0 else ''))

    # ------------------------------------------------------------ part 2
    print('\nPART 2 -- the rules (PREREG_v16.md section 3), bp/day (NW t)')
    base_of = {'K1': ('v11 + K1', 'v11'), 'K2': ('v11 + K2', 'v11'), 'K3': ('K3', 'v11')}
    res = {}
    for k in CAND:
        x, y = base_of[k]
        if x in D:
            r = diff(D, x, y); res[k] = r
            print(f'  {k}  {x:10s} - {y:4s}: {fmt(r)}   -> ' + ('val > 0: selected' if r['val'][0] > 0 else 'not selected'))
        else:
            print(f'  {k}  not built (leak check failed) -> not judged')
    sel = [k for k in CAND if k in res and res[k]['val'][0] > 0]
    print(f'  selected: {", ".join(sel) if sel else "none"}')
    passed, r16, sel16 = False, None, None
    if sel:
        b16 = beta['K1'] if 'K1' in sel else None
        t16 = K2['T'] if 'K2' in sel else None
        v11p = d.run(pe11, True, beta=b16, timing=t16)
        if 'K3' in sel:
            W0p = d.run(pe11, True, w_t=0.0, lam=0.5, beta=b16)
            D['v16'], _, sel16, _ = L.k3_run(v11p, W0p)
        else:
            D['v16'] = v11p
        r16 = diff(D, 'v16', 'v11')
        passed = r16['hold'][0] > 0 and r16['post'][0] > 0
        print(f'  v16 = v11 + {" + ".join(sel)}' + (f' (K3 q per fold: {", ".join(f"{L.K3_Q[g]:.1e}" for g in sel16.values())})' if sel16 else '') + ':')
        print(f'    v16 - v11: {fmt(r16)}')
        print('    ' + ('hold and post both > 0 -> PASSED (research result; a daily program only if you decide so before 11/23)' if passed
                        else 'hold or post <= 0 -> NOT PASSED: v9, v11 and W0 stay as they are'))
        print(f'    stricter standard, all-period NW t >= 2: {r16["all"][1]:.2f} -> ' + ('met' if r16['all'][1] >= 2 else 'not met'))
        if 'K1' in sel:
            print(f'    report only, W0 + K1 - W0 (the W0 part of v16): {fmt(diff(D, "W0 + K1", "W0"))}')
    else:
        print('  nothing selected -> NOT PASSED: v9, v11 and W0 stay as they are')

    # ------------------------------------------------------------ part 3
    print('\nPART 3 -- report only')
    print('  W0 and v9 versions, and the nested references:')
    rep = [('W0 + K1', 'W0'), ('K3', 'W0'), ('v9 + K2', 'v9'), ('v11 + K1 q=0', 'v11'), ('v11 + E1 Vasicek', 'v11'),
           ('v11 + K1', 'v11 + E1 Vasicek'), ('v11 + K1', 'v11 + K1 q=0'), ('W0 + K1 q=0', 'W0'), ('W0 + E1 Vasicek', 'W0')]
    for x, y in rep:
        if x in D and y in D:
            print(f'    {x:18s} - {y:18s}: {fmt(diff(D, x, y))}')
    print('\n  stricter standard (all-period NW t >= 2), each candidate against its base:')
    for k in CAND:
        if k in res:
            print(f'    {k}: all {res[k]["all"][0]:+.2f} (t {res[k]["all"][1]:.2f}) -> ' + ('met' if res[k]['all'][1] >= 2 else 'not met'))
    print('\n  by year, candidate - base (bp/day, NW t):')
    for k in CAND:
        x, y = base_of[k]
        if x in D:
            print(f'    {k}  {fmt(diff(D, x, y, YEARS))}')
    print('\n  2026-10-01 .. 10-22 (15 days, information only):')
    for k in CAND:
        x, y = base_of[k]
        if x in D:
            print(f'    {k}  {fmt(diff(D, x, y, OCT))}')
    print(f'    W0 + K1 - W0  {fmt(diff(D, "W0 + K1", "W0", OCT))}' if 'W0 + K1' in D else '')
    print('\n  power of the rule (normal approximation; P(selected and both later periods > 0 | true effect delta), alone):')
    pw = {}
    for k in CAND:
        x, y = base_of[k]
        if x in D:
            sds, pr = power(D, x, y); pw[k] = pr
            print(f'    {k}: daily sd ' + ', '.join(f'{p} {sds[p][0]:.1f}' for p in sds) + ';  ' + ', '.join(f'{dl:+.1f} -> {pv:.2f}' for dl, pv in pr.items()))

    # beta forecast error
    print('\n  K1: beta forecast error on the Top50 stock-days (README_v16 4.4): L = mean over the day of (r - beta m)^2;')
    print('  the paired daily difference against the 120-day OLS divided by mean(m^2) estimates the change of the beta MSE')
    r = d.INFO.target_r; mm = S['m'].reindex(r.index.get_level_values('date')).values
    Ls = {'OLS120': d.INFO.beta, 'E1 Vasicek': beta['E1'], 'K1 q=0': beta['K1q0'], 'K1': beta['K1']}
    daily = {k: ((r - b.reindex(r.index).values * mm) ** 2).groupby(level='date').mean() * 1e8 for k, b in Ls.items()}
    m2 = (S['m'] ** 2 * 1e8).reindex(daily['OLS120'].index)
    for k in Ls:
        line = f'    {k:11s} L (bp^2): ' + ' / '.join(f'{per(daily[k])[p]:.0f}' for p in PER)
        if k != 'OLS120':
            dd = daily[k] - daily['OLS120']
            parts = []
            for p in PER:
                sel_ = w(dd.index, p)
                parts.append(f'{p} {dd[sel_].mean() / m2[sel_].mean():+.4f} (t {nw_t(dd[sel_].values):.2f})')
            line += '   change of beta MSE vs OLS120: ' + '  '.join(parts)
        print(line)
    dK = daily['K1'] - daily['E1 Vasicek']
    print('    K1 - E1 Vasicek, change of beta MSE: ' + '  '.join(f'{p} {dK[w(dK.index, p)].mean() / m2[w(dK.index, p)].mean():+.4f} '
                                                          f'(t {nw_t(dK[w(dK.index, p)].values):.2f})' for p in PER))

    # K2 diagnostics
    print('\n  K2 diagnostics:')
    xy = pd.DataFrame({'x': K2['x'], 'y': K2['y']})
    print('    corr(x = pbar\' - 0.5, m50) by period: ' + '  '.join(f'{p} {xy[w(xy.index, p)].corr().iloc[0, 1]:+.3f}' for p in PER)
          + '   (by year: ' + '  '.join(f'{p} {xy[w(xy.index, p, YEARS)].corr().iloc[0, 1]:+.3f}' for p in YEARS) + ')')
    print(f'    sd of x {float(K2["x"].std()):.4f}; sd of m50 {float(K2["y"].std() * 1e4):.1f} bp')
    for p in PER:
        ss = w(dates, p)
        mu_, ic_ = K2['mult'][ss], K2['icpt'][ss]
        print(f'    {p}: multiplier min {mu_.min():+.2f} median {mu_.median():+.2f} max {mu_.max():+.2f} (share < 0: {(mu_ < 0).mean():.2f}); '
              f'intercept term |.| mean {ic_.abs().mean():.4f}, max {ic_.abs().max():.4f}')
    t_ = np.arange(len(dates))
    dll = pd.Series(K2['llt'][t_, K2['gsel']] - K2['llt'][:, 0], index=dates)
    print('    out-of-sample one-step log-likelihood of the chosen q minus (0, 0), summed: ' + '  '.join(f'{p} {dll[w(dates, p)].sum():+.2f}' for p in PER))
    print('    forced adaptation (q_a 0, fixed q_b in every fold; v11 + K2 fixed - v11):')
    for qb, name in forced2.items():
        g = L.K2_GRID.index((0.0, qb)); dlf = pd.Series(K2['llt'][:, g] - K2['llt'][:, 0], index=dates)
        print(f'      q_b {qb:g}: {fmt(diff(D, name, "v11"))};  log-lik vs (0, 0): ' + '  '.join(f'{p} {dlf[w(dates, p)].sum():+.1f}' for p in ('val', 'hold', 'post')))

    # K3 diagnostics
    print('\n  K3 diagnostics: share of days on W0, and the filtered mean of v11 - W0 (bp/day) at the end of each period')
    dlt = (D['v11'].AR - D['W0'].AR)
    for p in PER:
        ss = w(use3.index, p)
        print(f'    {p}: W0 on {(~use3[ss]).mean():.2f} of the days; mu at the end {mu3[np.flatnonzero(ss)[-1]]:+.2f}; '
              f'v11 - W0 that period {dlt[w(dlt.index, p)].mean():+.2f}')
    print('    forced adaptation (fixed half-life in every fold; K3 fixed - v11, and the share of days on W0):')
    for h, (name, q) in forced3.items():
        _, u, _, _ = L.k3_run(D['v11'], D['W0'], qs=[q], fixed=0)
        print(f'      half-life {h:3d} d (q {q:.1e}): {fmt(diff(D, name, "v11"))};  W0 share ' + ' / '.join(f'{(~u[w(u.index, p)]).mean():.2f}' for p in ('val', 'hold', 'post')))
    print(f'    K3 on 2026-10-22 (last walk-forward day): mu {mu3[-1]:+.2f} -> {"v11" if use3.iloc[-1] else "W0"}')

    rows = []
    for k, Dk in D.items():
        rr = per(Dk.AR); r19 = Dk.AR.rolling(19).mean().dropna()
        rows.append({'config': k[:30], **rr, 'sd': Dk.AR.std(), 'timing': Dk.timing.mean(), 'xsec': Dk.xsec.mean(),
                     'share1': (Dk.sbar.mean() + 1) / 2, 'P19>0': (r19 > 0).mean()})
    print('\nTop50 bp/day (walk-forward 2022-01 .. 2026-10-22)')
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    os.makedirs(L.RESULTS, exist_ok=True)
    pd.DataFrame({k: v.AR for k, v in D.items()}).to_csv(os.path.join(L.RESULTS, 'v16_daily.csv'), float_format='%.4f')
    json.dump({'rebuild_ok': bool(ok), 'each_minus_base': res, 'selected': sel, 'v16_minus_v11': r16, 'passed': bool(passed),
               'k1_selection': {fa: L.K1_GRID[g] for fa, g in S['sel'].items()}, 'k2_selection': {fa: L.K2_GRID[g] for fa, g in K2['sel'].items()},
               'k3_selection': {fa: float(L.K3_Q[g]) for fa, g in sel3.items()}, 'power': pw,
               'checks': {'k1_leak_ok': bool(S.get('leak_ok')), 'k2_leak_max': worst2, 'k3_leak_max': worst3, 'k2_nesting_max': nest,
                          'k2_nesting_ar_max': nest_ar}},
              open(os.path.join(L.RESULTS, 'v16_selection.json'), 'w'), indent=1, default=float)
    print(f'\nsaved results\\v16_daily.csv, results\\v16_selection.json ({time.time() - t0:.0f}s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())

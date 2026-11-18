r"""
v16_signals.py -- K1 of v16 (research\PREREG_v16.md section 2): the Kalman beta for the hedge, plus the betas it is
compared with (120-day OLS = hedge_beta, Vasicek = v15 E1, the K1 filter with q = 0) and the leak check. numpy + pandas only.

  K1  r_it = alpha_i + beta_it m_t + eps_it, eps ~ N(0, R_it), R_it = 250-day OLS residual variance (days before t);
      beta_it - 1 = phi (beta_i,t-1 - 1) + eta_it, eta ~ N(0, q); alpha_i fixed; prior alpha ~ N(0, 0.002^2), beta ~ N(1, 0.6^2)
      (phi, q) on 37 grid points, chosen per fold by the pooled log-likelihood of the days before the fold (all 100 stocks,
      each from its 61st update); signal for day D = beta_{D | D-1}

Leak check: the prices are cut at D-1 for 6 random days D from 2022 on and K1 is recomputed; beta at D must be identical.
-> ..\cache\signals16.pkl ; report -> stdout
usage (in v16\research): python v16_signals.py
"""
import os, sys, time, math
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v16lib as L


def k1_build(end=None, extra_dates=()):
    M, m, m50, top50 = L.price_panel(end, extra_dates)
    V = L.resid_var(M, m)
    pred, snap = L.k1_filter(M, m, V)
    K1, sel = L.k1_select(pred, snap, M.index, M.columns)
    return dict(M=M, m=m, m50=m50, top50=top50, V=V, pred=pred, snap=snap, K1=K1, sel=sel)


# ------------------------------------------------------------------ E1 (= v15_signals.betas_with_se + vasicek)
def betas_with_se(end=None):
    R, mk, _ = L.price_long(end)
    X = R.merge(mk, left_on='date', right_index=True).sort_values(['stock_id', 'date'])
    out = []
    for sid, g in X.groupby('stock_id', sort=False):
        x, y = g.m100, g.r
        roll = lambda s: s.rolling(120, min_periods=60)
        mx, my = roll(x).mean(), roll(y).mean()
        cv = roll(x * y).mean() - mx * my
        vx = roll(x).var(ddof=0); vy = roll(y).var(ddof=0)
        n = roll((x.notna() & y.notna()).astype(float)).sum()
        b = cv / vx
        se2 = (vy - b * b * vx).clip(lower=0) / ((n - 2).clip(lower=1) * vx)
        out.append(pd.DataFrame({'date': g.date.values, 'stock_id': sid, 'beta': b.shift(1).values, 'se2': se2.shift(1).values}))
    B = pd.concat(out, ignore_index=True)
    B['date'] = pd.to_datetime(B.date).astype('datetime64[ns]'); B['stock_id'] = B.stock_id.astype(str)
    return B.set_index(['date', 'stock_id']).sort_index()


def vasicek(B):
    g = B.groupby(level='date')
    bbar = g.beta.transform('mean')
    s2 = (g.beta.transform('var') - g.se2.transform('mean')).clip(lower=0.01)
    bv = (s2 * B.beta + B.se2 * bbar) / (s2 + B.se2)
    return bv.where(B.beta.notna() & B.se2.notna())


def main():
    t0 = time.time()
    print('v16_signals.py -- K1 Kalman beta (research\\PREREG_v16.md)')
    print(f'  {time.strftime("%Y-%m-%d %H:%M")}, pandas {pd.__version__}, numpy {np.__version__}')
    K = k1_build()
    M, m, pred, snap, sel = K['M'], K['m'], K['pred'], K['snap'], K['sel']
    print(f'  prices {M.index.min().date()} .. {M.index.max().date()}, {M.shape[0]} days x {M.shape[1]} stocks; '
          f'{len(L.K1_GRID)} grid points; filter + selection {time.time() - t0:.0f}s')

    print('\nK1: chosen (phi, q) per fold, by the pooled log-likelihood of the days before the fold')
    print('  gain = log-likelihood of the chosen point minus that of q = 0 (fixed beta, all data since 2020);')
    print('  best RW / best phi<1 = the best point with phi = 1 / phi < 1; memory = 1 / steady-state gain (README_v16 theorem 3)')
    rows = []
    for fa, _ in L.FOLDS:
        f = pd.Timestamp(fa); ll, n = snap[f]
        g = sel[fa]; phi, q = L.K1_GRID[g]
        rw = [i for i, (p, qq) in enumerate(L.K1_GRID) if p == 1.0 and qq > 0]
        mr = [i for i, (p, qq) in enumerate(L.K1_GRID) if p < 1.0]
        before = M.index < f
        m2 = float(np.nanmean(m[before].values ** 2))
        Rmed = float(np.nanmedian(K['V'][before].values))
        Kg, mem, sd, hl = L.k1_memory(phi, q, m2, Rmed)
        brw = max(rw, key=lambda i: ll[i]); bmr = max(mr, key=lambda i: ll[i])
        rows.append(dict(fold=fa, phi=phi, q=q, gain_vs_q0=ll[g] - ll[0], best_rw=f'q {L.K1_GRID[brw][1]:g} ({ll[brw] - ll[0]:+.1f})',
                         best_mr=f'phi {L.K1_GRID[bmr][0]:g} q {L.K1_GRID[bmr][1]:g} ({ll[bmr] - ll[0]:+.1f})',
                         memory_days=mem, stationary_sd=sd, reversion_half_life=hl, n_terms=n))
        print(f'  {fa}: phi {phi:g}, q {q:g}; log-lik gain over q = 0 {ll[g] - ll[0]:+.1f} ({n:,} terms); '
              f'best RW {rows[-1]["best_rw"]}, best phi<1 {rows[-1]["best_mr"]}; memory {mem:.0f} days'
              + (f', stationary sd {sd:.2f}, reversion half-life {hl:.0f} days' if phi < 1 else ''))

    # leak check
    rng = np.random.default_rng(16)
    days = [d for d in M.index if d >= pd.Timestamp('2022-01-03') and d <= pd.Timestamp('2026-10-22')]
    pick = sorted(rng.choice(len(days), size=6, replace=False))
    worst = 0.0
    for i in pick:
        D = days[i]; prev = M.index[M.index.get_loc(D) - 1]
        Kc = k1_build(end=prev, extra_dates=[D])
        a = K['K1'].loc[D].values; b = Kc['K1'].reindex(columns=K['K1'].columns).loc[D].values
        ok = np.isfinite(a) | np.isfinite(b)
        diff = np.nanmax(np.abs(a[ok] - b[ok])) if ok.any() else 0.0
        if np.isnan(diff) or (np.isfinite(a) != np.isfinite(b)).any():
            diff = float('inf')
        worst = max(worst, float(diff))
    leak_ok = worst == 0.0 or worst < 1e-12
    print(f'\nleak check (prices cut at D-1, 6 random days from 2022 on; max |difference| of beta at D, must be 0): {worst:.1e}  '
          + ('OK' if leak_ok else 'FAILED: K1 uses data of day D or later'))

    # comparison betas
    OLS = L.ols_beta(M, m)
    B = betas_with_se()
    E1 = vasicek(B)
    K1q0 = pd.DataFrame(pred[:, 0, :], index=M.index, columns=M.columns)
    S = {'K1': L.long(K['K1'], 'K1'), 'K1q0': L.long(K1q0, 'K1q0'), 'OLS120': L.long(OLS, 'OLS120'), 'E1': E1.rename('E1'),
         'sel': sel, 'grid': L.K1_GRID, 'll': {fa: snap[pd.Timestamp(fa)][0] for fa, _ in L.FOLDS}, 'k1_table': rows,
         'leak_ok': bool(leak_ok), 'leak_worst': worst, 'M': M, 'm': m, 'm50': K['m50'], 'top50': K['top50']}
    wf = (M.index >= pd.Timestamp('2022-01-01')) & (M.index <= pd.Timestamp('2026-10-22'))
    T50 = sorted(K['top50'])
    o = L.long(OLS.loc[wf, T50], 'OLS120')
    X = pd.DataFrame({'OLS120': o, 'E1': E1.reindex(o.index), 'K1': L.long(K['K1'].loc[wf, T50], 'K1'),
                      'K1q0': L.long(K1q0.loc[wf, T50], 'K1q0')}).dropna().clip(*L.BETA_CLIP)
    print(f'\nTop50 betas on the walk-forward days (2022-01 .. 2026-10-22, {len(X):,} stock-days, clipped to [0.2, 2.5]):')
    print('  mean, cross-sectional sd (mean over days), mean |x - OLS120|, correlation with OLS120, day-to-day sd of the change:')
    for k in X.columns:
        s = X[k]
        csd = float(s.groupby(level=0).std().mean())
        chg = s.groupby(level='stock_id').diff()
        print(f'    {k:7s} mean {s.mean():.3f}, cs sd {csd:.3f}, |x - OLS120| {float((s - X.OLS120).abs().mean()):.3f}, '
              f'corr {float(np.corrcoef(s, X.OLS120)[0, 1]):.3f}, daily change sd {float(chg.std()):.4f}')
    os.makedirs(L.CACHE, exist_ok=True)
    pd.to_pickle(S, os.path.join(L.CACHE, 'signals16.pkl'))
    print(f'\nsaved {os.path.join(L.CACHE, "signals16.pkl")} ({time.time() - t0:.0f}s)')
    return 0 if leak_ok else 1


if __name__ == '__main__':
    sys.exit(main())

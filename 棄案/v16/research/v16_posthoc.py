r"""
v16_posthoc.py -- after the judgment (hindsight, report only; nothing here changes the PREREG_v16 verdict).

  1. K1 with every q fixed (no MLE), plus q 3e-3 and 1e-2 beyond the grid edge: beta forecast error against the 120-day
     OLS and the v11 / W0 AR difference on the walk-forward days. Answers: is K1 worse because of the chosen q, or for any q?
  2. Where the K1 log-likelihood gain comes from (days before 2026-06-01): sum of log F (variance part), sum of v^2 / F,
     and the plain squared one-step error sum of v^2 (point forecast), for q = 0, the chosen point and the 120-day OLS.
  3. The numbers that README_v16 theorem 4 needs (detectability): sd of m100, median idiosyncratic sd, sd of m50, sd of the
     daily v11 - W0 difference.
usage (in v16\research): python v16_posthoc.py > ..\results\v16_posthoc.txt
"""
import os, sys, time, math
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v16lib as L
from v16lib import per, w, nw_t, diff, fmt, PER

EXTRA = [(1.0, 3e-3), (1.0, 1e-2), (0.999, 3e-3), (0.999, 1e-2)]


def k1_parts(M, m, V, grid, start, end, burn=L.K1_BURN):
    """the K1 filter (= v16lib.k1_filter) with the log-likelihood split into its parts over start <= t < end"""
    y = M.values; mm = m.values; Vv = V.values
    T, N = y.shape; G = len(grid)
    phi = np.array([g[0] for g in grid])[:, None]; q = np.array([g[1] for g in grid])[:, None]
    A = np.zeros((G, N)); B = np.zeros((G, N)); P00 = np.zeros((G, N)); P01 = np.zeros((G, N)); P11 = np.zeros((G, N))
    started = np.zeros(N, bool); nupd = np.zeros(N, int)
    slogF = np.zeros(G); sv2F = np.zeros(G); sv2 = np.zeros(G); n = 0
    pred = np.full((T, G, N), np.nan)
    for t in range(T):
        obs = np.isfinite(y[t]) & np.isfinite(Vv[t]) & np.isfinite(mm[t])
        s = started
        if s.any():
            B[:, s] = 1.0 + phi * (B[:, s] - 1.0); P11[:, s] = phi * phi * P11[:, s] + q; P01[:, s] = phi * P01[:, s]
        new = obs & ~started
        if new.any():
            A[:, new] = 0.0; B[:, new] = L.K1_B_MEAN; P00[:, new] = L.K1_A_SD ** 2; P01[:, new] = 0.0; P11[:, new] = L.K1_B_SD ** 2
            started = started | new
        pred[t][:, started] = B[:, started]
        if obs.any():
            o = obs; mt = mm[t]
            v = y[t, o][None, :] - A[:, o] - B[:, o] * mt
            F = P00[:, o] + 2 * mt * P01[:, o] + mt * mt * P11[:, o] + Vv[t, o][None, :]
            K0 = (P00[:, o] + mt * P01[:, o]) / F; K1 = (P01[:, o] + mt * P11[:, o]) / F
            A[:, o] += K0 * v; B[:, o] += K1 * v
            P00[:, o] -= K0 * K0 * F; P01[:, o] -= K0 * K1 * F; P11[:, o] -= K1 * K1 * F
            nupd[o] += 1
            cnt = nupd[o] > burn
            if cnt.any() and start <= M.index[t] < end:
                slogF += np.log(F[:, cnt]).sum(axis=1); sv2F += (v[:, cnt] ** 2 / F[:, cnt]).sum(axis=1)
                sv2 += (v[:, cnt] ** 2).sum(axis=1); n += int(cnt.sum())
    return pred, slogF, sv2F, sv2, n


def main():
    t0 = time.time()
    print('v16_posthoc.py -- hindsight checks after the v16 judgment (report only)')
    print(f'  {time.strftime("%Y-%m-%d %H:%M")}, pandas {pd.__version__}, numpy {np.__version__}')
    M, m, m50, top50 = L.price_panel()
    V = L.resid_var(M, m)
    grid = L.K1_GRID + EXTRA
    S = pd.read_pickle(os.path.join(L.CACHE, 'signals16.pkl'))
    g_post = S['sel']['2026-06-01']

    # 2. likelihood parts before 2026-06-01
    pred, slogF, sv2F, sv2, n = k1_parts(M, m, V, grid, pd.Timestamp('2020-01-01'), pd.Timestamp('2026-06-01'))
    print(f'\n2. K1 log-likelihood parts, days before 2026-06-01 ({n:,} terms; log-lik = -(sum log F + sum v^2/F) / 2):')
    print('   point                 -sum(log F)/2   -sum(v^2/F)/2   log-lik vs q 0   mean v^2 (bp^2)   mean v^2 vs q 0')
    for g in sorted(set([0, g_post] + [i for i, (p, q) in enumerate(grid) if p == 1.0])):
        phi, q = grid[g]
        ll = -0.5 * (slogF[g] + sv2F[g]); ll0 = -0.5 * (slogF[0] + sv2F[0])
        print(f'   phi {phi:<5g} q {q:<7g}{" (chosen)" if g == g_post else "         "} {-0.5 * slogF[g]:14.1f} {-0.5 * sv2F[g]:15.1f} '
              f'{ll - ll0:+15.1f} {sv2[g] / n * 1e8:16.2f} {(sv2[g] - sv2[0]) / n * 1e8:+16.3f}')

    # 1. every q fixed: beta error and AR
    d = L.Data()
    pe11 = d.view([d.LGB, d.x1])
    D = {'v11': d.run(pe11, True), 'W0': d.run(pe11, True, w_t=0.0, lam=0.5)}
    r = d.INFO.target_r; dts = r.index.get_level_values('date'); mm = m.reindex(dts).values
    m2 = (m ** 2 * 1e8)
    base = ((r - d.INFO.beta.values * mm) ** 2).groupby(level='date').mean() * 1e8
    ols = L.long(L.ols_beta(M, m), 'b')
    print('\n1. K1 with q fixed in every fold (hindsight): change of the beta MSE against the 120-day OLS (val / hold / post / all, t of all),')
    print('   and the AR difference v11 + K1(q) - v11 and W0 + K1(q) - W0 (val / hold / post / all, NW t)')
    for g, (phi, q) in enumerate(grid):
        if phi not in (1.0, 0.999):
            continue
        Bq = pd.DataFrame(pred[:, g, :], index=M.index, columns=M.columns)
        b = L.hedge_ready(L.long(Bq, 'b'), d.INFO.index)
        lo = ((r - b.reindex(r.index).values * mm) ** 2).groupby(level='date').mean() * 1e8
        dd = lo - base; mm2 = m2.reindex(dd.index)
        mse = ' / '.join(f'{dd[w(dd.index, p)].mean() / mm2[w(dd.index, p)].mean():+.4f}' for p in PER)
        D['a'] = d.run(pe11, True, beta=b); D['b'] = d.run(pe11, True, w_t=0.0, lam=0.5, beta=b)
        ra, rb = diff(D, 'a', 'v11'), diff(D, 'b', 'W0')
        print(f'   phi {phi:<5g} q {q:<7g}{" (beyond the grid)" if (phi, q) in EXTRA else ""}: beta MSE {mse} (t {nw_t(dd.values):.2f});  '
              f'v11 {" / ".join(f"{ra[p][0]:+.2f}" for p in PER)} ({ra["all"][1]:.2f});  W0 {" / ".join(f"{rb[p][0]:+.2f}" for p in PER)} ({rb["all"][1]:.2f})')

    # 3. numbers for theorem 4
    wf = (M.index >= pd.Timestamp('2022-01-01')) & (M.index <= pd.Timestamp('2026-10-22'))
    sdm = float(m[wf].std()); sdi = float(np.sqrt(np.nanmedian(V.loc[wf, sorted(top50)].values)))
    dlt = (D['v11'].AR - D['W0'].AR)
    print('\n3. numbers for README_v16 theorem 4 (walk-forward days):')
    print(f'   sd of m100 {sdm * 1e4:.1f} bp, median idiosyncratic sd of the Top50 (sqrt of R) {sdi * 1e4:.1f} bp, sd of m50 {float(m50[wf].std()) * 1e4:.1f} bp')
    print(f'   per-day separation for a beta change of 0.2: 0.2 x {sdm * 1e4:.1f} / {sdi * 1e4:.1f} = {0.2 * sdm / sdi:.3f}')
    print(f'   daily v11 - W0: sd {float(dlt.std()):.1f} bp; mean by period ' + ', '.join(f'{p} {v:+.2f}' for p, v in per(dlt).items())
          + '; month-to-month autocorrelation of the monthly means '
          + f'{float(dlt.groupby(dlt.index.to_period("M")).mean().autocorr()):+.2f}')
    print(f'\n({time.time() - t0:.0f}s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())

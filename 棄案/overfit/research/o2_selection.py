r"""
o2_selection.py -- overfitting of the CHOICES (coefficients, components, versions), not of the model weights.
(numpy + pandas only; ~3-5 minutes; writes ..\results\o2_selection.txt and ..\cache\o2_*.npz)

  1. CSCV / PBO (Bailey, Borwein, Lopez de Prado & Zhu 2017) on
       a. v9's 4,375 coefficient settings (w_t, lam, lgb_w, tf_w, bias; v9\cache\adapt_grid.npz)
       b. the 35 settings that only move w_t and lam (the two coefficients of the 10/27 upload W0)
       c. every distinct daily series of the development (v5 .. v12, Alpha158 screens, ablations)
  2. Reality check (White 2000, studentized as Hansen 2005): is ANY of the 4,375 settings better than v9's own
     coefficients, after accounting for having looked at all of them?  walk-forward 1,164 days and rolling static 99 days.
  3. Choosing coefficients on a short recent window (what static / October did): for every window of L days pick the best
     of the 4,375 settings and score it on the next 19 trading days (the length of the private test) vs v9's coefficients.
  4. Deflated statistics of the final models (t against 0 and Deflated Sharpe Ratio with the number of settings tried).
run (in overfit\research): python o2_selection.py > ..\results\o2_selection.txt
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import ofit_lib as L
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
RNG = np.random.default_rng(20261028)


def cscv_report(name, M, cols, ref=None, S=16):
    t0 = time.time()
    r = L.cscv(M, S=S)
    T, N = M.shape
    line = (f'{name}: N = {N} settings, T = {T} days, S = {S} blocks, {len(r["rank"])} splits ({time.time() - t0:.0f}s)\n'
            f'  PBO = {r["pbo"]:.3f}   (share of splits where the IS-best setting lands in the bottom half OOS)\n'
            f'  IS-best: IS mean {r["is_best"].mean():.2f}, OOS mean {r["oos_of_best"].mean():.2f}; OOS median setting {r["oos_median"].mean():.2f} bp/day\n'
            f'  OOS relative rank of the IS-best: mean {r["rank"].mean():.3f}, quartiles {np.quantile(r["rank"], [.25, .5, .75]).round(3).tolist()}\n')
    b = np.polyfit(r['is_best'], r['oos_of_best'], 1)
    line += f'  degradation: OOS = {b[1]:.2f} + {b[0]:.3f} x IS over the splits (slope 1 = no overfitting, 0 = IS says nothing)\n'
    if ref is not None:
        # the reference (v9's own coefficients) in the same OOS halves
        parts = np.array_split(np.arange(T), S); size = np.array([len(p) for p in parts], float)
        Bref = np.array([M[p, ref].mean() for p in parts])
        import itertools
        combs = np.array(list(itertools.combinations(range(S), S // 2)))
        W = np.ones((len(combs), S)); W[np.arange(len(combs))[:, None], combs] = 0.0
        oref = (W * size) @ Bref / (W * size).sum(1)
        d = r['oos_of_best'] - oref
        line += (f'  IS-best minus {cols[ref]} in the same OOS half: mean {d.mean():+.2f} bp/day, share of splits > 0 {np.mean(d > 0):.3f}\n')
    return line, r


def section1(AW, settings, S):
    print('=' * 110)
    print('1. CSCV / PBO: if a setting is picked because it was best on half of the days, how does it do on the other half?')
    print('=' * 110)
    names = settings.apply(lambda r: f'w_t {r.w_t} lam {r.lam} lgb {r.lgb_w} tf {r.tf_w} bias {r.bias}', axis=1).tolist()
    k9 = L.grid_key()
    line, ra = cscv_report('1a. v9 coefficient grid', AW, names, ref=k9)
    print(line)
    picks = pd.Series(ra['pick']).value_counts().head(5)
    print('  most frequent IS-best settings:')
    for k, n in picks.items():
        print(f'    {names[k]}: {n} of {len(ra["pick"])} splits')
    sub = np.flatnonzero((settings.lgb_w == 0.5) & (settings.tf_w == 0.5) & (settings.bias == 0.0))
    line, rb = cscv_report('\n1b. only w_t and lam move (35 settings; W0 = w_t 0, lam 0.5 is one of them)', AW[:, sub],
                           [names[i] for i in sub], ref=int(np.flatnonzero(sub == k9)[0]))
    print(line)
    picks = pd.Series(sub[rb['pick']]).value_counts().head(5)
    print('  most frequent IS-best settings:')
    for k, n in picks.items():
        print(f'    {names[k]}: {n} of {len(rb["pick"])} splits')
    # robustness: other numbers of blocks
    for s in (8, 10, 12):
        r2 = L.cscv(AW, S=s)
        print(f'  (grid with S = {s}: PBO {r2["pbo"]:.3f}, OOS of IS-best {r2["oos_of_best"].mean():.2f}, OOS median {r2["oos_median"].mean():.2f})')
    # development pool
    uniq = {}
    for k, v in S.items():
        key = np.round(v.values, 6).tobytes()
        if key not in uniq:
            uniq[key] = k
    keep = list(uniq.values())
    M = np.column_stack([S[k].values for k in keep])
    line, rc = cscv_report(f'\n1c. development pool: every distinct daily series of v5..v12 and the screens', M, keep,
                           ref=keep.index('v9'))
    print(line)
    picks = pd.Series([keep[i] for i in rc['pick']]).value_counts().head(6)
    print('  most frequent IS-best series:')
    for k, n in picks.items():
        print(f'    {k}: {n} of {len(rc["pick"])} splits')
    np.savez_compressed(os.path.join(L.CACHE, 'o2_cscv.npz'), grid_rank=ra['rank'], grid_is=ra['is_best'], grid_oos=ra['oos_of_best'],
                        grid_med=ra['oos_median'], wl_rank=rb['rank'], pool_rank=rc['rank'], pool_is=rc['is_best'], pool_oos=rc['oos_of_best'],
                        pool_med=rc['oos_median'])
    return keep


def reality_check(D, B=1000, mean_block=10, label=''):
    """D: T x K daily differences (setting - reference).  H0: max_k E[D_k] <= 0.  Studentized max statistic,
    stationary bootstrap of the centred differences.  Returns (T_obs, p, argmax, crit95)"""
    T, K = D.shape
    m = D.mean(0)
    se = np.array([L.nw_se(D[:, k]) for k in range(K)])
    se = np.where(se > 1e-9, se, np.inf)
    tstat = m / se
    kbest = int(np.nanargmax(tstat)); Tobs = tstat[kbest]
    Dc = D - m
    I = L.stationary_bootstrap_idx(T, B, mean_block, RNG)
    mx = np.empty(B)
    for b in range(B):
        mx[b] = np.max(Dc[I[b]].mean(0) / se)
    return Tobs, float((mx >= Tobs).mean()), kbest, float(np.quantile(mx, 0.95)), m


def section2(dW, AW, dS, AS, settings):
    print('=' * 110)
    print('2. Reality check: is any of the 4,375 settings truly better than v9\'s coefficients? (studentized max, stationary bootstrap)')
    print('=' * 110)
    k9 = L.grid_key()
    names = settings.apply(lambda r: f'w_t {r.w_t} lam {r.lam} lgb {r.lgb_w} tf {r.tf_w} bias {r.bias}', axis=1).tolist()
    for lab, idx, A in (('walk-forward 2022-01..2026-10 (1,164 days)', np.arange(len(dW)), AW),
                        ('walk-forward val only (2022-2024)', np.flatnonzero(L.mask(dW, 'val')), AW),
                        ('rolling static 6/1-10/22 (99 days, the period the static coefficients were chosen on)', np.arange(len(dS)), AS)):
        D = A[idx] - A[idx][:, [k9]]
        D = np.delete(D, k9, axis=1); nm = [n for i, n in enumerate(names) if i != k9]
        Tobs, p, kb, crit, m = reality_check(D, B=1000)
        best_raw = int(np.argmax(m))
        print(f'  {lab}:')
        print(f'    best setting by t: {nm[kb]}  diff {m[kb]:+.2f} bp/day, t {Tobs:.2f}; 95% critical value of the max-t under H0 {crit:.2f}; p = {p:.3f}')
        print(f'    best setting by mean: {nm[best_raw]}  diff {m[best_raw]:+.2f} bp/day')
    print()


def section3(dW, AW, settings):
    print('=' * 110)
    print('3. Picking the coefficients on a short recent window and using them for the next 19 trading days')
    print('   (static was 84 days, static + October 99 days, October alone 15 days; the private test is about 19 days)')
    print('=' * 110)
    k9 = L.grid_key(); kW0 = L.grid_key(w_t=0.0, lam=0.5)
    names = settings.apply(lambda r: f'w_t {r.w_t} lam {r.lam} lgb {r.lgb_w} tf {r.tf_w} bias {r.bias}', axis=1).tolist()
    T = len(dW)
    rows = []
    for Lw in (15, 60, 84, 99, 250):
        res = []
        for start in range(0, T - Lw - 19 + 1, 19):              # non-overlapping 19-day test blocks
            w = slice(start, start + Lw); nx = slice(start + Lw, start + Lw + 19)
            mw = AW[w].mean(0); kb = int(np.argmax(mw))
            res.append((mw[kb] - mw[k9], AW[nx, kb].mean() - AW[nx, k9].mean(), AW[nx, kW0].mean() - AW[nx, k9].mean(),
                        mw[kW0] - mw[k9], kb))
        R = pd.DataFrame(res, columns=['in_window', 'next19', 'W0_next19', 'W0_in_window', 'kb'])
        d = R.next19.values
        t = d.mean() / d.std(ddof=1) * np.sqrt(len(d))
        top = pd.Series([names[k] for k in R.kb]).value_counts().index[0]
        rows.append({'window L': Lw, 'n windows': len(R), 'best in window, advantage over v9': R.in_window.mean(),
                     'same setting, next 19 days': d.mean(), 't': t, 'share next19 > 0': (d > 0).mean(),
                     'slope next19 on window': np.polyfit(R.in_window, R.next19, 1)[0], 'most picked': top})
        if Lw == 99:
            R99 = R
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    # W0 specifically: when W0 had beaten v9 over the previous 99 days, what happened next
    m = R99.W0_in_window > 0
    print(f'\n  W0 (w_t 0, lam 0.5; in v9\'s grid) after a 99-day window where it beat v9: next 19 days {R99.W0_next19[m].mean():+.2f} '
          f'(n = {int(m.sum())}); after a window where it lost: {R99.W0_next19[~m].mean():+.2f} (n = {int((~m).sum())})')
    print(f'  W0 vs v9 next-19-day difference over all {len(R99)} blocks: {R99.W0_next19.mean():+.2f}')
    print()


def deflated(series, n_trials, sr_var_trials, label):
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) on daily AR; also t vs 0 and the max-t of n_trials null t's"""
    x = series.dropna().values; T = len(x)
    sr = x.mean() / x.std(ddof=1)
    g3 = ((x - x.mean()) ** 3).mean() / x.std() ** 3; g4 = ((x - x.mean()) ** 4).mean() / x.std() ** 4
    emc = 0.5772156649
    from statistics import NormalDist
    Z = NormalDist().inv_cdf
    sr0 = np.sqrt(sr_var_trials) * ((1 - emc) * Z(1 - 1 / n_trials) + emc * Z(1 - 1 / (n_trials * np.e)))
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    dsr = NormalDist().cdf(z)
    return {'model': label, 'mean bp': x.mean(), 'daily SR': sr, 'NW t vs 0': L.nw_t(x), 'skew': g3, 'kurt': g4,
            'trials N': n_trials, 'SR0 (expected max of N null SRs)': sr0, 'DSR = P(true SR > 0 | N trials)': dsr}


def section4(S, AW):
    print('=' * 110)
    print('4. The final models against 0 after counting how many settings were tried (Deflated Sharpe Ratio)')
    print('=' * 110)
    srs = AW.mean(0) / AW.std(0, ddof=1)
    var_sr = srs.var(ddof=1)
    rows = []
    for k, lab in (('v9', 'v9'), ('v11', 'v11'), ('W0', 'W0 (10/27 upload)'), ('v5 (control)', 'v5')):
        for N in (100, 4375, 100000):
            rows.append(deflated(S[k], N, var_sr, lab))
    print(f'  variance of the daily Sharpe ratio across the 4,375 grid settings: {var_sr:.5f} (sd {np.sqrt(var_sr):.4f}); '
          f'E[max of N iid N(0,1)] for N = 4,375: {L.expected_max_normal(4375):.2f}')
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    # the increments
    print('\n  increments (paired daily differences), t against the max of K null t\'s:')
    for a, b, K in (('v11', 'v9', 7), ('v9', 'v5 (control)', 26), ('W0', 'v11', 240), ('W0', 'v9', 4375)):
        d = (S[a] - S[b]).dropna().values
        t = L.nw_t(d)
        print(f'    {a} - {b}: {d.mean():+.2f} bp/day, NW t {t:+.2f}; K = {K} candidates -> E[max null t] ~ {L.expected_max_normal(K):.2f}, '
              f'Bonferroni 5% one-sided t > {abs(__import__("statistics").NormalDist().inv_cdf(0.05 / K)):.2f}')
    print()


def main():
    t0 = time.time()
    (dW, AW), (dS, AS) = L.load_grid()
    settings = L.grid_settings()
    S = L.load_series()
    print(f'o2_selection.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}   grid {AW.shape}, static grid {AS.shape}, {len(S)} development series\n')
    section1(AW, settings, S)
    section2(dW, AW, dS, AS, settings)
    section3(dW, AW, settings)
    section4(S, AW)
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()

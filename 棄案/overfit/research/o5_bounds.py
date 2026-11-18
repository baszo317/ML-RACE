r"""
o5_bounds.py -- the numbers behind the theorems of README_overfit.md section 2 (numpy + pandas only; seconds).
  T2  Bernstein / Freedman bound for the walk-forward mean (finite sample, worst-case range from the +/-10% price limits)
  T3  selection: expected optimism of the best of K settings <= s sqrt(2 ln K), against what was actually observed
  T4  re-used holdout: what a hold / post confirmation can certify after b pass/fail decisions
  T0  why capacity (VC) bounds say nothing here
run (in overfit\research): python o5_bounds.py > ..\results\o5_bounds.txt
"""
import os, sys, time, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import ofit_lib as L
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
R_BP = 2 * (1.1 / 0.9 - 1) * 1e4        # |X_t - mu_t| <= 2 x 22.2%: open and close both within +/-10% of the previous close


def bernstein_eps(n, sigma, R, delta):
    """one-sided: P(mean - mu >= eps) <= delta when every conditional variance <= sigma^2 and X_t - mu_t <= R"""
    Lg = math.log(1 / delta)
    a = Lg * R / 3 + math.sqrt((Lg * R / 3) ** 2 + 2 * Lg * n * sigma ** 2)
    return a / n


def main():
    t0 = time.time()
    S = L.load_series()
    (dW, AW), (dS, AS) = L.load_grid()
    from statistics import NormalDist
    Z = NormalDist().inv_cdf
    print(f'o5_bounds.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}\n')
    print('=' * 110)
    print(f'T2. Finite-sample lower bound of the true average skill (one-sided 95%), R = {R_BP:.0f} bp')
    print('=' * 110)
    rows = []
    for k in ('v9', 'v11', 'W0', 'v5 (control)'):
        a = S[k].values; n = len(a); sd = a.std(ddof=1)
        for mult in (1.0, 1.5, 2.0):
            e = bernstein_eps(n, mult * sd, R_BP, 0.05)
            rows.append({'model': k, 'n': n, 'mean': a.mean(), 'assumed conditional sd': mult * sd, 'Bernstein eps': e,
                         'lower bound (Bernstein)': a.mean() - e, 'lower bound (NW, asymptotic)': a.mean() - 1.645 * L.nw_se(a),
                         'Hoeffding eps (range only)': R_BP * math.sqrt(math.log(20) / (2 * n))})
    T = pd.DataFrame(rows)
    print(T.round(2).to_string(index=False))
    print('  (Hoeffding uses only the range [-22.2%, +22.2%] and is useless; Bernstein uses the variance and certifies the level)\n')
    # per period
    rows = []
    for k in ('v9', 'v11', 'W0'):
        for p in ('val', 'hold', 'post', 'Oct'):
            a = S[k][L.mask(S[k].index, p)].values; n = len(a); sd = S[k].std(ddof=1)
            e = bernstein_eps(n, sd, R_BP, 0.05)
            rows.append({'model': k, 'period': p, 'n': n, 'mean': a.mean(), 'Bernstein eps (sd = full-period sd)': e, 'lower bound': a.mean() - e,
                         'NW lower bound': a.mean() - 1.645 * L.nw_se(a)})
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print()

    print('=' * 110)
    print('T3. Selection: how much can the best of K settings look better than v9 by luck alone?')
    print('    bound E[max_k noise_k] <= s sqrt(2 ln K), s = the largest NW se of (setting - v9) in that period (bp/day)')
    print('=' * 110)
    k9 = L.grid_key()
    rows = []
    for lab, A, m in (('walk-forward all (1,164 days)', AW, np.ones(len(dW), bool)), ('val 2022-24', AW, L.mask(dW, 'val')),
                      ('hold 2025-01..2026-05', AW, L.mask(dW, 'hold')), ('rolling static 6/1-9/30 (84 days)', AS, L.mask(dS, 'static')),
                      ('rolling static + October (99 days)', AS, np.ones(len(dS), bool)), ('October 10/1-10/22 (15 days)', AS, L.mask(dS, 'Oct'))):
        D = A[m] - A[m][:, [k9]]
        mean = D.mean(0); se = np.array([L.nw_se(D[:, j]) for j in range(D.shape[1]) if j != k9])
        s_max, s90 = se.max(), np.quantile(se, 0.9)
        r = {'period': lab, 'days': int(m.sum()), 's = max se': s_max, '90th pct se': s90}
        for K in (35, 240, 4375):
            r[f'bound K={K}'] = s_max * math.sqrt(2 * math.log(K))
        r['observed: best of 4,375 minus v9'] = mean.max()
        r['best setting'] = ', '.join(f'{k} {v}' for k, v in L.grid_settings().iloc[int(np.argmax(mean))].items())
        rows.append(r)
    T3 = pd.DataFrame(rows)
    print(T3.round(2).to_string(index=False))
    print('  (static_max = 32.18 vs v9 16.42 on the rolling static; the October-best of the 240 v11 settings was 33.02 vs v11 14.27)\n')

    print('=' * 110)
    print('T4. Re-used holdout: smallest difference a hold (or post) confirmation can certify after b pass/fail decisions')
    print('    eps_b = s sqrt(2 (b ln 2 + ln(2/delta))), delta = 0.05, two-sided')
    print('=' * 110)
    C = pd.read_csv(os.path.join(L.RESULTS, 'o3_changes.csv'))
    s_hold_change = float(np.median(C.se_hold)); s_post_change = float(np.median(C.se_post))
    d = (S['v11'] - S['v9']); s_hold_ver = L.nw_se(d[L.mask(d.index, 'hold')].values); s_post_ver = L.nw_se(d[L.mask(d.index, 'post')].values)
    rows = []
    for b in (0, 5, 10, 20, 30, 50):
        f = math.sqrt(2 * (b * math.log(2) + math.log(2 / 0.05)))
        rows.append({'b (decisions)': b, 'factor': f, 'hold, one change (s = %.2f)' % s_hold_change: f * s_hold_change,
                     'hold, v11 vs v9 (s = %.2f)' % s_hold_ver: f * s_hold_ver, 'post, one change (s = %.2f)' % s_post_change: f * s_post_change,
                     'post, v11 vs v9 (s = %.2f)' % s_post_ver: f * s_post_ver})
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    sdd = s_hold_ver * math.sqrt(L.mask(d.index, 'hold').sum())
    for b in (0, 20, 30):
        f = math.sqrt(2 * (b * math.log(2) + math.log(2 / 0.05)))
        print(f'  days needed to certify a 0.7 bp/day difference (v11 - v9 size, daily sd of the difference {sdd:.1f}) with b = {b}: {(f * sdd / 0.7) ** 2:,.0f}')
    print()

    print('=' * 110)
    print('T0. Capacity bounds (VC): |test error - training error| <= sqrt(8/n (d ln(2en/d) + ln(4/delta)))')
    print('=' * 110)
    for lab, dvc in (('linear rule on the 67 features (d = 68)', 68), ('MLP 72-64-32-1 (~6.8k weights, d ~ W log W ~ 6e4)', 60000)):
        for nlab, n in (('i.i.d. stock-days (optimistic), 2020-03..2026-05', 153000), ('days (what the market view really learns from)', 1530)):
            v = math.sqrt(8 / n * (dvc * math.log(2 * math.e * n / dvc) + math.log(4 / 0.05))) if n > dvc else float('inf')
            print(f'  {lab}, n = {nlab} {n:,}: bound {v:.3f} on the error-rate gap (the models\' OOS edge is about 0.04-0.06)')
    print(f'\ndone in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()

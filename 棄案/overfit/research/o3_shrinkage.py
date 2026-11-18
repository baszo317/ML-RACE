r"""
o3_shrinkage.py -- winner's curse of the screened changes: does a change that looked good on val also help on hold / post?
(numpy + pandas only; seconds; writes ..\results\o3_shrinkage.txt)

Every change screened in v9..v12 (and the Alpha158 screens of v10) whose daily series is cached is a pair
(base, base + change).  For each pair: paired daily difference d_t, its mean and Newey-West standard error in val / hold /
post.  Then
  * selected by the rule actually used (val > 0) vs not selected: mean difference in val, hold, post
  * regression of the hold (post) difference on the val difference across the changes
  * empirical Bayes: true effects theta_k ~ (m, tau^2), d_val,k = theta_k + e_k with se_k; tau^2 from
        (i)  the covariance of d_val and d_hold across changes (same theta in both periods, independent noise)
        (ii) method of moments Var(d_val) - mean(se_val^2)
    and the posterior mean tau^2 / (tau^2 + se^2) x d_val of the components that are in v9 / v11 now.
run (in overfit\research): python o3_shrinkage.py > ..\results\o3_shrinkage.txt
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import ofit_lib as L
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 200)


def table(S, C):
    rows = []
    for name, base, var, rule in C:
        d = (S[var] - S[base])
        r = {'change': name, 'rule': rule}
        for p in ('val', 'hold', 'post', 'all'):
            x = d[L.mask(d.index, p)].dropna().values
            r[p] = x.mean(); r['se_' + p] = L.nw_se(x)
        rows.append(r)
    return pd.DataFrame(rows)


def eb(T, label):
    x, y, z = T.val.values, T.hold.values, T.post.values
    sv = T.se_val.values
    tau2_cov = np.cov(x, y)[0, 1]
    tau2_mm = x.var(ddof=1) - np.mean(sv ** 2)
    print(f'  {label}: {len(T)} changes')
    print(f'    corr(val, hold) {np.corrcoef(x, y)[0, 1]:+.3f}, corr(val, post) {np.corrcoef(x, z)[0, 1]:+.3f}, corr(hold, post) {np.corrcoef(y, z)[0, 1]:+.3f}')
    bh = np.polyfit(x, y, 1); bp = np.polyfit(x, z, 1)
    print(f'    OLS hold = {bh[1]:+.3f} + {bh[0]:+.3f} x val ;  post = {bp[1]:+.3f} + {bp[0]:+.3f} x val')
    print(f'    sd of d_val across changes {x.std(ddof=1):.3f}; mean NW se of d_val {np.sqrt(np.mean(sv ** 2)):.3f} '
          f'-> tau^2 (moments) {tau2_mm:+.4f}; tau^2 (cov val-hold) {tau2_cov:+.4f}')
    tau2 = max(tau2_cov, 0.0)
    sel = x > 0
    print(f'    selected by val > 0: {sel.sum()} of {len(T)}: mean val {x[sel].mean():+.3f}, hold {y[sel].mean():+.3f}, post {z[sel].mean():+.3f};'
          f'  not selected: val {x[~sel].mean():+.3f}, hold {y[~sel].mean():+.3f}, post {z[~sel].mean():+.3f}')
    sel2 = x / sv > 2
    if sel2.any():
        print(f'    selected by val t > 2: {sel2.sum()}: mean val {x[sel2].mean():+.3f}, hold {y[sel2].mean():+.3f}, post {z[sel2].mean():+.3f}')
    return tau2


def main():
    t0 = time.time()
    S = L.load_series()
    C = L.candidates(S)
    T = table(S, C)
    print(f'o3_shrinkage.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}   {len(T)} screened changes with cached daily series (bp/day, paired differences)\n')
    print('=' * 110)
    print('1. Every change: mean paired difference (NW se) per period')
    print('=' * 110)
    show = T.copy()
    for p in ('val', 'hold', 'post', 'all'):
        show[p] = show.apply(lambda r: f'{r[p]:+.2f} ({r["se_" + p]:.2f})', axis=1)
    print(show[['change', 'rule', 'val', 'hold', 'post', 'all']].to_string(index=False))
    print()
    print('=' * 110)
    print('2. Do val differences predict hold / post differences?  (winner\'s curse)')
    print('=' * 110)
    scr = T[~T.change.str.startswith('v9 keep') & ~T.change.isin(['v9 vs v5', 'v11 vs v9'])]
    tau_all = eb(scr, 'all screened single changes (v10 Alpha158 groups, v11b, X5, v12, F1, F2, V10)')
    eb(scr[scr.change.str.startswith('v10 lgb') | scr.change.str.startswith('v10 mlp')], 'Alpha158 groups only (v10 screen)')
    eb(scr[~(scr.change.str.startswith('v10 lgb') | scr.change.str.startswith('v10 mlp'))], 'everything except the Alpha158 groups')
    print()
    print('=' * 110)
    print('3. Shrunken (posterior-mean) effect of what is in the models now')
    print('=' * 110)
    tau2 = max(tau_all, 1e-6)
    rows = []
    for name in ['v11b T2', 'v11b T3', 'v11b X1', 'v11 vs v9', 'v9 vs v5'] + [c for c in T.change if c.startswith('v9 keep')]:
        r = T[T.change == name].iloc[0]
        B = tau2 / (tau2 + r.se_val ** 2)
        Ba = tau2 / (tau2 + r.se_all ** 2)
        rows.append({'component': name, 'val (selection data)': r.val, 'se val': r.se_val, 'shrink factor': B, 'posterior mean from val': B * r.val,
                     'hold': r.hold, 'post': r.post, 'all-period mean': r['all'], 'posterior mean from all': Ba * r['all']})
    print(f'  tau^2 = {tau2:.4f} (sd of true effects {np.sqrt(tau2):.3f} bp/day), from the covariance of val and hold across the screened changes')
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print('\n  Note: the posterior from "all" re-uses hold / post, which were looked at during the development; it is an upper reference.')
    T.to_csv(os.path.join(L.RESULTS, 'o3_changes.csv'), index=False, float_format='%.4f')
    print(f'\nper-change table written to ..\\results\\o3_changes.csv; done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()

r"""
o4_drift.py -- does the walk-forward level carry over to a new period?  (drift between periods, the private test, live)
(numpy + pandas only; seconds; writes ..\results\o4_drift.txt)

  1. half-year means of v5 / v9 / v11 / W0 (and their timing / stock-picking parts) with Newey-West standard errors
  2. random-effects model over the half-years (DerSimonian-Laird): true half-year means mu_h ~ (m, tau^2);
     tau = how much the true level itself moves from period to period (beyond sampling noise)
  3. predictive distribution of the mean of a new 19-day period (the private test): m +/- sqrt(tau^2 + s19^2),
     P(mean > 0), compared with the empirical share of 19-day blocks with a mean > 0
  4. where 2026-06.. and October 2026 sit; the live day 10/27 inside each model's daily distribution
run (in overfit\research): python o4_drift.py > ..\results\o4_drift.txt
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import ofit_lib as L
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
LIVE_1027 = {'W0': -69.80, 'v9': -79.53, 'v11': -84.29}       # v6\phase2\scores.csv and v11\results\rerun_1027_us (bp)


def load_parts():
    b = L._read('v11/cache/v11b_daily.pkl'); f = L._read('v11/cache/v11f_daily.pkl'); d9 = L._read('v9/cache/v9_daily.pkl')
    return {'v5': d9['control: v4 rule, current members (= v5 live)'], 'v9': b['v9'], 'v11': b['v11'], 'W0': f['W0 (10/27 upload)']}


def dl(y, se):
    """DerSimonian-Laird random-effects: returns m, se(m), tau^2, Q, k"""
    w = 1 / se ** 2; m_fe = np.sum(w * y) / w.sum(); Q = np.sum(w * (y - m_fe) ** 2); k = len(y)
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - np.sum(w ** 2) / w.sum()))
    ws = 1 / (se ** 2 + tau2); m = np.sum(ws * y) / ws.sum()
    return m, np.sqrt(1 / ws.sum()), tau2, Q, k


def main():
    t0 = time.time()
    D = load_parts()
    print(f'o4_drift.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}   walk-forward 2022-01-03 .. 2026-10-22, Top50 bp/day\n')
    print('=' * 110)
    print('1. Half-year means (NW se)')
    print('=' * 110)
    rows = []
    for name, X in D.items():
        ix = pd.DatetimeIndex(X.index)
        h = ix.year.astype(str) + np.where(ix.month <= 6, 'H1', 'H2')
        for part in ('AR', 'timing', 'xsec'):
            g = X[part].groupby(h)
            for k, s in g:
                rows.append({'model': name, 'part': part, 'half': k, 'n': len(s), 'mean': s.mean(), 'se': L.nw_se(s.values)})
    H = pd.DataFrame(rows)
    tab = H[H.part == 'AR'].pivot(index='half', columns='model', values='mean')[list(D)]
    se = H[H.part == 'AR'].pivot(index='half', columns='model', values='se')[list(D)]
    n = H[(H.part == 'AR') & (H.model == 'v9')].set_index('half').n
    out = tab.round(2).astype(str) + ' (' + se.round(1).astype(str) + ')'
    out.insert(0, 'days', n)
    print(out.to_string())
    print()
    print('=' * 110)
    print('2. Random effects over the 10 half-years: how much does the TRUE level move between periods?')
    print('=' * 110)
    rows = []
    for name in D:
        for part in ('AR', 'timing', 'xsec'):
            X = H[(H.model == name) & (H.part == part)]
            m, sem, tau2, Q, k = dl(X['mean'].values, X['se'].values)
            rows.append({'model': name, 'part': part, 'RE mean m': m, 'se(m)': sem, 'tau (sd of true half-year means)': np.sqrt(tau2),
                         'Q': Q, 'df': k - 1, 'mean sampling se of a half-year': np.sqrt(np.mean(X['se'] ** 2))})
    RE = pd.DataFrame(rows)
    print(RE.round(3).to_string(index=False))
    print('  (Q ~ chi2(df) if the true level never moved; Q much larger than df = the level itself drifts)\n')
    print('=' * 110)
    print('3. The private test (about 19 trading days): predictive distribution of the 19-day mean')
    print('=' * 110)
    rows = []
    for name, X in D.items():
        a = X.AR.values; nT = len(a)
        r = RE[(RE.model == name) & (RE.part == 'AR')].iloc[0]
        tau = r['tau (sd of true half-year means)']
        # 19-day sampling sd with the serial correlation of the daily scores (NW variance ratio)
        vr = (L.nw_se(a) ** 2 * nT) / a.var(ddof=1)
        s19 = a.std(ddof=1) * np.sqrt(vr / 19)
        sd_pred = np.sqrt(tau ** 2 + s19 ** 2 + r['se(m)'] ** 2)
        from statistics import NormalDist
        N = NormalDist()
        blocks = np.array([a[i:i + 19].mean() for i in range(0, nT - 18, 19)])
        rows.append({'model': name, 'm': r['RE mean m'], 'tau': tau, '19-day sampling sd': s19, 'predictive sd': sd_pred,
                     '95% interval low': r['RE mean m'] - 1.96 * sd_pred, 'high': r['RE mean m'] + 1.96 * sd_pred,
                     'P(19-day mean > 0) model': 1 - N.cdf(-r['RE mean m'] / sd_pred), 'empirical share of 19-day blocks > 0': (blocks > 0).mean(),
                     'n blocks': len(blocks)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print()
    print('=' * 110)
    print('4. Recent periods against the earlier ones, and the live day 10/27')
    print('=' * 110)
    for name, X in D.items():
        ix = pd.DatetimeIndex(X.index); a = X.AR
        pre = a[ix < pd.Timestamp('2026-06-01')]; post = a[ix >= pd.Timestamp('2026-06-01')]; octo = a[ix >= pd.Timestamp('2026-10-01')]
        z = (post.mean() - pre.mean()) / np.sqrt(L.nw_se(post.values) ** 2 + L.nw_se(pre.values) ** 2)
        line = (f'  {name}: 2022-01..2026-05 {pre.mean():.2f}; 2026-06..10-22 {post.mean():.2f} (z of the difference {z:+.2f}); '
                f'October (15 days) {octo.mean():.2f}')
        if name in LIVE_1027:
            v = LIVE_1027[name]
            pct = (a.values <= v).mean()
            line += f'; live 10/27 {v:.1f} bp = {(v - a.mean()) / a.std():+.2f} daily sd, {100 * pct:.1f}% of walk-forward days were at least this bad'
        print(line)
    print(f'\ndone in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()

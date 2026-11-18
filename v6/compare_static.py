"""
compare_static.py -- compare static-leaderboard submissions (2026-06..09) with the competition metric, and put the
noise and the effect of selection on the static period next to every difference.

  python compare_static.py A.csv B.csv [C.csv ...] [--base A.csv] [--tried B.csv=20 C.csv=5]
                           [--start 2026-06-01] [--end 2026-09-30] [--boot 5000] [--out result.csv]

Metric (course slides): R = (2p - 1)(close - open)/open per stock-day; AR_t = average over the 50 stocks; score = mean
of AR_t over the days. bp/day below. timing_t = mean_i(2p_i - 1) * mean_i(r_i), xsec_t = AR_t - timing_t.

Output
  1. per file: days, AR, daily sd, SE, Newey-West t, timing, xsec, share of 1s, AR by month
  2. paired difference to --base (default: the first file) on the same days: diff, sd(d), SE = the larger of the iid
     and the Newey-West (5 lags) standard error (conservative: with 84 days NW alone can come out too small), t,
     95 % interval (stationary bootstrap, mean block 10 days), MDE (two-sided 5 %, power 80 %), and the chance that the
     file beats the base over 19 days if the static difference were the true one
  3. with --tried FILE=N (the author evaluated N settings on the static period and kept the best): the best of N
     equally good settings is ahead by SE_d * E[max of N standard normals] on average, so that amount is subtracted,
     and the one-sided Bonferroni bar z(1 - 0.05/N) * SE_d is shown
Labels come from data/market/*.csv (etl:adj_open, etl:adj_close; the same-day ratio is unaffected by adjustments).
Placeholder rows (-1) of the prediction day are ignored. Only numpy and pandas are needed.
"""
import os, sys, json, argparse
from statistics import NormalDist
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); DATA = os.path.join(ROOT, 'data')
N01 = NormalDist()


def load_sub(path):
    s = pd.read_csv(path, dtype={'id': str}, encoding='utf-8-sig')
    s['id'] = s['id'].astype(str).str.strip()
    L = s.melt(id_vars='id', var_name='date', value_name='p')
    L['date'] = pd.to_datetime(L.date)
    bad = ~L.p.isin([0, 1])
    if bad.any():
        raise ValueError(f'{path}: {int(bad.sum())} cells are not 0/1')
    return L


def load_returns(ids):
    rows = []
    for sid in ids:
        m = pd.read_csv(os.path.join(DATA, 'market', f'{sid}.csv'), parse_dates=['date'])
        o, c = m['etl:adj_open'], m['etl:adj_close']
        ok = (o > 0) & (c > 0)
        rows.append(pd.DataFrame({'id': sid, 'date': m.date[ok], 'r': ((c - o) / o)[ok]}))
    return pd.concat(rows, ignore_index=True)


def daily(L, R, start, end):
    x = L.merge(R, on=['id', 'date'])
    x = x[(x.date >= start) & (x.date <= end)].copy()
    x['s'] = 2 * x.p - 1
    x['sr'] = x.s * x.r
    g = x.groupby('date')
    AR = g.sr.mean() * 1e4
    T = g.s.mean() * g.r.mean() * 1e4
    return pd.DataFrame({'AR': AR, 'timing': T, 'xsec': AR - T, 'share1': g.p.mean(), 'n_stocks': g.size()})


def nw_se(d, lags=5):
    d = np.asarray(d, float); n = len(d); e = d - d.mean(); s = e @ e / n
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return np.sqrt(max(s, 0.0) / n)


def sboot_means(d, B=5000, block=10, seed=0):
    """stationary bootstrap (Politis & Romano 1994) of the mean of d"""
    d = np.asarray(d, float); n = len(d); p = 1.0 / block; rng = np.random.default_rng(seed)
    starts = rng.integers(n, size=(B, n)); jump = rng.random((B, n)) < p
    idx = np.empty((B, n), int); idx[:, 0] = starts[:, 0]
    for k in range(1, n):
        idx[:, k] = np.where(jump[:, k], starts[:, k], (idx[:, k - 1] + 1) % n)
    return d[idx].mean(axis=1)


def emax(N):
    """E[max of N iid standard normals] = integral of z * N * phi(z) * Phi(z)^(N-1)"""
    if N <= 1:
        return 0.0
    z = np.linspace(-8, 8, 8001); dz = z[1] - z[0]
    phi = np.exp(-z * z / 2) / np.sqrt(2 * np.pi); Phi = np.array([N01.cdf(v) for v in z])
    f = z * N * phi * Phi ** (N - 1)
    return float((f[:-1] + f[1:]).sum() * dz / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--base', default=None)
    ap.add_argument('--tried', nargs='*', default=[], help='FILE=N: N settings were evaluated on the static period')
    ap.add_argument('--start', default='2026-06-01'); ap.add_argument('--end', default='2026-09-30')
    ap.add_argument('--boot', type=int, default=5000); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    tried = {}
    for t in a.tried:
        f, n = t.rsplit('=', 1); tried[os.path.abspath(f)] = int(n)
    files = list(dict.fromkeys(a.files if a.base is None or a.base in a.files else [a.base] + a.files))
    base = a.base or files[0]
    subs = {f: load_sub(f) for f in files}
    ids = sorted(set().union(*[set(s.id) for s in subs.values()]))
    top50 = json.load(open(os.path.join(DATA, 'info.json')))['Top50@Y']
    if set(ids) != set(top50):
        print(f'warning: ids differ from info.json Top50@Y (extra {sorted(set(ids) - set(top50))}, missing {sorted(set(top50) - set(ids))})')
    R = load_returns(ids)
    D = {f: daily(s, R, start, end) for f, s in subs.items()}
    B = subs[base]
    for lab, v in (('(always 0)', 0), ('(always 1)', 1)):
        D[lab] = daily(B.assign(p=v), R, start, end)
    days = D[base].index
    print(f'scored days: {days.min().date()} .. {days.max().date()} ({len(days)} days); base = {os.path.basename(base)}\n')

    rows = []
    for f, x in D.items():
        x = x.reindex(days)
        se = x.AR.std() / np.sqrt(len(x))
        r = {'file': os.path.basename(f), 'days': int(x.AR.notna().sum()), 'AR': x.AR.mean(), 'sd': x.AR.std(), 'SE': se,
             't_NW': x.AR.mean() / nw_se(x.AR.dropna()), 'timing': x.timing.mean(), 'xsec': x.xsec.mean(), 'share1': x.share1.mean()}
        for m, v in x.AR.groupby(x.index.to_period('M')).mean().items():
            r[str(m)] = v
        rows.append(r)
    T1 = pd.DataFrame(rows)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print('1. each file (bp/day)'); print(T1.round(2).to_string(index=False))

    rows = []
    b = D[base].AR.reindex(days)
    for f, x in D.items():
        if f == base:
            continue
        d = (x.AR.reindex(days) - b).dropna()
        se = max(nw_se(d), d.std() / np.sqrt(len(d))); bt = sboot_means(d.values, B=a.boot)
        r = {'file': os.path.basename(f), 'diff': d.mean(), 'sd(d)': d.std(), 'SE': se, 't': d.mean() / se if se > 0 else np.nan,
             'CI95_lo': np.quantile(bt, 0.025), 'CI95_hi': np.quantile(bt, 0.975), 'MDE80': 2.80 * se,
             'd_timing': (x.timing.reindex(days) - D[base].timing.reindex(days)).mean(),
             'd_xsec': (x.xsec.reindex(days) - D[base].xsec.reindex(days)).mean(),
             'P(win 19d)': N01.cdf(d.mean() * np.sqrt(19) / d.std()) if d.std() > 0 else np.nan}
        N = tried.get(os.path.abspath(f)) if os.path.exists(f) else None
        if N:
            r['tried'] = N; r['selection_bias'] = se * emax(N); r['diff_adj'] = d.mean() - r['selection_bias']
            r['bar_Bonf'] = N01.inv_cdf(1 - 0.05 / N) * se
        rows.append(r)
    T2 = pd.DataFrame(rows)
    print(f'\n2. paired difference to the base on the same {len(days)} days (bp/day; CI = stationary bootstrap, block 10)')
    print(T2.round(2).to_string(index=False))
    print('\nreading: |t| < 2 or a CI that contains 0 means the static period cannot tell the two apart; '
          'MDE80 is the true difference that would be detected 80 % of the time.')
    if tried:
        print('selection: E[max of N standard normals] =', {n: round(emax(n), 2) for n in sorted(set(tried.values()))})
    if a.out:
        pd.concat([T1.assign(table='each'), T2.assign(table='paired')], ignore_index=True).to_csv(a.out, index=False)
        print(f'\nwritten {a.out}')


if __name__ == '__main__':
    main()

"""Independent AR check for 2026-11-11 (does not use phase2_check.py).

  python ar_check_1111.py COURSE.zip FINMIND_PRICES.csv SUBMISSION.csv [SUBMISSION.csv ...]

r = (adj_close - adj_open) / adj_open of each of the 50 stocks of sample_submission.csv on 11/11 from the course zip;
AR score = mean over the 50 stocks of (2p - 1) * r; timing = mean(2p - 1) * mean(r); xsec = AR - timing.
Baselines as on the board: Always down (p = 0) and Previous day (p = y of 11/10 = adj_close > adj_open).
The FinMind raw open / close give the same r (ratio); the largest difference is printed.
"""
import sys, zipfile, io
import numpy as np, pandas as pd

zpath, fm_path, subs = sys.argv[1], sys.argv[2], sys.argv[3:]
DAY, PREV = '2026-11-11', '2026-11-10'
z = zipfile.ZipFile(zpath)
names = {n.split('/')[-1][:-4]: n for n in z.namelist() if '/market/' in n and n.endswith('.csv')}
ids = pd.read_csv(sys.argv[3], dtype={'id': str})['id'].tolist()
r, yprev = {}, {}
for s in ids:
    m = pd.read_csv(io.BytesIO(z.read(names[s])))
    d = m[m.date == DAY].iloc[0]; p = m[m.date == PREV].iloc[0]
    assert d['etl:adj_open'] > 0 and d['etl:adj_close'] > 0
    r[s] = d['etl:adj_close'] / d['etl:adj_open'] - 1
    yprev[s] = int(p['etl:adj_close'] > p['etl:adj_open'])
r = pd.Series(r); yprev = pd.Series(yprev)
fm = pd.read_csv(fm_path, dtype={'id': str}).set_index('id')
rf = (fm.close / fm.open - 1).reindex(ids)
print(f'{DAY}: 50 stocks, Top50 mean r {r.mean() * 1e4:.2f} bp, {int((r > 0).sum())} up; '
      f'course zip vs FinMind largest |r difference| {float((r - rf).abs().max()):.1e}')

def score(p):
    s = 2 * p.astype(float) - 1
    ar = float((s * r).mean()); tm = float(s.mean() * r.mean())
    return ar, ar * 1e4, tm * 1e4, (ar - tm) * 1e4, float(p.mean()), int(((r > 0).astype(int) == p).sum())

rows = [('(baseline) Always down', score(pd.Series(0, index=ids))), ('(baseline) Previous day', score(yprev.reindex(ids)))]
for f in subs:
    sub = pd.read_csv(f, dtype={'id': str}).set_index('id')
    assert list(sub.columns) == [DAY] and list(sub.index) == ids, f
    rows.append((f.split('/')[-1], score(sub[DAY].astype(int).reindex(ids))))
print(f"{'AR score':>10} {'AR bp':>8} {'timing':>8} {'xsec':>8} {'1s':>5} {'hits':>5}  file")
for name, (ar, bp, tm, xs, s1, h) in rows:
    print(f'{ar:10.6f} {bp:8.2f} {tm:8.2f} {xs:8.2f} {s1:5.2f} {h:3d}/50  {name}')

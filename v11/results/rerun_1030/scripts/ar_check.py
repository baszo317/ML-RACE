"""Independent AR check (not the user's phase2_check.py): AR = mean over the 50 template stocks of (2p - 1)(close - open)/open,
prices = adj_open / adj_close of the day in the course zip.

  python ar_check.py ZIP TEMPLATE DAY file1.csv [file2.csv ...]
"""
import sys, io, zipfile
import numpy as np, pandas as pd

zp, tpl, day = sys.argv[1], sys.argv[2], sys.argv[3]
files = sys.argv[4:]
stocks = pd.read_csv(tpl, dtype={'id': str})['id'].tolist()
z = zipfile.ZipFile(zp)
rows, prev_y = {}, {}
for s in stocks:
    df = pd.read_csv(io.BytesIO(z.read(f'data/market/{s}.csv')))
    df = df[df['etl:adj_open'] != -1]
    r = df[df.date == day].iloc[0]
    rows[s] = (r['etl:adj_close'] - r['etl:adj_open']) / r['etl:adj_open']
    prev_y[s] = int(df[df.date < day].iloc[-1]['y'])
r = pd.Series(rows)
print(f'{day}: {len(r)} stocks, mean intraday return {r.mean() * 1e4:+.2f} bp, {int((r > 0).sum())} closed above the open')

def score(p, name):
    p = pd.Series(p).reindex(stocks).astype(float)
    a = (2 * p - 1) * r
    timing = (2 * p - 1).mean() * r.mean()
    print(f'{name:<45s} AR {a.mean():+.6f} ({a.mean() * 1e4:+8.2f} bp)  timing {timing * 1e4:+8.2f}  xsec {(a.mean() - timing) * 1e4:+8.2f}  '
          f'share1 {p.mean():.2f}  hits {int(((p == 1) == (r > 0)).sum())}/50')

score({s: 0 for s in stocks}, '(baseline) Always down')
score(prev_y, '(baseline) Previous day')
for f in files:
    S = pd.read_csv(f, dtype={'id': str}).set_index('id')
    assert list(S.columns) == [day], (f, list(S.columns))
    score(S[day].to_dict(), f.split('/')[-1])

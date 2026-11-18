"""
score_static.py -- score a submission with the competition metric on the dates whose labels are in data/market
  python score_static.py submission_ensemble.csv
AR = mean over days of the 50-stock average of (2p-1)(close-open)/open.  Reporting only: do NOT tune models on the
static-leaderboard period (Jun-Sep 2026) -- that would make the score meaningless for the live (private) test.
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(os.path.dirname(HERE), 'data')
sub = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'submission_ensemble.csv'), dtype={'id': str})
long = sub.melt(id_vars='id', var_name='date', value_name='p'); long['date'] = pd.to_datetime(long.date)
rows = []
for sid in sub['id']:
    m = pd.read_csv(os.path.join(DATA, 'market', f'{sid}.csv'), parse_dates=['date'])
    rows.append(pd.DataFrame({'id': sid, 'date': m.date, 'r': (m['etl:adj_close'] - m['etl:adj_open']) / m['etl:adj_open']}))
x = long.merge(pd.concat(rows), on=['id', 'date'])
x['R'] = (2 * x.p - 1) * x.r
d = x.groupby('date').R.mean()
print(f'scored dates: {d.index.min().date()} .. {d.index.max().date()} ({len(d)} days, {x.id.nunique()} stocks)')
print(f'AR = {d.mean():.6f}  ({d.mean() * 1e4:.2f} bp per day);  by month (bp/day): {(d.groupby(d.index.to_period("M")).mean() * 1e4).round(2).to_dict()}')
print(f'reference always-0 on the same days: {(-x.groupby("date").r.mean()).mean() * 1e4:.2f} bp per day')

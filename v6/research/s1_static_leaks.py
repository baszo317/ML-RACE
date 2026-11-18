"""s1 (v6, static only): what static-leaderboard scores are made of. Scores of simple rules on the static period
(2026-06-01..09-30, Top50, bp/day) computed with the competition metric from data/market/*.csv:
  * the two leaderboard baselines (to confirm that this scoring = the leaderboard's: Previous day 0.000722, Always down -0.000204)
  * rules that use day-t information, which the static rule forbids ("use 6/1 to predict 6/2, not the other way round"):
    today's close (or anything computed from it: the RSI / SMA / MACD columns of the same row) and today's open
  * an allowed rule for comparison
For calibration only: none of these is a submission. run from this folder: python s1_static_leaks.py > results/s1_static_leaks.txt"""
import os, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
top50 = json.load(open(os.path.join(ROOT, 'data', 'info.json')))['Top50@Y']
rows = []
for sid in top50:
    m = pd.read_csv(os.path.join(ROOT, 'data', 'market', f'{sid}.csv'), parse_dates=['date'])
    m = m[m['etl:adj_open'] > 0].sort_values('date').reset_index(drop=True)
    c, o = m['etl:adj_close'], m['etl:adj_open']
    rows.append(pd.DataFrame({'date': m.date, 'sid': sid, 'r': (c - o) / o, 'y_prev': m.y.shift(1),
                              'ret1_t': c / c.shift() - 1, 'rsi_t': m.RSI.diff(), 'macd_t': m.MACD.diff(), 'gap_t': o / c.shift() - 1,
                              'rsi_prev': m.RSI.diff().shift(1)}))
X = pd.concat(rows); X = X[(X.date >= '2026-06-01') & (X.date <= '2026-09-30')].copy()
X['gap_c'] = X.gap_t - X.groupby('date').gap_t.transform('mean')
sg = lambda v: np.where(v > 0, 1, -1)
def ar(s): return ((s * X.r).groupby(X.date).mean() * 1e4).mean()
print(f'static period: {X.date.min().date()} .. {X.date.max().date()}, {X.date.nunique()} days, {X.sid.nunique()} stocks; bp/day\n')
print('leaderboard baselines (this scoring reproduces them):')
print(f'  Previous day  (predict y_t-1)               {ar(2 * X.y_prev - 1):8.2f}   leaderboard 0.000722 = 7.22')
print(f'  Always down   (all 0)                       {ar(-1 + 0 * X.r):8.2f}   leaderboard -0.000204 = -2.04')
print('\nforbidden for static (day-t information):')
print(f'  perfect foresight (close_t > open_t)        {ar(sg(X.r)):8.2f}')
print(f'  close_t > close_t-1 (today\'s close)         {ar(sg(X.ret1_t)):8.2f}')
print(f'  RSI_t > RSI_t-1 (same-row RSI)              {ar(sg(X.rsi_t)):8.2f}')
print(f'  MACD_t > MACD_t-1 (same-row MACD)           {ar(sg(X.macd_t)):8.2f}')
print(f'  gap_t < 0 (today\'s open)                    {ar(-sg(X.gap_t)):8.2f}')
print(f'  gap_t below the day median (today\'s open)   {ar(-sg(X.gap_c)):8.2f}')
print('\nallowed, for comparison:')
print(f'  RSI_t-1 > RSI_t-2                           {ar(sg(X.rsi_prev)):8.2f}')
print('  v4 static upload (rolling, every day trained up to the day before): 15.36 (compare_static.py)')
print('  v4 + the real 09:00 open as a model input (v7 e6, upper bound, not allowed for static): 17.37')

"""e7 (H2, README_v7 section 6): TAIEX futures night-session features, merged into the core panel for the market-view
MLP members. FinMind dates a night ('after_market') row by the trading day it belongs to: the row dated D runs from
15:00 of the previous trading day to 05:00 of D (checked below: its open ~ the previous day-session close of the same
contract), so it is known before 09:00 of D.
  contract    : the TX contract with the largest volume in that night session
  night_ret   = log(night close / day-session close of the same contract on the previous trading day)
  night_range = (night high - night low) / that day-session close
writes ../cache/panel_night.parquet; run from this folder: python e7_build_night_panel.py > results/e7_build_night_panel.txt"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); V7 = os.path.dirname(HERE); ROOT = os.path.dirname(V7)
sys.path.insert(0, os.path.join(ROOT, 'v5'))
import numpy as np, pandas as pd
import ens_lib as E
tx = pd.read_csv(os.path.join(V7, 'data', 'tx_daily.csv'), dtype={'contract_date': str})
tx['date'] = pd.to_datetime(tx.date).astype('datetime64[ns]')
tx = tx[(tx.futures_id == 'TX') & tx.contract_date.str.fullmatch(r'\d{6}')]          # monthly contracts only (no spreads)
day = tx[tx.trading_session == 'position'][['date', 'contract_date', 'open', 'close']]
ngt = tx[(tx.trading_session == 'after_market') & (tx.volume > 0)]
ngt = ngt.sort_values(['date', 'volume'], ascending=[True, False]).drop_duplicates('date')[['date', 'contract_date', 'open', 'max', 'min', 'close', 'volume']]
ddates = np.sort(day.date.unique())
i = np.searchsorted(ddates, ngt.date.values) - 1                                   # previous day-session date
ngt = ngt[i >= 0].copy(); ngt['prev_day'] = ddates[i[i >= 0]]
ref = day.rename(columns={'date': 'prev_day', 'close': 'prev_close', 'open': 'prev_open'})
N = ngt.merge(ref, on=['prev_day', 'contract_date'], how='left')
N['night_ret'] = np.log(N.close / N.prev_close); N['night_range'] = (N['max'] - N['min']) / N.prev_close
chk = (N.open / N.prev_close - 1).abs()
print(f'night sessions: {len(N)} ({N.date.min().date()} .. {N.date.max().date()}); matched to a previous day close: {N.prev_close.notna().mean():.3f}')
print(f'dating check |night open / previous day close - 1|: median {chk.median():.5f}, p95 {chk.quantile(.95):.5f} (small = the row dated D is the night before D)')
alt = N.merge(day.rename(columns={'close': 'same_close'})[['date', 'contract_date', 'same_close']], on=['date', 'contract_date'], how='left')
print(f'   for comparison |night open / same-date day close - 1|: median {(alt.open / alt.same_close - 1).abs().median():.5f}')
P = E.build_panel(E.DATA); P['date'] = P.date.astype('datetime64[ns]')
P = P.merge(N[['date', 'night_ret', 'night_range']], on='date', how='left')
fs = E.feature_sets()
keep = list(dict.fromkeys(['date', 'stock_id', 'target_r', 'y', 'resid_r', 'y_resid', 'top50'] + fs['B+XR'] + ['night_ret', 'night_range']))
Q = P[keep].copy(); num = [c for c in keep if c not in ('date', 'stock_id', 'top50')]; Q[num] = Q[num].astype('float32')
Q.to_parquet(os.path.join(V7, 'cache', 'panel_night.parquet'), index=False, compression='zstd')
D = P.groupby('date').agg(night_ret=('night_ret', 'first'), night_range=('night_range', 'first'), ixic=('us_ixic_ret1', 'first'),
                         m=('target_r', 'mean')).dropna()
print(f'panel days with night data: {P.groupby("date").night_ret.first().notna().mean():.3f} of {P.date.nunique()} days')
print(f'descriptive: corr(night_ret, Nasdaq previous session) {D.night_ret.corr(D.ixic):.3f}; corr(night_ret, Top100 intraday same day) {D.night_ret.corr(D.m):.3f}; '
      f'corr(night_range, |intraday|) {D.night_range.corr(D.m.abs()):.3f}')

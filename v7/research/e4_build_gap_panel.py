"""e4: core panel (competition Top100, all v4 features) + the SAME-DAY opening gap, for an upper-bound study of a
pre-open indicative-price feature (live only; the static rule forbids day-t data, so nothing here goes into v6).
  gap_today    = log(adj_open_t / adj_close_{t-1})            (known only at 09:00; live proxy = 08:5x MIS indicative price)
  xr_gap_today = cross-sectional percentile rank of gap_today among the 100 stocks of the day, centred at 0
writes ../cache/panel_gap.parquet (training data for e5 in the cloud)"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); V7 = os.path.dirname(HERE); ROOT = os.path.dirname(V7)
sys.path.insert(0, os.path.join(ROOT, 'v5'))
import numpy as np, pandas as pd
import ens_lib as E
P = E.build_panel(E.DATA); P['date'] = P.date.astype('datetime64[ns]')
rows = []
for sid in P.stock_id.unique():
    m = pd.read_csv(os.path.join(ROOT, 'data', 'market', f'{sid}.csv'), parse_dates=['date'], usecols=['date', 'etl:adj_open', 'etl:adj_close'])
    m = m[(m['etl:adj_open'] > 0) & (m['etl:adj_close'] > 0)].sort_values('date')
    rows.append(pd.DataFrame({'date': m.date.astype('datetime64[ns]'), 'stock_id': sid, 'gap_today': np.log(m['etl:adj_open'] / m['etl:adj_close'].shift(1))}))
P = P.merge(pd.concat(rows), on=['date', 'stock_id'], how='left')
P['xr_gap_today'] = P.groupby('date').gap_today.rank(pct=True) - 0.5
fs = E.feature_sets()
keep = list(dict.fromkeys(['date', 'stock_id', 'target_r', 'y', 'resid_r', 'y_resid', 'top50'] + fs['B+XR'] + ['gap_today', 'xr_gap_today']))
Q = P[keep].copy(); num = [c for c in keep if c not in ('date', 'stock_id', 'top50')]; Q[num] = Q[num].astype('float32')
Q.to_parquet(os.path.join(V7, 'cache', 'panel_gap.parquet'), index=False, compression='zstd')
json.dump({'B': fs['B'], 'B+XR': fs['B+XR']}, open(os.path.join(V7, 'cache', 'feature_sets.json'), 'w'))
print(Q.shape, 'gap_today missing (labelled rows):', round(float(Q[Q.target_r.notna()].gap_today.isna().mean()), 4))

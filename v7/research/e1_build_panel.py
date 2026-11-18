"""e1: build the extended training panel (core = competition Top100, all v4 features unchanged; extra = point-in-time
Top300 public stocks) and write it to cache/panel_ext.parquet for the walk-forward training (e2)"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); V7 = os.path.dirname(HERE); sys.path.insert(0, V7)
import numpy as np, pandas as pd
import panel_ext as PX
t = time.time()
Q, U, extra = PX.build()
os.makedirs(os.path.join(V7, 'cache'), exist_ok=True)
Q.to_parquet(os.path.join(V7, 'cache', 'panel_ext.parquet'), index=False, compression='zstd')
print(f'{time.time() - t:.0f}s; rows {len(Q)} (core {int(Q.core.sum())}, extra {int((~Q.core).sum())}); extra stocks {len(extra)}')
print('extra stocks in the universe by year:', {y: len(s - set(Q[Q.core].stock_id)) for y, s in U.items()})
x = Q[~Q.core]; print('extra rows by year:', x.groupby(x.date.dt.year).size().to_dict())
print('B feature missing share, extra rows (top 8):'); print(x.iloc[:, 8:].isna().mean().sort_values(ascending=False).head(8).round(3).to_string())

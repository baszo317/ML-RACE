"""
make_train_set.py -- write the best-feature training set (run after build_features.py)

Input : panel_candidates.parquet (all 123 candidate features), final_feature_lists.json
Output: train_best.parquet and train_best.csv
  rows    : 100 stocks (Top100@Y), 2020-03-02 .. 2026-05-29  (nothing from Jun-Sep 2026)
  columns : date, stock_id, top50, new_data_covered, y, target_r, sample_weight, exdiv_today_flag,
            then the 89 features of final_feature_lists.json["F_best"]
  * y            : 1 if close > open on that day (label)
  * target_r     : intraday return (close - open) / open; the competition score is mean((2p-1) * target_r)
  * sample_weight: |target_r| clipped at 0.05 -> training a classifier with this weight optimises the score
  * exdiv_today_flag: 1 on ex-dividend days -> NOT a model feature; override the prediction to 0 on those days
  * new_data_covered = False for 8069, 3293, 6223, 3529, 6488, 5347, 5274, 8299, 6274, 2812 (no new data,
    their nd_* features are NaN; LightGBM handles NaN natively)
  * Competition-data-only fallback: use final_feature_lists.json["B_competition_only"] (drop all nd_* columns)
"""
import os, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P = pd.read_parquet(os.path.join(HERE, 'panel_candidates.parquet'))
L = json.load(open(os.path.join(HERE, 'final_feature_lists.json')))
F = L['F_best']
missing = set(F) - set(P.columns); assert not missing, missing
assert pd.to_datetime(P.date).max() <= pd.Timestamp('2026-05-31'), 'panel must stop before June 2026'

T = P[['date', 'sid', 'top50', 'new_data_covered', 'y', 'target_r'] + F].rename(columns={'sid': 'stock_id'}).copy()
T.insert(6, 'sample_weight', T.target_r.abs().clip(upper=0.05).astype('float32'))
T.insert(7, 'exdiv_today_flag', P['nd_exdiv_today'].fillna(0).astype('int8').values)
T = T.sort_values(['date', 'stock_id']).reset_index(drop=True)

T.to_parquet(os.path.join(HERE, 'train_best.parquet'), index=False)
T.to_csv(os.path.join(HERE, 'train_best.csv'), index=False, encoding='utf-8-sig', float_format='%.6g')
print(f'train_best: {T.shape[0]} rows x {T.shape[1]} columns  ({len(F)} features)  '
      f'{T.date.min()} .. {T.date.max()}  stocks {T.stock_id.nunique()}  top50 rows {int(T.top50.sum())}')

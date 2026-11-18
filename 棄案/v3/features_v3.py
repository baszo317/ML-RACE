r"""
features_v3.py -- the 5 new stock-picking (cross-sectional) features of v3, all known before 09:00 of day D.

  gap_bias_60  : mean overnight gap log(open_d / close_{d-1}) over the last 60 trading days (at least 30)
  gap_bias_250 : the same over the last 250 trading days (at least 120)
                 -> stocks whose gains come overnight tend to lose intraday ("tug of war", Lou, Polk & Skouras 2019)
  resid_vol_60 : std of the residual intraday return (close/open - 1 minus the same-day mean of the 100 stocks),
                 last 60 days (at least 30) -> high idiosyncratic volatility, weak intraday (Ang et al. 2006)
  dow_bias_e   : mean residual intraday return (close/open - 1 minus the same-day mean of the 100 stocks) of this stock
                 on the same weekday, last 52 such days (at least 20) -> return seasonality (Keloharju et al. 2016)
  hi52         : close_{d-1} / highest close of the last 250 trading days - 1   (George & Hwang 2004)
  xr_<name>    : percentile rank among the 100 stocks of the same day, centred at 0 (as the xr_* of features_extra)

Timing (no look-ahead): every value on row D is computed from rows dated D-1 or earlier (shift by one row of the same
stock; dow_bias_e uses earlier same-weekday rows only). Screening evidence (daily Spearman IC with the residual
intraday return, Top50, 2020-03 .. 2026-10): research\screen_candidates.py and README_v3.md.
"""
import os
import numpy as np, pandas as pd
from comp_features import read_market          # v1 reader (-1 placeholder rows -> NaN), from ..\transformer

NEW = ['gap_bias_60', 'gap_bias_250', 'resid_vol_60', 'dow_bias_e', 'hi52']
XRN = ['xr_' + c for c in NEW]

def _raw_prices(data_dir, sids):
    R = []
    for s in sids:
        df = read_market(os.path.join(data_dir, 'market', f'{s}.csv'))
        R.append(pd.DataFrame({'date': df.date, 'stock_id': s, '_o': df['etl:adj_open'], '_c': df['etl:adj_close']}))
    return pd.concat(R, ignore_index=True)

def add_v3(P, data_dir):
    """P: panel of ens_lib.build_panel (one row per date x stock, incl. the day(s) to predict). Returns P + NEW + XRN."""
    sids = sorted(P.stock_id.unique())
    Q = P[['date', 'stock_id', 'target_r']].merge(_raw_prices(data_dir, sids), on=['date', 'stock_id'], how='left')
    Q = Q.sort_values(['stock_id', 'date']).reset_index(drop=True)
    g = Q.groupby('stock_id', sort=False)
    Q['_gap'] = np.log(Q._o) - g._c.shift(1).pipe(np.log)                     # log(open_d / close_{d-1})
    Q['_e'] = Q.target_r - Q.groupby('date').target_r.transform('mean')       # residual intraday return of day d
    g = Q.groupby('stock_id', sort=False)
    roll = lambda col, n, mp, fn: g[col].transform(lambda s: getattr(s.rolling(n, min_periods=mp), fn)().shift(1))
    Q['gap_bias_60'] = roll('_gap', 60, 30, 'mean')
    Q['gap_bias_250'] = roll('_gap', 250, 120, 'mean')
    Q['resid_vol_60'] = roll('_e', 60, 30, 'std')
    Q['hi52'] = g._c.transform(lambda s: (s / s.rolling(250, min_periods=120).max() - 1).shift(1))
    Q['_dow'] = Q.date.dt.dayofweek
    Q['dow_bias_e'] = Q.groupby(['stock_id', '_dow'], sort=False)._e.transform(lambda s: s.shift(1).rolling(52, min_periods=20).mean())
    for c in NEW:
        Q['xr_' + c] = Q.groupby('date')[c].rank(pct=True) - 0.5
    keep = Q[['date', 'stock_id'] + NEW + XRN]
    P = P.drop(columns=[c for c in NEW + XRN if c in P.columns]).merge(keep, on=['date', 'stock_id'], how='left')
    return P.sort_values(['date', 'stock_id']).reset_index(drop=True)

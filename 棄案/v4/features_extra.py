"""
features_extra.py -- extra inputs (v2 code; v4 only adds data/ref/exdiv_history.csv to the ex-dividend sources),
all known before 09:00 of day D (no look-ahead):
  * xr_*   : cross-sectional percentile rank (among the 100 stocks of the same day, centred at 0) of 30 stock features
             that comp_features already shifts by one day
  * cal_*  : trading-calendar flags (Monday, Friday, before / after a holiday >= 3 days, TX settlement Wednesday,
             first 3 trading days of the month, last trading day of the month); the calendar is published in advance
  * ex_*   : ex-dividend day / day before / 1 and 2 days after (ex-dates are announced weeks ahead)
  * resid_r, y_resid : intraday return minus the same-day mean over the 100 stocks (training target of the
             cross-sectional model; never an input)
"""
import os
import numpy as np, pandas as pd

XR_BASE = ['ret1', 'ret5', 'ret20', 'intraday_ret', 'gap', 'range', 'clv', 'upper_shadow', 'lower_shadow', 'vol_ratio', 'amt_ratio',
           'trades_ratio', 'foreign_net', 'foreign_net_5d', 'trust_net', 'trust_net_5d', 'dealer_self_net', 'dealer_hedge_net',
           'foreign_hold_chg5', 'intra_bias_20', 'intra_bias_60', 'intra_bias_250', 'gap_bias_20', 'RSI', 'close_SMA30', 'BB_pctb',
           'ATR_pct', 'vol_ratio_5_60', 'beta_x_us', 'y_mean5']
XR = ['xr_' + c for c in XR_BASE]
CAL = ['cal_dow0', 'cal_dow4', 'cal_pre_holiday', 'cal_post_holiday', 'cal_settle', 'cal_tom', 'cal_month_end']
EXDIV = ['ex_today', 'ex_next1', 'ex_prev1', 'ex_prev2']
MKT_EXTRA = ['mkt_intra_20', 'mkt_absr_20']

def add_xrank(P):
    g = P.groupby('date')
    R = pd.DataFrame({'xr_' + c: g[c].rank(pct=True) - 0.5 for c in XR_BASE}, index=P.index)
    return pd.concat([P, R], axis=1)

def trading_calendar(data_dir, dates):
    """official calendar (data/ref/calendar.parquet) if it covers the requested dates, else the data's own dates"""
    f = os.path.join(data_dir, 'ref', 'calendar.parquet')
    if os.path.exists(f):
        try:
            cal = pd.to_datetime(pd.read_parquet(f).date).sort_values().drop_duplicates().reset_index(drop=True)
            if cal.max() > max(dates):
                return pd.Series(sorted(set(cal) | set(dates)))
        except Exception as e:                                   # e.g. pyarrow missing
            print(f'warning: cannot read {f} ({e}); calendar flags use the data dates only')
    return pd.Series(sorted(set(dates)))

def add_calendar(P, data_dir):
    dts = pd.Series(sorted(P.date.unique())); cal = trading_calendar(data_dir, list(dts))
    cv = cal.values; i = np.searchsorted(cv, dts.values)
    nxt = cv[np.minimum(i + 1, len(cv) - 1)]; prv = cv[np.maximum(i - 1, 0)]
    C = pd.DataFrame({'date': dts})
    C['cal_dow0'] = (dts.dt.dayofweek == 0).astype(float); C['cal_dow4'] = (dts.dt.dayofweek == 4).astype(float)
    C['cal_pre_holiday'] = ((nxt - dts.values) / np.timedelta64(1, 'D') > 3).astype(float)
    C['cal_post_holiday'] = ((dts.values - prv) / np.timedelta64(1, 'D') > 3).astype(float)
    C['cal_settle'] = ((dts.dt.dayofweek == 2) & dts.dt.day.between(15, 21)).astype(float)
    ym = dts.dt.to_period('M')
    C['cal_tom'] = (dts.groupby(ym).cumcount() < 3).astype(float)
    C['cal_month_end'] = (pd.Series(nxt).dt.to_period('M').values != ym.values).astype(float)
    return P.merge(C, on='date', how='left')

def exdiv_table(data_dir):
    """ex-dividend dates: history in data/ref/corporate_actions.parquet (if present, up to 2026-07-31) +
    data/ref/exdiv_history.csv (TWSE results table TWT49U, written by fetch_exdiv.py --history; 2026-06 onwards) +
    data/ref/exdiv_upcoming.csv (announcements, fetch_exdiv.py) + data/ref/exdiv_manual.csv (hand-made, stock_id,date)"""
    E = []
    f = os.path.join(data_dir, 'ref', 'corporate_actions.parquet')
    if os.path.exists(f):
        try:
            ca = pd.read_parquet(f); ca = ca[ca.source == 'dividend']
            E.append(pd.DataFrame({'stock_id': ca.stock_id.astype(str), 'date': pd.to_datetime(ca.date)}))
        except Exception as e:                                   # e.g. pyarrow missing: history only matters for backtests
            print(f'warning: cannot read {f} ({e}); past ex-dividend dates unknown (pip install pyarrow)')
    for g in ('exdiv_history.csv', 'exdiv_upcoming.csv', 'exdiv_manual.csv'):
        g = os.path.join(data_dir, 'ref', g)
        if os.path.exists(g):
            x = pd.read_csv(g, dtype={'stock_id': str})
            if len(x):
                E.append(pd.DataFrame({'stock_id': x.stock_id.str.strip(), 'date': pd.to_datetime(x.date)}))
    if not E:
        return pd.DataFrame({'stock_id': pd.Series(dtype=str), 'date': pd.Series(dtype='datetime64[ns]')})
    E = pd.concat(E, ignore_index=True); E['date'] = E.date.astype('datetime64[ns]')
    return E.drop_duplicates().reset_index(drop=True)

def add_exdiv(P, data_dir):
    E = exdiv_table(data_dir); E['_ex'] = 1.0
    P = P.merge(E, on=['stock_id', 'date'], how='left')
    P = P.assign(ex_today=P._ex.fillna(0.0)).drop(columns='_ex').sort_values(['stock_id', 'date']).reset_index(drop=True)
    g = P.groupby('stock_id').ex_today
    return P.assign(ex_next1=g.shift(-1).fillna(0.0), ex_prev1=g.shift(1).fillna(0.0), ex_prev2=g.shift(2).fillna(0.0))

def add_market_extra(P):
    d = P.groupby('date').target_r.mean().sort_index()
    x = pd.DataFrame({'mkt_intra_20': d.rolling(20, min_periods=10).mean().shift(1),
                      'mkt_absr_20': d.abs().rolling(20, min_periods=10).mean().shift(1)})
    return P.merge(x, left_on='date', right_index=True, how='left')

def add_residual_targets(P):
    m = P.groupby('date').target_r.mean().rename('mkt_r')
    P = P.merge(m, left_on='date', right_index=True, how='left')
    rr = P.target_r - P.mkt_r
    P = P.assign(resid_r=rr, y_resid=(rr > 0).astype(float).where(rr.notna())).sort_values(['stock_id', 'date']).reset_index(drop=True)
    return P.assign(absres_60=P.groupby('stock_id').resid_r.transform(lambda s: s.abs().rolling(60, min_periods=20).mean().shift(1)))

def add_all(P, data_dir):
    P = add_xrank(P.copy()).copy(); P = add_calendar(P, data_dir).copy(); P = add_exdiv(P, data_dir).copy()
    P = add_market_extra(P).copy(); P = add_residual_targets(P).copy()
    return P.sort_values(['date', 'stock_id']).reset_index(drop=True)

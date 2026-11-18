"""
ext_data.py -- rebuild competition-format stock files (the 20 columns of data/market/<id>.csv) for stocks outside the
competition's Top100, from the public data already downloaded by the old pipeline (棄案/data):
  raw/finmind/TaiwanStockPrice              open, max, min, close, Trading_Volume, Trading_money, Trading_turnover
  pit/price.parquet                         adjusted open/high/low/close (dividend- and split-adjusted)
  raw/finmind/TaiwanStockShareholding       ForeignInvestmentSharesRatio (= 全體外資及陸資持股比率), NumberOfSharesIssued
  raw/finmind/TaiwanStockInstitutionalInvestorsBuySell   buy - sell of Foreign_Investor / Investment_Trust / Dealer_self / Dealer_Hedging
Technical indicators recomputed on the adjusted prices with the parameters that reproduce the competition columns
exactly (checked on 2330, 2882, 1101; median relative error 0 after the warm-up):
  SMA = 30-day simple mean of close; RSI = Wilder 14; MACD = EMA12 - EMA26 - EMA9 of that (histogram);
  ATR = Wilder 14 of the true range; BBANDS = SMA5 +- 2 x population sd (5 days).
All data end on 2026-07-31 (date of the old download); enough for walk-forward folds up to 2026-05.
"""
import os, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OLD = os.path.join(ROOT, '棄案', 'data'); RAW = os.path.join(OLD, 'raw', 'finmind')
COLS = ['date', 'etl:adj_close', 'etl:adj_open', 'etl:adj_high', 'etl:adj_low', 'price:成交筆數', 'price:成交金額', 'price:成交股數',
        'foreign_investors_shareholding:全體外資及陸資持股比率',
        'institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)',
        'institutional_investors_trading_summary:投信買賣超股數',
        'institutional_investors_trading_summary:自營商買賣超股數(自行買賣)',
        'institutional_investors_trading_summary:自營商買賣超股數(避險)',
        'RSI', 'SMA', 'MACD', 'ATR', 'BBANDS_upper', 'BBANDS_mid', 'BBANDS_lower', 'y']
FLOW = {'Foreign_Investor': COLS[9], 'Investment_Trust': COLS[10], 'Dealer_self': COLS[11], 'Dealer_Hedging': COLS[12]}

def _wilder(x, n): return x.ewm(alpha=1 / n, adjust=False).mean()

def indicators(o, h, l, c):
    d = c.diff()
    rsi = 100 - 100 / (1 + _wilder(d.clip(lower=0), 14) / _wilder((-d).clip(lower=0), 14))
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    mid = c.rolling(5).mean(); sd = c.rolling(5).std(ddof=0)
    return {'RSI': rsi, 'SMA': c.rolling(30).mean(), 'MACD': macd - macd.ewm(span=9, adjust=False).mean(), 'ATR': _wilder(tr, 14),
            'BBANDS_upper': mid + 2 * sd, 'BBANDS_mid': mid, 'BBANDS_lower': mid - 2 * sd}

_PIT = None
def _pit():
    global _PIT
    if _PIT is None:
        _PIT = pd.read_parquet(os.path.join(OLD, 'pit', 'price.parquet'), columns=['stock_id', 'date', 'adj_open', 'adj_high', 'adj_low', 'adj_close'])
        _PIT['stock_id'] = _PIT.stock_id.astype(str); _PIT['date'] = pd.to_datetime(_PIT.date)
    return _PIT

def available_ids():
    """stocks with all four sources"""
    s = None
    for ds in ('TaiwanStockPrice', 'TaiwanStockShareholding', 'TaiwanStockInstitutionalInvestorsBuySell'):
        ids = {f[:-8] for f in os.listdir(os.path.join(RAW, ds)) if f.endswith('.parquet')}
        s = ids if s is None else s & ids
    return sorted(s & set(_pit().stock_id.unique()))

def build(sid):
    """DataFrame with the competition's 21 columns (+ shares_issued) for one stock, all dates in the raw data"""
    p = pd.read_parquet(os.path.join(RAW, 'TaiwanStockPrice', f'{sid}.parquet'))
    p['date'] = pd.to_datetime(p.date); p = p[(p.open > 0) & (p.close > 0)].drop_duplicates('date').sort_values('date')
    a = _pit(); a = a[a.stock_id == sid].drop(columns='stock_id')
    x = p.merge(a, on='date', how='inner').sort_values('date').reset_index(drop=True)
    sh = pd.read_parquet(os.path.join(RAW, 'TaiwanStockShareholding', f'{sid}.parquet'))
    if not len(sh) or 'NumberOfSharesIssued' not in sh:
        sh = pd.DataFrame({'date': pd.Series(dtype='datetime64[ns]'), 'ForeignInvestmentSharesRatio': [], 'NumberOfSharesIssued': []})
    sh = sh[['date', 'ForeignInvestmentSharesRatio', 'NumberOfSharesIssued']].copy()
    sh['date'] = pd.to_datetime(sh.date); sh = sh.drop_duplicates('date')
    fl = pd.read_parquet(os.path.join(RAW, 'TaiwanStockInstitutionalInvestorsBuySell', f'{sid}.parquet'))
    x = x.merge(sh, on='date', how='left')
    if len(fl) and {'date', 'buy', 'sell', 'name'} <= set(fl.columns):                # some files are empty
        fl['date'] = pd.to_datetime(fl.date); fl['net'] = fl.buy - fl.sell
        fl = fl[fl.name.isin(FLOW)].pivot_table(index='date', columns='name', values='net', aggfunc='sum').rename(columns=FLOW).reset_index()
        x = x.merge(fl, on='date', how='left')
    o, h, l, c = x.adj_open, x.adj_high, x.adj_low, x.adj_close
    out = pd.DataFrame({'date': x.date, COLS[1]: c, COLS[2]: o, COLS[3]: h, COLS[4]: l, COLS[5]: x.Trading_turnover.astype(float),
                        COLS[6]: x.Trading_money.astype(float), COLS[7]: x.Trading_Volume.astype(float), COLS[8]: x.ForeignInvestmentSharesRatio})
    for c_ in FLOW.values():
        out[c_] = x[c_] if c_ in x else np.nan
    for k, v in indicators(o, h, l, c).items():
        out[k] = v
    out['y'] = (c > o).astype(float)
    out['shares_issued'] = x.NumberOfSharesIssued
    return out[COLS + ['shares_issued']]

def industry():
    info = pd.read_parquet(os.path.join(RAW, 'TaiwanStockInfo', '_ALL.parquet'))
    info['stock_id'] = info.stock_id.astype(str)
    return info.sort_values('date').drop_duplicates('stock_id', keep='last').set_index('stock_id')[['industry_category', 'type']]

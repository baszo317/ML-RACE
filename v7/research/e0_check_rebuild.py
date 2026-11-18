"""e0: rebuild the competition's own 100 stocks from the public sources and compare with data/market/<id>.csv on
2020-01-02..2026-07-31, in the form comp_features uses them (prices as returns / ratios, indicators relative to the close,
because the two price adjustments use different anchor dates)"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); V7 = os.path.dirname(HERE); ROOT = os.path.dirname(V7); sys.path.insert(0, V7)
import numpy as np, pandas as pd
import ext_data as X
C = X.COLS
ids = json.load(open(os.path.join(ROOT, 'data', 'info.json')))['Top100@Y']
have = set(X.available_ids()); print(f'competition Top100 with all public sources: {len(set(ids) & have)} / 100; public stocks with all sources: {len(have)}')
def feats(d, s):
    c, o, h, l = d[C[1] + s], d[C[2] + s], d[C[3] + s], d[C[4] + s]
    return {'ret1': c / c.shift() - 1, 'intraday': c / o - 1, 'gap': o / c.shift() - 1, 'range': (h - l) / c,
            'trades': d[C[5] + s], 'value': d[C[6] + s], 'volume': d[C[7] + s], 'foreign_hold': d[C[8] + s],
            'foreign_net': d[C[9] + s], 'trust_net': d[C[10] + s], 'dealer_self': d[C[11] + s], 'dealer_hedge': d[C[12] + s],
            'RSI': d['RSI' + s], 'close_SMA30': c / d['SMA' + s] - 1, 'MACD/c': d['MACD' + s] / c, 'ATR/c': d['ATR' + s] / c,
            'BB_width': (d['BBANDS_upper' + s] - d['BBANDS_lower' + s]) / d['BBANDS_mid' + s], 'y': d['y' + s]}
agree, mad = [], []
for sid in sorted(set(ids) & have):
    a = pd.read_csv(os.path.join(ROOT, 'data', 'market', f'{sid}.csv'), parse_dates=['date']); a = a[a[C[1]] > 0]
    m = a.merge(X.build(sid), on='date', suffixes=('_c', '_p'))
    m = m[(m.date >= '2020-01-02') & (m.date <= '2026-07-31')]
    fc, fp = feats(m, '_c'), feats(m, '_p')
    ra, rm = {'sid': sid, 'days': len(m)}, {'sid': sid}
    for k in fc:
        u, v = fc[k].values.astype(float), fp[k].values.astype(float); ok = np.isfinite(u) & np.isfinite(v)
        tol = 1e-3 * np.maximum(np.abs(u), 1e-9) + (1e-6 if k not in ('trades', 'value', 'volume', 'foreign_net', 'trust_net', 'dealer_self', 'dealer_hedge') else 0.5)
        ra[k] = float(np.mean(np.abs(u - v)[ok] <= tol[ok])) if ok.any() else np.nan
        rm[k] = float(np.nanmedian(np.abs(u - v)[ok] / np.maximum(np.abs(u[ok]), 1e-12))) if ok.any() else np.nan
    agree.append(ra); mad.append(rm)
A = pd.DataFrame(agree); M = pd.DataFrame(mad)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print('\nshare of stock-days that agree (relative 1e-3), over the stocks: mean / 5th percentile / min')
print(A.drop(columns=['sid', 'days']).agg(['mean', lambda s: s.quantile(.05), 'min']).set_axis(['mean', 'p05', 'min']).round(4).T.to_string())
print('\nstocks with any return / flow column below 0.95 agreement:')
cols = ['ret1', 'gap', 'trades', 'volume', 'foreign_net', 'trust_net', 'dealer_self', 'dealer_hedge', 'close_SMA30', 'ATR/c']
print(A[(A[cols] < 0.95).any(axis=1)][['sid', 'days'] + cols].round(3).to_string(index=False))
A.to_csv(os.path.join(HERE, 'results', 'e0_check_rebuild.csv'), index=False)

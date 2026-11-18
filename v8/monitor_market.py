r"""
monitor_market.py (v8) -- where do the "market" (大盤) gains and losses of the live predictions come from?

  python monitor_market.py                          reads ..\v5\out\history_predictions.csv (written by v5\daily.bat)
  python monitor_market.py --sub a.csv b.csv ...    or any submission files (id column + one column per date, 0/1)

Run it after update_data.py (so that the predicted days have their close in data\market). Only numpy + pandas.

For every predicted day with a known result (Top50, bp):
  AR        competition score
  timing    net exposure x market move  = mean(2d-1) x mean(r)                (the explicit market view, pbar / w_T)
  beta_bet  beta tilt x market move     = mean[(s - sbar)(beta - mean beta)] x mean(r)
            (the hidden market bet of the stock view: long low-beta / short high-beta loses on days the market rises)
  rest      AR - timing - beta_bet      (stock picking that does not depend on the market's direction)
  B         beta tilt of the book (walk-forward 2022-26 average of the v4 rule: -0.30; long side beta ~0.6, short side ~1.2)
beta = 120-day rolling beta of the stock's intraday return on the Top100 mean intraday return, using days before the
predicted day only (README_v8.md, section 3).  Walk-forward reference of the v4 rule, 2022-01..2026-10 (bp/day):
AR 13.38, timing 1.55, beta_bet 1.23, rest 10.60; daily sd: timing 21, beta_bet 26.
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); DATA = os.path.join(ROOT, 'data')

def intraday_returns(sids):
    R = []
    for sid in sids:
        m = pd.read_csv(os.path.join(DATA, 'market', f'{sid}.csv'), parse_dates=['date'])
        m = m[(m['etl:adj_open'] > 0) & (m['etl:adj_close'] > 0)]
        R.append(pd.DataFrame({'date': m.date, 'stock_id': sid, 'r': m['etl:adj_close'] / m['etl:adj_open'] - 1}))
    return pd.concat(R, ignore_index=True)

def rolling_beta(R):
    mk = R.groupby('date').r.mean().rename('m100')
    X = R.merge(mk, left_on='date', right_index=True).sort_values(['stock_id', 'date'])
    def _b(g):
        x, y = g.m100, g.r
        cv = (x * y).rolling(120, min_periods=60).mean() - x.rolling(120, min_periods=60).mean() * y.rolling(120, min_periods=60).mean()
        return (cv / x.rolling(120, min_periods=60).var(ddof=0)).shift(1)       # days before t only
    X['beta'] = X.groupby('stock_id', group_keys=False)[['m100', 'r']].apply(_b)
    return X[['date', 'stock_id', 'beta']]

def read_predictions(subs):
    if not subs:
        f = os.path.join(ROOT, 'v5', 'out', 'history_predictions.csv')
        if not os.path.exists(f):
            sys.exit('no ..\\v5\\out\\history_predictions.csv yet: run v5\\daily.bat first, or pass --sub <submission files>')
        H = pd.read_csv(f, dtype={'stock_id': str}, parse_dates=['date'])
        return H[['date', 'stock_id', 'pred']]
    out = []
    for f in subs:
        S = pd.read_csv(f, dtype={'id': str})
        L = S.melt(id_vars='id', var_name='date', value_name='pred').rename(columns={'id': 'stock_id'})
        L['date'] = pd.to_datetime(L.date); out.append(L)
    H = pd.concat(out, ignore_index=True).drop_duplicates(['date', 'stock_id'], keep='last')
    return H

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--sub', nargs='*', default=None, help='submission csv files (id + date columns)')
    a = ap.parse_args()
    info = json.load(open(os.path.join(DATA, 'info.json'))); top50 = set(info['Top50@Y'])
    H = read_predictions(a.sub)
    R = intraday_returns(info['Top100@Y'])
    Bt = rolling_beta(R)
    x = H[H.stock_id.isin(top50)].merge(R, on=['date', 'stock_id']).merge(Bt, on=['date', 'stock_id'], how='left')
    if x.empty:
        print('no predicted day has a realised result yet'); return
    x['beta'] = x.beta.clip(0.2, 2.5).fillna(x.groupby('date').beta.transform('median')).fillna(1.0)
    x['s'] = 2 * x.pred.astype(int) - 1
    rows = []
    for d, g in x.groupby('date'):
        sb, rb, bb = g.s.mean(), g.r.mean(), g.beta.mean()
        B = ((g.s - sb) * (g.beta - bb)).mean()
        AR = (g.s * g.r).mean()
        rows.append({'date': d.date(), 'm': rb * 1e4, 'share1': g.pred.mean(), 'B': B, 'AR': AR * 1e4, 'timing': sb * rb * 1e4,
                     'beta_bet': B * rb * 1e4, 'rest': (AR - sb * rb - B * rb) * 1e4,
                     'beta_long': g.beta[g.s > 0].mean(), 'beta_short': g.beta[g.s < 0].mean()})
    T = pd.DataFrame(rows).set_index('date')
    pd.set_option('display.width', 200)
    print(T.round(2).to_string())
    print(f'\naverage over {len(T)} days:'); print(T.mean().round(2).to_string())
    mk = T.timing + T.beta_bet
    print(f'\nmarket-related part (timing + beta_bet): {mk.mean():.2f} bp/day; stock picking not tied to the market (rest): {T.rest.mean():.2f} bp/day')
    print('walk-forward reference (v4 rule, 2022-01..2026-10): timing 1.55, beta_bet 1.23, rest 10.60 bp/day; mean B -0.30')

if __name__ == '__main__':
    main()

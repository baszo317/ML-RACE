r"""
monitor_live.py (v5) -- realised score of the live predictions, split into the parts that can drift

  python monitor_live.py            (after update_data.py, so that yesterday's close is in data\market)

For every predicted day with a known result (Top50, bp):
  AR       competition score of the uploaded predictions
  timing   share of 1s x market move (the market view)
  between  sector bet: (sector share of 1s - overall share) x (sector move - market move)   [fin / semi / elec / trad]
  within   stock picking inside each sector                                                 (xsec = between + within)
  always0  score of predicting 0 everywhere
From the 5th day on, the average of the last n = min(20, days) days is compared with every n-day stretch of the
walk-forward history of the same rule (research\ref_daily_components.csv, 2022-01..2026-10) and printed as a
percentile. A part below the 1st percentile is flagged. A flag is a reason to look (data problem? code? sector
rotation?), not a reason to change the model: with nothing broken, some part is flagged at least once in about 11 % of
19-day periods (research\s14_monitor_reference.py; the first version, 5 % of 20-day averages, fired in 43 %).
"""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); DATA = os.path.join(ROOT, 'data')
sys.path.insert(0, os.path.join(ROOT, 'transformer'))

def group_of(sids):
    from comp_features import group_code
    return {s: group_code(s, sids) for s in sids}

def main():
    f = os.path.join(HERE, 'out', 'history_predictions.csv')
    if not os.path.exists(f):
        sys.exit('no out\\history_predictions.csv yet: run train_and_predict_v5.py first')
    H = pd.read_csv(f, dtype={'stock_id': str}, parse_dates=['date'])
    info = json.load(open(os.path.join(DATA, 'info.json'))); top50 = set(info['Top50@Y']); grp = group_of(info['Top100@Y'])
    R = []
    for sid in sorted(set(H.stock_id) & top50):
        m = pd.read_csv(os.path.join(DATA, 'market', f'{sid}.csv'), parse_dates=['date'])
        m = m[(m['etl:adj_open'] > 0) & (m['etl:adj_close'] > 0)]
        R.append(pd.DataFrame({'date': m.date, 'stock_id': sid, 'r': m['etl:adj_close'] / m['etl:adj_open'] - 1}))
    R = pd.concat(R)
    x = H[H.stock_id.isin(top50)].merge(R, on=['date', 'stock_id'])
    if x.empty:
        print('no predicted day has a realised result yet'); return
    x['g'] = x.stock_id.map(grp); x['s'] = 2 * x.pred - 1
    rows = []
    for d, g in x.groupby('date'):
        sb, rb = g.s.mean(), g.r.mean()
        sg = g.groupby('g').s.transform('mean'); rg = g.groupby('g').r.transform('mean')
        rows.append({'date': d.date(), 'AR': (g.s * g.r).mean() * 1e4, 'timing': sb * rb * 1e4,
                     'between': ((sg - sb) * (rg - rb)).mean() * 1e4, 'within': ((g.s - sg) * (g.r - rg)).mean() * 1e4,
                     'always0': -rb * 1e4, 'share1': g.pred.mean(), 'pbar': g.pbar.mean()})
    T = pd.DataFrame(rows).set_index('date'); T['xsec'] = T.AR - T.timing
    pd.set_option('display.width', 200)
    print(T.round(2).to_string())
    print(f'\naverage over {len(T)} days:'); print(T.mean().round(2).to_string())
    ref = os.path.join(HERE, 'research', 'ref_daily_components.csv')
    if len(T) >= 5 and os.path.exists(ref):
        Hd = pd.read_csv(ref, index_col=0); n = min(20, len(T)); last = T.tail(n).mean()
        print(f'\nlast {n} days vs every {n}-day stretch of the walk-forward history (2022-01..2026-10, same rule):')
        for k in ('AR', 'timing', 'between', 'within', 'xsec'):
            hist = Hd[k].rolling(n).mean().dropna(); p = (hist <= last[k]).mean()
            flag = '  <-- lower than 99 % of the history: check data, code, sector rotation' if p < 0.01 else ''
            print(f'  {k:8s} {last[k]:7.2f}   percentile {100 * p:5.1f} %   (history: 5 % {hist.quantile(0.05):6.2f}, '
                  f'median {hist.median():6.2f}, 95 % {hist.quantile(0.95):6.2f}){flag}')
        print('  (with nothing broken, some part is flagged at least once in about 1 of 9 periods of 19 days)')

if __name__ == '__main__':
    main()

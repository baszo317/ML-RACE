r"""
monitor_live.py -- realised competition score of the predictions made so far (the "observe / detect" step)

  python monitor_live.py            (after update_data.py, so that yesterday's close is in data\market)

Reads out\history_predictions.csv (written by train_and_predict_v3.py; copy of v2\monitor_live.py) and the realised intraday returns in
data\market\*.csv. For every predicted day with a known result it prints, over the Top50:
  AR (bp)   : score of the uploaded predictions (v2: timing shrinkage + ex-dividend rule)
  plain     : score of the same models without the two adjustments
  timing    : part of AR that comes from the overall up/down call (share of 1s x market move)
  xsec      : part of AR that comes from picking which stocks beat the others
  always0   : score of predicting 0 everywhere
and the running averages. If "timing" keeps losing while "xsec" earns, the shrinkage should be larger (--lam);
if both lose for weeks, the models need retraining / the regime has changed.
"""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); DATA = os.path.join(ROOT, 'data')

def main():
    f = os.path.join(HERE, 'out', 'history_predictions.csv')
    if not os.path.exists(f):
        sys.exit('no out\\history_predictions.csv yet: run train_and_predict_v3.py first')
    H = pd.read_csv(f, dtype={'stock_id': str}, parse_dates=['date'])
    top50 = set(json.load(open(os.path.join(DATA, 'info.json')))['Top50@Y'])
    R = []
    for sid in sorted(set(H.stock_id) & top50):
        m = pd.read_csv(os.path.join(DATA, 'market', f'{sid}.csv'), parse_dates=['date'])
        m = m[(m['etl:adj_open'] > 0) & (m['etl:adj_close'] > 0)]
        R.append(pd.DataFrame({'date': m.date, 'stock_id': sid, 'r': m['etl:adj_close'] / m['etl:adj_open'] - 1}))
    R = pd.concat(R)
    x = H[H.stock_id.isin(top50)].merge(R, on=['date', 'stock_id'])
    if x.empty:
        print('no predicted day has a realised result yet'); return
    rows = []
    for d, g in x.groupby('date'):
        s = 2 * g.pred - 1; sp = 2 * g.pred_plain - 1; mk = g.r.mean()
        rows.append({'date': d.date(), 'AR': (s * g.r).mean() * 1e4, 'plain': (sp * g.r).mean() * 1e4,
                     'timing': s.mean() * mk * 1e4, 'xsec': ((s * g.r).mean() - s.mean() * mk) * 1e4,
                     'always0': -mk * 1e4, 'share1': g.pred.mean(), 'pbar': g.pbar.mean()})
    T = pd.DataFrame(rows).set_index('date')
    pd.set_option('display.width', 200)
    print(T.round(2).to_string())
    print('\naverage over', len(T), 'days:'); print(T.mean().round(2).to_string())
    if len(T) >= 5:
        print('\nlast 5 days:'); print(T.tail(5).mean().round(2).to_string())

if __name__ == '__main__':
    main()

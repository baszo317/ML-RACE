r"""
hedge_beta.py (v8) -- optional: cancel part of the stock view's hidden market bet (the low-beta tilt) in a v5 prediction file.

  python hedge_beta.py                       latest ..\v5\out\probabilities_v5_*.csv, lambda 0.5 -> out\submission_v8_hedge_<tag>.csv
  python hedge_beta.py --prob FILE --lam 0.5 any probabilities file written by v4/v5 (columns date, stock_id, score, ex_today)
  python hedge_beta.py --prob ..\v4\out\probabilities_v4_static.csv --tag static     (the static months, for checking)

What it does, for every date in the file (README_v8.md, section 11):
  1. v5 decision s_i = +1 if score_i > 0 else -1 (ex-dividend day -> -1), Top50 = the stocks of sample_submission.csv
  2. beta_i = 120-day rolling beta of the stock's intraday return on the Top100 mean, days before the date only, clipped [0.2, 2.5]
  3. the book's beta exposure E = mean(s_i beta_i) = sbar*mean(beta) + B, where B = mean[(s - sbar)(beta - mean beta)] is the tilt
  4. target E* = sbar_v5*mean(beta) + lambda*B_v5: lambda = 1 keeps v5 as is, 0.5 cancels half of the tilt, 0 all of it
  5. new decision: score_i + c*beta_i > 0 (ex-dividend -> 0), with c in [-0.3, 0.3] (step 0.0025) giving E closest to E*
Walk-forward 2022-01..2026-10 (README_v8.md): lambda 0.5 -> mean 13.52 vs 13.38 bp/day (difference within noise), daily sd
49.6 -> 44.1, P(19-day mean > 0) 0.911 -> 0.937, P(> 20) 0.237 -> 0.222; about 3 of the 50 predictions change per day.
lambda 0.5 was picked after looking at 4 values (exploratory), so it is a risk setting to decide on 11/23 together with w_T,
not a proven improvement.  Only numpy + pandas.
"""
import os, sys, glob, json, argparse
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); DATA = os.path.join(ROOT, 'data')
CGRID = np.round(np.arange(-0.3, 0.3 + 1e-9, 0.0025), 6)

def betas(dates):
    info = json.load(open(os.path.join(DATA, 'info.json')))
    R = []
    for sid in info['Top100@Y']:
        m = pd.read_csv(os.path.join(DATA, 'market', f'{sid}.csv'), parse_dates=['date'])
        m = m[(m['etl:adj_open'] > 0) & (m['etl:adj_close'] > 0)]
        R.append(pd.DataFrame({'date': m.date, 'stock_id': sid, 'r': m['etl:adj_close'] / m['etl:adj_open'] - 1}))
    R = pd.concat(R, ignore_index=True)
    mk = R.groupby('date').r.mean().rename('m100')
    X = R.merge(mk, left_on='date', right_index=True)
    # add the requested dates (the day to predict has no price yet) so that shift(1) gives the beta known before it
    extra = pd.MultiIndex.from_product([sorted(set(pd.to_datetime(dates)) - set(X.date)), info['Top100@Y']], names=['date', 'stock_id'])
    if len(extra):
        X = pd.concat([X, extra.to_frame(index=False)], ignore_index=True)
    X = X.sort_values(['stock_id', 'date'])
    def _b(g):
        x, y = g.m100, g.r
        cv = (x * y).rolling(120, min_periods=60).mean() - x.rolling(120, min_periods=60).mean() * y.rolling(120, min_periods=60).mean()
        return (cv / x.rolling(120, min_periods=60).var(ddof=0)).shift(1)
    X['beta'] = X.groupby('stock_id', group_keys=False)[['m100', 'r']].apply(_b)
    return X[['date', 'stock_id', 'beta']]

def hedge_day(score, beta, ex, lam):
    n = len(score)
    s0 = np.where(ex, -1, np.where(score > 0, 1, -1))
    sbar0, bbar = s0.mean(), beta.mean()
    B0 = ((s0 - sbar0) * (beta - bbar)).mean()
    target = sbar0 * bbar + lam * B0
    S = (score[None, :] + CGRID[:, None] * beta[None, :] > 0) & ~ex[None, :]
    E = ((2 * S - 1) @ beta) / n
    d = np.abs(E - target); cand = np.flatnonzero(d <= d.min() + 1e-12); j = cand[np.argmin(np.abs(CGRID[cand]))]
    s1 = np.where(S[j], 1, -1)
    B1 = ((s1 - s1.mean()) * (beta - bbar)).mean()
    return (s1 > 0).astype(int), dict(c=CGRID[j], B_before=B0, B_after=B1, E_before=(s0 * beta).mean(), E_after=(s1 * beta).mean(),
                                      share1_before=(s0 > 0).mean(), share1_after=(s1 > 0).mean(), changed=int((s1 != s0).sum()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prob', default=None); ap.add_argument('--lam', type=float, default=0.5)
    ap.add_argument('--template', default=os.path.join(ROOT, 'sample_submission.csv')); ap.add_argument('--tag', default=None)
    a = ap.parse_args()
    f = a.prob
    if f is None:
        fs = sorted(glob.glob(os.path.join(ROOT, 'v5', 'out', 'probabilities_v5_*.csv')), key=os.path.getmtime)
        fs = [x for x in fs if not x.endswith('_static.csv')]
        if not fs:
            sys.exit('no ..\\v5\\out\\probabilities_v5_*.csv yet: run v5\\daily.bat first, or pass --prob')
        f = fs[-1]
    tag = a.tag or os.path.basename(f).split('_')[-1].replace('.csv', '')
    Pr = pd.read_csv(f, dtype={'stock_id': str}, parse_dates=['date'])
    tpl = pd.read_csv(a.template, dtype={'id': str}); stocks = tpl['id'].tolist()
    Pr = Pr[Pr.stock_id.isin(stocks)]
    dates = sorted(Pr.date.unique())
    B = betas(dates)
    X = Pr.merge(B, on=['date', 'stock_id'], how='left')
    X['beta'] = X.beta.clip(0.2, 2.5); X['beta'] = X.beta.fillna(X.groupby('date').beta.transform('median')).fillna(1.0)
    sub = pd.DataFrame({'id': stocks}); logs = []
    for d in dates:
        g = X[X.date == d].set_index('stock_id').reindex(stocks)
        if g.score.isna().any():
            sys.exit(f'{pd.Timestamp(d).date()}: some of the 50 stocks have no score in {f}')
        pred, info = hedge_day(g.score.values, g.beta.values, g.ex_today.values == 1, a.lam)
        sub[pd.Timestamp(d).strftime('%Y-%m-%d')] = pred
        logs.append({'date': pd.Timestamp(d).date(), **info})
    out = os.path.join(HERE, 'out'); os.makedirs(out, exist_ok=True)
    fn = os.path.join(out, f'submission_v8_hedge_{tag}.csv')
    vals = sub.drop(columns='id')
    assert list(sub.id) == stocks and set(np.unique(vals.values)) <= {0, 1}
    sub.to_csv(fn, index=False)
    L = pd.DataFrame(logs).set_index('date')
    pd.set_option('display.width', 200)
    print(L.tail(10).round(3).to_string())
    print(f'\n{len(L)} date(s), lambda {a.lam}: beta tilt B {L.B_before.mean():+.3f} -> {L.B_after.mean():+.3f}, beta exposure E {L.E_before.mean():+.3f} -> '
          f'{L.E_after.mean():+.3f}, share of 1 {L.share1_before.mean():.3f} -> {L.share1_after.mean():.3f}, changed {L.changed.mean():.1f} of 50 per day')
    print(f'wrote {fn}  (source: {os.path.basename(f)})')

if __name__ == '__main__':
    main()

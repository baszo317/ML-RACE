r"""
v14_try.py -- a sandbox: test your own signal on the same footing as v14 in about a minute (numpy and pandas only).

1. Write a function in MY_SIGNALS below. It gets the market data as tables (rows = trading days T, columns = the 100 stocks) and
   returns a table of the same shape: the value known at the CLOSE of day T (use only rows <= T: rolling(), shift(+k), cumsum() ...;
   never shift(-k)). The program moves it to the next trading day D = T + 1 by itself, which is when it is used (before 09:00 of D).
   Higher value = the stock should do better than the others from the open to the close of D.
2. python v14_try.py                       -> every signal in MY_SIGNALS
   python v14_try.py --only rev1 --w 0.2  -> one signal, placed with weight 0.2 instead of 1/3
   python v14_try.py --list               -> names of the tables you can use
3. Output: a leak check (the signal of a few days recomputed from data cut at T must be identical), its own rank IC, the
   correlation with the v11 stock view, README_v14 theorem 2 (gain condition, best weight w*, largest helpful weight w0, best IC),
   and v11 / W0 with the signal added (spread-matched new family, PREREG_v14 1), by period with Newey-West t.
   This is exploration: if something looks good, write the rule down first (as in research\PREREG_v14.md) before trusting it.

Tables (all adjusted prices; shares): close, open, high, low, volume, amount, trades, foreign_hold, foreign_net, trust_net,
dealer_self, dealer_hedge, RSI, SMA, MACD, ATR, bb_up, bb_mid, bb_low, y (1 if close > open on day T).
"""
import os, sys, math, argparse, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import v14lib as L
from v14lib import per, w, ic_daily, zday, with_families, diff, fmt, PER, ROOT

COLS = {'etl:adj_close': 'close', 'etl:adj_open': 'open', 'etl:adj_high': 'high', 'etl:adj_low': 'low', 'price:成交股數': 'volume',
        'price:成交金額': 'amount', 'price:成交筆數': 'trades', 'foreign_investors_shareholding:全體外資及陸資持股比率': 'foreign_hold',
        'institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)': 'foreign_net',
        'institutional_investors_trading_summary:投信買賣超股數': 'trust_net',
        'institutional_investors_trading_summary:自營商買賣超股數(自行買賣)': 'dealer_self',
        'institutional_investors_trading_summary:自營商買賣超股數(避險)': 'dealer_hedge',
        'RSI': 'RSI', 'SMA': 'SMA', 'MACD': 'MACD', 'ATR': 'ATR', 'BBANDS_upper': 'bb_up', 'BBANDS_mid': 'bb_mid', 'BBANDS_lower': 'bb_low', 'y': 'y'}


# ---------------------------------------------------------------------------------------------- your signals
def rev1(M):
    """yesterday's intraday reversal: -(close / open - 1) of day T (B already has this as intraday_ret, so expect little)"""
    return -(M['close'] / M['open'] - 1)


def vol_shock(M):
    """volume of day T relative to its 20-day mean (log)"""
    return np.log(M['volume'] / M['volume'].rolling(20, min_periods=10).mean())


def range_pos(M):
    """where the close of day T sits in the 20-day high-low range (0 = at the low, 1 = at the high), sign flipped (reversal)"""
    hi, lo = M['high'].rolling(20, min_periods=10).max(), M['low'].rolling(20, min_periods=10).min()
    return -((M['close'] - lo) / (hi - lo))


MY_SIGNALS = {'rev1': rev1, 'vol_shock': vol_shock, 'range_pos': range_pos}
# ---------------------------------------------------------------------------------------------------------------


def load_tables():
    mdir = os.path.join(ROOT, 'data', 'market')
    ids = sorted(f[:-4] for f in os.listdir(mdir) if f.endswith('.csv'))
    T = {v: {} for v in COLS.values()}
    for s in ids:
        m = pd.read_csv(os.path.join(mdir, f'{s}.csv'), parse_dates=['date'])
        m = m[(m['etl:adj_close'] > 0) & (m['etl:adj_open'] > 0)].set_index('date').sort_index()
        for k, v in COLS.items():
            if k in m:
                T[v][s] = m[k]
    out = {}
    for v, d in T.items():
        df = pd.DataFrame(d).sort_index(); df.index = df.index.astype('datetime64[ns]'); out[v] = df
    return out


def to_D(table, days):
    """value at the close of T -> row of the next trading day D, stacked to (date, stock_id)"""
    nxt = pd.Series(days[1:], index=days[:-1])
    t = table.reindex(days)
    t = t[t.index.isin(nxt.index)].copy()
    t.index = pd.DatetimeIndex(nxt.loc[t.index].values)
    try:
        s = t.stack(future_stack=True)
    except TypeError:
        s = t.stack(dropna=False)
    s.index.names = ['date', 'stock_id']
    return s.replace([np.inf, -np.inf], np.nan)


def leak_check(fn, M, days, n=4, seed=7):
    full = fn(M)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for i in sorted(rng.choice(np.flatnonzero(days >= pd.Timestamp('2022-01-03')), n, replace=False)):
        T = days[i - 1]
        cut = fn({k: v.loc[:T] for k, v in M.items()})
        a, b = cut.loc[T], full.loc[T]
        both = a.notna() & b.notna()
        if (a.notna() != b.notna()).any():
            return np.inf
        if both.any():
            worst = max(worst, float((a[both] - b[both]).abs().max()))
    return worst


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--only', default=''); ap.add_argument('--w', type=float, default=1 / 3)
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    if a.list:
        print('tables: ' + ', '.join(COLS.values())); return
    t0 = time.time()
    M = load_tables(); days = M['close'].index
    d = L.Data(); idx = d.idx
    pe = d.view([d.LGB, d.x1])
    base = {'v11': d.run(pe, True), 'W0': d.run(pe, True, w_t=0.0, lam=0.5)}
    ICx = ic_daily(pe, d.resid)
    print(f'v14_try.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  ({time.time() - t0:.0f}s to load)  weight of the new family w = {a.w:.3f}')
    print('  v11 stock view: rank IC ' + ' / '.join(f'{p} {per(ICx)[p]:.4f}' for p in PER)
          + '   (base Top50 bp/day v11 ' + ' / '.join(f'{per(base["v11"].AR)[p]:.2f}' for p in PER) + ')')
    names = [a.only] if a.only else list(MY_SIGNALS)
    for name in names:
        fn = MY_SIGNALS[name]
        print(f'\n== {name}: {(fn.__doc__ or "").strip()}')
        lk = leak_check(fn, M, days)
        print(f'  leak check (recomputed from data cut at T): max |diff| {lk:.2e} -> ' + ('ok' if lk < 1e-10 else 'LEAK: the value at T depends on rows after T'))
        if not lk < 1e-10:
            continue
        z = to_D(fn(M), days).reindex(idx)
        print(f'  coverage {z.notna().mean():.3f} of the stock-days')
        icz = ic_daily(z, d.resid); rho = ic_daily(z, pe)
        print('  own rank IC      ' + '  '.join(f'{p} {per(icz)[p]:+.4f} (t {nw_t_(icz, p):.1f})' for p in PER))
        for p in ('val', 'hold', 'post', 'all'):
            ax, bz, r_ = per(ICx)[p], per(icz)[p], float(rho[w(rho.index, p)].mean())
            g_ = bz - r_ * ax
            ws = g_ / ((ax - r_ * bz) + g_) if ((ax - r_ * bz) + g_) != 0 else np.nan
            best = math.sqrt(max(ax * ax + bz * bz - 2 * r_ * ax * bz, 0) / (1 - r_ * r_))
            den = 2 * ax * ax * (1 - r_) - (bz - ax) ** 2
            w0 = 0.0 if g_ <= 0 else (1.0 if den <= 0 else min(2 * ax * g_ / den, 1.0))
            print(f'  {p:4s} rho {r_:+.3f}  gain {g_:+.4f}  w* {ws:+.2f}  w0 {w0:.2f}  best IC {best:.4f} (now {ax:.4f})')
        v = with_families(pe, [(zday(z), None)]) if abs(a.w - 1 / 3) < 1e-12 else _with_weight(pe, zday(z), a.w)
        D = dict(base)
        D['v11 + s'] = d.run(v, True); D['W0 + s'] = d.run(v, True, w_t=0.0, lam=0.5)
        print('  v11 + signal - v11: ' + fmt(diff(D, 'v11 + s', 'v11')))
        print('  W0  + signal - W0 : ' + fmt(diff(D, 'W0 + s', 'W0')))
    print(f'\n({time.time() - t0:.0f}s)')


def nw_t_(s, p):
    x = s[w(s.index, p)].values
    return L.nw_t(x)


def _with_weight(pe, z, wt):
    """(1 - wt) * pe + wt * sd(pe) * z, spread-matched (wt = 1/3 is the same as with_families)"""
    from v14lib import sdd, spread_match
    return spread_match((1 - wt) * pe + wt * sdd(pe) * z, pe)


if __name__ == '__main__':
    main()

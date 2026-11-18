r"""
v14_signals.py -- the rule-based signals of v14 (research\PREREG_v14.md section 2 and 4), built from data\market\*.csv only.
needs numpy and pandas only. Writes ..\cache\signals.pkl (rows = prediction day D x stock) and ..\cache\calendar.pkl (rows = D).

Every signal of day D uses closes up to D-1 (the trading day before D), except gap (E1, report only), which uses the open of D.
  al_k{1,3,5,10}   E2a  Avellaneda-Lee s-score, sign flipped (high = expected to rise): 252-day PCA of close-to-close returns of the
                        100 stocks, k eigenportfolios, 60-day OLS with intercept, OU fit of the cumulated residuals
  rr_h{1,5,10,20}  E2b  residual reversal: -(sum of the last h residuals of the k = 5 regression) / sd(60 residuals)
  beta{1..5}             E3b  the k = 5 regression betas (exposures to the 5 eigenportfolios)
  trust60          E4   sum of the last 60 days of investment-trust net buying (shares) / mean daily volume (shares) of the same 60 days
  gap              E1   open_D / close_{D-1} - 1 (adjusted prices): a 09:00 number, an upper bound only
calendar.pkl: q_last{5,10,20,40} (D is among the last n trading days of its calendar quarter), q_first3, quarter_id.
The leak check at the end recomputes the signals of a few days from data cut at D-1 and requires identical values.

usage (in v14\research):  python v14_signals.py  > ..\results\v14_signals.txt      (about 1-2 minutes)
"""
import os, sys, json, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); V14 = os.path.dirname(HERE)
ROOT = os.environ.get('MLRACE_ROOT', os.path.dirname(V14))
MARKET = os.path.join(ROOT, 'data', 'market')
CACHE = os.path.join(V14, 'cache')
W_PCA, W_REG = 252, 60                       # Avellaneda & Lee (2010): 252-day correlation window, 60-day estimation window
KS = (1, 3, 5, 10)                           # number of eigenportfolios (5 = the candidate; others report only)
HS = (1, 5, 10, 20)                          # residual-reversal horizons (5 = the candidate)
B_MAX = float(np.exp(-(252 / 30) / 252))     # kappa > 252/30  <=>  b < exp(-8.4/252) = 0.9672
W_TRUST = 60
TRUST_COL = 'institutional_investors_trading_summary:投信買賣超股數'
VOL_COL = 'price:成交股數'


def load_market():
    """adjusted close / open, trust net buying and volume of the 100 stocks; rows = trading days with real prices"""
    ids = sorted(f[:-4] for f in os.listdir(MARKET) if f.endswith('.csv'))
    C, O, TR, VO = {}, {}, {}, {}
    for s in ids:
        m = pd.read_csv(os.path.join(MARKET, f'{s}.csv'), parse_dates=['date'])
        m = m[(m['etl:adj_close'] > 0) & (m['etl:adj_open'] > 0)].set_index('date').sort_index()
        C[s], O[s] = m['etl:adj_close'], m['etl:adj_open']
        TR[s], VO[s] = m[TRUST_COL].where(m[TRUST_COL] > -1e15), m[VOL_COL].where(m[VOL_COL] > 0)
    C, O, TR, VO = (pd.DataFrame(x).sort_index() for x in (C, O, TR, VO))
    C.index = C.index.astype('datetime64[ns]')
    for x in (O, TR, VO):
        x.index = x.index.astype('datetime64[ns]')
    return C, O, TR, VO


def window_signals(R, t, ks=KS, hs=HS):
    """signals at the close of row t (uses rows t - W_PCA - W_REG + 2 .. t only).
    R: (T, N) array of close-to-close returns (nan = missing). Returns dict name -> (N,) array (nan = no signal)."""
    N = R.shape[1]
    out = {}
    lo = t - W_PCA + 1
    if lo < 1:
        return out
    Wn = R[lo:t + 1]                                             # 252 x N
    ok = np.isfinite(Wn).all(axis=0)
    ok &= np.isfinite(R[t - W_REG + 1:t + 1]).all(axis=0)
    idx = np.flatnonzero(ok)
    if len(idx) < 30:
        return out
    X = Wn[:, idx]
    mu, sd = X.mean(axis=0), X.std(axis=0, ddof=1)
    good = sd > 0
    idx, X, mu, sd = idx[good], X[:, good], mu[good], sd[good]
    Z = (X - mu) / sd
    Cm = Z.T @ Z / (len(Z) - 1)
    ev, V = np.linalg.eigh(Cm)                                   # ascending
    order = np.argsort(ev)[::-1]
    V = V[:, order]
    Rr = R[t - W_REG + 1:t + 1][:, idx]                          # 60 x n
    for k in ks:
        Q = V[:, :k] / sd[:, None]                               # eigenportfolio weights (Avellaneda & Lee eq. 9)
        F = Rr @ Q                                               # 60 x k factor returns
        A = np.column_stack([np.ones(len(F)), F])
        coef, *_ = np.linalg.lstsq(A, Rr, rcond=None)            # (k+1) x n
        E = Rr - A @ coef                                        # 60 x n residuals (sum to 0: intercept)
        Xc = np.cumsum(E, axis=0)
        x, y = Xc[:-1], Xc[1:]
        xm, ym = x.mean(axis=0), y.mean(axis=0)
        sxx = ((x - xm) ** 2).sum(axis=0)
        b = np.where(sxx > 0, ((x - xm) * (y - ym)).sum(axis=0) / np.where(sxx > 0, sxx, 1), np.nan)
        a = ym - b * xm
        zeta = y - a - b * x
        vz = (zeta ** 2).sum(axis=0) / (len(x) - 2)
        valid = (b > 0) & (b < B_MAX) & np.isfinite(b)
        m = np.where(valid, a / (1 - np.where(valid, b, 0.5)), np.nan)
        seq = np.where(valid, np.sqrt(vz / (1 - np.where(valid, b, 0.5) ** 2)), np.nan)
        mbar = np.nanmean(m) if np.isfinite(m).any() else 0.0
        s = (Xc[-1] - (m - mbar)) / seq                          # Xc[-1] = 0 up to rounding
        sig = np.full(N, np.nan); sig[idx] = -s
        out[f'al_k{k}'] = sig
        if k == 5:
            rsd = E.std(axis=0, ddof=k + 1)
            for h in hs:
                v = np.full(N, np.nan); v[idx] = np.where(rsd > 0, -E[-h:].sum(axis=0) / np.where(rsd > 0, rsd, 1), np.nan)
                out[f'rr_h{h}'] = v
            for j in range(k):
                v = np.full(N, np.nan); v[idx] = coef[1 + j]
                out[f'beta{j + 1}'] = v
    return out


def build(C, O, TR, VO, rows=None):
    """all signals, indexed by the CLOSE day T (row of C); rows = list of row numbers to compute (default: all)"""
    R = (C / C.shift(1) - 1).values                              # close-to-close return of each row (nan if a close is missing)
    T, N = R.shape
    rows = list(range(W_PCA + W_REG, T)) if rows is None else list(rows)
    rec = {}
    for t in rows:
        o = window_signals(R, t)
        for k, v in o.items():
            rec.setdefault(k, {})[C.index[t]] = v
    S = {k: pd.DataFrame(v, index=C.columns).T.sort_index() for k, v in rec.items()}
    tr = TR.rolling(W_TRUST, min_periods=40).sum() / VO.rolling(W_TRUST, min_periods=40).mean()
    S['trust60'] = tr.loc[C.index[rows]]
    return S


def to_prediction_day(S, days):
    """value computed at the close of day T -> row of the next trading day D"""
    nxt = pd.Series(days[1:], index=days[:-1])
    out = []
    for k, df in S.items():
        df = df[df.index.isin(nxt.index)].copy()
        df.index = pd.DatetimeIndex(nxt.loc[df.index].values)
        try:
            st = df.stack(future_stack=True)                         # pandas >= 2.1 (keeps missing values)
        except TypeError:
            st = df.stack(dropna=False)                              # older pandas
        out.append(st.rename(k))
    X = pd.concat(out, axis=1)
    X.index.names = ['date', 'stock_id']
    return X


def calendar_flags(days):
    d = pd.Series(days, index=days)
    q = d.dt.year * 10 + d.dt.quarter
    pos = d.groupby(q.values).cumcount() + 1
    n = d.groupby(q.values).transform('size')
    back = n - pos + 1                                           # 1 = last trading day of the quarter
    F = pd.DataFrame({'quarter_id': q.values, 'pos_in_quarter': pos.values, 'from_end': back.values}, index=days)
    for k in (5, 10, 20, 40):
        F[f'q_last{k}'] = F.from_end <= k
    F['q_first3'] = F.pos_in_quarter <= 3
    return F


def main():
    t0 = time.time()
    print(f'v14_signals.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  data: {MARKET}')
    C, O, TR, VO = load_market()
    days = C.index
    print(f'  {C.shape[1]} stocks, {len(days)} trading days {days[0]:%Y-%m-%d} .. {days[-1]:%Y-%m-%d}')
    S = build(C, O, TR, VO)
    X = to_prediction_day(S, days)
    gap = (O / C.shift(1) - 1).stack().rename('gap')
    gap.index.names = ['date', 'stock_id']
    X = X.join(gap, how='outer')
    X = X.sort_index()
    cal = calendar_flags(days)
    # quarters whose end is beyond the data are incomplete: drop their q_last flags (they matter only for live use)
    last_q = cal.quarter_id.iloc[-1]
    cal.loc[cal.quarter_id == last_q, [c for c in cal.columns if c.startswith('q_last')]] = False
    os.makedirs(CACHE, exist_ok=True)
    X.to_pickle(os.path.join(CACHE, 'signals.pkl')); cal.to_pickle(os.path.join(CACHE, 'calendar.pkl'))
    print(f'  built in {time.time() - t0:.0f}s -> cache\\signals.pkl ({len(X):,} rows, {X.shape[1]} columns), cache\\calendar.pkl')
    w = X[X.index.get_level_values('date') >= '2022-01-01']
    print('\n  coverage from 2022-01 (share of non-missing rows) and mean daily cross-sectional sd:')
    for c in X.columns:
        print(f'    {c:8s} {w[c].notna().mean():.3f}   sd {w[c].groupby(level="date").std().mean():.4f}')
    print(f'\n  calendar: q_last20 {int(cal.q_last20.sum())} days, q_first3 {int(cal.q_first3.sum())} days (2020-01 .. {days[-1]:%Y-%m-%d};'
          f' the last, unfinished quarter has no q_last flags)')

    # ---------------------------------------------------------------- leak check
    print('\n  leak check: recompute from data cut at the close of D-1, compare with the full run (max |difference|)')
    rng = np.random.default_rng(14)
    pick = sorted(rng.choice(np.flatnonzero(days >= pd.Timestamp('2022-01-03')), 6, replace=False))
    worst = 0.0
    for i in pick:
        D, T = days[i], days[i - 1]
        Ct, Ot, TRt, VOt = (x.loc[:T] for x in (C, O, TR, VO))
        St = build(Ct, Ot, TRt, VOt, rows=[len(Ct) - 1])
        diffs = []
        for k, df in St.items():
            a = df.loc[T].astype(float)
            b = X.xs(D, level='date')[k].reindex(a.index).astype(float)
            both = a.notna() & b.notna()
            if (a.notna() != b.notna()).any():
                diffs.append(np.inf)
            elif both.any():
                diffs.append(float((a[both] - b[both]).abs().max()))
        mx = max(diffs) if diffs else 0.0
        worst = max(worst, mx)
        print(f'    D = {D:%Y-%m-%d} (cut at {T:%Y-%m-%d}): {len(diffs)} signals, max |diff| {mx:.2e}')
    print('  leak check ' + ('PASSED (identical up to rounding)' if worst < 1e-10 else 'FAILED: find the cause before using the signals'))
    json.dump({'built': f'{pd.Timestamp.now():%Y-%m-%d %H:%M}', 'rows': int(len(X)), 'columns': list(X.columns), 'leak_check_max_abs_diff': worst},
              open(os.path.join(CACHE, 'signals_meta.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()

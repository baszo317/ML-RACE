r"""
alpha158.py -- Qlib's Alpha158 factor set computed from the competition files (v10).

Definitions: microsoft/qlib, qlib/contrib/data/loader.py Alpha158DL.get_feature_config with the default config of
qlib/contrib/data/handler.py Alpha158 (kbar; price OPEN HIGH LOW VWAP at window 0; rolling windows 5 10 20 30 60 with all
29 operators) = 9 + 4 + 29 x 5 = 158 features. The source text is kept in research\qlib_alpha158_source.txt (commit
5435523, 2026-10-08); the operator semantics in research\qlib_ops_source.txt (qlib/data/ops.py, qlib/data/_libs/rolling.pyx).
research\check_alpha158.py compares this file with a line-by-line port of those operators.

Operator semantics reproduced (qlib/data/ops.py):
  * every rolling operator is pandas .rolling(N, min_periods=1): the first N-1 rows of a stock use the shorter window
  * Std = sample std (ddof 1); Quantile = linear interpolation; Rank = pct rank of the newest value, ties averaged
  * IdxMax / IdxMin = 1-based position of the max / min inside the window (oldest = 1, newest = N)
  * Slope / Rsquare / Resi = OLS of the window values on their positions 1..N (missing values skipped) as in rolling.pyx;
    Resi = newest value minus its fitted value; Rsquare and Corr are NaN where a rolling std is ~0 (np.isclose atol 2e-5)
  * x > y and x < y are False when a side is missing (numpy comparison); Greater / Less = np.maximum / np.minimum
Inputs per stock (competition files only):
  $open $high $low $close = etl:adj_open/high/low/close. The course's adjusted prices are 後復權: raw x F_t with F_t
      constant between ex-dates and growing at each dividend (2330: about 1.92 in 2020, 2.21 in 2026), so past values do
      not change when a new dividend happens.
  $volume = price:成交股數 (raw shares; qlib divides volume by its adjustment factor, which here would only move these ratio
      features by the size of a dividend at an ex-date)
  $vwap = price:成交金額 / price:成交股數 (raw VWAP) x F_t, F_t estimated causally by vwap_factor(): each day bounds it,
      adj_low / raw_vwap <= F_t <= adj_high / raw_vwap; the bounds are intersected over the days since the last restart;
      dividend factors (data\ref\corporate_actions.parquet; later ex-dates from exdiv_history.csv, f = 1 - value /
      prev_close) carry the estimate across ex-dates; an ex-date without a factor, or an empty intersection, restarts it.
      F_t uses days <= t only.
Timing (no look-ahead; the rule of transformer\comp_features.py): a value of day t uses data of day t, known after the
close, so every feature is shifted by one row of the same stock: row D holds the values computed from data up to D-1.
Columns: 'a158_' + the qlib name (a158_KMID ... a158_VSUMD60).
"""
import os, sys, json
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as _swv

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
WINDOWS = (5, 10, 20, 30, 60)
KBAR = ['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2']
PRICE = ['OPEN0', 'HIGH0', 'LOW0', 'VWAP0']
OPS = ['ROC', 'MA', 'STD', 'BETA', 'RSQR', 'RESI', 'MAX', 'MIN', 'QTLU', 'QTLD', 'RANK', 'RSV', 'IMAX', 'IMIN', 'IMXD',
       'CORR', 'CORD', 'CNTP', 'CNTN', 'CNTD', 'SUMP', 'SUMN', 'SUMD', 'VMA', 'VSTD', 'WVMA', 'VSUMP', 'VSUMN', 'VSUMD']
NAMES = KBAR + PRICE + [f'{op}{d}' for op in OPS for d in WINDOWS]          # qlib's order
PREFIX = 'a158_'
COLS = [PREFIX + n for n in NAMES]
assert len(NAMES) == 158 and len(set(NAMES)) == 158

# screening groups (v10 PREREG): one qlib operator family each, except families that are exact transforms of each other,
# which carry the same information and form one group: CNTD = CNTP - CNTN; SUMN = 1 - SUMP and SUMD = 2 SUMP - 1 (up to the
# 1e-12 in the denominator), the same for VSUM*; IMXD = IMAX - IMIN
_FAMILY = {'IDX': ['IMAX', 'IMIN', 'IMXD'], 'CNT': ['CNTP', 'CNTN', 'CNTD'], 'SUM': ['SUMP', 'SUMN', 'SUMD'],
           'VSUM': ['VSUMP', 'VSUMN', 'VSUMD']}

def groups():
    """group name -> list of column names (23 groups, together the 158 columns)"""
    g = {'KBAR': list(KBAR), 'PRICE': list(PRICE)}
    seen = set()
    for op in OPS:
        fam = next((k for k, v in _FAMILY.items() if op in v), None)
        key = fam or op
        if key in seen:
            continue
        seen.add(key)
        ops = _FAMILY[fam] if fam else [op]
        g[key] = [f'{o}{d}' for o in ops for d in WINDOWS]
    out = {k: [PREFIX + n for n in v] for k, v in g.items()}
    assert sorted(sum(out.values(), [])) == sorted(COLS)
    return out

# ------------------------------------------------------------------ operators on one stock (numpy arrays, oldest first)
def _win(x, n, pad):
    """windows of length n ending at every row (oldest first); positions before the first row hold `pad`"""
    return _swv(np.concatenate([np.full(n - 1, pad, dtype=float), x]), n)

def _regression(y, n):
    """rolling.pyx Slope / Rsquare / Resi: OLS of the valid window values on positions 1..n (newest = n)"""
    W = _win(y, n, np.nan)
    m = ~np.isnan(W)
    x = np.arange(1, n + 1, dtype=float)[None, :]
    # centre y by the window mean of the valid values: slope, r2 and the residual do not depend on a shift of y, and the
    # centring avoids the cancellation of n*Syy - Sy^2 for price levels in the thousands
    cnt = m.sum(1).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        mu = np.where(cnt > 0, np.where(m, W, 0.0).sum(1) / np.maximum(cnt, 1), np.nan)
        Y = np.where(m, W - mu[:, None], 0.0); X = np.where(m, x, 0.0)
        Sx, Sxx = X.sum(1), (X * X).sum(1)
        Sy, Syy, Sxy = Y.sum(1), (Y * Y).sum(1), (X * Y).sum(1)
        num = cnt * Sxy - Sx * Sy
        den = cnt * Sxx - Sx * Sx
        slope = num / den
        r2 = (num / np.sqrt(den * (cnt * Syy - Sy * Sy))) ** 2
        interp = Sy / cnt - slope * Sx / cnt
        resi = (y - mu) - (slope * n + interp)                  # newest value (centred) minus its fitted value
    slope[cnt < 1] = np.nan; r2[cnt < 1] = np.nan
    return slope, r2, resi

def _idx(y, n, kind):
    """IdxMax / IdxMin: 1-based position inside the window (pandas rolling(n, min_periods=1).apply(argmax + 1));
    the first n-1 rows use the rows available so far; a missing value inside a window wins, as numpy argmax / argmin do"""
    pad = -np.inf if kind == 'max' else np.inf
    W = _win(y, n, pad)
    p = (np.argmax(W, axis=1) if kind == 'max' else np.argmin(W, axis=1)).astype(float)
    t = np.arange(len(y))
    p = np.where(t < n - 1, p - (n - 1 - t), p) + 1.0
    return p

def _rank(y, n):
    """Rank: pandas rolling(n, min_periods=1).rank(pct=True) -- average rank of the newest value among the valid window
    values divided by their count; NaN if the newest value is missing"""
    W = _win(y, n, np.nan)
    last = y[:, None]
    m = ~np.isnan(W)
    cnt = m.sum(1)
    less = (m & (W < last)).sum(1); eq = (m & (W == last)).sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (less + (eq + 1) / 2.0) / cnt
    r[np.isnan(y)] = np.nan
    return r

def _near0(s):
    return np.isclose(s.values, 0, atol=2e-05)

def stock_alpha158(o, h, l, c, v, vwap):
    """the 158 values of every day (not shifted) for one stock; inputs are 1-d arrays in date order"""
    o, h, l, c, v, vwap = [np.asarray(a, dtype=float) for a in (o, h, l, c, v, vwap)]
    F = {}
    S = lambda a: pd.Series(a)
    with np.errstate(divide='ignore', invalid='ignore'):
        hl = h - l + 1e-12; mx = np.maximum(o, c); mn = np.minimum(o, c)
        F['KMID'] = (c - o) / o; F['KLEN'] = (h - l) / o; F['KMID2'] = (c - o) / hl
        F['KUP'] = (h - mx) / o; F['KUP2'] = (h - mx) / hl; F['KLOW'] = (mn - l) / o; F['KLOW2'] = (mn - l) / hl
        F['KSFT'] = (2 * c - h - l) / o; F['KSFT2'] = (2 * c - h - l) / hl
        F['OPEN0'] = o / c; F['HIGH0'] = h / c; F['LOW0'] = l / c; F['VWAP0'] = vwap / c
        cs, hs, ls, vs = S(c), S(h), S(l), S(v)
        c1, v1 = cs.shift(1), vs.shift(1)
        up, dn, ad = np.maximum(cs - c1, 0), np.maximum(c1 - cs, 0), (cs - c1).abs()
        vup, vdn, vad = np.maximum(vs - v1, 0), np.maximum(v1 - vs, 0), (vs - v1).abs()
        gt = (cs > c1).astype(float); lt = (cs < c1).astype(float)
        lv = np.log(vs + 1); cr = cs / c1; lvr = np.log(vs / v1 + 1)
        wv = (cs / c1 - 1).abs() * vs
        for d in WINDOWS:
            R = lambda s: s.rolling(d, min_periods=1)
            F[f'ROC{d}'] = (cs.shift(d) / cs).values
            F[f'MA{d}'] = (R(cs).mean() / cs).values
            F[f'STD{d}'] = (R(cs).std() / cs).values
            sl, r2, rs = _regression(c, d)
            r2[_near0(R(cs).std())] = np.nan
            F[f'BETA{d}'] = sl / c; F[f'RSQR{d}'] = r2; F[f'RESI{d}'] = rs / c
            hmax, lmin = R(hs).max(), R(ls).min()
            F[f'MAX{d}'] = (hmax / cs).values; F[f'MIN{d}'] = (lmin / cs).values
            F[f'QTLU{d}'] = (R(cs).quantile(0.8) / cs).values; F[f'QTLD{d}'] = (R(cs).quantile(0.2) / cs).values
            F[f'RANK{d}'] = _rank(c, d)
            F[f'RSV{d}'] = ((cs - lmin) / (hmax - lmin + 1e-12)).values
            im, iv = _idx(h, d, 'max'), _idx(l, d, 'min')
            F[f'IMAX{d}'] = im / d; F[f'IMIN{d}'] = iv / d; F[f'IMXD{d}'] = (im - iv) / d
            for nm, a, b in (('CORR', cs, lv), ('CORD', cr, lvr)):
                r = R(a).corr(b)
                r[_near0(R(a).std()) | _near0(R(b).std())] = np.nan
                F[f'{nm}{d}'] = r.values
            p, n = R(gt).mean(), R(lt).mean()
            F[f'CNTP{d}'] = p.values; F[f'CNTN{d}'] = n.values; F[f'CNTD{d}'] = (p - n).values
            su, sd, sa = R(up).sum(), R(dn).sum(), R(ad).sum() + 1e-12
            F[f'SUMP{d}'] = (su / sa).values; F[f'SUMN{d}'] = (sd / sa).values; F[f'SUMD{d}'] = ((su - sd) / sa).values
            F[f'VMA{d}'] = (R(vs).mean() / (vs + 1e-12)).values; F[f'VSTD{d}'] = (R(vs).std() / (vs + 1e-12)).values
            F[f'WVMA{d}'] = (R(wv).std() / (R(wv).mean() + 1e-12)).values
            su, sd, sa = R(vup).sum(), R(vdn).sum(), R(vad).sum() + 1e-12
            F[f'VSUMP{d}'] = (su / sa).values; F[f'VSUMN{d}'] = (sd / sa).values; F[f'VSUMD{d}'] = ((su - sd) / sa).values
    return pd.DataFrame({n: np.asarray(F[n], dtype=float) for n in NAMES})

# ------------------------------------------------------------------ VWAP adjustment factor
def event_table(data_dir):
    """ex-dates with the dividend factor f (adjusted price before the ex-date = raw x f in 前復權 terms; in the course's
    後復權 prices F grows by 1/f): corporate_actions.parquet (history; every row has f), exdiv_history.csv (TWT49U results,
    f = 1 - value / prev_close), exdiv_upcoming.csv and exdiv_manual.csv (dates only: f unknown -> restart)"""
    rows = []
    ref = os.path.join(data_dir, 'ref')
    f = os.path.join(ref, 'corporate_actions.parquet')
    if os.path.exists(f):
        try:
            ca = pd.read_parquet(f)
            rows.append(pd.DataFrame({'stock_id': ca.stock_id.astype(str), 'date': pd.to_datetime(ca.date), 'f': ca.f.astype(float), 'pri': 0}))
        except Exception as e:                                   # e.g. pyarrow missing
            print(f'warning: cannot read {f} ({e}); VWAP factor restarts at every ex-date')
    f = os.path.join(ref, 'exdiv_history.csv')
    if os.path.exists(f):
        x = pd.read_csv(f, dtype={'stock_id': str})
        if len(x):
            fv = 1 - pd.to_numeric(x.value, errors='coerce') / pd.to_numeric(x.prev_close, errors='coerce')
            rows.append(pd.DataFrame({'stock_id': x.stock_id.str.strip(), 'date': pd.to_datetime(x.date), 'f': fv, 'pri': 1}))
    for g in ('exdiv_upcoming.csv', 'exdiv_manual.csv'):
        g = os.path.join(ref, g)
        if os.path.exists(g):
            x = pd.read_csv(g, dtype={'stock_id': str})
            if len(x):
                rows.append(pd.DataFrame({'stock_id': x.stock_id.str.strip(), 'date': pd.to_datetime(x.date), 'f': np.nan, 'pri': 2}))
    if not rows:
        return {}
    E = pd.concat(rows, ignore_index=True)
    E['date'] = E.date.astype('datetime64[ns]')
    E = E[E.f.isna() | ((E.f > 0.2) & (E.f < 5))]
    E = E.sort_values(['stock_id', 'date', 'pri']).drop_duplicates(['stock_id', 'date'])  # the factor of the best source
    return {s: dict(zip(g.date.values, g.f.values)) for s, g in E.groupby('stock_id')}

def vwap_factor(dates, adj_low, adj_high, vwap_raw, events=None, full=False):
    """causal estimate of F_t (adjusted / raw price) from the daily bounds adj_low / raw_vwap <= F_t <= adj_high / raw_vwap.
    Works with h_t = log F_t minus the known jumps (constant between unknown events): the bounds are intersected over the
    days since the last restart and the estimate is the middle of the intersection, so while no unknown event happens the
    error of log F is at most half the width of the intersection (returned with full=True).
    A day whose bounds contradict the intersection is set aside (an odd VWAP, e.g. trades outside the session's range);
    if the next day agrees with it and not with the intersection, F really moved (an event missing from the tables) and the
    intersection restarts from those two days. An ex-date without a factor restarts it at once.
    Returns (F, number of restarts) or, with full=True, (F, restarts, half width, number of days set aside)."""
    events = events or {}
    dates = pd.DatetimeIndex(dates).values.astype('datetime64[ns]')
    with np.errstate(divide='ignore', invalid='ignore'):
        lo = np.log(np.asarray(adj_low, float) / np.asarray(vwap_raw, float))
        hi = np.log(np.asarray(adj_high, float) / np.asarray(vwap_raw, float))
    n = len(dates); out = np.full(n, np.nan); half = np.full(n, np.nan)
    S = 0.0; L, U = -np.inf, np.inf; cand = None; restarts = 0; aside = 0
    for i, d in enumerate(dates):
        if d in events:
            f = events[d]
            if np.isfinite(f) and f > 0:
                S -= np.log(f)                                   # F grows by 1 / f from the ex-date on
            else:
                L, U, cand = -np.inf, np.inf, None               # unknown factor: start again from this day
        a, b = lo[i] - S, hi[i] - S
        if np.isfinite(a) and np.isfinite(b) and a <= b:
            if max(L, a) <= min(U, b):                           # agrees with the intersection
                L, U, cand = max(L, a), min(U, b), None
            elif cand is not None and max(cand[0], a) <= min(cand[1], b):
                L, U, cand = max(cand[0], a), min(cand[1], b), None   # two days agree on a new level: restart
                restarts += 1
            else:
                cand = (a, b); aside += 1                        # set aside; the estimate keeps the intersection
        if np.isfinite(L) and np.isfinite(U):
            out[i] = np.exp((L + U) / 2 + S); half[i] = (U - L) / 2
    return (out, restarts, half, aside) if full else (out, restarts)

# ------------------------------------------------------------------ panel
def _read_market(path):
    sys.path.insert(0, os.path.join(ROOT, 'transformer'))
    from comp_features import read_market                      # same rows as the B features (placeholder row kept, NaN)
    return read_market(path)

def stock_frame(df, events=None):
    """one stock file (comp_features.read_market) -> DataFrame of the 158 day-t values (not shifted) + VWAP diagnostics"""
    o, h, l, c = [df[f'etl:adj_{k}'].values for k in ('open', 'high', 'low', 'close')]
    v = df['price:成交股數'].values.astype(float); amt = df['price:成交金額'].values.astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        vw_raw = np.where(v > 0, amt / v, np.nan)
    Fct, nre = vwap_factor(df.date.values, l, h, vw_raw, events)
    # the VWAP of a day lies in its [low, high]; on a day whose bounds were set aside (odd VWAP or a factor jump not yet
    # confirmed) the estimate is clipped to that range
    vwap = np.clip(vw_raw * Fct, l, h)
    A = stock_alpha158(o, h, l, c, v, vwap)
    A.attrs['restarts'] = nre
    return A

def build_alpha158(data_dir, pred_dates=(), sids=None, log=None):
    """long table date, stock_id, a158_* for every row of the stock files (+ the next requested date beyond the data, as
    comp_features.build_comp_features); row D = values computed from data up to D-1"""
    info = json.load(open(os.path.join(data_dir, 'info.json'))); sids = sids or info['Top100@Y']
    raw = {s: _read_market(os.path.join(data_dir, 'market', f'{s}.csv')) for s in sids}
    last = max(d.date.max() for d in raw.values())
    nxt = sorted(pd.Timestamp(d) for d in pred_dates if pd.Timestamp(d) > last)[:1]
    ev = event_table(data_dir)
    out, nres = [], {}
    for s, df in raw.items():
        if nxt and df.date.max() < nxt[0]:
            df = pd.concat([df, pd.DataFrame({'date': nxt})], ignore_index=True)
        A = stock_frame(df, ev.get(s))
        nres[s] = A.attrs.get('restarts', 0)
        A = A.shift(1)                                           # known after the close of day D-1 -> row D
        A.columns = [PREFIX + c for c in A.columns]
        A.insert(0, 'stock_id', s); A.insert(0, 'date', pd.to_datetime(df.date).values.astype('datetime64[ns]'))
        out.append(A)
    X = pd.concat(out, ignore_index=True).sort_values(['date', 'stock_id']).reset_index(drop=True)
    X.attrs['vwap_restarts'] = nres
    if log:
        log(f'alpha158: {len(X)} rows, {X.stock_id.nunique()} stocks, VWAP factor restarts {sum(nres.values())}')
    return X

def add_alpha158(P, data_dir, pred_dates=()):
    """merge the 158 columns into a panel with date, stock_id (left join; rows without stock data get NaN)"""
    X = build_alpha158(data_dir, pred_dates, sids=sorted(P.stock_id.astype(str).unique()))
    P = P.drop(columns=[c for c in COLS if c in P.columns])
    P = P.merge(X, on=['date', 'stock_id'], how='left')
    return P

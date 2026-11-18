r"""
check_alpha158.py -- checks v10\alpha158.py against Qlib's definitions.

1. The 158 expressions: the templates below are copied from qlib/contrib/data/loader.py (Alpha158DL.get_feature_config);
   every template string must occur verbatim in research\qlib_alpha158_source.txt (the fetched source text), and the
   expanded names must equal alpha158.NAMES in the same order.
2. Values: an independent reference evaluator -- the expressions are evaluated with a line-by-line port of the operators
   of qlib/data/ops.py (pandas rolling with min_periods=1, the Corr / Rsquare masks) and of the Cython deque algorithms
   of qlib/data/_libs/rolling.pyx (Slope, Resi, Rsquare); compared with alpha158.stock_alpha158 on several stocks.
3. Timing: build_alpha158 row D equals the day D-1 values of the same stock; consistency with the existing B features
   (KMID = exp(intraday_ret) - 1 and ROC5 = exp(-ret5) on the same rows of the v9 panel).
4. VWAP factor: the causal estimate against the factor implied by corporate_actions.parquet (F_t x prod f = constant).
run: python check_alpha158.py [stock ...]  > ..\results\check_alpha158.txt
"""
import os, sys, re, math, json, time
from collections import deque
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); V10 = os.path.dirname(HERE); ROOT = os.path.dirname(V10)
sys.path.insert(0, V10)
import alpha158 as A
DATA = os.path.join(ROOT, 'data')

# ------------------------------------------------------------------ 1. expressions (copied from loader.py)
KBAR_FIELDS = ["($close-$open)/$open", "($high-$low)/$open", "($close-$open)/($high-$low+1e-12)",
               "($high-Greater($open, $close))/$open", "($high-Greater($open, $close))/($high-$low+1e-12)",
               "(Less($open, $close)-$low)/$open", "(Less($open, $close)-$low)/($high-$low+1e-12)",
               "(2*$close-$high-$low)/$open", "(2*$close-$high-$low)/($high-$low+1e-12)"]
PRICE_TEMPLATE = "$%s/$close"
ROLL = [  # (name template, field template, number of %d per field)
    ("ROC%d", "Ref($close, %d)/$close", 1), ("MA%d", "Mean($close, %d)/$close", 1), ("STD%d", "Std($close, %d)/$close", 1),
    ("BETA%d", "Slope($close, %d)/$close", 1), ("RSQR%d", "Rsquare($close, %d)", 1), ("RESI%d", "Resi($close, %d)/$close", 1),
    ("MAX%d", "Max($high, %d)/$close", 1), ("MIN%d", "Min($low, %d)/$close", 1),
    ("QTLU%d", "Quantile($close, %d, 0.8)/$close", 1), ("QTLD%d", "Quantile($close, %d, 0.2)/$close", 1),
    ("RANK%d", "Rank($close, %d)", 1), ("RSV%d", "($close-Min($low, %d))/(Max($high, %d)-Min($low, %d)+1e-12)", 3),
    ("IMAX%d", "IdxMax($high, %d)/%d", 2), ("IMIN%d", "IdxMin($low, %d)/%d", 2), ("IMXD%d", "(IdxMax($high, %d)-IdxMin($low, %d))/%d", 3),
    ("CORR%d", "Corr($close, Log($volume+1), %d)", 1), ("CORD%d", "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), %d)", 1),
    ("CNTP%d", "Mean($close>Ref($close, 1), %d)", 1), ("CNTN%d", "Mean($close<Ref($close, 1), %d)", 1),
    ("CNTD%d", "Mean($close>Ref($close, 1), %d)-Mean($close<Ref($close, 1), %d)", 2),
    ("SUMP%d", "Sum(Greater($close-Ref($close, 1), 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)", 2),
    ("SUMN%d", "Sum(Greater(Ref($close, 1)-$close, 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)", 2),
    ("SUMD%d", "(Sum(Greater($close-Ref($close, 1), 0), %d)-Sum(Greater(Ref($close, 1)-$close, 0), %d))"
               "/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)", 3),
    ("VMA%d", "Mean($volume, %d)/($volume+1e-12)", 1), ("VSTD%d", "Std($volume, %d)/($volume+1e-12)", 1),
    ("WVMA%d", "Std(Abs($close/Ref($close, 1)-1)*$volume, %d)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, %d)+1e-12)", 2),
    ("VSUMP%d", "Sum(Greater($volume-Ref($volume, 1), 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)", 2),
    ("VSUMN%d", "Sum(Greater(Ref($volume, 1)-$volume, 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)", 2),
    ("VSUMD%d", "(Sum(Greater($volume-Ref($volume, 1), 0), %d)-Sum(Greater(Ref($volume, 1)-$volume, 0), %d))"
                "/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)", 3)]
WINDOWS = [5, 10, 20, 30, 60]

def expressions():
    fields = list(KBAR_FIELDS); names = ['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2']
    for f in ['OPEN', 'HIGH', 'LOW', 'VWAP']:
        fields.append(PRICE_TEMPLATE % f.lower()); names.append(f + '0')
    for nt, ft, k in ROLL:
        fields += [ft % ((d,) * k) for d in WINDOWS]; names += [nt % d for d in WINDOWS]
    return fields, names

def check_source():
    src = open(os.path.join(HERE, 'qlib_alpha158_source.txt'), encoding='utf-8').read()
    flat = re.sub(r'"\s*\n\s*"', '', src)                       # join the implicitly concatenated string literals
    miss = [f for f in KBAR_FIELDS if f'"{f}"' not in flat]
    miss += [ft for _, ft, _ in ROLL if f'"{ft}"' not in flat]
    miss += [nt for nt, _, _ in ROLL if f'"{nt}"' not in flat]
    if '"$%s/$close" % field' not in flat:
        miss.append(PRICE_TEMPLATE)
    for w in ('"windows": [0]', '"feature": ["OPEN", "HIGH", "LOW", "VWAP"]', '"windows", [5, 10, 20, 30, 60]'):
        if w not in flat:
            miss.append(w)
    return miss

# ------------------------------------------------------------------ 2. reference operators (port of qlib/data/ops.py)
_nan = float('nan')
def _div(a, b):                                                  # C double division (cdivision): x/0 = +-inf, 0/0 = nan
    return float(np.float64(a) / np.float64(b)) if b != 0 else (_nan if (a == 0 or a != a) else math.copysign(math.inf, a) * math.copysign(1, b))

class _CyRolling:                                                # rolling.pyx class Rolling + Slope / Resi / Rsquare
    def __init__(self, window, kind):
        self.window, self.kind = window, kind; self.na_count = window; self.barv = deque([_nan] * window)
        self.i_sum = self.x_sum = self.x2_sum = self.y_sum = self.y2_sum = self.xy_sum = 0.0
    def update(self, val):
        w = self.window
        self.barv.append(val)
        self.xy_sum = self.xy_sum - self.y_sum
        self.x2_sum = self.x2_sum + self.i_sum - 2 * self.x_sum
        self.x_sum = self.x_sum - self.i_sum
        _val = self.barv[0]
        if not math.isnan(_val):
            self.i_sum -= 1; self.y_sum -= _val; self.y2_sum -= _val * _val
        else:
            self.na_count -= 1
        self.barv.popleft()
        if math.isnan(val):
            self.na_count += 1
        else:
            self.i_sum += 1; self.x_sum += w; self.x2_sum += w * w; self.y_sum += val; self.y2_sum += val * val; self.xy_sum += w * val
        N = w - self.na_count
        num = N * self.xy_sum - self.x_sum * self.y_sum
        if self.kind == 'slope':
            return _div(num, N * self.x2_sum - self.x_sum * self.x_sum)
        if self.kind == 'resi':
            slope = _div(num, N * self.x2_sum - self.x_sum * self.x_sum)
            x_mean = _div(self.x_sum, N); y_mean = _div(self.y_sum, N)
            interp = y_mean - slope * x_mean
            return val - (slope * w + interp)
        den = (N * self.x2_sum - self.x_sum * self.x_sum) * (N * self.y2_sum - self.y_sum * self.y_sum)
        r = _div(num, float(np.sqrt(np.float64(den))))
        return r * r

def _cy(series, N, kind):
    r = _CyRolling(N, kind)
    return pd.Series([r.update(float(v)) for v in series.values], index=series.index)

def Ref(s, N): return s.shift(N)
def Mean(s, N): return s.rolling(N, min_periods=1).mean()
def Sum(s, N): return s.rolling(N, min_periods=1).sum()
def Std(s, N): return s.rolling(N, min_periods=1).std()
def Max(s, N): return s.rolling(N, min_periods=1).max()
def Min(s, N): return s.rolling(N, min_periods=1).min()
def Quantile(s, N, q): return s.rolling(N, min_periods=1).quantile(q)
def Rank(s, N): return s.rolling(N, min_periods=1).rank(pct=True)
def IdxMax(s, N): return s.rolling(N, min_periods=1).apply(lambda x: x.argmax() + 1, raw=True)
def IdxMin(s, N): return s.rolling(N, min_periods=1).apply(lambda x: x.argmin() + 1, raw=True)
def Slope(s, N): return _cy(s, N, 'slope')
def Resi(s, N): return _cy(s, N, 'resi')
def Rsquare(s, N):
    r = _cy(s, N, 'rsquare')
    r.loc[np.isclose(s.rolling(N, min_periods=1).std(), 0, atol=2e-05)] = np.nan
    return r
def Corr(a, b, N):
    r = a.rolling(N, min_periods=1).corr(b)
    r.loc[np.isclose(a.rolling(N, min_periods=1).std(), 0, atol=2e-05) | np.isclose(b.rolling(N, min_periods=1).std(), 0, atol=2e-05)] = np.nan
    return r
def Greater(a, b): return np.maximum(a, b)
def Less(a, b): return np.minimum(a, b)
def Abs(s): return np.abs(s)
def Log(s): return np.log(s)

def reference(o, h, l, c, v, vwap):
    fields, names = expressions()
    ns = dict(Ref=Ref, Mean=Mean, Sum=Sum, Std=Std, Max=Max, Min=Min, Quantile=Quantile, Rank=Rank, IdxMax=IdxMax, IdxMin=IdxMin,
              Slope=Slope, Resi=Resi, Rsquare=Rsquare, Corr=Corr, Greater=Greater, Less=Less, Abs=Abs, Log=Log)
    for k, a in dict(open=o, high=h, low=l, close=c, volume=v, vwap=vwap).items():
        ns['S_' + k] = pd.Series(np.asarray(a, float))
    out = {}
    with np.errstate(all='ignore'):
        for f, n in zip(fields, names):
            out[n] = np.asarray(eval(re.sub(r'\$(\w+)', r'S_\1', f), {'__builtins__': {}}, ns), dtype=float)
    return pd.DataFrame(out)

def compare(P, R):
    """max_rel: largest |a - b| / |b|; max_scaled: largest |a - b| / sd(b) (a difference in units of the feature's spread;
    max_rel alone is large where the value itself is ~0, e.g. a residual of exactly 0)"""
    rows = []
    for n in A.NAMES:
        p, r = P[n].values, R[n].values
        both = np.isfinite(p) & np.isfinite(r)
        nan_mismatch = int((np.isnan(p) != np.isnan(r)).sum())
        d = np.abs(p[both] - r[both])
        rel = d / np.maximum(np.abs(r[both]), 1e-300)
        sd = np.std(r[both]) if both.sum() > 1 else 1.0
        rows.append((n, int(both.sum()), nan_mismatch, float(rel.max()) if both.any() else 0.0,
                     float(d.max() / sd) if both.any() and sd > 0 else 0.0))
    return pd.DataFrame(rows, columns=['feature', 'n', 'nan_mismatch', 'max_rel', 'max_scaled'])

def main():
    t0 = time.time()
    fields, names = expressions()
    print(f'1. expressions: {len(fields)} fields; names equal alpha158.NAMES in order: {names == A.NAMES}')
    miss = check_source()
    print('   every template occurs verbatim in qlib_alpha158_source.txt' if not miss else f'   NOT FOUND in the source: {miss}')
    stocks = sys.argv[1:] or ['2330', '2317', '2882', '2603', '6446', '2327']
    ev = A.event_table(DATA)
    worst = None
    for s in stocks:
        df = A._read_market(os.path.join(DATA, 'market', f'{s}.csv'))
        df = df[df['etl:adj_close'].notna()].reset_index(drop=True)
        o, h, l, c = [df[f'etl:adj_{k}'].values for k in ('open', 'high', 'low', 'close')]
        v = df['price:成交股數'].values.astype(float); vw = df['price:成交金額'].values / v
        Fct, nre = A.vwap_factor(df.date.values, l, h, vw, ev.get(s))
        P = A.stock_alpha158(o, h, l, c, v, vw * Fct); R = reference(o, h, l, c, v, vw * Fct)
        C = compare(P, R)
        exact = int((C.max_rel == 0).sum())
        print(f'2. {s}: {len(df)} days; identical in {exact} of 158 features; NaN pattern mismatches {int(C.nan_mismatch.sum())}; '
              f'largest difference {C.max_scaled.max():.1e} sd of the feature ({C.loc[C.max_scaled.idxmax(), "feature"]})')
        worst = C if worst is None else worst.assign(max_scaled=np.maximum(worst.max_scaled, C.max_scaled),
                                                     max_rel=np.maximum(worst.max_rel, C.max_rel), nan_mismatch=worst.nan_mismatch + C.nan_mismatch)
    fam = worst.assign(fam=worst.feature.str.replace(r'\d+$', '', regex=True)).groupby('fam', sort=False).max_scaled.max()
    print('   families that are not identical (largest difference in sd of the feature): ' +
          (', '.join(f'{k} {v:.1e}' for k, v in fam.items() if v > 0) or 'none'))
    # exact rational arithmetic on the rows where Slope / Resi differ most: which side is right
    from fractions import Fraction as Fr
    df = A._read_market(os.path.join(DATA, 'market', f'{stocks[0]}.csv')); df = df[df['etl:adj_close'].notna()].reset_index(drop=True)
    c = df['etl:adj_close'].values; em = eq = 0.0
    for d in (5, 10, 20):
        sl, _, rs = A._regression(c, d); qs, qr = Slope(pd.Series(c), d).values, Resi(pd.Series(c), d).values
        for i in np.argsort(-np.abs(rs - qr))[:20]:
            if i < d - 1:
                continue
            w = [Fr(float(x)) for x in c[i - d + 1: i + 1]]; xs = [Fr(k + 1) for k in range(d)]
            mx, my = sum(xs) / d, sum(w) / d
            b = sum((x - mx) * (y - my) for x, y in zip(xs, w)) / sum((x - mx) ** 2 for x in xs)
            r = float(w[-1] - (my + b * (xs[-1] - mx)))
            em = max(em, abs(rs[i] - r) / c[i]); eq = max(eq, abs(qr[i] - r) / c[i])
    print(f'   Slope/Resi rows with the largest differences ({stocks[0]}), error vs exact rational arithmetic / close: '
          f'alpha158.py {em:.1e}, qlib algorithm {eq:.1e} (qlib updates its sums incrementally, so rounding accumulates)')
    # 3. timing / consistency with the v9 panel
    X = A.build_alpha158(DATA)
    s = stocks[0]
    df = A._read_market(os.path.join(DATA, 'market', f'{s}.csv'))
    day = A.stock_frame(df, ev.get(s)); xs = X[X.stock_id == s].reset_index(drop=True)
    sh = np.nanmax(np.abs(xs[A.COLS].values[1:] - day[A.NAMES].values[:-1]))
    print(f'3. build_alpha158 row D == values of day D-1 ({s}): max abs diff {sh:.1e}; rows {len(X)}, stocks {X.stock_id.nunique()}, '
          f'last date {X.date.max().date()}')
    core = os.path.join(ROOT, 'v9', 'cache', 'core9.pkl')
    if os.path.exists(core):
        P9 = pd.read_pickle(core)[['date', 'stock_id', 'intraday_ret', 'ret5', 'ret1', 'range']]
        M = P9.merge(X[['date', 'stock_id', 'a158_KMID', 'a158_ROC5', 'a158_KLEN']], on=['date', 'stock_id'])
        M = M.dropna()
        d1 = np.abs(M.a158_KMID - np.expm1(M.intraday_ret)).max(); d2 = np.abs(M.a158_ROC5 - np.exp(-M.ret5)).max()
        d3 = np.abs(M.a158_KLEN - M['range'] * np.exp(M.intraday_ret)).max()
        print(f'   vs the v9 panel ({len(M)} rows): KMID vs exp(intraday_ret)-1 {d1:.1e}; ROC5 vs exp(-ret5) {d2:.1e}; '
              f'KLEN vs range*close/open {d3:.1e}')
    # 4. VWAP factor: bound on the error, restarts, days set aside, the resulting VWAP0
    H, nre, nas, ndays = [], 0, 0, 0
    for s in sorted(X.stock_id.unique()):
        df = A._read_market(os.path.join(DATA, 'market', f'{s}.csv')); df = df[df['etl:adj_close'].notna()].reset_index(drop=True)
        l, h = df['etl:adj_low'].values, df['etl:adj_high'].values; vw = df['price:成交金額'].values / df['price:成交股數'].values
        _, r, half, a = A.vwap_factor(df.date.values, l, h, vw, ev.get(s), full=True)
        H.append(pd.Series(half, index=df.date.values)); nre += r; nas += a; ndays += len(df)
    H = pd.concat(H); H22 = H[H.index >= '2022-01-01']
    print(f'4. VWAP factor ({ndays} stock-days): restarts {nre}, days set aside {nas}; bound on |log F error| (half width of the '
          f'intersection, valid while no unknown event): median {np.nanmedian(H):.1e}, 90% {np.nanquantile(H, 0.9):.1e}, '
          f'99% {np.nanquantile(H, 0.99):.1e}; from 2022 on median {np.nanmedian(H22):.1e}, 99% {np.nanquantile(H22, 0.99):.1e}')
    v0 = X['a158_VWAP0']
    print(f'   VWAP0 (vwap / close of the day before): median {v0.median():.4f}, 1% {v0.quantile(0.01):.4f}, 99% {v0.quantile(0.99):.4f}, '
          f'min {v0.min():.4f}, max {v0.max():.4f}; within [LOW0, HIGH0] on {((v0 >= X.a158_LOW0 - 1e-12) & (v0 <= X.a158_HIGH0 + 1e-12))[v0.notna()].mean():.4f} of the rows')
    nanfrac = X[X.date >= '2020-06-01'][A.COLS].isna().mean()
    print(f'5. missing values from 2020-06 on: {int((nanfrac > 0.0011).sum())} features above 0.11% of the rows'
          + (f' ({", ".join(nanfrac[nanfrac > 0.0011].index.str[5:])})' if (nanfrac > 0.0011).any() else ''))
    print(f'done in {time.time() - t0:.0f}s')

if __name__ == '__main__':
    main()

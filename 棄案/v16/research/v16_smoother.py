r"""
v16_smoother.py -- v16 part 2 (research\PREREG_v16b.md): the Kalman smoother (Rauch-Tung-Striebel). numpy + pandas only.

  S1  smoother on the data up to D-1 only: must give exactly the filter's prediction for D (README_v16 theorem 5) -- a check
  S2  smoother on all the data, the value of D (or of D-1) used for day D: it uses day D and later -> report only, NOT usable
  S3  smoother only inside the training window, to make the labels of a linear model; judged against the same model with
      the raw label (S3-0): (v11 + S3x) - (v11 + S3-0), val > 0 -> selected, combined hold > 0 and post > 0 -> passed

reads  ..\cache\signals16.pkl (v16_signals.py), the cached walk-forward predictions (v16lib.Data), v9\cache\core9.pkl
writes ..\results\v16_smoother_daily.csv, ..\results\v16_smoother_selection.json; the report goes to stdout
usage (in v16\research): python v16_smoother.py > ..\results\v16_smoother.txt
"""
import os, sys, json, math, time
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v16lib as L
import v16_eval as E
from v16lib import per, w, nw_t, diff, fmt, PER, YEARS

TRAIN_START = pd.Timestamp('2020-03-02')
S3_HALF = {'S3a': 5, 'S3b': 20}
S3F_HALF = 5
COEF_SHOW = ['rel_intra_prev', 'intraday_ret', 'ret1', 'y_mean5', 'intra_bias_20', 'mkt_intra_prev']


def gain(h):
    return 1 - 0.5 ** (1.0 / h)


# ================================================================== smoothers
def k1_fs(M, m, V, phi, q):
    """K1 filter (= v16lib.k1_filter for one grid point) plus the RTS smoother. returns (pred beta t|t-1, smoothed beta t|T), T x N"""
    y = M.values; mm = m.values; Vv = V.values
    T, N = y.shape
    A = np.zeros(N); B = np.zeros(N); P00 = np.zeros(N); P01 = np.zeros(N); P11 = np.zeros(N)
    started = np.zeros(N, bool); st = np.zeros((T, N), bool)
    fa = np.full((T, N), np.nan); fb = np.full((T, N), np.nan)
    f00 = np.full((T, N), np.nan); f01 = np.full((T, N), np.nan); f11 = np.full((T, N), np.nan)
    pb = np.full((T, N), np.nan)
    for t in range(T):
        obs = np.isfinite(y[t]) & np.isfinite(Vv[t]) & np.isfinite(mm[t])
        s = started
        if s.any():
            B[s] = 1.0 + phi * (B[s] - 1.0); P11[s] = phi * phi * P11[s] + q; P01[s] = phi * P01[s]
        new = obs & ~started
        if new.any():
            A[new] = 0.0; B[new] = L.K1_B_MEAN; P00[new] = L.K1_A_SD ** 2; P01[new] = 0.0; P11[new] = L.K1_B_SD ** 2
            started = started | new
        pb[t, started] = B[started]
        if obs.any():
            o = obs; mt = mm[t]
            v = y[t, o] - A[o] - B[o] * mt
            F = P00[o] + 2 * mt * P01[o] + mt * mt * P11[o] + Vv[t, o]
            K0 = (P00[o] + mt * P01[o]) / F; K1 = (P01[o] + mt * P11[o]) / F
            A[o] += K0 * v; B[o] += K1 * v
            P00[o] -= K0 * K0 * F; P01[o] -= K0 * K1 * F; P11[o] -= K1 * K1 * F
        st[t] = started
        fa[t, started] = A[started]; fb[t, started] = B[started]
        f00[t, started] = P00[started]; f01[t, started] = P01[started]; f11[t, started] = P11[started]
    sa = fa.copy(); sb = fb.copy()
    for t in range(T - 2, -1, -1):
        ok = st[t]
        if not ok.any():
            continue
        p00 = f00[t]; p01 = phi * f01[t]; p11 = phi * phi * f11[t] + q                  # P_{t+1|t}
        det = p00 * p11 - p01 * p01
        i00, i01, i11 = p11 / det, -p01 / det, p00 / det
        a00, a01, a10, a11 = f00[t], phi * f01[t], f01[t], phi * f11[t]                  # P_{t|t} Phi'
        J00 = a00 * i00 + a01 * i01; J01 = a00 * i01 + a01 * i11
        J10 = a10 * i00 + a11 * i01; J11 = a10 * i01 + a11 * i11
        da = sa[t + 1] - fa[t]; db = sb[t + 1] - (1.0 + phi * (fb[t] - 1.0))
        sa[t] = np.where(ok, fa[t] + J00 * da + J01 * db, np.nan)
        sb[t] = np.where(ok, fb[t] + J10 * da + J11 * db, np.nan)
    return pb, sb


def k2_fs(y, x, Rv, qa, qb):
    """K2 filter (= v16lib.k2_filter for one grid point) plus the RTS smoother; returns pa, pb (t|t-1) and sa, sb (t|T)"""
    T = len(y)
    a = 0.0; b = L.K2_B_MEAN; P00 = L.K2_A_SD ** 2; P01 = 0.0; P11 = L.K2_B_SD ** 2
    pa = np.empty(T); pb = np.empty(T); fa = np.empty(T); fb = np.empty(T)
    f = np.empty((T, 3)); pp = np.empty((T, 3))
    for t in range(T):
        if t > 0:
            P00 = P00 + qa * Rv[t]; P11 = P11 + qb * Rv[t] / L.K2_SX ** 2
        pa[t] = a; pb[t] = b; pp[t] = (P00, P01, P11)
        if np.isfinite(y[t]) and np.isfinite(x[t]) and np.isfinite(Rv[t]):
            xt = x[t]; v = y[t] - a - b * xt
            F = P00 + 2 * xt * P01 + xt * xt * P11 + Rv[t]
            K0 = (P00 + xt * P01) / F; K1 = (P01 + xt * P11) / F
            a = a + K0 * v; b = b + K1 * v
            P00, P01, P11 = P00 - K0 * K0 * F, P01 - K0 * K1 * F, P11 - K1 * K1 * F
        fa[t] = a; fb[t] = b; f[t] = (P00, P01, P11)
    sa = fa.copy(); sb = fb.copy()
    for t in range(T - 2, -1, -1):
        p00, p01, p11 = pp[t + 1]                                                          # P_{t+1|t} (Phi = I)
        det = p00 * p11 - p01 * p01
        i00, i01, i11 = p11 / det, -p01 / det, p00 / det
        g00, g01, g11 = f[t]
        J00 = g00 * i00 + g01 * i01; J01 = g00 * i01 + g01 * i11
        J10 = g01 * i00 + g11 * i01; J11 = g01 * i01 + g11 * i11
        da = sa[t + 1] - fa[t]; db = sb[t + 1] - fb[t]
        sa[t] = fa[t] + J00 * da + J01 * db; sb[t] = fb[t] + J10 * da + J11 * db
    return pa, pb, sa, sb


def ll_fs(y, q, P0=1.0):
    """local level (= v16lib.local_level for one q, P in units of Var(eps)) plus the RTS smoother; returns pred (t|t-1), smoothed (t|T)"""
    y = np.asarray(y, float); T = len(y)
    a = 0.0; P = P0
    pred = np.empty(T); fa = np.empty(T); fp = np.empty(T)
    for t in range(T):
        pred[t] = a
        if np.isfinite(y[t]):
            F = P + 1.0; K = P / F
            a = a + K * (y[t] - a); P = P * (1 - K)
        fa[t] = a; fp[t] = P
        P = P + q
    s = fa.copy()
    for t in range(T - 2, -1, -1):
        J = fp[t] / (fp[t] + q) if (fp[t] + q) > 0 else 1.0
        s[t] = fa[t] + J * (s[t + 1] - fa[t])
    return pred, s


def ll_smooth_matrix(Y, q, P0=1e4):
    """local level RTS smoother of every column of Y (T x N, NaN = no observation), P in units of Var(eps)"""
    T, N = Y.shape
    a = np.zeros(N); P = np.full(N, P0)
    fa = np.empty((T, N)); fp = np.empty((T, N))
    for t in range(T):
        o = np.isfinite(Y[t])
        F = P + 1.0; K = P / F
        a = np.where(o, a + K * (np.where(o, Y[t], 0.0) - a), a)
        P = np.where(o, P * (1 - K), P)
        fa[t] = a; fp[t] = P
        P = P + q
    s = fa.copy()
    for t in range(T - 2, -1, -1):
        J = fp[t] / (fp[t] + q)
        s[t] = fa[t] + J * (s[t + 1] - fa[t])
    return s


def forward_ewma_matrix(Y, K):
    """f_t = K e_t + (1 - K) f_{t+1} within the window, last day f = e; a missing e_t carries f_{t+1}"""
    T, N = Y.shape
    f = np.full((T, N), np.nan); nxt = np.full(N, np.nan)
    for t in range(T - 1, -1, -1):
        e = Y[t]; o = np.isfinite(e)
        cur = np.where(o & np.isfinite(nxt), K * np.where(o, e, 0.0) + (1 - K) * np.where(np.isfinite(nxt), nxt, 0.0),
                       np.where(o, e, nxt))
        f[t] = cur; nxt = cur
    return f


# ================================================================== helpers (= v15lib)
def c(s):
    return s - s.groupby(level='date').transform('mean')


def sdd(s):
    return s.groupby(level='date').transform('std')


def scale_to(y, ref):
    k = (sdd(c(ref)) / sdd(c(y))).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return ref.groupby(level='date').transform('mean') + c(y) * k


def ic_daily(pred, r):
    X = pd.DataFrame({'p': pred, 'r': r}).dropna()
    g = X.groupby(level='date')
    R = pd.DataFrame({'p': g.p.rank(), 'r': g.r.rank()})
    R = R - R.groupby(level='date').transform('mean')
    s = pd.DataFrame({'xy': R.p * R.r, 'xx': R.p ** 2, 'yy': R.r ** 2}).groupby(level='date').sum()
    return s.xy / np.sqrt(s.xx * s.yy)


def k2_term(a, b, a0, b0, x):
    den = np.maximum(b0, L.K2_FLOOR)
    mult = np.clip(1.0 + (b - b0) / den, *L.K2_MULT)
    icpt = np.clip((a - a0) / den, -L.K2_ICPT, L.K2_ICPT)
    return L.W_T * (np.asarray(x, float) * mult + icpt)


def fold_of(dates):
    out = np.empty(len(dates), dtype=object)
    for fa, fb in L.FOLDS:
        out[(dates >= pd.Timestamp(fa)) & (dates < pd.Timestamp(fb))] = fa
    return out


# ================================================================== S3: smoothed labels for a linear model
def s3_predictions(log):
    B = list(json.load(open(os.path.join(L.ROOT, 'transformer', 'final_feature_lists.json')))['B_competition_only'])
    P = pd.read_pickle(os.path.join(L.ROOT, 'v9', 'cache', 'core9.pkl'))[['date', 'stock_id', 'resid_r'] + B].copy()
    P['date'] = pd.to_datetime(P.date).astype('datetime64[ns]'); P['stock_id'] = P.stock_id.astype(str)
    P = P.sort_values(['date', 'stock_id']).reset_index(drop=True)
    P['e'] = P.resid_r.clip(-0.05, 0.05)
    E_ = P.pivot(index='date', columns='stock_id', values='e').sort_index()
    preds = {k: [] for k in ('S3-0', 'S3a', 'S3b', 'S3f')}
    coefs = {k: [] for k in preds}; r2lag = {k: [] for k in preds}
    for fa, fb in L.FOLDS:
        f0, f1 = pd.Timestamp(fa), pd.Timestamp(fb)
        tr = (P.date >= TRAIN_START) & (P.date < f0) & P.e.notna()
        te = (P.date >= f0) & (P.date < f1)
        Xtr = P.loc[tr, B].to_numpy(float); Xte = P.loc[te, B].to_numpy(float)
        lo = np.nanquantile(Xtr, 0.01, axis=0); hi = np.nanquantile(Xtr, 0.99, axis=0)
        Xtr = np.clip(Xtr, lo, hi); Xte = np.clip(Xte, lo, hi)
        mu = np.nanmean(Xtr, axis=0); sd = np.nanstd(Xtr, axis=0); sd[~(sd > 0)] = 1.0
        Xtr = np.nan_to_num((Xtr - mu) / sd); Xte = np.nan_to_num((Xte - mu) / sd)
        Xtr = np.column_stack([np.ones(len(Xtr)), Xtr]); Xte = np.column_stack([np.ones(len(Xte)), Xte])
        W = E_[(E_.index >= TRAIN_START) & (E_.index < f0)]
        lab = {'S3-0': W.values}
        for k, h in S3_HALF.items():
            K = gain(h); lab[k] = ll_smooth_matrix(W.values, K * K / (1 - K))
        lab['S3f'] = forward_ewma_matrix(W.values, gain(S3F_HALF))
        keys = pd.MultiIndex.from_arrays([P.loc[tr, 'date'].values, P.loc[tr, 'stock_id'].values])
        rip = P.loc[tr, 'rel_intra_prev'].to_numpy(float)
        for k, Mx in lab.items():
            Sx = pd.DataFrame(Mx, index=W.index, columns=W.columns).stack(future_stack=True) if hasattr(pd.DataFrame, 'stack') else None
            yk = Sx.reindex(keys).to_numpy(float)
            ok = np.isfinite(yk)
            coef, *_ = np.linalg.lstsq(Xtr[ok], yk[ok], rcond=None)
            preds[k].append(pd.Series(Xte @ coef, index=pd.MultiIndex.from_arrays([P.loc[te, 'date'].values, P.loc[te, 'stock_id'].values],
                                                                                names=['date', 'stock_id'])))
            coefs[k].append({f: float(coef[1 + B.index(f)]) for f in COEF_SHOW})
            okr = ok & np.isfinite(rip)
            r2lag[k].append(float(np.corrcoef(yk[okr], rip[okr])[0, 1] ** 2))
        log(f'  S3 fold {fa}: {int(tr.sum()):,} training rows, {int(te.sum()):,} test rows')
    out = {k: pd.concat(v).sort_index() for k, v in preds.items()}
    return out, coefs, r2lag, P.set_index(['date', 'stock_id']).resid_r


def main():
    t0 = time.time()
    print('v16_smoother.py -- v16 part 2: the Kalman smoother (research\\PREREG_v16b.md)')
    print(f'  {time.strftime("%Y-%m-%d %H:%M")}, pandas {pd.__version__}, numpy {np.__version__}')
    S = pd.read_pickle(os.path.join(L.CACHE, 'signals16.pkl'))
    d = L.Data(); dates = d.dates; fold = fold_of(dates)
    pe11 = d.view([d.LGB, d.x1])
    D = {'v11': d.run(pe11, True), 'W0': d.run(pe11, True, w_t=0.0, lam=0.5)}
    x11 = d.pbar(night=True).reindex(dates) - 0.5
    K2 = E.k2_build(d, x11, S['m50'], dates)
    D['K3'], use3, sel3, mu3 = L.k3_run(D['v11'], D['W0'])
    print(f'  part-1 objects rebuilt ({time.time() - t0:.0f}s)')

    # ------------------------------------------------------------ S1
    print('\nS1 -- smoother on the data up to D-1 only vs the part-1 filter prediction for D (must be identical, README_v16 theorem 5)')
    rng = np.random.default_rng(1603)
    M, m = S['M'], S['m']
    wfd = [t for t in dates if t >= pd.Timestamp('2022-01-03')]
    pick = [wfd[i] for i in sorted(rng.choice(len(wfd), size=6, replace=False))]
    worst1 = 0.0
    for D_ in pick:
        prev = M.index[M.index.get_loc(D_) - 1]
        Mc, mc, _, _ = L.price_panel(end=prev, extra_dates=[D_])
        Vc = L.resid_var(Mc, mc)
        phi, q = L.K1_GRID[S['sel'][fold[dates.get_loc(D_)]]]
        _, sbc = k1_fs(Mc, mc, Vc, phi, q)
        a = pd.Series(sbc[-1], index=Mc.columns); b = S['K1'].xs(D_, level='date').reindex(Mc.columns)
        ok = a.notna() | b.notna()
        worst1 = max(worst1, float((a[ok] - b[ok]).abs().max()))
    print(f'  K1: 6 random days, max |smoothed-on-data-to-D-1 beta at D - filter beta D|D-1| = {worst1:.1e}')
    worst2 = 0.0
    pick2 = [t for t in pick if t >= pd.Timestamp('2023-01-01')] or pick
    for D_ in pick2:
        i = dates.get_loc(D_)
        yc = K2['y'].values[: i + 1].copy(); yc[-1] = np.nan
        g = K2['sel'][fold[i]]; qa, qb = L.K2_GRID[g]
        _, _, sa, sb = k2_fs(yc, K2['x'].values[: i + 1], K2['R'].values[: i + 1], qa, qb)
        _, _, sa0, sb0 = k2_fs(yc, K2['x'].values[: i + 1], K2['R'].values[: i + 1], 0.0, 0.0)
        Tt = k2_term(sa[-1], sb[-1], sa0[-1], sb0[-1], K2['x'].values[i])
        worst2 = max(worst2, abs(float(Tt) - float(K2['T'].iloc[i])))
    print(f'  K2: {len(pick2)} days, max |timing term from the smoother on data to D-1 - part-1 K2 term| = {worst2:.1e}')
    dl = (D['v11'].AR - D['W0'].AR).reindex(dates).values
    worst3 = 0.0
    for D_ in pick:
        i = dates.get_loc(D_)
        yc = dl[: i + 1].copy(); yc[-1] = np.nan
        q = L.K3_Q[sel3[fold[i]]]
        _, s = ll_fs(yc, q)
        worst3 = max(worst3, abs(float(s[-1]) - float(mu3[i])))
    print(f'  K3: 6 days, max |smoothed mu at D (data to D-1) - part-1 mu D|D-1| = {worst3:.1e}')
    s1_ok = max(worst1, worst2, worst3) < 1e-12
    print('  ' + ('identical: S1 = K1, K2, K3 of part 1 (same AR), as theorem 5 says' if s1_ok else 'NOT identical: check the programs'))

    # ------------------------------------------------------------ S2
    print('\nS2 -- smoother on ALL the data (uses day D and later): report only, cannot be used live')
    rows2 = []
    # K1
    settings = sorted(set(L.K1_GRID[g] for g in S['sel'].values()))
    sm = {}
    for phi, q in settings:
        _, sb = k1_fs(M, m, S['M'].pipe(lambda X: L.resid_var(X, m)), phi, q)
        sm[(phi, q)] = pd.DataFrame(sb, index=M.index, columns=M.columns)
    BT = pd.DataFrame(np.nan, index=M.index, columns=M.columns)
    for fa, fb in L.FOLDS:
        rr = (M.index >= pd.Timestamp(fa)) & (M.index < pd.Timestamp(fb))
        BT.loc[rr] = sm[L.K1_GRID[S['sel'][fa]]].loc[rr].values
    BT1 = pd.DataFrame(np.nan, index=M.index, columns=M.columns)
    for fa, fb in L.FOLDS:
        rr = (M.index >= pd.Timestamp(fa)) & (M.index < pd.Timestamp(fb))
        BT1.loc[rr] = sm[L.K1_GRID[S['sel'][fa]]].shift(1).loc[rr].values
    for lab, Bx in (('D|T', BT), ('D-1|T', BT1)):
        b = L.hedge_ready(L.long(Bx, 'b'), d.INFO.index)
        D[f'S2-K1 {lab}'] = d.run(pe11, True, beta=b); D[f'S2-K1 {lab} W0'] = d.run(pe11, True, w_t=0.0, lam=0.5, beta=b)
        rows2.append((f'S2-K1 smoothed beta {lab}', diff(D, f'S2-K1 {lab}', 'v11'), diff(D, f'S2-K1 {lab} W0', 'W0')))
    # K2
    yv, xv, Rv = K2['y'].values, K2['x'].values, K2['R'].values
    cache2 = {}
    def run2(qa, qb):
        if (qa, qb) not in cache2:
            cache2[(qa, qb)] = k2_fs(yv, xv, Rv, qa, qb)
        return cache2[(qa, qb)]
    _, _, sa0, sb0 = run2(0.0, 0.0)
    def term(sa, sb, lag):
        if lag:
            sa, sb, a0, b0 = np.r_[sa[0], sa[:-1]], np.r_[sb[0], sb[:-1]], np.r_[sa0[0], sa0[:-1]], np.r_[sb0[0], sb0[:-1]]
        else:
            a0, b0 = sa0, sb0
        return pd.Series(k2_term(sa, sb, a0, b0, xv), index=dates)
    k2_sets = {'chosen q per fold': None, '(1e-3, 0)': (1e-3, 0.0), '(0, 1e-3)': (0.0, 1e-3), '(0, 1e-2)': (0.0, 1e-2), '(1e-3, 1e-2)': (1e-3, 1e-2)}
    for name, qq in k2_sets.items():
        for lag, lab in ((False, 'D|T'), (True, 'D-1|T')):
            if qq is None:
                Tt = pd.Series(np.nan, index=dates)
                for fa, _ in L.FOLDS:
                    qa, qb = L.K2_GRID[K2['sel'][fa]]
                    _, _, sa, sb = run2(qa, qb)
                    tt = term(sa, sb, lag); rr = fold == fa
                    Tt[rr] = tt[rr]
            else:
                _, _, sa, sb = run2(*qq); Tt = term(sa, sb, lag)
            key = f'S2-K2 {name} {lab}'
            D[key] = d.run(pe11, True, timing=Tt)
            rows2.append((f'S2-K2 {name} {lab}', diff(D, key, 'v11'), None))
    # K3
    k3_sets = {'chosen q per fold': None, 'half-life 5': 5, 'half-life 20': 20, 'half-life 60': 60}
    for name, h in k3_sets.items():
        for lag, lab in ((False, 'D|T'), (True, 'D-1|T')):
            if h is None:
                mu = np.full(len(dates), np.nan)
                for fa, _ in L.FOLDS:
                    _, s = ll_fs(dl, L.K3_Q[sel3[fa]]); rr = fold == fa
                    mu[rr] = (np.r_[0.0, s[:-1]] if lag else s)[rr]
            else:
                _, s = ll_fs(dl, L.q_from_half_life(h)); mu = np.r_[0.0, s[:-1]] if lag else s
            key = f'S2-K3 {name} {lab}'
            D[key] = L.switch_frame(D['v11'], D['W0'], pd.Series(mu >= 0, index=dates))
            rows2.append((f'S2-K3 {name} {lab} (W0 on {np.mean(mu < 0):.2f} of days)', diff(D, key, 'v11'), None))
    print('  each row: the leaky version - v11 (val / hold / post / all, NW t); for S2-K1 also the W0 version - W0')
    for name, r, rw in rows2:
        print(f'    {name:44s} {fmt(r)}' + (f'\n    {"":44s} W0: {fmt(rw)}' if rw is not None else ''))

    # ------------------------------------------------------------ S3
    print('\nS3 -- the smoother only for the training labels of a linear model (OLS on the 67 B features)')
    pr, coefs, r2lag, resid = s3_predictions(lambda s: print(s))
    idx = d.idx
    scaled = {}
    for k, p in pr.items():
        p = p.reindex(idx)
        miss = int(p.isna().sum())
        if miss:
            p = p.fillna(p.groupby(level='date').transform('mean'))
        scaled[k] = scale_to(p, d.LGB)
        D[f'v11 + {k}'] = d.run(d.view([d.LGB, d.x1, scaled[k]]), True)
        D[f'W0 + {k}'] = d.run(d.view([d.LGB, d.x1, scaled[k]]), True, w_t=0.0, lam=0.5)
    r_ = resid.reindex(idx)
    ICd = {k: ic_daily(c(pr[k].reindex(idx)), r_) for k in pr}
    print('  own daily rank IC with the excess return (100 stocks), val / hold / post / all:')
    for k in pr:
        line = f'    {k:5s} ' + ' / '.join(f'{per(ICd[k])[p]:.4f}' for p in PER)
        if k != 'S3-0':
            dd = (ICd[k] - ICd['S3-0']).dropna()
            line += '   minus S3-0: ' + '  '.join(f'{p} {dd[w(dd.index, p)].mean():+.4f} ({nw_t(dd[w(dd.index, p)].values):.2f})' for p in PER)
        print(line)
    print('  standardized OLS coefficients (mean over the 6 folds) and R^2 of the label on yesterday\'s excess return (rel_intra_prev) in training:')
    print('    label  ' + '  '.join(f'{f:>15s}' for f in COEF_SHOW) + '   R^2 on e_{t-1}')
    for k in pr:
        mc_ = {f: np.mean([cf[f] for cf in coefs[k]]) for f in COEF_SHOW}
        print(f'    {k:5s}  ' + '  '.join(f'{mc_[f] * 1e4:+15.2f}' for f in COEF_SHOW) + f'   {np.mean(r2lag[k]):.3f}')
    print('    (coefficients in bp of the label per 1 sd of the feature; theorem 6 for white noise: R^2 0.104 (5 days), 0.032 (20 days))')

    print('\n  the rule (PREREG_v16b section 3): (v11 + S3x) - (v11 + S3-0), bp/day (NW t)')
    res = {}
    for k in ('S3a', 'S3b'):
        res[k] = diff(D, f'v11 + {k}', 'v11 + S3-0')
        print(f'    {k}: {fmt(res[k])}   -> ' + ('val > 0: selected' if res[k]['val'][0] > 0 else 'not selected'))
    sel = [k for k in ('S3a', 'S3b') if res[k]['val'][0] > 0]
    passed, rc = False, None
    if sel:
        comb = pd.concat([scaled[k] for k in sel], axis=1).mean(axis=1)
        D['v11 + S3 selected'] = d.run(d.view([d.LGB, d.x1, comb]), True)
        rc = diff(D, 'v11 + S3 selected', 'v11 + S3-0'); passed = rc['hold'][0] > 0 and rc['post'][0] > 0
        print(f'    selected {", ".join(sel)}: {fmt(rc)} -> ' + ('PASSED (research only)' if passed else 'NOT PASSED'))
        print(f'    against v11 itself: {fmt(diff(D, "v11 + S3 selected", "v11"))}')
    else:
        print('    nothing selected -> NOT PASSED: v9, v11 and W0 stay as they are')
    print('\n  report only:')
    for k in pr:
        print(f'    v11 + {k:4s} - v11: {fmt(diff(D, f"v11 + {k}", "v11"))}')
    for k in ('S3a', 'S3b', 'S3f'):
        print(f'    {k} - S3-0, W0 versions: {fmt(diff(D, f"W0 + {k}", "W0 + S3-0"))}')
    print('    by year, (v11 + S3x) - (v11 + S3-0):')
    for k in ('S3a', 'S3b', 'S3f'):
        print(f'      {k}: {fmt(diff(D, f"v11 + {k}", "v11 + S3-0", YEARS))}')

    os.makedirs(L.RESULTS, exist_ok=True)
    pd.DataFrame({k: v.AR for k, v in D.items()}).to_csv(os.path.join(L.RESULTS, 'v16_smoother_daily.csv'), float_format='%.4f')
    json.dump({'s1_max_diff': {'K1': worst1, 'K2': worst2, 'K3': worst3}, 's1_identical': bool(s1_ok),
               's2': {name: {'v11': r, 'W0': rw} for name, r, rw in rows2}, 's3_each_minus_control': res, 's3_selected': sel,
               's3_combined_minus_control': rc, 's3_passed': bool(passed),
               's3_ic': {k: per(ICd[k]) for k in ICd}, 's3_r2_lag': {k: float(np.mean(v)) for k, v in r2lag.items()},
               's3_coefs': {k: {f: float(np.mean([cf[f] for cf in coefs[k]])) for f in COEF_SHOW} for k in coefs}},
              open(os.path.join(L.RESULTS, 'v16_smoother_selection.json'), 'w'), indent=1, default=float)
    print(f'\nsaved results\\v16_smoother_daily.csv, results\\v16_smoother_selection.json ({time.time() - t0:.0f}s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())

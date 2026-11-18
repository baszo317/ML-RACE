r"""
adapt_eval.py -- research\PREREG_adapt.md: choose v9's combination coefficients every day from the results of the last
N trading days (only days whose close is known before the day) instead of keeping them fixed.  Nothing is retrained:
the member predictions are those of v9_eval.py (walk-forward 2022-01..2026-10) and, as a second check, the rolling static
members out\members_v9_static.csv.gz (2026-06-01..10-22, every day predicted with models trained up to the day before).

  cd v9\research
  python adapt_eval.py > ..\results\adapt_eval.txt        (numpy + pandas only; a few minutes; grids cached in ..\cache)

Coefficients (v9 value; grid):  w_t 0.5 (0..1.5), lam 0.75 (0..1), lgb_w 0.5 (0..1), tf_w 0.5 (0..1), bias 0 (-0.02..0.02)
  p_raw = (1 - tf_w) * mean(3 raw MLPs) + tf_w * Transformer,  pbar = day mean of p_raw over the 100 stocks
  p_e   = (1 - lgb_w) * mean(4 resid MLPs) + lgb_w * LightGBM,  p_e_c = p_e - day mean over the 100
  score = w_t * (pbar - 0.5) + p_e_c - bias ; 1 if score > 0 ; 0 on the 4 ex-dividend flags ; beta hedge with lam
With the v9 values every day equals v9_eval.py ('v9: all 11 changes', cache\v9_daily.pkl) and out\probabilities_v9_static.csv.
Daily AR (bp, Top50) of every one of the 4375 combinations is computed once (grid_day) and cached; a scheme then only
picks, for day t, the combination with the best mean AR over days t-N..t-1 (ties: the one closest to v9).
"""
import os, sys, time, tempfile
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); V9 = os.path.dirname(HERE); ROOT = os.path.dirname(V9)
sys.path.insert(0, os.path.join(ROOT, 'v8', 'research'))
import mlib as L                                      # periods, plain_t, nw_t (v8 helpers, unchanged)
CACHE = os.path.join(V9, 'cache'); V8C = os.path.join(ROOT, 'v8', 'cache'); OUT = os.path.join(V9, 'out')
GRID_FILE = os.path.join(CACHE, 'adapt_grid.npz')
CGRID = np.round(np.arange(-0.3, 0.3 + 1e-9, 0.0025), 6)
EX4 = ('ex_today', 'ex_next1', 'ex_prev1', 'ex_prev2')
RAW9 = ['mlp_raw_B9', 'mlp_raw_xr9', 'mlp_raw_w4_9']; RES9 = ['mlp_res_B9', 'mlp_res_xr9', 'mlp_res_BN9', 'mlp_res_xrN9']
NAMES = ['w_t', 'lam', 'lgb_w', 'tf_w', 'bias']
GRID = {'w_t': [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5], 'lam': [0.0, 0.25, 0.5, 0.75, 1.0], 'lgb_w': [0.0, 0.25, 0.5, 0.75, 1.0],
        'tf_w': [0.0, 0.25, 0.5, 0.75, 1.0], 'bias': [-0.02, -0.01, 0.0, 0.01, 0.02]}
V9DEF = {'w_t': 0.5, 'lam': 0.75, 'lgb_w': 0.5, 'tf_w': 0.5, 'bias': 0.0}
SHAPE = tuple(len(GRID[k]) for k in NAMES)
IDX = np.indices(SHAPE).reshape(len(SHAPE), -1).T                  # (K, 5) grid positions of every combination
DEFPOS = np.array([GRID[k].index(V9DEF[k]) for k in NAMES])
K_DEF = int(np.ravel_multi_index(tuple(DEFPOS), SHAPE))
STATIC0, STATIC1 = pd.Timestamp('2026-06-01'), pd.Timestamp('2026-09-30')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)

# ------------------------------------------------------------------ inputs
def read_csv(f):
    x = pd.read_csv(f, dtype={'stock_id': str}, parse_dates=['date']); x['date'] = x.date.astype('datetime64[ns]')
    return x.set_index(['date', 'stock_id']).sort_index()

def info_table(idx):
    """v9_eval.INFO: Top50 rows with a result, the 4 ex-dividend flags, beta (clip 0.2..2.5, day median, 1.0)"""
    P = pd.read_pickle(os.path.join(CACHE, 'core9.pkl')); P['date'] = P.date.astype('datetime64[ns]'); P['stock_id'] = P.stock_id.astype(str)
    P = P.set_index(['date', 'stock_id'])
    B = pd.read_pickle(os.path.join(V8C, 'beta_panel.pkl')); B['date'] = B.date.astype('datetime64[ns]'); B['stock_id'] = B.stock_id.astype(str)
    B = B.set_index(['date', 'stock_id']).beta
    I = pd.DataFrame({'top50': P.top50, 'target_r': P.target_r, **{c: P[c] for c in EX4}}).reindex(idx)
    I['beta'] = B.reindex(idx).clip(0.2, 2.5)
    I['beta'] = I.beta.fillna(I.groupby(level='date').beta.transform('median')).fillna(1.0)
    return I[I.top50.astype(bool) & I.target_r.notna()]

def families(raw_mlp, tf, res_mlp, lgb, idx):
    """pbar per tf_w and p_e_c per lgb_w, with the same pandas operations as v9_eval.views / rolling9.combine for the v9 values"""
    pbar, pec = {}, {}
    for w in GRID['tf_w']:
        p = pd.concat([raw_mlp, tf], axis=1).mean(axis=1) if w == 0.5 else (1 - w) * raw_mlp + w * tf
        pbar[w] = p.groupby(level='date').transform('mean')
    for w in GRID['lgb_w']:
        p = pd.concat([res_mlp, lgb], axis=1).mean(axis=1) if w == 0.5 else (1 - w) * res_mlp + w * lgb
        pec[w] = p - p.groupby(level='date').transform('mean')
    return pd.DataFrame({f'pbar_{w}': v for w, v in pbar.items()}).reindex(idx), pd.DataFrame({f'pec_{w}': v for w, v in pec.items()}).reindex(idx)

def research_inputs():
    W9 = read_csv(os.path.join(CACHE, 'wf9_preds.csv.gz'))
    M = read_csv(os.path.join(V8C, 'wf_members.csv.gz'))
    LGB = read_csv(os.path.join(V8C, 'tab_preds.csv.gz')).lgb_resid
    idx = M.index.intersection(W9.dropna().index).intersection(LGB.index)
    M, W9, LGB = M.loc[idx], W9.loc[idx], LGB.loc[idx]
    raw_mlp = pd.concat([W9[c] for c in RAW9], axis=1).mean(axis=1)
    res_mlp = pd.concat([W9[c] for c in RES9], axis=1).mean(axis=1)
    pb, pe = families(raw_mlp, pd.concat([M.tf_raw_B], axis=1).mean(axis=1), res_mlp, pd.concat([LGB], axis=1).mean(axis=1), idx)
    return idx, pb, pe

def static_inputs():
    S = read_csv(os.path.join(OUT, 'members_v9_static.csv.gz'))
    raw_mlp = pd.concat([S[c].dropna() for c in RAW9], axis=1).mean(axis=1)
    res_mlp = pd.concat([S[c].dropna() for c in RES9], axis=1).mean(axis=1)
    tf = pd.concat([S['tf_raw_B'].dropna()], axis=1).mean(axis=1); lgb = pd.concat([S['lgb_res'].dropna()], axis=1).mean(axis=1)
    idx = raw_mlp.index
    pb, pe = families(raw_mlp, tf, res_mlp, lgb, idx)
    return idx, pb, pe

def day_arrays(I, pb, pe):
    """per date: pec (n_lgb, n), pbar (n_tf,), ex (n,), beta (n,), r (n,) for the Top50 rows (stock_id order as v9_eval)"""
    X = I.join(pb, how='inner').join(pe, how='inner').sort_index()
    out = []
    pe_cols = [f'pec_{w}' for w in GRID['lgb_w']]; pb_cols = [f'pbar_{w}' for w in GRID['tf_w']]
    for d, g in X.groupby(level='date'):
        ex = np.zeros(len(g), bool)
        for c in EX4:
            ex |= g[c].values == 1
        out.append((pd.Timestamp(d), g[pe_cols].values.T.copy(), g[pb_cols].values[0].copy(), ex, g.beta.values.copy(), g.target_r.values.copy(),
                    g.index.get_level_values('stock_id').tolist()))
    return out

# ------------------------------------------------------------------ one day, every combination
WT = np.array(GRID['w_t']); LAMS = np.array(GRID['lam']); BIAS = np.array(GRID['bias'])

def grid_day(pec, pbar, ex, b, r, want=None):
    """daily AR (bp) of every combination, shape SHAPE = (w_t, lam, lgb_w, tf_w, bias); v9_eval.decide vectorised.
    want = flat combination index -> also return its 0/1 decisions (n,)"""
    n = len(r); nwt, nlam, nlgb, ntf, nb = SHAPE
    C = (WT[:, None] * (pbar[None, :] - 0.5))[:, :, None] - BIAS[None, None, :]      # market constant per (w_t, tf_w, bias)
    Cf = C.reshape(-1); Q = len(Cf)
    bb = b.mean(); notex = ~ex; cb = CGRID[:, None] * b[None, :]
    out = np.empty(SHAPE); dec = None
    wpos = np.unravel_index(want, SHAPE) if want is not None else None
    for k in range(nlgb):
        SC = pec[k][None, :] + Cf[:, None]                                            # (Q, n)  (v9: market + stock, same sum)
        s0 = np.where(ex[None, :], -1, np.where(SC > 0, 1, -1))
        ar1 = (s0 * r[None, :]).mean(axis=1) * 1e4
        sb = s0.mean(axis=1); B0 = ((s0 - sb[:, None]) * (b - bb)[None, :]).mean(axis=1)
        S = (SC[:, None, :] + cb[None, :, :] > 0) & notex[None, None, :]               # (Q, G, n)
        E = ((2 * S - 1) @ b) / n                                                     # (Q, G)
        for l, lam in enumerate(LAMS):
            if lam >= 1.0:
                ar, s = ar1, s0
            else:
                Et = sb * bb + lam * B0
                dd = np.abs(E - Et[:, None])
                cand = dd <= dd.min(axis=1, keepdims=True) + 1e-12
                j = np.argmin(np.where(cand, np.abs(CGRID)[None, :], np.inf), axis=1)
                s = np.where(S[np.arange(Q), j, :], 1, -1)
                ar = (s * r[None, :]).mean(axis=1) * 1e4
            out[:, l, k, :, :] = ar.reshape(nwt, ntf, nb)
            if wpos is not None and wpos[1] == l and wpos[2] == k:
                q = np.ravel_multi_index((wpos[0], wpos[3], wpos[4]), (nwt, ntf, nb))
                dec = (s[q] > 0).astype(int)
    return (out, dec) if want is not None else out

def grids(days, log, part, deadline=None):
    """AR of every combination for every day; resumable: progress is kept in `part` (.npz) when the time budget runs out"""
    K = int(np.prod(SHAPE)); A = np.full((len(days), K), np.nan); start = 0; t0 = time.time()
    if os.path.exists(part):
        z = np.load(part); n = int(z['n']); A[:n] = z['A'][:n]; start = n
        log(f'  resuming at day {start + 1}/{len(days)} ({os.path.basename(part)})')
    for i in range(start, len(days)):
        d, pec, pbar, ex, b, r, _ = days[i]
        A[i] = grid_day(pec, pbar, ex, b, r).reshape(-1)
        if (i + 1) % 200 == 0:
            log(f'  {i + 1}/{len(days)} days ({time.time() - t0:.0f}s)')
        if deadline is not None and time.time() > deadline and i + 1 < len(days):
            np.savez(part, A=A[:i + 1], n=i + 1)
            sys.exit(f'time budget used: {i + 1}/{len(days)} days done and kept in {part}; run again to continue')
    try:
        if os.path.exists(part):
            os.remove(part)
    except OSError:
        pass
    return A

# ------------------------------------------------------------------ schemes
def subset(free, near=False):
    """combinations whose fixed coefficients equal v9 (only the `free` ones move); near: at most one grid step from v9"""
    m = np.ones(len(IDX), bool)
    for i, k in enumerate(NAMES):
        if k not in free:
            m &= IDX[:, i] == DEFPOS[i]
        elif near:
            m &= np.abs(IDX[:, i] - DEFPOS[i]) <= 1
    return np.flatnonzero(m)

def dist(ks):
    return np.abs(IDX[ks] - DEFPOS[None, :]).sum(axis=1)

def choose(A, ks, N=15, ew=None):
    """flat index chosen for every day: argmax over ks of the mean AR of days t-N..t-1 (ew: exponential weights over all
    earlier days, half-life ew); ties -> fewest grid steps from v9; v9 itself while fewer than N earlier days exist"""
    T = len(A); X = A[:, ks]; pen = 1e-9 * dist(ks)[None, :]
    ch = np.full(T, K_DEF)
    if ew:
        rho = 0.5 ** (1.0 / ew); acc = np.zeros(len(ks))
        for t in range(T):
            if t >= N:
                ch[t] = ks[np.argmax(acc - pen[0])]
            acc = rho * acc + X[t]
        return ch
    cs = np.vstack([np.zeros((1, len(ks))), np.cumsum(X, axis=0)])
    for t in range(N, T):
        w = (cs[t] - cs[t - N]) / N
        ch[t] = ks[np.argmax(w - pen[0])]
    return ch

SCHEMES = [('S1', 'w_t only', ['w_t'], False, None), ('S2', 'lam only', ['lam'], False, None), ('S3', 'lgb_w only', ['lgb_w'], False, None),
           ('S4', 'tf_w only', ['tf_w'], False, None), ('S5', 'bias only', ['bias'], False, None),
           ('S6', 'w_t + lam + lgb_w (primary)', ['w_t', 'lam', 'lgb_w'], False, None), ('S7', 'all five', NAMES, False, None),
           ('S8', 'S6, exp. weights half-life 15', ['w_t', 'lam', 'lgb_w'], False, 15), ('S9', 'S6, at most one step from v9', ['w_t', 'lam', 'lgb_w'], True, None)]

def summary_row(name, ar, base, dates):
    pr = L.periods(dates); d = ar - base; r = {'scheme': name}
    for p in ('val', 'hold', 'post'):
        r[p] = ar[pr[p]].mean()
    r['all'] = ar.mean(); r['sd'] = ar.std(ddof=1)
    for p in ('val', 'hold', 'post'):
        r['d_' + p] = d[pr[p]].mean()
    r['d_all'] = d.mean(); r['t_all'] = L.plain_t(d); r['nw_all'] = L.nw_t(d, 5)
    return r

def coef_stats(ch):
    c = IDX[ch]
    return {k: np.array(GRID[k])[c[:, i]].mean() for i, k in enumerate(NAMES)} | {'moved': float((ch != K_DEF).mean())}

def ic_persistence():
    """smoother signal than AR: daily IC (Pearson, 100 stocks, return minus the day mean) of the two stock-view families,
    and the daily hit of the market view; is the recent winner still ahead on the next day?"""
    W9 = read_csv(os.path.join(CACHE, 'wf9_preds.csv.gz')); M = read_csv(os.path.join(V8C, 'wf_members.csv.gz'))
    LGB = read_csv(os.path.join(V8C, 'tab_preds.csv.gz')).lgb_resid
    idx = M.index.intersection(W9.dropna().index).intersection(LGB.index)
    P = pd.read_pickle(os.path.join(CACHE, 'core9.pkl')); P['date'] = P.date.astype('datetime64[ns]'); P['stock_id'] = P.stock_id.astype(str)
    P = P.set_index(['date', 'stock_id'])
    X = pd.DataFrame({'res': W9.loc[idx, RES9].mean(axis=1), 'lgb': LGB.loc[idx], 'raw': W9.loc[idx, RAW9].mean(axis=1),
                      'tf': M.loc[idx, 'tf_raw_B'], 'r': P.target_r.reindex(idx), 'top50': P.top50.reindex(idx)}).dropna()
    X['rc'] = X.r - X.groupby(level='date').r.transform('mean')
    def pers(d):
        d = np.asarray(d, float); tr = pd.Series(d).rolling(15).mean().shift(1).values; m = ~np.isnan(tr)
        nb = len(d) // 15; bm = d[:nb * 15].reshape(nb, 15).mean(axis=1)
        return np.corrcoef(tr[m], d[m])[0, 1], np.corrcoef(bm[:-1], bm[1:])[0, 1], np.mean(bm > 0)
    ic = X.groupby(level='date').apply(lambda g: pd.Series({c: np.corrcoef(g[c] - g[c].mean(), g.rc)[0, 1] for c in ('res', 'lgb')}))
    d = (ic.lgb - ic.res).values; c1, c2, w = pers(d)
    print(f'  stock view IC: 4 resid MLPs {ic.res.mean():.4f}, LightGBM {ic.lgb.mean():.4f}; LightGBM minus MLPs {d.mean():+.4f} '
          f'(daily sd {d.std():.4f}, so a 15-day mean has se {d.std() / np.sqrt(15):.4f}); corr15 {c1:+.3f}, block {c2:+.3f}, '
          f'LightGBM ahead in {w:.0%} of the 15-day blocks')
    day = X.groupby(level='date').agg(raw=('raw', 'mean'), tf=('tf', 'mean'))
    day = day.join(X[X.top50.astype(bool)].groupby(level='date').r.mean().rename('m')).dropna()
    hit = (np.sign((day.raw + day.tf) / 2 - 0.5) * day.m).values * 1e4
    c1, c2, w = pers(hit)
    print(f'  market view hit sign(pbar - 0.5) x Top50 mean return: {hit.mean():+.2f} bp/day; corr15 {c1:+.3f}, block {c2:+.3f}')
    d = ((np.sign(day.tf - day.tf.mean()) - np.sign(day.raw - day.raw.mean())) * day.m).values * 1e4
    c1, c2, w = pers(d)
    print(f'  market view, Transformer minus raw MLPs (sign of the day mean vs its average): {d.mean():+.2f} bp/day; corr15 {c1:+.3f}, block {c2:+.3f}')

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=float, default=None, help='seconds; stop and keep the progress when exceeded (rerun to continue)')
    a = ap.parse_args()
    t0 = time.time(); log = lambda s: print(f'[{time.time() - t0:5.0f}s] {s}', flush=True)
    deadline = t0 + a.budget if a.budget else None
    # ---------------- grids (cached)
    if os.path.exists(GRID_FILE):
        z = np.load(GRID_FILE, allow_pickle=False)
        dW, AW, dS, AS = pd.DatetimeIndex(z['dW']), z['AW'], pd.DatetimeIndex(z['dS']), z['AS']
        log(f'grids read from {GRID_FILE}')
    else:
        idx, pb, pe = research_inputs(); I = info_table(idx)
        days = day_arrays(I, pb, pe); log(f'walk-forward: {len(days)} days {days[0][0].date()}..{days[-1][0].date()}; {np.prod(SHAPE)} combinations per day')
        AW = grids(days, log, os.path.join(tempfile.gettempdir(), 'adapt_grid_wf.part.npz'), deadline); dW = pd.DatetimeIndex([d[0] for d in days])
        idx2, pb2, pe2 = static_inputs(); I2 = info_table(idx2)
        sdays = day_arrays(I2, pb2, pe2); log(f'rolling static: {len(sdays)} days {sdays[0][0].date()}..{sdays[-1][0].date()}')
        AS = grids(sdays, log, os.path.join(tempfile.gettempdir(), 'adapt_grid_static.part.npz'), deadline); dS = pd.DatetimeIndex([d[0] for d in sdays])
        np.savez_compressed(GRID_FILE, dW=dW.values.astype('datetime64[ns]'), AW=AW, dS=dS.values.astype('datetime64[ns]'), AS=AS)
        log(f'wrote {GRID_FILE}')
    # ---------------- checks: the v9 combination reproduces the official numbers
    ref = pd.read_pickle(os.path.join(CACHE, 'v9_daily.pkl'))['v9: all 11 changes'].AR
    ref.index = pd.DatetimeIndex(ref.index).astype('datetime64[ns]')
    mine = pd.Series(AW[:, K_DEF], dW)
    print(f'check 1: v9 combination vs v9_eval ({len(ref)} days): max |diff| {np.abs(mine.reindex(ref.index) - ref).max():.2e} bp; '
          f'mean {mine.mean():.4f} vs {ref.mean():.4f}')
    P = pd.read_csv(os.path.join(OUT, 'probabilities_v9_static.csv'), dtype={'stock_id': str}, parse_dates=['date'])
    P = P.merge(pd.read_pickle(os.path.join(CACHE, 'core9.pkl'))[['date', 'stock_id', 'target_r']].assign(stock_id=lambda x: x.stock_id.astype(str)), on=['date', 'stock_id'])
    P = P[P.target_r.notna()]; refS = ((2 * P.pred - 1) * P.target_r).groupby(P.date).mean() * 1e4
    mineS = pd.Series(AS[:, K_DEF], dS)
    st = (dS >= STATIC0) & (dS <= STATIC1)
    print(f'check 2: v9 combination vs out\\probabilities_v9_static.csv ({len(refS)} days): max |diff| {np.abs(mineS - refS.reindex(dS)).max():.2e} bp; '
          f'static {mineS[st].mean():.2f} (official 16.42), Oct {mineS[dS > STATIC1].mean():.2f} (official 21.00)')
    # ---------------- spliced history for the static check: walk-forward before 2026-06-01, rolling static after
    keep = dW < STATIC0
    dX = dW[keep].append(dS); AX = np.vstack([AW[keep], AS]); inS = dX >= STATIC0
    baseW = AW[:, K_DEF]; baseX = AX[:, K_DEF]
    # ---------------- schemes, N = 15
    rows, rows_s, chosen = [], [], {}
    rows.append(summary_row('v9 fixed (reference)', baseW, baseW, dW))
    for code, lab, free, near, ew in SCHEMES:
        ks = subset(free, near)
        ch = choose(AW, ks, 15, ew); arW = AW[np.arange(len(AW)), ch]
        rows.append({**summary_row(f'{code} {lab} [{len(ks)}]', arW, baseW, dW), **coef_stats(ch[15:])})
        chX = choose(AX, ks, 15, ew); arX = AX[np.arange(len(AX)), chX][inS]; bX = baseX[inS]; dd = dS
        s1 = (dd >= STATIC0) & (dd <= STATIC1)
        rows_s.append({'scheme': f'{code} {lab}', 'static': arX[s1].mean(), 'd_static': (arX - bX)[s1].mean(), 't_static': L.plain_t((arX - bX)[s1]),
                       'Oct': arX[~s1].mean(), 'd_Oct': (arX - bX)[~s1].mean(), 'moved': float((chX[inS] != K_DEF).mean())})
        chosen[code] = (ks, ew, chX)
    T = pd.DataFrame(rows)
    print('\n=== walk-forward 2022-01..2026-10 (Top50 bp/day), N = 15 trading days; d_ = adaptive minus fixed v9 (same days)')
    print(T[['scheme', 'val', 'hold', 'post', 'all', 'sd', 'd_val', 'd_hold', 'd_post', 'd_all', 't_all', 'nw_all']].round(2).to_string(index=False))
    print('\naverage coefficient chosen (days with a full window) and share of days that moved away from v9:')
    print(T[['scheme'] + NAMES + ['moved']].iloc[1:].round(3).to_string(index=False))
    print(f'\n=== rolling static splice (history before 6/1 from the walk-forward), N = 15; v9 fixed: static {baseX[inS][(dS >= STATIC0) & (dS <= STATIC1)].mean():.2f}, '
          f'Oct {baseX[inS][dS > STATIC1].mean():.2f}')
    print(pd.DataFrame(rows_s).round(2).to_string(index=False))
    # ---------------- pre-registered decision
    pr = T.iloc[6]
    ok = pr.d_val > 0 and pr.d_hold > 0 and pr.d_post > 0 and pr.nw_all >= 2
    print(f"\npre-registered rule, primary S6 N=15: d_val {pr.d_val:+.2f}, d_hold {pr.d_hold:+.2f}, d_post {pr.d_post:+.2f}, NW t {pr.nw_all:.2f} -> "
          + ('PASS: candidate for live' if ok else 'FAIL: live keeps the fixed v9 coefficients'))
    cands = [r.scheme for _, r in T.iloc[1:].iterrows() if r.scheme[:2] != 'S6' and r.d_val > 0 and r.d_hold > 0 and r.d_post > 0 and r.nw_all >= 2.73]
    print('secondary schemes passing (three periods > 0 and NW t >= 2.73): ' + (', '.join(cands) if cands else 'none'))
    # ---------------- window length (report only)
    print('\n=== window length N (report only, not used for picking): full-period difference to fixed v9, bp/day (NW t)')
    NS = [5, 10, 15, 20, 30, 60, 120, 250]; rr = []
    for code, lab, free, near, ew in [s for s in SCHEMES if s[0] in ('S1', 'S3', 'S6')]:
        ks = subset(free, near); r = {'scheme': f'{code} {lab}'}
        for N in NS:
            ch = choose(AW, ks, N); d = AW[np.arange(len(AW)), ch] - baseW
            d = d[250:]                                  # same days for every N (the first 250 days use v9 when N = 250)
            r[f'N{N}'] = f'{d.mean():+.2f} ({L.nw_t(d, 5):+.1f})'
        rr.append(r)
    print(pd.DataFrame(rr).to_string(index=False))
    print('  (days from 2023-01 on, so that every N has a full window)')
    # ---------------- persistence: does the last 15 days' advantage of a coefficient predict the next day's?
    print('\n=== persistence (walk-forward): one coefficient changed, the others at v9; D = daily AR difference to v9')
    print('    corr15 = corr(mean D over the previous 15 days, D of the day); block = corr of consecutive 15-day block means;')
    print('    mean D = average advantage over the whole period (what a fixed switch would give)')
    pp = []
    for i, k in enumerate(NAMES):
        for g in range(SHAPE[i]):
            if g == DEFPOS[i]:
                continue
            pos = DEFPOS.copy(); pos[i] = g; kk = int(np.ravel_multi_index(tuple(pos), SHAPE))
            D = AW[:, kk] - baseW
            tr = pd.Series(D).rolling(15).mean().shift(1).values
            m = ~np.isnan(tr)
            nb = len(D) // 15; bm = D[:nb * 15].reshape(nb, 15).mean(axis=1)
            pp.append({'coef': k, 'value': GRID[k][g], 'mean D': D.mean(), 't': L.plain_t(D), 'corr15': np.corrcoef(tr[m], D[m])[0, 1],
                       'block': np.corrcoef(bm[:-1], bm[1:])[0, 1], 'sd D': D.std(ddof=1)})
    PP = pd.DataFrame(pp); print(PP.round(3).to_string(index=False))
    print(f'  average corr15 {PP.corr15.mean():+.3f} (median {PP.corr15.median():+.3f}); average block corr {PP.block.mean():+.3f}; '
          f'1/sqrt(n) = {1 / np.sqrt(len(baseW)):.3f}')
    print('  same question with smoother daily signals than AR:')
    ic_persistence()
    # ---------------- hindsight reference: best fixed combination per period (not available in real time)
    print('\n=== hindsight (not usable live): the best fixed combination of each period, all 4375')
    pr = L.periods(dW)
    for p in ('val', 'hold', 'post'):
        mu = AW[pr[p]].mean(axis=0); kb = int(np.argmax(mu)); c = IDX[kb]
        oth = {q: AW[pr[q]][:, kb].mean() for q in ('val', 'hold', 'post') if q != p}
        print(f'  best in {p:4s}: ' + ', '.join(f'{k} {GRID[k][c[i]]}' for i, k in enumerate(NAMES)) +
              f' -> {p} {mu[kb]:.2f} (v9 {AW[pr[p]][:, K_DEF].mean():.2f}); same combination elsewhere: ' + ', '.join(f'{q} {v:.2f}' for q, v in oth.items()))
    log('done')

if __name__ == '__main__':
    main()

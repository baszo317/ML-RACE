r"""
build_panel.py -- RACE-STD v1 step 1 (research\PREREG_race_std.md sections 1.1, 1.2, 2.1, appendices A-C).

One panel, one row per (date, stock) of the 100 competition stocks, with
  labels / rule columns : top50, target_r (r), resid_r (e), y_resid, ex_today .. ex_prev2, absres_60
  U__*  111  Baszo v9 / v11 features (v9\rolling9.build_core, the user's code, unchanged)
  P__*   61  FinPred v2.3 (rewritten from race ver\framework_features_v2.3.md, appendix A)
  C__*  129  Chen V5, the part rebuildable from competition data + night session + calendar (appendix B)
  M__*   39  NTU M2 (its own create_features + enrich; market columns from the Top50, appendix C)

  python build_panel.py                  build cache\panel.pkl, cache\features.json, results\snapshot.json
  python build_panel.py --ic             same-day IC check on D days (results\leak_check_ic.csv; > 0.30 removed)
  python build_panel.py --leak 0 3       truncation test on test days 0..2 of 9 for P, C, M (results\leak_check_trunc.txt)
  python build_panel.py --leak-u 0 1     truncation test of U (rebuilds v9 build_core on truncated copies of data\)
"""
import os, sys, json, time, argparse, shutil, tempfile, platform
import numpy as np, pandas as pd
import rslib as L

COLS = {'O': 'etl:adj_open', 'H': 'etl:adj_high', 'L': 'etl:adj_low', 'C': 'etl:adj_close', 'V': 'price:成交股數',
        'AMT': 'price:成交金額', 'N': 'price:成交筆數', 'FH': 'foreign_investors_shareholding:全體外資及陸資持股比率',
        'FN': 'institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)',
        'TN': 'institutional_investors_trading_summary:投信買賣超股數',
        'DS': 'institutional_investors_trading_summary:自營商買賣超股數(自行買賣)',
        'DH': 'institutional_investors_trading_summary:自營商買賣超股數(避險)',
        'RSI': 'RSI', 'SMA': 'SMA', 'MACD': 'MACD', 'ATR': 'ATR', 'BBU': 'BBANDS_upper', 'BBM': 'BBANDS_mid', 'BBL': 'BBANDS_lower'}


# ------------------------------------------------------------------ raw data
def read_info(data_dir):
    with open(os.path.join(data_dir, 'info.json'), encoding='utf-8') as fh:
        info = json.load(fh)
    return [str(s) for s in info['Top100@Y']], [str(s) for s in info['Top50@Y']]


def read_market_long(data_dir, ids, until=None, blank_last=False):
    """the competition csv files, long format; rows whose prices are the -1 placeholder become missing.
    until / blank_last: truncation test (keep rows <= until; values of the day `until` set to missing)"""
    out = []
    for s in ids:
        df = pd.read_csv(os.path.join(data_dir, 'market', f'{s}.csv'), encoding='utf-8')
        df['date'] = pd.to_datetime(df['date']).astype('datetime64[ns]')
        num = [c for c in df.columns if c != 'date']
        for c in num:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        ph = (df[COLS['O']] == -1) & (df[COLS['C']] == -1)
        df.loc[ph, num] = np.nan
        if until is not None:
            df = df[df.date <= until].copy()
            if blank_last:
                df.loc[df.date == until, num] = np.nan
        df['stock_id'] = s
        out.append(df)
    return pd.concat(out, ignore_index=True)


def read_global(data_dir, until=None, blank_last=False):
    g = pd.read_csv(os.path.join(data_dir, 'global', 'global_data.csv'))
    g['date'] = pd.to_datetime(g['date']).astype('datetime64[ns]')
    for c in g.columns:
        if c != 'date':
            g[c] = pd.to_numeric(g[c], errors='coerce')
            g.loc[g[c] == -1, c] = np.nan
    if until is not None:
        g = g[g.date <= until].copy()
        if blank_last:                               # on day t only the U.S. close of the night before is known
            for c in ('^TWII', '^KS11', 'TX_OI'):
                g.loc[g.date == until, c] = np.nan
    return g.drop_duplicates('date', keep='last').sort_values('date').reset_index(drop=True)


def wide(long, dates, ids):
    W = {}
    for k, c in COLS.items():
        x = long.pivot(index='date', columns='stock_id', values=c) if c in long.columns else pd.DataFrame(index=dates, columns=ids)
        W[k] = x.reindex(index=dates, columns=ids).astype(float)
    return W


# ------------------------------------------------------------------ helpers (wide: rows = dates, columns = stocks)
def rpct(X):
    """cross-sectional percentile rank - 0.5 over the non-missing stocks (-0.5 .. 0.5)"""
    R = X.rank(axis=1, method='average')
    n = X.notna().sum(axis=1)
    return R.sub(1.0).div((n - 1).where(n > 1), axis=0) - 0.5


def rmean(X, k, mp=None):
    return X.rolling(k, min_periods=mp or max(2, k // 2)).mean()


def rstd(X, k, mp=None):
    return X.rolling(k, min_periods=mp or max(3, k // 2)).std()


def rslope(y, x, k, mp):
    """rolling OLS slope of y on x (both wide, same shape), common support only"""
    ok = y.notna() & x.notna()
    xm, ym = x.where(ok), y.where(ok)
    mx, my = rmean(xm, k, mp), rmean(ym, k, mp)
    cxy = rmean(xm * ym, k, mp) - mx * my
    vx = rmean(xm * xm, k, mp) - mx * mx
    return cxy / vx.where(vx > 1e-14)


def rcorr(a, b, k, mp):
    ok = a.notna() & b.notna()
    am, bm = a.where(ok), b.where(ok)
    ma, mb = rmean(am, k, mp), rmean(bm, k, mp)
    c = rmean(am * bm, k, mp) - ma * mb
    va = rmean(am * am, k, mp) - ma * ma
    vb = rmean(bm * bm, k, mp) - mb * mb
    return c / np.sqrt((va * vb).where((va > 1e-14) & (vb > 1e-14)))


def ew_stats(X, hl, minobs):
    """exponentially weighted mean and standard error of the mean (half-life hl observations), as of each row;
    se = sqrt(v / n_eff), v = weighted variance, n_eff = (sum w)^2 / sum w^2. Missing values do not update."""
    lam = 0.5 ** (1.0 / hl)
    A = X.to_numpy(float)
    T, N = A.shape
    S0, S1, S2, W2, cnt = (np.zeros(N) for _ in range(5))
    M = np.full((T, N), np.nan); SE = np.full((T, N), np.nan)
    for t in range(T):
        x = A[t]; ok = ~np.isnan(x)
        S0[ok] = lam * S0[ok] + 1.0; S1[ok] = lam * S1[ok] + x[ok]; S2[ok] = lam * S2[ok] + x[ok] ** 2
        W2[ok] = lam * lam * W2[ok] + 1.0; cnt[ok] += 1
        v = cnt >= minobs
        if v.any():
            m = S1[v] / S0[v]
            var = np.maximum(S2[v] / S0[v] - m * m, 0.0)
            neff = S0[v] ** 2 / W2[v]
            M[t, v] = m; SE[t, v] = np.sqrt(var / neff)
    return pd.DataFrame(M, X.index, X.columns), pd.DataFrame(SE, X.index, X.columns)


def bcast(s, like):
    """a per-date Series as a wide frame shaped like `like`"""
    return pd.DataFrame(np.repeat(s.reindex(like.index).to_numpy(float)[:, None], like.shape[1], axis=1), like.index, like.columns)


def stack(F, prefix):
    """dict name -> wide frame  ->  long frame indexed (date, stock_id) with prefix + name columns"""
    cols = {}
    for k, X in F.items():
        X = X.copy(); X.index.name = 'date'; X.columns.name = 'stock_id'
        try:
            cols[prefix + k] = X.stack(future_stack=True)
        except TypeError:
            cols[prefix + k] = X.stack(dropna=False)
    out = pd.DataFrame(cols)
    out.index.names = ['date', 'stock_id']
    return out


def streak(sign):
    """signed run length of a sign matrix (+1 / -1 / 0 / nan) along rows"""
    A = sign.to_numpy(float); T, N = A.shape
    out = np.full((T, N), np.nan); run = np.zeros(N)
    for t in range(T):
        x = A[t]
        same = (x == np.sign(run)) & (x != 0)
        run = np.where(np.isnan(x), run, np.where(x == 0, 0.0, np.where(same, run + x, x)))
        out[t] = np.where(np.isnan(x), np.nan, run)
    return pd.DataFrame(out, sign.index, sign.columns)


# ------------------------------------------------------------------ trading calendar
def calendar(dates, data_dir):
    """trading days: the panel dates, extended with the official calendar after the last panel date (for month-end and
    settlement-day facts, which are published in advance)"""
    days = pd.DatetimeIndex(dates)
    try:
        cal = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(os.path.join(data_dir, 'ref', 'calendar.parquet')).date)).astype('datetime64[ns]')
        closed = set()
        f = os.path.join(data_dir, 'ref', 'closures_manual.csv')
        if os.path.exists(f):
            closed = set(pd.to_datetime(pd.read_csv(f, dtype=str)['date'].dropna().str.strip()))
        later = [d for d in cal.sort_values() if d > days.max() and d not in closed]
        days = days.append(pd.DatetimeIndex(later))
    except Exception:
        extra = pd.bdate_range(days.max() + pd.Timedelta(days=1), periods=60)
        days = days.append(extra)
    return pd.DatetimeIndex(sorted(set(days))).astype('datetime64[ns]')


def settle_days(days):
    """TX settlement: third Wednesday of each month; if that is not a trading day, the next trading day"""
    out = []
    dset = days
    for per in pd.period_range(days.min(), days.max(), freq='M'):
        first = per.start_time
        wd = (2 - first.weekday()) % 7
        third = first + pd.Timedelta(days=wd + 14)
        nxt = dset[dset >= third]
        if len(nxt):
            out.append(nxt[0])
    return pd.DatetimeIndex(sorted(set(out)))


def calendar_feats(dates, data_dir):
    days = calendar(dates, data_dir)
    D = pd.DataFrame(index=days)
    s = pd.Series(days, index=days)
    prev = s.shift(1)
    D['dow'] = days.weekday.astype(float)
    D['month'] = days.month.astype(float)
    D['dom'] = days.day.astype(float)
    D['gap_days'] = (s - prev).dt.days.astype(float)
    nb = [len(pd.bdate_range(p + pd.Timedelta(days=1), d - pd.Timedelta(days=1))) if pd.notna(p) else 0 for p, d in zip(prev, days)]
    D['post_holiday'] = (np.asarray(nb) > 0).astype(float)
    mon = days.to_period('M')
    rk = s.groupby(mon).rank(method='first')
    rk_desc = s.groupby(mon).rank(method='first', ascending=False)
    D['tom_first2'] = (rk <= 2).astype(float).values
    D['is_month_first_bd'] = (rk == 1).astype(float).values
    D['is_month_last_bd'] = (rk_desc == 1).astype(float).values
    q = days.to_period('Q')
    D['is_quarter_last_bd'] = (s.groupby(q).rank(method='first', ascending=False) == 1).astype(float).values
    st = settle_days(days)
    D['settle_day'] = days.isin(st).astype(float)
    pos = np.arange(len(days))
    st_pos = np.searchsorted(days.values, st.values)
    nxt = np.searchsorted(st_pos, pos)                    # index of the first settlement position >= pos
    D['bdays_to_settle'] = np.where(nxt < len(st_pos), st_pos[np.minimum(nxt, len(st_pos) - 1)] - pos, np.nan).astype(float)
    return D.reindex(pd.DatetimeIndex(dates))


# ------------------------------------------------------------------ P: FinPred v2.3 (appendix A)
def feat_P(W, G, CAL, dates, ids):
    O, H, Lo, C, V, AMT = W['O'], W['H'], W['L'], W['C'], W['V'], W['AMT']
    r = C / O - 1
    e = r.sub(r.mean(axis=1), axis=0)
    gap = O / C.shift(1) - 1
    cc = C / C.shift(1) - 1
    A = {}                                                 # as of the close of day d (shifted by one day below)
    _, A['ew120_se'] = ew_stats(r, 120, 60)
    _, A['ew500_se'] = ew_stats(r, 500, 60)
    n = e.notna().cumsum()
    mean_all = e.expanding(min_periods=60).mean()
    sd_all = e.expanding(min_periods=60).std()
    A['res_bias_all'] = mean_all
    A['res_bias_t'] = mean_all / (sd_all / np.sqrt(n)).where(sd_all > 0)
    m500, se500 = ew_stats(e, 500, 60)
    A['res_ew500'] = m500
    mbar = m500.mean(axis=1)
    tau2 = (m500.var(axis=1) - (se500 ** 2).mean(axis=1)).clip(lower=0)
    shrink = bcast(tau2, m500) / (bcast(tau2, m500) + se500 ** 2)
    A['res_ew500_shrunk'] = bcast(mbar, m500) + shrink.where(shrink.notna(), 0.0) * m500.sub(mbar, axis=0)
    for k in (20, 60):
        A[f'res_intra_mean{k}'] = rmean(e, k)
        A[f'up_rate{k}'] = rmean((r > 0).astype(float).where(r.notna()), k)
        A[f'gap_mean{k}'] = rmean(gap, k)
        A[f'intra_std{k}'] = rstd(r, k)
    lc = np.log(C)
    for k in (5, 20, 60, 250):
        A[f'mom{k}'] = lc - lc.shift(k)
    A['hi_dist250'] = lc - np.log(C.rolling(250, min_periods=120).max())
    A['macd_l1'] = W['MACD'] / C
    A['sma_dev_l1'] = C / W['SMA'] - 1
    A['bb_pos_l1'] = (C - W['BBM']) / (W['BBU'] - W['BBL']).where((W['BBU'] - W['BBL']).abs() > 0)
    A['bb_width_l1'] = (W['BBU'] - W['BBL']) / W['BBM']
    A['atr_l1'] = W['ATR'] / C
    A['ushadow_l1'] = (H - np.maximum(O, C)) / C
    la = np.log(AMT.where(AMT > 0))
    A['amount_z20'] = la - rmean(la, 20)
    A['vol20'] = rstd(cc, 20); A['vol60'] = rstd(cc, 60)
    mkt_r = bcast(r.mean(axis=1), r)
    A['beta_intra'] = rslope(r, mkt_r, 250, 120)
    A['beta_gap_on_intra'] = rslope(r, gap, 250, 120)
    spx = G['^GSPC'].pct_change(fill_method=None)            # row d: the U.S. session before day d's open
    bgu = rslope(gap, bcast(spx, gap), 250, 120)
    A['beta_gap_us'] = bgu
    vol = V.where(V > 0)
    flows = {'foreign': W['FN'] / vol, 'trust': W['TN'] / vol, 'dealer': W['DS'] / vol, 'hedge': W['DH'] / vol}
    A['foreign_flow_l1'] = flows['foreign']; A['foreign_flow20'] = rmean(flows['foreign'], 20)
    A['trust_flow_l1'] = flows['trust']; A['trust_flow5'] = rmean(flows['trust'], 5); A['trust_flow20'] = rmean(flows['trust'], 20)
    A['dealer_flow_l1'] = flows['dealer']; A['dealer_flow5'] = rmean(flows['dealer'], 5); A['dealer_flow20'] = rmean(flows['dealer'], 20)
    A['hedge_flow_l1'] = flows['hedge']; A['hedge_flow20'] = rmean(flows['hedge'], 20)
    fh = W['FH'].ffill()
    A['fhold_l1'] = fh; A['fhold_chg1'] = fh - fh.shift(1); A['fhold_chg5'] = fh - fh.shift(5)
    lag = {k: X.shift(1) for k, X in A.items()}
    # same-weekday EW of e over earlier occurrences only (already lagged)
    dow = pd.Series(pd.DatetimeIndex(e.index).weekday, index=e.index)
    sd = pd.DataFrame(np.nan, e.index, e.columns)
    for wdy in range(5):
        idx = dow.index[dow.values == wdy]
        if len(idx):
            m, _ = ew_stats(e.loc[idx], 100, 20)
            sd.loc[idx] = m.shift(1).to_numpy()
    lag['stock_dow_res_ew'] = sd
    F = {('z_' + k): rpct(X) for k, X in lag.items()}
    F['raw_beta_gap_us'] = lag['beta_gap_us']
    F['z_exp_gap'] = rpct(lag['beta_gap_us'] * bcast(spx.shift(1), gap))
    mkt = pd.DataFrame({'intra': r.mean(axis=1), 'gap': gap.mean(axis=1), 'ret': cc.mean(axis=1)})
    day = pd.DataFrame(index=r.index)
    day['mkt_intra_l1'] = mkt.intra.shift(1); day['mkt_gap_l1'] = mkt.gap.shift(1); day['mkt_ret_l1'] = mkt.ret.shift(1)
    day['mkt_ret5'] = mkt.ret.rolling(5, min_periods=3).sum().shift(1)
    day['mkt_ret20'] = mkt.ret.rolling(20, min_periods=10).sum().shift(1)
    day['spx_ret'] = spx.shift(1)
    day['ndx_ret'] = G['^IXIC'].pct_change(fill_method=None).shift(1)
    day['vix'] = G['^VIX'].shift(1)
    day['vix_chg'] = G['^VIX'].pct_change(fill_method=None).shift(1)
    day['vix_z20'] = ((G['^VIX'] - G['^VIX'].rolling(20, min_periods=10).mean()) / G['^VIX'].rolling(20, min_periods=10).std()).shift(1)
    for k in ('dow', 'post_holiday', 'tom_first2', 'settle_day'):
        day[k] = CAL[k]
    for k in day.columns:
        F[k] = bcast(day[k], r)
    return stack(F, 'P__')


# ------------------------------------------------------------------ C: Chen V5 rebuildable part (appendix B)
def feat_C(W, G, CAL, NIGHT, dates, ids, top50):
    O, H, Lo, C, V, AMT, N = W['O'], W['H'], W['L'], W['C'], W['V'], W['AMT'], W['N']
    r = C / O - 1
    cc = C / C.shift(1) - 1
    gap = O / C.shift(1) - 1
    hl = (H - Lo).where((H - Lo) > 0)
    A = {}
    A['kmid'] = (C - O) / O; A['klen'] = (H - Lo) / O; A['kup'] = (H - np.maximum(O, C)) / O
    A['klow'] = (np.minimum(O, C) - Lo) / O; A['ksft'] = (2 * C - H - Lo) / O; A['kmid2'] = (C - O) / hl
    A['ret1'] = cc
    for k in (5, 10, 20, 60):
        A[f'roc_{k}'] = C / C.shift(k) - 1
        A[f'ma_bias_{k}'] = C / rmean(C, k, k) - 1
        lo, hi = Lo.rolling(k, min_periods=k).min(), H.rolling(k, min_periods=k).max()
        A[f'rsv_{k}'] = (C - lo) / (hi - lo).where((hi - lo) > 0)
    B13 = ['ret1'] + [f'{p}_{k}' for p in ('roc', 'ma_bias', 'rsv') for k in (5, 10, 20, 60)]
    for k in (5, 20, 60):
        A[f'std_{k}'] = rstd(cc, k)
        A[f'range_{k}'] = rmean((H - Lo) / C, k)
    A['std_ratio_5_20'] = A['std_5'] / A['std_20'].where(A['std_20'] > 0)
    A['atr_rel'] = W['ATR'] / C
    A['bb_width'] = (W['BBU'] - W['BBL']) / W['BBM']
    lv = np.log(V.where(V > 0))
    A['vchg_1'] = lv - lv.shift(1)
    for k in (5, 20, 60):
        A[f'vrel_{k}'] = V / rmean(V, k)
    A['amt_rel20'] = AMT / rmean(AMT, 20)
    size = AMT / N.where(N > 0)
    A['size_rel20'] = size / rmean(size, 20)
    A['corr_rv_20'] = rcorr(cc, lv, 20, 10)
    vol = V.where(V > 0)
    nets = {'f_net': W['FN'], 'trust_net': W['TN'], 'dself_net': W['DS'], 'dhedge_net': W['DH']}
    for k, x in nets.items():
        A[k] = x / vol
        A[f'{k}_5'] = x.rolling(5, min_periods=3).sum() / vol.rolling(5, min_periods=3).sum()
        A[f'{k}_20'] = x.rolling(20, min_periods=10).sum() / vol.rolling(20, min_periods=10).sum()
    inst = W['FN'] + W['TN'] + W['DS'] + W['DH']
    A['inst'] = inst / vol
    A['inst_5'] = inst.rolling(5, min_periods=3).sum() / vol.rolling(5, min_periods=3).sum()
    A['f_streak'] = streak(np.sign(W['FN']))
    fh = W['FH'].ffill()
    A['fhold'] = fh; A['fhold_d1'] = fh - fh.shift(1); A['fhold_d5'] = fh - fh.shift(5); A['fhold_d20'] = fh - fh.shift(20)
    A['rsi'] = W['RSI']; A['rsi_chg5'] = W['RSI'] - W['RSI'].shift(5)
    A['macd_rel'] = W['MACD'] / C; A['sma_bias'] = C / W['SMA'] - 1
    A['bb_pctb'] = (C - W['BBL']) / (W['BBU'] - W['BBL']).where((W['BBU'] - W['BBL']).abs() > 0)
    for k in (5, 20, 60):
        A[f'intra_mean_{k}'] = rmean(r, k)
    A['up_rate_20'] = rmean((r > 0).astype(float).where(r.notna()), 20)
    A['intra_std_20'] = rstd(r, 20)
    A['intra_sharpe_20'] = A['intra_mean_20'] / A['intra_std_20'].where(A['intra_std_20'] > 0)
    A['gap'] = gap
    A['gap_mean_5'] = rmean(gap, 5); A['gap_mean_20'] = rmean(gap, 20)
    A['gap_intra_corr_60'] = rcorr(gap, r, 60, 30)
    lag = {k: X.shift(1) for k, X in A.items()}
    F = dict(lag)
    for k in B13:                                                      # H: ranks of B in the 100
        F['rank_' + k] = rpct(lag[k])
    t50 = [s for s in ids if s in set(top50)]
    day = pd.DataFrame(index=r.index)
    day['mkt_kmid'] = lag['kmid'][t50].mean(axis=1)
    day['mkt_ret1'] = lag['ret1'][t50].mean(axis=1)
    day['mkt_up_ratio'] = (lag['kmid'][t50] > 0).astype(float).where(lag['kmid'][t50].notna()).mean(axis=1)
    day['mkt_f_net'] = lag['f_net'][t50].mean(axis=1)
    F['kmid_vs_mkt'] = lag['kmid'].sub(day['mkt_kmid'], axis=0)
    F['ret1_vs_mkt'] = lag['ret1'].sub(day['mkt_ret1'], axis=0)
    # G (global row t-1, Chen's lag 1)
    g = pd.DataFrame(index=r.index)
    for nm, col in (('ixic', '^IXIC'), ('gspc', '^GSPC'), ('twii', '^TWII'), ('ks11', '^KS11')):
        x = G[col]; rr = x.pct_change(fill_method=None)
        g[f'{nm}_ret1'] = rr; g[f'{nm}_ret5'] = x.pct_change(5, fill_method=None); g[f'{nm}_ret20'] = x.pct_change(20, fill_method=None)
        g[f'{nm}_ma20_bias'] = x / x.rolling(20, min_periods=10).mean() - 1
        g[f'{nm}_std20'] = rr.rolling(20, min_periods=10).std()
    g['vix'] = G['^VIX']; g['vix_chg1'] = G['^VIX'].pct_change(fill_method=None)
    g['vix_rel20'] = G['^VIX'] / G['^VIX'].rolling(20, min_periods=10).mean() - 1
    oi = G['TX_OI']
    g['txoi_chg1'] = (oi - oi.shift(1)) / 1e4; g['txoi_chg5'] = (oi - oi.shift(5)) / 1e4
    g['txoi_z20'] = (oi - oi.rolling(20, min_periods=10).mean()) / oi.rolling(20, min_periods=10).std()
    g['us_minus_tw_ret1'] = g['gspc_ret1'] - g['twii_ret1']
    g = g.shift(1)
    s = pd.Series(r.index, index=r.index)
    g['staleness_days'] = (s - s.shift(1)).dt.days.astype(float)
    for k in g.columns:
        day[k] = g[k]
    for k in ('dow', 'month', 'dom', 'gap_days', 'is_month_last_bd'):              # I (lag 0)
        day['I_' + k] = CAL[k]
    day['is_settle_day'] = CAL['settle_day']; day['bdays_to_settle'] = CAL['bdays_to_settle']   # M_cal (lag 0)
    day['M_is_month_last_bd'] = CAL['is_month_last_bd']; day['is_month_first_bd'] = CAL['is_month_first_bd']
    day['is_quarter_last_bd'] = CAL['is_quarter_last_bd']
    day['tx_night_ret'] = NIGHT['night_ret']; day['tx_night_absret'] = NIGHT['night_ret'].abs()   # K_pre night (lag 0)
    day['tx_night_range'] = NIGHT['night_range']
    for k in day.columns:
        F[k] = bcast(day[k], r)
    return stack(F, 'C__'), list(day.columns)


# ------------------------------------------------------------------ K: Chen's crawled groups (PREREG section 6, appendix D)
EXT = os.path.join(L.STD, 'ext')
ADR = {'2330': ('TSM', 5.0), '2303': ('UMC', 5.0), '3711': ('ASX', 2.0), '2412': ('CHT', 10.0)}
ETF_RULES = [(('半導體',), 'SOXX'), (('金融', '保險'), 'XLF'), (('電腦', '電子零組件', '光電', '其他電子', '電子通路', '資訊服務', '通信網路'), 'XLK'),
             (('水泥', '塑膠', '鋼鐵', '化學', '玻璃', '造紙'), 'XLB'), (('航運',), 'IYT'), (('食品',), 'XLP'),
             (('汽車', '貿易百貨', '紡織', '橡膠', '居家生活', '運動休閒', '觀光'), 'XLY'), (('電機', '電器電纜', '建材營造'), 'XLI'),
             (('油電燃氣',), 'XLE'), (('生技醫療',), 'XLV')]
CARRIERS = {'2412': 'XLC', '3045': 'XLC', '4904': 'XLC'}
K_STOCK = ['adr_ret_1', 'adr_prem', 'adr_prem_chg_1', 'adr_prem_z20', 'sector_ret_1', 'sector_ret_5', 'sector_rel_1',
           'margin_chg_vol', 'margin_util', 'margin_chg5', 'short_chg_vol', 'short_margin_ratio', 'short_chg5', 'offset_vol',
           'sbl_sell_vol', 'sbl_bal_v20', 'sbl_chg_vol', 'sbl_chg5_v20', 'daytrade_ratio', 'daytrade_ratio_rel20', 'daytrade_imb',
           'rank_margin_chg_vol', 'rank_sbl_sell_vol', 'rank_daytrade_ratio']
K_DAY = ['n225_gap', 'ks11_gap', 'asia_gap_mean', 'sox_ret_1', 'sox_ret_5', 'sox_rel_ixic_1', 'tsm_ret_1', 'K_adr_prem',
         'K_adr_prem_chg_1', 'K_adr_prem_z20', 'twd_ret_1', 'twd_ret_5', 'tnx_chg_1', 'tnx_chg_5', 'dxy_ret_1', 'dxy_ret_5']


def ext_files():
    out = {}
    if os.path.isdir(EXT):
        for dp, _, fs in os.walk(EXT):
            for f in sorted(fs):
                if f.endswith('.csv'):
                    p = os.path.join(dp, f)
                    out['ext/' + os.path.relpath(p, EXT).replace(os.sep, '/')] = L.sha256(p)
    return out


def yahoo_series(sym, until=None):
    f = os.path.join(EXT, 'yahoo', sym.replace('^', 'IDX_').replace('=', '_') + '.csv')
    if not os.path.exists(f):
        return None
    y = pd.read_csv(f)
    y['date'] = pd.to_datetime(y['date']).astype('datetime64[ns]')
    y = y.dropna(subset=['close']).drop_duplicates('date', keep='last').set_index('date').sort_index()
    if until is not None:
        y = y[y.index <= until]
    return y


def asof_before(s, dates):
    """value of the last observation of s dated strictly before each date (s: Series on its own dates)"""
    if s is None or len(s) == 0:
        return pd.Series(np.nan, index=dates)
    s = s.sort_index()
    i = np.searchsorted(s.index.values, pd.DatetimeIndex(dates).values, side='left') - 1
    v = np.where(i >= 0, s.to_numpy(float)[np.maximum(i, 0)], np.nan)
    return pd.Series(v, index=dates)


def asof_on_or_before(s, dates):
    s = s.sort_index()
    i = np.searchsorted(s.index.values, pd.DatetimeIndex(dates).values, side='right') - 1
    return pd.Series(np.where(i >= 0, s.to_numpy(float)[np.maximum(i, 0)], np.nan), index=dates)


def adr_premium(adr, tw, fx, ratio):
    """premium on the ADR's own (U.S.) dates: ADR close x USD/TWD / (ratio x Taiwan close of the same or earlier day) - 1"""
    if adr is None or tw is None or fx is None:
        return None
    d = adr.index
    p = adr['close'] * asof_on_or_before(fx['close'], d).values / (ratio * asof_on_or_before(tw['close'], d).values) - 1
    p = p.replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame({'prem': p, 'chg_1': p - p.shift(1),
                         'z20': (p - p.rolling(20, min_periods=10).mean()) / p.rolling(20, min_periods=10).std()})


def us_ret(y, k):
    return None if y is None else y['close'] / y['close'].shift(k) - 1


def industry_etf(ids, top_group, log=print):
    """FinMind TaiwanStockInfo can list several categories per stock (e.g. 2330: 半導體業 and 電子工業); the first rule of
    ETF_RULES matched by any of them wins; no match -> the user's 4 groups (0 fin, 1 semi, 2 electronics, 3 other)"""
    f = os.path.join(EXT, 'finmind', 'TaiwanStockInfo.csv')
    cats = {}
    if os.path.exists(f):
        I = pd.read_csv(f, dtype={'stock_id': str})
        for sid, g in I.groupby('stock_id'):
            cats[sid] = [c for c in g['industry_category'].astype(str).unique() if c and c != 'nan' and 'ETF' not in c]
    out, how = {}, {}
    for s in ids:
        if s in CARRIERS:
            out[s] = CARRIERS[s]; how[s] = 'carrier'; continue
        cs = cats.get(s, [])
        etf = next((e for keys, e in ETF_RULES if any(k in c for c in cs for k in keys)), None)
        if etf is None:
            etf = {0: 'XLF', 1: 'SOXX', 2: 'XLK'}.get(top_group.get(s), 'SPY'); how[s] = 'group'
        else:
            how[s] = 'category'
        out[s] = etf
    log('sector ETF map: ' + ', '.join(f'{e} {sum(1 for v in out.values() if v == e)}' for e in sorted(set(out.values())))
        + f'; by category {sum(1 for v in how.values() if v == "category")}, by group {sum(1 for v in how.values() if v == "group")}')
    return out


def finmind_wide(ds, ids, dates, cols, until=None):
    W = {c: pd.DataFrame(np.nan, index=dates, columns=ids) for c in cols}
    for s in ids:
        f = os.path.join(EXT, 'finmind', ds, f'{s}.csv')
        if not os.path.exists(f):
            continue
        x = pd.read_csv(f)
        if not len(x) or 'date' not in x.columns:
            continue
        x['date'] = pd.to_datetime(x['date']).astype('datetime64[ns]')
        if until is not None:
            x = x[x.date < until]                          # published after the close: never the day itself
        x = x.drop_duplicates('date', keep='last').set_index('date')
        for c in cols:
            if c in x.columns:
                W[c][s] = pd.to_numeric(x[c], errors='coerce').reindex(dates)
    return W


def unit_factor(ratio_median, name, log, expect_lo=0.005):
    """FinMind unit check (PREREG appendix D): if the typical ratio to shares traded is tiny, the data are in lots"""
    if not np.isfinite(ratio_median):
        log(f'unit check {name}: no data'); return 1.0
    f = 1000.0 if ratio_median < expect_lo else 1.0
    log(f'unit check {name}: median ratio to shares traded {ratio_median:.4g} -> factor {f:g}')
    return f


def feat_K(W, dates, ids, top_group, until=None, log=print):
    r_like = W['C']
    F, day = {}, pd.DataFrame(index=dates)
    ys = {k: yahoo_series(k, until) for k in ('TSM', 'UMC', 'ASX', 'CHT', '2330.TW', '2303.TW', '3711.TW', '2412.TW', '^SOX',
                                               '^N225', '^KS11', '^IXIC', 'TWD=X', '^TNX', 'DX-Y.NYB', 'SPY', 'SOXX', 'XLK',
                                               'XLF', 'XLB', 'XLI', 'XLP', 'XLY', 'XLE', 'XLV', 'XLC', 'IYT')}
    missing = [k for k, v in ys.items() if v is None]
    if missing:
        log(f'K: Yahoo files missing -> those features are missing values: {missing}')
    for nm, sym in (('n225_gap', '^N225'), ('ks11_gap', '^KS11')):
        y = ys[sym]
        if y is None:
            day[nm] = np.nan; continue
        g = (y['open'] / y['close'].shift(1) - 1).where(y['open'] > 0)
        if until is not None:
            g = g[g.index <= until]
        day[nm] = g.reindex(dates).values                  # the opening of day t itself (08:00 Taipei)
    day['asia_gap_mean'] = day[['n225_gap', 'ks11_gap']].mean(axis=1)
    sox1, sox5, ixic1 = us_ret(ys['^SOX'], 1), us_ret(ys['^SOX'], 5), us_ret(ys['^IXIC'], 1)
    day['sox_ret_1'] = asof_before(sox1, dates).values
    day['sox_ret_5'] = asof_before(sox5, dates).values
    day['sox_rel_ixic_1'] = (asof_before(sox1, dates) - asof_before(ixic1, dates)).values
    day['tsm_ret_1'] = asof_before(us_ret(ys['TSM'], 1), dates).values
    prem = {s: adr_premium(ys[a], ys[s + '.TW'], ys['TWD=X'], ratio) for s, (a, ratio) in ADR.items()}
    pt = prem['2330']
    for k, c in (('K_adr_prem', 'prem'), ('K_adr_prem_chg_1', 'chg_1'), ('K_adr_prem_z20', 'z20')):
        day[k] = asof_before(pt[c], dates).values if pt is not None else np.nan
    for nm, sym, kind in (('twd', 'TWD=X', 'ret'), ('tnx', '^TNX', 'chg'), ('dxy', 'DX-Y.NYB', 'ret')):
        y = ys[sym]
        for k in (1, 5):
            col = f'{nm}_{kind}_{k}'
            if y is None:
                day[col] = np.nan
            elif kind == 'ret':
                day[col] = asof_before(y['close'] / y['close'].shift(k) - 1, dates).values
            else:
                day[col] = asof_before(y['close'] - y['close'].shift(k), dates).values
    for k in K_DAY:
        F[k] = bcast(day[k], r_like)
    # K_pre_stk
    nan = pd.DataFrame(np.nan, index=dates, columns=ids)
    for k in ('adr_ret_1', 'adr_prem', 'adr_prem_chg_1', 'adr_prem_z20'):
        F[k] = nan.copy()
    for s, (a, ratio) in ADR.items():
        if s not in ids:
            continue
        F['adr_ret_1'][s] = asof_before(us_ret(ys[a], 1), dates).values
        if prem[s] is not None:
            F['adr_prem'][s] = asof_before(prem[s]['prem'], dates).values
            F['adr_prem_chg_1'][s] = asof_before(prem[s]['chg_1'], dates).values
            F['adr_prem_z20'][s] = asof_before(prem[s]['z20'], dates).values
    emap = industry_etf(ids, top_group, log)
    spy1 = asof_before(us_ret(ys['SPY'], 1), dates)
    for k in ('sector_ret_1', 'sector_ret_5', 'sector_rel_1'):
        F[k] = nan.copy()
    for s in ids:
        y = ys.get(emap[s])
        if y is None:
            continue
        r1 = asof_before(us_ret(y, 1), dates)
        F['sector_ret_1'][s] = r1.values
        F['sector_ret_5'][s] = asof_before(us_ret(y, 5), dates).values
        F['sector_rel_1'][s] = (r1 - spy1).values
    # T1_chip (FinMind, as of the close of d, then shifted one day)
    V = W['V'].where(W['V'] > 0)
    M = finmind_wide('TaiwanStockMarginPurchaseShortSale', ids, dates, ['MarginPurchaseTodayBalance', 'MarginPurchaseYesterdayBalance',
                     'MarginPurchaseLimit', 'ShortSaleTodayBalance', 'ShortSaleYesterdayBalance', 'OffsetLoanAndShort'], until)
    S = finmind_wide('TaiwanDailyShortSaleBalances', ids, dates, ['SBLShortSalesShortSales', 'SBLShortSalesCurrentDayBalance',
                     'SBLShortSalesPreviousDayBalance', 'MarginShortSalesCurrentDayBalance'], until)
    D = finmind_wide('TaiwanStockDayTrading', ids, dates, ['Volume', 'BuyAmount', 'SellAmount'], until)
    mb, sb = M['MarginPurchaseTodayBalance'], M['ShortSaleTodayBalance']
    um = unit_factor(float(np.nanmedian((mb / V).to_numpy())), 'margin balance', log)
    us_ = unit_factor(float(np.nanmedian((S['SBLShortSalesCurrentDayBalance'] / V.rolling(20, min_periods=10).mean()).to_numpy())), 'SBL balance', log)
    ud = unit_factor(float(np.nanmedian((D['Volume'] / V).to_numpy())), 'day-trade volume', log)
    v5 = V.rolling(5, min_periods=3).sum(); v20 = V.rolling(20, min_periods=10).mean()
    A = {}
    A['margin_chg_vol'] = (mb - M['MarginPurchaseYesterdayBalance']) * um / V
    A['margin_util'] = mb / M['MarginPurchaseLimit'].where(M['MarginPurchaseLimit'] > 0)
    A['margin_chg5'] = (mb - mb.shift(5)) * um / v5
    A['short_chg_vol'] = (sb - M['ShortSaleYesterdayBalance']) * um / V
    A['short_margin_ratio'] = sb / mb.where(mb > 0)
    A['short_chg5'] = (sb - sb.shift(5)) * um / v5
    A['offset_vol'] = M['OffsetLoanAndShort'] * um / V
    sbl = S['SBLShortSalesCurrentDayBalance']
    A['sbl_sell_vol'] = S['SBLShortSalesShortSales'] * us_ / V
    A['sbl_bal_v20'] = sbl * us_ / v20
    A['sbl_chg_vol'] = (sbl - S['SBLShortSalesPreviousDayBalance']) * us_ / V
    A['sbl_chg5_v20'] = (sbl - sbl.shift(5)) * us_ / v20
    dr = D['Volume'] * ud / V
    A['daytrade_ratio'] = dr
    A['daytrade_ratio_rel20'] = dr / dr.rolling(20, min_periods=10).mean()
    A['daytrade_imb'] = (D['BuyAmount'] - D['SellAmount']) / (D['BuyAmount'] + D['SellAmount']).where((D['BuyAmount'] + D['SellAmount']) > 0)
    for k, X in A.items():
        F[k] = X.replace([np.inf, -np.inf], np.nan).shift(1)
    F['rank_margin_chg_vol'] = rpct(F['margin_chg_vol'])
    F['rank_sbl_sell_vol'] = rpct(F['sbl_sell_vol'])
    F['rank_daytrade_ratio'] = rpct(F['daytrade_ratio'])
    cov = {k: float(F[k].notna().to_numpy().mean()) for k in K_STOCK + K_DAY}
    log('K coverage (share of non-missing rows): ' + ', '.join(f'{k} {v:.2f}' for k, v in cov.items()))
    return stack({k: F[k] for k in K_STOCK + K_DAY}, 'C__'), list(K_DAY), cov


# ------------------------------------------------------------------ M: NTU M2 (appendix C)
def feat_M(long, gdf, ids, top50):
    root = L.M2ROOT
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.features.base_features import create_features
    from src.features.market import enrich
    from src.features.builder import KEEP
    df = long.copy()
    meta, base = create_features(df, gdf)
    t50 = set(top50)
    meta50, base50 = create_features(df[df.stock_id.isin(t50)].reset_index(drop=True), gdf)
    mk = pd.DataFrame({'date': meta50.date.values, 'b': base50.breadth_p1.values, 'm': base50.market_avg_ret_p1.values,
                       'd': base50.market_dispersion_p1.values}).groupby('date').first()
    base = base.copy()
    base['breadth_p1'] = meta.date.map(mk.b).astype('float32').values
    base['market_avg_ret_p1'] = meta.date.map(mk.m).astype('float32').values
    base['market_dispersion_p1'] = meta.date.map(mk.d).astype('float32').values
    base['stock_minus_market_p1'] = (base['intraday_p1'] - base['market_avg_ret_p1']).astype('float32')
    f = enrich(meta, base)[KEEP].copy()
    f.columns = ['M__' + c for c in f.columns]
    f.index = pd.MultiIndex.from_arrays([pd.to_datetime(meta.date).astype('datetime64[ns]'), meta.stock_id.astype(str)],
                                        names=['date', 'stock_id'])
    return f.astype(float)


# ------------------------------------------------------------------ U: the user's panel
def user_panel(data_dir):
    v9 = os.path.join(L.ROOT, 'v9')
    if v9 not in sys.path:
        sys.path.insert(0, v9)
    import rolling9 as R9
    P = R9.build_core(data_dir)
    fs = R9.feature_sets()
    U = fs['B9+XR+NEW+XRN+SIG']
    B, NIGHT = fs['B'], ['night_ret', 'night_range']
    nas = [f for f in B if f not in U]
    allU = [f for f in fs['B9+SIG'] if f not in ('absr_60', 'absres_60')]          # B9
    xr = [f for f in fs['B9+XR+SIG'] if f not in fs['B9+SIG']]
    new = [f for f in fs['B9+NEW+SIG'] if f not in fs['B9+SIG']]
    xrn = [f for f in U if f not in set(allU) | set(xr) | set(new) | {'absr_60', 'absres_60'}]
    order = allU + nas + NIGHT + ['absr_60', 'absres_60'] + xr + new + xrn
    sub = {'UB': B, 'UB9S': fs['B9+SIG'], 'UXR9': fs['B9+XR+SIG'], 'UBN9': fs['B9+NEW+SIG'], 'UXRN9': fs['B9+XR+NEW+XRN+SIG']}
    return P, order, sub


def tx_night(P):
    n = P.groupby('date')[['night_ret', 'night_range']].first()
    n.index = pd.DatetimeIndex(n.index).astype('datetime64[ns]')
    return n


# ------------------------------------------------------------------ assemble
def build_all(data_dir, until=None, blank_last=False, need_u=True, Uprev=None, log=None):
    ids, top50 = read_info(data_dir)
    if need_u:
        P, uorder, usub = user_panel(data_dir)
    else:
        P, uorder, usub = Uprev
    P = P.copy()
    P['date'] = pd.to_datetime(P.date).astype('datetime64[ns]'); P['stock_id'] = P.stock_id.astype(str)
    if until is not None:
        P = P[P.date <= until].copy()
    dates = pd.DatetimeIndex(sorted(P.date.unique())).astype('datetime64[ns]')
    long = read_market_long(data_dir, ids, until, blank_last)
    gdf = read_global(data_dir, until, blank_last)
    W = wide(long, dates, ids)
    G = gdf.set_index('date').reindex(dates).ffill()
    CAL = calendar_feats(dates, data_dir)
    NIGHT = tx_night(P).reindex(dates)
    FP = feat_P(W, G, CAL, dates, ids)
    FC, cday = feat_C(W, G, CAL, NIGHT, dates, ids, top50)
    grp = P.groupby('stock_id')['group'].last().to_dict()
    FK, kday, cov = feat_K(W, dates, ids, grp, until, log=(print if log is None else log))
    FC = FC.join(FK, how='left')
    cday = cday + kday
    FM = feat_M(long, gdf, ids, top50)
    return P, uorder, usub, FP, FC, cday, FM, ids, top50


def assemble(P, uorder, usub, FP, FC, cday, FM):
    keep = ['date', 'stock_id', 'top50', 'has_label', 'target_r', 'resid_r', 'y_resid', 'absres_60', 'group'] + list(L.EX4)
    meta = P[keep].copy().reset_index(drop=True)
    meta['date'] = pd.to_datetime(meta.date).astype('datetime64[ns]'); meta['stock_id'] = meta.stock_id.astype(str)
    idx = pd.MultiIndex.from_arrays([meta.date, meta.stock_id], names=['date', 'stock_id'])
    blocks = [P[uorder].to_numpy(np.float32)]
    names = ['U__' + c for c in uorder]
    for Fx in (FP, FC, FM):
        blocks.append(Fx.reindex(idx).to_numpy(np.float32)); names += list(Fx.columns)
    X = pd.DataFrame(np.hstack(blocks), columns=names)
    out = pd.concat([meta, X], axis=1).sort_values(['date', 'stock_id']).reset_index(drop=True)
    feats = {'U': ['U__' + c for c in uorder], 'P': list(FP.columns), 'C': list(FC.columns), 'M': list(FM.columns)}
    for k, v in usub.items():
        feats[k] = ['U__' + c for c in v]
    feats['C_DAY'] = ['C__' + c for c in cday]
    return out, feats


def snapshot(data_dir):
    files = {}
    for sub in ('market', 'global', 'ref'):
        d = os.path.join(data_dir, sub)
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if os.path.isfile(p):
                files[f'data/{sub}/{f}'] = L.sha256(p)
    files['data/info.json'] = L.sha256(os.path.join(data_dir, 'info.json'))
    if os.path.exists(L.TX_CSV):
        files['v7/data/tx_daily.csv'] = L.sha256(L.TX_CSV)
    return files


def versions():
    v = {'python': platform.python_version(), 'platform': platform.platform(), 'numpy': np.__version__, 'pandas': pd.__version__}
    for m in ('lightgbm', 'sklearn', 'scipy'):
        try:
            v[m] = __import__(m).__version__
        except Exception as e:
            v[m] = f'missing ({e.__class__.__name__})'
    return v


def ic_check(out, feats, log):
    """same-day rank IC of every feature with r, D days only (PREREG 1.2 b): stock-level features = mean over days of the
    within-day Spearman correlation (100 stocks); day-level features = time-series Spearman with the Top50 mean r"""
    a, b = L.PER['D']
    X = out[(out.date >= a) & (out.date < b) & out.target_r.notna()].reset_index(drop=True)
    allf = [c for g in L.GROUPS for c in feats[g] if c in X.columns]
    nun = X.groupby('date')[allf].nunique()
    daylevel = [c for c in allf if (nun[c] <= 1).mean() > 0.9]
    stock = [c for c in allf if c not in set(daylevel)]
    rows = []
    d = X['date']
    rr = X.target_r.groupby(d).rank()
    rc = (rr - rr.groupby(d).transform('mean')).astype('float32')
    for i in range(0, len(stock), 60):
        cols = stock[i:i + 60]
        R = X[cols].groupby(d).rank().astype('float32')
        Rc = R - R.groupby(d).transform('mean')
        xy = Rc.mul(rc, axis=0).groupby(d).sum()
        xx = (Rc ** 2).groupby(d).sum()
        yy = Rc.notna().mul(rc ** 2, axis=0).groupby(d).sum()
        ic = (xy / np.sqrt(xx * yy)).replace([np.inf, -np.inf], np.nan)
        for c in cols:
            rows.append((c, 'stock', float(ic[c].mean())))
    T50 = X[X.top50.astype(bool)].groupby('date')
    m = T50.target_r.mean()
    for c in daylevel:
        v = T50[c].first()
        rows.append((c, 'day', float(v.rank().corr(m.rank()))))
    T = pd.DataFrame(rows, columns=['feature', 'type', 'ic']).sort_values('ic', key=lambda s: -s.abs())
    drop = T[T.ic.abs() > 0.30].feature.tolist()
    warn = T[(T.ic.abs() > 0.08) & (T.ic.abs() <= 0.30)]
    log(f'same-day IC check (D days): {len(T)} features, max |IC| {T.ic.abs().max():.4f}; > 0.30 (removed): {drop}; '
        f'0.08 .. 0.30 (listed): {len(warn)}')
    return T, drop


# ------------------------------------------------------------------ truncation tests
LEAK_SEED = 20261115


def leak_days(out):
    days = pd.DatetimeIndex(sorted(out[out.target_r.notna()].date.unique()))
    days = days[days >= pd.Timestamp('2021-06-01')]
    rng = np.random.default_rng(LEAK_SEED)
    pick = sorted(rng.choice(len(days), 9, replace=False))
    return [days[i] for i in pick]


def compare_row(full, part, t, cols):
    a = full[full.date == t].set_index('stock_id')[cols].astype(float)
    b = part.loc[part.index.get_level_values('date') == t].reset_index(level='date', drop=True)
    b = b.reindex(a.index)[cols].astype(float)
    A, B = a.to_numpy(), b.to_numpy()
    both_nan = np.isnan(A) & np.isnan(B)
    diff = np.where(both_nan, 0.0, np.abs(A - B))
    diff = np.where(np.isnan(diff), np.inf, diff)
    tol = 1e-6 * np.maximum(1.0, np.nan_to_num(np.abs(A)))
    bad = diff > tol
    badcols = [cols[j] for j in np.flatnonzero(bad.any(axis=0))]
    return float(np.nanmax(np.where(np.isinf(diff), np.nan, diff))) if diff.size else 0.0, badcols


def leak_test(i0, i1, log):
    full = pd.read_pickle(L.PANEL)
    feats = L.features()
    days = leak_days(full)
    P0, uorder, usub = user_panel(L.DATA)
    lines = []
    for t in days[i0:i1]:
        t0 = time.time()
        _, _, _, FP, FC, cday, FM, ids, top50 = build_all(L.DATA, until=t, blank_last=True, need_u=False, Uprev=(P0, uorder, usub))
        for g, part in (('P', FP), ('C', FC), ('M', FM)):
            mx, bad = compare_row(full, part, t, feats[g])
            lines.append(f'{t.date()} {g}: max |diff| {mx:.3g}; features that differ: {bad if bad else "none"}')
            log(lines[-1])
        log(f'  ({time.time() - t0:.0f} s)')
    return lines


def leak_test_u(i0, i1, log):
    full = pd.read_pickle(L.PANEL)
    feats = L.features()
    days = leak_days(full)
    rep = []
    for t in days[i0:i1]:
        ts = t.strftime('%Y-%m-%d')
        tmp = tempfile.mkdtemp(prefix='race_leak_')
        try:
            ids, top50 = read_info(L.DATA)
            os.makedirs(os.path.join(tmp, 'market')); os.makedirs(os.path.join(tmp, 'global'))
            shutil.copytree(os.path.join(L.DATA, 'ref'), os.path.join(tmp, 'ref'))
            shutil.copy(os.path.join(L.DATA, 'info.json'), os.path.join(tmp, 'info.json'))
            for s in ids:                  # text-level truncation: earlier rows stay byte-identical
                with open(os.path.join(L.DATA, 'market', f'{s}.csv'), encoding='utf-8') as fh:
                    lines = fh.read().splitlines()
                out_l = [lines[0]]
                for ln in lines[1:]:
                    d = ln[:10]
                    if d < ts:
                        out_l.append(ln)
                    elif d == ts:                  # the course placeholder of the day to predict
                        out_l.append(ln.split(',', 1)[0] + ',' + ','.join(['-1'] * (len(lines[0].split(',')) - 1)))
                with open(os.path.join(tmp, 'market', f'{s}.csv'), 'w', encoding='utf-8', newline='') as fh:
                    fh.write('\n'.join(out_l) + '\n')
            with open(os.path.join(L.DATA, 'global', 'global_data.csv'), encoding='utf-8') as fh:
                lines = fh.read().splitlines()
            hdr = lines[0].split(','); out_l = [lines[0]]
            for ln in lines[1:]:
                d = ln[:10]
                if d < ts:
                    out_l.append(ln)
                elif d == ts:                      # on day t only last night's U.S. close is known
                    v = ln.split(',')
                    for c in ('^TWII', '^KS11', 'TX_OI'):
                        v[hdr.index(c)] = '-1'
                    out_l.append(','.join(v))
            with open(os.path.join(tmp, 'global', 'global_data.csv'), 'w', encoding='utf-8', newline='') as fh:
                fh.write('\n'.join(out_l) + '\n')
            P, uorder, _ = user_panel(tmp)
            P['date'] = pd.to_datetime(P.date).astype('datetime64[ns]'); P['stock_id'] = P.stock_id.astype(str)
            part = P.set_index(['date', 'stock_id'])[uorder]
            part.columns = ['U__' + c for c in uorder]
            mx, bad = compare_row(full, part, t, feats['U'])
            rep.append(f'{t.date()} U: max |diff| {mx:.3g}; features that differ: {bad if bad else "none"}')
            log(rep[-1])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--leak', nargs=2, type=int)
    ap.add_argument('--leak-u', nargs=2, type=int)
    ap.add_argument('--ic', action='store_true')
    a = ap.parse_args()
    os.makedirs(L.CACHE, exist_ok=True); os.makedirs(L.RESULTS, exist_ok=True)
    log = L.log_to(os.path.join(L.RESULTS, 'build_panel.log'))
    if a.leak:
        lines = leak_test(a.leak[0], a.leak[1], log)
        with open(os.path.join(L.RESULTS, 'leak_check_trunc.txt'), 'a', encoding='utf-8') as fh:
            fh.write('\n'.join(lines) + '\n')
        return
    if a.leak_u:
        lines = leak_test_u(a.leak_u[0], a.leak_u[1], log)
        with open(os.path.join(L.RESULTS, 'leak_check_trunc.txt'), 'a', encoding='utf-8') as fh:
            fh.write('\n'.join(lines) + '\n')
        return
    if a.ic:
        out = pd.read_pickle(L.PANEL); feats = L.features()
        T, drop = ic_check(out, feats, log)
        T.to_csv(os.path.join(L.RESULTS, 'leak_check_ic.csv'), index=False)
        for g in list(feats):
            if not g.startswith('_'):
                feats[g] = [c for c in feats[g] if c not in drop]
        feats['_removed_by_ic'] = drop
        with open(L.FEATS, 'w', encoding='utf-8') as fh:
            json.dump(feats, fh, ensure_ascii=False, indent=1)
        return
    t0 = time.time()
    P, uorder, usub, FP, FC, cday, FM, ids, top50 = build_all(L.DATA, log=log)
    log(f'features built in {time.time() - t0:.0f} s: U {len(uorder)}, P {FP.shape[1]}, C {FC.shape[1]}, M {FM.shape[1]}')
    out, feats = assemble(P, uorder, usub, FP, FC, cday, FM)
    feats['_removed_by_ic'] = None
    out.to_pickle(L.PANEL)
    out[['date', 'stock_id', 'top50', 'target_r', 'resid_r', 'group'] + list(L.EX4)].to_pickle(os.path.join(L.CACHE, 'info.pkl'))
    day = out.groupby('date')[feats['C_DAY']].first()
    day.to_pickle(os.path.join(L.CACHE, 'day_features.pkl'))
    with open(L.FEATS, 'w', encoding='utf-8') as fh:
        json.dump(feats, fh, ensure_ascii=False, indent=1)
    lab = out[out.target_r.notna()]
    data = snapshot(L.DATA); data.update(ext_files())
    snap = dict(written=time.strftime('%Y-%m-%d %H:%M:%S'), prereg_sha256=L.sha256(L.PREREG), data=data,
                data_sha256=L.dict_sha256(data),
                rows=int(len(out)), stocks=int(out.stock_id.nunique()), first_day=str(out.date.min().date()),
                last_day=str(out.date.max().date()), last_labelled_day=str(lab.date.max().date()),
                n_features={g: len(feats[g]) for g in L.GROUPS}, versions=versions(), code=L.code_hashes())
    with open(os.path.join(L.RESULTS, 'snapshot.json'), 'w', encoding='utf-8') as fh:
        json.dump(snap, fh, ensure_ascii=False, indent=1)
    log(f'panel {out.shape} written in {time.time() - t0:.0f} s; last labelled day {snap["last_labelled_day"]}')


if __name__ == '__main__':
    main()

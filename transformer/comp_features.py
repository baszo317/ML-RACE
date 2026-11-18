"""
comp_features.py -- the 67 competition-data features ("B_competition_only") for every date, plus the next
trading day after the last data date (so it can be predicted).

Uses ONLY the competition files: data/market/*.csv, data/global/global_data.csv, data/info.json.
Timing rule (no look-ahead): every feature on row D uses information published before 09:00 of day D.
  * stock data of day d (price, volume, institutional flows, indicators) -> usable from the next trading day
  * global_data ^VIX ^IXIC ^GSPC: row d already holds the previous U.S. session -> usable on d
  * global_data ^TWII ^KS11 TX_OI: row d = same-day close -> usable from the next row
Identical (to float32 precision) to the C_* columns of best_features/panel_candidates.parquet.
"""
import os, json
import numpy as np, pandas as pd

SEMI = {'2330','2454','2303','3711','2379','3034','3661','5274','3443','2408','2344','6415','3529','5347','8299','6488','2449','6223','5269'}
TRAD = {'1216','6505','2207','2002','1303','2912','1301','2615','2609','2603','2618','1101','1326','1402','2633','1504',
        '1102','2105','2610','9910','8464','1476','1590','1519','2412','3045','4904','6446','3293'}

def group_code(s, sids):
    fin = {x for x in sids if x.startswith('28')} | {'5871', '5876', '5880'}
    return 0 if s in fin else 1 if s in SEMI else 3 if s in TRAD else 2   # 0 fin, 1 semi, 2 electronics, 3 traditional/other

def _stock(df):
    o, h, l, c = [df[f'etl:adj_{k}'] for k in ('open', 'high', 'low', 'close')]
    v, amt, n, cols = df['price:成交股數'], df['price:成交金額'], df['price:成交筆數'], df.columns
    F = pd.DataFrame(index=df.index); r1 = np.log(c).diff(); rr = (c - o) / o; gap = np.log(o / c.shift())
    F['ret1'] = r1; F['ret5'] = np.log(c).diff(5); F['ret20'] = np.log(c).diff(20)
    F['intraday_ret'] = np.log(c / o); F['gap'] = gap; F['y_mean5'] = df['y'].rolling(5).mean()
    F['range'] = (h - l) / c; F['clv'] = ((c - l) / (h - l)).where(h > l, 0.5)
    F['upper_shadow'] = (h - np.maximum(o, c)) / c; F['lower_shadow'] = (np.minimum(o, c) - l) / c
    F['vol_ratio'] = np.log(v / v.rolling(20).mean()); F['amt_ratio'] = np.log(amt / amt.rolling(20).mean())
    F['trades_ratio'] = np.log(n / n.rolling(20).mean()); F['avg_trade_size'] = np.log((v / n) / (v / n).rolling(20).mean())
    F['RSI'] = df.RSI; F['close_SMA30'] = c / df.SMA - 1; F['MACD_hist_norm'] = df.MACD / c; F['ATR_pct'] = df.ATR / c
    F['BB_pctb'] = ((c - df.BBANDS_lower) / (df.BBANDS_upper - df.BBANDS_lower)).clip(-2, 3)
    F['BB_width'] = (df.BBANDS_upper - df.BBANDS_lower) / df.BBANDS_mid
    F['vol_ratio_5_60'] = r1.rolling(5).std() / r1.rolling(60).std()
    F['limit_up_prev'] = (r1 > 0.093).astype(float); F['limit_dn_prev'] = (r1 < -0.093).astype(float)
    yy = df['y']; F['streak_y'] = (yy.groupby((yy != yy.shift()).cumsum()).cumcount() + 1) * np.where(yy == 1, 1, -1)
    F['intra_bias_20'] = rr.rolling(20, min_periods=10).mean(); F['intra_bias_60'] = rr.rolling(60, min_periods=30).mean()
    F['intra_bias_250'] = rr.rolling(250, min_periods=120).mean(); F['intra_bias_exp'] = rr.expanding(min_periods=60).mean()
    F['gap_bias_20'] = gap.rolling(20, min_periods=10).mean()
    fh = df[cols[8]]; F['foreign_hold'] = fh; F['foreign_hold_chg1'] = fh.diff(); F['foreign_hold_chg5'] = fh.diff(5)
    adv = v.rolling(20).mean()
    for i, nm in [(9, 'foreign_net'), (10, 'trust_net'), (11, 'dealer_self_net'), (12, 'dealer_hedge_net')]:
        F[nm] = df[cols[i]] / adv; F[nm + '_5d'] = df[cols[i]].rolling(5).sum() / adv / 5
    fs = np.sign(df[cols[9]]); F['foreign_streak'] = (fs.groupby((fs != fs.shift()).cumsum()).cumcount() + 1) * fs
    F = F.shift(1)                       # known only after the close -> usable on the next trading day
    F['gap_today_raw'] = gap             # helper for beta (never a feature)
    F['date'] = df.date; F['target_r'] = rr; F['y'] = df['y']
    return F

def read_market(path):
    """Read one stock file. The course server marks the day still to be predicted with a row whose values are all -1:
    that row is kept (it is the day to predict) but its values become NaN."""
    df = pd.read_csv(path, parse_dates=['date']).sort_values('date').drop_duplicates('date', keep='last').reset_index(drop=True)
    vals = df.columns.drop('date')
    df.loc[(df[vals] == -1).all(axis=1), vals] = np.nan
    return df

def read_global(path):
    """Read global_data.csv; -1 placeholders (today's ^TWII ^KS11 TX_OI before the close) become NaN."""
    g = pd.read_csv(path, parse_dates=['date']).sort_values('date').drop_duplicates('date', keep='last').reset_index(drop=True)
    ph = g['^TWII'] == -1
    g.loc[ph, ['^TWII', '^KS11', 'TX_OI']] = np.nan
    for c in ['^VIX', '^IXIC', '^GSPC']:
        g.loc[g[c] == -1, c] = np.nan
    return g

def placeholder_dates(data_dir):
    """Dates whose market rows are all -1 (the day(s) the course server asks you to predict)."""
    info = json.load(open(os.path.join(data_dir, 'info.json')))
    df = pd.read_csv(os.path.join(data_dir, 'market', f"{info['Top50@Y'][0]}.csv"), parse_dates=['date'])
    return sorted(df.loc[(df.drop(columns='date') == -1).all(axis=1), 'date'])

def build_comp_features(data_dir, pred_dates=()):
    """data_dir: folder containing market/, global/, info.json.  pred_dates: dates you want to predict.
    Returns one row per (date, stock) for all data dates (incl. -1 placeholder days) and, if a requested date lies
    beyond the data, for the first such date."""
    info = json.load(open(os.path.join(data_dir, 'info.json'))); sids = info['Top100@Y']; top50 = set(info['Top50@Y'])
    raw = {s: read_market(os.path.join(data_dir, 'market', f'{s}.csv')) for s in sids}
    last = max(d.date.max() for d in raw.values())
    nxt = sorted(pd.Timestamp(d) for d in pred_dates if pd.Timestamp(d) > last)[:1]   # next day only: needs data of the day before
    P = []
    for s, df in raw.items():
        if nxt and df.date.max() < nxt[0]:
            df = pd.concat([df, pd.DataFrame({'date': nxt})], ignore_index=True)        # placeholder row, values NaN
        F = _stock(df); F['stock_id'] = s; P.append(F)
    P = pd.concat(P, ignore_index=True)
    P['group'] = P.stock_id.map(lambda s: group_code(s, sids)).astype(float); P['is_fin'] = (P.group == 0).astype(float)
    d = P.groupby('date').agg(m=('target_r', 'mean'), b=('y', 'mean')).sort_index()
    d['mkt_intra_prev'] = d.m.shift(1); d['breadth_prev'] = d.b.shift(1); d['mkt_intra_5'] = d.m.rolling(5).mean().shift(1)
    P = P.merge(d[['mkt_intra_prev', 'breadth_prev', 'mkt_intra_5']], left_on='date', right_index=True, how='left')
    sg = P.groupby(['group', 'date']).target_r.mean().rename('s').reset_index().sort_values(['group', 'date'])
    sg['sector_intra_prev'] = sg.groupby('group').s.shift(1)
    P = P.merge(sg[['group', 'date', 'sector_intra_prev']], on=['group', 'date'], how='left')
    P = P.sort_values(['stock_id', 'date']).reset_index(drop=True)
    P['rel_intra_prev'] = P.groupby('stock_id').target_r.shift(1) - P.mkt_intra_prev
    g = read_global(os.path.join(data_dir, 'global', 'global_data.csv'))
    U = pd.DataFrame({'date': g.date}); U['us_vix'] = g['^VIX']; U['us_vix_chg'] = np.log(g['^VIX']).diff()
    for k, nm in [('^IXIC', 'ixic'), ('^GSPC', 'gspc')]:
        U[f'us_{nm}_ret1'] = np.log(g[k]).diff(); U[f'us_{nm}_ret5'] = np.log(g[k]).diff(5)
    for nm in ['us_vix_chg', 'us_ixic_ret1', 'us_gspc_ret1']:
        U[nm + '_lag1'] = U[nm].shift(1)
    W = pd.DataFrame({'date': g.date})
    for k, nm in [('^TWII', 'twii'), ('^KS11', 'ks11')]:
        W[f'{nm}_ret1'] = np.log(g[k]).diff(); W[f'{nm}_ret5'] = np.log(g[k]).diff(5)
    W['txoi'] = g['TX_OI'] / 1e4; W['txoi_chg'] = g['TX_OI'].diff() / 1e4; W['txoi_chg5'] = g['TX_OI'].diff(5) / 1e4
    P = P.sort_values('date')
    P = pd.merge_asof(P, U, on='date', direction='backward')
    P = pd.merge_asof(P, W, on='date', direction='backward', allow_exact_matches=False)
    P = P.sort_values(['stock_id', 'date']).reset_index(drop=True)
    def _beta(x):
        cv = x.gap_today_raw.rolling(120, min_periods=60).cov(x.us_ixic_ret1); vr = x.us_ixic_ret1.rolling(120, min_periods=60).var()
        return (cv / vr).shift(1)
    P['beta_us'] = P.groupby('stock_id', group_keys=False)[['gap_today_raw', 'us_ixic_ret1']].apply(_beta)
    P['beta_x_us'] = P.beta_us * P.us_ixic_ret1; P['beta_x_us_lag1'] = P.beta_us * P.us_ixic_ret1_lag1
    P = P.drop(columns='gap_today_raw')
    P['top50'] = P.stock_id.isin(top50)
    P['has_label'] = P.target_r.notna()
    P['stale_us'] = False
    if nxt:   # the U.S. session right before the extra day is not in global_data yet -> U.S. features are one session old
        P.loc[P.date == nxt[0], 'stale_us'] = True
    return P.sort_values(['date', 'stock_id']).reset_index(drop=True)

FEATURES = None   # filled from final_feature_lists.json by the caller (B_competition_only)

"""
build_features.py -- candidate feature panel for ML RACE (predict close > open on day D)

Timing rule: every feature on row D uses only information published before 09:00 of day D.
  * competition stock data of day d (price, flows, indicators)   -> usable from the next trading day
  * global_data: ^VIX ^IXIC ^GSPC row d = previous U.S. session  -> usable on d itself
                 ^TWII ^KS11 TX_OI row d = same-day close         -> usable from the next row
  * data/pit tables: usable on D when available_from <= D (as-of join)
  * data/raw tables: data date d -> available_from = next Taiwan trading day after d
User rule: NEW data (data/pit, data/raw) rows dated on/after 2026-06-01 are dropped, and the
panel ends 2026-05-31 so the static-leaderboard period (Jun-Sep 2026) is never touched.
Output: best_features/panel_candidates.parquet + feature_catalog.csv
Run:    python build_features.py      (from anywhere; paths are relative to this file)
"""
import os, json, time
import numpy as np, pandas as pd, pyarrow.dataset as pads

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); DATA = os.path.join(ROOT, 'data')
NEW_DATA_END = pd.Timestamp('2026-05-31')
PANEL_START, PANEL_END = pd.Timestamp('2020-03-02'), pd.Timestamp('2026-05-31')
info = json.load(open(os.path.join(DATA, 'info.json'))); SIDS = info['Top100@Y']; TOP50 = set(info['Top50@Y'])
SEMI = {'2330','2454','2303','3711','2379','3034','3661','5274','3443','2408','2344','6415','3529','5347','8299','6488','2449','6223','5269'}
FIN = {s for s in SIDS if s.startswith('28')} | {'5871','5876','5880'}
TRAD = {'1216','6505','2207','2002','1303','2912','1301','2615','2609','2603','2618','1101','1326','1402','2633','1504',
        '1102','2105','2610','9910','8464','1476','1590','1519','2412','3045','4904','6446','3293'}
def group_code(s):  # 0 financial, 1 semiconductor, 2 electronics / AI hardware, 3 traditional / telecom / shipping / other
    return 0 if s in FIN else 1 if s in SEMI else 3 if s in TRAD else 2

CAT = {}  # feature -> (group, source, timing, meaning)
def reg(names, group, source, timing, meaning):
    for n in ([names] if isinstance(names, str) else names):
        CAT[n] = (group, source, timing, meaning)

cal = pd.to_datetime(pd.read_parquet(os.path.join(DATA, 'raw/finmind/TaiwanStockTradingDate/_ALL.parquet')).date)
cal = cal.sort_values().drop_duplicates().reset_index(drop=True); CALV = cal.values
def next_tday(s):
    i = np.searchsorted(CALV, pd.to_datetime(s).values, side='right')
    return pd.Series(CALV[np.minimum(i, len(CALV) - 1)], index=s.index)

# ============================================================ 1. competition stock data
def comp_stock(df):
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
    F = F.shift(1)                       # everything above is known only after the close -> next trading day
    F['gap_today_raw'] = gap             # helper for beta only (needs today's open, never used as a feature)
    F['date'] = df.date; F['target_r'] = rr; F['y'] = df['y']
    return F

P = []
for sid in SIDS:
    df = pd.read_csv(os.path.join(DATA, 'market', f'{sid}.csv'), parse_dates=['date']).sort_values('date').reset_index(drop=True)
    F = comp_stock(df); F['sid'] = sid; P.append(F)
P = pd.concat(P, ignore_index=True)
reg(['ret1','ret5','ret20','intraday_ret','gap','y_mean5','range','clv','upper_shadow','lower_shadow','vol_ratio','amt_ratio',
     'trades_ratio','avg_trade_size','RSI','close_SMA30','MACD_hist_norm','ATR_pct','BB_pctb','BB_width','vol_ratio_5_60',
     'limit_up_prev','limit_dn_prev','streak_y'], 'C_price_tech', 'competition market csv', 't-1 close',
    'price / volume / technical state of the previous day')
reg(['intra_bias_20','intra_bias_60','intra_bias_250','intra_bias_exp','gap_bias_20'], 'C_bias', 'competition market csv',
    't-1 close', 'rolling mean of past intraday returns (stock drift) and of past opening gaps')
reg(['foreign_hold','foreign_hold_chg1','foreign_hold_chg5','foreign_net','foreign_net_5d','trust_net','trust_net_5d',
     'dealer_self_net','dealer_self_net_5d','dealer_hedge_net','dealer_hedge_net_5d','foreign_streak'], 'C_flows',
    'competition market csv', 't-1 (published after close)', 'institutional flows / foreign holding, scaled by 20d ADV')

P['group'] = P.sid.map(group_code).astype(float); P['is_fin'] = (P.group == 0).astype(float)
reg(['group','is_fin'], 'C_identity', 'manual sector map', 'static', '0 fin, 1 semiconductor, 2 electronics, 3 traditional/other')
d = P.groupby('date').agg(m=('target_r', 'mean'), b=('y', 'mean')).sort_index()
d['mkt_intra_prev'] = d.m.shift(1); d['breadth_prev'] = d.b.shift(1); d['mkt_intra_5'] = d.m.rolling(5).mean().shift(1)
P = P.merge(d[['mkt_intra_prev','breadth_prev','mkt_intra_5']], left_on='date', right_index=True, how='left')
sg = P.groupby(['group','date']).target_r.mean().rename('s').reset_index().sort_values(['group','date'])
sg['sector_intra_prev'] = sg.groupby('group').s.shift(1)
P = P.merge(sg[['group','date','sector_intra_prev']], on=['group','date'], how='left')
P = P.sort_values(['sid','date']).reset_index(drop=True)
P['rel_intra_prev'] = P.groupby('sid').target_r.shift(1) - P.mkt_intra_prev
reg(['mkt_intra_prev','breadth_prev','mkt_intra_5','sector_intra_prev','rel_intra_prev'], 'C_xsection', 'competition market csv',
    't-1 close', 'yesterday market / sector intraday state and the stock residual')

# ============================================================ 2. competition global data
g = pd.read_csv(os.path.join(DATA, 'global', 'global_data.csv'), parse_dates=['date']).sort_values('date').reset_index(drop=True)
U = pd.DataFrame({'date': g.date}); U['us_vix'] = g['^VIX']; U['us_vix_chg'] = np.log(g['^VIX']).diff()
for k, nm in [('^IXIC','ixic'), ('^GSPC','gspc')]:
    U[f'us_{nm}_ret1'] = np.log(g[k]).diff(); U[f'us_{nm}_ret5'] = np.log(g[k]).diff(5)
for nm in ['us_vix_chg','us_ixic_ret1','us_gspc_ret1']:
    U[nm + '_lag1'] = U[nm].shift(1)
W = pd.DataFrame({'date': g.date})
for k, nm in [('^TWII','twii'), ('^KS11','ks11')]:
    W[f'{nm}_ret1'] = np.log(g[k]).diff(); W[f'{nm}_ret5'] = np.log(g[k]).diff(5)
W['txoi'] = g['TX_OI'] / 1e4; W['txoi_chg'] = g['TX_OI'].diff() / 1e4; W['txoi_chg5'] = g['TX_OI'].diff(5) / 1e4
P = P.sort_values('date')
P = pd.merge_asof(P, U, on='date', direction='backward')                              # U.S. row d usable on d
P = pd.merge_asof(P, W, on='date', direction='backward', allow_exact_matches=False)    # TW / KR / TX_OI row d usable after d
reg(['us_vix','us_vix_chg','us_ixic_ret1','us_ixic_ret5','us_gspc_ret1','us_gspc_ret5'], 'C_us', 'competition global csv',
    'last U.S. close (before TW open)', 'U.S. market overnight')
reg(['us_vix_chg_lag1','us_ixic_ret1_lag1','us_gspc_ret1_lag1'], 'C_us', 'competition global csv', 'U.S. session before last',
    'U.S. market two nights ago')
reg(['twii_ret1','twii_ret5','ks11_ret1','ks11_ret5'], 'C_tw_kr', 'competition global csv', 't-1 close', 'TAIEX / KOSPI returns')
reg(['txoi','txoi_chg','txoi_chg5'], 'C_txoi', 'competition global csv', 't-1 (after close)', 'foreign net open interest in TX futures')
P = P.sort_values(['sid','date']).reset_index(drop=True)
def _beta(x):
    cv = x.gap_today_raw.rolling(120, min_periods=60).cov(x.us_ixic_ret1); vr = x.us_ixic_ret1.rolling(120, min_periods=60).var()
    return (cv / vr).shift(1)
P['beta_us'] = P.groupby('sid', group_keys=False)[['gap_today_raw','us_ixic_ret1']].apply(_beta)
P['beta_x_us'] = P.beta_us * P.us_ixic_ret1; P['beta_x_us_lag1'] = P.beta_us * P.us_ixic_ret1_lag1
P = P.drop(columns='gap_today_raw')
reg(['beta_us','beta_x_us','beta_x_us_lag1'], 'C_us', 'competition csv', 't-1 / last U.S. close',
    'rolling sensitivity of the opening gap to Nasdaq, times the overnight move')

dts = pd.Series(sorted(P.date.unique())); i = np.searchsorted(CALV, dts.values)
assert (CALV[i] == dts.values).all(), 'competition dates missing from trading calendar'
nxt = pd.Series(CALV[np.minimum(i + 1, len(CALV) - 1)]); prv = pd.Series(CALV[np.maximum(i - 1, 0)])
C = pd.DataFrame({'date': dts, 'dow': dts.dt.dayofweek.astype(float),
                  'pre_holiday': ((nxt - dts).dt.days > 3).astype(float), 'post_holiday': ((dts - prv).dt.days > 3).astype(float)})
ym = dts.dt.to_period('M'); C['tdom'] = (C.groupby(ym).cumcount() + 1).astype(float)
C['turn_of_month'] = ((C.tdom <= 3) | (C.groupby(ym).cumcount(ascending=False) == 0)).astype(float)
C['settle_monthly'] = ((C.dow == 2) & dts.dt.day.between(15, 21)).astype(float)
P = P.merge(C, on='date', how='left')
reg(['dow','pre_holiday','post_holiday','tdom','turn_of_month','settle_monthly'], 'C_calendar', 'exchange calendar',
    'known in advance', 'calendar / settlement effects')
print(f'[{time.time()-T0:.0f}s] competition features done {P.shape}', flush=True)

# ============================================================ 3. NEW data (rows dated after NEW_DATA_END dropped)
def read_pit(name, cols, stock=True):
    t = pads.dataset(os.path.join(DATA, 'pit', f'{name}.parquet'))
    x = t.to_table(columns=cols, filter=pads.field('stock_id').isin(SIDS) if stock else None).to_pandas()
    for c in ('date', 'available_from'):
        x[c] = pd.to_datetime(x[c])
    return x[x.date <= NEW_DATA_END]

def asof(P, T, cols, tol_days, by=True):
    # ties: several data rows can become available on the same day (e.g. U.S. sessions during a Taiwan holiday);
    # sort by (available_from, date) so the as-of join always takes the most recent data row
    T = T.sort_values(['available_from', 'date'], kind='mergesort'); L = P[['date','sid']].reset_index().sort_values('date')
    if by:
        m = pd.merge_asof(L, T.rename(columns={'stock_id': 'sid'})[['sid','available_from'] + cols], left_on='date',
                          right_on='available_from', by='sid', direction='backward', tolerance=pd.Timedelta(days=tol_days))
    else:
        m = pd.merge_asof(L, T[['available_from'] + cols], left_on='date', right_on='available_from',
                          direction='backward', tolerance=pd.Timedelta(days=tol_days))
    m = m.set_index('index')
    for c in cols:
        P[c] = m[c].reindex(P.index)
    return P

# 3a margin: pit (utilisation / 5d changes) + raw daily margin & short, credit day-trade ratio; raw price level
x = read_pit('margin', ['stock_id','date','available_from','margin_util','short_to_margin','margin_chg_5','short_chg_5'])
x = x.rename(columns={c: 'nd_' + c for c in ['margin_util','short_to_margin','margin_chg_5','short_chg_5']})
P = asof(P, x, ['nd_margin_util','nd_short_to_margin','nd_margin_chg_5','nd_short_chg_5'], 10)
R = []
for sid in SIDS:
    fm = os.path.join(DATA, 'raw/finmind/TaiwanStockMarginPurchaseShortSale', f'{sid}.parquet')
    fp = os.path.join(DATA, 'raw/finmind/TaiwanStockPrice', f'{sid}.parquet')
    if not (os.path.exists(fm) and os.path.exists(fp)):
        continue
    m = pd.read_parquet(fm, columns=['date','MarginPurchaseBuy','MarginPurchaseTodayBalance','MarginPurchaseYesterdayBalance',
                                     'OffsetLoanAndShort','ShortSaleSell','ShortSaleTodayBalance','ShortSaleYesterdayBalance'])
    p = pd.read_parquet(fp, columns=['date','Trading_Volume','close'])
    for z in (m, p):
        z['date'] = pd.to_datetime(z.date)
    z = p.merge(m, on='date', how='left').sort_values('date'); z = z[z.date <= NEW_DATA_END].reset_index(drop=True)
    vol = z.Trading_Volume.replace(0, np.nan); adv = vol.rolling(20, min_periods=10).mean()
    o = pd.DataFrame({'stock_id': sid, 'date': z.date})
    o['nd_margin_flow'] = (z.MarginPurchaseTodayBalance - z.MarginPurchaseYesterdayBalance) * 1000 / adv   # FinMind margin unit = lots of 1000 shares
    o['nd_short_flow'] = (z.ShortSaleTodayBalance - z.ShortSaleYesterdayBalance) * 1000 / adv
    o['nd_offset_ratio'] = z.OffsetLoanAndShort * 1000 / vol
    o['nd_offset_ratio_20'] = o.nd_offset_ratio.rolling(20, min_periods=10).mean()
    o['nd_margin_buy_ratio'] = z.MarginPurchaseBuy * 1000 / vol; o['nd_short_sell_ratio'] = z.ShortSaleSell * 1000 / vol
    c = z.close.where(z.close > 0)        # FinMind records close = 0 on some no-trade days -> missing
    tick = np.select([c < 10, c < 50, c < 100, c < 500, c < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
    o['nd_tick_rel'] = tick / c; o['nd_log_price'] = np.log(c)
    R.append(o)
R = pd.concat(R, ignore_index=True); R['available_from'] = next_tday(R.date)
rawm = ['nd_margin_flow','nd_short_flow','nd_offset_ratio','nd_offset_ratio_20','nd_margin_buy_ratio','nd_short_sell_ratio']
P = asof(P, R, rawm + ['nd_tick_rel','nd_log_price'], 10)
reg(['nd_margin_util','nd_short_to_margin','nd_margin_chg_5','nd_short_chg_5'], 'N_margin', 'data/pit/margin',
    't-1 (available_from<=D)', 'margin utilisation, short/margin ratio, 5d changes')
reg(rawm, 'N_margin', 'data/raw TaiwanStockMarginPurchaseShortSale', 't-1 (next trading day)',
    'daily margin/short flow vs ADV, credit day-trade (資券互抵) share of volume, margin-buy and short-sell share of volume')
reg(['nd_tick_rel','nd_log_price'], 'N_rawprice', 'data/raw TaiwanStockPrice', 't-1 close', 'tick size relative to raw price, raw price level')
print(f'[{time.time()-T0:.0f}s] margin / raw price done', flush=True)

# 3b valuation, revenue, other windows of flows / holding
x = read_pit('per', ['stock_id','date','available_from','earnings_yield','book_to_price','dividend_yield'])
x = x.rename(columns={c: 'nd_' + c for c in ['earnings_yield','book_to_price','dividend_yield']})
P = asof(P, x, ['nd_earnings_yield','nd_book_to_price','nd_dividend_yield'], 10)
reg(['nd_earnings_yield','nd_book_to_price','nd_dividend_yield'], 'N_valuation', 'data/pit/per', 't-1', 'E/P, B/P, dividend yield')
x = read_pit('revenue', ['stock_id','date','available_from','rev_yoy','rev_mom','rev_surprise'])
x = x.rename(columns={c: 'nd_' + c for c in ['rev_yoy','rev_mom','rev_surprise']}); x['nd_rev_avail'] = x.available_from
P = asof(P, x, ['nd_rev_yoy','nd_rev_mom','nd_rev_surprise','nd_rev_avail'], 75)
P['nd_rev_fresh'] = ((P.date - P.nd_rev_avail).dt.days <= 4).astype(float).where(P.nd_rev_avail.notna())
P = P.drop(columns='nd_rev_avail')
reg(['nd_rev_yoy','nd_rev_mom','nd_rev_surprise','nd_rev_fresh'], 'N_revenue', 'data/pit/revenue',
    'after the 10th-of-month filing deadline', 'monthly revenue growth / surprise / newly-released flag')
x = read_pit('flows', ['stock_id','date','available_from','foreign_20','trust_20','dealer_20'])
x = x.rename(columns={c: 'nd_' + c for c in ['foreign_20','trust_20','dealer_20']})
P = asof(P, x, ['nd_foreign_20','nd_trust_20','nd_dealer_20'], 10)
x = read_pit('shareholding', ['stock_id','date','available_from','foreign_ratio_chg_20']).rename(columns={'foreign_ratio_chg_20': 'nd_foreign_ratio_chg_20'})
P = asof(P, x, ['nd_foreign_ratio_chg_20'], 10)
reg(['nd_foreign_20','nd_trust_20','nd_dealer_20','nd_foreign_ratio_chg_20'], 'N_dup_flows', 'data/pit/flows, shareholding', 't-1',
    '20-day institutional flows / holding change (same underlying source as the competition data, longer window)')

# 3c dividend events (ex-dates are announced weeks ahead; ex-dates after NEW_DATA_END dropped)
pos = {dd: k for k, dd in enumerate(cal)}
E = []
for sid in SIDS:
    f = os.path.join(DATA, 'raw/finmind/TaiwanStockDividendResult', f'{sid}.parquet')
    if os.path.exists(f):
        e = pd.to_datetime(pd.read_parquet(f, columns=['date']).date); e = e[e <= NEW_DATA_END]
        E.append(pd.DataFrame({'sid': sid, 'p': e.map(pos)}))
E = pd.concat(E).dropna(); have = set(E.sid) | {s for s in SIDS if os.path.exists(os.path.join(DATA, 'raw/finmind/TaiwanStockDividendResult', f'{s}.parquet'))}
EXP = {s: np.sort(v.values.astype(int)) for s, v in E.groupby('sid').p}
P['_p'] = P.date.map(pos); assert P['_p'].notna().all()
ev = np.full((len(P), 3), np.nan)
for s, idx in P.groupby('sid').indices.items():
    if s not in have:
        continue
    a = EXP.get(s, np.array([], dtype=int)); pp = P['_p'].values[idx].astype(int)
    j = np.searchsorted(a, pp, side='left')
    nx = np.where(j < len(a), a[np.minimum(j, max(len(a) - 1, 0))] - pp if len(a) else 999, 999)
    pv = np.where(j - 1 >= 0, pp - a[np.maximum(j - 1, 0)] if len(a) else 999, 999)
    ev[idx, 0] = (nx == 0); ev[idx, 1] = (nx >= 1) & (nx <= 5); ev[idx, 2] = (pv >= 1) & (pv <= 5)
P['nd_exdiv_today'], P['nd_exdiv_next5'], P['nd_exdiv_prev5'] = ev[:, 0], ev[:, 1], ev[:, 2]
P = P.drop(columns='_p')
reg(['nd_exdiv_today','nd_exdiv_next5','nd_exdiv_prev5'], 'N_events', 'data/raw TaiwanStockDividendResult',
    'scheduled (announced in advance)', 'ex-dividend day / within 5 trading days before or after')
print(f'[{time.time()-T0:.0f}s] valuation / revenue / events done', flush=True)

# 3d market-level tables
x = read_pit('mkt_taiex', ['date','available_from','mkt_ret_20','mkt_vol_20','mkt_value'], stock=False).sort_values('date')
x['mkt_value'] = x.mkt_value.where(x.mkt_value > 0)   # vendor TAIEX turnover is 0 on 52 days (2017-18, 2026-02) -> missing
x['nd_mkt_value_ratio'] = np.log(x.mkt_value / x.mkt_value.rolling(20, min_periods=10).mean())
x = x.rename(columns={'mkt_ret_20': 'nd_mkt_ret_20', 'mkt_vol_20': 'nd_mkt_vol_20'})
P = asof(P, x, ['nd_mkt_ret_20','nd_mkt_vol_20','nd_mkt_value_ratio'], 10, by=False)
reg(['nd_mkt_ret_20','nd_mkt_vol_20','nd_mkt_value_ratio'], 'N_mkt_taiex', 'data/pit/mkt_taiex', 't-1',
    'TAIEX 20d return / volatility, market turnover vs its 20d mean')
fl = ['mkt_foreign_1','mkt_foreign_5','mkt_trust_1','mkt_trust_5','mkt_dealer_1','mkt_dealer_5']
x = read_pit('mkt_flows', ['date','available_from'] + fl, stock=False).rename(columns={c: 'nd_' + c for c in fl})
x = x.replace([np.inf, -np.inf], np.nan)              # flows are scaled by market turnover, which is 0 on the bad vendor days
P = asof(P, x, ['nd_' + c for c in fl], 10, by=False)
reg(['nd_' + c for c in fl], 'N_mkt_flows', 'data/pit/mkt_flows', 't-1', 'market-wide institutional net buying (1d / 5d)')
x = read_pit('mkt_margin', ['date','available_from','mkt_margin_chg_5','mkt_short_chg_5'], stock=False)
x = x.rename(columns={'mkt_margin_chg_5': 'nd_mkt_margin_chg_5', 'mkt_short_chg_5': 'nd_mkt_short_chg_5'})
P = asof(P, x, ['nd_mkt_margin_chg_5','nd_mkt_short_chg_5'], 10, by=False)
tm = pd.read_parquet(os.path.join(DATA, 'raw/finmind/TaiwanStockTotalMarginPurchaseShortSale/_ALL.parquet'))
tm['date'] = pd.to_datetime(tm.date); tm = tm[tm.date <= NEW_DATA_END]
tm = tm.pivot_table(index='date', columns='name', values=['TodayBalance','YesBalance'])
x = pd.DataFrame({'date': tm.index,
                  'nd_mkt_margin_chg_1': (tm['TodayBalance']['MarginPurchaseMoney'] / tm['YesBalance']['MarginPurchaseMoney'] - 1).values,
                  'nd_mkt_short_chg_1': (tm['TodayBalance']['ShortSale'] / tm['YesBalance']['ShortSale'] - 1).values})
x['available_from'] = next_tday(x.date)
P = asof(P, x, ['nd_mkt_margin_chg_1','nd_mkt_short_chg_1'], 10, by=False)
reg(['nd_mkt_margin_chg_5','nd_mkt_short_chg_5','nd_mkt_margin_chg_1','nd_mkt_short_chg_1'], 'N_mkt_margin',
    'data/pit/mkt_margin + raw total margin', 't-1', 'market-wide margin / short balance changes')
x = read_pit('mkt_fx', ['date','available_from','usdtwd_chg_1','usdtwd_chg_20'], stock=False)
x = x.rename(columns={'usdtwd_chg_1': 'nd_usdtwd_chg_1', 'usdtwd_chg_20': 'nd_usdtwd_chg_20'})
P = asof(P, x, ['nd_usdtwd_chg_1','nd_usdtwd_chg_20'], 10, by=False)
reg(['nd_usdtwd_chg_1','nd_usdtwd_chg_20'], 'N_fx', 'data/pit/mkt_fx', 't-1', 'USD/TWD change (positive = TWD weaker)')
x = read_pit('mkt_sox', ['date','available_from','sox_ret_1','sox_ret_5'], stock=False).sort_values('date')
x['nd_sox_ret_1_lag1'] = x.sox_ret_1.shift(1); x = x.rename(columns={'sox_ret_1': 'nd_sox_ret_1', 'sox_ret_5': 'nd_sox_ret_5'})
P = asof(P, x, ['nd_sox_ret_1','nd_sox_ret_5','nd_sox_ret_1_lag1'], 10, by=False)
P['nd_sox_rel_ixic'] = P.nd_sox_ret_1 - P.us_ixic_ret1
reg(['nd_sox_ret_1','nd_sox_ret_5','nd_sox_ret_1_lag1','nd_sox_rel_ixic'], 'N_sox', 'data/pit/mkt_sox',
    'last U.S. close (before TW open)', 'Philadelphia semiconductor index overnight / two nights ago / minus Nasdaq')

# 3e broad-market breadth from the ~478 listed stocks in data/pit/price
t = pads.dataset(os.path.join(DATA, 'pit', 'price.parquet')).to_table(columns=['stock_id','date','open','close','value']).to_pandas()
t['date'] = pd.to_datetime(t.date); t = t[(t.date >= '2019-09-01') & (t.date <= NEW_DATA_END) & (t.open > 0)]
t['r'] = t.close / t.open - 1; t['up'] = (t.close > t.open).astype(float); t['top'] = t.stock_id.isin(SIDS); t['rv'] = t.r * t.value
b = t.groupby('date').agg(nd_all_intra=('r','mean'), nd_all_breadth=('up','mean'), rv=('rv','sum'), v=('value','sum')).reset_index()
b['nd_all_intra_vw'] = b.rv / b.v
sm = t[~t.top].groupby('date').r.mean(); tp = t[t.top].groupby('date').r.mean()
b['nd_small_minus_top'] = b.date.map(sm - tp); b['nd_all_intra_5'] = b.nd_all_intra.rolling(5).mean()
b['available_from'] = next_tday(b.date)
bc = ['nd_all_intra','nd_all_breadth','nd_all_intra_vw','nd_small_minus_top','nd_all_intra_5']
P = asof(P, b, bc, 10, by=False); del t
reg(bc, 'N_breadth', 'data/pit/price (~478 listed stocks)', 't-1 close', 'broad-market intraday return / breadth, small-vs-large spread')
print(f'[{time.time()-T0:.0f}s] market tables done', flush=True)

# ============================================================ 4. finish
P = P[(P.date >= PANEL_START) & (P.date <= PANEL_END)].sort_values(['date','sid']).reset_index(drop=True)
P['top50'] = P.sid.isin(TOP50); P['new_data_covered'] = P.sid.isin(have)
FEATS = list(CAT)
missing = set(FEATS) - set(P.columns); assert not missing, missing
P = P[['date','sid','top50','new_data_covered','y','target_r'] + FEATS]
n_inf = int(np.isinf(P[FEATS].to_numpy(dtype='float64')).sum())
P[FEATS] = P[FEATS].replace([np.inf, -np.inf], np.nan)   # safety net: no infinite feature values
print(f'infinite values replaced by NaN: {n_inf}')
for c in FEATS:
    P[c] = P[c].astype('float32')
P.to_parquet(os.path.join(HERE, 'panel_candidates.parquet'), index=False)
pd.DataFrame([(k,) + v for k, v in CAT.items()], columns=['feature','group','source','timing','meaning']).to_csv(
    os.path.join(HERE, 'feature_catalog.csv'), index=False, encoding='utf-8-sig')
print(f'[{time.time()-T0:.0f}s] saved {P.shape} dates {P.date.min().date()}..{P.date.max().date()} features {len(FEATS)}')

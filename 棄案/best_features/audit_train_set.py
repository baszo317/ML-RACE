"""Independent audit of train_best.parquet: re-derive sampled values straight from the source files with plain filtering
(no code shared with build_features.py) and check (1) values match, (2) only information from before day D is used."""
import os, json, numpy as np, pandas as pd
ROOT = os.path.expanduser('~/mnt/ML RACE'); DATA = f'{ROOT}/data'
TR = pd.read_parquet(f'{ROOT}/best_features/train_best.parquet')
T = pd.read_parquet(f'{ROOT}/best_features/panel_candidates.parquet').rename(columns={'sid': 'stock_id'})   # superset of train_best
# train_best must be exactly the panel restricted to its columns
_c = [c for c in TR.columns if c in T.columns]
assert len(TR) == len(T) and all(np.array_equal(TR[c].values, T[c].values) or TR[c].equals(T[c]) for c in _c), 'train_best != panel subset'
assert (TR.exdiv_today_flag.values == T.nd_exdiv_today.fillna(0).values).all()
assert np.allclose(TR.sample_weight.values, np.minimum(np.abs(T.target_r.values), 0.05), atol=1e-7)
T['date'] = pd.to_datetime(T.date)
rng = np.random.default_rng(2026)
S = pd.concat([T[T.new_data_covered].sample(250, random_state=1), T[~T.new_data_covered].sample(30, random_state=2),
               T[T.date >= '2026-05-20'].sample(20, random_state=3)])
g = pd.read_csv(f'{DATA}/global/global_data.csv', parse_dates=['date']).sort_values('date').reset_index(drop=True)
cal = sorted(pd.to_datetime(pd.read_parquet(f'{DATA}/raw/finmind/TaiwanStockTradingDate/_ALL.parquet').date))
sox = pd.read_parquet(f'{DATA}/raw/finmind/USStockPrice/IDX_SOX.parquet'); sox['date'] = pd.to_datetime(sox.date); sox = sox.sort_values('date').reset_index(drop=True)
pm = pd.read_parquet(f'{DATA}/pit/margin.parquet'); pm['date'] = pd.to_datetime(pm.date); pm['available_from'] = pd.to_datetime(pm.available_from)
cache = {}
def mk(sid):
    if sid not in cache:
        cache[sid] = pd.read_csv(f'{DATA}/market/{sid}.csv', parse_dates=['date']).sort_values('date').reset_index(drop=True)
    return cache[sid]
bad = {}; n = {}
def chk(name, got, exp, tol=1e-4):
    n[name] = n.get(name, 0) + 1
    if (pd.isna(got) and pd.isna(exp)):
        return
    if pd.isna(got) or pd.isna(exp) or abs(float(got) - float(exp)) > tol * max(1.0, abs(float(exp))):
        bad.setdefault(name, []).append((got, exp))
for _, r in S.iterrows():
    sid, D = r.stock_id, r.date; m = mk(sid); i = m.index[m.date == D][0]
    # label from competition csv and raw open/close
    chk('y', r.y, int(m.loc[i, 'etl:adj_close'] > m.loc[i, 'etl:adj_open']))
    chk('target_r', r.target_r, (m.loc[i, 'etl:adj_close'] - m.loc[i, 'etl:adj_open']) / m.loc[i, 'etl:adj_open'])
    # competition stock features: must come from the previous trading row of the stock
    if i >= 2: chk('ret1', r.ret1, np.log(m.loc[i-1, 'etl:adj_close'] / m.loc[i-2, 'etl:adj_close']))
    if i >= 21:
        adv = m.loc[i-20:i-1, 'price:成交股數'].mean()
        chk('foreign_net', r.foreign_net, m.iloc[i-1, 9] / adv)
        chk('trust_net', r.trust_net, m.iloc[i-1, 10] / adv)
    chk('RSI', r.RSI, m.loc[i-1, 'RSI'])
    if i >= 61: chk('intra_bias_60', r.intra_bias_60, ((m['etl:adj_close'] - m['etl:adj_open']) / m['etl:adj_open']).loc[i-60:i-1].mean())
    # global: U.S. row <= D (row already = previous U.S. session); TW/KR/TX_OI row strictly < D
    gu = g[g.date <= D]; gw = g[g.date < D]
    chk('us_ixic_ret1', r.us_ixic_ret1, np.log(gu['^IXIC'].iloc[-1] / gu['^IXIC'].iloc[-2]))
    chk('us_vix', r.us_vix, gu['^VIX'].iloc[-1])
    chk('txoi_chg', r.txoi_chg, (gw['TX_OI'].iloc[-1] - gw['TX_OI'].iloc[-2]) / 1e4)
    chk('twii_ret1', r.twii_ret1, np.log(gw['^TWII'].iloc[-1] / gw['^TWII'].iloc[-2]))
    # SOX: last U.S. session with date < D (simple return), and the one before
    sx = sox[sox.date < D]
    chk('nd_sox_ret_1', r.nd_sox_ret_1, sx.Close.iloc[-1] / sx.Close.iloc[-2] - 1)
    chk('nd_sox_ret_1_lag1', r.nd_sox_ret_1_lag1, sx.Close.iloc[-2] / sx.Close.iloc[-3] - 1)
    prev_td = max(d for d in cal if d < D)
    if r.new_data_covered:
        qq = pm[(pm.stock_id == sid) & (pm.available_from <= D)].sort_values('available_from')
        if len(qq) == 0:                       # e.g. 6446 moved from TPEx to TWSE in 2023-07: no data before -> NaN expected
            chk('nd_margin_util(no data yet -> NaN)', float(pd.isna(r.nd_margin_util)), 1.0)
            continue
        q = qq.iloc[-1]
        assert q.date < D
        chk('nd_margin_util', r.nd_margin_util, q.margin_util)
        mr = pd.read_parquet(f'{DATA}/raw/finmind/TaiwanStockMarginPurchaseShortSale/{sid}.parquet'); mr['date'] = pd.to_datetime(mr.date)
        pr = pd.read_parquet(f'{DATA}/raw/finmind/TaiwanStockPrice/{sid}.parquet'); pr['date'] = pd.to_datetime(pr.date)
        a = mr[mr.date == prev_td]; b = pr[pr.date == prev_td]
        if len(a) and len(b):
            chk('nd_offset_ratio', r.nd_offset_ratio, a.OffsetLoanAndShort.iloc[0] * 1000 / b.Trading_Volume.iloc[0])
            chk('nd_margin_buy_ratio', r.nd_margin_buy_ratio, a.MarginPurchaseBuy.iloc[0] * 1000 / b.Trading_Volume.iloc[0])
    else:
        chk('uncovered_nd_margin_util_is_nan', float(pd.isna(r.nd_margin_util)), 1.0)
# breadth: previous trading day over the pit price universe
pp = pd.read_parquet(f'{DATA}/pit/price.parquet', columns=['date', 'open', 'close']); pp['date'] = pd.to_datetime(pp.date)
for _, r in S.sample(40, random_state=5).iterrows():
    prev_td = max(d for d in cal if d < r.date); z = pp[(pp.date == prev_td) & (pp.open > 0)]
    chk('nd_all_breadth', r.nd_all_breadth, (z.close > z.open).mean())
# global constraints
assert T.date.max() <= pd.Timestamp('2026-05-31') and T.date.min() >= pd.Timestamp('2020-03-02')
assert not T.duplicated(['date', 'stock_id']).any()
F = json.load(open(f'{ROOT}/best_features/final_feature_lists.json'))['F_best']
assert [c for c in TR.columns if c in F] == F, 'feature columns of train_best differ from F_best'
assert np.isfinite(TR[F].to_numpy('float64')[~np.isnan(TR[F].to_numpy('float64'))]).all(), 'infinite values'
print('checked rows:', len(S), '| train_best features:', len(F), '| train_best == panel subset: yes | date range', T.date.min().date(), T.date.max().date())
for k in n:
    print(f'{k:32s} checks {n[k]:4d}  mismatches {len(bad.get(k, []))}', ('' if k not in bad else bad[k][:3]))
print('RESULT:', 'PASS' if not bad else 'FAIL')

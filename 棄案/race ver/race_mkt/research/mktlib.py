r"""
mktlib.py -- race_mkt (PREREG_race_mkt.md): market view with any number of 1s, and futures positioning features.
Imports the frozen RACE-STD code (..\..\race_std\research\rslib.py) read only; numpy + pandas only.
"""
import os, sys, json, math, hashlib, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); MKT = os.path.dirname(HERE); RACEVER = os.path.dirname(MKT)
STD = os.path.join(RACEVER, 'race_std'); LIVE = os.path.join(RACEVER, 'race_live')
sys.path.insert(0, os.path.join(STD, 'research'))
import rslib as L                                                     # noqa: E402  (frozen, read only)

ROOT = L.ROOT
EXT = os.path.join(MKT, 'ext'); RES = os.path.join(MKT, 'results'); CACHE = os.path.join(MKT, 'cache')
PREREG = os.path.join(HERE, 'PREREG_race_mkt.md')
LOCK = os.path.join(RES, 'LOCK.json'); SEALED = os.path.join(RES, 'SEALED_V.json')
LAST_LABEL = pd.Timestamp('2026-11-09')
TRAIN_START = pd.Timestamp('2020-03-02')
OOS_START = pd.Timestamp('2021-01-01')
D_A, D_B = pd.Timestamp('2022-01-01'), pd.Timestamp('2025-01-01')
V_A, V_B = pd.Timestamp('2025-01-01'), pd.Timestamp('2026-11-10')
EX4 = L.EX4
LAMS = [3, 10, 30, 100, 300, 1000, 3000, 10000]
K_SET = [10, 15, 20, 25, 30, 35, 40]
DROP_DAY = ['C__I_dow', 'C__I_month', 'C__I_dom']
GAP_VARS = ['C__tx_night_ret', 'C__asia_gap_mean', 'C__n225_gap', 'C__ks11_gap', 'C__gspc_ret1', 'C__ixic_ret1',
            'C__sox_ret_1', 'C__tsm_ret_1', 'C__K_adr_prem_chg_1', 'C__us_minus_tw_ret1']
MIN_CAL, MIN_C, MIN_K = 120, 60, 120

FUT_FEATURES = ['f_txeq_net_z250', 'f_txeq_net_d1n', 'f_txeq_net_d5n', 'it_txeq_net_z250', 'it_txeq_net_d1n',
                'dl_txeq_net_z250', 'dl_txeq_net_d1n', 'f_txeq_flow_n',
                'rt_small_ratio', 'rt_small_ratio_z250', 'rt_small_ratio_d1n', 'rt_mtx_ratio_z250', 'rt_tmf_ratio_z250',
                'rt_tx_ratio_z250', 'div_retail_foreign',
                'lt10_all_z250', 'lt10_all_d1n', 'lt10_spec_z250', 'lt10_nonspec_z250', 'lt10_nonspec_d1n', 'lt5_all_z250',
                'lt10_near_minus_all_z250',
                'fo_call_net_z250', 'fo_put_net_z250', 'fo_bias_z250', 'fo_bias_d1n', 'pc_oi_z250', 'pc_oi_d1n', 'pc_vol_z60',
                'basis_dev', 'term_spread_dev', 'oi_txeq_d1n', 'oi_price_regime', 'small_vol_share_z60', 'night_day_vol_z60',
                'tw_foreign_netbuy_n', 'tw_margin_chg_n']


def log_to(path):
    return L.log_to(path)


def sha256(path):
    return L.sha256(path)


# ------------------------------------------------------------------ frozen data
def load_top50():
    """Top50 rows (labelled, date <= LAST_LABEL) indexed (date, stock_id): target_r and the ex-dividend flags"""
    X = L.load_info()
    X = X[X.top50.astype(bool) & (X.date <= LAST_LABEL) & X.target_r.notna()].copy()
    return X.set_index(['date', 'stock_id'])[['target_r'] + list(EX4)].sort_index()


def all_days():
    X = L.load_info()
    return pd.DatetimeIndex(sorted(X.date.unique()))


def market_m(T50):
    """m_t = mean target_r of the labelled Top50 (bp)"""
    return (T50.target_r.groupby(level='date').mean() * 1e4).rename('m')


def day_features():
    D = pd.read_pickle(os.path.join(STD, 'cache', 'day_features.pkl'))
    D.index = pd.DatetimeIndex(D.index).astype('datetime64[ns]')
    return D.drop(columns=[c for c in DROP_DAY if c in D.columns]).astype(float).sort_index()


def open_gap(T50):
    """g_t = Top50 mean of adj_open_t / adj_close_{t-1} - 1 (bp), from data\\market (dates <= LAST_LABEL)"""
    rows = []
    for sid in sorted(set(T50.index.get_level_values('stock_id'))):
        m = pd.read_csv(os.path.join(ROOT, 'data', 'market', f'{sid}.csv'), parse_dates=['date'])
        m = m[(m.date <= LAST_LABEL) & (m['etl:adj_open'] > 0) & (m['etl:adj_close'] > 0)].sort_values('date')
        g = m['etl:adj_open'].to_numpy() / m['etl:adj_close'].shift(1).to_numpy() - 1
        rows.append(pd.DataFrame({'date': m.date.values, 'stock_id': sid, 'g': g}))
    G = pd.concat(rows, ignore_index=True)
    G['date'] = pd.to_datetime(G.date).astype('datetime64[ns]')
    G = G.set_index(['date', 'stock_id']).g
    G = G.reindex(T50.index)
    return (G.groupby(level='date').mean() * 1e4).rename('gap')


def wf3_scores(days):
    """= race_live walkforward.scores(k=3): score of day t = mean of the models trained on the 3 trading days before t (D only)"""
    U = os.path.join(LIVE, 'cache', 'wf_units')
    preds = {}
    for x in sorted(os.listdir(U)):
        if x.endswith('.npz') and '.tmp' not in x:
            s, m = L.load_npz(os.path.join(U, x))
            preds[pd.Timestamp(m['train_end'])] = s
    days = pd.DatetimeIndex(days)
    pos = {d: i for i, d in enumerate(days)}
    rows = []
    for t in days[(days >= D_A) & (days < D_B)]:
        i = pos[t]
        ms = [days[i - h] for h in range(1, 4)]
        if not all(m in preds for m in ms):
            continue
        parts = [preds[m].xs(t, level='date', drop_level=False) for m in ms]
        rows.append(pd.concat(parts, axis=1).mean(axis=1))
    return pd.concat(rows).sort_index().rename('score')


def clu_scores(folds):
    """C.LU annual folds (seed mean), = race_std candidate C.LU"""
    s = L.load_unit('C.LU', folds)
    if s is None:
        raise RuntimeError('C.LU unit files missing')
    return s.rename('score')


def betas_for(T50):
    v8 = os.path.join(ROOT, 'v8')
    if v8 not in sys.path:
        sys.path.insert(0, v8)
    import hedge_beta as H
    dates = sorted(T50.index.get_level_values('date').unique())
    B = H.betas(dates)
    B['date'] = pd.to_datetime(B.date).astype('datetime64[ns]'); B['stock_id'] = B.stock_id.astype(str)
    B = B.set_index(['date', 'stock_id']).beta.reindex(T50.index)
    B = B.clip(0.2, 2.5)
    B = B.fillna(B.groupby(level='date').transform('median')).fillna(1.0)
    return B.rename('beta')


def twii_close():
    g = pd.read_csv(os.path.join(ROOT, 'data', 'global', 'global_data.csv'), parse_dates=['date'])
    s = g.set_index('date')['^TWII'].astype(float)
    s.index = pd.DatetimeIndex(s.index).astype('datetime64[ns]')
    return s.where(s > 0)


# ------------------------------------------------------------------ futures positioning features (PREREG section 2)
def _num(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '').str.strip(), errors='coerce')
    return df


def z_roll(x, w, mp):
    return ((x - x.rolling(w, min_periods=mp).mean()) / x.rolling(w, min_periods=mp).std()).clip(-5, 5)


def d1n(x):
    d = x.diff()
    return (d / d.rolling(60, min_periods=20).std()).clip(-5, 5)


def d5n(x):
    d = x.diff(5)
    return (d / d.rolling(250, min_periods=60).std()).clip(-5, 5)


def norm60(x):
    return (x / x.rolling(60, min_periods=20).std()).clip(-5, 5)


def _inv(s):
    s = str(s)
    return 'F' if '外資' in s else ('T' if '投信' in s else ('D' if '自營' in s else None))


def raw_tables(cut=None):
    """the downloaded tables; with cut: rows dated < cut, except the night-session rows of the futures daily tables, which may be
    dated cut (that night closes at 05:00 of the day cut) -- the truncation test of the information time.  Missing -> None"""
    def rd(*p, night_ok=False):
        f = os.path.join(EXT, *p)
        if not os.path.exists(f):
            return None
        d = pd.read_csv(f, dtype=str)
        dc = 'date' if 'date' in d.columns else '日期'
        d['date'] = pd.to_datetime(d[dc].astype(str).str.replace('/', '-').str.strip(), errors='coerce')
        d = d[d.date.notna()]
        if cut is not None:
            if night_ok:
                d = d[(d.date < cut) | ((d.date == cut) & (d.trading_session == 'after_market'))]
            else:
                d = d[d.date < cut]
        return d
    T = {}
    for c in ('TX', 'MTX', 'TMF'):
        T['inst_' + c] = rd('finmind', f'fut_inst_{c}.csv')
        T['daily_' + c] = rd('finmind', f'fut_daily_{c}.csv', night_ok=True)
    T['opt'] = rd('finmind', 'opt_inst_TXO.csv')
    T['tinst'] = rd('finmind', 'total_inst.csv')
    T['tmarg'] = rd('finmind', 'total_margin.csv')
    T['lt'] = rd('taifex', 'large_trader_TX.csv')
    T['pc'] = rd('taifex', 'pc_ratio.csv')
    return T


def futures_features(pred_days, T=None, twii=None):
    """the 37 features of PREREG section 2, indexed by prediction day t (inputs dated < t; night volume: the night dated t).
    Returns (F, dropped) where dropped lists features whose source is missing."""
    T = T if T is not None else raw_tables()
    twii = twii if twii is not None else twii_close()
    pred_days = pd.DatetimeIndex(pred_days).sort_values()
    dropped = []
    mult = {'TX': 1.0, 'MTX': 0.25, 'TMF': 0.05}
    S = {}
    # institutional net OI / net trading volume per contract and investor type
    net, flow = {}, {}
    for c in ('TX', 'MTX', 'TMF'):
        d = T.get('inst_' + c)
        if d is None or not len(d):
            continue
        d = _num(d.copy(), ['long_deal_volume', 'short_deal_volume', 'long_open_interest_balance_volume',
                            'short_open_interest_balance_volume'])
        d['inv'] = d.institutional_investors.map(_inv)
        d['n'] = d.long_open_interest_balance_volume - d.short_open_interest_balance_volume
        d['f'] = d.long_deal_volume - d.short_deal_volume
        net[c] = d.pivot_table(index='date', columns='inv', values='n', aggfunc='sum')
        flow[c] = d.pivot_table(index='date', columns='inv', values='f', aggfunc='sum')
    # whole-market OI and volume (non-spread contracts), day session; night volume; TX monthly contracts
    oi, dvol, nvol, mon_tx = {}, {}, {}, None
    for c in ('TX', 'MTX', 'TMF'):
        d = T.get('daily_' + c)
        if d is None or not len(d):
            continue
        d = _num(d.copy(), ['close', 'volume', 'settlement_price', 'open_interest', 'spread_per'])
        d = d[~d.contract_date.astype(str).str.contains('/')]
        pos = d[d.trading_session == 'position']
        oi[c] = pos.groupby('date').open_interest.sum()
        dvol[c] = pos.groupby('date').volume.sum()
        nvol[c] = d[d.trading_session == 'after_market'].groupby('date').volume.sum()
        if c == 'TX':
            mon_tx = pos[pos.contract_date.astype(str).str.fullmatch(r'\d{6}')].copy()
    if 'TX' in net:
        idx = net['TX'].index
    elif 'TX' in oi:
        idx = oi['TX'].index
    else:
        idx = pd.DatetimeIndex([])
    idx = pd.DatetimeIndex(sorted(idx))

    def ser(dct, c, inv=None, fill=np.nan):
        if c not in dct:
            return pd.Series(fill, index=idx, dtype=float)
        s = dct[c] if inv is None else (dct[c][inv] if inv in dct[c].columns else pd.Series(np.nan, index=dct[c].index))
        return s.reindex(idx).astype(float)

    def tmf(s):
        return s.fillna(0.0)                                          # TMF before 2024-07-29 (or missing) -> 0

    have_inst = all(c in net for c in ('TX', 'MTX'))
    have_oi = all(c in oi for c in ('TX', 'MTX'))
    if have_inst:
        def eq(dct, inv):
            return ser(dct, 'TX', inv) + 0.25 * ser(dct, 'MTX', inv) + 0.05 * tmf(ser(dct, 'TMF', inv, 0.0))
        Feq, Teq, Deq, Fflow = eq(net, 'F'), eq(net, 'T'), eq(net, 'D'), eq(flow, 'F')
        S['f_txeq_net_z250'] = z_roll(Feq, 250, 60); S['f_txeq_net_d1n'] = d1n(Feq); S['f_txeq_net_d5n'] = d5n(Feq)
        S['it_txeq_net_z250'] = z_roll(Teq, 250, 60); S['it_txeq_net_d1n'] = d1n(Teq)
        S['dl_txeq_net_z250'] = z_roll(Deq, 250, 60); S['dl_txeq_net_d1n'] = d1n(Deq)
        S['f_txeq_flow_n'] = norm60(Fflow)
    else:
        dropped += ['institutional (8)']
    if have_inst and have_oi:
        def retail(c):
            return -(ser(net, c, 'F', 0.0) + ser(net, c, 'T', 0.0) + ser(net, c, 'D', 0.0))
        rt_mtx = retail('MTX') / ser(oi, 'MTX')
        rt_tx = retail('TX') / ser(oi, 'TX')
        num = 0.25 * retail('MTX') + 0.05 * tmf(retail('TMF') if 'TMF' in net else pd.Series(0.0, index=idx))
        den = 0.25 * ser(oi, 'MTX') + 0.05 * tmf(ser(oi, 'TMF', fill=0.0))
        small = (num / den).clip(-1, 1)
        S['rt_small_ratio'] = small; S['rt_small_ratio_z250'] = z_roll(small, 250, 60); S['rt_small_ratio_d1n'] = d1n(small)
        S['rt_mtx_ratio_z250'] = z_roll(rt_mtx, 250, 60)
        if 'TMF' in net and 'TMF' in oi:
            S['rt_tmf_ratio_z250'] = z_roll(retail('TMF') / ser(oi, 'TMF').where(ser(oi, 'TMF') > 0), 250, 60)
        else:
            dropped += ['rt_tmf_ratio_z250']
        S['rt_tx_ratio_z250'] = z_roll(rt_tx, 250, 60)
        S['div_retail_foreign'] = (S['rt_small_ratio_z250'] - S['f_txeq_net_z250']).clip(-10, 10)
    else:
        dropped += ['retail (7)']
    # large traders (TAIFEX), TX composite = TX + MTX/4 + TMF/20
    lt = T.get('lt')
    if lt is not None and len(lt):
        lt = lt.rename(columns={c: c.strip() for c in lt.columns})
        mcol = [c for c in lt.columns if c.startswith('到期月份')][0]
        tcol = [c for c in lt.columns if c.startswith('交易人類別')][0]
        b5, s5, b10, s10, oic = '前五大交易人買方', '前五大交易人賣方', '前十大交易人買方', '前十大交易人賣方', '全市場未沖銷部位數'
        lt = _num(lt.copy(), [b5, s5, b10, s10, oic, tcol])
        lt[mcol] = lt[mcol].astype(str).str.strip()
        allm = lt[lt[mcol] == '999999']
        a0 = allm[allm[tcol] == 0].groupby('date')[[b5, s5, b10, s10, oic]].sum()
        a1 = allm[allm[tcol] == 1].groupby('date')[[b5, s5, b10, s10, oic]].sum()
        oi0 = a0[oic].where(a0[oic] > 0)
        n10 = ((a0[b10] - a0[s10]) / oi0).reindex(idx)
        n10s = ((a1[b10] - a1[s10]).reindex(oi0.index) / oi0).reindex(idx)
        n5 = ((a0[b5] - a0[s5]) / oi0).reindex(idx)
        mon = lt[lt[mcol].str.fullmatch(r'\d{6}') & ~lt[mcol].isin(['999999', '666666']) & (lt[tcol] == 0)]   # 666666 = all weeklies
        if len(mon):
            first = mon.sort_values(['date', mcol]).groupby('date').head(1).set_index('date')
            n10near = ((first[b10] - first[s10]) / first[oic].where(first[oic] > 0)).reindex(idx)
        else:
            n10near = pd.Series(np.nan, index=idx)
        S['lt10_all_z250'] = z_roll(n10, 250, 60); S['lt10_all_d1n'] = d1n(n10)
        S['lt10_spec_z250'] = z_roll(n10s, 250, 60)
        S['lt10_nonspec_z250'] = z_roll(n10 - n10s, 250, 60); S['lt10_nonspec_d1n'] = d1n(n10 - n10s)
        S['lt5_all_z250'] = z_roll(n5, 250, 60)
        S['lt10_near_minus_all_z250'] = z_roll(n10near - n10, 250, 60)
    else:
        dropped += ['large traders (7)']
    # options: foreign TXO call / put net OI amount; P/C ratios
    op = T.get('opt')
    if op is not None and len(op):
        op = _num(op.copy(), ['long_open_interest_balance_amount', 'short_open_interest_balance_amount'])
        op = op[op.institutional_investors.map(_inv) == 'F'].copy()
        op['n'] = op.long_open_interest_balance_amount - op.short_open_interest_balance_amount
        call = op[op.call_put.astype(str).str.contains('買')].groupby('date').n.sum().reindex(idx)
        put = op[op.call_put.astype(str).str.contains('賣')].groupby('date').n.sum().reindex(idx)
        S['fo_call_net_z250'] = z_roll(call, 250, 60); S['fo_put_net_z250'] = z_roll(put, 250, 60)
        S['fo_bias_z250'] = z_roll(call - put, 250, 60); S['fo_bias_d1n'] = d1n(call - put)
    else:
        dropped += ['options institutional (4)']
    pc = T.get('pc')
    if pc is not None and len(pc):
        pc = pc.rename(columns={c: c.strip() for c in pc.columns})
        oc = [c for c in pc.columns if '未平倉量比率' in c][0]; vc = [c for c in pc.columns if '成交量比率' in c][0]
        pc = _num(pc.copy(), [oc, vc]).groupby('date')[[oc, vc]].last().reindex(idx)
        S['pc_oi_z250'] = z_roll(pc[oc], 250, 60); S['pc_oi_d1n'] = d1n(pc[oc]); S['pc_vol_z60'] = z_roll(pc[vc], 60, 20)
    else:
        dropped += ['P/C (3)']
    # price / volume of the futures daily table
    ret_sign = None
    if mon_tx is not None and len(mon_tx):
        top = mon_tx.sort_values(['date', 'volume'], ascending=[True, False]).groupby('date').head(1).set_index('date')
        tw = twii.reindex(top.index)
        basis = (top.close / tw - 1).where(tw > 0).reindex(idx)
        S['basis_dev'] = ((basis - basis.rolling(20, min_periods=10).median()) * 1e4).clip(-500, 500)
        ok = mon_tx[(mon_tx.settlement_price > 0) & (mon_tx.open_interest > 0)].sort_values(['date', 'contract_date'])
        first2 = ok.groupby('date').head(2)
        rk = first2.groupby('date').cumcount()
        s1 = first2[rk.values == 0].set_index('date').settlement_price
        s2 = first2[rk.values == 1].set_index('date').settlement_price
        term = (s2.reindex(idx) / s1.reindex(idx) - 1)
        S['term_spread_dev'] = ((term - term.rolling(60, min_periods=20).median()) * 1e4).clip(-500, 500)
        ret_sign = np.sign(top.spread_per.reindex(idx))
    else:
        dropped += ['basis_dev', 'term_spread_dev']
    if have_oi:
        OIeq = ser(oi, 'TX') + 0.25 * ser(oi, 'MTX') + 0.05 * tmf(ser(oi, 'TMF', fill=0.0))
        S['oi_txeq_d1n'] = d1n(OIeq)
        if ret_sign is not None:
            S['oi_price_regime'] = ret_sign * S['oi_txeq_d1n']
        else:
            dropped += ['oi_price_regime']
        vtx, vm, vt = ser(dvol, 'TX'), ser(dvol, 'MTX'), tmf(ser(dvol, 'TMF', fill=0.0))
        share = (0.25 * vm + 0.05 * vt) / (vtx + 0.25 * vm + 0.05 * vt)
        S['small_vol_share_z60'] = z_roll(share, 60, 20)
    else:
        dropped += ['oi / volume (3)']
    # whole stock market (FinMind)
    ti = T.get('tinst')
    if ti is not None and len(ti):
        ti = _num(ti.copy(), ['buy', 'sell'])
        fo = ti[ti.name.isin(['Foreign_Investor', 'Foreign_Dealer_Self'])]
        fnet = (fo.buy - fo.sell).groupby(fo.date).sum().reindex(idx)
        S['tw_foreign_netbuy_n'] = norm60(fnet)
    else:
        dropped += ['tw_foreign_netbuy_n']
    tm = T.get('tmarg')
    if tm is not None and len(tm):
        tm = _num(tm.copy(), ['TodayBalance', 'YesBalance'])
        mm = tm[tm.name == 'MarginPurchaseMoney'].groupby('date')[['TodayBalance', 'YesBalance']].last().reindex(idx)
        S['tw_margin_chg_n'] = norm60(mm.TodayBalance - mm.YesBalance)
    else:
        dropped += ['tw_margin_chg_n']
    # align: the last futures day strictly before t, at most 7 calendar days old
    FD = pd.DataFrame(S, index=idx).sort_index()
    left = pd.DataFrame({'t': pred_days})
    right = pd.DataFrame({'d': FD.index})
    A = pd.merge_asof(left, right, left_on='t', right_on='d', allow_exact_matches=False, direction='backward')
    stale = ((A.t - A.d).dt.days > 7) | A.d.isna()
    F = FD.reindex(A.d.values)
    F.index = pd.DatetimeIndex(A.t.values)
    F.loc[stale.values, :] = np.nan
    # night volume: the night dated t (closes 05:00 on t) / the day-session volume of the futures day before t
    if 'TX' in nvol and 'TX' in dvol:
        dv_prev = dvol['TX'].reindex(A.d.values).to_numpy(float)
        nv = nvol['TX'].reindex(A.t.values).to_numpy(float)
        ratio = pd.Series(nv / np.where(dv_prev > 0, dv_prev, np.nan), index=F.index)
        ratio[stale.values] = np.nan
        F['night_day_vol_z60'] = z_roll(ratio, 60, 20)
    else:
        dropped += ['night_day_vol_z60']
    F = F.reindex(columns=[c for c in FUT_FEATURES if c in F.columns])
    F.index.name = 'date'
    return F.astype(float), dropped


# ------------------------------------------------------------------ market models (PREREG section 3)
class Prep:
    """winsorize at the training 1 % / 99 % quantiles, standardize with training mean / sd, missing -> 0"""
    def fit(self, X):
        X = np.asarray(X, float)
        with np.errstate(all='ignore'):
            self.lo = np.nanpercentile(X, 1, axis=0); self.hi = np.nanpercentile(X, 99, axis=0)
            Xc = np.clip(X, self.lo, self.hi)
            self.mu = np.nanmean(Xc, axis=0); self.sd = np.nanstd(Xc, axis=0)
        self.ok = np.isfinite(self.mu) & np.isfinite(self.sd) & (self.sd > 1e-12)
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        with np.errstate(all='ignore'):
            Z = (np.clip(X, self.lo, self.hi) - self.mu) / self.sd
        Z[:, ~self.ok] = 0.0
        return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


def wins_y(y):
    lo, hi = np.percentile(y, 1), np.percentile(y, 99)
    return np.clip(y, lo, hi)


def ridge_fit(Z, y, lam):
    p = Z.shape[1]
    ym = y.mean()
    b = np.linalg.solve(Z.T @ Z + lam * np.eye(p), Z.T @ (y - ym))
    return ym, b


def ridge_cv(X, y):
    """lambda by forward-chaining CV on 5 consecutive blocks of the training window (blocks 2..5 validated)"""
    n = len(y); edges = np.linspace(0, n, 6).astype(int)
    mse = {lam: [] for lam in LAMS}
    for b in range(1, 5):
        tr, va = slice(0, edges[b]), slice(edges[b], edges[b + 1])
        P = Prep().fit(X[tr]); Ztr, Zva = P.transform(X[tr]), P.transform(X[va])
        ytr = wins_y(y[tr])
        for lam in LAMS:
            a, w = ridge_fit(Ztr, ytr, lam)
            mse[lam].append(np.mean((y[va] - (a + Zva @ w)) ** 2))
    return min(LAMS, key=lambda l: (np.mean(mse[l]), -l))


def ols1(x, y):
    """y = a + b x"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    vx = x.var()
    b = ((x - x.mean()) * (y - y.mean())).mean() / vx if vx > 0 else 0.0
    return y.mean() - b * x.mean(), b


def fit_predict(model, Xtr, ytr, Xte, gtr=None, Xg_tr=None, Xg_te=None):
    """one refit: returns (predictions for Xte, info dict)"""
    if model in ('MB', 'MF', 'MFO'):
        lam = ridge_cv(Xtr, ytr)
        P = Prep().fit(Xtr)
        a, w = ridge_fit(P.transform(Xtr), wins_y(ytr), lam)
        return a + P.transform(Xte) @ w, {'lam': lam}
    if model == 'MS':
        P = Prep().fit(Xg_tr)
        Ztr, Zte = P.transform(Xg_tr), P.transform(Xg_te)
        g = wins_y(gtr)
        Z1 = np.c_[np.ones(len(Ztr)), Ztr]
        coef = np.linalg.lstsq(Z1, g, rcond=None)[0]
        ghat_tr = Z1 @ coef; ghat_te = np.c_[np.ones(len(Zte)), Zte] @ coef
        a, b = ols1(ghat_tr, wins_y(ytr))
        r2 = 1 - np.var(g - ghat_tr) / np.var(g)
        return a + b * ghat_te, {'gap_r2': float(r2), 'b': float(b)}
    if model == 'MV':
        P = Prep().fit(Xtr)
        Ztr, Zte = P.transform(Xtr), P.transform(Xte)
        yw = wins_y(ytr); n = len(yw)
        zs = Ztr.std(axis=0)
        r = np.where(zs > 0, ((Ztr - Ztr.mean(0)) * (yw - yw.mean())[:, None]).mean(0) / (np.where(zs > 0, zs, 1) * yw.std()), 0.0)
        t = r * np.sqrt(n - 2) / np.sqrt(np.maximum(1 - r ** 2, 1e-12))
        sel = np.abs(t) >= 2
        if not sel.any():
            return np.full(len(Zte), yw.mean()), {'n_sel': 0}
        vtr = (Ztr[:, sel] * np.sign(r[sel])).mean(1); vte = (Zte[:, sel] * np.sign(r[sel])).mean(1)
        a, b = ols1(vtr, yw)
        return a + b * vte, {'n_sel': int(sel.sum())}
    raise ValueError(model)


def walk_forward(model, X, m, gap=None, Xg=None, start=OOS_START, end=LAST_LABEL):
    """monthly refits; training = labelled days in [TRAIN_START, refit day); returns (Series m_hat for OOS days, list of infos)"""
    days = X.index
    lab = m.reindex(days)
    oos_days = days[(days >= start) & (days <= end)]
    months = sorted(set(oos_days.to_period('M')))
    out, infos = [], []
    for M in months:
        md = oos_days[oos_days.to_period('M') == M]
        s = md[0]
        tr = days[(days >= TRAIN_START) & (days < s) & lab.reindex(days).notna().to_numpy()]
        ytr = lab.reindex(tr).to_numpy(float)
        Xtr, Xte = X.reindex(tr).to_numpy(float), X.reindex(md).to_numpy(float)
        if model == 'MS':
            gl = gap.reindex(tr)
            ok = gl.notna().to_numpy()
            pred, info = fit_predict(model, Xtr[ok], ytr[ok], Xte, gtr=gl.to_numpy(float)[ok],
                                     Xg_tr=Xg.reindex(tr).to_numpy(float)[ok], Xg_te=Xg.reindex(md).to_numpy(float))
        else:
            pred, info = fit_predict(model, Xtr, ytr, Xte)
        info.update(month=str(M), n_train=int(len(tr)))
        infos.append(info)
        out.append(pd.Series(pred, index=md))
    return pd.concat(out).rename('mhat'), infos


# ------------------------------------------------------------------ calibration (PREREG section 4)
def eb(coef, se):
    """empirical-Bayes shrinkage of one coefficient: coef * max(0, 1 - 1/t^2)"""
    if not np.isfinite(se) or se <= 0:
        return 0.0
    t = coef / se
    return coef * max(0.0, 1.0 - 1.0 / (t * t)) if t != 0 else 0.0


def ols_ab_se(x, y):
    n = len(x)
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if n < 3 or sxx <= 0:
        return 0.0, 0.0, np.nan, np.nan
    b = ((x - xm) * (y - ym)).sum() / sxx
    a = ym - b * xm
    res = y - a - b * x
    s2 = (res @ res) / (n - 2)
    se_b = math.sqrt(s2 / sxx); se_a = math.sqrt(s2 * (1.0 / n + xm * xm / sxx))
    return a, b, se_a, se_b


def market_calib(mhat, m, eval_days, gated=False):
    """mu_hat_t = a + b * mhat_t from the OOS days before t (>= OOS_START, at least MIN_CAL); slope >= 0, EB-shrunk.
    gated (DL4): only days with |z| >= 1 (z = mhat standardized with the calibration window's mean / sd) bet, and (a, b) is
    estimated on the past days with |z| >= 1.  Returns DataFrame indexed by eval day: mu, a, b, z, n_cal"""
    mh = mhat.dropna(); ml = m.reindex(mh.index)
    ok = ml.notna()
    hist_x, hist_y, hist_d = mh[ok].to_numpy(float), ml[ok].to_numpy(float), mh[ok].index
    rows = []
    for t in eval_days:
        if t not in mh.index:
            rows.append((t, 0.0, 0.0, 0.0, np.nan, 0)); continue
        k = hist_d.searchsorted(t)                                   # days strictly before t
        x, y = hist_x[:k], hist_y[:k]
        if len(x) < MIN_CAL:
            rows.append((t, 0.0, 0.0, 0.0, np.nan, len(x))); continue
        mu_x, sd_x = x.mean(), x.std()
        z = (mh[t] - mu_x) / sd_x if sd_x > 0 else 0.0
        if gated:
            sel = np.abs((x - mu_x) / sd_x) >= 1 if sd_x > 0 else np.zeros(len(x), bool)
            if sel.sum() < 30 or abs(z) < 1:
                rows.append((t, 0.0, 0.0, 0.0, z, int(sel.sum()))); continue
            x, y = x[sel], y[sel]
        a, b, se_a, se_b = ols_ab_se(x, y)
        b_s = eb(max(b, 0.0), se_b) if b > 0 else 0.0
        a_s = eb(a, se_a)
        rows.append((t, a_s + b_s * mh[t], a_s, b_s, z, len(x)))
    return pd.DataFrame(rows, columns=['date', 'mu', 'a', 'b', 'z', 'n_cal']).set_index('date')


def normal_scores(n):
    from statistics import NormalDist
    nd = NormalDist()
    return np.array([nd.inv_cdf((k - 0.5) / n) for k in range(1, n + 1)])


def day_frame(T50, score, beta=None):
    """per-day arrays: sid, r, ex, q (normal score of the rank of score, high = +, NaN score lowest), score, beta"""
    X = T50.join(score.rename('score'), how='left')
    if beta is not None:
        X = X.join(beta.rename('beta'), how='left')
    out = {}
    NS = {}
    for d, g in X.groupby(level='date', sort=True):
        sid = g.index.get_level_values('stock_id').to_numpy()
        sc = g.score.to_numpy(float)
        if np.isnan(sc).all():
            continue
        scn = np.where(np.isnan(sc), -np.inf, sc)
        order = np.lexsort((sid, -scn))                               # score desc, then stock id (= rslib.decide_sd)
        n = len(g)
        if n not in NS:
            NS[n] = normal_scores(n)
        q = np.empty(n); q[order] = NS[n][::-1]                       # best rank -> largest normal score
        ex = np.zeros(n, bool)
        for c in EX4:
            ex |= g[c].to_numpy() == 1
        out[d] = dict(sid=sid, r=g.target_r.to_numpy(float), ex=ex, q=q, order=order,
                      beta=(g.beta.to_numpy(float) if beta is not None else np.ones(n)))
    return out


def stock_calib(DF, m, eval_days):
    """c_hat_t = pooled through-origin OLS of e = r - m on q over the days with scores before t (at least MIN_C days)"""
    ds = sorted(DF)
    sxy = np.zeros(len(ds)); sxx = np.zeros(len(ds))
    for j, d in enumerate(ds):
        f = DF[d]; e = f['r'] - f['r'].mean()
        sxy[j] = (f['q'] * e).sum(); sxx[j] = (f['q'] ** 2).sum()
    csy, csx = np.cumsum(sxy), np.cumsum(sxx)
    pos = pd.DatetimeIndex(ds)
    out = {}
    for t in eval_days:
        k = pos.searchsorted(t)                                      # days strictly before t
        out[t] = (csy[k - 1] / csx[k - 1] * 1e4) if k >= MIN_C else np.nan     # bp per unit q
    return pd.Series(out, name='c')


# ------------------------------------------------------------------ decisions (PREREG section 4)
def ar_of(s, f):
    return float((s * f['r']).mean() * 1e4)


def topk(f, k):
    s = -np.ones(len(f['r'])); s[f['order'][:min(k, len(s))]] = 1.0; s[f['ex']] = -1.0
    return s


def decide(f, kind, mu=0.0, c=np.nan, k=None):
    if kind == 'DL0' or (kind in ('DL1', 'DL2') and (not np.isfinite(c) or c <= 0)):
        return topk(f, 25)
    if kind == 'DL1':
        s = np.where(mu + c * f['q'] > 0, 1.0, -1.0)
    elif kind == 'DL2':
        s = np.where(f['beta'] * mu + c * f['q'] > 0, 1.0, -1.0)
    elif kind == 'DLK':
        return topk(f, k)
    else:
        raise ValueError(kind)
    s[f['ex']] = -1.0
    return s


def policy_k(DF, mhat, days_eval):
    """DL3: for each eval day t, OLS of G_s(k) = AR_s(Top-k) - AR_s(Top25) on mhat_s over the earlier days with both a score
    and an OOS mhat (EB-shrunk a, b per k); k_t = argmax of the predicted G (k = 25 -> 0). At least MIN_K days, else 25."""
    ds = [d for d in sorted(DF) if d in mhat.index and np.isfinite(mhat[d])]
    G = np.array([[ar_of(topk(DF[d], k), DF[d]) - ar_of(topk(DF[d], 25), DF[d]) for k in K_SET] for d in ds]) if ds else np.zeros((0, len(K_SET)))
    x = np.array([mhat[d] for d in ds])
    pos = pd.DatetimeIndex(ds)
    out = {}
    for t in days_eval:
        j = pos.searchsorted(t)
        if j < MIN_K or t not in mhat.index or not np.isfinite(mhat[t]):
            out[t] = 25; continue
        best_k, best_g = 25, 0.0
        for i, k in enumerate(K_SET):
            if k == 25:
                continue
            a, b, se_a, se_b = ols_ab_se(x[:j], G[:j, i])
            g = eb(a, se_a) + eb(b, se_b) * mhat[t]
            if g > best_g + 1e-12:
                best_k, best_g = k, g
        out[t] = best_k
    return pd.Series(out, name='k')


def run_decision(DF, days_eval, kind, mu=None, c=None, kser=None):
    rows = []
    for t in days_eval:
        if t not in DF:
            continue
        f = DF[t]
        if kind == 'DLK':
            s = decide(f, 'DLK', k=int(kser.get(t, 25)))
        else:
            s = decide(f, kind, mu=(0.0 if mu is None else float(mu.get(t, 0.0))), c=(np.nan if c is None else float(c.get(t, np.nan))))
        r = f['r']; mm = r.mean()
        ar = float((s * r).mean() * 1e4)
        rows.append((t, ar, float(s.mean() * mm * 1e4), float(mm * 1e4), int((s > 0).sum())))
    D = pd.DataFrame(rows, columns=['date', 'AR', 'timing', 'm', 'ones']).set_index('date')
    D['xsec'] = D.AR - D.timing
    return D


# ------------------------------------------------------------------ metrics
def nw_t(d, lags=5):
    return L.nw_t(np.asarray(d, float), lags)


def years(D):
    return {str(y): float(g.mean()) for y, g in D.groupby(pd.DatetimeIndex(D.index).year)}


def one_sided_p(t):
    from statistics import NormalDist
    return 1.0 - NormalDist().cdf(t) if np.isfinite(t) else 1.0


def holm(ps):
    """Holm step-down adjusted p-values"""
    ks = sorted(ps, key=lambda k: ps[k]); n = len(ks); out = {}; run = 0.0
    for i, k in enumerate(ks):
        run = max(run, min(1.0, (n - i) * ps[k])); out[k] = run
    return out


def roll19(ar):
    r = pd.Series(np.asarray(ar, float)).rolling(19).mean().dropna()
    return {f'P19>{x}': float((r > x).mean()) for x in (0, 10, 20, 30, 40)} | {'worst19': float(r.min()) if len(r) else np.nan}

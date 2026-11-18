"""d1: why is the market view (大盤觀點, pbar) bad?  Diagnostics only, nothing is selected here.
(1) reproduce v4's walk-forward reference (v5/research/ref_daily_components.csv) from the exported member predictions
(2) how pbar relates to the Top50 intraday return m_t: correlation, calibration slope, hit rate, exposure, by period / year
(3) which market-level inputs (all known before 09:00) are related to m_t, period by period
run: python d1_market_diag.py > ../results/d1_market_diag.txt"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mlib as L
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)

P = L.panel(); M = L.members(); V = L.views(M)
score = 0.5 * (V.pbar - 0.5) + V.p_e_c
D = L.daily(score)
ref = pd.read_csv(os.path.join(L.ROOT, 'v5', 'research', 'ref_daily_components.csv'), parse_dates=['date']).set_index('date')
j = D.index.intersection(ref.index)
print(f'(1) reproduction of v4 reference: {len(j)} common days; max |AR diff| {np.abs(D.AR[j] - ref.AR[j]).max():.4f}, '
      f'max |timing diff| {np.abs(D.timing[j] - ref.timing[j]).max():.4f}; mean AR {D.AR.mean():.2f}, timing {D.timing.mean():.2f}, xsec {D.xsec.mean():.2f}')

# ---------------------------------------------------------------- (2) market view
mk = L.market_daily(P)
pb = V.groupby(level='date')[['pbar']].first()
pb['pbar_mlp'] = V.groupby(level='date').p_raw_mlp.mean(); pb['pbar_tf'] = V.groupby(level='date').p_raw_tf.mean()
X = pb.join(mk, how='inner').join(D[['sbar', 'timing', 'xsec', 'AR']], how='inner')
X['m_bp'] = X.m * 1e4
def lab_period(ix):
    y = ix.year.astype(str).values.astype(object)
    y[(ix >= '2026-01-01') & (ix < '2026-06-01')] = '2026H1'
    y[ix >= '2026-06-01'] = '2026-06..10'
    return y
X['per'] = lab_period(X.index)
rows = []
for k, g in list(X.groupby('per')) + [('ALL 2022-26', X)]:
    b = np.polyfit(g.pbar - 0.5, g.m_bp, 1)[0]
    rows.append({'period': k, 'days': len(g), 'mean m (bp)': g.m_bp.mean(), 'sd m': g.m_bp.std(), 'mean pbar': g.pbar.mean(), 'sd pbar': g.pbar.std(),
                 'corr(pbar,m)': L.corr(g.pbar, g.m), 'corr mlp': L.corr(g.pbar_mlp, g.m), 'corr tf': L.corr(g.pbar_tf, g.m),
                 'slope bp per 0.01 pbar': b / 100, 'hit sign': np.mean(np.sign(g.pbar - 0.5) == np.sign(g.m)),
                 'mean sbar': g.sbar.mean(), 'sd sbar': g.sbar.std(), 'corr(sbar,m)': L.corr(g.sbar, g.m),
                 'timing': g.timing.mean(), 'timing cov part': ((g.sbar - g.sbar.mean()) * (g.m_bp - g.m_bp.mean())).mean(),
                 'timing drift part': g.sbar.mean() * g.m_bp.mean(), 'xsec': g.xsec.mean()})
T = pd.DataFrame(rows)
print('\n(2) market view by period (m = Top50 mean intraday return; sbar = net exposure 2*share1-1 of the v4 rule)')
print(T.round(3).to_string(index=False))

# exposure mapping: how much of sbar is explained by pbar?  (the stock view's dispersion sets the scale)
disp = V.groupby(level='date').p_e_c.std().rename('disp')
X = X.join(disp)
print(f'\n    corr(pbar, sbar) = {L.corr(X.pbar, X.sbar):.3f};  sd(0.5*(pbar-0.5)) = {(0.5 * (X.pbar - 0.5)).std():.4f} vs mean daily sd of p_e_c = {X.disp.mean():.4f}')
print('    -> the market view moves the share of 1s by roughly Phi(0.5*(pbar-0.5)/disp); a 1-sd move of pbar changes sbar by '
      f'{np.polyfit(X.pbar, X.sbar, 1)[0] * X.pbar.std():.3f}')

# biggest days: what did we do on the 20 largest |m| days of 2026-06..10?
Q = X[X.index >= L.POST].copy(); Q['abs'] = Q.m_bp.abs()
print('\n    2026-06..10, the 15 days with the largest |m|: m (bp), pbar, sbar, timing (bp)')
print(Q.sort_values('abs', ascending=False).head(15)[['m_bp', 'pbar', 'sbar', 'timing', 'xsec']].round(3).to_string())
print(f'    share of post timing coming from these 15 days: {Q.sort_values("abs", ascending=False).head(15).timing.sum() / Q.timing.sum():.2f} '
      f'(sum {Q.sort_values("abs", ascending=False).head(15).timing.sum():.0f} of {Q.timing.sum():.0f} bp-days)')

# ---------------------------------------------------------------- (3) market-level inputs
tx = pd.read_csv(os.path.join(L.ROOT, 'v7', 'data', 'tx_daily.csv'), dtype={'contract_date': str})
tx['date'] = pd.to_datetime(tx.date).astype('datetime64[ns]')
tx = tx[(tx.futures_id == 'TX') & tx.contract_date.str.fullmatch(r'\d{6}')]
day = tx[tx.trading_session == 'position'][['date', 'contract_date', 'open', 'close']]
ngt = tx[(tx.trading_session == 'after_market') & (tx.volume > 0)]
ngt = ngt.sort_values(['date', 'volume'], ascending=[True, False]).drop_duplicates('date')[['date', 'contract_date', 'open', 'max', 'min', 'close']]
dd = np.sort(day.date.unique()); i = np.searchsorted(dd, ngt.date.values) - 1
ngt = ngt[i >= 0].copy(); ngt['prev_day'] = dd[i[i >= 0]]
N = ngt.merge(day.rename(columns={'date': 'prev_day', 'close': 'prev_close', 'open': 'prev_open'}), on=['prev_day', 'contract_date'], how='left')
N['night_ret'] = np.log(N.close / N.prev_close); N['night_range'] = (N['max'] - N['min']) / N.prev_close
N = N.set_index('date')[['night_ret', 'night_range']]
# day-session futures intraday return of the front contract (open->close) as a check of the target's proxy
fr = day.sort_values(['date', 'contract_date']).drop_duplicates('date').set_index('date')
fut_intra = np.log(fr.close / fr.open).rename('fut_intra')

G = P.groupby('date').agg(**{c: (c, 'first') for c in ['us_ixic_ret1', 'us_gspc_ret1', 'us_vix_chg', 'us_vix', 'us_ixic_ret5', 'us_gspc_ret5',
                                                       'twii_ret1', 'twii_ret5', 'ks11_ret1', 'txoi', 'txoi_chg', 'txoi_chg5', 'mkt_intra_prev',
                                                       'breadth_prev', 'mkt_intra_5', 'mkt_intra_20', 'mkt_absr_20', 'cal_dow0', 'cal_dow4',
                                                       'cal_tom', 'cal_month_end', 'cal_settle', 'cal_pre_holiday', 'cal_post_holiday']})
# previous-day market gap and close-to-close of the 100 stocks (shifted features already exist per stock: gap, ret1)
G['mkt_gap_prev'] = P.groupby('date').gap.mean(); G['mkt_ret1_prev'] = P.groupby('date').ret1.mean()
G['mkt_range_prev'] = P.groupby('date')['range'].mean(); G['mkt_clv_prev'] = P.groupby('date').clv.mean()
G['foreign_net_sum'] = P.groupby('date').foreign_net.mean(); G['trust_net_sum'] = P.groupby('date').trust_net.mean()
G = G.join(N).join(mk, how='inner').join(fut_intra.shift(0).rename('fut_intra_same'))   # fut_intra_same: same-day (NOT usable), sanity only
G['night_minus_us'] = G.night_ret - 0.6 * G.us_ixic_ret1
print(f'\n    sanity: corr(Top50 intraday m, TX front-month day-session open->close, same day) = {L.corr(G.m, G.fut_intra_same):.3f} '
      '(the futures market follows the same intraday move)')
feats = ['us_ixic_ret1', 'us_gspc_ret1', 'us_ixic_ret5', 'us_vix_chg', 'us_vix', 'night_ret', 'night_range', 'night_minus_us', 'twii_ret1', 'twii_ret5',
         'ks11_ret1', 'txoi', 'txoi_chg', 'txoi_chg5', 'mkt_intra_prev', 'mkt_intra_5', 'mkt_intra_20', 'mkt_absr_20', 'breadth_prev', 'mkt_gap_prev',
         'mkt_ret1_prev', 'mkt_range_prev', 'mkt_clv_prev', 'foreign_net_sum', 'trust_net_sum', 'cal_dow0', 'cal_dow4', 'cal_tom', 'cal_month_end',
         'cal_settle', 'cal_pre_holiday', 'cal_post_holiday']
pr = L.periods(G.index)
rows = []
for f in feats:
    r = {'feature': f}
    for k in ('pre', 'val', 'hold', 'post'):
        r[k] = L.corr(G[f][pr[k]], G.m[pr[k]])
    r['all'] = L.corr(G[f], G.m)
    r['t_all'] = r['all'] * np.sqrt(G[f].notna().sum() - 2) / np.sqrt(1 - r['all'] ** 2)
    r['same sign 4/4'] = int(np.sign(r['pre']) == np.sign(r['val']) == np.sign(r['hold']) == np.sign(r['post']))
    rows.append(r)
F = pd.DataFrame(rows)
print('\n(3) correlation of each pre-09:00 input with the Top50 intraday return m_t, by period '
      f'(pre 2020-03..21 n={pr["pre"].sum()}, val n={pr["val"].sum()}, hold n={pr["hold"].sum()}, post n={pr["post"].sum()}; |r| > {2 / np.sqrt(pr["val"].sum()):.3f} is ~2 se in val)')
print(F.round(3).to_string(index=False))
G.to_pickle(os.path.join(L.CACHE, 'daily_market.pkl'))
X.to_pickle(os.path.join(L.CACHE, 'daily_view.pkl'))

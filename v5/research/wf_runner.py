r"""
wf_runner.py -- walk-forward runs for the v5 experiments (MLP members only; resumable; parallel processes)

  python wf_runner.py --scheme year  --variant orig sig_a0 sig_a05 sig_a1 --seeds 1-5 --workers 2
  python wf_runner.py --scheme month --variant orig --seeds 1-5 --workers 2
  python wf_runner.py --scheme year  --variant drop_us_nasdaq drop_us_other drop_us_beta drop_tw_market drop_flows
                      drop_intra_bias drop_price_tech drop_sector drop_us_nasdaq_all --seeds 1-3 --workers 2
  (these three commands = the 2,288 models behind s9, s10, s11; about 1-2 hours)

scheme  year  : test folds 2022, 2023, 2024, 2025, 2026-01..05 (expanding window from 2020-03-02, train up to the day
                before the fold) + post 2026-06-01.. predicted by models trained up to 2026-05-29 (= overnight protocol)
        month : one test fold per calendar month 2022-01 .. 2026-10, each predicted by models trained up to the last
                trading day before the month (expanding window) -> "retrain every month" incl. the post months
variant orig    : v2/v4 training exactly (BCE weighted by min(|r|, 5%))
        sig_aX  : weight min(|r|, 5%) / sigma^X and sigma added as inputs (X = 0, 0.5, 1); sigma = trailing 60-day mean
                  of min(|r|, 5%) of the stock (raw target) or of |r - same-day mean| (residual target), known at t-1
        drop_G  : the features of group G (GROUPS below) and their xr_ ranks removed from the inputs, training as orig
Each unit is saved to cache\wf5\<member>__<variant>__<scheme>\<fold start>_seed<s>.pkl as soon as it finishes.
"""
import os, sys, time, math, copy, argparse, traceback
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rlib as L
E = L.E
OUT = os.path.join(L.CACHE, 'wf5')
MEMBERS = ['mlp_raw_B', 'mlp_raw_xr', 'mlp_res_B', 'mlp_res_xr']
SIG_COLS = ['absr_60', 'absres_60']
# input-feature groups of the 67 B features (drop-one-group ablation: variant drop_<group>; the matching xr_ rank
# features are dropped from the B+XR members as well)
GROUPS = {
    'us_nasdaq': ['us_ixic_ret1', 'us_ixic_ret5', 'us_ixic_ret1_lag1'],
    'us_other': ['us_vix', 'us_vix_chg', 'us_vix_chg_lag1', 'us_gspc_ret1', 'us_gspc_ret5', 'us_gspc_ret1_lag1'],
    'us_beta': ['beta_us', 'beta_x_us', 'beta_x_us_lag1'],
    'tw_market': ['twii_ret1', 'twii_ret5', 'ks11_ret1', 'ks11_ret5', 'txoi', 'txoi_chg', 'txoi_chg5', 'mkt_intra_prev',
                  'breadth_prev', 'mkt_intra_5', 'sector_intra_prev', 'rel_intra_prev'],
    'flows': ['foreign_hold', 'foreign_hold_chg1', 'foreign_hold_chg5', 'foreign_net', 'foreign_net_5d', 'trust_net',
              'trust_net_5d', 'dealer_self_net', 'dealer_self_net_5d', 'dealer_hedge_net', 'dealer_hedge_net_5d', 'foreign_streak'],
    'intra_bias': ['intra_bias_20', 'intra_bias_60', 'intra_bias_250', 'intra_bias_exp', 'gap_bias_20'],
    'price_tech': ['ret1', 'ret5', 'ret20', 'intraday_ret', 'gap', 'y_mean5', 'range', 'clv', 'upper_shadow', 'lower_shadow',
                   'vol_ratio', 'amt_ratio', 'trades_ratio', 'avg_trade_size', 'RSI', 'close_SMA30', 'MACD_hist_norm', 'ATR_pct',
                   'BB_pctb', 'BB_width', 'vol_ratio_5_60', 'limit_up_prev', 'limit_dn_prev', 'streak_y'],
    'sector': ['group', 'is_fin'],
    # every input that carries Nasdaq information (us_nasdaq + us_beta: beta_us is estimated on Nasdaq returns and
    # beta_x_us = beta_us * Nasdaq return), added after the user's linear-model finding that Nasdaq and S&P get
    # opposite signs
    'us_nasdaq_all': ['us_ixic_ret1', 'us_ixic_ret5', 'us_ixic_ret1_lag1', 'beta_us', 'beta_x_us', 'beta_x_us_lag1']}

def add_sigma(P):
    if 'absr_60' in P:
        return P
    P = P.sort_values(['stock_id', 'date']).copy()
    a = P.target_r.abs().clip(upper=0.05)
    P['absr_60'] = a.groupby(P.stock_id).transform(lambda s: s.rolling(60, min_periods=20).mean().shift(1))
    return P.sort_values(['date', 'stock_id']).reset_index(drop=True)

def folds(scheme, last_label):
    if scheme == 'year':
        return [(a, b, pd.Timestamp(a) - pd.Timedelta(days=1)) for a, b in L.YEAR_FOLDS] + \
               [('2026-06-01', '2100-01-01', pd.Timestamp('2026-05-31'))]
    out = []
    for m in pd.period_range('2022-01', last_label.to_period('M'), freq='M'):
        a = m.start_time; b = (m + 1).start_time
        out.append((str(a.date()), str(b.date()), a - pd.Timedelta(days=1)))
    return out

def train_custom(Q, feats, train_end, seed, alpha, sig_col):
    """tfmodel.train_model for the MLP with weights min(|r|, 5%) / sigma^alpha (alpha None -> v2 weights exactly)"""
    import torch, tfmodel as TM
    hp = {**dict(d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.2, lr=1e-3, wd=1e-2, bs=512, epochs=15, patience=3), **E.HP}
    torch.manual_seed(seed); np.random.seed(seed)
    D = TM.SeqData(Q, feats, 1)
    te = pd.Timestamp(train_end)
    tm = ((D.P.date >= E.TRAIN_START) & (D.P.date <= te)).values & D.P.target_r.notna().values
    scaler = TM.Scaler().fit(D.raw[tm]); D.scale(scaler)
    idx_all = D.valid(tm); dates = D.P.date.values[idx_all]
    ud = np.sort(np.unique(dates)); cut = ud[-120] if len(ud) > 240 else ud[-1]
    y = D.P.y.values.astype(np.float32)
    if alpha is None:
        w = TM._weights(D.P.target_r.fillna(0).values)
    else:
        w = np.minimum(np.abs(D.P.target_r.fillna(0).values), 0.05)
        sig = D.P[sig_col].values.astype(float)
        med = np.nanmedian(sig[idx_all])
        sig = np.where(np.isfinite(sig) & (sig > 0), sig, med)
        w = w / sig ** alpha
        w = (w / w[idx_all].mean()).astype(np.float32)
    tr, va = idx_all[dates < cut], idx_all[dates >= cut]
    model = TM.make_model('mlp', len(feats), 1, dropout=hp['dropout'])
    model, best_ep = TM._fit(model, D, tr, y[tr], w[tr], hp['epochs'], hp['lr'], hp['wd'], hp['bs'], seed, val=(va, y[va], w[va]), patience=hp['patience'])
    best_ep = max(best_ep, 1)
    torch.manual_seed(seed)
    model = TM.make_model('mlp', len(feats), 1, dropout=hp['dropout'])
    model, _ = TM._fit(model, D, idx_all, y[idx_all], w[idx_all], best_ep, hp['lr'], hp['wd'], hp['bs'], seed)
    return dict(model=model.eval(), scaler=scaler, kind='mlp', seq_len=1, features=feats, hp=hp, best_epoch=best_ep)

def unit_path(name, variant, scheme, a, s):
    return os.path.join(OUT, f'{name}__{variant}__{scheme}', f'{a}_seed{s}.pkl')

_G = {}
def run_unit(u):
    name, variant, scheme, a, b, te, s = u
    f = unit_path(name, variant, scheme, a, s)
    if os.path.exists(f):
        return (u, 0.0, None)
    t0 = time.time()
    try:
        import torch, tfmodel as TM
        torch.set_num_threads(1)
        P = _G['P']; m = L.member_def(name)
        feats = list(_G['fs'][m['feats']])
        Q = E.target_panel(P[P.date <= max(pd.Timestamp(b), te)], m['target'])
        Qtr = Q[Q.date <= te]
        if variant == 'orig':
            bd = train_custom(Qtr, feats, te, s, None, None)
        elif variant.startswith('drop_'):
            gone = set(GROUPS[variant[5:]]); gone |= {'xr_' + f for f in gone}
            bd = train_custom(Qtr, [f for f in feats if f not in gone], te, s, None, None)
        else:
            alpha = {'sig_a0': 0.0, 'sig_a05': 0.5, 'sig_a1': 1.0}[variant]
            bd = train_custom(Qtr, feats + SIG_COLS, te, s, alpha, 'absr_60' if m['target'] == 'raw' else 'absres_60')
        a_, b_ = pd.Timestamp(a), pd.Timestamp(b)
        o = TM.predict(bd, Q, lambda d: (d.date >= a_) & (d.date < b_) & d.target_r.notna())[['date', 'stock_id', 'p']]
        os.makedirs(os.path.dirname(f), exist_ok=True); o.to_pickle(f + '.tmp'); os.replace(f + '.tmp', f)
        return (u, time.time() - t0, None)
    except Exception:
        return (u, time.time() - t0, traceback.format_exc())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scheme', default='year', choices=['year', 'month'])
    ap.add_argument('--variant', nargs='+', default=['orig'])
    ap.add_argument('--members', nargs='+', default=MEMBERS)
    ap.add_argument('--seeds', default='1-5')
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split('-')); seeds = range(lo, hi + 1)
    P = add_sigma(L.panel()); _G['P'] = P; _G['fs'] = E.feature_sets()
    last = P[P.target_r.notna()].date.max()
    units = [(n, v, a.scheme, fa, fb, te, s) for v in a.variant for n in a.members for fa, fb, te in folds(a.scheme, last) for s in seeds]
    todo = [u for u in units if not os.path.exists(unit_path(u[0], u[1], u[2], u[3], u[6]))]
    print(f'{len(units)} units, {len(todo)} to do, workers {a.workers}', flush=True)
    t0 = time.time(); done = 0
    import multiprocessing as mp
    with mp.get_context('fork').Pool(a.workers) as pool:
        for u, sec, err in pool.imap_unordered(run_unit, todo):
            done += 1
            if err:
                print('FAILED', u, err, flush=True)
            elif done % 20 == 0 or done == len(todo):
                print(f'[{done}/{len(todo)}] {u[0]} {u[1]} {u[3]} s{u[6]} {sec:.1f}s; elapsed {(time.time() - t0) / 60:.1f} min', flush=True)

if __name__ == '__main__':
    main()

"""m1: daily market model (PREREG_m.md, T2/T3): ridge on 32 pre-09:00 market-level inputs -> forecast of the Top50 mean
intraday return mu_t, out of sample (expanding window from 2020-03-02, refit on the first trading day of every month,
lambda chosen inside the training window on its last 250 days).  Writes ../cache/mu_ridge.pkl.
Two versions: 'all' (32 inputs incl. the TX night session) and 'comp' (29 inputs, competition data only).
run: python m1_market_model.py > ../results/m1_market_model.txt"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mlib as L
pd.set_option('display.width', 250)
G = pd.read_pickle(os.path.join(L.CACHE, 'daily_market.pkl'))
FEATS = ['us_ixic_ret1', 'us_gspc_ret1', 'us_ixic_ret5', 'us_vix_chg', 'us_vix', 'night_ret', 'night_range', 'night_minus_us', 'twii_ret1',
         'twii_ret5', 'ks11_ret1', 'txoi', 'txoi_chg', 'txoi_chg5', 'mkt_intra_prev', 'mkt_intra_5', 'mkt_intra_20', 'mkt_absr_20',
         'breadth_prev', 'mkt_gap_prev', 'mkt_ret1_prev', 'mkt_range_prev', 'mkt_clv_prev', 'foreign_net_sum', 'trust_net_sum', 'cal_dow0',
         'cal_dow4', 'cal_tom', 'cal_month_end', 'cal_settle', 'cal_pre_holiday', 'cal_post_holiday']
NIGHT = ['night_ret', 'night_range', 'night_minus_us']
LAMBDAS = [1, 3, 10, 30, 100, 300, 1000, 3000]
START = pd.Timestamp('2020-03-02'); FIRST_OOS = pd.Timestamp('2021-03-01')
G = G[G.index >= START].copy()
y_all = G.m.clip(-0.03, 0.03)

class Ridge:
    """ridge with an unpenalised intercept (same as sklearn.linear_model.Ridge(alpha=lam)), numpy only"""
    def __init__(self, alpha):
        self.alpha = alpha
    def fit(self, Z, y):
        zm, ym = Z.mean(axis=0), y.mean()
        Zc = Z - zm
        self.coef_ = np.linalg.solve(Zc.T @ Zc + self.alpha * np.eye(Z.shape[1]), Zc.T @ (y - ym))
        self.intercept_ = ym - zm @ self.coef_
        return self
    def predict(self, Z):
        return Z @ self.coef_ + self.intercept_

def fit(X, y):
    mu, sd = X.mean(), X.std().replace(0, 1)
    Z = ((X - mu) / sd).fillna(0.0).values
    n = len(Z); cut = n - 250
    best, bl = None, np.inf
    for lam in LAMBDAS:
        r = Ridge(lam).fit(Z[:cut], y[:cut])
        e = np.mean((r.predict(Z[cut:]) - y[cut:]) ** 2)
        if e < bl:
            bl, best = e, lam
    r = Ridge(best).fit(Z, y)
    return r, mu, sd, best

out = {}
for ver, feats in (('all', FEATS), ('comp', [f for f in FEATS if f not in NIGHT])):
    days = G.index
    months = sorted(set(d.to_period('M') for d in days if d >= FIRST_OOS))
    preds, lams, coefs = [], [], []
    for mo in months:
        a = mo.start_time; b = (mo + 1).start_time
        tr = days[days < a]; te = days[(days >= a) & (days < b)]
        if len(te) == 0:
            continue
        X = G.loc[tr, feats]; y = y_all.loc[tr].values
        k = ~np.isnan(y)
        r, mu, sd, lam = fit(X[k], y[k])
        Zt = ((G.loc[te, feats] - mu) / sd).fillna(0.0).values
        preds.append(pd.Series(r.predict(Zt), index=te)); lams.append((str(mo), lam)); coefs.append(pd.Series(r.coef_, index=feats, name=str(mo)))
    mu_hat = pd.concat(preds).rename(f'mu_{ver}')
    out[ver] = mu_hat
    C = pd.DataFrame(coefs)
    print(f'[{ver}] {len(feats)} inputs; OOS forecasts {mu_hat.index.min().date()}..{mu_hat.index.max().date()} ({len(mu_hat)} days)')
    print('   lambda chosen (count): ' + ', '.join(f'{l}: {c}' for l, c in pd.Series([l for _, l in lams]).value_counts().sort_index().items()))
    pr = L.periods(mu_hat.index); m = G.m.reindex(mu_hat.index)
    print('   corr(mu_hat, m): ' + ', '.join(f'{k} {L.corr(mu_hat[v], m[v]):.3f} (n {v.sum()})' for k, v in pr.items() if v.sum() > 20) +
          f'; 2021 (pre, OOS) {L.corr(mu_hat[mu_hat.index < "2022-01-01"], m[mu_hat.index < "2022-01-01"]):.3f}')
    print(f'   sd(mu_hat) {mu_hat.std() * 1e4:.2f} bp, mean {mu_hat.mean() * 1e4:.2f} bp; sd(m) {m.std() * 1e4:.1f} bp')
    last = C.iloc[-1].sort_values(key=abs, ascending=False)
    print('   largest standardised coefficients in the last refit (bp per sd): ' + ', '.join(f'{f} {v * 1e4:+.2f}' for f, v in last.head(10).items()))
    sgn = np.sign(C).mean().sort_values()
    print('   share of monthly refits with a positive coefficient, most stable: ' +
          ', '.join(f'{f} {(C[f] > 0).mean():.2f}' for f in list(sgn.index[:5]) + list(sgn.index[-5:])))
M = pd.concat(out.values(), axis=1)
M.to_pickle(os.path.join(L.CACHE, 'mu_ridge.pkl')); M.to_csv(os.path.join(L.CACHE, 'mu_ridge.csv'), float_format='%.8f')
X = pd.read_pickle(os.path.join(L.CACHE, 'daily_view.pkl'))
J = M.join(X[['pbar']], how='inner')
print(f'\ncorr(mu_all, pbar) 2022-26: {L.corr(J.mu_all, J.pbar):.3f}; corr(mu_comp, pbar): {L.corr(J.mu_comp, J.pbar):.3f}')

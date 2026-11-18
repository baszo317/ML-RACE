"""t5_eval: the pre-registered T5 test (PREREG_m.md, second round).
control  : v4 rule, stock view = cached mlp_res_B / mlp_res_xr seeds 1-5 (x1 export), market view = cached raw members
T5       : stock view retrained on the beta-adjusted residual e = r - beta*m (t5_train, target 'beta'), market term scaled by beta_i/mean beta
check    : t5_train target 'orig' must reproduce the cached seeds 1-5 (same code as the cache)
uses ../cache/t5/ if present (and writes ../cache/t5_preds.csv.gz), otherwise ../cache/t5_preds.csv.gz
run: python t5_eval.py > ../results/t5_eval.txt"""
import os, sys, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mlib as L
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)

def load(member, target):
    fs = sorted(glob.glob(os.path.join(L.CACHE, 't5', f'{member}__{target}', '*.pkl')))
    parts = {}
    for f in fs:
        fold = os.path.basename(f).split('_seed')[0]
        x = pd.read_pickle(f); x['date'] = pd.to_datetime(x.date).astype('datetime64[ns]'); x['stock_id'] = x.stock_id.astype(str)
        parts.setdefault(fold, []).append(x.set_index(['date', 'stock_id']).p)
    assert all(len(v) == 5 for v in parts.values()) and len(parts) == 6, {k: len(v) for k, v in parts.items()}
    return pd.concat([pd.concat(v, axis=1).mean(axis=1) for v in parts.values()]).sort_index()

CSV = os.path.join(L.CACHE, 't5_preds.csv.gz')            # seed means of t5_train's output (so this runs without retraining)
if os.path.exists(os.path.join(L.CACHE, 't5')) and glob.glob(os.path.join(L.CACHE, 't5', '*', '*.pkl')):
    T5P = pd.DataFrame({f'{n}__{t}': load(n, t) for t in ('orig', 'beta') for n in ['mlp_res_B', 'mlp_res_xr']})
    T5P.reset_index().to_csv(CSV, index=False, float_format='%.7f')
else:
    T5P = pd.read_csv(CSV, dtype={'stock_id': str}, parse_dates=['date']); T5P['date'] = T5P.date.astype('datetime64[ns]')
    T5P = T5P.set_index(['date', 'stock_id']).sort_index()
M = L.members(); V = L.views(M)
C5 = pd.read_csv(os.path.join(L.CACHE, 'wf_res_seed1_5.csv.gz'), dtype={'stock_id': str}, parse_dates=['date'])
C5['date'] = C5.date.astype('datetime64[ns]'); C5 = C5.set_index(['date', 'stock_id']).sort_index()
orig = pd.DataFrame({n: T5P[f'{n}__orig'] for n in ['mlp_res_B', 'mlp_res_xr']})
beta = pd.DataFrame({n: T5P[f'{n}__beta'] for n in ['mlp_res_B', 'mlp_res_xr']})
j = orig.index.intersection(C5.index)
print(f'check: retrained target "orig" vs cached seeds 1-5: {len(j)} rows, max |diff| B {np.abs(orig.mlp_res_B[j] - C5.mlp_res_B[j]).max():.2e}, '
      f'xr {np.abs(orig.mlp_res_xr[j] - C5.mlp_res_xr[j]).max():.2e}')

Bp = pd.read_pickle(os.path.join(L.CACHE, 'beta_panel.pkl'))
F = V[['pbar']].join(C5.rename(columns=lambda c: c + '_c5')).join(beta.rename(columns=lambda c: c + '_bt'))
F = F.reset_index().merge(Bp[['date', 'stock_id', 'beta']], on=['date', 'stock_id'], how='left')
F['beta'] = F.beta.clip(0.2, 2.5); F['beta'] = F.beta.fillna(F.groupby('date').beta.transform('median')).fillna(1.0)
F['bbar'] = F.groupby('date').beta.transform('mean')
F['pe_c5'] = F[['mlp_res_B_c5', 'mlp_res_xr_c5']].mean(axis=1); F['pe_c5'] -= F.groupby('date').pe_c5.transform('mean')
F['pe_bt'] = F[['mlp_res_B_bt', 'mlp_res_xr_bt']].mean(axis=1); F['pe_bt'] -= F.groupby('date').pe_bt.transform('mean')
F = F.set_index(['date', 'stock_id'])
ctrl_score = 0.5 * (F.pbar - 0.5) + F.pe_c5
t5_score = 0.5 * (F.pbar - 0.5) * F.beta / F.bbar + F.pe_bt

def run(score):
    D = L.daily(score)
    q = score.rename('score').reset_index().merge(Bp[['date', 'stock_id', 'top50', 'target_r', 'ex_today', 'beta']], on=['date', 'stock_id'])
    q = q[q.top50 & q.target_r.notna() & q.beta.notna()].copy()
    pred = np.where(q.ex_today == 1, 0, (q.score > 0).astype(int)); q['s'] = 2 * pred - 1
    g = q.groupby('date'); q['sb'] = g.s.transform('mean'); q['bb'] = g.beta.transform('mean')
    D['B'] = ((q.s - q.sb) * (q.beta - q.bb)).groupby(q.date).mean().reindex(D.index)
    D['beta_bet'] = D.B * D.m; D['rest'] = D.xsec - D.beta_bet
    return D
base, t5 = run(ctrl_score), run(t5_score)
rows = []
for lab, D, b in (('control: v4 rule, res seeds 1-5 (cache)', base, None), ('T5: beta-adjusted residual target + beta-scaled market term', t5, base)):
    r = L.table(D, b, lab); pr = L.periods(D.index)
    r.update({'beta_bet': D.beta_bet.mean(), 'rest': D.rest.mean(), 'mean B': D.B.mean(), 'sd_beta_bet': D.beta_bet.std(),
              'corr(xsec,m)': L.corr(D.xsec, D.m), 'P19>0': (D.AR.rolling(19).mean().dropna() > 0).mean()})
    rows.append(r)
T = pd.DataFrame(rows)
cols = ['config', 'val', 'hold', 'post', 'all22', 'sd', 'timing', 'xsec', 'beta_bet', 'rest', 'mean B', 'sd_beta_bet', 'corr(xsec,m)', 'P19>0',
        'd_val', 't_val', 'd_hold', 't_hold', 'd_post', 't_post']
print('\nTop50, bp/day, walk-forward 2022-01..2026-10; d_/t_ = paired difference to the control')
print(T[[c for c in cols if c in T]].round(3).to_string(index=False))
r = T.iloc[1]; ok = r.t_val >= 2.0 and r.d_hold > 0
print(f'\npre-registered decision: val t = {r.t_val:.2f} (needs >= 2.0), hold diff = {r.d_hold:+.2f} (needs > 0) -> {"ADOPT" if ok else "do not adopt"}')
for lab, a, b in (('static 2026-06-01..09-30', '2026-06-01', '2026-09-30'), ('live 2026-10-01..10-22', '2026-10-01', '2026-10-22')):
    m = (base.index >= a) & (base.index <= b); d = (t5.AR - base.AR)[m]
    print(f'for information, {lab}: control {base.AR[m].mean():.2f} (beta_bet {base.beta_bet[m].mean():.2f}), T5 {t5.AR[m].mean():.2f} '
          f'(beta_bet {t5.beta_bet[m].mean():.2f}), diff {d.mean():+.2f} (t {L.plain_t(d):.2f}, {m.sum()} days)')
# member quality: rank IC of the stock view against the beta-adjusted residual and the plain residual, Top50
P = L.panel()
Z = F[['pe_c5', 'pe_bt', 'beta']].reset_index().merge(P[['date', 'stock_id', 'top50', 'target_r', 'resid_r', 'mkt_r']], on=['date', 'stock_id'])
Z = Z[Z.top50 & Z.target_r.notna()]; Z['e_beta'] = Z.target_r - Z.beta * Z.mkt_r
def ic(a, b):
    return Z.groupby('date')[[a, b]].apply(lambda g: g[a].rank().corr(g[b].rank()))
for tgt in ('resid_r', 'e_beta'):
    i1, i2 = ic('pe_c5', tgt), ic('pe_bt', tgt); pr = L.periods(i1.index)
    print(f'daily rank IC vs {tgt:8s}: control ' + ', '.join(f'{k} {i1[m].mean():.3f}' for k, m in pr.items() if m.sum()) +
          ' | T5 ' + ', '.join(f'{k} {i2[m].mean():.3f}' for k, m in pr.items() if m.sum()))
pd.to_pickle({'base': base, 't5': t5}, os.path.join(L.CACHE, 't5_daily.pkl'))

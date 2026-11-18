"""g2_tab_eval: the pre-registered test of the sixth round (PREREG_m.md): LightGBM and logistic regression in the market view
(instead of / next to the Transformer) and in the stock view (next to the two MLPs).
uses ../cache/tab/ if present (and writes ../cache/tab_preds.csv.gz), otherwise ../cache/tab_preds.csv.gz
run: python g2_tab_eval.py > ../results/g2_tab_eval.txt"""
import os, sys, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mlib as L
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
CSV = os.path.join(L.CACHE, 'tab_preds.csv.gz')
NAMES = {'lgb_raw': 5, 'lgb_resid': 5, 'lr_raw': 1, 'lr_resid': 1}

def load(name, nseed):
    parts = {}
    for f in sorted(glob.glob(os.path.join(L.CACHE, 'tab', name, '*.pkl'))):
        x = pd.read_pickle(f); x['date'] = pd.to_datetime(x.date).astype('datetime64[ns]'); x['stock_id'] = x.stock_id.astype(str)
        parts.setdefault(os.path.basename(f).split('_seed')[0], []).append(x.set_index(['date', 'stock_id']).p)
    assert len(parts) == 6 and all(len(v) == nseed for v in parts.values()), (name, {k: len(v) for k, v in parts.items()})
    return pd.concat([pd.concat(v, axis=1).mean(axis=1) for v in parts.values()]).sort_index()

if glob.glob(os.path.join(L.CACHE, 'tab', '*', '*.pkl')):
    TB = pd.DataFrame({n: load(n, k) for n, k in NAMES.items()})
    TB.reset_index().to_csv(CSV, index=False, float_format='%.7f')
else:
    TB = pd.read_csv(CSV, dtype={'stock_id': str}, parse_dates=['date']); TB['date'] = TB.date.astype('datetime64[ns]')
    TB = TB.set_index(['date', 'stock_id']).sort_index()

M = L.members()
idx = M.index.intersection(TB.dropna().index)
print(f'rows: members {len(M)}, tabular {len(TB.dropna())}; common {len(idx)}')
M, TB = M.loc[idx], TB.loc[idx]
mlp = M[['mlp_raw_B', 'mlp_raw_xr']].mean(axis=1); tf = M.tf_raw_B
res_mlp = M[['mlp_res_B', 'mlp_res_xr']].mean(axis=1)
def centre(p):
    return p - p.groupby(level='date').transform('mean')
def pbar_of(p_raw):
    return p_raw.groupby(level='date').transform('mean')
PB0 = pbar_of((mlp + tf) / 2)
mk = L.market_daily()

def run(pbar, p_e):
    return L.daily(0.5 * (pbar - 0.5) + centre(p_e))
def report(variants, kind, thr, base_key):
    R = {k: run(*v) for k, v in variants.items()}
    base = R[base_key]
    rows = []
    for k, D in R.items():
        r = L.table(D, None if k == base_key else base, k); r['P19>0'] = (D.AR.rolling(19).mean().dropna() > 0).mean(); rows.append(r)
    T = pd.DataFrame(rows)
    cols = ['config', 'val', 'hold', 'post', 'all22', 'sd', 'timing', 'xsec', 'P19>0', 'd_val', 't_val', 'd_hold', 't_hold', 'd_post', 't_post']
    print(f'\n{kind}: Top50 bp/day, walk-forward 2022-01..2026-10; d_/t_ = paired difference to the control')
    print(T[cols].round(3).to_string(index=False))
    tr = T[T.config.str.match(r'[LGS]\d')].sort_values('t_val', ascending=False).reset_index(drop=True); alive = True
    print(f'Holm ({len(tr)} tests) and hold > 0:')
    for i, r in tr.iterrows():
        ok = alive and r.t_val >= thr[i] and r.d_hold > 0; alive = alive and r.t_val >= thr[i]
        print(f'  {r.config:36s} val t {r.t_val:+.2f} (needs {thr[i]:.2f}), hold diff {r.d_hold:+.2f} -> {"ADOPT" if ok else "do not adopt"}')
    for lab, a, b in (('static 2026-06-01..09-30', '2026-06-01', '2026-09-30'), ('live 2026-10-01..10-22', '2026-10-01', '2026-10-22')):
        print(f'  for information, {lab}: ' + '; '.join(f'{k.split(" ")[0]} {D.AR[(D.index >= a) & (D.index <= b)].mean():.2f}' for k, D in R.items()))
    return R

# ---------------------------------------------------------------- market view (stock view fixed = the two MLPs)
mv = {'control: (MLP + Transformer) / 2': (PB0, res_mlp),
      'L1 (MLP + LightGBM) / 2': (pbar_of((mlp + TB.lgb_raw) / 2), res_mlp),
      'L2 (MLP + Transformer + LightGBM) / 3': (pbar_of((mlp + tf + TB.lgb_raw) / 3), res_mlp),
      'G1 (MLP + logistic) / 2': (pbar_of((mlp + TB.lr_raw) / 2), res_mlp),
      'G2 (MLP + Transformer + logistic) / 3': (pbar_of((mlp + tf + TB.lr_raw) / 3), res_mlp),
      '(info) LightGBM only': (pbar_of(TB.lgb_raw), res_mlp),
      '(info) logistic only': (pbar_of(TB.lr_raw), res_mlp)}
report(mv, 'MARKET VIEW', [2.24, 2.13, 1.96, 1.645], 'control: (MLP + Transformer) / 2')
print('\ncorrelation of each daily pbar with the Top50 intraday return m (val / hold / post / all):')
for k, (pb, _) in mv.items():
    d = pb.groupby(level='date').first(); j = d.index.intersection(mk.index); pr = L.periods(j)
    print(f'  {k:38s} ' + ' / '.join(f'{L.corr(d[j][v], mk.m[j][v]):+.3f}' for p, v in pr.items() if p in ('val', 'hold', 'post', 'all22')) +
          f'   sd(pbar) {d.std():.4f}')

# ---------------------------------------------------------------- stock view (market view fixed = MLP + Transformer)
sv = {'control: p_e = MLP residual members': (PB0, res_mlp),
      'S1 p_e = (MLP + LightGBM) / 2': (PB0, (res_mlp + TB.lgb_resid) / 2),
      'S2 p_e = (MLP + logistic) / 2': (PB0, (res_mlp + TB.lr_resid) / 2),
      '(info) LightGBM only': (PB0, TB.lgb_resid),
      '(info) logistic only': (PB0, TB.lr_resid)}
report(sv, 'STOCK VIEW', [1.96, 1.645], 'control: p_e = MLP residual members')
P = L.panel().set_index(['date', 'stock_id'])
Z = pd.DataFrame({'resid_r': P.resid_r.reindex(idx), 'top50': P.top50.reindex(idx)})
Z = Z[Z.top50.astype(bool) & Z.resid_r.notna()]
print('\ndaily rank IC of the stock view vs the residual intraday return, Top50 (val / hold / post / all):')
for k, (_, pe) in sv.items():
    q = pd.DataFrame({'s': pe.reindex(Z.index), 'y': Z.resid_r})
    ic = q.groupby(level='date').apply(lambda g: g.s.rank().corr(g.y.rank()))
    pr = L.periods(ic.index)
    print(f'  {k:38s} ' + ' / '.join(f'{ic[v].mean():.3f}' for p, v in pr.items() if p in ('val', 'hold', 'post', 'all22')))

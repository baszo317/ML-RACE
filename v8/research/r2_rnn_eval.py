"""r2_rnn_eval: the pre-registered test of the fifth round (PREREG_m.md): a GRU instead of (R1) or next to (R2) the Transformer
in the market view.  uses ../cache/rnn/ if present (and writes ../cache/rnn_preds.csv.gz), otherwise ../cache/rnn_preds.csv.gz
run: python r2_rnn_eval.py > ../results/r2_rnn_eval.txt"""
import os, sys, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mlib as L
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)

CSV = os.path.join(L.CACHE, 'rnn_preds.csv.gz')
fs = sorted(glob.glob(os.path.join(L.CACHE, 'rnn', '*.pkl')))
if fs:
    parts = {}
    for f in fs:
        x = pd.read_pickle(f); x['date'] = pd.to_datetime(x.date).astype('datetime64[ns]'); x['stock_id'] = x.stock_id.astype(str)
        parts.setdefault(os.path.basename(f).split('_seed')[0], []).append(x.set_index(['date', 'stock_id']).p)
    assert len(parts) == 6 and all(len(v) == 5 for v in parts.values()), {k: len(v) for k, v in parts.items()}
    cnn = pd.concat([pd.concat(v, axis=1).mean(axis=1) for v in parts.values()]).sort_index().rename('rnn_raw_B')
    cnn.reset_index().to_csv(CSV, index=False, float_format='%.7f')
else:
    c = pd.read_csv(CSV, dtype={'stock_id': str}, parse_dates=['date']); c['date'] = c.date.astype('datetime64[ns]')
    cnn = c.set_index(['date', 'stock_id']).rnn_raw_B

M = L.members(); V = L.views(M)
idx = M.index.intersection(cnn.index)
M = M.loc[idx]; V = V.loc[idx]; cnn = cnn.loc[idx]
print(f'rows: members {len(L.members())}, GRU {len(cnn)}; common {len(idx)}')
mlp = M[['mlp_raw_B', 'mlp_raw_xr']].mean(axis=1); tf = M.tf_raw_B
variants = {'control: (MLP + Transformer) / 2': (mlp + tf) / 2, 'R1 (MLP + GRU) / 2': (mlp + cnn) / 2,
            'R2 (MLP + Transformer + GRU) / 3': (mlp + tf + cnn) / 3, '(info) Transformer only': tf, '(info) GRU only': cnn}
mk = L.market_daily()
R, PB = {}, {}
for k, p_raw in variants.items():
    pbar = p_raw.groupby(level='date').transform('mean')
    R[k] = L.daily(0.5 * (pbar - 0.5) + V.p_e_c)
    PB[k] = p_raw.groupby(level='date').mean()
base = R['control: (MLP + Transformer) / 2']
rows = []
for k, D in R.items():
    r = L.table(D, None if k.startswith('control') else base, k)
    r['P19>0'] = (D.AR.rolling(19).mean().dropna() > 0).mean()
    rows.append(r)
T = pd.DataFrame(rows)
cols = ['config', 'val', 'hold', 'post', 'all22', 'sd', 'timing', 'xsec', 'P19>0', 'd_val', 't_val', 'd_hold', 't_hold', 'd_post', 't_post']
print('\nv4 rule with each market view, Top50 bp/day, walk-forward 2022-01..2026-10; d_/t_ = paired difference to the control')
print(T[cols].round(3).to_string(index=False))
tr = T[T.config.str.match(r'R\d')].sort_values('t_val', ascending=False).reset_index(drop=True); thr = [1.96, 1.645]; alive = True
print('\nHolm (2 tests) and hold > 0:')
for i, r in tr.iterrows():
    ok = alive and r.t_val >= thr[i] and r.d_hold > 0; alive = alive and r.t_val >= thr[i]
    print(f'  {r.config:34s} val t {r.t_val:+.2f} (needs {thr[i]:.2f}), hold diff {r.d_hold:+.2f} -> {"ADOPT" if ok else "do not adopt"}')
print('\ncorrelation of each pbar with the Top50 intraday return m:')
for k, pb in PB.items():
    j = pb.index.intersection(mk.index); pr = L.periods(j)
    print(f'  {k:34s} ' + ', '.join(f'{p} {L.corr(pb[j][v], mk.m[j][v]):+.3f}' for p, v in pr.items() if v.sum()))
pm = (M[['mlp_raw_B', 'mlp_raw_xr']].mean(axis=1)).groupby(level='date').mean(); pt, pc = PB['(info) Transformer only'], PB['(info) GRU only']
print(f'\ncorrelation between the market views (daily pbar): MLP-Transformer {L.corr(pm, pt):.3f}, MLP-GRU {L.corr(pm, pc):.3f}, Transformer-GRU {L.corr(pt, pc):.3f}')
for lab, a, b in (('static 2026-06-01..09-30', '2026-06-01', '2026-09-30'), ('live 2026-10-01..10-22', '2026-10-01', '2026-10-22')):
    print(f'\nfor information, {lab}:')
    for k, D in R.items():
        m_ = (D.index >= a) & (D.index <= b); dd = (D.AR - base.AR.reindex(D.index))[m_]
        print(f'  {k:34s} AR {D.AR[m_].mean():7.2f}  timing {D.timing[m_].mean():6.2f}  diff {dd.mean():+6.2f} (t {L.plain_t(dd):.2f})')

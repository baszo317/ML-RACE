"""k2_seq_eval: the pre-registered test of the seventh round (PREREG_m.md): LSTM in the market view (K1, K2), GRU and LSTM in the
stock view (S3, S4); the fifth-round GRU market-view results and the sixth-round LightGBM stock view are shown for comparison.
uses ../cache/seq/ if present (and writes ../cache/seq_preds.csv.gz), otherwise ../cache/seq_preds.csv.gz
run: python k2_seq_eval.py > ../results/k2_seq_eval.txt"""
import os, sys, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mlib as L
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
CSV = os.path.join(L.CACHE, 'seq_preds.csv.gz')
NAMES = ['lstm_raw', 'rnn_resid', 'lstm_resid']

def load(name):
    parts = {}
    for f in sorted(glob.glob(os.path.join(L.CACHE, 'seq', name, '*.pkl'))):
        x = pd.read_pickle(f); x['date'] = pd.to_datetime(x.date).astype('datetime64[ns]'); x['stock_id'] = x.stock_id.astype(str)
        parts.setdefault(os.path.basename(f).split('_seed')[0], []).append(x.set_index(['date', 'stock_id']).p)
    assert len(parts) == 6 and all(len(v) == 5 for v in parts.values()), (name, {k: len(v) for k, v in parts.items()})
    return pd.concat([pd.concat(v, axis=1).mean(axis=1) for v in parts.values()]).sort_index()

def read_csv(f):
    x = pd.read_csv(f, dtype={'stock_id': str}, parse_dates=['date']); x['date'] = x.date.astype('datetime64[ns]')
    return x.set_index(['date', 'stock_id']).sort_index()

if glob.glob(os.path.join(L.CACHE, 'seq', '*', '*.pkl')):
    SQ = pd.DataFrame({n: load(n) for n in NAMES}); SQ.reset_index().to_csv(CSV, index=False, float_format='%.7f')
else:
    SQ = read_csv(CSV)
GRU_RAW = read_csv(os.path.join(L.CACHE, 'rnn_preds.csv.gz')).rnn_raw_B
LGB_RES = read_csv(os.path.join(L.CACHE, 'tab_preds.csv.gz')).lgb_resid

M = L.members()
idx = M.index.intersection(SQ.dropna().index).intersection(GRU_RAW.index).intersection(LGB_RES.index)
print(f'rows: members {len(M)}, new sequence members {len(SQ.dropna())}; common {len(idx)}')
M, SQ, GRU_RAW, LGB_RES = M.loc[idx], SQ.loc[idx], GRU_RAW.loc[idx], LGB_RES.loc[idx]
mlp = M[['mlp_raw_B', 'mlp_raw_xr']].mean(axis=1); tf = M.tf_raw_B; res_mlp = M[['mlp_res_B', 'mlp_res_xr']].mean(axis=1)
def centre(p):
    return p - p.groupby(level='date').transform('mean')
def pbar_of(p_raw):
    return p_raw.groupby(level='date').transform('mean')
PB0 = pbar_of((mlp + tf) / 2)
mk = L.market_daily()

def report(variants, kind, thr, base_key, tag):
    R = {k: L.daily(0.5 * (pb - 0.5) + centre(pe)) for k, (pb, pe) in variants.items()}
    base = R[base_key]; rows = []
    for k, D in R.items():
        r = L.table(D, None if k == base_key else base, k); r['P19>0'] = (D.AR.rolling(19).mean().dropna() > 0).mean(); rows.append(r)
    T = pd.DataFrame(rows)
    cols = ['config', 'val', 'hold', 'post', 'all22', 'sd', 'timing', 'xsec', 'P19>0', 'd_val', 't_val', 'd_hold', 't_hold', 'd_post', 't_post']
    print(f'\n{kind}: Top50 bp/day, walk-forward 2022-01..2026-10; d_/t_ = paired difference to the control')
    print(T[cols].round(3).to_string(index=False))
    tr = T[T.config.str.match(tag)].sort_values('t_val', ascending=False).reset_index(drop=True); alive = True
    print(f'Holm ({len(tr)} tests) and hold > 0:')
    for i, r in tr.iterrows():
        ok = alive and r.t_val >= thr[i] and r.d_hold > 0; alive = alive and r.t_val >= thr[i]
        print(f'  {r.config:40s} val t {r.t_val:+.2f} (needs {thr[i]:.2f}), hold diff {r.d_hold:+.2f} -> {"ADOPT" if ok else "do not adopt"}')
    for lab, a, b in (('static 2026-06-01..09-30', '2026-06-01', '2026-09-30'), ('live 2026-10-01..10-22', '2026-10-01', '2026-10-22')):
        print(f'  for information, {lab}: ' + '; '.join(f'{k.split(" ")[0]} {D.AR[(D.index >= a) & (D.index <= b)].mean():.2f}' for k, D in R.items()))

mv = {'control: (MLP + Transformer) / 2': (PB0, res_mlp),
      'K1 (MLP + LSTM) / 2': (pbar_of((mlp + SQ.lstm_raw) / 2), res_mlp),
      'K2 (MLP + Transformer + LSTM) / 3': (pbar_of((mlp + tf + SQ.lstm_raw) / 3), res_mlp),
      '(info, round 5) R1 (MLP + GRU) / 2': (pbar_of((mlp + GRU_RAW) / 2), res_mlp),
      '(info, round 5) R2 (MLP + Transformer + GRU) / 3': (pbar_of((mlp + tf + GRU_RAW) / 3), res_mlp),
      '(info) LSTM only': (pbar_of(SQ.lstm_raw), res_mlp),
      '(info) GRU only': (pbar_of(GRU_RAW), res_mlp)}
report(mv, 'MARKET VIEW', [1.96, 1.645], 'control: (MLP + Transformer) / 2', r'K\d')
print('\ncorrelation of each daily pbar with the Top50 intraday return m (val / hold / post / all):')
for k, (pb, _) in mv.items():
    d = pb.groupby(level='date').first(); j = d.index.intersection(mk.index); pr = L.periods(j)
    print(f'  {k:48s} ' + ' / '.join(f'{L.corr(d[j][v], mk.m[j][v]):+.3f}' for p, v in pr.items() if p in ('val', 'hold', 'post', 'all22')))

sv = {'control: p_e = MLP residual members': (PB0, res_mlp),
      'S3 p_e = (MLP + GRU) / 2': (PB0, (res_mlp + SQ.rnn_resid) / 2),
      'S4 p_e = (MLP + LSTM) / 2': (PB0, (res_mlp + SQ.lstm_resid) / 2),
      '(info, round 6) S1 p_e = (MLP + LightGBM) / 2': (PB0, (res_mlp + LGB_RES) / 2),
      '(info) GRU only': (PB0, SQ.rnn_resid),
      '(info) LSTM only': (PB0, SQ.lstm_resid)}
report(sv, 'STOCK VIEW', [1.96, 1.645], 'control: p_e = MLP residual members', r'S[34]')
P = L.panel().set_index(['date', 'stock_id'])
Z = pd.DataFrame({'resid_r': P.resid_r.reindex(idx), 'top50': P.top50.reindex(idx)})
Z = Z[Z.top50.astype(bool) & Z.resid_r.notna()]
print('\ndaily rank IC of the stock view vs the residual intraday return, Top50 (val / hold / post / all):')
for k, (_, pe) in sv.items():
    q = pd.DataFrame({'s': pe.reindex(Z.index), 'y': Z.resid_r})
    ic = q.groupby(level='date').apply(lambda g: g.s.rank().corr(g.y.rank())); pr = L.periods(ic.index)
    print(f'  {k:48s} ' + ' / '.join(f'{ic[v].mean():.3f}' for p, v in pr.items() if p in ('val', 'hold', 'post', 'all22')))

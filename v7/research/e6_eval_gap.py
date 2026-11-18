"""e6: upper bound of a same-day gap feature for the live stage (as if the 09:00 open were known before the deadline).
control  : v4 rule; stock view = mean(res_B core [e2], res_xr_ctrl [e5]); market view = existing walk-forward members
treatment: stock view = mean(res_B_gap, res_xr_gap) [e5]
Not a decision test: the live feature would be the 08:5x MIS indicative price, which only approximates the open.
run from this folder: python e6_eval_gap.py > results/e6_eval_gap.txt"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); V7 = os.path.dirname(HERE); ROOT = os.path.dirname(V7)
sys.path.insert(0, os.path.join(ROOT, 'v5', 'research'))
import compat
import numpy as np, pandas as pd
import rlib as L
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
L._P = L.E.build_panel(L.E.DATA); L._P['date'] = L._P['date'].astype('datetime64[ns]')
S10, S5 = range(1, 11), range(1, 6)
pr = {n: L.load_wf(n, s) for n, s in [('mlp_raw_B', S10), ('mlp_raw_xr', S10), ('tf_raw_B', S5)]}
def rd(path):
    x = pd.read_parquet(path); x['date'] = pd.to_datetime(x.date).astype('datetime64[ns]'); x['stock_id'] = x.stock_id.astype(str)
    return x.set_index(['date', 'stock_id']).p
G = os.path.join(V7, 'cache', 'wf_gap')
res = {'res_B_core': rd(os.path.join(V7, 'cache', 'wf_ext', 'core_mean.parquet'))}
for v in ('res_xr_ctrl', 'res_B_gap', 'res_xr_gap'):
    res[v] = rd(os.path.join(G, f'{v}_mean.parquet'))
mem = [L.member_def(n) for n in ['mlp_raw_B', 'mlp_raw_xr', 'tf_raw_B', 'mlp_res_B', 'mlp_res_xr']]
def run(rb, rx, w=0.5):
    probs = {**pr, 'mlp_res_B': rb, 'mlp_res_xr': rx}
    return L.daily(L.E.combine({m['name']: probs[m['name']] for m in mem}, mem, w_t=w)[['score']])
base = run(res['res_B_core'], res['res_xr_ctrl']); trt = run(res['res_B_gap'], res['res_xr_gap'])
cols = ['config', 'val', 'hold', 'post', 'all', 'sd', 'timing', 'xsec', 'd_val', 't_val', 'd_hold', 't_hold', 'd_post', 't_post']
print('v4 rule (w_T 0.5, ex-dividend -> 0), bp/day, Top50; d_/t_ = paired difference to the control')
print(pd.DataFrame([L.summary_row(base, None, 'control (cloud res_B + res_xr)'), L.summary_row(trt, base, '+ same-day gap in both stock members')])[cols].round(2).to_string(index=False))
for lab, a, b in (('static 2026-06-01..09-30', '2026-06-01', '2026-09-30'), ('live 2026-10-01..10-22', '2026-10-01', '2026-10-22')):
    m = (base.index >= a) & (base.index <= b); d = (trt.AR - base.AR)[m]
    print(f'{lab}: control {base.AR[m].mean():.2f}, with gap {trt.AR[m].mean():.2f}, diff {d.mean():+.2f} (t {L.plain_t(d):.2f}, {m.sum()} days)')
P = L._P[L._P.top50 & L._P.resid_r.notna()][['date', 'stock_id', 'resid_r']].set_index(['date', 'stock_id'])
for lab in ('res_xr_ctrl', 'res_xr_gap', 'res_B_core', 'res_B_gap'):
    q = P.join(res[lab].rename('p'), how='inner').reset_index()
    ic = q.groupby('date')[['p', 'resid_r']].apply(lambda z: z.p.rank().corr(z.resid_r.rank()))
    print(f'IC {lab:12s}: ' + ', '.join(f'{k} {ic[m].mean():.4f}' for k, m in L.periods(ic.index).items()))

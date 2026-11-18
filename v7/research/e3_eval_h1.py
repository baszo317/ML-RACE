"""e3: the pre-registered test of H1 (README_v7 section 1).
control  : v4 rule with mlp_res_B trained on the competition 100 (e2 variant core, seeds 1-5)
treatment: the same with mlp_res_B trained on core + point-in-time Top300 extras (e2 variant ext, seeds 1-5)
other members = the existing walk-forward predictions (mlp_raw_B, mlp_raw_xr x10, tf_raw_B x5, mlp_res_xr x10).
Decision: adopt iff the val (2022-24) paired t >= 2.0 and the hold (2025-01..2026-05) difference > 0; post is reported only.
run from this folder: python e3_eval_h1.py > results/e3_eval_h1.txt"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); V7 = os.path.dirname(HERE); ROOT = os.path.dirname(V7)
sys.path.insert(0, os.path.join(ROOT, 'v5', 'research'))
import compat
import numpy as np, pandas as pd
import rlib as L
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
L._P = L.E.build_panel(L.E.DATA); L._P['date'] = L._P['date'].astype('datetime64[ns]')          # no panel cache on disk
S10, S5 = range(1, 11), range(1, 6)
pr = {n: L.load_wf(n, s) for n, s in [('mlp_raw_B', S10), ('mlp_raw_xr', S10), ('tf_raw_B', S5), ('mlp_res_xr', S10)]}
def cloud(variant):
    x = pd.read_parquet(os.path.join(V7, 'cache', 'wf_ext', f'{variant}_mean.parquet'))
    x['date'] = pd.to_datetime(x.date).astype('datetime64[ns]'); x['stock_id'] = x.stock_id.astype(str)
    return x.set_index(['date', 'stock_id']).p
core, ext = cloud('core'), cloud('ext')
mem = [L.member_def(n) for n in ['mlp_raw_B', 'mlp_raw_xr', 'tf_raw_B', 'mlp_res_B', 'mlp_res_xr']]
def run(res_B, w=0.5):
    probs = {**pr, 'mlp_res_B': res_B}
    return L.daily(L.E.combine({m['name']: probs[m['name']] for m in mem}, mem, w_t=w)[['score']])
base, trt = run(core), run(ext)
cache10 = run(L.load_wf('mlp_res_B', S10))
rows = [L.summary_row(base, None, 'control: res_B core (seeds 1-5, cloud)'),
        L.summary_row(trt, base, 'H1: res_B_ext (seeds 1-5, cloud)'),
        L.summary_row(cache10, base, 'reference: res_B core seeds 1-10 (existing)')]
cols = ['config', 'val', 'hold', 'post', 'all', 'sd', 'timing', 'xsec', 'd_val', 't_val', 'd_hold', 't_hold', 'd_post', 't_post', 'nwt_val']
T = pd.DataFrame(rows)[cols]
print('v4 rule (w_T 0.5, ex-dividend day -> 0), bp/day, Top50; d_ and t_ = paired difference to the control')
print(T.round(2).to_string(index=False))
r = T.iloc[1]
ok = (r.t_val >= 2.0) and (r.d_hold > 0)
print(f'\npre-registered decision: val t = {r.t_val:.2f} (needs >= 2.0), hold diff = {r.d_hold:+.2f} (needs > 0) -> {"ADOPT" if ok else "do not adopt"}')
print('\nfor information only (not part of the decision): stock view alone (w_T = 0)')
b0, t0 = run(core, 0.0), run(ext, 0.0)
print(pd.DataFrame([L.summary_row(b0, None, 'control w_T=0'), L.summary_row(t0, b0, 'H1 w_T=0')])[cols].round(2).to_string(index=False))
# daily IC of the two members' centred probabilities with the residual return (Top50)
P = L._P[L._P.top50 & L._P.resid_r.notna()][['date', 'stock_id', 'resid_r']].set_index(['date', 'stock_id'])
for lab, s in (('core', core), ('ext', ext)):
    q = P.join(s.rename('p'), how='inner').reset_index()
    ic = q.groupby('date')[['p', 'resid_r']].apply(lambda z: z.p.rank().corr(z.resid_r.rank()))
    per = L.periods(ic.index)
    print(f'IC of {lab}: ' + ', '.join(f'{k} {ic[m].mean():.4f}' for k, m in per.items()))

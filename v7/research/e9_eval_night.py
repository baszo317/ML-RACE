"""e9: the pre-registered test of H2 (README_v7 section 6).
control  : v4 rule; market view = raw_B_ctrl, raw_xr_ctrl (e8, seeds 1-5) + tf_raw_B (existing, 5 seeds)
treatment: the same with raw_B_night, raw_xr_night (e8, + night_ret, night_range)
stock view (mlp_res_B, mlp_res_xr x10) = the existing walk-forward predictions, identical in both.
Decision: adopt iff the val (2022-24) paired t >= 2.0 and the hold (2025-01..2026-05) difference > 0; post is reported only.
run from this folder: python e9_eval_night.py > results/e9_eval_night.txt"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); V7 = os.path.dirname(HERE); ROOT = os.path.dirname(V7)
sys.path.insert(0, os.path.join(ROOT, 'v5', 'research'))
import compat
import numpy as np, pandas as pd
import rlib as L
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
L._P = L.E.build_panel(L.E.DATA); L._P['date'] = L._P['date'].astype('datetime64[ns]')
S10, S5 = range(1, 11), range(1, 6)
pr = {n: L.load_wf(n, s) for n, s in [('tf_raw_B', S5), ('mlp_res_B', S10), ('mlp_res_xr', S10)]}
def rd(v):
    x = pd.read_parquet(os.path.join(V7, 'cache', 'wf_night', f'{v}_mean.parquet'))
    x['date'] = pd.to_datetime(x.date).astype('datetime64[ns]'); x['stock_id'] = x.stock_id.astype(str)
    return x.set_index(['date', 'stock_id']).p
mem = [L.member_def(n) for n in ['mlp_raw_B', 'mlp_raw_xr', 'tf_raw_B', 'mlp_res_B', 'mlp_res_xr']]
def run(rb, rx, w=0.5):
    probs = {**pr, 'mlp_raw_B': rb, 'mlp_raw_xr': rx}
    return L.daily(L.E.combine({m['name']: probs[m['name']] for m in mem}, mem, w_t=w)[['score']])
base = run(rd('raw_B_ctrl'), rd('raw_xr_ctrl')); trt = run(rd('raw_B_night'), rd('raw_xr_night'))
ref = run(L.load_wf('mlp_raw_B', S10), L.load_wf('mlp_raw_xr', S10))
cols = ['config', 'val', 'hold', 'post', 'all', 'sd', 'timing', 'xsec', 'd_val', 't_val', 'd_hold', 't_hold', 'd_post', 't_post', 'nwt_val']
T = pd.DataFrame([L.summary_row(base, None, 'control: raw MLPs without night (seeds 1-5, cloud)'),
                  L.summary_row(trt, base, 'H2: raw MLPs + night_ret, night_range'),
                  L.summary_row(ref, base, 'reference: existing raw MLPs (seeds 1-10)')])[cols]
print('v4 rule (w_T 0.5, ex-dividend day -> 0), bp/day, Top50; d_ and t_ = paired difference to the control')
print(T.round(2).to_string(index=False))
r = T.iloc[1]; ok = (r.t_val >= 2.0) and (r.d_hold > 0)
print(f'\npre-registered decision: val t = {r.t_val:.2f} (needs >= 2.0), hold diff = {r.d_hold:+.2f} (needs > 0) -> {"ADOPT" if ok else "do not adopt"}')
for lab, a, b in (('static 2026-06-01..09-30', '2026-06-01', '2026-09-30'), ('live 2026-10-01..10-22', '2026-10-01', '2026-10-22')):
    m = (base.index >= a) & (base.index <= b); d = (trt.AR - base.AR)[m]
    print(f'for information, {lab}: control {base.AR[m].mean():.2f}, with night {trt.AR[m].mean():.2f}, diff {d.mean():+.2f} (t {L.plain_t(d):.2f}, {m.sum()} days)')
# how well the market view (pbar) tracks the market's intraday direction, with and without the night features
P = L._P[L._P.target_r.notna()]
mk = P.groupby('date').target_r.mean()
for lab, rb, rx in (('control', 'raw_B_ctrl', 'raw_xr_ctrl'), ('night', 'raw_B_night', 'raw_xr_night')):
    e = L.E.combine({m['name']: {**pr, 'mlp_raw_B': rd(rb), 'mlp_raw_xr': rd(rx)}[m['name']] for m in mem}, mem, w_t=0.5)
    pb = e.groupby(level='date').pbar.first().reindex(mk.index).dropna(); mm = mk.reindex(pb.index)
    per = L.periods(pb.index)
    print(f'corr(pbar, market intraday) {lab:8s}: ' + ', '.join(f'{k} {np.corrcoef(pb[m_], mm[m_])[0, 1]:.3f}' for k, m_ in per.items()))

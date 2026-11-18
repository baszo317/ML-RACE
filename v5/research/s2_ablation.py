"""s2: ablation of v4's components.
(A) long walk-forward 2022-01..2026-10 (yearly retrain; post = frozen 2026-05-29 models), all members = v4/v2 default
(B) static months 2026-06..09 with v4's real rolling pool (each day uses models trained up to d-1..d-5)"""
import os, sys, numpy as np, pandas as pd, rlib as L
sys.path.insert(0, L.V4)
import rolling as R
E = L.E
S10, S5 = range(1, 11), range(1, 6)
# ---------------- (A) long walk-forward
pr = {n: L.load_wf(n, s) for n, s in [('mlp_raw_B', S10), ('mlp_raw_xr', S10), ('tf_raw_B', S5), ('mlp_res_B', S10), ('mlp_res_xr', S10)]}

def score_from(pr, raw, res, w_t=0.5, stock_from_raw=False):
    mem = [L.member_def(n) for n in raw + res]
    idx = None
    for n in raw + res:
        idx = pr[n].index if idx is None else idx.intersection(pr[n].index)
    probs = {n: pr[n].reindex(idx) for n in raw + res}
    if stock_from_raw:            # stock view from the raw members (same w_T): score = w (pbar-.5) + (p_raw - pbar)
        out = E.combine(probs, mem)
        out['score'] = w_t * (out.pbar - 0.5) + (out.p_raw - out.pbar)
        return out
    return E.combine(probs, mem, w_t=w_t)

RAW = ['mlp_raw_B', 'mlp_raw_xr', 'tf_raw_B']; RES = ['mlp_res_B', 'mlp_res_xr']
def variants(pr, raw_all=RAW, res_all=RES, static=False):
    V = [('v4 full', dict(raw=raw_all, res=res_all)),
         ('- market view (w_T = 0)', dict(raw=raw_all, res=res_all, w_t=0.0)),
         ('- residual target (stock view from raw model)', dict(raw=raw_all, res=res_all, stock_from_raw=True)),
         ('- Transformer (market view = MLP only)', dict(raw=[r for r in raw_all if not r.startswith('tf')], res=res_all)),
         ('- XR rank members (B only)', dict(raw=[r for r in raw_all if 'xr' not in r], res=['mlp_res_B'])),
         ('- B members in stock view (res_xr only)', dict(raw=raw_all, res=['mlp_res_xr'])),
         ('- XR in stock view (res_B only)', dict(raw=raw_all, res=['mlp_res_B'])),
         ('- exdiv rule', dict(raw=raw_all, res=res_all, exdiv=False)),
         ('v1 rule (raw members, p > 0.5)', dict(raw=raw_all, res=[], v1=True))]
    out = []
    for lab, kw in V:
        exd = kw.pop('exdiv', True); v1 = kw.pop('v1', False)
        if v1:
            mem = [L.member_def(n) for n in kw['raw']]
            idx = None
            for n in kw['raw']:
                idx = pr[n].index if idx is None else idx.intersection(pr[n].index)
            o = E.combine({n: pr[n].reindex(idx) for n in kw['raw']}, mem)
            D = L.daily(o[['score']], exdiv=False)
        else:
            o = score_from(pr, **kw); D = L.daily(o[['score']], exdiv=exd)
        out.append((lab, D))
    return out

rowsA = []
VA = variants(pr)
base = VA[0][1]
for lab, D in VA:
    rowsA.append(L.summary_row(D, None if lab == 'v4 full' else base, lab))
TA = pd.DataFrame(rowsA)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print('(A) walk-forward 2022-01..2026-10-22 (yearly retrain; post frozen at 2026-05-29)')
print(TA.round(2).to_string(index=False)); TA.to_csv('out_s2_ablation_wf.csv', index=False)

# ---------------- (B) static months with the v4 rolling pool
X = pd.read_csv(os.path.join(L.CACHE, 'v4_static_model_preds.csv.gz'), dtype={'stock_id': str})
X['date'] = pd.to_datetime(X.date).astype('datetime64[ns]'); X['te'] = pd.to_datetime(X.te).astype('datetime64[ns]')
P = L.panel()
tpl = pd.read_csv(os.path.join(L.ROOT, 'sample_submission.csv'), dtype={'id': str})
dates = [pd.Timestamp(c) for c in tpl.columns if c != 'id']
members = [L.member_def(n) for n in RAW + RES]
sched = {k: dict(v) for k, v in R.DEFAULT['static'].items()}
units, use, prev = R.static_plan(P, members, dates, sched)
roll, froz = {}, {}
for m in members:
    x = X[X.member == m['name']]
    roll[m['name']] = R.pool_average(x, use, m['name'], dates)
    t0 = x.te.min(); froz[m['name']] = x[x.te == t0].groupby(['date', 'stock_id']).p.mean()
rowsB = []
VB = variants(roll)
baseB = VB[0][1]
for lab, D in VB:
    rowsB.append(L.summary_row(D, None if lab == 'v4 full' else baseB, lab))
for lab, D in variants(froz)[:1]:
    rowsB.append(L.summary_row(D, baseB, '- rolling (all frozen at 2026-05-29)'))
mixed = dict(roll); mixed['tf_raw_B'] = froz['tf_raw_B']
rowsB.append(L.summary_row(variants(mixed)[0][1], baseB, '- rolling Transformer only'))
TB = pd.DataFrame(rowsB)[['config', 'post', 'sd', 'timing', 'xsec', 'share1', 'd_post', 't_post']]
print('\n(B) static 2026-06-01..09-30 (84 days), v4 rolling pool')
print(TB.round(2).to_string(index=False)); TB.to_csv('out_s2_ablation_static.csv', index=False)
pd.to_pickle({'roll': roll, 'froz': froz}, os.path.join(L.CACHE, 'v4_static_pool_probs.pkl'))

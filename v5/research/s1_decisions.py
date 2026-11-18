"""s1: reproduce the overnight summary and test the three pending options against v2 default (paired, same days)."""
import numpy as np, pandas as pd, rlib as L
S10, S5, S3 = range(1, 11), range(1, 6), range(1, 4)
pr = {}
for n, s in [('mlp_raw_B', S10), ('mlp_raw_xr', S10), ('tf_raw_B', S5), ('mlp_res_B', S10), ('mlp_res_xr', S10),
             ('tf_res_B', S3), ('mlp_raw_w4', S10), ('tf_raw_w4', S3)]:
    pr[n] = L.load_wf(n, s); print(n, len(pr[n]))
pr['tf_raw_B3'] = L.load_wf('tf_raw_B', S3); pr['mlp_raw_B3'] = L.load_wf('mlp_raw_B', S3)
DEF = ['mlp_raw_B', 'mlp_raw_xr', 'tf_raw_B', 'mlp_res_B', 'mlp_res_xr']
def run(names, w, exdiv=True, rename=None):
    probs = dict(pr)
    mem = []
    for n in names:
        src = (rename or {}).get(n, n)
        m = L.member_def(n if n in L.SPEC else src); probs[m['name']] = pr[src]; mem.append(m)
    ens = L.E.combine({m['name']: probs[m['name']] for m in mem}, mem, w_t=w) if w is not None else \
        L.E.combine({m['name']: probs[m['name']] for m in mem}, [m for m in mem if m['target'] == 'raw'])
    return L.daily(ens[['score']], exdiv=exdiv)
base = run(DEF, 0.5)
rows = [L.summary_row(base, None, 'v2 default w_T=0.5 (+exdiv)')]
v1like = run(['mlp_raw_B', 'tf_raw_B'], None, exdiv=False, rename={'mlp_raw_B': 'mlp_raw_B3', 'tf_raw_B': 'tf_raw_B3'})
rows.append(L.summary_row(v1like, base, 'v1-like (mlp3+tf3, v1 rule)'))
for w in (0.0, 0.25, 0.75, 1.0):
    rows.append(L.summary_row(run(DEF, w), base, f'v2 default w_T={w}'))
rows.append(L.summary_row(run(DEF, 0.5, exdiv=False), base, 'v2 default, no exdiv'))
rows.append(L.summary_row(run(DEF + ['tf_res_B'], 0.5), base, '+ tf_res (--tf-resid)'))
rows.append(L.summary_row(run(DEF + ['mlp_raw_w4'], 0.5), base, '+ mlp_raw_w4 (4y window, market view)'))
rows.append(L.summary_row(run(DEF + ['mlp_raw_w4', 'tf_raw_w4'], 0.5), base, '+ mlp_w4 + tf_w4'))
T = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(T.round(2).to_string(index=False))
T.to_csv('out_s1_decisions.csv', index=False)

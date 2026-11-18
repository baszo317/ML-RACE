"""s10: concept drift -- does retraining more often help over 4.8 years? (MLP members; Transformer = overnight yearly run
in both rows). yearly: models trained up to the end of the previous year (post: frozen at 2026-05-29);
monthly: models trained up to the end of the previous month (post months included). Same seeds 1-5, same code."""
import os, numpy as np, pandas as pd, rlib as L
from s9_sigma_loss import load5, MEM
tf = L.load_wf('tf_raw_B', range(1, 6))
def rule(scheme, with_tf=True, w=0.5):
    pr = {n: load5(n, 'orig', scheme) for n in MEM}
    names = MEM + (['tf_raw_B'] if with_tf else [])
    if with_tf:
        pr['tf_raw_B'] = tf
    return L.daily(L.combine(pr, names, w)[['score']])
rows = []
for with_tf in (True, False):
    y = rule('year', with_tf); m = rule('month', with_tf)
    lab = '' if with_tf else ' (MLP only, no Transformer)'
    rows.append(L.summary_row(y, None, 'yearly retrain' + lab))
    rows.append(L.summary_row(m, y, 'monthly retrain' + lab))
T = pd.DataFrame(rows); pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(T.round(2).to_string(index=False)); T.to_csv('out_s10_retrain_freq.csv', index=False)
y = rule('year'); m = rule('month')
d = (m - y)
print('\ndifference monthly - yearly by part (bp/day):')
for k, msk in L.periods(d.index).items():
    print(f"  {k:5s} AR {d.AR[msk].mean():+.2f}  timing {d.timing[msk].mean():+.2f}  xsec {d.xsec[msk].mean():+.2f}")

r"""svm_grid_probe2.py -- does the best C move with n?  (fold, ES part, val = last 120 days; OOS for information)
usage: python svm_grid_probe2.py <fold_a>"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rslib as L
from sklearn.svm import SVC
a = sys.argv[1]; b = dict(L.FOLDS)[a]; m = L.members()['svm_res']
S = L.split(L.panel(), m, a, b)
Q, R, yq, wq, trm = S['Q'], S['R'], S['yq'], S['wq'], S['trm']
sc = L.Scaler().fit(Q[m['feats']].to_numpy(np.float64))
Z = sc.transform(Q[m['feats']].to_numpy(np.float64)).astype(np.float64); ZR = sc.transform(R[m['feats']].to_numpy(np.float64)).astype(np.float64)
Dv = Q[~trm]
rng = np.random.default_rng(100 + L.FOLDS.index((a, b)))
perm = rng.permutation(np.flatnonzero(trm))            # nested: the 10k rows are inside the 40k rows
plan = [(10000, C, g) for C in (0.01, 0.03) for g in (0.25, 1.0)] + [(40000, C, 0.5) for C in (0.01, 0.03, 0.1, 0.3)] + [(10000, C, 0.5) for C in (0.01, 0.03, 0.1, 0.3)]
for n, C, g in plan:
    ix = np.sort(perm[:n])
    t = time.time()
    svc = SVC(C=C, kernel='rbf', gamma=g / Z.shape[1], cache_size=1500).fit(Z[ix], 2 * yq[ix] - 1, sample_weight=wq[ix])
    tf = time.time() - t
    fv = svc.decision_function(Z[~trm]); fo = svc.decision_function(ZR)
    print(json.dumps({'fold': a, 'n': n, 'C': C, 'g': g, 'fit_sec': round(tf, 1), 'n_sv': int(svc.n_support_.sum()),
                      'val_ic': round(L.rank_ic(Dv.date.values, fv, Dv.resid_r.values), 4), 'val_hinge': round(L.whinge(fv, yq[~trm], wq[~trm]), 4),
                      'oos_ic': round(L.rank_ic(R.date.values, fo, R.resid_r.values), 4)}), flush=True)

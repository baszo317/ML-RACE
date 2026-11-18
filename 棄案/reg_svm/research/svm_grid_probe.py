r"""
svm_grid_probe.py -- where is the best (C, gamma) of the exact RBF SVC?  n rows drawn from the ES part of a fold's window,
scored on the last 120 days of the window (val) and, for information only, on the fold itself (OOS).
usage: python svm_grid_probe.py <fold_a> <n> > ../results/svm_grid_probe_<fold>.txt
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rslib as L
from sklearn.svm import SVC

a = sys.argv[1]; n = int(sys.argv[2])
b = dict(L.FOLDS)[a]; m = L.members()['svm_res']
S = L.split(L.panel(), m, a, b)
Q, R, yq, wq, trm = S['Q'], S['R'], S['yq'], S['wq'], S['trm']
sc = L.Scaler().fit(Q[m['feats']].to_numpy(np.float64))
Z = sc.transform(Q[m['feats']].to_numpy(np.float64)).astype(np.float64); ZR = sc.transform(R[m['feats']].to_numpy(np.float64)).astype(np.float64)
Dv = Q[~trm]
ix = np.sort(np.random.default_rng(100 + L.FOLDS.index((a, b))).choice(np.flatnonzero(trm), n, replace=False))
print(json.dumps({'fold': a, 'n': n, 'es_rows': int(trm.sum()), 'val_rows': int((~trm).sum())}), flush=True)
for C in (0.1, 1.0, 10.0):
    for g in (0.25, 1.0, 4.0):
        t = time.time()
        svc = SVC(C=C, kernel='rbf', gamma=g / Z.shape[1], cache_size=1000).fit(Z[ix], 2 * yq[ix] - 1, sample_weight=wq[ix])
        tf = time.time() - t
        fv = svc.decision_function(Z[~trm]); fo = svc.decision_function(ZR)
        print(json.dumps({'C': C, 'g': g, 'fit_sec': round(tf, 1), 'n_sv': int(svc.n_support_.sum()),
                          'val_ic': round(L.rank_ic(Dv.date.values, fv, Dv.resid_r.values), 4),
                          'val_hinge': round(L.whinge(fv, yq[~trm], wq[~trm]), 4),
                          'oos_ic': round(L.rank_ic(R.date.values, fo, R.resid_r.values), 4)}), flush=True)

r"""
svm_bench.py -- how long does an RBF SVM take on this computer? (stock view, the 2026-06 fold: window 2020-03-02 .. 2026-05-31)
usage: python svm_bench.py <core9.pkl> <final_feature_lists.json> <n list, e.g. 5000,10000,20000> [nystroem]
prints one JSON line per n: fit seconds, seconds to predict 5,000 rows, support vectors, libsvm iterations
"""
import sys, time, json, resource
import numpy as np, pandas as pd
from sklearn.svm import SVC

core, flist, ns = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(',') if x]
P = pd.read_pickle(core); P['date'] = pd.to_datetime(P.date)
B = json.load(open(flist))['B_competition_only']
a = pd.Timestamp('2026-06-01')
Q = P[(P.date >= pd.Timestamp('2020-03-02')) & (P.date < a) & P.resid_r.notna()]
R = P[(P.date >= a) & P.resid_r.notna()]
X = Q[B].to_numpy(np.float64); y = Q.y_resid.values.astype(int)
w = np.minimum(np.abs(Q.resid_r.values), 0.05); w = w / w.mean()
lo = np.nanquantile(X, 0.01, axis=0); hi = np.nanquantile(X, 0.99, axis=0)
Xc = np.clip(X, lo, hi); mu = np.nan_to_num(np.nanmean(Xc, axis=0)); sd = np.nanstd(Xc, axis=0); sd[~(sd > 1e-12)] = 1.0
tf = lambda A: np.nan_to_num((np.clip(A, lo, hi) - mu) / sd, nan=0.0)
Z = tf(X); ZR = tf(R[B].to_numpy(np.float64))[:5000]
print(json.dumps({'rows_in_window': len(Z), 'features': Z.shape[1]}), flush=True)
rng = np.random.default_rng(0)
for n in ns:
    ix = rng.choice(len(Z), n, replace=False)
    t = time.time()
    svc = SVC(C=1.0, kernel='rbf', gamma=1.0 / Z.shape[1], cache_size=1000).fit(Z[ix], 2 * y[ix] - 1, sample_weight=w[ix])
    t_fit = time.time() - t
    t = time.time(); svc.decision_function(ZR); t_pred = time.time() - t
    print(json.dumps({'n': n, 'fit_sec': round(t_fit, 2), 'pred_5000_sec': round(t_pred, 2), 'n_sv': int(svc.n_support_.sum()),
                      'iters': int(np.sum(svc.n_iter_))}), flush=True)
if len(sys.argv) > 4 and sys.argv[4] == 'nystroem':
    from sklearn.kernel_approximation import Nystroem
    from sklearn.svm import LinearSVC
    import warnings
    warnings.simplefilter('ignore')
    for k in (500, 1000):
        t = time.time()
        ny = Nystroem(kernel='rbf', gamma=1.0 / Z.shape[1], n_components=k, random_state=1).fit(Z)
        F = ny.transform(Z); t_tr = time.time() - t
        t = time.time(); LinearSVC(C=1e-2, loss='hinge', dual=True, max_iter=20000, tol=1e-4).fit(F, 2 * y - 1, sample_weight=w)
        print(json.dumps({'nystroem_components': k, 'n': len(Z), 'transform_sec': round(t_tr, 1), 'linear_svm_fit_sec': round(time.time() - t, 1),
                          'peak_rss_gb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)}), flush=True)
        del F

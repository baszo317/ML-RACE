"""x1: seed 1-5 means of the cached walk-forward stock-view members (mlp_res_B, mlp_res_xr) = the control of the T5 test
(PREREG_m.md, second round). Same loader as v5/research/rlib.py."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'v5', 'research')); sys.path.insert(0, os.path.join(ROOT, 'v7', 'research'))
import compat  # noqa: F401
import pandas as pd, rlib as L
X = pd.DataFrame({n: L.load_wf(n, range(1, 6)) for n in ['mlp_res_B', 'mlp_res_xr']}).reset_index()
out = os.path.join(ROOT, 'v8', 'cache', 'wf_res_seed1_5.csv.gz'); X.to_csv(out, index=False, float_format='%.7f')
print(X.shape, '->', out)

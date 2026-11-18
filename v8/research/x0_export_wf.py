"""x0: export the seed-averaged walk-forward predictions of the 5 v4 members (yearly folds 2022-01..2026-05 + post
frozen at 2026-05-29) into one compact file, so the market-view research can run anywhere without the 215 MB of pickles.
Same seeds as v5/research/s3_drift.py (MLP seeds 1-10, Transformer 1-5)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'v5', 'research'))
sys.path.insert(0, os.path.join(ROOT, "v7", "research"))
import compat  # noqa: F401  (pandas < 3 reading pandas-3 pickles)
import pandas as pd, rlib as L
S10, S5 = range(1, 11), range(1, 6)
cols = {}
for n, s in [('mlp_raw_B', S10), ('mlp_raw_xr', S10), ('tf_raw_B', S5), ('mlp_res_B', S10), ('mlp_res_xr', S10)]:
    cols[n] = L.load_wf(n, s); print(n, len(cols[n]), flush=True)
X = pd.DataFrame(cols).reset_index()
out = os.path.join(ROOT, 'v8', 'cache', 'wf_members.csv.gz')
X.to_csv(out, index=False, float_format='%.7f')
print(X.shape, X.date.min(), X.date.max(), '->', out)

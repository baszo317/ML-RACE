"""Does training X1 here give the laptop's live X1 models? Retrain the points in BASE/v11/models_v11/lgb_res_xrN that the
laptop trained itself (10/27, 10/28, 10/30, 11/02; seed yymmdd01) and the cloud start point 10/22 (seed 1), compare trees
and predictions on the day each one first predicted.

  python verify_x1.py BASE [te1,te2,...]   (11/02 needs a tree with labelled data after 11/02, e.g. the 11/05 as-of copy)
"""
import os, sys, re
BASE = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.join(BASE, 'v9'))
import numpy as np, pandas as pd
import rolling9 as V
X1 = V.mem('lgb_res_xrN', 'lgb', 'B9+XR+NEW+XRN+SIG', 'resid', train='lgb')
core = V.build_core(V.DATA, [])
core['date'] = core['date'].astype('datetime64[ns]'); core['stock_id'] = core.stock_id.astype(str)
fs = V.feature_sets(); lab = V.R.labelled_days(core)
root = os.path.join(BASE, 'v11', 'models_v11')
strip = lambda s: re.sub(r'\n\[[^\n]*\n', '\n', s.split('end of trees')[0])     # trees only (no parameter echo)
PTS = [('2026-10-22', 1), ('2026-10-27', 26102701), ('2026-10-28', 26102801), ('2026-10-30', 26103001), ('2026-11-02', 26110201)]
ONLY = sys.argv[2].split(',') if len(sys.argv) > 2 else None
for te, s in [x for x in PTS if ONLY is None or x[0] in ONLY]:
    b0 = V.load_bundle(os.path.join(root, X1['name'], f'{te}_s{s}.pt'))
    b1 = V.train_lgb(core, X1, pd.Timestamp(te), [s], fs[X1['feats']])[s]
    d = lab[lab > pd.Timestamp(te)][0]
    p0 = V.predict(b0, core, X1, [d]); p1 = V.predict(b1, core, X1, [d])
    same = strip(b0['model'].model_to_string()) == strip(b1['model'].model_to_string())
    print(f'{te} s{s}: laptop {b0["best_epoch"]} trees, here {b1["best_epoch"]}; trees identical {same}; '
          f'max |p diff| on {d.date()} {np.abs(p0.values - p1.reindex(p0.index).values).max():.2e}', flush=True)

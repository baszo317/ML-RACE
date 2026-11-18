r"""
ens_v3.py -- shared code of v3 = v2's ens_lib (..\v2\ens_lib.py, NOT modified) + the 5 new stock-picking features.

Everything of v2 is re-exported unchanged (members, training, the two-view combine(), ex-dividend rule, scoring);
only the panel and the feature sets are extended:
  build_panel()  : v2 panel + features_v3.add_v3()  (gap_bias_60/250, resid_vol_60, dow_bias_e, hi52 and their ranks)
  feature_sets() : v2 sets + 'B+N' = B + 5 new,  'B+XR+N' = B + 30 ranks + 5 new + their 5 ranks
"""
import os, sys
_V3 = os.path.dirname(os.path.abspath(__file__)); _V2 = os.path.join(os.path.dirname(_V3), 'v2')
for _p in (_V2, _V3):                       # v3 first on sys.path, then v2 (ens_lib itself adds ..\transformer)
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
from ens_lib import *                       # noqa: F401,F403  (v2 shared code, unchanged)
import ens_lib as _v2
import features_v3 as f3

V3_DIR = _V3

def build_panel(data_dir=_v2.DATA, pred_dates=()):
    return f3.add_v3(_v2.build_panel(data_dir, pred_dates), data_dir)

def feature_sets():
    fs = _v2.feature_sets()
    fs['B+N'] = fs['B'] + f3.NEW
    fs['B+XR+N'] = fs['B+XR'] + f3.NEW + f3.XRN
    return fs

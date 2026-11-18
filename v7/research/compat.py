"""pandas < 3 cannot unpickle the string columns that pandas 3 writes (transformer/results/wf_transformer_B.pkl).
Importing this module patches StringArray.__setstate__ for that one state format; with pandas >= 3 it does nothing."""
import numpy as np, pandas as pd
if int(pd.__version__.split('.')[0]) < 3:
    from pandas.core.arrays.string_ import StringArray
    from pandas._libs.arrays import NDArrayBacked
    _orig = StringArray.__setstate__
    def _setstate(self, state):
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], np.ndarray):
            NDArrayBacked.__init__(self, state[1].astype(object), pd.StringDtype('python', na_value=np.nan))
            return
        return _orig(self, state)
    StringArray.__setstate__ = _setstate

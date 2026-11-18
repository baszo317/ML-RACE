r"""
p2_derive.py -- part 2, F5: the monthly refit on a quarter's first month uses exactly the same training rows, seed and settings
as the quarterly refit on that day, so the model is the same (checked: 12 such pairs trained both ways, max |difference| 0.0).
This writes those monthly units from the quarterly ones (the predictions of that month only), marked in their meta.

  python p2_derive.py
"""
import os, json
import numpy as np, pandas as pd
import p2lib as Q, p2_train as PT
L = Q.L


def main():
    qd = dict(PT.refit_days('Q'))
    n = 0
    for a, b in PT.refit_days('M'):
        if a not in qd:
            continue
        for s in PT.SEEDS:
            fm, fq = PT.unit_file('M', a, s), PT.unit_file('Q', a, s)
            if os.path.exists(fm) or not os.path.exists(fq):
                continue
            q, meta = L.load_npz(fq)
            d = q.index.get_level_values('date')
            m = q[(d >= pd.Timestamp(a)) & (d < pd.Timestamp(b))]
            meta = dict(meta); meta.update(unit='C.LU@M', fold=a, seed=s, derived_from=os.path.relpath(fq, L.CACHE), n_pred=int(len(m)))
            L.save_unit(fm, m.index.get_level_values('date'), m.index.get_level_values('stock_id'), m.values, meta)
            n += 1
    print(f'{n} monthly units written from the identical quarterly units')


if __name__ == '__main__':
    main()

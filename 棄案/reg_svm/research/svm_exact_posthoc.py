r"""
svm_exact_posthoc.py -- after the exact SVM run: report only.  Not part of PREREG_svm.md; the verdict stays as
svm_exact_eval.py printed it (v11 + SVM: selected on val, not passed on hold / post).
  1. paired daily rank-IC differences: SVM minus lgb_res / X1 / the stock MLPs / the Nystroem preview (mean, NW t)
  2. what if the SVM REPLACED lgb_res instead of being added (v9: LightGBM part = SVM; v11 / W0: (SVM + X1) / 2),
     the SVM rescaled exactly as in svm_exact_eval.py
usage (in reg_svm\research):  python svm_exact_posthoc.py > ..\results\svm_exact_posthoc.txt
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import svm_exact_eval as E


def load_dir(d, idx):
    return pd.concat([E.load_npz(os.path.join(d, f'{fa}.npz')) for fa, _ in E.FOLDS]).sort_index().reindex(idx)


def main():
    d = E.Data()
    svm = load_dir(os.path.join(E.RS, 'cache', 'svm_exact'), d.idx)
    assert svm.notna().all()
    M = {'lgb_res': d.LGB, 'X1': d.x1, 'stock MLPs (mean of 4)': pd.concat(d.mlp, axis=1).mean(axis=1)}
    nd = os.path.join(E.RS, 'cache', 'svm_nystroem_preview')
    if all(os.path.exists(os.path.join(nd, f'{fa}.npz')) for fa, _ in E.FOLDS):
        M['Nystroem SVM (preview)'] = load_dir(nd, d.idx)
    ics = E.ic_daily(E.c(svm), d.resid)
    print('svm_exact_posthoc.py -- report only (not part of the pre-registered rule)\n')
    print('1. paired daily rank IC (all 100 stocks), exact SVM minus each member: mean difference (NW t)')
    for k, s in M.items():
        dd = (ics - E.ic_daily(E.c(s), d.resid)).dropna()
        print(f'  SVM - {k:24s} ' + '  '.join(f'{p} {dd[E.w(dd.index, p)].mean():+.4f} ({E.nw_t(dd[E.w(dd.index, p)].values):5.2f})'
                                           for p in E.PER))
    scale = lambda y: d.LGB.groupby(level='date').transform('mean') + E.c(y) * E.sdd(E.c(d.LGB)) / E.sdd(E.c(y))
    S = scale(svm)
    D = {'v9': d.run(d.view([d.LGB]), False), 'v11': d.run(d.view([d.LGB, d.x1]), True),
         'W0': d.run(d.view([d.LGB, d.x1]), True, w_t=0.0, lam=0.5)}
    D['v9, lgb_res -> SVM'] = d.run(d.view([S]), False)
    D['v11, lgb_res -> SVM'] = d.run(d.view([S, d.x1]), True)
    D['W0, lgb_res -> SVM'] = d.run(d.view([S, d.x1]), True, w_t=0.0, lam=0.5)
    print('\n2. the SVM replacing lgb_res (bp/day, minus the base, NW t)')
    for x, y in (('v9, lgb_res -> SVM', 'v9'), ('v11, lgb_res -> SVM', 'v11'), ('W0, lgb_res -> SVM', 'W0')):
        dd = (D[x].AR - D[y].AR).dropna()
        print(f'  {x:22s} - {y:3s}: ' + '  '.join(f'{p} {dd[E.w(dd.index, p)].mean():+.2f} ({E.nw_t(dd[E.w(dd.index, p)].values):.2f})'
                                                for p in E.PER))
    for nm, pe in (('v11 stock view', d.view([d.LGB, d.x1])), ('v11 with lgb_res -> SVM', d.view([S, d.x1]))):
        r = E.per(E.ic_daily(pe, d.resid))
        print(f'  rank IC of the {nm}: ' + ' / '.join(f'{r[p]:.4f}' for p in E.PER))
    rows = []
    for k, Dk in D.items():
        rows.append({'config': k, **E.per(Dk.AR), 'sd': Dk.AR.std()})
    print('\nTop50 bp/day (val / hold / post / all)')
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == '__main__':
    main()

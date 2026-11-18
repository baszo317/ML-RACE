r"""
agg_diag.py -- race_agg E3 (report only, research\PREREG_race_agg.md section 3E): can the out-of-bag estimate replace the
forward validation here?

For each D fold and leaf in {50, 250, 1000, 4000}:
  (a) row-level OOB of an ordinary RF (200 trees, max_samples 0.5, sqrt) fitted on the whole training window
  (b) day-level OOB of an RF whose trees each draw half of the training days with replacement (OOB = days not drawn)
  (c) forward validation: the RF of (a) fitted on the fit part, scored on the last 120 training days
  (d) the test year (after the fact): the RF of (a) on the fold's rows
Metrics: weighted log-loss (w = min(|e|, 5%)) and the mean daily Spearman IC with e (all stocks of the day).
  python agg_diag.py            -> results\agg_E3.txt, agg_E3.json
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '1')
import sys, json, time
import numpy as np, pandas as pd
import agglib as A

LEAVES = (50, 250, 1000, 4000)
N_TREES = 200


def wll(p, y, w):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(np.sum(w * -(y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w))


def dic(p, e, d):
    s = pd.DataFrame({'d': d, 'p': p, 'e': e}).dropna()
    return float(s.groupby('d').apply(lambda g: g.p.rank().corr(g.e.rank())).mean())


def rf(leaf, seed, oob=False):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=N_TREES, max_features='sqrt', max_samples=0.5, bootstrap=True, oob_score=oob,
                                  min_samples_leaf=leaf, criterion='gini', random_state=seed, n_jobs=1)


def day_rf_oob(X, y, w, d, rows_tr, leaf, seed):
    """RF with whole-day bootstrap (half of the days per tree); returns the OOB probability of every training row"""
    from sklearn.tree import DecisionTreeClassifier
    rng = np.random.default_rng(seed)
    idx = rows_tr[np.argsort(d[rows_tr], kind='stable')]
    dd = d[idx]; days = np.unique(dd)
    st = np.searchsorted(dd, days, 'left'); en = np.searchsorted(dd, days, 'right')
    s = np.zeros(len(idx)); c = np.zeros(len(idx))
    for t in range(N_TREES):
        pos = rng.integers(0, len(days), len(days) // 2)
        inb = np.zeros(len(days), bool); inb[pos] = True
        rows = np.concatenate([np.arange(st[k], en[k]) for k in pos])
        tree = DecisionTreeClassifier(max_features='sqrt', min_samples_leaf=leaf, criterion='gini',
                                      random_state=int(rng.integers(1 << 30)))
        tree.fit(X[idx[rows]], y[idx[rows]], sample_weight=w[idx[rows]])
        oob = np.concatenate([np.arange(st[k], en[k]) for k in np.flatnonzero(~inb)])
        s[oob] += tree.predict_proba(X[idx[oob]])[:, 1]; c[oob] += 1
    p = np.full(len(rows_tr), np.nan)
    order = np.argsort(np.argsort(d[rows_tr], kind='stable'), kind='stable')   # back to rows_tr order
    pp = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    p[:] = pp[order]
    return p, float((c > 0).mean())


def main():
    t0 = time.time()
    P = A.load_panel(); F = A.feats_C(); X = P[F].to_numpy(np.float32)
    d = P['date'].to_numpy()
    out = {}
    lines = ['race_agg E3 (report only): out-of-bag estimate vs forward validation vs the test year (RF, 200 trees, sqrt, half samples)',
             'wll = weighted log-loss (lower is better); IC = mean daily Spearman with e (higher is better)', '']
    for fa, fb in A.D_FOLDS:
        tr, te = A.masks(P, fa, fb)
        e, y, w = A.label_weight(P, tr)
        fit, val = A.split_last(d, tr)
        rows_tr = np.flatnonzero(tr)
        R = {}
        for lf in LEAVES:
            t1 = time.time()
            m_a = rf(lf, 1, oob=True).fit(X[tr], y[tr], sample_weight=w[tr])
            p_a = m_a.oob_decision_function_[:, 1]
            ok = np.isfinite(p_a)
            p_b, cover = day_rf_oob(X, y, w, d, rows_tr, lf, 1000 * int(fa[:4]) + lf)
            okb = np.isfinite(p_b)
            m_c = rf(lf, 1).fit(X[fit], y[fit], sample_weight=w[fit])
            p_c = m_c.predict_proba(X[val])[:, 1]
            lab_te = te & np.isfinite(e)
            p_d = m_a.predict_proba(X[lab_te])[:, 1]
            R[lf] = dict(oob_row_wll=wll(p_a[ok], y[tr][ok], w[tr][ok]), oob_row_ic=dic(p_a[ok], e[tr][ok], d[tr][ok]),
                         oob_day_wll=wll(p_b[okb], y[tr][okb], w[tr][okb]), oob_day_ic=dic(p_b[okb], e[tr][okb], d[tr][okb]),
                         oob_day_cover=cover,
                         val_wll=wll(p_c, y[val], w[val]), val_ic=dic(p_c, e[val], d[val]),
                         test_wll=wll(p_d, (e[lab_te] > 0).astype(float), np.minimum(np.abs(e[lab_te]), 0.05) / np.nanmean(np.minimum(np.abs(e[tr]), 0.05))),
                         test_ic=dic(p_d, e[lab_te], d[lab_te]), seconds=round(time.time() - t1, 1))
            A.log(f'E3 {fa} leaf {lf}: ' + ', '.join(f'{k} {v:.4f}' for k, v in R[lf].items() if k != 'seconds') + f' ({R[lf]["seconds"]} s)')
        pick = {m: min(LEAVES, key=lambda lf: R[lf][f'{m}_wll']) for m in ('oob_row', 'oob_day', 'val', 'test')}
        pick_ic = {m: max(LEAVES, key=lambda lf: R[lf][f'{m}_ic']) for m in ('oob_row', 'oob_day', 'val', 'test')}
        out[fa] = dict(results={str(k): v for k, v in R.items()}, pick_wll=pick, pick_ic=pick_ic)
        lines.append(f'fold {fa}:')
        lines.append('  leaf   row-OOB wll    IC    | day-OOB wll    IC    | forward-val wll  IC    | test-year wll   IC')
        for lf in LEAVES:
            r = R[lf]
            lines.append(f'  {lf:>5}  {r["oob_row_wll"]:.5f} {r["oob_row_ic"]:+.4f} | {r["oob_day_wll"]:.5f} {r["oob_day_ic"]:+.4f} | '
                         f'{r["val_wll"]:.5f} {r["val_ic"]:+.4f} | {r["test_wll"]:.5f} {r["test_ic"]:+.4f}')
        lines.append(f'  leaf picked by wll: {pick};  by IC: {pick_ic}')
        lines.append('')
    with open(os.path.join(A.RES, 'agg_E3.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    txt = '\n'.join(lines)
    with open(os.path.join(A.RES, 'agg_E3.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    print(txt)
    print(f'({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()

r"""
svm_exact_eval.py -- the exact RBF SVM as an extra member of the stock view (reg_svm\research\PREREG_svm.md section 3).
needs numpy and pandas only (any version: it rebuilds v9 / v11 from the cached walk-forward predictions with the code of
v11\research\v11b_lib.py copied below, because v11b_lib reads v10\cache\panel10.pkl, which only pandas 3 can read).

Part 0  rebuild v9, v11 and W0 and check them against the published numbers:
        v9 13.99 / 14.33 / 16.59 / 14.31, v11 14.60 / 14.34 / 20.35 / 15.02, W0 11.79 / 10.35 / 21.71 / 12.22 (val / hold / post / all)
Part 1  (report) the SVM's own daily rank IC next to lgb_res, X1 and the stock-view MLPs; within-day correlations
Part 2  (the rule) the SVM, rescaled like X5 / v12 (centred per day, lgb_res's daily sd), added to the LightGBM part:
          v11 + SVM = v11 with LightGBM part (lgb_res + X1 + SVM) / 3     <- decides
          val > 0 -> selected; selected and hold > 0 and post > 0 -> passed
Part 3  (report only) v9 + SVM (LightGBM part (lgb_res + SVM) / 2); W0 + SVM; the SVM as its own third of the stock view
usage (in reg_svm\research):  python svm_exact_eval.py [--preds DIR] [--name NAME] [--tag TAG] [--test]  > ..\results\svm_exact_eval.txt
  --preds: folder with <fold>.npz (default ..\cache\svm_exact; --test uses ..\cache\svm_exact_test)
  --tag:   output names ..\results\svm_<tag>_daily.csv / svm_<tag>_selection.json (default exact, or exact_test with --test)
"""
import os, sys, json, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); RS = os.path.dirname(HERE)
ROOT = os.environ.get('MLRACE_ROOT', os.path.dirname(RS))
for p in (os.path.join(ROOT, 'v9', 'research'), os.path.join(ROOT, 'v8')):
    if p not in sys.path:
        sys.path.insert(0, p)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
FOLDS = [('2022-01-01', '2023-01-01'), ('2023-01-01', '2024-01-01'), ('2024-01-01', '2025-01-01'), ('2025-01-01', '2026-01-01'),
         ('2026-01-01', '2026-06-01'), ('2026-06-01', '2100-01-01')]
PER = {'val': ('2022-01-01', '2025-01-01'), 'hold': ('2025-01-01', '2026-06-01'), 'post': ('2026-06-01', '2100-01-01'),
       'all': ('2022-01-01', '2100-01-01')}
W_T, LAM, NIGHT_W, EXTRA_ONES = 0.5, 0.75, 0.5, 5                          # = v11b_lib
EX4 = ('ex_today', 'ex_next1', 'ex_prev1', 'ex_prev2')
CGRID = np.round(np.arange(-0.3, 0.3 + 1e-9, 0.0025), 6)                   # = v10lib.CGRID
PUBLISHED = {'v9': (13.99, 14.33, 16.59, 14.31), 'v11': (14.60, 14.34, 20.35, 15.02), 'W0': (11.79, 10.35, 21.71, 12.22)}


def read_csv(f):
    x = pd.read_csv(f, dtype={'stock_id': str}, parse_dates=['date']); x['date'] = x.date.astype('datetime64[ns]')
    return x.set_index(['date', 'stock_id']).sort_index()


def load_npz(f):
    z = np.load(f, allow_pickle=True)                                     # our own files (the sid column may be an object array)
    idx = pd.MultiIndex.from_arrays([pd.to_datetime(z['date']).astype('datetime64[ns]'), z['sid'].astype(str)], names=['date', 'stock_id'])
    return pd.Series(z['p'], index=idx, name='p')


def w(ix, k):
    a, b = PER[k]; ix = pd.DatetimeIndex(ix)
    return (ix >= pd.Timestamp(a)) & (ix < pd.Timestamp(b))


def nw_t(d, lags=5):                                                         # = v10lib.nw_t
    d = np.asarray(d, float); d = d[~np.isnan(d)]; n = len(d)
    if n < 3:
        return np.nan
    e = d - d.mean(); s = e @ e / n
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return d.mean() / np.sqrt(s / n) if s > 0 else np.nan


c = lambda s: s - s.groupby(level='date').transform('mean')
sdd = lambda s: s.groupby(level='date').transform('std')


class Data:
    """= v11b_lib.Data for what the SVM needs, with core9.pkl instead of panel10.pkl (same rows and columns used)"""
    def __init__(self):
        W9 = read_csv(os.path.join(ROOT, 'v9', 'cache', 'wf9_preds.csv.gz'))
        M = read_csv(os.path.join(ROOT, 'v8', 'cache', 'wf_members.csv.gz'))
        LGB = read_csv(os.path.join(ROOT, 'v8', 'cache', 'tab_preds.csv.gz')).lgb_resid
        idx = M.index.intersection(W9.dropna().index).intersection(LGB.index)
        self.W9, self.TF, self.LGB, self.idx = W9.loc[idx], M.loc[idx, 'tf_raw_B'], LGB.loc[idx], idx
        P = pd.read_pickle(os.path.join(ROOT, 'v9', 'cache', 'core9.pkl'))
        P = P[['date', 'stock_id', 'top50', 'target_r', 'resid_r'] + list(EX4)].copy()
        P['date'] = pd.to_datetime(P.date).astype('datetime64[ns]'); P['stock_id'] = P.stock_id.astype(str)
        Q = P.set_index(['date', 'stock_id'])
        import hedge_beta as H                                              # = v10lib.info_table
        Bt = H.betas(sorted(set(idx.get_level_values('date'))))
        Bt['date'] = pd.to_datetime(Bt.date).astype('datetime64[ns]'); Bt['stock_id'] = Bt.stock_id.astype(str)
        Bt = Bt.set_index(['date', 'stock_id']).beta
        INFO = pd.DataFrame({'top50': Q.top50, 'target_r': Q.target_r, **{k: Q[k] for k in EX4}}).reindex(idx)
        INFO['beta'] = Bt.reindex(idx).clip(0.2, 2.5)
        INFO['beta'] = INFO.beta.fillna(INFO.groupby(level='date').beta.transform('median')).fillna(1.0)
        self.INFO = INFO[INFO.top50.astype(bool) & INFO.target_r.notna()]
        self.resid = Q.resid_r.reindex(idx)
        self.raw = pd.concat([pd.concat([self.W9.mlp_raw_B9, self.W9.mlp_raw_xr9, self.W9.mlp_raw_w4_9], axis=1).mean(axis=1), self.TF],
                             axis=1).mean(axis=1)
        self.mlp = [self.W9.mlp_res_B9, self.W9.mlp_res_xr9, self.W9.mlp_res_BN9, self.W9.mlp_res_xrN9]
        xdir = os.path.join(ROOT, 'v11', 'cache', 'screen', 'x', 'lgb_res_xrN', 'lgb_res_xrN')         # = v11b_train.load_member
        parts = []
        for fa, _ in FOLDS:
            parts.append(pd.concat([load_npz(os.path.join(xdir, f'{fa}_seed{s}.npz')) for s in range(1, 6)], axis=1).mean(axis=1))
        self.x1 = pd.concat(parts).sort_index().reindex(idx)
        import v9lib
        N = v9lib.night_table().set_index('date').night_ret.sort_index()
        sd = N.rolling(250, min_periods=60).std().shift(1)
        self.z_night = (N / sd).clip(-3, 3)
        days = pd.DatetimeIndex(sorted(P.date.unique()))
        first3 = pd.Series(days, index=days).groupby(days.to_period('M')).transform(lambda s: s.rank(method='first') <= 3)
        self.month_start = set(days[first3.values.astype(bool)])

    def pbar(self, night=False):
        pb = self.raw.groupby(level='date').mean()
        if night:
            sd = pb.rolling(250, min_periods=20).std().shift(1)
            z = self.z_night.reindex(pb.index)
            pb = pb - (NIGHT_W * sd * z).fillna(0.0)
        return pb

    def view(self, lgbs, third=None):
        """centred p_e: (MLP mean + LightGBM-part mean) / 2, or with a third part (MLP + LightGBM part + third) / 3"""
        m = pd.concat(self.mlp, axis=1).mean(axis=1); lg = pd.concat(lgbs, axis=1).mean(axis=1)
        return c((m + lg) / 2) if third is None else c((m + lg + third) / 3)

    def run(self, pe, v11, w_t=W_T, lam=LAM):
        pb = self.pbar(night=v11)
        s = w_t * (pb.reindex(pe.index.get_level_values('date')).values - 0.5) + pe
        return self.decide(s, month_start=v11, lam=lam)

    def decide(self, score, month_start=False, lam=LAM):                   # = v11b_lib.Data.decide
        X = self.INFO.join(score.rename('score'), how='inner')
        rows = []
        for d, g in X.groupby(level='date'):
            sc, b, r = g.score.values.copy(), g.beta.values, g.target_r.values
            ex = np.zeros(len(g), bool)
            for col in EX4:
                ex |= g[col].values == 1
            if month_start and d in self.month_start:
                cand = np.sort(sc[~ex & (sc <= 0)])[::-1]
                if len(cand) > EXTRA_ONES:
                    sc = sc - (cand[EXTRA_ONES - 1] + cand[EXTRA_ONES]) / 2
                elif len(cand):
                    sc = sc - cand[-1] + 1e-9
            s = np.where(ex, -1, np.where(sc > 0, 1, -1))
            if lam < 1.0:
                sb, bb = s.mean(), b.mean(); B0 = ((s - sb) * (b - bb)).mean(); Et = sb * bb + lam * B0
                S = (sc[None, :] + CGRID[:, None] * b[None, :] > 0) & ~ex[None, :]
                E = ((2 * S - 1) @ b) / len(b)
                dd = np.abs(E - Et); cnd = np.flatnonzero(dd <= dd.min() + 1e-12); j = cnd[np.argmin(np.abs(CGRID[cnd]))]
                s = np.where(S[j], 1, -1)
            m = r.mean()
            rows.append((d, (s * r).mean() * 1e4, s.mean() * m * 1e4, m * 1e4, s.mean()))
        D = pd.DataFrame(rows, columns=['date', 'AR', 'timing', 'm', 'sbar']).set_index('date'); D['xsec'] = D.AR - D.timing
        return D


def ic_daily(pred, r):
    X = pd.DataFrame({'p': pred, 'r': r}).dropna()
    g = X.groupby(level='date')
    R = pd.DataFrame({'p': g.p.rank(), 'r': g.r.rank()})
    R = R - R.groupby(level='date').transform('mean')
    s = pd.DataFrame({'xy': R.p * R.r, 'xx': R.p ** 2, 'yy': R.r ** 2}).groupby(level='date').sum()
    return s.xy / np.sqrt(s.xx * s.yy)


def per(s):
    return {p: float(s[w(s.index, p)].mean()) for p in PER}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--preds', default=''); ap.add_argument('--name', default='SVM')
    ap.add_argument('--test', action='store_true'); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    tag = a.tag or ('exact_test' if a.test else 'exact')
    pdir = a.preds or os.path.join(RS, 'cache', 'svm_exact_test' if a.test else 'svm_exact')
    have = [fa for fa, _ in FOLDS if os.path.exists(os.path.join(pdir, f'{fa}.npz'))]
    print(f'svm_exact_eval.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  predictions: {pdir}  folds found: {len(have)} of 6'
          + ('   [TEST: models trained on 3,000 rows, numbers mean nothing]' if a.test else ''))
    metas = []
    for fa in have:
        z = np.load(os.path.join(pdir, f'{fa}.npz'), allow_pickle=True)
        if 'meta' in z:
            metas.append(json.loads(str(z['meta'])))
    if metas:
        T = pd.DataFrame(metas)
        cols = [k for k in ('fold', 'rows', 'C_tuned', 'g', 'C', 'n_sv', 'fit_sec', 'pred_sec', 'oos_ic', 'finished') if k in T]
        T = T[cols].copy()
        if 'fit_sec' in T:
            T['fit_min'] = T.pop('fit_sec') / 60
        print('\nfolds (C_tuned, g = chosen on the last 120 days of the window; C = used for the full fit):')
        print(T.round(4).to_string(index=False))
    if len(have) < 6:
        print('\nnot all folds are done yet: parts 0-3 need all six (run svm_exact.py again to continue)')
        return

    d = Data()
    svm = pd.concat([load_npz(os.path.join(pdir, f'{fa}.npz')) for fa, _ in FOLDS]).sort_index()
    miss = int(svm.reindex(d.idx).isna().sum())
    svm = svm.reindex(d.idx)
    print(f'\nrows: {len(d.idx):,} (walk-forward 2022-01-03 .. 2026-10-22); SVM missing on {miss} of them')
    assert miss == 0, 'the SVM predictions do not cover every row of the walk-forward'

    # ------------------------------------------------------------ part 0
    print('\nPART 0 -- rebuild check (val / hold / post / all, bp/day)')
    D = {'v9': d.run(d.view([d.LGB]), False), 'v11': d.run(d.view([d.LGB, d.x1]), True),
         'W0': d.run(d.view([d.LGB, d.x1]), True, w_t=0.0, lam=0.5)}
    ok = True
    for k, ref in PUBLISHED.items():
        got = tuple(round(per(D[k].AR)[p], 2) for p in PER)
        same = all(abs(x - y) <= 0.005 for x, y in zip(got, ref)); ok &= same
        print(f'  {k:4s} rebuilt {" / ".join(f"{x:.2f}" for x in got)}   published {" / ".join(f"{x:.2f}" for x in ref)}   '
              + ('same' if same else 'DIFFERENT'))
    if not ok:
        print('  WARNING: the rebuild differs from the published numbers; the comparisons below are still paired (same rebuild)')

    # ------------------------------------------------------------ part 1
    print(f'\nPART 1 -- {a.name}: own daily rank IC (all 100 stocks) and within-day correlation (report only)')
    mlp_mean = pd.concat(d.mlp, axis=1).mean(axis=1)
    M = {a.name: svm, 'lgb_res': d.LGB, 'X1 (lgb_res_xrN)': d.x1, 'stock MLPs (mean of 4)': mlp_mean}
    rows = []
    for k, s in M.items():
        r = per(ic_daily(c(s), d.resid)); rows.append({'member': k, **r})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    cc = lambda x, y: float(np.corrcoef(c(x).values, c(y).values)[0, 1])
    print(f'  within-day correlation: {a.name}-lgb_res {cc(svm, d.LGB):.3f}, {a.name}-X1 {cc(svm, d.x1):.3f}, '
          f'{a.name}-MLPs {cc(svm, mlp_mean):.3f}; for scale: lgb_res-X1 {cc(d.LGB, d.x1):.3f}, lgb_res-MLPs {cc(d.LGB, mlp_mean):.3f}')
    scale = lambda y: d.LGB.groupby(level='date').transform('mean') + c(y) * sdd(c(d.LGB)) / sdd(c(y))
    S = scale(svm)
    pe11 = d.view([d.LGB, d.x1]); pe11s = d.view([d.LGB, d.x1, S])
    r0, r1 = per(ic_daily(pe11, d.resid)), per(ic_daily(pe11s, d.resid))
    print('  rank IC of the whole stock view: v11 ' + ' / '.join(f'{r0[p]:.4f}' for p in PER)
          + f';  v11 + {a.name} ' + ' / '.join(f'{r1[p]:.4f}' for p in PER))

    # ------------------------------------------------------------ part 2 + 3
    D[f'v11 + {a.name}'] = d.run(d.view([d.LGB, d.x1, S]), True)
    D[f'v9 + {a.name}'] = d.run(d.view([d.LGB, S]), False)
    D[f'W0 + {a.name}'] = d.run(d.view([d.LGB, d.x1, S]), True, w_t=0.0, lam=0.5)
    D[f'v11, {a.name} as a third part'] = d.run(d.view([d.LGB, d.x1], third=S), True)
    D[f'v9, {a.name} as a third part'] = d.run(d.view([d.LGB], third=S), False)

    def diff(x, y):
        dd = (D[x].AR - D[y].AR).dropna()
        return {p: (float(dd[w(dd.index, p)].mean()), float(nw_t(dd[w(dd.index, p)].values))) for p in PER}
    fmt = lambda r: '  '.join(f'{p} {r[p][0]:+.2f} ({r[p][1]:.2f})' for p in PER)
    print(f'\nPART 2 -- the rule (PREREG_svm.md 3.1): v11 + {a.name} minus v11, bp/day (NW t)')
    r = diff(f'v11 + {a.name}', 'v11')
    sel = r['val'][0] > 0; passed = sel and r['hold'][0] > 0 and r['post'][0] > 0
    print(f'  {fmt(r)}')
    print(f'  val {"> 0 -> selected" if sel else "<= 0 -> not selected"}; '
          + ('hold and post both > 0 -> PASSED: next step is a daily program (PREREG_svm.md 3.2)' if passed
             else ('hold or post <= 0 -> not passed: v9 / v11 stay as they are' if sel else 'v9 / v11 stay as they are')))
    print(f'\nPART 3 -- report only, each minus its base (bp/day, NW t)')
    for x, y in ((f'v9 + {a.name}', 'v9'), (f'W0 + {a.name}', 'W0'), (f'v11, {a.name} as a third part', 'v11'),
                 (f'v9, {a.name} as a third part', 'v9')):
        print(f'  {x:28s} - {y:4s}: {fmt(diff(x, y))}')
    rows = []
    for k, Dk in D.items():
        rr = per(Dk.AR); r19 = Dk.AR.rolling(19).mean().dropna()
        rows.append({'config': k, **rr, 'sd': Dk.AR.std(), 'timing': Dk.timing.mean(), 'xsec': Dk.xsec.mean(), 'P19>0': (r19 > 0).mean()})
    print('\nTop50 bp/day (walk-forward 2022-01 .. 2026-10)')
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    pd.DataFrame({k: v.AR for k, v in D.items()}).to_csv(os.path.join(RS, 'results', f'svm_{tag}_daily.csv'), float_format='%.4f')
    json.dump({'name': a.name, 'rebuild_ok': bool(ok), 'selected': bool(sel), 'passed': bool(passed), 'v11 + SVM minus v11': r,
               'folds': metas}, open(os.path.join(RS, 'results', f'svm_{tag}_selection.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()

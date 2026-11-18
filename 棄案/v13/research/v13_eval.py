r"""
v13_eval.py -- the three v13 candidates as extra stock-view members of v11, judged by the rules written before their results:
  LSTM-R, CNN-R   raw-sequence LSTM / 1D CNN, each as a NEW FAMILY of the stock view       (PREREG_v13.md;
                  decided on the cloud run of 2026-11-05: not selected; a local re-run is reported next to it)
  F-LOAM          2x50 LSTM -> h_t + B -> LightGBM tuned by Optuna, inside the LIGHTGBM PART (PREREG_v13b.md)
needs numpy and pandas only (pandas 2 or 3): v9 / v11 / W0 are rebuilt from the cached walk-forward predictions with the
code of reg_svm\research\svm_exact_eval.py (itself = v11\research\v11b_lib.py), copied below.

Part 0  rebuild v9, v11 and W0 and check them against the published numbers
Part 1  (report) each member's own daily rank IC, paired IC differences (raw vs B input; F-LOAM vs lgb_res, without Optuna,
        its own LSTM), within-day correlations, the two-signal diagnostics of README_v13.md section 3
Part 2  (the rules)  LSTM-R / CNN-R: v11 + k (new family, p_e = c((MLP + LightGBM part + k) / 3)) - v11
                     F-LOAM: v11 with LightGBM part (lgb_res + X1 + F-LOAM) / 3 - v11
                     val > 0 -> selected; the selected ones together: hold > 0 and post > 0 -> passed
Part 3  (report only) W0 / v9 versions, other placements, all three together, by year, power, October
usage (in v13\research):  python v13_eval.py [--preds DIR] [--cloud DIR] [--test] [--tag TAG]  > ..\results\v13_local_eval.txt
  --preds  units trained here: <kind>\<fold>_seed<s>.npz (default ..\cache\seq_raw; --test: ..\cache\seq_raw_test)
  --cloud  the cloud units of LSTM-R / CNN-R (default ..\cache\seq_raw_cloud): used when --preds does not have all of
           them, and compared with them when it does
  --tag    output names ..\results\<tag>_daily.csv, <tag>_selection.json (default v13_local, or test)
"""
import os, sys, json, math, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); V13 = os.path.dirname(HERE)
ROOT = os.environ.get('MLRACE_ROOT', os.path.dirname(V13))
for p in (HERE, os.path.join(ROOT, 'v9', 'research'), os.path.join(ROOT, 'v8')):
    if p not in sys.path:
        sys.path.insert(0, p)
import seqlib as S                                                          # FOLDS, load_member, load_npz (no torch needed)
import floamlib as FL                                                       # load_member for F-LOAM (no torch needed)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
FOLDS = S.FOLDS
PER = {'val': ('2022-01-01', '2025-01-01'), 'hold': ('2025-01-01', '2026-06-01'), 'post': ('2026-06-01', '2100-01-01'),
       'all': ('2022-01-01', '2100-01-01')}
YEARS = {'2022': ('2022-01-01', '2023-01-01'), '2023': ('2023-01-01', '2024-01-01'), '2024': ('2024-01-01', '2025-01-01'),
         '2025': ('2025-01-01', '2026-01-01'), '2026-01..05': ('2026-01-01', '2026-06-01'), 'post': ('2026-06-01', '2100-01-01')}
OCT = {'2026-10-01..10-22': ('2026-10-01', '2026-10-23')}
W_T, LAM, NIGHT_W, EXTRA_ONES = 0.5, 0.75, 0.5, 5                          # = v11b_lib
EX4 = ('ex_today', 'ex_next1', 'ex_prev1', 'ex_prev2')
CGRID = np.round(np.arange(-0.3, 0.3 + 1e-9, 0.0025), 6)                   # = v10lib.CGRID
PUBLISHED = {'v9': (13.99, 14.33, 16.59, 14.31), 'v11': (14.60, 14.34, 20.35, 15.02), 'W0': (11.79, 10.35, 21.71, 12.22)}
CLOUD_1105 = {'LSTM-R': {'val': -0.19, 'hold': -0.52, 'post': 4.02}, 'CNN-R': {'val': -0.02, 'hold': -1.36, 'post': 0.92}}


def read_csv(f):
    x = pd.read_csv(f, dtype={'stock_id': str}, parse_dates=['date']); x['date'] = x.date.astype('datetime64[ns]')
    return x.set_index(['date', 'stock_id']).sort_index()


def w(ix, k, table=PER):
    a, b = table[k]; ix = pd.DatetimeIndex(ix)
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
    r"""= reg_svm\research\svm_exact_eval.Data (= v11b_lib.Data for the stock view), with any number of extra families"""
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
            parts.append(pd.concat([S.load_npz(os.path.join(xdir, f'{fa}_seed{s}.npz')) for s in range(1, 6)], axis=1).mean(axis=1))
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

    def view(self, lgbs, extras=()):
        """centred p_e: (MLP mean + LightGBM-part mean + extra families) / (2 + number of extra families)"""
        m = pd.concat(self.mlp, axis=1).mean(axis=1); lg = pd.concat(lgbs, axis=1).mean(axis=1)
        tot = m + lg
        for e in extras:
            tot = tot + e
        return c(tot / (2 + len(extras)))

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
                Sg = (sc[None, :] + CGRID[:, None] * b[None, :] > 0) & ~ex[None, :]
                E = ((2 * Sg - 1) @ b) / len(b)
                dd = np.abs(E - Et); cnd = np.flatnonzero(dd <= dd.min() + 1e-12); j = cnd[np.argmin(np.abs(CGRID[cnd]))]
                s = np.where(Sg[j], 1, -1)
            m = r.mean()
            rows.append((d, (s * r).mean() * 1e4, s.mean() * m * 1e4, m * 1e4, s.mean()))
        D = pd.DataFrame(rows, columns=['date', 'AR', 'timing', 'm', 'sbar']).set_index('date'); D['xsec'] = D.AR - D.timing
        return D


def ic_daily(pred, r):
    """daily Spearman correlation of pred with r (any two series on (date, stock_id))"""
    X = pd.DataFrame({'p': pred, 'r': r}).dropna()
    g = X.groupby(level='date')
    R = pd.DataFrame({'p': g.p.rank(), 'r': g.r.rank()})
    R = R - R.groupby(level='date').transform('mean')
    s = pd.DataFrame({'xy': R.p * R.r, 'xx': R.p ** 2, 'yy': R.r ** 2}).groupby(level='date').sum()
    return s.xy / np.sqrt(s.xx * s.yy)


def per(s, table=PER):
    return {p: float(s[w(s.index, p, table)].mean()) for p in table}


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def unit_metas(kind, pdir, seeds):
    out = []
    for fa, _ in FOLDS:
        for s in seeds:
            f = os.path.join(pdir, kind, f'{fa}_seed{s}.npz')
            if os.path.exists(f):
                out.append(S.load_meta(f))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--preds', default=''); ap.add_argument('--cloud', default='')
    ap.add_argument('--test', action='store_true'); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    tag = a.tag or ('test' if a.test else 'v13_local')
    pdir = a.preds or os.path.join(V13, 'cache', 'seq_raw_test' if a.test else 'seq_raw')
    cdir = '' if a.test else (a.cloud or os.path.join(V13, 'cache', 'seq_raw_cloud'))
    seeds = (1,) if a.test else (1, 2, 3, 4, 5)
    need = len(FOLDS) * len(seeds)
    print(f'v13_eval.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  units: {pdir}' + (f'  (cloud LSTM-R / CNN-R: {cdir})' if cdir else '')
          + ('   [TEST: 1 epoch, 20,000 rows, 3 Optuna trials -- numbers mean nothing]' if a.test else ''))

    # ------------------------------------------------------------ which units, training summary
    src, metas = {}, {}
    for k in ('lstm', 'tcn', 'floam'):
        here = unit_metas(k, pdir, seeds)
        if len(here) == need:
            src[k], metas[k] = pdir, here
        elif k != 'floam' and cdir and len(unit_metas(k, cdir, seeds)) == need:
            src[k], metas[k] = cdir, unit_metas(k, cdir, seeds)
        else:
            src[k], metas[k] = None, here
    names = {'lstm': 'LSTM-R', 'tcn': 'CNN-R', 'floam': 'F-LOAM'}
    for k in ('lstm', 'tcn', 'floam'):
        lab = 'this run' if src[k] == pdir else ('cloud 2026-11-05' if src[k] else 'MISSING')
        print(f'  {names[k]:7s}: {len(metas[k])} of {need} units ({lab})')
    T = pd.DataFrame([m for k in metas for m in metas[k] if src[k]])
    if len(T):
        T['minutes'] = (T.train_sec + T.pred_sec) / 60
        g = T.groupby(['kind', 'fold'])
        tab = pd.DataFrame({'rows': g.rows.first(), 'best_epochs': g.best_epoch.apply(lambda s: ' '.join(str(int(x)) for x in s)),
                            'min_per_unit': g.minutes.mean().round(1)})
        print('\nbest LSTM / CNN epoch of each seed (early stopping on the last 120 days of the window; cap 30) and minutes per unit:')
        print(tab.to_string())
        print('total minutes: ' + ', '.join(f'{names[k]} {v:.0f}' for k, v in T.groupby('kind').minutes.sum().items())
              + f'  (threads per unit: {sorted(set(T.threads))})')
        F = T[T.kind == 'floam']
        if len(F):
            bp = pd.DataFrame(list(F.best_params))
            print('F-LOAM Optuna choices (all units): num_leaves ' + ', '.join(f'{int(v)}: {n}' for v, n in bp.num_leaves.value_counts().sort_index().items())
                  + f'; learning_rate median {bp.learning_rate.median():.3f} [{bp.learning_rate.min():.3f}, {bp.learning_rate.max():.3f}]'
                  + f'; bagging_freq ' + ', '.join(f'{int(v)}: {n}' for v, n in bp.bagging_freq.value_counts().sort_index().items())
                  + f'; rounds median {int(F.best_rounds.median())} (lgb_res settings: {int(F.fixed_rounds.median())})')
            imp = (F.fixed_val_logloss - F.best_val_logloss)
            print(f'  validation log-loss gain of Optuna over lgb_res settings (same 120 days it was tuned on): median {imp.median():.5f}')
    if not (src['lstm'] and src['tcn']):
        print('\nLSTM-R / CNN-R units are not complete here or in the cloud folder: run v13_train.py first'); return

    d = Data()

    def get(series, name):
        x = series.reindex(d.idx); miss = int(x.isna().sum())
        if miss:
            x = x.fillna(x.groupby(level='date').transform('mean'))
            print(f'  {name}: {miss} of {len(d.idx):,} rows missing (filled with the day mean)')
        return x
    C = {names[k]: get(S.load_member(k, seeds=seeds, pdir=src[k]), names[k]) for k in ('lstm', 'tcn')}
    have_f = src['floam'] is not None
    if have_f:
        C['F-LOAM'] = get(FL.load_member('p', seeds, pdir=src['floam']), 'F-LOAM')
        EXTRA = {'F-LOAM without Optuna (lgb_res settings)': get(FL.load_member('p_fixed', seeds, pdir=src['floam']), 'F-LOAM fixed'),
                 'F-LOAM own LSTM (2x50)': get(FL.load_member('p_lstm', seeds, pdir=src['floam']), 'F-LOAM LSTM')}
    else:
        EXTRA = {}
        print('\nF-LOAM units missing: the F-LOAM parts below are skipped (python v13_train.py --kinds floam)')
    if not a.test and cdir:
        for k in ('lstm', 'tcn'):
            if src[k] == pdir and len(unit_metas(k, cdir, seeds)) == need:
                y = get(S.load_member(k, seeds=seeds, pdir=cdir), names[k] + ' (cloud)')
                r_ = per(ic_daily(c(C[names[k]]), c(y)))
                print(f'  {names[k]} here vs the cloud run: max |p diff| {float((C[names[k]] - y).abs().max()):.2e}, within-day rank corr '
                      + ' / '.join(f'{r_[p]:.4f}' for p in PER))
    SQ = read_csv(os.path.join(ROOT, 'v8', 'cache', 'seq_preds.csv.gz')).reindex(d.idx)
    REF = {'LSTM-B (v8)': SQ.lstm_resid, 'GRU-B (v8)': SQ.rnn_resid}
    for k, v in REF.items():
        if v.isna().any():
            REF[k] = v.fillna(v.groupby(level='date').transform('mean'))
    svm_dir = os.path.join(ROOT, 'reg_svm', 'cache', 'svm_exact')
    SVM = None
    if all(os.path.exists(os.path.join(svm_dir, f'{fa}.npz')) for fa, _ in FOLDS):
        SVM = pd.concat([S.load_npz(os.path.join(svm_dir, f'{fa}.npz')) for fa, _ in FOLDS]).sort_index().reindex(d.idx)

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
        print('  WARNING: the rebuild differs from the published numbers: find the cause before deciding')

    # ------------------------------------------------------------ part 1
    print('\nPART 1 -- own daily rank IC with the excess return (all 100 stocks; report only)')
    mlp_mean = pd.concat(d.mlp, axis=1).mean(axis=1)
    pe11 = d.view([d.LGB, d.x1])
    M = {**C, **EXTRA, **REF, 'lgb_res': d.LGB, 'X1 (lgb_res_xrN)': d.x1, 'stock MLPs (mean of 4)': mlp_mean}
    if SVM is not None:
        M['SVM (exact, reg_svm)'] = SVM
    M['v11 stock view p_e'] = pe11
    ICd = {k: ic_daily(c(s), d.resid) for k, s in M.items()}
    print(pd.DataFrame([{'member': k, **per(v)} for k, v in ICd.items()]).round(4).to_string(index=False))
    print('\n  paired daily IC differences, mean (NW t):')
    pairs = [('LSTM-R', 'LSTM-B (v8)'), ('CNN-R', 'LSTM-B (v8)'), ('LSTM-R', 'stock MLPs (mean of 4)'), ('LSTM-R', 'CNN-R')]
    if have_f:
        pairs += [('F-LOAM', 'lgb_res'), ('F-LOAM without Optuna (lgb_res settings)', 'lgb_res'),
                  ('F-LOAM', 'F-LOAM without Optuna (lgb_res settings)'), ('F-LOAM', 'X1 (lgb_res_xrN)'),
                  ('F-LOAM own LSTM (2x50)', 'LSTM-B (v8)')]
    for x, y in pairs:
        dd = (ICd[x] - ICd[y]).dropna()
        print(f'    {x[:26]:26s} - {y[:26]:26s}: ' + '  '.join(f'{p} {dd[w(dd.index, p)].mean():+.4f} ({nw_t(dd[w(dd.index, p)].values):.2f})' for p in PER))
    print('\n  within-day rank correlation (mean over days, all periods):')
    others = {'v11 p_e': pe11, 'lgb_res': d.LGB, 'X1': d.x1, 'MLPs': mlp_mean, 'LSTM-B': REF['LSTM-B (v8)']}
    for name in C:
        print(f'    {name:7s}: ' + ', '.join(f'{k} {ic_daily(c(C[name]), c(v)).mean():.3f}' for k, v in others.items()))
    print('    for scale: lgb_res-X1 %.3f, lgb_res-MLPs %.3f, LSTM-R-CNN-R %.3f' % (
        ic_daily(c(d.LGB), c(d.x1)).mean(), ic_daily(c(d.LGB), c(mlp_mean)).mean(), ic_daily(c(C['LSTM-R']), c(C['CNN-R'])).mean()))
    print('\n  two-signal diagnostics (README_v13 3.2; x = v11 stock view, z = candidate; ranks, averaged over days, approximate):')
    print('    adding a little of z raises the IC iff IC_z > rho * IC_x;  best weight w* and best IC of a mix of the two')
    diag = {}
    for name in list(C) + list(REF) + (['SVM (exact, reg_svm)'] if SVM is not None else []):
        rho_d = ic_daily(c(M[name]), c(pe11))
        for p in PER:
            ix, iz, rho = per(ICd['v11 stock view p_e'])[p], per(ICd[name])[p], float(rho_d[w(rho_d.index, p)].mean())
            a_, b_ = ix - rho * iz, iz - rho * ix
            wstar = b_ / (a_ + b_) if (a_ + b_) != 0 else np.nan
            icopt = math.sqrt(max(ix * ix + iz * iz - 2 * rho * ix * iz, 0) / (1 - rho * rho)) if abs(rho) < 1 else np.nan
            diag[(name, p)] = dict(IC_x=ix, IC_z=iz, rho=rho, gain_condition=b_, w_star=wstar, IC_best=icopt)
        print(f'    {name[:20]:20s} rho {diag[(name, "all")]["rho"]:.3f}  ' + '  '.join(
            f'{p}: IC_z-rho*IC_x {diag[(name, p)]["gain_condition"]:+.4f}, w* {diag[(name, p)]["w_star"]:.2f}, '
            f'best IC {diag[(name, p)]["IC_best"]:.4f} (now {diag[(name, p)]["IC_x"]:.4f})' for p in ('val', 'hold', 'post')))
    scale = lambda y: d.LGB.groupby(level='date').transform('mean') + c(y) * sdd(c(d.LGB)) / sdd(c(y))
    SC = {k: scale(v) for k, v in C.items()}
    SR = {k: scale(v) for k, v in REF.items()}
    fam = [SC['LSTM-R'], SC['CNN-R']]
    views = {'v11 + LSTM-R': d.view([d.LGB, d.x1], extras=[SC['LSTM-R']]), 'v11 + CNN-R': d.view([d.LGB, d.x1], extras=[SC['CNN-R']]),
             'v11 + LSTM-R + CNN-R': d.view([d.LGB, d.x1], extras=fam)}
    if have_f:
        SF = scale(EXTRA['F-LOAM without Optuna (lgb_res settings)'])
        views['v11 + F-LOAM (LightGBM part)'] = d.view([d.LGB, d.x1, SC['F-LOAM']])
        views['v11 + all three'] = d.view([d.LGB, d.x1, SC['F-LOAM']], extras=fam)
    print('  rank IC of the whole stock view: v11 ' + ' / '.join(f'{per(ICd["v11 stock view p_e"])[p]:.4f}' for p in PER))
    for k, v in views.items():
        r1 = per(ic_daily(v, d.resid))
        print(f'                                   {k:28s} ' + ' / '.join(f'{r1[p]:.4f}' for p in PER))

    # ------------------------------------------------------------ part 2
    def diff(x, y, table=PER):
        dd = (D[x].AR - D[y].AR).dropna()
        return {p: (float(dd[w(dd.index, p, table)].mean()), float(nw_t(dd[w(dd.index, p, table)].values))) for p in table}
    fmt = lambda r: '  '.join(f'{p} {r[p][0]:+.2f} ({r[p][1]:.2f})' for p in r)
    for k, v in views.items():
        D[k] = d.run(v, True)
    print('\nPART 2 -- the rules, bp/day (NW t)')
    print('  PREREG_v13.md (LSTM-R, CNN-R as new families). Official decision = the cloud run of 2026-11-05 (results\\v13_eval.txt):')
    res = {}
    for k in ('LSTM-R', 'CNN-R'):
        r = diff(f'v11 + {k}', 'v11'); res[k] = r
        offi = CLOUD_1105[k]
        here = 'this run IS the cloud run' if src['lstm' if k == 'LSTM-R' else 'tcn'] == cdir else \
            ('re-run here: val > 0, the 11/5 decision would differ -> report both' if r['val'][0] > 0 else 're-run here agrees (val <= 0)')
        print(f'    v11 + {k:6s} - v11: {fmt(r)}   11/5: val {offi["val"]:+.2f} -> not selected; {here}')
    sel = []
    passed = False; rsel = None
    if have_f:
        r = diff('v11 + F-LOAM (LightGBM part)', 'v11'); res['F-LOAM'] = r
        s_ = r['val'][0] > 0
        print('  PREREG_v13b.md (F-LOAM in the LightGBM part: (lgb_res + X1 + F-LOAM) / 3). LSTM-R and CNN-R are not selected (11/5),')
        print('  so F-LOAM is judged alone:')
        print(f'    v11 + F-LOAM - v11: {fmt(r)}   -> ' + ('val > 0: selected' if s_ else 'val <= 0: not selected'))
        if s_:
            sel = ['F-LOAM']; rsel = r
            passed = r['hold'][0] > 0 and r['post'][0] > 0
            print('    ' + ('hold and post both > 0 -> PASSED: next step is a daily program (PREREG_v13b.md 2.4)' if passed
                            else 'hold or post <= 0 -> NOT PASSED: v9, v11 and W0 stay as they are'))
        else:
            print('    nothing selected -> NOT PASSED: v9, v11 and W0 stay as they are')
    else:
        print('  PREREG_v13b.md (F-LOAM): no F-LOAM units yet, not judged')

    # ------------------------------------------------------------ part 3
    print('\nPART 3 -- report only, each minus its base (bp/day, NW t)')
    for k in ('LSTM-R', 'CNN-R'):
        D[f'W0 + {k}'] = d.run(d.view([d.LGB, d.x1], extras=[SC[k]]), True, w_t=0.0, lam=0.5)
        D[f'v9 + {k}'] = d.run(d.view([d.LGB], extras=[SC[k]]), False)
        D[f'v11, {k} in the LightGBM part'] = d.run(d.view([d.LGB, d.x1, SC[k]]), True)
    D['W0 + LSTM-R + CNN-R'] = d.run(d.view([d.LGB, d.x1], extras=fam), True, w_t=0.0, lam=0.5)
    for k, v in SR.items():
        D[f'v11 + {k}'] = d.run(d.view([d.LGB, d.x1], extras=[v]), True)
    rows3 = [(f'W0 + {k}', 'W0') for k in ('LSTM-R', 'CNN-R')] + [('W0 + LSTM-R + CNN-R', 'W0')] + \
            [(f'v9 + {k}', 'v9') for k in ('LSTM-R', 'CNN-R')] + [(f'v11, {k} in the LightGBM part', 'v11') for k in ('LSTM-R', 'CNN-R')] + \
            [('v11 + LSTM-R + CNN-R', 'v11')] + [(f'v11 + {k}', 'v11') for k in SR]
    if have_f:
        D['W0 + F-LOAM (LightGBM part)'] = d.run(d.view([d.LGB, d.x1, SC['F-LOAM']]), True, w_t=0.0, lam=0.5)
        D['v9 + F-LOAM (LightGBM part)'] = d.run(d.view([d.LGB, SC['F-LOAM']]), False)
        D['v11, F-LOAM as a new family'] = d.run(d.view([d.LGB, d.x1], extras=[SC['F-LOAM']]), True)
        D['v11, F-LOAM replaces lgb_res'] = d.run(d.view([SC['F-LOAM'], d.x1]), True)
        D['v11, F-LOAM without Optuna (LightGBM part)'] = d.run(d.view([d.LGB, d.x1, SF]), True)
        D['W0 + all three'] = d.run(d.view([d.LGB, d.x1, SC['F-LOAM']], extras=fam), True, w_t=0.0, lam=0.5)
        rows3 += [('W0 + F-LOAM (LightGBM part)', 'W0'), ('v9 + F-LOAM (LightGBM part)', 'v9'), ('v11, F-LOAM as a new family', 'v11'),
                  ('v11, F-LOAM replaces lgb_res', 'v11'), ('v11, F-LOAM without Optuna (LightGBM part)', 'v11'),
                  ('v11 + all three', 'v11'), ('W0 + all three', 'W0')]
    for x, y in rows3:
        print(f'  {x[:42]:42s} - {y:4s}: {fmt(diff(x, y))}')
    print('\n  by year, v11 + candidate - v11 (bp/day, NW t):')
    for k in ['LSTM-R', 'CNN-R'] + (['F-LOAM (LightGBM part)'] if have_f else []):
        print(f'    {k:22s} ' + fmt(diff(f'v11 + {k}', 'v11', YEARS)))
    print('\n  2026-10-01 .. 10-22 (15 days, information only):')
    for x, y in [('v11 + LSTM-R', 'v11'), ('v11 + CNN-R', 'v11'), ('W0 + LSTM-R', 'W0'), ('W0 + CNN-R', 'W0')] + \
            ([('v11 + F-LOAM (LightGBM part)', 'v11'), ('W0 + F-LOAM (LightGBM part)', 'W0')] if have_f else []):
        print(f'    {x:30s} - {y:4s}: {fmt(diff(x, y, OCT))}')
    print('\n  power of the rules (normal approximation, daily difference sd and days of each period, v11 + candidate - v11):')
    pw = {}
    for k, base in [('LSTM-R', 'v11 + LSTM-R'), ('CNN-R', 'v11 + CNN-R')] + ([('F-LOAM', 'v11 + F-LOAM (LightGBM part)')] if have_f else []):
        dd = (D[base].AR - D['v11'].AR).dropna()
        sds = {p: (float(dd[w(dd.index, p)].std()), int(w(dd.index, p).sum())) for p in ('val', 'hold', 'post')}
        pw[k] = {}
        for delta in (0.0, 0.3, 0.5, 1.0):
            pr = 1.0
            for p, (sd_, n_) in sds.items():
                pr *= phi(delta / (sd_ / math.sqrt(n_))) if sd_ > 0 else 1.0
            pw[k][delta] = pr
        print(f'    {k:7s}: daily sd ' + ', '.join(f'{p} {sds[p][0]:.1f} bp ({sds[p][1]} days)' for p in sds)
              + ';  P(pass | true effect) ' + ', '.join(f'{dl:+.1f} -> {pv:.2f}' for dl, pv in pw[k].items()))
    rows = []
    for k, Dk in D.items():
        rr = per(Dk.AR); r19 = Dk.AR.rolling(19).mean().dropna()
        rows.append({'config': k, **rr, 'sd': Dk.AR.std(), 'timing': Dk.timing.mean(), 'xsec': Dk.xsec.mean(), 'P19>0': (r19 > 0).mean()})
    print('\nTop50 bp/day (walk-forward 2022-01 .. 2026-10)')
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    os.makedirs(os.path.join(V13, 'results'), exist_ok=True)
    pd.DataFrame({k: v.AR for k, v in D.items()}).to_csv(os.path.join(V13, 'results', f'{tag}_daily.csv'), float_format='%.4f')
    json.dump({'rebuild_ok': bool(ok), 'sources': {names[k]: (src[k] or '') for k in src},
               'lstm_cnn_official': 'not selected (cloud run 2026-11-05)', 'floam_selected': bool(sel), 'floam_passed': bool(passed),
               'each_minus_v11': res, 'floam_selected_minus_v11': rsel, 'diagnostics': {f'{k[0]} {k[1]}': v for k, v in diag.items()},
               'power': pw, 'ic': {k: per(v) for k, v in ICd.items()}},
              open(os.path.join(V13, 'results', f'{tag}_selection.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()

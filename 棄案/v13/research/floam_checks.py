r"""
floam_checks.py (v13) -- why the published F-LOAM numbers cannot be taken over to this competition, measured on our data
(README_v13.md section 7).  Report only; nothing here is a model of ours.  Needs numpy, pandas and vmdpy (pip install vmdpy).

F-LOAM = Liao & Lin (2025), "F-LOAM: an efficient hybrid model for stock price prediction based on SVMD denoising",
Discover Artificial Intelligence 5:347.  Their protocol, as written in the paper: closing-price LEVEL as target (R2 0.98-0.999),
a RANDOM 80/20 train/test split, Optuna minimising the MSE ON THE TEST SET, and SVMD = VMD (K 7, alpha 2000) whose IMFs are
kept when |Spearman(IMF, original series)| > 0.5, with no statement that the decomposition sees the training part only.

A1  price-level R2 of "tomorrow's close = today's close" (min-max scaled like the paper), Top50, 2020-01-02 .. 2026-10-22
A2  random 80/20 split: predicting a test day by interpolating the training days just before and after it (uses the future)
A3  SVMD on the whole series vs. SVMD on the past 250 days only: daily rank IC of the latest change of the denoised log
    close (known at D-1) with day D's intraday return and close-to-close return, Top50, 2025-01..2026-10
usage (in v13\research):  python floam_checks.py [--workers 2] [--start 2025-01-01]  > ..\results\floam_checks.txt
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); V13 = os.path.dirname(HERE)
ROOT = os.environ.get('MLRACE_ROOT', os.path.dirname(V13))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import seqlib as S                                              # _read_market (no torch needed)

ALPHA, K, TAU, TOL, THR, WIN = 2000, 7, 0.0, 1e-7, 0.5, 250
END = pd.Timestamp('2026-10-22')


def svmd(x):
    """VMD (vmdpy; K 7, alpha 2000) + keep the IMFs with |Spearman(IMF, x)| > 0.5, summed (the paper's SVMD).
    vmdpy drops the LAST sample of an odd-length series, so an odd series loses its FIRST sample instead."""
    from vmdpy import VMD
    x = np.asarray(x, float)
    off = len(x) % 2
    xe = x[off:]
    u, _, _ = VMD(xe, ALPHA, TAU, K, 0, 1, TOL)
    rx = pd.Series(xe).rank().values
    keep = [k for k in range(u.shape[0]) if abs(np.corrcoef(pd.Series(u[k]).rank().values, rx)[0, 1]) > THR]
    rec = u[keep].sum(axis=0) if keep else np.zeros_like(xe)
    return np.concatenate([np.full(off, np.nan), rec]), len(keep)


def _causal_one(args):
    sid, x, pos_list = args                                     # x: log close of the stock (trading days), pos = index of D
    out = []
    for p in pos_list:
        if p - WIN < 0:
            out.append((p, np.nan, np.nan)); continue
        rec, nk = svmd(x[p - WIN:p])                            # days D-250 .. D-1 only
        out.append((p, rec[-1] - rec[-2], nk))
    return sid, out


def load():
    info = json.load(open(os.path.join(ROOT, 'data', 'info.json'), encoding='utf-8'))
    top50 = info['Top50@Y']
    px = {}
    for s in top50:
        df = S._read_market(os.path.join(ROOT, 'data', 'market', f'{s}.csv'))
        df = df[(df.date <= END) & df['etl:adj_close'].notna()].reset_index(drop=True)
        px[s] = df[['date', 'etl:adj_open', 'etl:adj_close']].rename(columns={'etl:adj_open': 'o', 'etl:adj_close': 'c'})
    core = pd.read_pickle(os.path.join(ROOT, 'v9', 'cache', 'core9.pkl'))[['date', 'stock_id', 'target_r', 'resid_r']].copy()
    core['date'] = pd.to_datetime(core.date).astype('datetime64[ns]'); core['stock_id'] = core.stock_id.astype(str)
    return top50, px, core.set_index(['date', 'stock_id']).sort_index()


def r2(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float); m = np.isfinite(y) & np.isfinite(f)
    return 1 - ((y[m] - f[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()


def rank_ic(df, fcol, ycol):
    g = df[[fcol, ycol]].dropna().groupby(level='date')
    ic = g.apply(lambda q: q[fcol].rank().corr(q[ycol].rank()) if len(q) > 5 else np.nan).dropna()
    e = ic - ic.mean(); n = len(ic); s = (e ** 2).sum() / n
    for L in range(1, 6):
        s += 2 * (1 - L / 6) * (e.values[L:] * e.values[:-L]).sum() / n
    return ic.mean(), ic.mean() / np.sqrt(s / n), n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=2); ap.add_argument('--start', default='2025-01-01')
    a = ap.parse_args()
    t0 = time.time()
    top50, px, core = load()
    print(f'floam_checks.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  Top50, prices 2020-01-02 .. {END:%Y-%m-%d}')

    # ---------------------------------------------------------------- A1, A2
    rng = np.random.default_rng(2025)
    rows = []
    for s, df in px.items():
        c = df.c.values; cn = (c - c.min()) / (c.max() - c.min())                 # min-max like the paper
        pers = np.r_[np.nan, cn[:-1]]
        test = np.zeros(len(cn), bool); test[rng.choice(np.arange(1, len(cn) - 1), int(0.2 * len(cn)), replace=False)] = True
        tr = np.flatnonzero(~test); te = np.flatnonzero(test)
        j = np.searchsorted(tr, te); a_, b_ = tr[j - 1], tr[np.minimum(j, len(tr) - 1)]
        interp = cn[a_] + (cn[b_] - cn[a_]) * (te - a_) / np.maximum(b_ - a_, 1)
        lr = np.diff(np.log(c))
        rows.append({'stock': s, 'R2 persistence (all days)': r2(cn[1:], pers[1:]), 'MSE persistence': np.nanmean((cn[1:] - pers[1:]) ** 2),
                     'R2 persistence (random 20%)': r2(cn[te], pers[te]), 'R2 interpolation (random 20%)': r2(cn[te], interp),
                     'MSE interpolation': np.mean((cn[te] - interp) ** 2), 'R2 log return, persistence': r2(lr[1:], lr[:-1]),
                     'R2 log return, zero': r2(lr, np.zeros_like(lr))})
    A = pd.DataFrame(rows).set_index('stock')
    print('\nA1/A2 -- closing price scaled to [0, 1] per stock (as in the paper); 50 stocks: median [min, max]')
    for k in A.columns:
        print(f'  {k:32s} {A[k].median():+.6f}  [{A[k].min():+.6f}, {A[k].max():+.6f}]')
    print('  (paper: R2 0.9809 best ablation, 0.99915 Optuna-tuned 600883; MSE 0.0028 for 603880)')

    # ---------------------------------------------------------------- A3
    start = pd.Timestamp(a.start)
    full, jobs, nk_full = {}, [], []
    for s, df in px.items():
        x = np.log(df.c.values)
        rec, nk = svmd(x); nk_full.append(nk)
        f = pd.Series(np.r_[np.nan, np.nan, (rec[1:-1] - rec[:-2])], index=df.date.values)   # row D: rec[D-1] - rec[D-2]
        full[s] = f
        pos = [int(p) for p in np.flatnonzero(df.date.values >= np.datetime64(start))]
        jobs.append((s, x, pos))
    print(f'\nA3 -- SVMD (K {K}, alpha {ALPHA}, keep |Spearman| > {THR}); whole series: {np.mean(nk_full):.1f} of {K} IMFs kept on average')
    t1 = time.time(); causal = {}; nk_c = []
    if a.workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn')) as ex:
            res = list(ex.map(_causal_one, jobs))
    else:
        res = [_causal_one(j) for j in jobs]
    for s, out in res:
        d = px[s].date.values
        causal[s] = pd.Series({d[p]: v for p, v, _ in out})
        nk_c += [n for _, _, n in out if np.isfinite(n)]
    print(f'  past-250-day SVMD: {sum(len(v) for v in causal.values()):,} decompositions in {(time.time() - t1) / 60:.1f} min, '
          f'{np.mean(nk_c):.1f} IMFs kept on average')
    X = []
    for s, df in px.items():
        cc = pd.Series(np.r_[np.nan, np.diff(np.log(df.c.values))], index=df.date.values)
        X.append(pd.DataFrame({'date': df.date.values, 'stock_id': s, 'f_full': full[s].values,
                               'f_causal': causal[s].reindex(df.date.values).values, 'cc_r': cc.values}))
    X = pd.concat(X).set_index(['date', 'stock_id']).sort_index()
    X = X.join(core, how='left')
    X = X[X.index.get_level_values('date') >= start]
    print(f'  daily rank IC over the Top50, {X.index.get_level_values("date").min():%Y-%m-%d} .. {X.index.get_level_values("date").max():%Y-%m-%d} '
          f'(feature = latest change of the denoised log close, known at D-1 only if the decomposition is causal):')
    print(f'  {"feature":44s} {"IC intraday r(D)":>18s} {"IC close-to-close":>18s} {"pooled corr, intraday":>22s}')
    for fcol, lab in (('f_full', 'SVMD on the whole series (as in the paper)'), ('f_causal', 'SVMD on days D-250 .. D-1 only')):
        vals = [rank_ic(X, fcol, y) for y in ('target_r', 'cc_r')]
        pooled = float(X[[fcol, 'target_r']].dropna().corr().iloc[0, 1])
        print(f'  {lab:44s} ' + ' '.join(f'{m:+.4f} (t {t:5.1f})' for m, t, n in vals) + f'   {pooled:+.4f}            {vals[0][2]} days')
    print('  (rank IC inside a day is the same for the intraday and the excess intraday return: they differ by the day mean)')
    print(f'\ndone in {(time.time() - t0) / 60:.1f} min')
    os.makedirs(os.path.join(V13, 'results'), exist_ok=True)
    A.to_csv(os.path.join(V13, 'results', 'floam_checks_a1a2.csv'), float_format='%.6g')


if __name__ == '__main__':
    main()

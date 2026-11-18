r"""
race_live.py -- the race version, every day (PREREG_race_live.md sections 1 and 3).

  python race_live.py train              evening: build the panel, train the models the next trading day needs that are missing
                                         (daily mode: normally only today's)
  python race_live.py predict            morning: build the panel, complete the pool, predict the -1 placeholder day, decide,
                                         write out\submission_race_MMDD.csv (upload), out\pred_race_MMDD.csv, out\report_race_MMDD.txt
  python race_live.py predict --no-train only the models already trained (if the pool is short: WARN, the mean of what exists)
  python race_live.py selftest {lu,panel}  checks on the frozen data of part 1 (VM)

Mode (race_mode.txt, or --mode): daily = the models of the last 3 labelled days, seed = the day (PREREG section 1, default);
monthly = 3 seeds (1, 2, 3) trained on the data before the month of the day to predict (part 2 MR; the fallback of section 2).
Options: --data <folder> (default ..\..\data), --models <folder>, --out <folder>, --tag <text> (added to the file names)
Exit code 1 on a FATAL check (then no upload file is written).
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import livelib as LL                                                 # noqa: E402

L, B = LL.L, LL.B
MODE_FILE = os.path.join(HERE, 'race_mode.txt')


class Report:
    def __init__(self):
        self.items = []

    def add(self, level, msg):
        self.items.append((level, msg))

    def fatal(self):
        return [m for l, m in self.items if l == 'FATAL']

    def lines(self):
        return [f'{l:5s} {m}' for l, m in self.items]


def logger(path):
    return L.log_to(path)


def get_mode(a):
    if a.mode:
        return a.mode
    try:
        m = open(MODE_FILE, encoding='utf-8').read().strip().split()[0].lower()
        return m if m in ('daily', 'monthly') else 'daily'
    except (OSError, IndexError):
        return 'daily'


# ------------------------------------------------------------------ pool
def pool_keys(P, t, mode):
    """(training end, seed) of the models that predict day t"""
    lab = LL.labelled_days(P)
    lab = lab[lab < pd.Timestamp(t)]
    if mode == 'monthly':
        ms = pd.Timestamp(t).to_period('M').start_time
        d_end = lab[lab < ms][-1]
        return [(d_end, s) for s in (1, 2, 3)]
    return [(d, LL.seed_of(d)) for d in lab[-LL.POOL:]]


def ensure_pool(P, feats, keys, folder, train=True, log=print):
    metas = []
    for d, s in keys:
        m = LL.load_meta(d, s, folder)
        if m is None and train:
            log(f'pool: training the model of {d.date()} seed {s} ({LL.threads()} threads) ...')
            b, meta = LL.train_model(P, feats, d, seed=s)
            LL.save_model(b, meta, folder)
            m = LL.load_meta(d, s, folder)
            log(f'pool: {d.date()} seed {s} done: {meta["best_iter"]} rounds, {meta["n_train"]} rows, {meta["secs"]}s')
        if m is not None:
            metas.append(m)
    return metas


# ------------------------------------------------------------------ freshness of the crawled data (WARN only)
def ext_checks(t, last_lab, rep):
    def last_date(f):
        try:
            return pd.to_datetime(pd.read_csv(f, usecols=['date'])['date']).max()
        except Exception:
            return None
    y = os.path.join(LL.EXT, 'yahoo')
    us_last = [last_date(os.path.join(y, f + '.csv')) for f in ('IDX_IXIC', 'IDX_SOX', 'SPY', 'TSM')]
    us_last = [d for d in us_last if d is not None]
    exp_us = (pd.Timestamp(t) - pd.offsets.BDay(1)).normalize()
    if not us_last or min(us_last) < exp_us:
        rep.add('WARN', f'U.S. data in race_live\\ext\\yahoo end {min(us_last).date() if us_last else "?"}; expected '
                        f'{exp_us.date()} (run fetch_live.py --yahoo after the U.S. close; a U.S. holiday is fine)')
    for f, nm in (('IDX_N225', 'Nikkei'), ('IDX_KS11', 'KOSPI')):
        d = last_date(os.path.join(y, f + '.csv'))
        if d is None or d < pd.Timestamp(t):
            rep.add('WARN', f'{nm} has no row for {pd.Timestamp(t).date()} yet (opens 08:00 Taipei; a holiday there is fine) '
                            f'-> its opening-gap feature is missing today')
    fm = os.path.join(LL.EXT, 'finmind')
    for ds in ('TaiwanStockMarginPurchaseShortSale', 'TaiwanDailyShortSaleBalances', 'TaiwanStockDayTrading'):
        dd = [last_date(os.path.join(fm, ds, f)) for f in ('2330.csv', '2317.csv', '2454.csv')]
        dd = [d for d in dd if d is not None]
        if not dd or max(dd) < pd.Timestamp(last_lab):
            rep.add('WARN', f'FinMind {ds} ends {max(dd).date() if dd else "?"}, before {pd.Timestamp(last_lab).date()} '
                            f'(run fetch_live.py --finmind after 21:30) -> its lag-1 features are missing today')


def missing_share(rows, feats):
    cols = [c for c in feats if not c.startswith(LL.ADR_COLS_PREFIX)]
    X = rows.loc[rows.top50.astype(bool), cols].to_numpy(float)
    return float(np.isnan(X).mean()), cols


# ------------------------------------------------------------------ commands
def cmd_train(a):
    log = logger(os.path.join(LL.RESULTS, 'live.log'))
    mode = get_mode(a)
    P, feats, cov = LL.build_panel_C(a.data, log=log)
    lab = LL.labelled_days(P)
    R9 = LL.r9()
    t_next = pd.Timestamp(R9.next_trading_day(lab[-1], a.data))
    keys = pool_keys(P, t_next, mode)
    log(f'train ({mode}): labelled data up to {lab[-1].date()}; next trading day {t_next.date()}; models '
        + ', '.join(f'{d.date()}/seed {s}' for d, s in keys))
    metas = ensure_pool(P, feats, keys, a.models, train=True, log=log)
    log(f'train: pool ready for {t_next.date()}: ' + ', '.join(f'{m["train_end"]}/seed {m["seed"]}' for m in metas))
    return 0


def cmd_predict(a):
    log = logger(os.path.join(LL.RESULTS, 'live.log'))
    mode = get_mode(a)
    rep = Report()
    P, feats, cov = LL.build_panel_C(a.data, log=log)
    lab = LL.labelled_days(P)
    ph = LL.placeholder_days(P)
    if not len(ph):
        log(f'FATAL no -1 placeholder day after the last labelled day {lab[-1].date()}: merge this morning\'s zip first '
            f'(..\\..\\transformer\\update_data.py)')
        return 1
    t = ph[-1]
    R9 = LL.r9()
    nxt = R9.next_trading_day(lab[-1], a.data)
    if pd.Timestamp(nxt) != pd.Timestamp(t) or len(ph) > 1:
        rep.add('FATAL', f'the day to predict {t.date()} is not the trading day after the last labelled day {lab[-1].date()} '
                         f'(calendar: {pd.Timestamp(nxt).date()}; placeholder days {[str(x.date()) for x in ph]})')
    keys = pool_keys(P, t, mode)
    metas = ensure_pool(P, feats, keys, a.models, train=not a.no_train, log=log)
    if not metas:
        rep.add('FATAL', 'no model in the pool')
    elif len(metas) < len(keys):
        rep.add('WARN', f'pool has {len(metas)} of {len(keys)} models: ' + ', '.join(f'{m["train_end"]}/seed {m["seed"]}' for m in metas))
    for m in metas:
        if pd.Timestamp(m['train_end']) >= pd.Timestamp(t):
            rep.add('FATAL', f'model {m["train_end"]} is not before {t.date()}')
    ext_checks(t, lab[-1], rep)
    rows = P[P.date == t]
    ms, cols = missing_share(rows, feats)
    prev = P[P.date == lab[-1]]
    ms_prev, _ = missing_share(prev, feats)
    gone = [c for c in cols if rows.loc[rows.top50.astype(bool), c].isna().all() and prev.loc[prev.top50.astype(bool), c].notna().any()]
    if gone:
        rep.add('WARN', f'{len(gone)} C columns are missing for every Top50 stock today but not yesterday: {gone[:12]}')
    if ms > 0.20:
        rep.add('FATAL', f'{ms:.0%} of the non-ADR C values of the Top50 are missing today (yesterday {ms_prev:.0%})')
    if rep.fatal() and not a.force:
        for x in rep.lines():
            log(x)
        log('FATAL: no upload file written')
        return 1
    R = LL.predict_pool(P, feats, t, metas)
    dec = LL.decide_top25(R)
    n50, ones = len(dec), int(dec.pred.sum())
    if n50 != 50:
        rep.add('FATAL', f'{n50} Top50 rows on {t.date()} (expected 50)')
    if not 15 <= ones <= 25:
        rep.add('FATAL', f'{ones} ones (expected 15..25)')
    if rep.fatal() and not a.force:
        for x in rep.lines():
            log(x)
        log('FATAL: no upload file written')
        return 1
    f, sub = LL.write_submission(dec, t, a.out, a.tag)
    tag = pd.Timestamp(t).strftime('%m%d') + a.tag
    cols_out = ['rank', 'stock_id', 'p'] + [c for c in R.columns if c.startswith('p_')] + ['top25', 'ex_rule'] + list(LL.EX4) + ['pred']
    dec[cols_out].to_csv(os.path.join(a.out, f'pred_race_{tag}.csv'), index=False, encoding='utf-8-sig')
    lines = [f'race version ({mode} models), day to predict {t.date()} (written {time.strftime("%Y-%m-%d %H:%M:%S")})',
             f'labelled data up to {lab[-1].date()}; panel {len(P)} rows; C columns {len(feats)}; missing non-ADR C values '
             f'of the Top50 today {ms:.1%} (yesterday {ms_prev:.1%})',
             'pool: ' + ', '.join(f'{m["train_end"]}/seed {m["seed"]} ({m["best_iter"]} rounds)' for m in metas),
             f'ones {ones} (Top25 {int(dec.top25.sum())}, ex-dividend rule turned {int((dec.top25 & dec.ex_rule).sum())} to 0)',
             f'upload file: {f}', '', 'checks:'] + (rep.lines() or ['  none']) + ['', 'Top50 by probability:']
    for _, r in dec.iterrows():
        lines.append(f'{int(r["rank"]):3d} {r.stock_id:>6s} p {r.p:.4f}  ' + ('1' if r.pred else '0') +
                     ('  ex-dividend' if r.ex_rule else ''))
    txt = '\n'.join(lines)
    with open(os.path.join(a.out, f'report_race_{tag}.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    for x in lines[:7] + rep.lines():
        log(x)
    log(f'upload file written: {f}')
    return 0


def cmd_selftest(a):
    """lu: train_model = learners.fit_LU (part 1 C.LU, fold 2024, seed 1), bitwise; panel: the live panel = part 1 panel (C)"""
    log = logger(os.path.join(LL.RESULTS, 'selftest.log'))
    if a.what == 'lu':
        import train as T
        P = T.load_panel()
        feats = LL.c_features()
        d = P['date'].to_numpy()
        days = np.unique(d[d < np.datetime64('2024-01-01')])
        b, meta = LL.train_model(P, feats, pd.Timestamp(days[-1]), seed=1)
        te = (d >= np.datetime64('2024-01-01')) & (d < np.datetime64('2025-01-01'))
        p = b.predict(P.loc[te, feats].to_numpy(np.float32))
        ref, rm = L.load_npz(L.unit_file('C.LU', '2024-01-01', 1))
        mine = pd.Series(p, index=pd.MultiIndex.from_arrays([P.date[te], P.stock_id[te]])).reindex(ref.index)
        diff = float(np.nanmax(np.abs(mine.to_numpy() - ref.to_numpy())))
        log(f'selftest lu: best_iter {meta["best_iter"]} vs part 1 {rm.get("best_iter")}; max |p - p(part 1)| = {diff:.3g} '
            f'over {len(ref)} rows -> ' + ('IDENTICAL' if diff == 0 else 'DIFFERENT'))
        return 0 if diff == 0 else 1
    if a.what == 'panel':
        import train as T
        F = T.load_panel()
        feats = LL.c_features()
        P, f2, cov = LL.build_panel_C(a.data, log=log, fix_next=False, use_cache=False)
        A = F.set_index(['date', 'stock_id'])[feats]
        Bx = P.set_index(['date', 'stock_id'])[feats].reindex(A.index)
        a_, b_ = A.to_numpy(np.float32), Bx.to_numpy(np.float32)
        nan_diff = int((np.isnan(a_) != np.isnan(b_)).sum())
        both = ~np.isnan(a_) & ~np.isnan(b_)
        mx = float(np.abs(a_[both] - b_[both]).max()) if both.any() else 0.0
        keep = ['date', 'stock_id', 'top50', 'target_r', 'resid_r'] + list(LL.EX4)
        meta_same = bool(F[keep].reset_index(drop=True).equals(P[keep].reset_index(drop=True)))
        log(f'selftest panel: rows {len(F)} vs {len(P)}; C values: NaN pattern differs in {nan_diff} cells, max |diff| {mx:.3g}; '
            f'labels and ex flags equal: {meta_same}')
        return 0 if (nan_diff == 0 and mx == 0 and len(F) == len(P)) else 1
    raise SystemExit('selftest lu | panel')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['train', 'predict', 'selftest'])
    ap.add_argument('what', nargs='?', default='')
    ap.add_argument('--data', default=LL.DATA); ap.add_argument('--models', default=LL.MODELS)
    ap.add_argument('--out', default=LL.OUT); ap.add_argument('--tag', default='')
    ap.add_argument('--mode', default='', choices=['', 'daily', 'monthly'])
    ap.add_argument('--no-train', action='store_true'); ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(LL.RESULTS, exist_ok=True)
    rc = {'train': cmd_train, 'predict': cmd_predict, 'selftest': cmd_selftest}[a.cmd](a)
    sys.exit(rc)


if __name__ == '__main__':
    main()

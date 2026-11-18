r"""
news_run.py -- race_news (research\PREREG_race_news.md): cnyes 台股新聞 features in the stock model.

  python news_run.py --check              coverage, publish-time check, truncation test (writes results\check.txt)
  python news_run.py --train [--budget S] harness check C.LU 2024 seed 1, then CN.LU on the D folds (seeds 1-3); stops after S s
  python news_run.py --eval               CN.LU vs C.LU on D (results\news_results.txt / .json, news_daily.csv)

numpy + pandas (+ lightgbm for --train). Imports the frozen race_std code read only.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '1')
os.environ.setdefault('RACE_THREADS', '2')
import sys, glob, json, time, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); NEWS = os.path.dirname(HERE); RACEVER = os.path.dirname(NEWS)
STD = os.path.join(RACEVER, 'race_std')
sys.path.insert(0, os.path.join(STD, 'research'))
import rslib as L                                                     # noqa: E402  (frozen)

EXT = os.path.join(NEWS, 'ext', 'cnyes'); RES = os.path.join(NEWS, 'results'); UNITS = os.path.join(NEWS, 'cache', 'units')
PREREG = os.path.join(HERE, 'PREREG_race_news.md')
LOG = L.log_to(os.path.join(RES, 'race_news.log'))
POS = ['漲停', '大漲', '強漲', '飆漲', '創新高', '新高', '創高', '利多', '上修', '調升', '看好', '優於預期', '超乎預期', '轉盈',
       '獲利成長', '營收成長', '買超', '擴產', '漲價', '樂觀', '強勁', '回溫', '亮眼']
NEG = ['跌停', '大跌', '重挫', '崩跌', '暴跌', '新低', '利空', '下修', '調降', '看淡', '低於預期', '不如預期', '轉虧', '虧損',
       '衰退', '賣超', '減產', '降價', '保守', '疲弱', '下滑', '砍單', '示警']
KW = {'kw_rev_on': ['營收'], 'kw_earn_on': ['EPS', '獲利', '財報', '季報', '年報', '每股純益', '稅後'],
      'kw_rating_on': ['目標價', '評等', '調升', '調降', '喊進', '喊出'], 'kw_div_on': ['股利', '配息', '除息', '配股']}
NEWS_FEATS = ['nw_on', 'nw_on_solo', 'nw_pd', 'nw_abn', 'tone_on', 'tone_pd', 'kw_rev_on', 'kw_earn_on', 'kw_rating_on',
              'kw_div_on', 'mkt_n_on', 'mkt_tone_on']
H_CUT, H_CLOSE, H_OPEN = pd.Timedelta(hours=8, minutes=50), pd.Timedelta(hours=13, minutes=30), pd.Timedelta(hours=8, minutes=30)


def load_articles():
    fs = sorted(glob.glob(os.path.join(EXT, '*.csv')))
    if not fs:
        raise SystemExit('no ext\\cnyes\\*.csv: run run_fetch_news.bat first')
    A = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in fs], ignore_index=True)
    A = A.drop_duplicates('newsId')
    A['ts'] = pd.to_datetime(pd.to_numeric(A.publishAt, errors='coerce'), unit='s', utc=True).dt.tz_convert('Asia/Taipei').dt.tz_localize(None)
    A = A[A.ts.notna()].copy()
    txt = (A.title.fillna('') + ' ' + A.summary.fillna(''))
    A['pos'] = sum(txt.str.count(w) for w in POS); A['neg'] = sum(txt.str.count(w) for w in NEG)
    for k, ws in KW.items():
        A[k] = txt.apply(lambda s, ws=ws: int(any(w in s for w in ws)))
    A['tags'] = A.stock.fillna('').apply(lambda s: [x for x in s.split('|') if x.isdigit() and len(x) == 4])
    A['n_tw'] = A.tags.apply(len)
    return A.sort_values('ts').reset_index(drop=True)


def assign(A, days):
    """per article: t_att = first trading day t with t 08:50 >= ts (ATT window (p 08:50, t 08:50]); on = ts > p 13:30 (ON);
    t_pd = next trading day after the day d with d 08:30 < ts <= d 13:30 (PD)"""
    days = pd.DatetimeIndex(sorted(days))
    cut = days + H_CUT
    j = cut.searchsorted(A.ts.values, side='left')                    # first t with t 08:50 >= ts
    ok = j < len(days)
    t_att = pd.Series(pd.NaT, index=A.index, dtype='datetime64[ns]')
    t_att[ok] = days[j[ok]]
    prev = pd.Series(pd.NaT, index=A.index, dtype='datetime64[ns]')
    okp = ok & (j > 0)
    prev[okp] = days[j[okp] - 1]
    on = okp & (A.ts.values > (prev + H_CLOSE).values)
    # PD: the trading day d whose session (d 08:30, d 13:30] contains ts, then t = next day
    dd = A.ts.dt.normalize()
    is_day = dd.isin(days)
    in_sess = is_day & (A.ts > dd + H_OPEN) & (A.ts <= dd + H_CLOSE)
    k = days.searchsorted(dd.values, side='left')
    t_pd = pd.Series(pd.NaT, index=A.index, dtype='datetime64[ns]')
    okd = in_sess.values & (k + 1 < len(days))
    t_pd[okd] = days[k[okd] + 1]
    A = A.copy(); A['t_att'] = t_att.values; A['on'] = on; A['t_pd'] = t_pd.values
    return A


def build(A, days, stocks, upto=None):
    """features indexed (date, stock_id) for the trading days `days` and the stocks `stocks`; only articles with ts <= t 08:50"""
    days = pd.DatetimeIndex(sorted(days))
    if upto is not None:
        A = A[A.ts <= upto + H_CUT]
    A = assign(A, days)
    sset = set(stocks)
    E = A[['t_att', 'on', 't_pd', 'tags', 'n_tw', 'pos', 'neg'] + list(KW)].explode('tags')
    E = E[E.tags.isin(sset)].rename(columns={'tags': 'stock_id'})
    idx = pd.MultiIndex.from_product([days, sorted(sset)], names=['date', 'stock_id'])
    F = pd.DataFrame(index=idx)
    on = E[E.on & E.t_att.notna()]
    g = on.groupby(['t_att', 'stock_id'])
    F['nw_on'] = np.log1p(g.size().rename_axis(['date', 'stock_id']).reindex(idx).fillna(0))
    F['nw_on_solo'] = np.log1p(on[on.n_tw == 1].groupby(['t_att', 'stock_id']).size().rename_axis(['date', 'stock_id']).reindex(idx).fillna(0))
    pdx = E[E.t_pd.notna()]
    F['nw_pd'] = np.log1p(pdx.groupby(['t_pd', 'stock_id']).size().rename_axis(['date', 'stock_id']).reindex(idx).fillna(0))
    att = np.log1p(E[E.t_att.notna()].groupby(['t_att', 'stock_id']).size().rename_axis(['date', 'stock_id']).reindex(idx).fillna(0))
    base = att.groupby(level='stock_id').transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    F['nw_abn'] = att - base
    def tone(G, key):
        s = G.groupby([key, 'stock_id'])[['pos', 'neg']].sum().rename_axis(['date', 'stock_id']).reindex(idx).fillna(0)
        return (s.pos - s.neg) / (s.pos + s.neg + 1)
    F['tone_on'] = tone(on, 't_att'); F['tone_pd'] = tone(pdx, 't_pd')
    for k in KW:
        F[k] = on.groupby(['t_att', 'stock_id'])[k].max().rename_axis(['date', 'stock_id']).reindex(idx).fillna(0)
    allon = A[A.on & A.t_att.notna()].groupby('t_att')
    F['mkt_n_on'] = np.log1p(allon.size().reindex(days).fillna(0)).reindex(idx.get_level_values('date')).to_numpy()
    s = allon[['pos', 'neg']].sum().reindex(days).fillna(0)
    F['mkt_tone_on'] = ((s.pos - s.neg) / (s.pos + s.neg + 1)).reindex(idx.get_level_values('date')).to_numpy()
    return F[NEWS_FEATS].astype(float)


def panel_days_stocks():
    X = L.load_info()
    return pd.DatetimeIndex(sorted(X.date.unique())), sorted(X.stock_id.astype(str).unique())


def check():
    A = load_articles()
    days, stocks = panel_days_stocks()
    F = build(A, days, stocks)
    os.makedirs(RES, exist_ok=True)
    out = [f'articles {len(A)} ({A.ts.min()} .. {A.ts.max()}), with a Top100 tag {int(A.tags.apply(lambda t: any(x in set(stocks) for x in t)).sum())}']
    hh = A.ts.dt.hour * 60 + A.ts.dt.minute
    out.append(f'share published 08:50-09:00 {((hh >= 530) & (hh < 540)).mean():.4f}, 08:00-08:50 {((hh >= 480) & (hh < 530)).mean():.4f}, '
               f'13:30-15:00 {((hh >= 810) & (hh < 900)).mean():.4f}')
    by_year = A.groupby(A.ts.dt.year).agg(n=('newsId', 'size'), tagged=('n_tw', lambda s: float((s > 0).mean())))
    out.append('articles per year / share with a Taiwan tag: ' + ', '.join(f'{y}: {r.n} / {r.tagged:.2f}' for y, r in by_year.iterrows()))
    X = L.load_info(); X = X[X.top50.astype(bool)].set_index(['date', 'stock_id'])
    FT = F.reindex(X.index)
    for y in range(2020, 2027):
        sub = FT[FT.index.get_level_values('date').year == y]
        out.append(f'  {y}: Top50 stock-days with an ON article {float((sub.nw_on > 0).mean()):.3f}, solo {float((sub.nw_on_solo > 0).mean()):.3f}, '
                   f'PD {float((sub.nw_pd > 0).mean()):.3f}')
    rng = np.random.default_rng(1117)
    cand = days[(days >= pd.Timestamp('2020-06-01')) & (days <= pd.Timestamp('2026-11-09'))]
    pick = sorted(pd.DatetimeIndex(rng.choice(cand, 9, replace=False)))
    bad = []
    for t in pick:
        Fc = build(A, days[days <= t], stocks, upto=t)
        a = F.xs(t, level='date'); b = Fc.xs(t, level='date').reindex(a.index)
        d = (a - b).abs().to_numpy(); d = d[np.isfinite(d)]
        nan_diff = int((a.isna() != b.isna()).to_numpy().sum())
        if (d.max() if d.size else 0) > 1e-12 or nan_diff:
            bad.append((str(t.date()), float(d.max() if d.size else 0), nan_diff))
    out.append(f'truncation test on {[str(x.date()) for x in pick]}: ' + ('PASS' if not bad else f'FAIL {bad}'))
    D = X[(X.index.get_level_values('date') >= '2022-01-01') & (X.index.get_level_values('date') < '2025-01-01')]
    J = FT.reindex(D.index).join(D.target_r)
    ics = {c: float(J.groupby(level='date').apply(lambda x, c=c: x[c].rank().corr(x.target_r.rank()) if x[c].nunique() > 1 else np.nan).mean())
           for c in NEWS_FEATS}
    out.append('same-day rank IC with r on D (|IC| > 0.30 = stop and look for a leak): ' + ', '.join(f'{k} {v:+.4f}' for k, v in ics.items()))
    txt = '\n'.join(out)
    with open(os.path.join(RES, 'check.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    print(txt)
    return not bad and all(abs(v) <= 0.30 for v in ics.values() if v == v)


# ------------------------------------------------------------------ training (= race_mkt part 2)
def fit_LU_imp(P, feats, tr, te, seed):
    import lightgbm as lgb
    import learners
    e = P['resid_r'].to_numpy(float); d = P['date'].to_numpy()
    y = (e > 0).astype(float)
    w = np.minimum(np.abs(e), 0.05); w = w / np.nanmean(w[tr])
    prm = dict(objective='binary', learning_rate=0.02, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=learners.THREADS, verbose=-1, seed=seed,
               deterministic=True, force_col_wise=True)
    fit, val = learners._split_last_days(d, tr, learners.ES_DAYS)
    Xf = lambda m: P.loc[m, feats].to_numpy(np.float32)
    dtr = lgb.Dataset(Xf(fit), y[fit], weight=w[fit], free_raw_data=True)
    dva = lgb.Dataset(Xf(val), y[val], weight=w[val], reference=dtr)
    b = lgb.train(prm, dtr, num_boost_round=1000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    best = max(int(b.best_iteration or b.current_iteration()), 1)
    full = lgb.train(prm, lgb.Dataset(Xf(tr), y[tr], weight=w[tr]), num_boost_round=best)
    return full.predict(Xf(te)), dict(best_iter=best, importance=dict(zip(feats, full.feature_importance('gain').astype(float))))


TASKS = [('C.LU', '2024-01-01', '2025-01-01', 1)] + [('CN.LU', fa, fb, s) for fa, fb in L.D_FOLDS for s in (1, 2, 3)]


def unit_file(u, fa, s):
    return os.path.join(UNITS, u, f'{fa}_seed{s}.npz')


def train(budget):
    t_start = time.time()
    if all(os.path.exists(unit_file(u, fa, s)) for u, fa, fb, s in TASKS):
        LOG('all units exist'); return
    P = pd.read_pickle(L.PANEL)
    P['date'] = pd.to_datetime(P.date).astype('datetime64[ns]'); P['stock_id'] = P.stock_id.astype(str)
    C = L.features()['C']
    P = P[['date', 'stock_id', 'target_r', 'resid_r'] + C].copy()
    days = pd.DatetimeIndex(sorted(P.date.unique()))
    F = build(load_articles(), days, sorted(P.stock_id.unique())).add_prefix('NEWS__')
    P = P.merge(F, left_on=['date', 'stock_id'], right_index=True, how='left')
    NF = list(F.columns)
    d = P['date'].to_numpy()
    lab = P['target_r'].notna().to_numpy() & P['resid_r'].notna().to_numpy()
    LOG(f'panel {P.shape}, news features {len(NF)}')
    for (u, fa, fb, s) in TASKS:
        f = unit_file(u, fa, s)
        if os.path.exists(f):
            continue
        if u == 'CN.LU' and not os.path.exists(unit_file('C.LU', '2024-01-01', 1)):
            LOG('harness check missing: stop'); return
        if time.time() - t_start > budget:
            LOG('budget reached: run again to continue'); return
        feats = C if u == 'C.LU' else C + NF
        tr = lab & (d >= np.datetime64(L.TRAIN_START)) & (d < np.datetime64(pd.Timestamp(fa)))
        te = (d >= np.datetime64(pd.Timestamp(fa))) & (d < np.datetime64(pd.Timestamp(fb)))
        t0 = time.time()
        pred, meta = fit_LU_imp(P, feats, tr, te, s)
        meta.update(unit=u, fold=fa, seed=s, n_feats=len(feats), train_last=str(pd.Timestamp(d[tr].max()).date()))
        L.save_unit(f, d[te], P['stock_id'].to_numpy()[te], pred, meta)
        LOG(f'{u} {fa} seed {s}: best_iter {meta["best_iter"]}, {time.time() - t0:.0f}s')
        if u == 'C.LU':
            ref, mref = L.load_npz(os.path.join(L.CACHE, 'units', 'C.LU', f'{fa}_seed{s}.npz'))
            new, _ = L.load_npz(f)
            j = ref.index.intersection(new.index)
            dmax = float((ref[j] - new[j]).abs().max())
            LOG(f'harness check vs race_std C.LU {fa} seed {s}: rows {len(j)}/{len(ref)}, max |diff| {dmax:.3g}')
            if dmax != 0.0 or len(j) != len(ref):
                os.replace(f, f + '.FAILED'); LOG('HARNESS CHECK FAILED: stop'); return
    LOG('training done')


def evaluate():
    X = L.load_info()
    INFO = L.info_frame(X[(X.date >= '2022-01-01') & (X.date < '2025-01-01')])
    base = L.load_unit('C.LU', L.D_FOLDS)
    parts, metas = [], []
    for fa, _ in L.D_FOLDS:
        ss = []
        for s in (1, 2, 3):
            p, m = L.load_npz(unit_file('CN.LU', fa, s)); ss.append(p); metas.append(m)
        parts.append(pd.concat(ss, axis=1).mean(axis=1))
    cn = pd.concat(parts).sort_index()
    Db, Dc = L.decide_sd(base, INFO), L.decide_sd(cn, INFO)
    j = Db.index.intersection(Dc.index)
    d = (Dc.AR - Db.AR).reindex(j)
    yrs = {str(y): float(g.mean()) for y, g in d.groupby(pd.DatetimeIndex(d.index).year)}
    t = float(L.nw_t(d.to_numpy()))
    ok = bool(d.mean() > 0 and t >= 2 and sum(v > 0 for v in yrs.values()) >= 2)
    imp = pd.DataFrame([m['importance'] for m in metas]).fillna(0.0)
    share = imp.div(imp.sum(axis=1), axis=0).mean()
    nshare = share[[c for c in share.index if c.startswith('NEWS__')]].sort_values(ascending=False)
    # univariate IC of each news feature with the intraday residual return (D, Top50)
    days, stocks = panel_days_stocks()
    F = build(load_articles(), days, stocks)
    XT = X[X.top50.astype(bool) & (X.date >= '2022-01-01') & (X.date < '2025-01-01')].set_index(['date', 'stock_id'])
    J = F.reindex(XT.index).join(XT.resid_r)
    ics = {c: float(J.groupby(level='date').apply(lambda x, c=c: x[c].rank().corr(x.resid_r.rank()) if x[c].nunique() > 1 else np.nan).mean())
           for c in NEWS_FEATS}
    lines = [f'race_news, D (2022-2024), decision SD, {len(j)} days',
             f'  C.LU  (169 features)          AR {Db.AR.mean():6.2f}  IC {Db.ic.mean():.4f}  ' + ' '.join(f'{y}:{v:.2f}' for y, v in L.by_fold(Db, L.D_FOLDS).items()),
             f'  CN.LU (C + 12 news features)  AR {Dc.AR.mean():6.2f}  IC {Dc.ic.mean():.4f}  ' + ' '.join(f'{y}:{v:.2f}' for y, v in L.by_fold(Dc, L.D_FOLDS).items()),
             f'  CN.LU - C.LU: {d.mean():+.2f} bp/day, NW t {t:.2f}, years ' + ' '.join(f'{y}:{v:+.2f}' for y, v in yrs.items())
             + f'  -> {"PASS" if ok else "not passed"} (mean > 0, t >= 2, >= 2 of 3 years > 0)',
             f'  gain importance share of the news features {float(nshare.sum()):.3f}: ' + ', '.join(f'{k[6:]} {v:.4f}' for k, v in nshare.items()),
             '  rank IC with the intraday residual return (D, Top50, mean of daily IC; days where the feature is constant skipped): '
             + ', '.join(f'{k} {v:+.4f}' for k, v in ics.items())]
    txt = '\n'.join(lines)
    with open(os.path.join(RES, 'news_results.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    pd.DataFrame({'C.LU': Db.AR, 'CN.LU': Dc.AR, 'ic_C': Db.ic, 'ic_CN': Dc.ic}).to_csv(os.path.join(RES, 'news_daily.csv'))
    with open(os.path.join(RES, 'news_results.json'), 'w', encoding='utf-8') as fh:
        json.dump({'diff': float(d.mean()), 't': t, 'years': yrs, 'pass': ok, 'AR_C': float(Db.AR.mean()), 'AR_CN': float(Dc.AR.mean()),
                   'news_gain_share': float(nshare.sum()), 'ic': ics, 'prereg_sha256': L.sha256(PREREG)}, fh, indent=1)
    print(txt)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true'); ap.add_argument('--train', action='store_true')
    ap.add_argument('--eval', action='store_true'); ap.add_argument('--budget', type=float, default=1e9)
    a = ap.parse_args()
    if a.check:
        check()
    if a.train:
        train(a.budget)
    if a.eval:
        evaluate()

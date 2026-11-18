r"""
mkt_eval.py -- race_mkt (research\PREREG_race_mkt.md).

  python mkt_eval.py --check         data coverage of the futures features + truncation test of the information time
  python mkt_eval.py --d             D (2022-2024): market models, decisions P1-P8 vs DL0, diagnostics, LOCK.json
  python mkt_eval.py --seal-v        V (2025-01-02 .. 2026-11-09) once, after LOCK.json; writes SEALED_V.json and then refuses

Results go to ..\results\ (d_results.txt / .json, d_daily.csv, d_diag.txt, v_results.txt ...). numpy + pandas only.
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mktlib as K                                                    # noqa: E402

LOG = K.log_to(os.path.join(K.RES, 'race_mkt.log'))
CANDS = {  # name: (decision, market model, label)
    'P1': ('DL1', 'MB', 'DL1 x MB  calibrated sum, existing pre-market info'),
    'P2': ('DL1', 'MF', 'DL1 x MF  + futures positioning'),
    'P3': ('DL1', 'MFO', 'DL1 x MFO futures positioning only'),
    'P4': ('DL2', 'MF', 'DL2 x MF  beta version'),
    'P5': ('DLK', 'MF', 'DL3 x MF  learn k directly'),
    'P6': ('DL4', 'MF', 'DL4 x MF  bet only on |z| >= 1'),
    'P7': ('DL1', 'MS', 'DL1 x MS  gap-reversal structural model'),
    'P8': ('DL1', 'MV', 'DL1 x MV  vote'),
}
MODELS = ['MB', 'MF', 'MFO', 'MS', 'MV']


def code_files():
    return {f: K.sha256(os.path.join(HERE, f)) for f in sorted(os.listdir(HERE)) if f.endswith(('.py', '.md'))}


def ext_files():
    out = {}
    for sub in ('finmind', 'taifex'):
        d = os.path.join(K.EXT, sub)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith('.csv'):
                    out[f'{sub}/{f}'] = K.sha256(os.path.join(d, f))
    return out


def build():
    T50 = K.load_top50()
    m = K.market_m(T50)
    days = K.all_days()
    DAYF = K.day_features().reindex(days)
    FUT, dropped = K.futures_features(days)
    gap = K.open_gap(T50)
    X = {'MB': DAYF, 'MF': pd.concat([DAYF, FUT], axis=1), 'MFO': FUT, 'MV': pd.concat([DAYF, FUT], axis=1)}
    snap = {'written': time.strftime('%Y-%m-%d %H:%M:%S'), 'prereg_sha256': K.sha256(K.PREREG), 'code': code_files(),
            'ext': ext_files(), 'n_day_features': int(DAYF.shape[1]), 'n_fut_features': int(FUT.shape[1]),
            'fut_features': list(FUT.columns), 'dropped_sources': dropped,
            'frozen': {f: K.sha256(os.path.join(K.STD, 'cache', f)) for f in ('info.pkl', 'day_features.pkl')}}
    os.makedirs(K.RES, exist_ok=True)
    with open(os.path.join(K.RES, 'snapshot.json'), 'w', encoding='utf-8') as fh:
        json.dump(snap, fh, ensure_ascii=False, indent=1)
    return T50, m, days, DAYF, FUT, gap, X, snap


def check():
    T50, m, days, DAYF, FUT, gap, X, snap = build()
    LOG(f'futures features: {FUT.shape[1]} (dropped: {snap["dropped_sources"]})')
    cov = FUT[(FUT.index >= K.TRAIN_START) & (FUT.index <= K.LAST_LABEL)].notna().mean().round(3)
    first = FUT.apply(lambda s: s.first_valid_index())
    lines = [f'{c:28s} coverage 2020-03..2026-11-09 {cov[c]:.3f}  first {str(first[c])[:10]}' for c in FUT.columns]
    # truncation test: rebuild with every input dated < t (night rows dated t allowed) and compare row t
    rng = np.random.default_rng(20261117)
    cand = days[(days >= pd.Timestamp('2021-01-01')) & (days <= K.LAST_LABEL)]
    pick = sorted(rng.choice(cand, 9, replace=False))
    tw = K.twii_close()
    worst = 0.0; bad = []
    for t in pick:
        t = pd.Timestamp(t)
        Tc = K.raw_tables(cut=t)
        Fc, _ = K.futures_features(days[days <= t], T=Tc, twii=tw[tw.index < t])
        a, b = FUT.loc[t], Fc.loc[t].reindex(FUT.columns)
        both = a.notna() & b.notna()
        nan_diff = int((a.isna() != b.isna()).sum())
        d = float((a[both] - b[both]).abs().max()) if both.any() else 0.0
        worst = max(worst, d)
        if nan_diff or d > 1e-9:
            bad.append((str(t.date()), nan_diff, d, list(a.index[(a.isna() != b.isna()) | ((a - b).abs() > 1e-9)])[:8]))
    lines.append(f'\ntruncation test on {len(pick)} days {[str(pd.Timestamp(x).date()) for x in pick]}: max |diff| {worst:.2e}; '
                 f'{"PASS" if not bad else "FAIL " + str(bad)}')
    txt = '\n'.join(lines)
    with open(os.path.join(K.RES, 'check_features.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    print(txt)
    return not bad


def models_oos(X, m, gap, DAYF, end):
    out, infos = {}, {}
    for mod in MODELS:
        t0 = time.time()
        if mod == 'MS':
            mh, inf = K.walk_forward('MS', DAYF[K.GAP_VARS], m, gap=gap, Xg=DAYF[K.GAP_VARS], end=end)
        else:
            mh, inf = K.walk_forward(mod, X[mod], m, end=end)
        out[mod], infos[mod] = mh, inf
        LOG(f'model {mod}: {len(mh)} OOS days to {mh.index.max().date()} ({time.time() - t0:.0f}s)')
    return out, infos


def evaluate(DF, days_eval, m, mhat, label):
    c = K.stock_calib(DF, m, days_eval)
    mu = {mod: K.market_calib(mhat[mod], m, days_eval) for mod in MODELS}
    mu_g = K.market_calib(mhat['MF'], m, days_eval, gated=True)
    kser = K.policy_k(DF, mhat['MF'], days_eval)
    R = {'DL0': K.run_decision(DF, days_eval, 'DL0')}
    for name, (dl, mod, _) in CANDS.items():
        if dl == 'DLK':
            R[name] = K.run_decision(DF, days_eval, 'DLK', kser=kser)
        elif dl == 'DL4':
            R[name] = K.run_decision(DF, days_eval, 'DL1', mu=mu_g['mu'], c=c)
        else:
            R[name] = K.run_decision(DF, days_eval, dl, mu=mu[mod]['mu'], c=c)
    extra = {'c': c, 'mu': mu, 'mu_g': mu_g, 'k': kser}
    return R, extra


def summarize(R, label):
    base = R['DL0']
    res = {}
    for name, D in R.items():
        ar = D.AR.to_numpy()
        s = dict(days=int(len(D)), mean=float(ar.mean()), nw_t=float(K.nw_t(ar)), sd=float(ar.std(ddof=1)),
                 timing=float(D.timing.mean()), xsec=float(D.xsec.mean()), ones_mean=float(D.ones.mean()),
                 ones_sd=float(D.ones.std()), share_not25=float((D.ones != 25).mean()),
                 ones_q=[int(x) for x in np.percentile(D.ones, [5, 25, 50, 75, 95])], years=K.years(D.AR))
        s.update(K.roll19(ar))
        if name != 'DL0':
            d = (D.AR - base.AR.reindex(D.index)).dropna()
            s['diff'] = float(d.mean()); s['diff_t'] = float(K.nw_t(d.to_numpy()))
            s['diff_years'] = K.years(d); s['diff_p1'] = K.one_sided_p(s['diff_t'])
            yrs = list(s['diff_years'].values())
            s['pass'] = bool(s['diff'] > 0 and s['diff_t'] >= 2.0 and sum(v > 0 for v in yrs) >= 2) if label == 'D' else None
        res[name] = s
    ps = {k: v['diff_p1'] for k, v in res.items() if k != 'DL0'}
    for k, p in K.holm(ps).items():
        res[k]['holm_p'] = p
    return res


def fmt_table(res, title):
    L = [title, '', f'{"cand":5s} {"days":>4s} {"AR":>7s} {"t":>6s} {"sd":>6s} {"timing":>7s} {"xsec":>7s} {"ones":>5s} {"!=25":>5s} '
         f'{"diff":>7s} {"t":>6s} {"holm":>6s} {"pass":>5s}  years(diff)']
    for k, s in res.items():
        dy = ' '.join(f'{y}:{v:+.2f}' for y, v in s.get('diff_years', {}).items())
        L.append(f'{k:5s} {s["days"]:4d} {s["mean"]:7.2f} {s["nw_t"]:6.2f} {s["sd"]:6.1f} {s["timing"]:7.2f} {s["xsec"]:7.2f} '
                 f'{s["ones_mean"]:5.1f} {s["share_not25"]:5.2f} '
                 + (f'{s["diff"]:+7.2f} {s["diff_t"]:6.2f} {s["holm_p"]:6.3f} {str(s["pass"]):>5s}  {dy}' if k != 'DL0' else ''))
    L.append('')
    L.append('ones (5/25/50/75/95 %) and the 19-day view (share of rolling 19-day means above 0/10/20/30/40 bp, worst)')
    for k, s in res.items():
        L.append(f'{k:5s} ones {s["ones_q"]}  P19>0 {s["P19>0"]:.3f} >10 {s["P19>10"]:.3f} >20 {s["P19>20"]:.3f} '
                 f'>30 {s["P19>30"]:.3f} >40 {s["P19>40"]:.3f}  worst19 {s["worst19"]:.1f}')
    return '\n'.join(L)


def diagnostics(m, mhat, FUT, DF, c, infos, gap):
    L = []
    Dm = m[(m.index >= K.D_A) & (m.index < K.D_B)]
    L.append('1. market models: OOS corr(mhat, m) (2021 = calibration only), sign hit rate, E[sign(mhat) m] (bp)')
    vol = m.rolling(20, min_periods=10).std().shift(1)
    for mod in MODELS:
        mh = mhat[mod]
        row = []
        for y in (2021, 2022, 2023, 2024):
            idx = mh.index[(mh.index.year == y)].intersection(m.index)
            row.append(f'{y}: {np.corrcoef(mh[idx], m[idx])[0, 1]:+.3f}')
        idx = mh.index.intersection(Dm.index)
        cc = np.corrcoef(mh[idx], m[idx])[0, 1]
        hit = float((np.sign(mh[idx]) == np.sign(m[idx])).mean())
        allin = float((np.sign(mh[idx] - mh[idx].median()) * m[idx]).mean())
        tv = vol.reindex(idx); q = tv.quantile([1 / 3, 2 / 3]).to_numpy()
        terc = []
        for lo, hi, nm in ((-np.inf, q[0], 'low'), (q[0], q[1], 'mid'), (q[1], np.inf, 'high')):
            j = idx[(tv > lo).to_numpy() & (tv <= hi).to_numpy()]
            terc.append(f'{nm} {np.corrcoef(mh[j], m[j])[0, 1]:+.3f}')
        lam = [i.get('lam') for i in infos[mod] if i.get('lam') is not None]
        extra = f'  lambda median {np.median(lam):.0f}' if lam else ''
        if mod == 'MS':
            extra = f'  gap R2 (train) median {np.median([i["gap_r2"] for i in infos[mod]]):.3f}, b median {np.median([i["b"] for i in infos[mod]]):+.3f}'
        if mod == 'MV':
            extra = f'  features selected median {np.median([i["n_sel"] for i in infos[mod]]):.0f}'
        L.append(f'  {mod:4s} D corr {cc:+.3f} | ' + ' '.join(row) + f' | hit {hit:.3f} | all-in (vs median) {allin:+.2f} | vol terciles '
                 + ' '.join(terc) + extra)
    d = (mhat['MF'] - mhat['MB']).reindex(Dm.index)
    L.append(f'  MF - MB: corr diff by year ' + ' '.join(
        f'{y}: {np.corrcoef(mhat["MF"][Dm.index[Dm.index.year == y]], m[Dm.index[Dm.index.year == y]])[0, 1] - np.corrcoef(mhat["MB"][Dm.index[Dm.index.year == y]], m[Dm.index[Dm.index.year == y]])[0, 1]:+.3f}'
        for y in (2022, 2023, 2024)) + f'; corr(MF, MB) {np.corrcoef(mhat["MF"][Dm.index], mhat["MB"][Dm.index])[0, 1]:.3f}')
    L.append('')
    L.append('2. each futures feature vs m on D (Pearson; descriptive, in-sample; |t| ~ corr * sqrt(n))')
    FD = FUT.reindex(Dm.index)
    rows = []
    for col in FUT.columns:
        x = FD[col]; ok = x.notna()
        if ok.sum() < 60:
            rows.append((col, np.nan, int(ok.sum()))); continue
        rows.append((col, float(np.corrcoef(x[ok], Dm[ok])[0, 1]), int(ok.sum())))
    for col, cc, n in sorted(rows, key=lambda r: -abs(r[1]) if r[1] == r[1] else 0):
        L.append(f'  {col:28s} corr {cc:+.3f}  n {n}')
    L.append('')
    # 3. ceilings on D with the WF3 ranking
    L.append('3. ceilings on D (bp/day), WF3 ranking; ex-dividend rule only where noted')
    ds = [d for d in sorted(DF) if K.D_A <= d < K.D_B]
    best_k, allin, perf_rank, perf, base = [], [], [], [], []
    for d in ds:
        f = DF[d]; r = f['r']; n = len(r)
        ars = [K.ar_of(K.topk(f, k), f) for k in range(0, n + 1)]
        best_k.append(max(ars)); base.append(K.ar_of(K.topk(f, 25), f))
        allin.append(abs(r.mean()) * 1e4)
        o = np.argsort(-r); s = -np.ones(n); s[o[:25]] = 1; perf_rank.append(float((s * r).mean() * 1e4))
        perf.append(float(np.abs(r).mean() * 1e4))
    L.append(f'  DL0 (Top25 + ex-div)                         {np.mean(base):8.2f}')
    L.append(f'  (a) our ranking, best k known ex post        {np.mean(best_k):8.2f}')
    L.append(f'  (b) m known: all 1 if m > 0 else all 0       {np.mean(allin):8.2f}   (= E|m|)')
    L.append(f'  (c) true ranking, Top25                      {np.mean(perf_rank):8.2f}')
    L.append(f'  (d) everything known                         {np.mean(perf):8.2f}   (= mean |r|)')
    L.append('')
    # 4. how good must the market forecast be: synthetic z = rho m / sd + sqrt(1 - rho^2) eps, known calibration
    L.append('4. value of a market forecast with OOS correlation rho (synthetic, known calibration mu = rho * sd_m * z; 20 seeds)')
    mD = m.reindex(ds); sd_m = float(mD.std()); mean_m = float(mD.mean())
    cD = c.reindex(ds)
    sig_a = float(np.nanmean(cD))                                    # c is bp per unit q; q has sd ~1
    base_ar = np.array(base)
    for rho in (0.05, 0.10, 0.15, 0.20, 0.30):
        gains = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            z = rho * (mD.to_numpy() - mean_m) / sd_m + np.sqrt(1 - rho ** 2) * rng.standard_normal(len(ds))
            mu = mean_m + rho * sd_m * z
            ar = []
            for j, d in enumerate(ds):
                cj = cD.iloc[j]
                s = K.decide(DF[d], 'DL1', mu=mu[j], c=cj) if np.isfinite(cj) else K.topk(DF[d], 25)
                ar.append(K.ar_of(s, DF[d]))
            gains.append(np.mean(ar) - base_ar.mean())
        formula = rho ** 2 * sd_m ** 2 / (sig_a * np.sqrt(2 * np.pi))
        L.append(f'  rho {rho:.2f}: gain {np.mean(gains):+6.2f} bp/day (sd over seeds {np.std(gains):.2f}); formula rho^2 sd_m^2 / (sd_a sqrt(2 pi)) '
                 f'= {formula:+6.2f}  (sd_m {sd_m:.1f}, sd_a = c {sig_a:.1f})')
    return '\n'.join(L)


def run_d():
    if os.path.exists(K.LOCK):
        print('LOCK.json exists: D was already evaluated (results\\d_results.txt).'); return
    T50, m, days, DAYF, FUT, gap, X, snap = build()
    LOG(f'D run: futures features {FUT.shape[1]}, dropped {snap["dropped_sources"]}')
    mhat, infos = models_oos(X, m, gap, DAYF, end=K.D_B - pd.Timedelta(days=1))
    sc = K.wf3_scores(days)
    beta = K.betas_for(T50)
    DF = K.day_frame(T50[(T50.index.get_level_values('date') >= K.D_A) & (T50.index.get_level_values('date') < K.D_B)], sc, beta)
    days_eval = [d for d in sorted(DF)]
    R, extra = evaluate(DF, days_eval, m, mhat, 'D')
    res = summarize(R, 'D')
    # H2: P2 - P1
    d21 = (R['P2'].AR - R['P1'].AR).dropna()
    h2 = dict(diff=float(d21.mean()), t=float(K.nw_t(d21.to_numpy())), years=K.years(d21))
    passed = sorted([k for k, v in res.items() if v.get('pass')], key=lambda k: -res[k]['diff'])[:3]
    txt = fmt_table(res, f'race_mkt D (2022-01-03 .. 2024-12-31), {res["DL0"]["days"]} days; candidates vs DL0 (bp/day, NW 5 lags)')
    txt += f'\n\nH2 (futures information): P2 - P1 = {h2["diff"]:+.2f} (t {h2["t"]:.2f}) years ' + \
           ' '.join(f'{y}:{v:+.2f}' for y, v in h2['years'].items())
    txt += f'\n\npassed the D rule (mean > 0, NW t >= 2, >= 2 of 3 years > 0): {passed if passed else "none"}'
    cD = extra['c']
    txt += f'\nstock calibration c (bp per unit normal score): median {cD.median():.2f}, first usable {cD.first_valid_index()}'
    for mod in MODELS:
        mu = extra['mu'][mod]
        txt += f'\n  {mod}: calibration slope b median {mu.b.median():.3f} (share > 0 {(mu.b > 0).mean():.2f}), |mu| median {mu.mu.abs().median():.2f} bp'
    diag = diagnostics(m, mhat, FUT, DF, cD, infos, gap)
    with open(os.path.join(K.RES, 'd_results.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    with open(os.path.join(K.RES, 'd_diag.txt'), 'w', encoding='utf-8') as fh:
        fh.write(diag + '\n')
    daily = pd.concat({k: v.AR for k, v in R.items()}, axis=1)
    daily = pd.concat([daily, pd.concat({k + '_ones': v.ones for k, v in R.items() if k != 'DL0'}, axis=1),
                       m.reindex(daily.index).rename('m'), pd.concat({mod: mhat[mod] for mod in MODELS}, axis=1).add_prefix('mhat_').reindex(daily.index)], axis=1)
    daily.to_csv(os.path.join(K.RES, 'd_daily.csv'))
    out = {'results': res, 'h2': h2, 'passed': passed, 'snapshot': snap}
    with open(os.path.join(K.RES, 'd_results.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)
    lock = {'written': time.strftime('%Y-%m-%d %H:%M:%S'), 'prereg_sha256': K.sha256(K.PREREG), 'code': code_files(),
            'ext': snap['ext'], 'passed': passed, 'd_summary': {k: {'diff': v.get('diff'), 'diff_t': v.get('diff_t')} for k, v in res.items()}}
    with open(K.LOCK, 'w', encoding='utf-8') as fh:
        json.dump(lock, fh, ensure_ascii=False, indent=1, default=float)
    print(txt); print(); print(diag)


def seal_v():
    if not os.path.exists(K.LOCK):
        print('no LOCK.json: run --d first'); return
    if os.path.exists(K.SEALED):
        print('SEALED_V.json exists: V was already used once (results\\v_results.txt).'); return
    lock = json.load(open(K.LOCK, encoding='utf-8'))
    if lock['prereg_sha256'] != K.sha256(K.PREREG):
        print('PREREG changed since LOCK: refusing'); return
    changed = [f for f, h in code_files().items() if lock['code'].get(f) != h]
    if changed:
        print(f'code changed since LOCK ({changed}): refusing'); return
    T50, m, days, DAYF, FUT, gap, X, snap = build()
    mhat, infos = models_oos(X, m, gap, DAYF, end=K.LAST_LABEL)
    sc = K.clu_scores(K.L.FOLDS)                                      # 2022 .. 2026 folds (D folds feed the calibration)
    beta = K.betas_for(T50)
    DF = K.day_frame(T50[T50.index.get_level_values('date') >= K.D_A], sc, beta)
    days_eval = [d for d in sorted(DF) if K.V_A <= d < K.V_B]
    R, extra = evaluate(DF, days_eval, m, mhat, 'V')
    res = summarize(R, 'V')
    adopt = [k for k in lock['passed'] if res[k]['diff'] > 0]
    txt = fmt_table(res, f'race_mkt V (2025-01-02 .. 2026-11-09), {res["DL0"]["days"]} days; C.LU annual scores; candidates vs DL0')
    txt += f'\n\nD passed: {lock["passed"] or "none"}; adopted (D passed and V diff > 0): {adopt or "none"}'
    vm = m[(m.index >= K.V_A) & (m.index < K.V_B)]
    txt += '\n\nOOS corr(mhat, m) on V: ' + ' '.join(f'{mod} {np.corrcoef(mhat[mod].reindex(vm.index), vm)[0, 1]:+.3f}' for mod in MODELS)
    with open(os.path.join(K.RES, 'v_results.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    daily = pd.concat({k: v.AR for k, v in R.items()}, axis=1)
    daily = pd.concat([daily, pd.concat({k + '_ones': v.ones for k, v in R.items() if k != 'DL0'}, axis=1), m.reindex(daily.index).rename('m')], axis=1)
    daily.to_csv(os.path.join(K.RES, 'v_daily.csv'))
    sealed = {'written': time.strftime('%Y-%m-%d %H:%M:%S'), 'lock_written': lock['written'], 'results': res, 'adopted': adopt}
    with open(K.SEALED, 'w', encoding='utf-8') as fh:
        json.dump(sealed, fh, ensure_ascii=False, indent=1, default=float)
    print(txt)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true'); ap.add_argument('--d', action='store_true')
    ap.add_argument('--seal-v', action='store_true')
    a = ap.parse_args()
    if a.check:
        check()
    if a.d:
        run_d()
    if a.seal_v:
        seal_v()

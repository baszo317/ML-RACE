r"""
stage2_validate.py -- RACE-STD v1: the single use of V (research\PREREG_race_std.md section 3).

  python stage2_validate.py            score the LOCK finalists on V once, apply the rules of 3.1 and 3.2, write
                                       results\SEALED_V.json, stage2_V.txt, stage2_V_daily.csv; refuses if SEALED_V.json exists
  python stage2_validate.py --audit    recompute and compare with SEALED_V.json (no new decision, nothing overwritten)
  --test                               quick-test units and LOCK_test.json (the numbers mean nothing)
"""
import os, sys, json, time, argparse, platform
import numpy as np, pandas as pd
import rslib as L

OCT = ('2026-10-01', '2026-11-10')
B_BOOT, BLOCK, SEED = 5000, 10, 20261115


def check(lock, tag):
    now = L.code_hashes()
    bad = [f for f, h in lock['code'].items() if now.get(f) != h]
    extra = [f for f in now if f not in lock['code']]
    if bad or extra:
        sys.exit(f'code differs from LOCK{tag}.json: changed {bad}, new {extra} (PREREG 2.4)')
    if L.sha256(L.PREREG) != lock['prereg_sha256']:
        sys.exit('PREREG_race_std.md changed after the LOCK')
    snap = json.load(open(os.path.join(L.RESULTS, 'snapshot.json'), encoding='utf-8'))
    if snap['data_sha256'] != lock['data_sha256']:
        sys.exit('data snapshot changed after the LOCK')


def sb_idx(n, B, block, rng):
    """stationary bootstrap (Politis & Romano 1994), mean block length `block`"""
    start = rng.integers(n, size=(B, n)); new = rng.random((B, n)) < 1.0 / block
    new[:, 0] = True
    idx = np.empty((B, n), np.int64); idx[:, 0] = start[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(new[:, t], start[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def boot_means(X, idx, chunk=500):
    out = np.empty((idx.shape[0], X.shape[1]))
    for i in range(0, idx.shape[0], chunk):
        out[i:i + chunk] = X[idx[i:i + chunk]].mean(axis=1)
    return out


def mcs(AR, idx):
    """Hansen, Lunde & Nason (2011), T_max, loss = -AR; returns the MCS p-value of every column"""
    Lm = -AR; n, K = Lm.shape
    bm = boot_means(Lm, idx)
    M = list(range(K)); prun = 0.0; pv = {}
    while len(M) > 1:
        lb = Lm[:, M].mean(0); d = lb - lb.mean()
        bb = bm[:, M]; bd = bb - bb.mean(1, keepdims=True)
        sd = np.sqrt(((bd - d) ** 2).mean(0))
        t = d / sd
        Tb = ((bd - d) / sd).max(1)
        p = float((Tb >= t.max()).mean()); prun = max(prun, p)
        worst = M[int(np.argmax(t))]; pv[worst] = prun; M.remove(worst)
    pv[M[0]] = 1.0
    return pv


def spa(AR, b, idx):
    """Hansen (2005) SPA_c: H0 = no column beats column b; returns (p, T)"""
    n, K = AR.shape
    o = [k for k in range(K) if k != b]
    d = AR[:, o] - AR[:, [b]]; db = d.mean(0)
    bm = boot_means(d, idx)
    om = np.sqrt(n * ((bm - db) ** 2).mean(0)); om = np.where(om > 0, om, np.nan)
    st = np.sqrt(n) * db / om
    T = max(0.0, float(np.nanmax(st)))
    mu = np.where(st <= -np.sqrt(2 * np.log(np.log(n))), db, 0.0)
    Tb = np.maximum(0.0, np.nanmax(np.sqrt(n) * (bm - db + mu) / om, axis=1))
    return float((Tb > T).mean()), T


def ridge_mhat(day, INFO_all, folds, log):
    """Chen's market model (PREREG 3.2): Ridge on the day-level C features, target = Top50 mean r of the day"""
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV
    m = INFO_all[INFO_all.top50.astype(bool)].groupby('date').target_r.mean()
    X = day.copy(); X.index = pd.DatetimeIndex(X.index).astype('datetime64[ns]')
    y = m.reindex(X.index)
    out = []
    for fa, fb in folds:
        d = X.index
        tr = (d >= L.TRAIN_START) & (d < pd.Timestamp(fa)) & y.notna().values
        te = (d >= pd.Timestamp(fa)) & (d < pd.Timestamp(fb))
        pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler()),
                         ('r', RidgeCV(alphas=np.logspace(0, 5, 16)))])
        pipe.fit(X[tr].to_numpy(float), y[tr].to_numpy(float))
        out.append(pd.Series(pipe.predict(X[te].to_numpy(float)), index=d[te]))
        log(f'  Ridge fold {L.FOLD_NAME[fa]}: alpha {pipe.named_steps["r"].alpha_:.3g}, {int(tr.sum())} training days')
    return pd.concat(out).sort_index()


def score_of(name, defs, units):
    return L.candidate_score(name, defs, units)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audit', action='store_true')
    ap.add_argument('--test', action='store_true')
    a = ap.parse_args()
    tag = '_test' if a.test else ''
    lock_f = os.path.join(L.RESULTS, f'LOCK{tag}.json'); seal_f = os.path.join(L.RESULTS, f'SEALED_V{tag}.json')
    if not os.path.exists(lock_f):
        sys.exit(f'no {lock_f}: run freeze.py first')
    lock = json.load(open(lock_f, encoding='utf-8'))
    check(lock, tag)
    if os.path.exists(seal_f) and not a.audit:
        sys.exit(f'{seal_f} exists: V has already been used once (PREREG 3). Use --audit to recompute and compare.')
    log = L.log_to(os.path.join(L.RESULTS, f'stage2{tag}.log'))
    t0 = time.time()
    defs = lock['defs']; fin = lock['finalists']
    INFO_all = L.load_info(); INFO = L.info_frame(INFO_all)
    units, miss = {}, []
    for u in lock['finalist_units']:
        s = L.load_unit(u, L.V_FOLDS, a.test)
        if s is None:
            miss.append(u)
        else:
            units[u] = s
    if miss:
        sys.exit(f'V units missing: {miss} -- run train.py --stage 2')
    va, vb = L.PER['V']
    daily, res = {}, {}
    for n in fin:
        D = L.decide_sd(score_of(n, defs, units), INFO)
        D = D[L.in_per(D.index, va, vb)]
        daily[n] = D
        sm = L.summary(D, va, vb); sm['folds'] = L.by_fold(D, L.V_FOLDS)
        sm['oct_nov'] = float(D[L.in_per(D.index, *OCT)].AR.mean())
        w19 = D.AR.rolling(19).mean().dropna()
        sm['p19_neg'] = float((w19 < 0).mean()); sm['q05_19'] = float(w19.quantile(0.05))
        res[n] = sm
    AR = pd.DataFrame({n: daily[n].AR for n in fin}).dropna()
    # ---------------- 3.1 main decision
    w = max(fin, key=lambda n: res[n]['mean'])
    se = {n: (0.0 if n == w else float(L.nw_se((AR[w] - AR[n]).values))) for n in fin}
    within = [n for n in fin if res[w]['mean'] - res[n]['mean'] <= se[n] + 1e-12]
    cx = {n: tuple(defs[n]['complexity']) for n in fin}
    chosen = min(within, key=lambda n: (cx[n][0], cx[n][1], -res[n]['mean']))
    log(f'V winner {w} ({res[w]["mean"]:+.2f}); within 1 SE: {within}; chosen by the 1-SE rule: {chosen}')
    # ---------------- 3.2 market view for the chosen stock view
    day = pd.read_pickle(os.path.join(L.CACHE, 'day_features.pkl'))
    F = L.features(); day = day[[c for c in F['C_DAY'] if c in day.columns]]
    mh = ridge_mhat(day, INFO_all, L.FOLDS, log)
    sunits = {}
    for u in defs[chosen]['units']:
        sunits[u] = pd.concat([L.load_unit(u, L.D_FOLDS, a.test), units[u]]).sort_index()
    sc = score_of(chosen, defs, sunits)
    da, db = L.PER['D']
    D_sd = L.decide_sd(sc, INFO)
    ic_d = float(D_sd[L.in_per(D_sd.index, da, db)].ic.mean())
    E = INFO_all.copy(); E = E[E.resid_r.notna()]
    sig = E.groupby('date').resid_r.std().rolling(60, min_periods=20).mean().shift(1)
    z = L.zday(sc)
    dts = z.index.get_level_values('date')
    s_hat = ic_d * sig.reindex(dts).to_numpy() * z.to_numpy()
    mv = {}
    for lam in (1.0, 0.5):
        smv = pd.Series(lam * mh.reindex(dts).to_numpy() + s_hat, index=z.index)
        mv[lam] = L.decide_sd(smv, INFO, mode='zero')
    def per_mean(Dx, a_, b_):
        return float(Dx[L.in_per(Dx.index, a_, b_)].AR.mean())
    mvres = {}
    for lam, Dm in mv.items():
        dd = (Dm.AR - D_sd.AR).dropna()
        mvres[str(lam)] = dict(D_diff=per_mean(dd.to_frame('AR'), da, db), D_t=float(L.nw_t(dd[L.in_per(dd.index, da, db)].values)),
                               V_diff=per_mean(dd.to_frame('AR'), va, vb), V_t=float(L.nw_t(dd[L.in_per(dd.index, va, vb)].values)),
                               V_mean=per_mean(Dm, va, vb), D_mean=per_mean(Dm, da, db),
                               ones_V=float(Dm[L.in_per(Dm.index, va, vb)].ones.mean()))
    add_mv = mvres['1.0']['D_diff'] > 0 and mvres['1.0']['V_diff'] > 0
    log(f'market view (lambda 1): D diff {mvres["1.0"]["D_diff"]:+.2f}, V diff {mvres["1.0"]["V_diff"]:+.2f} -> '
        f'{"ADD" if add_mv else "do not add"}')
    # ---------------- report only
    rng = np.random.default_rng(SEED)
    idx = sb_idx(len(AR), B_BOOT, BLOCK, rng)
    X = AR.to_numpy()
    pv = mcs(X, idx)
    mcs_p = {fin[k]: v for k, v in pv.items()}
    nat = {'U-native': 'U-native', 'P-native': 'P.LP', 'C-native': 'C.LC', 'M-native': 'M.LM'}
    spa_r = {}
    for k, v in nat.items():
        if v in fin:
            p, T = spa(X, fin.index(v), idx)
            spa_r[k] = dict(p=p, T=T)
    bm = boot_means(X, idx)
    pbeat = {k: float((bm[:, fin.index(chosen)] > bm[:, fin.index(v)]).mean()) for k, v in nat.items() if v in fin and v != chosen}
    mu = np.array([res[n]['mean'] for n in fin]); s2 = np.array([L.nw_se(AR[n].values) ** 2 for n in fin])
    K = len(fin); dev = mu - mu.mean()
    shrink = max(0.0, 1 - (K - 3) * s2.mean() / max((dev ** 2).sum(), 1e-12)) if K > 3 else 1.0
    js = {n: float(mu.mean() + shrink * dev[i]) for i, n in enumerate(fin)}
    corr = AR.corr()
    sealed = dict(written=time.strftime('%Y-%m-%d %H:%M:%S'), lock_written=lock['written'], period=[va, str(AR.index.max().date())],
                  days=int(len(AR)), finalists=fin, results=res,
                  decision=dict(winner=w, se_vs_winner=se, within_1se=within, complexity={n: list(cx[n]) for n in fin},
                                chosen=chosen, chosen_def=defs[chosen]),
                  market_view=dict(stock_view=chosen, ic_D=ic_d, results=mvres, add=add_mv,
                                   ridge_features=list(day.columns)),
                  report_only=dict(mcs_p=mcs_p, mcs90=[n for n in fin if mcs_p[n] >= 0.10], spa_vs_native=spa_r,
                                   p_chosen_beats_native=pbeat, james_stein=js, js_shrink=shrink,
                                   corr=corr.round(4).to_dict()),
                  versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__), test=a.test)
    if a.audit:
        old = json.load(open(seal_f, encoding='utf-8'))
        diffs = {n: abs(old['results'][n]['mean'] - res[n]['mean']) for n in fin}
        same = max(diffs.values()) < 1e-9 and old['decision']['chosen'] == chosen and old['market_view']['add'] == add_mv
        print(f'AUDIT: {"identical" if same else "DIFFERENT"}; max |mean difference| {max(diffs.values()):.3g} bp; '
              f'chosen {old["decision"]["chosen"]} -> {chosen}; market view {old["market_view"]["add"]} -> {add_mv}')
        return
    with open(seal_f, 'w', encoding='utf-8') as fh:
        json.dump(sealed, fh, indent=1, ensure_ascii=False, default=float)
    pd.concat({n: daily[n] for n in fin}, axis=1).to_csv(os.path.join(L.RESULTS, f'stage2_V_daily{tag}.csv'))
    # ---------------- text report
    lines = [f'RACE-STD v1 Stage 2 -- V = {va} .. {AR.index.max().date()} ({len(AR)} days), the single use of V',
             f'sealed {sealed["written"]} (LOCK {lock["written"]}); decision SD = Top25 of Top50 + ex-dividend rule', '',
             'finalist          units feats    mean   NW t     sd   pos%    IC  | 2025  2026-01..05 2026-06.. | Oct-Nov  P(19d<0)  MCS p']
    for n in sorted(fin, key=lambda n: -res[n]['mean']):
        r = res[n]; f = r['folds']
        lab = n + (' (' + [k for k, v in nat.items() if v == n][0] + ')' if n in nat.values() else '')
        lines.append(f'{lab:<24.24}{cx[n][0]:>3} {cx[n][1]:>5} {r["mean"]:+7.2f} {r["nw_t"]:6.2f} {r["sd"]:6.1f} {100 * r["pos"]:5.1f} '
                     f'{r["ic"]:6.3f} | {f["2025"]:+6.2f} {f["2026-01..05"]:+8.2f} {f["2026-06.."]:+9.2f} | {r["oct_nov"]:+7.2f} '
                     f'{r["p19_neg"]:8.2f} {mcs_p[n]:7.3f}')
    lines += ['', f'3.1  winner on V: {w} ({res[w]["mean"]:+.2f} bp/day)',
              '     within 1 SE of the winner (paired NW SE): ' + ', '.join(f'{n} ({res[w]["mean"] - res[n]["mean"]:+.2f} <= {se[n]:.2f})' for n in within),
              f'     -> race stock view (fewest units, then fewest features): {chosen}  units {defs[chosen]["units"]}', '',
              f'3.2  market view for {chosen}: IC_D {ic_d:.4f}; SD_MV - SD (lambda 1): D {mvres["1.0"]["D_diff"]:+.2f} (t {mvres["1.0"]["D_t"]:.2f}), '
              f'V {mvres["1.0"]["V_diff"]:+.2f} (t {mvres["1.0"]["V_t"]:.2f}) -> {"ADD the market view" if add_mv else "no market view"}',
              f'     report only, lambda 0.5: D {mvres["0.5"]["D_diff"]:+.2f}, V {mvres["0.5"]["V_diff"]:+.2f}; share of 1s on V (lambda 1) '
              f'{mvres["1.0"]["ones_V"] / 50:.2f}', '',
              'report only:',
              f'  MCS 90% (Hansen, Lunde & Nason 2011): {sealed["report_only"]["mcs90"]}',
              '  SPA_c p (H0: no finalist beats the native): ' + ', '.join(f'{k} {v["p"]:.3f}' for k, v in spa_r.items()),
              f'  P(bootstrap mean of {chosen} > native): ' + ', '.join(f'{k} {v:.2f}' for k, v in pbeat.items()),
              f'  James-Stein shrinkage factor {shrink:.2f}: ' + ', '.join(f'{n} {v:+.2f}' for n, v in js.items()),
              '', 'correlation of daily AR on V:', corr.round(2).to_string()]
    txt = '\n'.join(lines)
    with open(os.path.join(L.RESULTS, f'stage2_V{tag}.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    print(txt)
    log(f'sealed in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()

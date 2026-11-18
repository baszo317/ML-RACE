r"""
rs_report.py -- tables of the regularization / SVM study from the unit files of rs_train.py and rs_svm.py.
run (in research): python rs_report.py > ../results/rs_report.txt   (works on partial results; missing parts are skipped)
"""
import os, sys, glob, pickle, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rslib as L
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 400)
FA = [fa for fa, _ in L.FOLDS]
UNITS = os.path.join(L.CACHE, 'units'); SVM = os.path.join(L.CACHE, 'svm')
OUTCSV = L.RESULTS


# ------------------------------------------------------------------ loading
def load_train():
    U = {}
    for f in glob.glob(os.path.join(UNITS, '*', '*', '*', '*.pkl')):
        r = pickle.load(open(f, 'rb'))
        member, pset, a, seed, frac = r['unit']
        U.setdefault((member, pset, frac), {}).setdefault(a, {})[seed] = r
    return U


def assemble(U, member, pset, frac, snap, seeds=None):
    """seed-mean OOS predictions over the 6 folds (Series on date, stock_id), None if a fold / seed / snapshot is missing"""
    d = U.get((member, pset, frac))
    if d is None or any(a not in d for a in FA):
        return None
    parts = []
    for a in FA:
        ss = sorted(d[a]) if seeds is None else [s for s in seeds if s in d[a]]
        if not ss or (seeds is not None and len(ss) < len(seeds)) or any(snap not in d[a][s]['snap'] for s in ss):
            return None
        k = d[a][ss[0]]['key']
        p = np.mean([d[a][s]['snap'][snap] for s in ss], axis=0)
        parts.append(pd.Series(p, index=pd.MultiIndex.from_frame(k.assign(date=pd.to_datetime(k.date)))))
    return pd.concat(parts).sort_index()


def n_seeds(U, member, pset, frac):
    d = U.get((member, pset, frac), {})
    return min((len(d[a]) for a in FA if a in d), default=0) if all(a in d for a in FA) else 0


# ------------------------------------------------------------------ metrics on the assembled series
def ic_by_period(p, Pp):
    r = Pp.resid.reindex(p.index)
    ic = L.rank_ic_daily(p.index.get_level_values('date'), p.values, r.values)
    pr = L.periods(ic.index)
    return {k: float(ic[v].mean()) for k, v in pr.items()}, ic


def corr_by_period(p, Pp):
    pb = p.groupby(level='date').mean()
    q = pd.DataFrame({'r': Pp.tr, 'top': Pp.top50}).reindex(p.index)
    m = q[q.top.astype(bool)].r.groupby(level='date').mean()
    j = pb.index.intersection(m.index); pb, m = pb[j], m[j]
    pr = L.periods(j)
    return {k: float(np.corrcoef(pb[v], m[v])[0, 1]) for k, v in pr.items()}


def wll_by_period(p, member, Pp):
    P = L.panel().set_index(['date', 'stock_id'])
    mcfg = L.members()[member]
    tcol, ycol = ('target_r', 'y') if mcfg['target'] == 'raw' else ('resid_r', 'y_resid')
    X = P.reindex(p.index)
    w = np.minimum(np.abs(X[tcol].values), 0.05)
    if mcfg['sig']:
        s = X[mcfg['sig']].values.astype(float); s = np.where(np.isfinite(s) & (s > 0), s, np.nanmedian(s)); w = w / s ** 0.5
    y = X[ycol].values.astype(float)
    pr = L.periods(p.index.get_level_values('date'))
    return {k: L.wll(p.values[v], y[v], w[v]) for k, v in pr.items()}


def fmt_period(d, nd=4):
    return ' / '.join(f'{d[k]:.{nd}f}' for k in ('val', 'hold', 'post', 'all'))


# ------------------------------------------------------------------ report parts
def part_members(U, Pp):
    print('=' * 120)
    print('1. Member level: production regularization vs removed / stronger (OOS, seed mean, walk-forward 2022-01 .. 2026-10)')
    print('=' * 120)
    rows = []
    spec = [('lgb_res', 'cur', 'es', 'production (ES, 15 leaves, min_data_in_leaf 500, bagging 0.8, lambda_l2 1)'),
            ('lgb_res', 'cur', 'n1000', 'no ES: 1000 trees, other settings as production'),
            ('lgb_res', 'cur', 'n2000', 'no ES: 2000 trees (seed 1 only)'),
            ('lgb_res', 'leaf20', 'es', 'min_data_in_leaf 500 -> 20, ES'),
            ('lgb_res', 'nol2bag', 'es', 'lambda_l2 0, no bagging / feature fraction, ES'),
            ('lgb_res', 'none', 'es', 'all penalties off, ES kept'),
            ('lgb_res', 'none', 'n1000', 'ALL OFF: penalties off and no ES (1000 trees)'),
            ('lgb_res', 'none', 'n2000', 'ALL OFF, 2000 trees'),
            ('lgb_res', 'strong', 'es', 'stronger: 7 leaves, min_data_in_leaf 2000, lambda_l2 10, ES'),
            ('lgb_res', 'default', 'n100', 'LightGBM defaults (lr 0.1, 31 leaves, min 20, 100 trees)'),
            ('mlp_res_xr9', 'cur', 'es', 'production (dropout 0.2, weight decay 1e-2, ES)'),
            ('mlp_res_xr9', 'cur', 'e10', 'no ES: 10 epochs'), ('mlp_res_xr9', 'cur', 'e20', 'no ES: 20 epochs'),
            ('mlp_res_xr9', 'nodw', 'es', 'dropout 0, weight decay 0, ES kept'),
            ('mlp_res_xr9', 'nodw', 'e10', 'ALL OFF, 10 epochs'), ('mlp_res_xr9', 'nodw', 'e20', 'ALL OFF, 20 epochs'),
            ('mlp_res_xr9', 'strong', 'es', 'stronger: dropout 0.5, weight decay 1e-1, ES'),
            ('mlp_raw_B9', 'cur', 'es', 'production (dropout 0.2, weight decay 1e-2, ES)'),
            ('mlp_raw_B9', 'cur', 'e10', 'no ES: 10 epochs'), ('mlp_raw_B9', 'cur', 'e20', 'no ES: 20 epochs'),
            ('mlp_raw_B9', 'nodw', 'es', 'dropout 0, weight decay 0, ES kept'),
            ('mlp_raw_B9', 'nodw', 'e10', 'ALL OFF, 10 epochs'), ('mlp_raw_B9', 'nodw', 'e20', 'ALL OFF, 20 epochs'),
            ('mlp_raw_B9', 'strong', 'es', 'stronger: dropout 0.5, weight decay 1e-1, ES')]
    for member, pset, snap, desc in spec:
        seeds = [1] if snap == 'n2000' and pset == 'cur' else None
        p = assemble(U, member, pset, 1.0, snap, seeds)
        if p is None:
            continue
        d = U[(member, pset, 1.0)]
        ss = seeds or sorted(d[FA[0]])
        best = np.mean([d[a][s]['best'] for a in FA for s in ss if s in d[a]])
        ins = [d[a][s]['ins'].get(snap) for a in FA for s in ss if s in d[a]]
        ins = [x for x in ins if x]
        r = {'member': member, 'pset': pset, 'snap': snap, 'setting': desc, 'seeds': len(ss), 'ES stop (mean)': best}
        if member.startswith('mlp_raw'):
            c = corr_by_period(p, Pp); wl = wll_by_period(p, member, Pp)
            r.update({'in: corr': np.mean([x['corr'] for x in ins]) if ins else np.nan, 'in: wll': np.mean([x['wll'] for x in ins]) if ins else np.nan,
                      'OOS corr val': c['val'], 'hold': c['hold'], 'post': c['post'], 'all': c['all'], 'OOS wll all': wl['all']})
        else:
            ic, _ = ic_by_period(p, Pp); wl = wll_by_period(p, member, Pp)
            r.update({'in: IC': np.mean([x['ic'] for x in ins]) if ins else np.nan, 'in: wll': np.mean([x['wll'] for x in ins]) if ins else np.nan,
                      'OOS IC val': ic['val'], 'hold': ic['hold'], 'post': ic['post'], 'all': ic['all'], 'OOS wll all': wl['all']})
        rows.append(r)
    T = pd.DataFrame(rows)
    if T.empty:
        print('  (no units yet)\n'); return T
    for member in ('lgb_res', 'mlp_res_xr9', 'mlp_raw_B9'):
        X = T[T.member == member].drop(columns=['member']).dropna(axis=1, how='all')
        k = 'corr' if member.startswith('mlp_raw') else 'IC'
        order = ['pset', 'snap', 'setting', 'seeds', 'ES stop (mean)', f'in: {k}', 'in: wll', f'OOS {k} val', 'hold', 'post', 'all', 'OOS wll all']
        X = X[[c for c in order if c in X.columns]]
        if not X.empty:
            print(f'\n  {member}  (stock view: daily rank IC with the same-day excess return; market view: corr(pbar_t, Top50 m_t); '
                  f'wll = weighted log-loss of the training objective, lower = better; in = on its own training window)')
            print(X.round(4).to_string(index=False))
    T.to_csv(os.path.join(OUTCSV, 'rs_members.csv'), index=False, float_format='%.5f')
    print()
    return T


def part_pipeline(U, Pp):
    print('=' * 120)
    print('2. Competition score (Top50 AR bp/day): mini v9 = market view mlp_raw_B9, stock view (mlp_res_xr9 + lgb_res) / 2,')
    print("   v9's decision (ex-dividend flags, beta hedge lambda 0.75); d / t = paired difference to 'all production', NW t")
    print('=' * 120)
    G = lambda m, ps, sn, seeds=None: assemble(U, m, ps, 1.0, sn, seeds)
    combos = {
        'all production (ES + penalties)': (('mlp_raw_B9', 'cur', 'es'), ('mlp_res_xr9', 'cur', 'es'), ('lgb_res', 'cur', 'es')),
        'ALL OFF (penalties off, no ES: MLP 20 epochs, LightGBM 1000 trees)': (('mlp_raw_B9', 'nodw', 'e20'), ('mlp_res_xr9', 'nodw', 'e20'), ('lgb_res', 'none', 'n1000')),
        'ALL OFF, MLP 10 epochs': (('mlp_raw_B9', 'nodw', 'e10'), ('mlp_res_xr9', 'nodw', 'e10'), ('lgb_res', 'none', 'n1000')),
        'only ES removed (penalties as production)': (('mlp_raw_B9', 'cur', 'e20'), ('mlp_res_xr9', 'cur', 'e20'), ('lgb_res', 'cur', 'n1000')),
        'only penalties removed (ES kept)': (('mlp_raw_B9', 'nodw', 'es'), ('mlp_res_xr9', 'nodw', 'es'), ('lgb_res', 'none', 'es')),
        'stronger everywhere (ES kept), seed 1': (('mlp_raw_B9', 'strong', 'es'), ('mlp_res_xr9', 'strong', 'es'), ('lgb_res', 'strong', 'es')),
        'all production, seed 1 (to compare with the line above)': (('mlp_raw_B9', 'cur', 'es'), ('mlp_res_xr9', 'cur', 'es'), ('lgb_res', 'cur', 'es')),
        'market member ALL OFF only': (('mlp_raw_B9', 'nodw', 'e20'), ('mlp_res_xr9', 'cur', 'es'), ('lgb_res', 'cur', 'es')),
        'stock MLP ALL OFF only': (('mlp_raw_B9', 'cur', 'es'), ('mlp_res_xr9', 'nodw', 'e20'), ('lgb_res', 'cur', 'es')),
        'LightGBM ALL OFF only': (('mlp_raw_B9', 'cur', 'es'), ('mlp_res_xr9', 'cur', 'es'), ('lgb_res', 'none', 'n1000')),
        'LightGBM defaults only': (('mlp_raw_B9', 'cur', 'es'), ('mlp_res_xr9', 'cur', 'es'), ('lgb_res', 'default', 'n100')),
    }
    R = {}
    for name, (a, b, c) in combos.items():
        if 'seed 1' in name:
            sa = sb = sc = [1]
        else:
            sa = sb = sc = None
        ra, rb, rc = G(*a, sa), G(*b, sb), G(*c, sc)
        if any(x is None for x in (ra, rb, rc)):
            continue
        ra, rb, rc = ra.reindex(Pp.idx), rb.reindex(Pp.idx), rc.reindex(Pp.idx)
        assert ra.notna().all() and rb.notna().all() and rc.notna().all(), name
        R[name] = Pp.decide(Pp.score([[ra]], [[rb], [rc]]))
    R['(reference) v9 full, cached members'] = Pp.v9()
    if 'all production (ES + penalties)' not in R:
        print('  (production units missing)\n'); return R
    base = R['all production (ES + penalties)']
    rows = [{'config': k, **L.ar_row(D, base)} for k, D in R.items()]
    T = pd.DataFrame(rows)
    print(T[['config', 'val', 'hold', 'post', 'all', 'sd', 'timing', 'xsec']].round(2).to_string(index=False))
    print()
    print(T[['config', 'd_val', 't_val', 'd_hold', 't_hold', 'd_post', 't_post', 'd_all', 't_all']].round(2).to_string(index=False))
    T.to_csv(os.path.join(OUTCSV, 'rs_pipeline.csv'), index=False, float_format='%.4f')
    pd.to_pickle(R, os.path.join(L.CACHE, 'rs_pipeline_daily.pkl'))
    print()
    return R


def part_curves(U):
    print('=' * 120)
    print('3. OOS learning curves of the refit (mean over folds and seeds): production penalties vs all penalties off')
    print('=' * 120)
    for member, psets, key in (('lgb_res', ('cur', 'none', 'strong'), 'ic'), ('mlp_res_xr9', ('cur', 'nodw', 'strong'), 'ic'),
                               ('mlp_raw_B9', ('cur', 'nodw', 'strong'), 'corr')):
        cols = {}
        for ps in psets:
            d = U.get((member, ps, 1.0))
            if not d:
                continue
            C = pd.concat([d[a][s]['curve'].assign(fold=a, seed=s) for a in d for s in d[a]])
            C = C[C.split == 'oos']
            cols[f'{ps}: {key}'] = C.groupby('step')[key].mean()
            cols[f'{ps}: wll'] = C.groupby('step')['wll'].mean()
            cols[f'{ps}: n'] = C.groupby('step')[key].size()
        if not cols:
            continue
        T = pd.DataFrame(cols)
        if member == 'lgb_res':
            T = T[T.index.isin([10, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000])]
        print(f'\n  {member} (step = {"trees" if member == "lgb_res" else "epochs"}; n = fold x seed units averaged)')
        print(T.round(4).to_string())
    print()


def part_size(U, Pp):
    print('=' * 120)
    print('4. Data size: a random fraction of the training DAYS (same calendar window), OOS rank IC of the stock view members')
    print('=' * 120)
    rows = []
    for frac in (0.125, 0.25, 0.5, 1.0):
        r = {'frac': frac}
        d = U.get(('lgb_res', 'cur', frac))
        if d and all(a in d for a in FA):
            r['train days (mean)'] = np.mean([d[a][s]['days_train'] for a in FA for s in d[a]])
            r['train rows (mean)'] = np.mean([d[a][s]['n_train'] for a in FA for s in d[a]])
        for (m, ps, sn, lab) in (('lgb_res', 'cur', 'es', 'LGB production'), ('lgb_res', 'none', 'n1000', 'LGB ALL OFF'),
                                 ('lgb_res', 'none', 'es', 'LGB penalties off, ES'),
                                 ('mlp_res_xr9', 'cur', 'es', 'MLP production'), ('mlp_res_xr9', 'nodw', 'e20', 'MLP ALL OFF'),
                                 ('mlp_res_xr9', 'nodw', 'es', 'MLP penalties off, ES')):
            seeds = [1, 2] if ps in ('cur', 'nodw') else [1]
            p = assemble(U, m, ps, frac, sn, seeds)
            if p is not None:
                ic, _ = ic_by_period(p, Pp); r[lab] = ic['all']
                dd = U[(m, ps, frac)]
                ins = [dd[a][s]['ins'].get(sn) for a in FA for s in seeds if s in dd[a]]
                ins = [x for x in ins if x]
                r[lab + ' (in)'] = np.mean([x['ic'] for x in ins]) if ins else np.nan
        rows.append(r)
    T = pd.DataFrame(rows)
    if len(T.columns) > 1:
        print(T.round(4).to_string(index=False))
        T.to_csv(os.path.join(OUTCSV, 'rs_size.csv'), index=False, float_format='%.5f')
    print('  (seeds 1-2 for production and MLP ALL OFF at every size, so the columns are comparable down the rows; (in) = IC on the '
          'model\'s own training rows)\n')
    return T


def part_seeds(U, Pp):
    print('=' * 120)
    print('5. Seed averaging (an implicit regularizer): OOS rank IC (all) of one seed vs the seed mean')
    print('=' * 120)
    rows = []
    for m, ps, sn in (('lgb_res', 'cur', 'es'), ('mlp_res_xr9', 'cur', 'es'), ('mlp_res_xr9', 'nodw', 'e20')):
        r = {'member': m, 'pset': ps, 'snap': sn}
        for s in (1, 2, 3):
            p = assemble(U, m, ps, 1.0, sn, [s])
            if p is not None:
                r[f'seed {s}'] = ic_by_period(p, Pp)[0]['all']
        p = assemble(U, m, ps, 1.0, sn, [1, 2, 3])
        if p is not None:
            r['mean of 3'] = ic_by_period(p, Pp)[0]['all']
        rows.append(r)
    print(pd.DataFrame(rows).round(4).to_string(index=False)); print()


# ------------------------------------------------------------------ SVM
def load_svm():
    S = {}
    for f in glob.glob(os.path.join(SVM, '*.pkl')):
        r = pickle.load(open(f, 'rb')); S[tuple(r['unit'])] = r
    return S


def svm_series(S, kind, view, C=None, rule=None, prob=False):
    parts = []
    for a in FA:
        r = S.get((kind, view, a))
        if r is None:
            return None, None
        rows = pd.DataFrame(r['rows'])
        if rule == 'hinge':
            c = float(rows.C[rows.val_hinge.idxmin()])
        elif rule == 'metric':
            c = float(rows.C[rows.val_metric.idxmax()])
        else:
            c = C
        v = (r['prob'] if prob else r['oos'])[c]
        k = r['key']
        parts.append((a, c, pd.Series(v, index=pd.MultiIndex.from_frame(k.assign(date=pd.to_datetime(k.date))))))
    return pd.concat([x[2] for x in parts]).sort_index(), [x[1] for x in parts]


def part_svm(S, Pp, Tm):
    print('=' * 120)
    print('6. SVM members (same folds, features B 67, labels, weights and ES split as lgb_res / v8 logistic)')
    print('=' * 120)
    TB = Pp.TB
    ref = {}
    for nm, col in (('lgb_res (cache, seed 1-5)', 'lgb_resid'), ('logistic regression (v8 lr_resid)', 'lr_resid')):
        ref[nm] = ic_by_period(TB[col], Pp)[0]
    ref['mlp_res_xr9 (cache, seed 1-10)'] = ic_by_period(Pp.W9.mlp_res_xr9, Pp)[0]
    print('\n  reference stock-view members, OOS rank IC val / hold / post / all:')
    for k, v in ref.items():
        print(f'    {k:40s} {fmt_period(v)}')
    refm = {}
    for nm, ser in (('mlp_raw_B9 (cache, seed 1-10)', Pp.W9.mlp_raw_B9), ('Transformer tf_raw_B (cache)', Pp.tf),
                    ('logistic regression (v8 lr_raw)', TB.lr_raw), ('v9 market view (3 MLP + TF)', None)):
        if ser is None:
            p_raw = pd.concat([pd.concat(Pp.raw9, axis=1).mean(axis=1), Pp.tf], axis=1).mean(axis=1)
            ser = p_raw
        refm[nm] = corr_by_period(ser, Pp)
    print('  reference market-view members, corr(pbar, m) val / hold / post / all:')
    for k, v in refm.items():
        print(f'    {k:40s} {fmt_period(v, 3)}')
    out = {}
    for kind, Cs in (('lin', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]), ('nys', [1e-3, 1e-2, 1e-1])):
        for view in ('res', 'raw'):
            have = [S.get((kind, view, a)) for a in FA]
            if any(h is None for h in have):
                continue
            print(f'\n  {kind} SVM, {"stock view (label beats the day average)" if view == "res" else "market view (label close > open)"}: '
                  f'path over C (OOS = all folds pooled; val = last 120 days of each window; in = own window)')
            rows = []
            for C in Cs:
                ser, _ = svm_series(S, kind, view, C=C)
                X = pd.concat([pd.DataFrame(h['rows']).assign(fold=a) for a, h in zip(FA, have)])
                X = X[X.C == C]
                r = {'C': C, 'lambda = 1/(2Cn)': 1 / (2 * C * 150000), 'val hinge': X.val_hinge.mean(), 'val metric': X.val_metric.mean(),
                     'in hinge': X.in_hinge.mean(), 'in metric': X.in_metric.mean(), 'OOS hinge': X.oos_hinge.mean(),
                     'share with y f <= 1': X.frac_sv_proxy.mean(), '|w|': X.w_norm.mean(), 'iters (max)': X.iters.max(),
                     'sec fit': (X.sec_es + X.sec_full).mean()}
                if view == 'res':
                    ic = ic_by_period(ser, Pp)[0]; r.update({f'OOS IC {k}': ic[k] for k in ('val', 'hold', 'post', 'all')})
                else:
                    c = corr_by_period(ser, Pp); r.update({f'OOS corr {k}': c[k] for k in ('val', 'hold', 'post', 'all')})
                    r['OOS wll (Platt)'] = X.oos_wll.mean()
                rows.append(r)
            T = pd.DataFrame(rows)
            print(T.round(4).to_string(index=False))
            for rule in ('hinge', 'metric'):
                ser, cs = svm_series(S, kind, view, rule=rule)
                if view == 'res':
                    v = ic_by_period(ser, Pp)[0]
                else:
                    v = corr_by_period(ser, Pp)
                print(f'    C chosen per fold by val {"hinge loss (pre-registered)" if rule == "hinge" else "rank IC / corr"}: '
                      f'{", ".join(f"{c:g}" for c in cs)} -> OOS {fmt_period(v)}')
                out[(kind, view, rule)] = (ser, cs)
            T.to_csv(os.path.join(OUTCSV, f'rs_svm_{kind}_{view}.csv'), index=False, float_format='%.6f')
    # exact RBF
    g = S.get(('grid', 'res', '2024-01-01', 10000))
    if g:
        print('\n  exact RBF SVC, n = 10,000 rows of the 2024 fold ES part, scored on its last 120 days (val):')
        print(pd.DataFrame(g['rows']).round(4).to_string(index=False))
    rb = [S[k] for k in S if k[0] == 'rbf']
    if rb:
        X = pd.concat([pd.DataFrame(r['rows']).assign(fold=r['unit'][2]) for r in rb])
        print('\n  exact RBF SVC on n random rows of the window vs LightGBM (production parameters, min_data_in_leaf scaled) on the same rows:')
        rows = []
        for n, g2 in X.groupby('n'):
            r = {'n': n, 'folds': len(g2), 'fit sec': g2.sec_fit.mean(), 'predict sec': g2.sec_pred.mean(), 'support vectors': g2.n_sv.mean(),
                 'share SV': g2.frac_sv.mean(), 'SVM OOS IC (fold mean)': g2.oos_metric.mean(), 'LightGBM same rows': g2.lgb_oos_metric.mean()}
            rows.append(r)
        T = pd.DataFrame(rows); print(T.round(4).to_string(index=False))
        T.to_csv(os.path.join(OUTCSV, 'rs_svm_rbf_exact.csv'), index=False, float_format='%.5f')
        if len(T) >= 3:
            b = np.polyfit(np.log(T.n), np.log(T['fit sec']), 1)[0]
            print(f'    fit time ~ n^{b:.2f}; at n = 150,000 about {T["fit sec"].iloc[-1] * (150000 / T.n.iloc[-1]) ** b / 60:.0f} minutes per fit '
                  f'(one C, gamma, fold), predicting one day of 100 stocks needs ~{T["share SV"].iloc[-1] * 150000:,.0f} kernel rows')
    print()
    return out


def part_svm_pipeline(S, out, Pp):
    print('=' * 120)
    print('7. SVM in the competition score: v9 with members replaced (Top50 AR bp/day; d / t = paired difference to v9, NW t)')
    print('   stock view: the SVM decision value is centred per day and rescaled to the daily sd of the member it replaces (as X5 / v12)')
    print('=' * 120)
    R = {'v9 (cached members)': Pp.v9()}
    raw_v9 = [Pp.raw9, [Pp.tf]]

    def stock_with(new):
        return [Pp.res9, [Pp.rescale_like(new.reindex(Pp.idx), Pp.lgb)]]
    cand = {}
    for kind in ('lin', 'nys'):
        for rule in ('hinge', 'metric'):
            if (kind, 'res', rule) in out:
                cand[f'LightGBM -> {kind} SVM (C by val {rule})'] = ('res', out[(kind, 'res', rule)][0])
            if (kind, 'raw', rule) in out:
                cand[f'market view -> {kind} SVM + Platt (C by val {rule})'] = ('raw', None)
    cand['LightGBM -> logistic regression (v8, reference)'] = ('res', Pp.TB.lr_resid)
    for name, (view, ser) in cand.items():
        if view == 'res':
            R[name] = Pp.decide(Pp.score(raw_v9, stock_with(ser)))
        else:
            kind = name.split('-> ')[1].split(' ')[0]; rule = 'hinge' if 'hinge' in name else 'metric'
            _, cs = out[(kind, 'raw', rule)]
            parts = []
            for a, c in zip(FA, cs):
                r = S[(kind, 'raw', a)]; k = r['key']
                parts.append(pd.Series(r['prob'][c], index=pd.MultiIndex.from_frame(k.assign(date=pd.to_datetime(k.date)))))
            pr = pd.concat(parts).sort_index().reindex(Pp.idx)
            R[name] = Pp.decide(Pp.score([[pr]], [Pp.res9, [Pp.lgb]]))
            out[(kind, 'raw', rule, 'prob')] = pr
    # everything SVM: market view = nys raw (Platt), stock view = nys res only
    if ('nys', 'raw', 'hinge', 'prob') in out and ('nys', 'res', 'hinge') in out:
        pe = out[('nys', 'res', 'hinge')][0].reindex(Pp.idx)
        pe = Pp.rescale_like(pe, pd.concat(Pp.res9 + [Pp.lgb], axis=1).mean(axis=1))
        R['ALL SVM: market nys + Platt, stock view nys only'] = Pp.decide(Pp.score([[out[('nys', 'raw', 'hinge', 'prob')]]], [[pe]]))
    if ('lin', 'raw', 'hinge', 'prob') in out and ('lin', 'res', 'hinge') in out:
        pe = out[('lin', 'res', 'hinge')][0].reindex(Pp.idx)
        pe = Pp.rescale_like(pe, pd.concat(Pp.res9 + [Pp.lgb], axis=1).mean(axis=1))
        R['ALL SVM: market lin + Platt, stock view lin only'] = Pp.decide(Pp.score([[out[('lin', 'raw', 'hinge', 'prob')]]], [[pe]]))
    base = R['v9 (cached members)']
    T = pd.DataFrame([{'config': k, **L.ar_row(D, base)} for k, D in R.items()])
    print(T[['config', 'val', 'hold', 'post', 'all', 'sd', 'timing', 'xsec']].round(2).to_string(index=False)); print()
    print(T[['config', 'd_val', 't_val', 'd_hold', 't_hold', 'd_post', 't_post', 'd_all', 't_all']].round(2).to_string(index=False))
    T.to_csv(os.path.join(OUTCSV, 'rs_svm_pipeline.csv'), index=False, float_format='%.4f')
    pd.to_pickle(R, os.path.join(L.CACHE, 'rs_svm_pipeline_daily.pkl'))
    print()


def part_facts(Pp):
    print('=' * 120)
    print('0. How much data: rows, days, and how much of it is signal')
    print('=' * 120)
    P = L.panel()
    rows = []
    for a, b in L.FOLDS:
        te = pd.Timestamp(a) - pd.Timedelta(days=1)
        Q = P[(P.date >= L.TRAIN_START) & (P.date <= te) & P.resid_r.notna()]
        rows.append({'fold': a[:7], 'train rows': len(Q), 'train days': Q.date.nunique()})
    print(pd.DataFrame(rows).to_string(index=False))
    q = P[P.target_r.notna()]
    # intraclass correlation of the same-day returns (one-way ANOVA estimator) and the design effect
    # intraclass correlation (one-way ANOVA estimator) and the design effect n / (1 + (k - 1) icc)
    g = q.groupby('date')['target_r']; k = g.size().mean()
    msb = (g.size() * (g.mean() - q.target_r.mean()) ** 2).sum() / (g.ngroups - 1)
    msw = ((q.target_r - g.transform('mean')) ** 2).sum() / (len(q) - g.ngroups)
    icc = (msb - msw) / (msb + (k - 1) * msw)
    print(f'  raw intraday return (market view label): same-day intraclass correlation {icc:.3f} -> one day of {k:.0f} stocks is worth '
          f'{k / (1 + (k - 1) * icc):.1f} independent rows; the market component itself has one value per day')
    # excess return: the day mean is removed by construction (icc = -1/(k-1)), dependence is inside industry groups
    if 'group' in q:
        X = q[['date', 'group', 'resid_r']].dropna()
        g = X.groupby(['date', 'group']).resid_r
        kg = g.size()
        msb = (kg * (g.mean() - X.resid_r.mean()) ** 2).sum() / (g.ngroups - 1)
        msw = ((X.resid_r - g.transform('mean')) ** 2).sum() / (len(X) - g.ngroups)
        k0 = kg.mean(); icc = (msb - msw) / (msb + (k0 - 1) * msw)
        print(f'  excess return (stock view label), same day and same industry group: intraclass correlation {icc:.3f} '
              f'(mean group size {k0:.1f}) -> one day of 100 stocks is worth about {100 / (1 + (k0 - 1) * icc):.0f} independent rows')
    print()


def main():
    Pp = L.Pipe()
    part_facts(Pp)
    U = load_train()
    print(f'rs_train units loaded: {sum(len(s) for d in U.values() for s in d.values())}')
    Tm = part_members(U, Pp)
    part_pipeline(U, Pp)
    part_curves(U)
    part_size(U, Pp)
    part_seeds(U, Pp)
    S = load_svm()
    print(f'rs_svm units loaded: {len(S)}')
    out = part_svm(S, Pp, Tm)
    part_svm_pipeline(S, out, Pp)


if __name__ == '__main__':
    main()

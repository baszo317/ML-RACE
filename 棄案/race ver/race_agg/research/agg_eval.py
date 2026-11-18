r"""
agg_eval.py -- race_agg (research\PREREG_race_agg.md section 4): every candidate on D, the rule, and the report-only diagnostics.

  python agg_eval.py          -> results\agg_D.txt, agg_D.json, agg_D_daily.csv, agg_blend.json
Nothing here reads a V prediction: only the D-fold unit files exist in this experiment, and every series is cut to 2022-2024.
"""
import os, sys, json, time, math
import numpy as np, pandas as pd
import agglib as A
L = A.L


def blend_frame(P, members_val, fa):
    """the last-120-day rows of fold fa with the within-day z of the 5 g^- predictions, label, weight and context"""
    Z = pd.concat([A.zday(members_val[m]).rename(m) for m in A.MEMBERS5], axis=1)
    Q = P.set_index(['date', 'stock_id'])[['resid_r', 'absres_60', 'C__vix']]
    Z = Z.join(Q, how='left')
    Z['y'] = (Z.resid_r > 0).astype(float)
    Z['w'] = np.minimum(Z.resid_r.abs(), 0.05)
    Z['vol_rank'] = Z.absres_60.groupby(level='date').rank(pct=True)
    return Z


def test_frame(P, members_test):
    Z = pd.concat([A.zday(members_test[m]).rename(m) for m in A.MEMBERS5], axis=1)
    Q = P.set_index(['date', 'stock_id'])[['absres_60', 'C__vix']]
    Z = Z.join(Q, how='left')
    Z['vol_rank'] = Z.absres_60.groupby(level='date').rank(pct=True)
    return Z


def fit_linear_blend(Z):
    """weighted logistic regression of y on the 5 z with an intercept, alpha >= 0, no penalty (L-BFGS-B)"""
    from scipy.optimize import minimize
    X = Z[A.MEMBERS5].to_numpy(float); yt = 2 * Z.y.to_numpy() - 1; w = Z.w.to_numpy(float); w = w / w.sum()

    def f(th):
        m = yt * (th[0] + X @ th[1:])
        l = np.logaddexp(0, -m)
        g = -yt * (1 - 1 / (1 + np.exp(-m)))                   # d l / d (b + Xa) = -y~ * sigmoid(-m)
        return float(w @ l), np.concatenate([[w @ g], X.T @ (w * g)])
    r = minimize(f, np.zeros(1 + len(A.MEMBERS5)), jac=True, method='L-BFGS-B',
                 bounds=[(None, None)] + [(0, None)] * len(A.MEMBERS5), options=dict(maxiter=2000, gtol=1e-10))
    return r.x, bool(r.success)


STACK_IN = A.MEMBERS5 + ['vol_rank', 'C__vix']


def fit_stack(Z):
    import lightgbm as lgb
    prm = dict(objective='binary', num_leaves=4, max_depth=2, min_data_in_leaf=1000, learning_rate=0.05, lambda_l2=10.0,
               feature_fraction=1.0, bagging_fraction=1.0, bagging_freq=0, seed=1, num_threads=1, verbose=-1, deterministic=True,
               force_col_wise=True)
    ds = lgb.Dataset(Z[STACK_IN].to_numpy(np.float32), Z.y.to_numpy(), weight=Z.w.to_numpy() / Z.w.mean())
    return lgb.train(prm, ds, num_boost_round=150)


def blends(P):
    """A3, A4 scores (test years) and their per-fold parameters"""
    mem_test = {m: A.load_std_unit(m) for m in A.MEMBERS5}
    out3, out4, info = [], [], {}
    for fa, fb in A.D_FOLDS:
        mv = {}
        for m, g in zip(A.MEMBERS5, A.GM_UNITS):
            s = A.load_agg_unit(g, seeds=L.SEEDS[m.split('.')[1]], folds=[(fa, fb)])
            if s is None:
                raise FileNotFoundError(f'{g} {fa}')
            mv[m] = s
        Zv = blend_frame(P, mv, fa)
        dts = Zv.index.get_level_values('date')
        assert dts.max() < pd.Timestamp(fa), 'blending rows must be before the fold'
        th, ok = fit_linear_blend(Zv)
        st = fit_stack(Zv)
        mt = {m: s[(s.index.get_level_values('date') >= pd.Timestamp(fa)) & (s.index.get_level_values('date') < pd.Timestamp(fb))]
              for m, s in mem_test.items()}
        Zt = test_frame(P, mt)
        a = th[1:]
        if np.all(a <= 0):
            a = np.ones_like(a)
        out3.append(pd.Series(Zt[A.MEMBERS5].to_numpy(float) @ a, index=Zt.index))
        out4.append(pd.Series(st.predict(Zt[STACK_IN].to_numpy(np.float32)), index=Zt.index))
        imp = st.feature_importance('gain'); imp = imp / imp.sum()
        # report-only: the g^- members' own IC on the blending days and their correlations
        ic_val = {m: float(Zv.groupby(level='date').apply(lambda g: g[m].rank().corr(g.resid_r.rank())).mean()) for m in A.MEMBERS5}
        info[fa] = dict(alpha=dict(zip(['intercept'] + A.MEMBERS5, [float(x) for x in th])), success=ok,
                        alpha_share={m: float(x / max(th[1:].sum(), 1e-12)) for m, x in zip(A.MEMBERS5, th[1:])},
                        stack_gain=dict(zip(STACK_IN, [float(x) for x in imp])), n_rows=int(len(Zv)),
                        days=[str(dts.min().date()), str(dts.max().date())], ic_val=ic_val)
    return A.cut_D(pd.concat(out3).sort_index()), A.cut_D(pd.concat(out4).sort_index()), info


def scores(P):
    S = {}
    mem = {m: A.load_std_unit(m) for m in A.MEMBERS5}
    S['SD'] = mem['C.LU']
    S['A1'] = A.zavg([mem['C.LU'], mem['C.LN']])
    S['A2'] = A.zavg([mem[m] for m in A.MEMBERS5])
    missing = []
    try:
        S['A3'], S['A4'], binfo = blends(P)
    except FileNotFoundError as e:
        missing += ['A3', 'A4']; binfo = {'missing': str(e)}
    b1 = A.load_agg_unit('B1')
    if b1 is None:
        missing.append('B1')
    else:
        S['B1'] = (mem['C.LU'] * 3 + b1.reindex(mem['C.LU'].index) * 7) / 10      # mean of seeds 1-10
    for c in ('B2', 'C1', 'C2', 'C3', 'D1', 'D2', 'E1', 'E4', 'F1'):
        s = A.load_agg_unit(c)
        if s is None:
            missing.append(c)
        else:
            S[c] = s
    if 'E1' in S:
        S['E2'] = A.zavg([mem['C.LU'], S['E1']])
    else:
        missing.append('E2')
    sig = P.set_index(['date', 'stock_id'])['absres_60'].reindex(mem['C.LU'].index)
    sig = sig.fillna(sig.groupby(level='date').transform('median'))
    S['F2'] = (2 * mem['C.LU'] - 1) * sig
    return S, mem, binfo, missing


def prop_B(mem_scores, blend_score, INFO):
    """report-only: uniform-blend IC predicted from the members' daily ICs and rank correlations vs the realized IC"""
    a = []
    for s in mem_scores:
        a.append(A.decide(s, INFO).ic.mean())
    K = len(mem_scores); R = np.eye(K)
    for i in range(K):
        for j in range(i + 1, K):
            R[i, j] = R[j, i] = A.daily_rank_corr(mem_scores[i], mem_scores[j])
    pred = float(np.sum(a) / np.sqrt(R.sum()))
    real = float(A.decide(blend_score, INFO).ic.mean())
    return dict(member_ic=[float(x) for x in a], corr=R.round(4).tolist(), predicted_ic=pred, realized_ic=real)


def main():
    t0 = time.time()
    P = A.load_panel(); INFO = A.load_INFO()
    S, mem, binfo, missing = scores(P)
    res = {k: A.decide(v, INFO) for k, v in S.items()}
    base = res['SD']
    assert abs(base.AR.mean() - 15.099561468823161) < 1e-9, base.AR.mean()
    st = {k: A.stats(v, base) for k, v in res.items()}
    st['SD'].update(diff=0.0, t=float('nan'), passes=False)
    cands = [c for c in A.CANDS if c in st]
    hp = A.holm({c: A.p_one_sided(st[c]['t']) for c in cands})
    for c in cands:
        st[c]['corr_CLU'] = A.daily_rank_corr(S[c], S['SD'])
        st[c]['top25_changes'] = A.top25_changes(S[c], S['SD'], INFO)
    # ---------------- rule (PREREG section 4)
    fam_win = {}
    for f in sorted(set(A.FAMILY.values())):
        ok = [c for c in cands if A.FAMILY[c] == f and st[c]['passes']]
        if ok:
            fam_win[f] = max(ok, key=lambda c: st[c]['diff'])
    combo = None
    if len(fam_win) >= 2:
        S['COMBO'] = A.zavg([S[c] for c in fam_win.values()])
        res['COMBO'] = A.decide(S['COMBO'], INFO); st['COMBO'] = A.stats(res['COMBO'], base)
        combo = dict(parts=list(fam_win.values()), passes=st['COMBO']['passes'])
    if not fam_win:
        adopted = []
    elif combo is not None and combo['passes']:
        adopted = ['COMBO'] + combo['parts']
    else:
        adopted = [max(fam_win.values(), key=lambda c: st[c]['diff'])]
    # ---------------- report-only diagnostics
    diag = {}
    try:
        diag['propB_A1'] = prop_B([mem['C.LU'], mem['C.LN']], S['A1'], INFO)
        diag['propB_A2'] = prop_B([mem[m] for m in A.MEMBERS5], S['A2'], INFO)
        if 'E1' in S:
            diag['propB_E2'] = prop_B([mem['C.LU'], S['E1']], S['E2'], INFO)
    except Exception as e:
        diag['propB_error'] = repr(e)
    metas = {}
    for c in ('B1', 'B2', 'C1', 'C2', 'C3', 'D1', 'D2', 'E1', 'E4', 'F1'):
        ms = A.load_metas(c)
        keep = ('best_iter', 'T', 'T_ran', 'T_refit', 'eps_first', 'eps_T', 'eps_refit_mean', 'val_loss_1', 'val_loss_T', 'val_loss_last',
                'leaf', 'val', 'leaves', 'depth', 'alpha', 'n_keep', 'unique_days', 'n_days', 'seconds', 'features', 'n_path_alphas')
        metas[c] = [{k: m[k] for k in ('fold', 'seed') + keep if k in m} for m in ms]
    clu_meta = [dict(fold=m['fold'], seed=m['seed'], best_iter=m.get('best_iter')) for m in L.unit_metas('C.LU', A.D_FOLDS)]
    out = dict(written=time.strftime('%Y-%m-%d %H:%M:%S'), period=[A.D_A, A.D_B], days=int(len(base)), stats=st, holm=hp,
               family_winners=fam_win, combo=combo, adopted=adopted, missing=missing, blend=binfo, diag=diag, metas=metas,
               clu_meta=clu_meta, prereg_sha256=A.sha256(A.PREREG), code=A.code_hashes(),
               versions=dict(numpy=np.__version__, pandas=pd.__version__, lightgbm=__import__('lightgbm').__version__,
                             sklearn=__import__('sklearn').__version__, python=sys.version.split()[0]))
    os.makedirs(A.RES, exist_ok=True)
    with open(os.path.join(A.RES, 'agg_D.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=float)
    pd.DataFrame({k: v.AR for k, v in res.items()}).to_csv(os.path.join(A.RES, 'agg_D_daily.csv'))
    lines = [f'race_agg -- aggregation methods of topic 6 / W12 on the race version, D only ({A.D_A} .. 2024-12-31, {len(base)} days)',
             f'written {out["written"]}; SD reproduces part 1 (C.LU on D {base.AR.mean():+.4f}); PREREG sha256 {out["prereg_sha256"][:16]}',
             f'missing: {missing if missing else "none"}', '',
             'cand fam description                              D mean   diff   NW t  2022   2023   2024  pass  Holm p   IC     sd  P19<0 corrCLU chg25']
    for k in ['SD'] + cands + (['COMBO'] if 'COMBO' in st else []):
        v = st[k]; y = v['years']
        lines.append(f'{k:<5}{A.FAMILY.get(k, "-"):<4}{A.LABEL[k]:<40.40} {v["mean"]:+7.2f} {v["diff"]:+6.2f} {v["t"]:6.2f} '
                     f'{y["2022"]:+6.2f} {y["2023"]:+6.2f} {y["2024"]:+6.2f}  {"yes" if v["passes"] else "no ":<4} '
                     f'{hp.get(k, float("nan")):6.3f} {v["ic"]:.4f} {v["sd"]:5.1f} {v["p19_neg"]:5.2f} '
                     f'{v.get("corr_CLU", float("nan")):6.3f} {v.get("top25_changes", float("nan")):5.1f}')
    lines += ['', f'family winners (mean > 0, NW t >= 2, >= 2 of 3 years > 0): {fam_win if fam_win else "none"}',
              f'combination: {combo}',
              f'ADOPTED: {adopted if adopted else "nothing (race version unchanged; V is not opened)"}', '']
    if isinstance(binfo, dict) and 'missing' not in binfo:
        lines.append('A3 linear blending weights (share of sum alpha) / A4 stacker gain share, per fold:')
        for fa, v in binfo.items():
            lines.append(f'  {fa}: alpha ' + ', '.join(f'{m} {v["alpha"][m]:.3f}' for m in A.MEMBERS5) +
                         f'  | stack gain ' + ', '.join(f'{k} {x:.2f}' for k, x in v['stack_gain'].items()))
            lines.append(f'         g- IC on the blending days ' + ', '.join(f'{m} {x:.4f}' for m, x in v['ic_val'].items()) +
                         f'  ({v["days"][0]} .. {v["days"][1]})')
    for k, v in diag.items():
        if k.startswith('propB'):
            lines.append(f'{k}: member IC {[round(x, 4) for x in v["member_ic"]]}, predicted blend IC {v["predicted_ic"]:.4f}, '
                         f'realized {v["realized_ic"]:.4f}')
    txt = '\n'.join(lines)
    with open(os.path.join(A.RES, 'agg_D.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    print(txt)
    print(f'({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()

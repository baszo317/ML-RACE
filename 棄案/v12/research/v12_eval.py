r"""
v12_eval.py -- research\PREREG_v12.md
  part 1 (report only): rank IC of every model (Y1..Y6, lgb_res, X1, X5), the pre-registered paired comparisons (daily IC
         differences, NW t) and the verdicts on C1, C2, C3 and H3; correlations with lgb_res next to the noise ceiling
         (two independent 5-seed blocks of lgb_res), top / bottom 10% returns, IC inside the top and the bottom half,
         early-stopping rounds
  part 2 (the user's rule): each Y added to v11 (LightGBM part (lgb_res + X1 + Y) / 3, Y rescaled as X5), val AR > 0 ->
         selected; all selected together = v12, kept if hold and post AR differences are both > 0
run (in v12\research, after v12_train.py): python v12_eval.py > ..\results\v12_eval.txt
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v12lib as L
import v11b_lib as B                          # v11's data, views and decision (read only)
import v11b_train as TB
V = L.V
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
PER = {'val': ('2022-01-01', '2025-01-01'), 'hold': ('2025-01-01', '2026-06-01'), 'post': ('2026-06-01', '2100-01-01'),
       'all': ('2022-01-01', '2100-01-01'), 'static': ('2026-06-01', '2026-10-01'), 'Oct': ('2026-10-01', '2100-01-01')}
YS = {'Y1': 'lgb_sign', 'Y2': 'lgb_dec_l2', 'Y3': 'lgb_size_l2', 'Y4': 'lgb_dec_pw', 'Y5': 'lgb_size_pw', 'Y6': 'lgb_size_rank'}
DESC = {'lgb_res': 'sign + size weights, binary', 'X1': 'lgb_res on 106 features', 'X5': 'decile, lambdarank',
        'Y1': 'sign, binary', 'Y2': 'decile, L2', 'Y3': 'size, L2', 'Y4': 'decile, pairwise', 'Y5': 'size, pairwise',
        'Y6': 'size, lambdarank'}

def w(ix, k):
    a, b = PER[k]; ix = pd.DatetimeIndex(ix)
    return (ix >= pd.Timestamp(a)) & (ix < pd.Timestamp(b))

c = lambda s: s - s.groupby(level='date').transform('mean')
sdd = lambda s: s.groupby(level='date').transform('std')

def daily_stats(pred, r):
    """per day: rank IC (Spearman), IC inside the top / bottom half by prediction, mean resid of the top / bottom 10% (bp)"""
    X = pd.DataFrame({'p': pred, 'r': r}).dropna()
    out = []
    for d, g in X.groupby(level='date'):
        p, r0 = g.p.values, g.r.values; y = r0 - r0.mean(); n = len(p)        # ranks from the raw return (demeaning can add float ties)
        o = np.argsort(-p, kind='stable'); h = n // 2; k = max(1, int(round(n / 10)))
        sp = lambda a, b: np.corrcoef(pd.Series(a).rank().values, pd.Series(b).rank().values)[0, 1]
        out.append((d, sp(p, r0), sp(p[o[:h]], r0[o[:h]]), sp(p[o[-h:]], r0[o[-h:]]), y[o[:k]].mean() * 1e4, y[o[-k:]].mean() * 1e4))
    return pd.DataFrame(out, columns=['date', 'IC', 'IC_top_half', 'IC_bottom_half', 'top10_bp', 'bottom10_bp']).set_index('date')

def verdict(dd):
    """pre-registered: A > B / A < B / ~ / ? from the daily IC differences A - B"""
    a, t = dd['all']; v, h = dd['val'][0], dd['hold'][0]
    if a > 0 and t >= 2 and v > 0 and h > 0:
        return 'A>B'
    if a < 0 and t <= -2 and v < 0 and h < 0:
        return 'A<B'
    if abs(a) < 0.002 and abs(t) < 2:
        return '~'
    return '?'

def main():
    t0 = time.time()
    d = B.Data()
    M = {'lgb_res': d.LGB, 'X1': d.x1}
    x5 = TB.load_member('lgb_rank')
    assert x5 is not None, 'X5 (v11 cache\\screen\\x\\lgb_rank) missing'
    M['X5'] = x5.reindex(d.idx)
    for k, n in YS.items():
        y = L.load_member(n)
        if y is None:
            sys.exit(f'{k} {n}: predictions missing, run v12_train.py first')
        M[k] = y.reindex(d.idx); assert M[k].notna().all(), (k, int(M[k].isna().sum()))
    # noise ceiling: two independent 5-seed blocks of lgb_res (v11 part A, seeds 6-10 and 11-15)
    root11 = os.path.join(B.V11, 'cache', 'screen')
    b1 = V.load_units('lgb', 'base', ['lgb_res'], range(6, 11), root=root11)['lgb_res'].reindex(d.idx)
    b2 = V.load_units('lgb', 'base', ['lgb_res'], range(11, 16), root=root11)['lgb_res'].reindex(d.idx)

    # ---------------------------------------------------------------- part 1
    S = {k: daily_stats(c(v), d.tr) for k, v in M.items()}
    chk = d.ic(c(d.LGB)); assert np.allclose(chk.values, S['lgb_res'].IC.reindex(chk.index).values, atol=1e-12), 'IC mismatch with v11b_lib.ic'
    print('PART 1 -- the interpretation (report only)\n')
    print('own rank IC on all stocks, 5-seed mean of each model')
    rows = []
    for k, s in S.items():
        rows.append({'model': k, 'label / loss': DESC[k], **{p: s.IC[w(s.index, p)].mean() for p in ('val', 'hold', 'post', 'all')}})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print('\nthe grid (all-period IC): rows = information in the label, columns = loss')
    g = pd.DataFrame({'pointwise': [S['Y1'].IC.mean(), S['lgb_res'].IC.mean(), S['Y2'].IC.mean(), S['Y3'].IC.mean()],
                      'pairwise': [np.nan, np.nan, S['Y4'].IC.mean(), S['Y5'].IC.mean()],
                      'lambdarank': [np.nan, np.nan, S['X5'].IC.mean(), S['Y6'].IC.mean()]},
                     index=['sign', 'sign + size weights', 'decile', 'size'])
    print(g.round(4).to_string())

    def comp(A, Bm):
        dd = (S[A].IC - S[Bm].IC).dropna()
        r = {p: (float(dd[w(dd.index, p)].mean()), float(V.nw_t(dd[w(dd.index, p)].values))) for p in ('val', 'hold', 'post', 'all')}
        return r, verdict(r)
    print('\npaired daily IC differences A - B (NW t) and the pre-registered verdict')
    PAIRS = [('C2/H3 decile row', 'Y2', 'Y4'), ('C2/H3 decile row', 'Y2', 'X5'), ('C2/H3 decile row', 'Y4', 'X5'),
             ('C2/H3 size row', 'Y3', 'Y5'), ('C2/H3 size row', 'Y3', 'Y6'), ('C2/H3 size row', 'Y5', 'Y6'),
             ('C3 size vs decile', 'Y3', 'Y2'), ('C3 size vs decile', 'Y5', 'Y4'), ('C3 size vs decile', 'Y6', 'X5'),
             ('C1', 'Y5', 'lgb_res'), ('report', 'Y3', 'lgb_res'), ('report', 'Y1', 'lgb_res'), ('report', 'X5', 'lgb_res'),
             ('report', 'Y4', 'lgb_res'), ('report', 'Y2', 'lgb_res')]
    R = {}
    for grp, A, Bm in PAIRS:
        r, v = comp(A, Bm); R[(A, Bm)] = (r, v)
        print(f'  {grp:18s} {A:7s} - {Bm:7s}: ' + '  '.join(f'{p} {r[p][0]:+.4f} ({r[p][1]:5.2f})' for p in ('val', 'hold', 'post', 'all'))
              + f'   -> {v}')
    V_ = {k: v for k, (r, v) in R.items()}
    print('\nverdicts (PREREG_v12 section 3)')
    rowQ = [V_[('Y2', 'Y4')], V_[('Y2', 'X5')], V_[('Y4', 'X5')]]; rowM = [V_[('Y3', 'Y5')], V_[('Y3', 'Y6')], V_[('Y5', 'Y6')]]
    allv = rowQ + rowM
    c2 = 'holds' if all(v == '~' for v in allv) else ('does not hold' if any(v in ('A>B', 'A<B') for v in allv) else 'uncertain')
    h3n = (V_[('Y4', 'X5')] == 'A>B') + (V_[('Y5', 'Y6')] == 'A>B')
    h3 = {2: 'holds', 1: 'partly holds', 0: 'does not hold'}[h3n]
    c3v = [V_[('Y3', 'Y2')], V_[('Y5', 'Y4')], V_[('Y6', 'X5')]]
    c3 = ('holds' if sum(v == 'A>B' for v in c3v) >= 2 and not any(v == 'A<B' for v in c3v)
          else 'does not hold' if all(v in ('~', 'A<B') for v in c3v) else 'uncertain')
    v1 = V_[('Y5', 'lgb_res')]; a1 = R[('Y5', 'lgb_res')][0]['all'][0]
    c1 = 'does not hold' if v1 == 'A>B' else ('holds' if v1 in ('~', 'A<B') or a1 <= 0 else 'uncertain')
    print(f'  C2 (changing the loss alone makes no difference): {c2}   [decile row {rowQ}, size row {rowM}]')
    print(f'  H3 (NDCG position discount hurts; Y4 > X5 and Y5 > Y6): {h3}   [{V_[("Y4", "X5")]}, {V_[("Y5", "Y6")]}]')
    print(f'  C3 (size label beats decile label under the same loss): {c3}   [{c3v}]')
    print(f'  C1 (a within-day pairwise loss with lgb_res\'s information does not beat lgb_res): {c1}   [Y5 - lgb_res {v1}, all {a1:+.4f}]')

    print('\ncorrelation of the day-centred predictions with lgb_res (all days 2022-01..2026-10)')
    cc = lambda a, b: float(np.corrcoef(c(a).values, c(b).values)[0, 1])
    print(f'  noise ceiling: lgb_res seeds 6-10 vs 11-15 {cc(b1, b2):.3f}; seeds 1-5 (v9) vs 6-10 {cc(d.LGB, b1):.3f}')
    print('  ' + ', '.join(f'{k} {cc(v, d.LGB):.3f}' for k, v in M.items() if k != 'lgb_res'))
    print('  pairs inside the grid: ' + ', '.join(f'{a}-{b} {cc(M[a], M[b]):.3f}' for a, b in
                                                 (('Y2', 'Y4'), ('Y4', 'X5'), ('Y2', 'X5'), ('Y3', 'Y5'), ('Y5', 'Y6'), ('Y3', 'Y6'),
                                                  ('Y3', 'Y2'), ('Y5', 'Y4'), ('Y6', 'X5'), ('Y1', 'Y3'))))

    print('\nwhere in the ranking (all days): IC inside the top / bottom half by prediction, mean excess return of the predicted '
          'top / bottom 10% (bp), spread')
    rows = []
    for k, s in S.items():
        rows.append({'model': k, 'IC': s.IC.mean(), 'IC_top_half': s.IC_top_half.mean(), 'IC_bottom_half': s.IC_bottom_half.mean(),
                     'top10_bp': s.top10_bp.mean(), 'bottom10_bp': s.bottom10_bp.mean(), 'spread_bp': (s.top10_bp - s.bottom10_bp).mean()})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    bf = os.path.join(L.V12, 'results', 'v12_train_units.csv')
    if os.path.exists(bf):
        Bt = pd.read_csv(bf)
        print('\nearly-stopping rounds (min / median / max over the 5 seeds) by fold')
        T = Bt.groupby(['member', 'fold']).best.agg(lambda x: f'{x.min()}/{int(x.median())}/{x.max()}').unstack('fold')
        print(T.reindex(list(YS.values())).to_string())

    # ---------------------------------------------------------------- part 2
    print('\n\nPART 2 -- v12 members (the user\'s rule)\n')
    scale = lambda y: d.LGB.groupby(level='date').transform('mean') + c(y) * sdd(c(d.LGB)) / sdd(c(y))
    YSC = {k: scale(M[k]) for k in YS}
    dt = lambda s: s.index.get_level_values('date')
    def view(lgbs):
        return c((pd.concat(d.mlp, axis=1).mean(axis=1) + pd.concat(lgbs, axis=1).mean(axis=1)) / 2)
    def run(pe, v11):
        pb = d.pbar(night=v11)
        s = B.W_T * (pb.reindex(dt(pe)).values - 0.5) + pe
        return d.decide(s, month_start=v11)
    D = {'v9': run(view([d.LGB]), False), 'v11': run(view([d.LGB, d.x1]), True)}
    for k in YS:
        D[f'v11 + {k}'] = run(view([d.LGB, d.x1, YSC[k]]), True)
        D[f'v9 + {k}'] = run(view([d.LGB, YSC[k]]), False)
    m9 = {p: D['v9'].AR[w(D['v9'].index, p)].mean() for p in ('val', 'hold', 'post', 'all')}
    m11 = {p: D['v11'].AR[w(D['v11'].index, p)].mean() for p in ('val', 'hold', 'post', 'all')}
    print(f'rebuilt: v9 {m9["val"]:.2f} / {m9["hold"]:.2f} / {m9["post"]:.2f} / {m9["all"]:.2f} (13.99 / 14.33 / 16.59 / 14.31); '
          f'v11 {m11["val"]:.2f} / {m11["hold"]:.2f} / {m11["post"]:.2f} / {m11["all"]:.2f} (14.60 / 14.34 / 20.35 / 15.02)')
    def diff(a, b):
        dd = (D[a].AR - D[b].AR).dropna()
        return {p: (float(dd[w(dd.index, p)].mean()), float(V.nw_t(dd[w(dd.index, p)].values))) for p in PER}
    out = {}
    print('\neach candidate added to v11, minus v11 (bp/day, NW t)')
    for k in YS:
        r = diff(f'v11 + {k}', 'v11'); out[k] = r
        print(f'  {k} {YS[k]:14s} ' + '  '.join(f'{p} {r[p][0]:+.2f} ({r[p][1]:.2f})' for p in PER)
              + ('   -> selected (val > 0)' if r['val'][0] > 0 else '   -> not selected'))
    print('\n(report) each candidate added to v9, minus v9')
    for k in YS:
        r = diff(f'v9 + {k}', 'v9')
        print(f'  {k} {YS[k]:14s} ' + '  '.join(f'{p} {r[p][0]:+.2f} ({r[p][1]:.2f})' for p in PER))
    sel = [k for k in YS if out[k]['val'][0] > 0]
    res = {'selected': sel, 'passed': False, 'single': out}
    if sel:
        D['v12'] = run(view([d.LGB, d.x1] + [YSC[k] for k in sel]), True)
        r = diff('v12', 'v11'); res['v12 vs v11'] = r
        ok = r['hold'][0] > 0 and r['post'][0] > 0; res['passed'] = bool(ok)
        print(f'\nselected: {sel} -> v12 = v11 with LightGBM part = mean of lgb_res, X1, ' + ', '.join(sel))
        print('v12 minus v11: ' + '  '.join(f'{p} {r[p][0]:+.2f} ({r[p][1]:.2f})' for p in PER))
        print('pre-registered check: hold and post both > 0 -> ' + ('PASSED: v12 becomes a daily program' if ok
                                                                   else 'not passed: no v12 daily program, v11 stays as it is'))
        r9 = diff('v12', 'v9'); res['v12 vs v9'] = r9
        print('(report) v12 minus v9: ' + '  '.join(f'{p} {r9[p][0]:+.2f} ({r9[p][1]:.2f})' for p in PER))
    else:
        print('\nno candidate has val > 0: no v12 daily program, v11 stays as it is')
    rows = []
    for k in ['v9', 'v11'] + [f'v11 + {k}' for k in YS] + (['v12'] if 'v12' in D else []):
        Dk = D[k]; r = {'config': k}
        for p in PER:
            r[p] = Dk.AR[w(Dk.index, p)].mean()
        r19 = Dk.AR.rolling(19).mean().dropna(); r['sd'] = Dk.AR.std(); r['P19>0'] = (r19 > 0).mean()
        rows.append(r)
    print('\nTop50 bp/day (walk-forward)')
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    res['verdicts'] = {'C1': c1, 'C2': c2, 'C3': c3, 'H3': h3}
    res['pairs'] = {f'{a} - {b}': {'diff': r, 'verdict': v} for (a, b), (r, v) in R.items()}
    json.dump(res, open(os.path.join(L.V12, 'results', 'v12_selection.json'), 'w'), indent=1)
    pd.to_pickle({'daily': D, 'stats': S}, os.path.join(L.V12, 'cache', 'v12_daily.pkl'))
    print(f'\ndone in {time.time() - t0:.0f}s')

if __name__ == '__main__':
    main()

r"""
agg_v.py -- race_agg V stage (research\PREREG_race_agg_V.md, user request 11/18 18:12): the 16 candidates on V.
  S (primary): against the 10-seed LU (C.LU seeds 1-3 + B1 seeds 4-10); candidates that contain C.LU are rebuilt on it.
  P (report only): against SD (C.LU seeds 1-3), the definitions of PREREG_race_agg.md.
Rule (S): tier 1 = V diff > 0, V NW t >= 2, >= 2 of 3 V segments > 0, D diff > 0; tier 2 = tier 1 and Holm(15) of the D+V p < 0.05;
recommend for live only tier 2. V is computed once: results\SEALED_V_agg.json; afterwards the script refuses (--audit only compares).

  python agg_v.py            -> results\agg_V.txt, agg_V.json, agg_V_daily.csv, SEALED_V_agg.json
  python agg_v.py --audit
"""
import os, sys, json, time
import numpy as np, pandas as pd
import agglib as A
import agg_eval as E
L = A.L

SEAL = os.path.join(A.RES, 'SEALED_V_agg.json')
SD_V_PART1 = 16.163461557302234                 # race_std SEALED_V.json: C.LU on V
SD_D_PART1 = 15.099561468823161
TRAINED = ('B2', 'C1', 'C2', 'C3', 'D1', 'D2', 'E1', 'E4', 'F1')
S_CANDS = [c for c in A.CANDS if c != 'B1']     # B1 is the S baseline


def lu10(period):
    per = [A.cut(pd.concat([L.load_npz(L.unit_file('C.LU', fa, s))[0] for fa, _ in A.FOLDS_OF[period]]).sort_index(), period)
           for s in (1, 2, 3)]
    for s in range(4, 11):
        x = A.load_period_unit('B1', period, seeds=(s,))
        if x is None:
            raise FileNotFoundError(f'B1 seed {s} {period}')
        per.append(x)
    idx = per[0].index
    return pd.concat([p.reindex(idx) for p in per], axis=1).mean(axis=1)


def blend_models(P, period):
    out = {}
    for fa, fb in A.FOLDS_OF[period]:
        mv = {}
        for m, g in zip(A.MEMBERS5, A.GM_UNITS):
            s = A.load_agg_unit(g, seeds=L.SEEDS[m.split('.')[1]], folds=[(fa, fb)])
            if s is None:
                raise FileNotFoundError(f'{g} {fa}')
            mv[m] = s
        Zv = E.blend_frame(P, mv, fa)
        dts = Zv.index.get_level_values('date')
        assert dts.max() < pd.Timestamp(fa), 'blending rows must be before the fold'
        th, ok = E.fit_linear_blend(Zv)
        ic_val = {m: float(Zv.groupby(level='date').apply(lambda g: g[m].rank().corr(g.resid_r.rank())).mean()) for m in A.MEMBERS5}
        out[fa] = dict(th=th, ok=ok, stack=E.fit_stack(Zv), days=[str(dts.min().date()), str(dts.max().date())], n=int(len(Zv)),
                       ic_val=ic_val)
    return out


def apply_blends(P, period, mem, models):
    o3, o4 = [], []
    for fa, fb in A.FOLDS_OF[period]:
        mt = {m: s[(s.index.get_level_values('date') >= pd.Timestamp(fa)) & (s.index.get_level_values('date') < pd.Timestamp(fb))]
              for m, s in mem.items()}
        Zt = E.test_frame(P, mt)
        a = models[fa]['th'][1:]
        if np.all(a <= 0):
            a = np.ones_like(a)
        o3.append(pd.Series(Zt[A.MEMBERS5].to_numpy(float) @ a, index=Zt.index))
        o4.append(pd.Series(models[fa]['stack'].predict(Zt[E.STACK_IN].to_numpy(np.float32)), index=Zt.index))
    return A.cut(pd.concat(o3).sort_index(), period), A.cut(pd.concat(o4).sort_index(), period)


def scores(P, period):
    mem = {m: A.load_member(m, period) for m in A.MEMBERS5}
    L10 = lu10(period)
    models = blend_models(P, period)
    trained = {}
    for c in TRAINED:
        x = A.load_period_unit(c, period)
        if x is None:
            raise FileNotFoundError(f'{c} {period}')
        trained[c] = x
    sig = P.set_index(['date', 'stock_id'])['absres_60']

    def f2(p):
        s = sig.reindex(p.index)
        return (2 * p - 1) * s.fillna(s.groupby(level='date').transform('median'))

    def build(lu):
        mm = dict(mem); mm['C.LU'] = lu
        S = {'A1': A.zavg([lu, mem['C.LN']]), 'A2': A.zavg([mm[m] for m in A.MEMBERS5])}
        S['A3'], S['A4'] = apply_blends(P, period, mm, models)
        S.update(trained)
        S['E2'] = A.zavg([lu, trained['E1']])
        S['F2'] = f2(lu)
        return S
    SP = build(mem['C.LU']); SP['SD'] = mem['C.LU']; SP['B1'] = L10
    SS = build(L10); SS['LU10'] = L10
    return SP, SS, models


def metas_v(c):
    out = []
    seeds = L.SEEDS[c.split('.')[1]] if c in A.MEMBER_V_TRAINED or c.startswith('GM.') else A.SEEDS[c]
    for fa, _ in A.FOLDS_OF['V']:
        for sd in seeds:
            f = A.unit_file(c, fa, sd)
            if os.path.exists(f):
                m = L.load_npz(f)[1]
                out.append({k: m[k] for k in ('fold', 'seed', 'best_iter', 'T', 'leaf', 'leaves', 'n_keep', 'best_epoch', 'eps_first')
                            if k in m})
    return out


def compute():
    P = A.load_panel(); INFO = A.load_INFO()
    res, models = {}, {}
    for per in ('D', 'V'):
        SP, SS, models[per] = scores(P, per)
        res[per] = ({k: A.decide(v, INFO, per) for k, v in SP.items()}, {k: A.decide(v, INFO, per) for k, v in SS.items()})
    assert abs(res['V'][0]['SD'].AR.mean() - SD_V_PART1) < 1e-9, res['V'][0]['SD'].AR.mean()
    assert abs(res['D'][0]['SD'].AR.mean() - SD_D_PART1) < 1e-9
    J = json.load(open(os.path.join(A.RES, 'agg_D.json'), encoding='utf-8'))
    for c in A.CANDS:                                        # P on D reproduces the D stage
        assert abs(res['D'][0][c].AR.mean() - J['stats'][c]['mean']) < 1e-6, (c, res['D'][0][c].AR.mean(), J['stats'][c]['mean'])
    comps = {}
    for mode, k, base, cands in (('S', 1, 'LU10', S_CANDS), ('P', 0, 'SD', A.CANDS)):
        for c in cands:
            rD, rV, bD, bV = res['D'][k][c], res['V'][k][c], res['D'][k][base], res['V'][k][base]
            sD, sV = A.stats_p(rD, bD, 'D'), A.stats_p(rV, bV, 'V')
            dd = pd.concat([(rD.AR - bD.AR).dropna(), (rV.AR - bV.AR).dropna()])
            t = float(L.nw_t(dd.values))
            comps[f'{mode}:{c}'] = dict(D=sD, V=sV, DV=dict(diff=float(dd.mean()), t=t, days=int(len(dd)), p=A.p_one_sided(t)))
    holm_S = A.holm({c: comps[f'S:{c}']['DV']['p'] for c in S_CANDS})
    holm_P = A.holm({c: comps[f'P:{c}']['DV']['p'] for c in A.CANDS})
    tier1 = [c for c in S_CANDS if comps[f'S:{c}']['V']['diff'] > 0 and comps[f'S:{c}']['V']['t'] >= 2
             and comps[f'S:{c}']['V']['pos_segments'] >= 2 and comps[f'S:{c}']['D']['diff'] > 0]
    tier2 = [c for c in tier1 if holm_S[c] < 0.05]
    rec = max(tier2, key=lambda c: comps[f'S:{c}']['DV']['diff']) if tier2 else None
    p_tier1 = [c for c in A.CANDS if comps[f'P:{c}']['V']['diff'] > 0 and comps[f'P:{c}']['V']['t'] >= 2
               and comps[f'P:{c}']['V']['pos_segments'] >= 2 and comps[f'P:{c}']['D']['diff'] > 0]
    base = {'SD_V': A.stats_p(res['V'][0]['SD'], res['V'][0]['SD'], 'V'), 'LU10_V': A.stats_p(res['V'][1]['LU10'], res['V'][1]['LU10'], 'V'),
            'SD_D': float(res['D'][0]['SD'].AR.mean()), 'LU10_D': float(res['D'][1]['LU10'].AR.mean())}
    blend = {per: {fa: dict(alpha=dict(zip(['intercept'] + A.MEMBERS5, [float(x) for x in m['th']])), days=m['days'], n=m['n'],
                            ic_val=m['ic_val']) for fa, m in models[per].items()} for per in ('V',)}
    return res, comps, holm_S, holm_P, tier1, tier2, rec, p_tier1, base, blend


def main(audit=False):
    t0 = time.time()
    assert os.path.exists(A.PREREG_V), 'no PREREG_race_agg_V.md'
    if os.path.exists(SEAL) and not audit:
        sys.exit('SEALED_V_agg.json exists: V was computed once; use --audit to compare')
    res, comps, holm_S, holm_P, tier1, tier2, rec, p_tier1, base, blend = compute()
    if audit:
        sealed = json.load(open(SEAL, encoding='utf-8'))
        bad = [k for k, v in comps.items() if abs(v['V']['diff'] - sealed['comps'][k]['V']['diff']) > 1e-9]
        print('audit:', 'identical' if not bad else f'DIFFERENT {bad}')
        return
    metas = {c: metas_v(c) for c in TRAINED + ('B1',) + A.MEMBER_V_TRAINED}
    out = dict(written=time.strftime('%Y-%m-%d %H:%M:%S'), prereg_sha256=A.sha256(A.PREREG), prereg_v_sha256=A.sha256(A.PREREG_V),
               code=A.code_hashes(), base=base, comps=comps, holm_S=holm_S, holm_P=holm_P, tier1=tier1, tier2=tier2,
               recommend=rec, p_tier1_report=p_tier1, blend=blend, metas=metas,
               versions=dict(numpy=np.__version__, pandas=pd.__version__, lightgbm=__import__('lightgbm').__version__,
                             sklearn=__import__('sklearn').__version__, python=sys.version.split()[0]))
    with open(os.path.join(A.RES, 'agg_V.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=float)
    daily = {}
    for mode, k in (('P', 0), ('S', 1)):
        for c, D in res['V'][k].items():
            daily[f'{mode}:{c}'] = D.AR
    pd.DataFrame(daily).to_csv(os.path.join(A.RES, 'agg_V_daily.csv'))
    with open(SEAL, 'w', encoding='utf-8') as fh:
        json.dump(dict(written=out['written'], prereg_v_sha256=out['prereg_v_sha256'], code=out['code'], tier1=tier1, tier2=tier2,
                       recommend=rec, comps={k: dict(V=dict(diff=v['V']['diff'], t=v['V']['t'])) for k, v in comps.items()}),
                  fh, indent=1, default=float)
    seg = [s for s, _, _ in A.SEGMENTS['V']]
    hdr = ('cand  description                              V mean  V diff  V t   ' + '  '.join(f'{s:>10}' for s in seg) +
           '  IC diff (t)       D diff  D+V diff  D+V t  Holm p')
    lines = [f'race_agg V stage (PREREG_race_agg_V.md, sha256 {out["prereg_v_sha256"][:16]}), V = 2025-01-02 .. 2026-11-09, '
             f'{base["SD_V"]["days"]} days; written {out["written"]}',
             f'SD on V {base["SD_V"]["mean"]:+.4f} (part 1: {SD_V_PART1:+.4f}); 10-seed LU: V {base["LU10_V"]["mean"]:+.2f}, '
             f'D {base["LU10_D"]:+.2f} (SD on D {base["SD_D"]:+.2f})', '',
             'S (primary): against the 10-seed LU; A1 A2 A3 A4 E2 F2 rebuilt on it', hdr]
    for mode, cands, hp in (('S', S_CANDS, holm_S), ('P', A.CANDS, holm_P)):
        if mode == 'P':
            lines += ['', 'P (report only): against SD = C.LU seeds 1-3, definitions of PREREG_race_agg.md', hdr]
        for c in cands:
            v = comps[f'{mode}:{c}']; V = v['V']
            lines.append(f'{c:<5} {A.LABEL[c]:<40.40} {V["mean"]:+7.2f} {V["diff"]:+7.2f} {V["t"]:5.2f} ' +
                         '  '.join(f'{V["segments"][s]:+10.2f}' for s in seg) +
                         f'  {V["ic_diff"]:+.4f} ({V["ic_t"]:5.2f})  {v["D"]["diff"]:+6.2f}  {v["DV"]["diff"]:+7.2f}  {v["DV"]["t"]:5.2f}  '
                         f'{hp.get(c, float("nan")):6.3f}')
    lines += ['', f'S tier 1 (V diff > 0, V t >= 2, >= 2 of 3 segments > 0, D diff > 0): {tier1 if tier1 else "none"}',
              f'S tier 2 (tier 1 and Holm(15) of the D+V p < 0.05): {tier2 if tier2 else "none"}',
              f'RECOMMEND for live: {rec if rec else "nothing (race version unchanged)"}',
              f'P tier-1 conditions (report only): {p_tier1 if p_tier1 else "none"}', '',
              'A3 weights on the V folds (z coefficients) and the g- IC on the blending rows:']
    for fa, m in blend['V'].items():
        lines.append(f'  {fa}: ' + ', '.join(f'{k} {x:.3f}' for k, x in m['alpha'].items() if k != 'intercept') +
                     f'  | IC ' + ', '.join(f'{k} {x:.4f}' for k, x in m['ic_val'].items()) + f'  ({m["days"][0]} .. {m["days"][1]})')
    txt = '\n'.join(lines)
    with open(os.path.join(A.RES, 'agg_V.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    print(txt)
    print(f'({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main(audit='--audit' in sys.argv)

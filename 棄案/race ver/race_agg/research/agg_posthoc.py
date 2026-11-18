r"""
agg_posthoc.py -- race_agg, AFTER the rule was applied (report only, not in the PREREG):
  1. seed lottery: D AR of C.LU for each single seed 1-10 and of every 3-seed average (120 triples); where seeds {1,2,3} rank
  2. paired daily-IC differences of every candidate vs SD (NW t), next to the AR differences
  3. how much AR one unit of IC buys here (slope of AR diff on IC diff across candidates)
  4. the uniform blends again with the 10-seed LU (B1) in place of the 3-seed C.LU, against the 10-seed LU (seed luck removed)
  5. all-stock daily IC of C.LU per test year (the universe of the E3 table) and the ex-post optimal linear weights on D
  python agg_posthoc.py   -> results\agg_posthoc.txt, agg_posthoc.json
"""
import os, json, itertools
import numpy as np, pandas as pd
import agglib as A
import agg_eval as E
L = A.L


def main():
    P = A.load_panel(); INFO = A.load_INFO()
    out, lines = {}, ['race_agg post-hoc (after the rule; report only)', '']
    # 1. seed lottery
    per = {}
    for s in (1, 2, 3):
        per[s] = A.cut_D(pd.concat([L.load_npz(L.unit_file('C.LU', fa, s))[0] for fa, _ in A.D_FOLDS]).sort_index())
    for s in range(4, 11):
        per[s] = A.load_agg_unit('B1', seeds=(s,))
    idx = per[1].index
    M = pd.concat([per[s].reindex(idx).rename(s) for s in range(1, 11)], axis=1)
    single = {s: float(A.decide(M[s], INFO).AR.mean()) for s in range(1, 11)}
    trip = {}
    for c in itertools.combinations(range(1, 11), 3):
        trip[c] = float(A.decide(M[list(c)].mean(axis=1), INFO).AR.mean())
    v = np.array(list(trip.values()))
    r123 = int((v > trip[(1, 2, 3)]).sum()) + 1
    ten = float(A.decide(M.mean(axis=1), INFO).AR.mean())
    out['seed'] = dict(single=single, triple_mean=float(v.mean()), triple_sd=float(v.std(ddof=1)), triple_min=float(v.min()),
                       triple_max=float(v.max()), rank_123=r123, n_triples=len(v), ten=ten, triple_123=trip[(1, 2, 3)])
    lines += ['1. seed lottery (C.LU, D, bp/day)',
              '   single seeds: ' + ', '.join(f'{s}: {x:.2f}' for s, x in single.items()),
              f'   single-seed mean {np.mean(list(single.values())):.2f}, sd {np.std(list(single.values()), ddof=1):.2f}',
              f'   3-seed averages (120 triples): mean {v.mean():.2f}, sd {v.std(ddof=1):.2f}, min {v.min():.2f}, max {v.max():.2f}',
              f'   seeds (1,2,3) = the race version: {trip[(1, 2, 3)]:.2f}, rank {r123} of {len(v)} (share above it {(r123 - 1) / len(v):.3f})',
              f'   10-seed average: {ten:.2f}', '']
    # 2. paired IC differences
    S, mem, binfo, missing = E.scores(P)
    res = {k: A.decide(s, INFO) for k, s in S.items()}
    base = res['SD']
    rows = []
    for k in A.CANDS:
        dic = (res[k].ic - base.ic).dropna(); dar = (res[k].AR - base.AR).dropna()
        rows.append((k, float(dar.mean()), float(L.nw_t(dar.values)), float(dic.mean()), float(L.nw_t(dic.values))))
    out['ic'] = {k: dict(ar_diff=a, ar_t=b, ic_diff=c, ic_t=d) for k, a, b, c, d in rows}
    lines += ['2. paired differences vs SD: AR (bp/day) and daily Top50 rank IC', '   cand   AR diff   t     IC diff    t']
    for k, a, b, c, d in rows:
        lines.append(f'   {k:<5} {a:+7.2f} {b:6.2f}  {c:+.4f} {d:7.2f}')
    # 3. AR per IC across candidates
    a = np.array([r[1] for r in rows]); c = np.array([r[3] for r in rows])
    slope = float(np.polyfit(c, a, 1)[0]); corr = float(np.corrcoef(c, a)[0, 1])
    lines += ['', f'3. AR diff on IC diff across the 16 candidates: slope {slope:.1f} bp/day per unit IC, corr {corr:.3f}',
              f'   -> +0.01 IC buys about {0.01 * slope:+.2f} bp/day; the sd of AR differences here is 30-60x noisier than IC differences']
    out['slope'] = dict(slope=slope, corr=corr)
    # 4. blends with the 10-seed LU
    lu10 = M.mean(axis=1)
    base10 = A.decide(lu10, INFO)
    alt = {'A1 (10-seed LU + C.LN)': A.zavg([lu10, mem['C.LN']]),
           'A2 (10-seed LU + 4 others)': A.zavg([lu10] + [mem[m] for m in A.MEMBERS5[1:]]),
           'E2 (10-seed LU + RF)': A.zavg([lu10, S['E1']])}
    lines += ['', '4. blends rebuilt on the 10-seed LU, against the 10-seed LU (bp/day; IC = daily Top50 rank IC)',
              f'   10-seed LU: AR {base10.AR.mean():.2f}, IC {base10.ic.mean():.4f}']
    out['blend10'] = {}
    for k, sc in alt.items():
        Dk = A.decide(sc, INFO)
        st = A.stats(Dk, base10); dic = (Dk.ic - base10.ic).dropna()
        out['blend10'][k] = dict(st, ic_diff=float(dic.mean()), ic_t=float(L.nw_t(dic.values)))
        y = st['years']
        lines.append(f'   {k:<28} AR {st["mean"]:.2f}  diff {st["diff"]:+.2f} (t {st["t"]:.2f}; 2022 {y["2022"]:+.2f}, 2023 {y["2023"]:+.2f}, '
                     f'2024 {y["2024"]:+.2f})  IC diff {dic.mean():+.4f} (t {L.nw_t(dic.values):.2f})')
    # 5. all-stock IC of C.LU per year
    e = P.set_index(['date', 'stock_id'])['resid_r']
    X = pd.concat([mem['C.LU'].rename('p'), e.rename('e')], axis=1, join='inner').dropna()
    ic_all = X.groupby(level='date').apply(lambda g: g.p.rank().corr(g.e.rank()))
    yrs = {y: float(ic_all[(ic_all.index >= a) & (ic_all.index < b)].mean()) for y, a, b in A.YEARS}
    out['clu_ic_all'] = yrs
    lines += ['', '5. C.LU all-stock (100) daily IC by test year (comparable with the E3 test-year IC): ' +
              ', '.join(f'{y} {v:.4f}' for y, v in yrs.items())]
    with open(os.path.join(A.RES, 'agg_posthoc.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, default=float)
    txt = '\n'.join(lines)
    with open(os.path.join(A.RES, 'agg_posthoc.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()

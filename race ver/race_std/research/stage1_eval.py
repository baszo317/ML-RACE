r"""
stage1_eval.py -- RACE-STD v1 Stage 1 on D only (research\PREREG_race_std.md 2.3).

  python stage1_eval.py            the 25-unit grid + U-native + ENS4 + ALL5 on D; L* = learner of the best grid unit;
                                   writes results\stage1_grid.json (train.py --stage 1b then trains the ablations)
  python stage1_eval.py --final    all 32 candidates -> results\stage1_D.json, stage1_D.txt, stage1_D_daily.csv
  --test                           quick-test units (cache\units_test; the numbers mean nothing)
Only D days (2022-01-03 .. 2024-12-31) are scored; nothing here reads a V prediction.
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd
import rslib as L


def load_units(units, test):
    out, missing = {}, []
    for u in units:
        s = L.load_unit(u, L.D_FOLDS, test)
        if s is None:
            missing.append(u)
        else:
            out[u] = s
    return out, missing


def evaluate(C, names, units, INFO):
    a, b = L.PER['D']
    res, daily = {}, {}
    for n in names:
        sc = L.candidate_score(n, C, units)
        sc = sc[L.in_per(sc.index.get_level_values('date'), a, b)]
        D = L.decide_sd(sc, INFO)
        D = D[L.in_per(D.index, a, b)]
        daily[n] = D
        sm = L.summary(D, a, b)
        sm['folds'] = L.by_fold(D, L.D_FOLDS)
        res[n] = sm
    return res, daily


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--final', action='store_true')
    ap.add_argument('--test', action='store_true')
    a = ap.parse_args()
    tag = '_test' if a.test else ''
    F = L.features()
    INFO = L.info_frame(L.load_info())
    C0 = L.stage1_candidates()
    grid = L.grid_units()
    units, miss = load_units(sorted({u for c in C0.values() for u in c['units']}), a.test)
    if miss:
        sys.exit(f'units missing on D: {miss[:10]} ... ({len(miss)}) -- run train.py --stage 1 first')
    t0 = time.time()
    res, daily = evaluate(C0, list(C0), units, INFO)
    best_grid = max(grid, key=lambda u: res[u]['mean'])
    lstar = L.unit_parts(best_grid)[1]
    gj = os.path.join(L.RESULTS, f'stage1_grid{tag}.json')
    if not a.final:
        with open(gj, 'w', encoding='utf-8') as fh:
            json.dump(dict(written=time.strftime('%Y-%m-%d %H:%M:%S'), best_grid=best_grid, lstar=lstar,
                           grid={u: res[u]['mean'] for u in grid}), fh, indent=1)
        print(f'L* = {lstar} (best grid unit on D: {best_grid}, {res[best_grid]["mean"]:+.2f} bp/day); next: train.py --stage 1b')
        return
    g = json.load(open(gj, encoding='utf-8'))
    assert g['lstar'] == lstar, 'L* changed: the units on D changed after stage1_grid.json'
    C = L.stage1_candidates(lstar)
    abl = [n for n in C if n.startswith('ALLm')]
    u2, miss = load_units([u for n in abl for u in C[n]['units']], a.test)
    if miss:
        sys.exit(f'ablation units missing: {miss} -- run train.py --stage 1b')
    units.update(u2)
    r2, d2 = evaluate(C, abl, units, INFO)
    res.update(r2); daily.update(d2)
    rows = []
    for n, c in C.items():
        cx = L.complexity(c, F)
        rows.append(dict(name=n, kind=c['kind'], units=len(c['units']), feats=cx[1], **{k: v for k, v in res[n].items() if k != 'folds'},
                         **{'f' + k: v for k, v in res[n]['folds'].items()}))
    T = pd.DataFrame(rows).sort_values(['mean', 'units'], ascending=[False, True]).reset_index(drop=True)
    T.index = T.index + 1
    out = dict(written=time.strftime('%Y-%m-%d %H:%M:%S'), period=L.PER['D'], lstar=lstar, best_grid=best_grid,
               candidates={n: dict(kind=C[n]['kind'], units=C[n]['units'], of=C[n].get('of'), complexity=list(L.complexity(C[n], F)),
                                   **res[n]) for n in C}, ranking=T.name.tolist(), test=a.test)
    with open(os.path.join(L.RESULTS, f'stage1_D{tag}.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, default=float)
    DD = pd.DataFrame({n: daily[n].AR for n in daily})
    DD.to_csv(os.path.join(L.RESULTS, f'stage1_D_daily{tag}.csv'))
    # report
    nat = {'U-native': 'U-native', 'P-native': 'P.LP', 'C-native': 'C.LC', 'M-native': 'M.LM'}
    lines = [f'RACE-STD v1 Stage 1 -- D = 2022-01-03 .. 2024-12-31 ({int(T.days.iloc[0])} days), decision SD (Top25 of Top50 + ex-dividend rule)',
             f'written {out["written"]}; L* = {lstar} ({L.LEARNER_LABEL[lstar]}), from the best grid unit {best_grid}', '',
             'rank  candidate        units feats    mean   NW t     sd   pos%    IC   worst19 | 2022   2023   2024']
    for i, r in T.iterrows():
        lab = r['name'] + (' *' + [k for k, v in nat.items() if v == r['name']][0] if r['name'] in nat.values() else '')
        lines.append(f'{i:>4}  {lab:<26.26}{r.units:>3} {r.feats:>5} {r["mean"]:+7.2f} {r.nw_t:6.2f} {r.sd:6.1f} {100 * r.pos:5.1f} '
                     f'{r.ic:6.3f} {r.worst19:+8.1f} | {r.f2022:+6.2f} {r.f2023:+6.2f} {r.f2024:+6.2f}')
    lines += ['', '* = the member-native candidate (P.LP = FinPred, C.LC = Chen V5, M.LM = NTU M2).', '',
              'grid: mean AR on D (rows = feature set, columns = learner)']
    G = pd.DataFrame({l: {g: res[f'{g}.{l}']['mean'] for g in L.GROUPS + ('ALL',)} for l in L.LEARNERS})
    lines.append(G.round(2).to_string())
    lines += ['', 'paired differences on D vs each native (NW t):']
    for n in T.name.head(6):
        parts = []
        for k, v in nat.items():
            if v == n:
                continue
            dd = (daily[n].AR - daily[v].AR).dropna()
            parts.append(f'vs {k} {dd.mean():+.2f} ({L.nw_t(dd.values):.2f})')
        lines.append(f'  {n:<16} ' + '  '.join(parts))
    lines += ['', 'correlation of daily AR on D among the natives and the top 3:']
    cols = list(dict.fromkeys(list(nat.values()) + T.name.head(3).tolist()))
    lines.append(DD[cols].corr().round(2).to_string())
    txt = '\n'.join(lines)
    with open(os.path.join(L.RESULTS, f'stage1_D{tag}.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    print(txt)
    print(f'({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()

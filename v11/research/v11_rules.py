r"""
v11_rules.py -- selection rules of research\PREREG_v11.md, their nested walk-forward comparison and the final v11 groups.

Rules (per family; a window W = [start, end) of walk-forward days is the only data a rule sees):
  R0  nothing (v9)
  R1  v10's rule: AR plain t > 0 against the seed 1-5 baseline, all 23 groups are candidates, union of the passing groups
  R2  rank-IC NW t > 0 (against the 4 baseline blocks), candidates after Stage 0, union
  R3  rank-IC NW t > 2 and the mean of every year of W > 0, candidates after Stage 0, union
  R4  v11's rule: rank-IC NW t > Bonferroni(K) and every year of W > 0 -> forward selection (keep g when IC(S+g) - IC(S) has
      NW t > 0 on W) -> AR guard (W's AR difference of the final S, NW t > -1, else nothing)
  R5  ALL 158 (no selection)
Nested walk-forward: outer periods 2023, 2024, 2025, 2026a (Jan-May), post; W = 2022-01-01 .. the day before the period.
A union of 2+ groups needs its own training: in an outer period only that fold (seed 1-5); R4's forward-selection steps all folds.
Missing trainings are written to ..\cache\jobs_needed.json (python v11_train.py --job list --file ..\cache\jobs_needed.json);
run again after training until nothing is missing.
run (in v11\research): python v11_rules.py > ..\results\v11_rules.txt
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v11lib as X
import v11_metrics as MT
import v11_train as T
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 300)

OUTER = [('2023', '2023-01-01', '2024-01-01', '2023-01-01'), ('2024', '2024-01-01', '2025-01-01', '2024-01-01'),
         ('2025', '2025-01-01', '2026-01-01', '2025-01-01'), ('2026a', '2026-01-01', '2026-06-01', '2026-01-01'),
         ('post', '2026-06-01', '2100-01-01', '2026-06-01')]
W0 = '2022-01-01'
RULES = ['R1', 'R2', 'R3', 'R4', 'R5']
NEED = []                                   # trainings still missing: dicts for v11_train.py --job list

class Missing(Exception):
    pass

def S0():
    return json.load(open(os.path.join(X.CACHE, 'stage0.json')))

def cands(fam, rule):
    return list(X.ORDER) if rule == 'R1' else list(S0()[fam]['kept'])

def win(d, start, end):
    return d[X.window(d.index, start, end)].dropna()

def ic_stats(fam, name, start, end):
    d = win(MT.delta(fam, name, 'ic'), start, end)
    return d.mean(), X.nw_t(d.values), d

V10SEL = json.load(open(os.path.join(X.ROOT, 'v10', 'results', 'v10_selection.json')))

def union_metrics(fam, groups, fold=None):
    """metrics of base + groups (2+ groups): all folds, or only `fold` (other days filled with baseline block 1, never used).
    All 23 groups = the ALL config of the v10 screen; v10's selection = the V10 units of v10 (seed 1-5)"""
    if list(groups) == list(X.ORDER):
        return MT.get(fam, 'ALL')
    name = T.config_name(groups)
    if len(groups) == 1:
        return MT.get(fam, name)
    if T.config_name(V10SEL[fam]) == name and (fam, name, 's1-5') not in MT.store():
        d = {m: X.unit_mean('v10', 'v10', m, range(1, 6)) for m in X.MEM[fam]}
        if all(v is not None for v in d.values()):
            return MT.get(fam, name, 's1-5', d=d)
    if fold is None:
        k = (fam, name, 's1-5')
        if k in MT.store():
            return MT.store()[k]
        d = X.config_preds(fam, groups)
        if d is None:
            NEED.append({'family': fam, 'groups': list(groups)}); raise Missing(name)
        return MT.get(fam, name, 's1-5', d=d)
    k = (fam, name, 's1-5')
    if k in MT.store():                     # trained on all folds already
        return MT.store()[k]
    k = (fam, name, f's1-5@{fold}')
    if k in MT.store():
        return MT.store()[k]
    d = X.config_preds(fam, groups, folds_only=[fold])
    if d is None:
        NEED.append({'family': fam, 'groups': list(groups), 'folds': [fold]}); raise Missing(name + '@' + fold)
    b1 = X.base_blocks(fam)[0]
    full = {m: pd.concat([d[m], b1[m][~b1[m].index.isin(d[m].index)]]).sort_index() for m in d}
    return MT.get(fam, name, f's1-5@{fold}', d=full)

def delta_of(fam, groups, what, start, end, fold=None):
    """daily difference (to the 4 baseline blocks) of base + groups on [start, end); zeros if groups is empty"""
    b = MT.base(fam, what)
    days = win(b, start, end).index
    if not groups:
        return pd.Series(0.0, index=days)
    m = MT.get(fam, groups[0]) if len(groups) == 1 else union_metrics(fam, groups, fold)   # all 23 -> ALL
    x = m[what].reindex(days)
    return (x - b.reindex(days)).dropna()

def select(rule, fam, start, end, log=None):
    """groups chosen by `rule` from the days [start, end)"""
    note = []
    if rule == 'R5':
        return list(X.ORDER), 'ALL'
    C = cands(fam, rule)
    if rule == 'R1':
        S = [g for g in C if X.V.plain_t(win(MT.delta(fam, g, 'ar', ref='b1'), start, end).values) > 0]
        return S, ''
    st = {g: ic_stats(fam, g, start, end) for g in C}
    if rule == 'R2':
        return [g for g in C if st[g][1] > 0], ''
    if rule == 'R3':
        return [g for g in C if st[g][1] > 2 and X.every_year_positive(st[g][2], start, end)], ''
    # R4
    tb = X.t_bonferroni(len(C))
    passed = sorted([g for g in C if st[g][1] > tb and X.every_year_positive(st[g][2], start, end)], key=lambda g: -st[g][1])
    note.append(f'screen (t > {tb:.2f}, every year > 0): ' + (', '.join(f'{g} {st[g][1]:.2f}' for g in passed) or 'none'))
    if not passed:
        return [], '; '.join(note)
    S = [passed[0]]
    for g in passed[1:]:
        cur = MT.get(fam, S[0])['ic'] if len(S) == 1 else union_metrics(fam, S)['ic']
        new = union_metrics(fam, S + [g])['ic']
        inc = win((new - cur.reindex(new.index)), start, end)
        t = X.nw_t(inc.values)
        note.append(f'+{g}: increment {inc.mean() * 1e3:+.2f}e-3 (t {t:.2f}) -> {"keep" if t > 0 else "drop"}')
        if t > 0:
            S = S + [g]
    dA = win(MT.delta(fam, S[0], 'ar') if len(S) == 1 else
             (union_metrics(fam, S)['ar'] - MT.base(fam, 'ar')).dropna(), start, end)
    tA = X.nw_t(dA.values)
    note.append(f'AR guard: {"+".join(S)} AR diff {dA.mean():+.2f} bp (t {tA:.2f}) -> {"ok" if tA > -1 else "fail: nothing"}')
    return (S if tA > -1 else []), '; '.join(note)

def nested(fam):
    rows, series = [], {}
    for r in RULES:
        dI, dA = [], []
        for name, a, b, fold in OUTER:
            S, note = select(r, fam, W0, a)
            xi = delta_of(fam, S, 'ic', a, b, fold); xa = delta_of(fam, S, 'ar', a, b, fold)
            dI.append(xi); dA.append(xa)
            rows.append({'rule': r, 'period': name, 'n_sel': len(S), 'selected': 'ALL' if S == list(X.ORDER) else '+'.join(S),
                         'dIC_e3': xi.mean() * 1e3, 'tIC': X.nw_t(xi.values), 'dAR': xa.mean(), 'tAR': X.nw_t(xa.values), 'note': note})
        series[r] = (pd.concat(dI), pd.concat(dA))
    return pd.DataFrame(rows), series

def main():
    t0 = time.time()
    res = {}
    missing = False
    for fam in ('lgb', 'mlp'):
        try:
            R, ser = nested(fam)
            res[fam] = (R, ser)
        except Missing as e:
            missing = True
            print(f'{fam}: training missing ({e}); collecting the rest ...')
            # collect every missing training of this family in one pass
            for r in RULES:
                for name, a, b, fold in OUTER:
                    try:
                        S, _ = select(r, fam, W0, a); delta_of(fam, S, 'ic', a, b, fold)
                    except Missing:
                        pass
    MT.save()
    if missing:
        jobs, seen = [], set()
        for j in NEED:
            k = (j['family'], tuple(j['groups']), tuple(j.get('folds') or []))
            if k not in seen:
                seen.add(k); jobs.append(j)
        f = os.path.join(X.CACHE, 'jobs_needed.json')
        json.dump(jobs, open(f, 'w'), indent=1)
        print(f'{len(jobs)} trainings missing -> {f}')
        for j in jobs:
            print('  ', j['family'], '+'.join(j['groups']), j.get('folds', 'all folds'))
        return
    print('nested walk-forward: every outer period is selected on 2022-01-01 .. the day before it; difference to the 4 baseline blocks')
    final = {}
    for fam in ('lgb', 'mlp'):
        R, ser = res[fam]
        print(f'\n{fam}: by period (dIC in 1e-3 rank IC of the family; dAR in Top50 bp/day, only this family replaced)')
        print(R.drop(columns='note').round(2).to_string(index=False))
        for r in RULES:
            for x in R[(R.rule == r) & (R.note != '')].itertuples():
                print(f'   {r} {x.period}: {x.note}')
        pooled = []
        for r in RULES:
            i, a = ser[r]
            pooled.append({'rule': r, 'dIC_e3': i.mean() * 1e3, 'tIC': X.nw_t(i.values), 'dAR': a.mean(), 'tAR': X.nw_t(a.values),
                           'days': len(i), 'periods_IC>0': int((R[R.rule == r].dIC_e3 > 0).sum()), 'periods_AR>0': int((R[R.rule == r].dAR > 0).sum())})
        Pd = pd.DataFrame(pooled)
        Pd['eligible'] = (Pd.tIC > 2) & (Pd.dAR > 0)
        print(f'{fam}: pooled over the outer periods (2023-01 .. 2026-10-22)')
        print(Pd.round(3).to_string(index=False))
        el = Pd[Pd.eligible]
        rule = el.sort_values('dIC_e3', ascending=False).rule.iloc[0] if len(el) else None
        final[fam] = {'rule': rule, 'pooled': Pd.round(4).to_dict('records')}
        print(f'{fam}: final rule (eligible: pooled IC t > 2 and pooled AR > 0; highest pooled dIC) -> {rule or "none: no Alpha158 group"}')
    # ---------------------------------------------------------------- the final groups (W_live) and the val-only report
    for fam in ('lgb', 'mlp'):
        r = final[fam]['rule']
        if r is None:
            final[fam].update(groups=[], groups_val=[]); continue
        try:
            S, note = select(r, fam, W0, '2100-01-01')
            Sv, notev = select(r, fam, W0, '2025-01-01')
            if len(S) > 1:
                union_metrics(fam, S)
            if len(Sv) > 1:
                union_metrics(fam, Sv)
        except Missing:
            missing = True; continue
        final[fam].update(groups=S, note=note, groups_val=Sv, note_val=notev)
        print(f'\n{fam}: {r} on all walk-forward days (2022-01-03 .. 2026-10-22) -> {"+".join(S) or "nothing"}   {note}')
        print(f'{fam}: {r} on val only (report) -> {"+".join(Sv) or "nothing"}   {notev}')
        for lab, G in (('final', S), ('val-only', Sv)):
            if not G:
                continue
            for nm, (a, b) in (('val', ('2022-01-01', '2025-01-01')), ('hold', ('2025-01-01', '2026-06-01')), ('post', ('2026-06-01', '2100-01-01')),
                               ('all', ('2022-01-01', '2100-01-01'))):
                i = delta_of(fam, G, 'ic', a, b); x = delta_of(fam, G, 'ar', a, b)
                print(f'   {lab} {"+".join(G)} {nm}: dIC {i.mean() * 1e3:+.2f}e-3 (t {X.nw_t(i.values):.2f}), dAR {x.mean():+.2f} (t {X.nw_t(x.values):.2f})')
    MT.save()
    if missing:
        jobs = [j for j in NEED]
        f = os.path.join(X.CACHE, 'jobs_needed.json'); json.dump(jobs, open(f, 'w'), indent=1)
        print(f'{len(jobs)} trainings missing for the final groups -> {f}'); return
    json.dump(final, open(os.path.join(X.RESULTS, 'v11_selection.json'), 'w'), indent=1, default=str)
    print(f'\nwritten ..\\results\\v11_selection.json; done in {time.time() - t0:.0f}s')

if __name__ == '__main__':
    main()

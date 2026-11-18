r"""
allcand_reference.py -- REFERENCE ONLY. The walk-forward (annual refit, expanding window from 2020-03-02, same folds as v9..v16)
of every candidate of part 1 and every variant of part 2, by fold, from the files that already exist. Nothing is retrained and
nothing here is used for a decision.

  D = 2022, 2023, 2024: all 32 candidates of part 1 + ENS_TOP3 + the 12 variants of part 2
  V = 2025, 2026-01..05, 2026-06..11-09: only the 8 finalists of part 1 were ever run on V (V used once, 11/15 23:55)

Writes results\allcand_wf_reference.txt / .csv and results\allcand_wf_daily.csv (date x candidate, AR in bp).
"""
import os, sys, json, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); RACEVER = os.path.dirname(HERE)
STD = os.path.join(RACEVER, 'race_std')
sys.path.insert(0, os.path.join(STD, 'research'))
import rslib as L                                                    # noqa: E402  (frozen, read only)

RES = os.path.join(HERE, 'results')
SRC = os.path.join(STD, 'results')
FOLD_COLS = [('2022', '2022-01-01', '2023-01-01'), ('2023', '2023-01-01', '2024-01-01'), ('2024', '2024-01-01', '2025-01-01'),
             ('2025', '2025-01-01', '2026-01-01'), ('2026-01..05', '2026-01-01', '2026-06-01'), ('2026-06..', '2026-06-01', '2100-01-01')]
LABEL = {'LU': 'LightGBM binary', 'LN': 'MLP', 'LP': 'HistGBR', 'LC': 'LightGBM huber', 'LM': 'LGBM median-imputer'}
P2_LABEL = {'SD': 'C.LU, Top25 (baseline of part 2)', 'H75': 'beta hedge 75%', 'H50': 'beta hedge 50%', 'T3': 'Top25 x 3-day hold',
            'TH': 'threshold', 'W0D': 'user W0 market view', 'MV25': 'market view 25%', 'MV50': 'market view 50%',
            'MV50T2': 'market view 50% + T2', 'V11D': 'v11 daily view', 'QR': 'quarterly refit', 'MR': 'monthly refit',
            'WF3': 'daily refit, mean of the last 3 days (race version)', 'WF1': 'daily refit, newest model only'}


def describe(name):
    if '.' in name and name.split('.')[-1] in LABEL:
        fk, ln = name.rsplit('.', 1)
        return f'features {fk}, {LABEL[ln]}'
    return {'U-native': 'user v9 W0 members (4 MLP + 2 LGBM)', 'ENS4': 'z-mean of the 4 natives',
            'ALL5': 'z-mean of ALL x 5 learners', 'ENS_TOP3': 'z-mean of ALL.LN, ALLmU.LN, C.LU'}.get(name, '')


def main():
    D = pd.read_csv(os.path.join(SRC, 'stage1_D_daily.csv'), index_col=0, parse_dates=True)
    lock = json.load(open(os.path.join(SRC, 'LOCK.json'), encoding='utf-8'))
    defs = lock['defs']
    units = {u: L.load_unit(u, L.D_FOLDS) for u in defs['ENS_TOP3']['units']}
    info = L.load_info()
    INFO_D = L.info_frame(info[info.date < pd.Timestamp('2025-01-01')])
    D['ENS_TOP3'] = L.decide_sd(L.candidate_score('ENS_TOP3', defs, units), INFO_D).AR.reindex(D.index)
    m_d = INFO_D.groupby(level='date').target_r.mean() * 1e4
    D['all 1'] = m_d.reindex(D.index); D['all 0'] = -m_d.reindex(D.index)
    V2 = pd.read_csv(os.path.join(SRC, 'stage2_V_daily.csv'), header=[0, 1], index_col=0)
    V2 = V2[[c for c in V2.columns if c[1] == 'AR']]
    V2.columns = [c[0] for c in V2.columns]
    V2.index = pd.to_datetime(V2.index, errors='coerce')
    V2 = V2[V2.index.notna()].astype(float)
    mv = pd.read_csv(os.path.join(SRC, 'stage2_V_daily.csv'), header=[0, 1], index_col=0)
    mv = mv[[c for c in mv.columns if c[1] == 'm']].iloc[:, 0]; mv.index = pd.to_datetime(mv.index, errors='coerce'); mv = mv[mv.index.notna()]
    V2['all 1'] = mv.astype(float); V2['all 0'] = -mv.astype(float)
    p2 = pd.read_csv(os.path.join(STD, 'part2', 'results', 'part2_D_daily.csv'), index_col=0, parse_dates=True)
    p2.columns = ['p2:' + c for c in p2.columns]
    f3 = os.path.join(RES, 'wf_daily_daily.csv')                    # part 3 (race_live walk-forward), if it has been run
    if os.path.exists(f3):
        p3 = pd.read_csv(f3, index_col=0, parse_dates=True)[['WF3', 'WF1']]
        p3.columns = ['p3:' + c for c in p3.columns]
        p2 = pd.concat([p2, p3], axis=1)
    A = pd.concat([pd.concat([D, p2], axis=1), V2], axis=0, sort=False).sort_index()
    A.index.name = 'date'
    order = list(D.columns) + list(p2.columns)
    A = A[order]

    rows = []
    for c in order:
        x = A[c].astype(float)
        r = dict(candidate=c, what=describe(c) if not c.startswith(('p2:', 'p3:')) else P2_LABEL.get(c[3:], ''))
        for nm, a, b in FOLD_COLS:
            y = x[(x.index >= a) & (x.index < b)].dropna()
            r[nm] = float(y.mean()) if len(y) else np.nan
        for per, (a, b) in L.PER.items():
            y = x[(x.index >= a) & (x.index < b)].dropna()
            r[f'{per} mean'] = float(y.mean()) if len(y) else np.nan
            r[f'{per} NW t'] = float(L.nw_t(y.values)) if len(y) > 20 else np.nan
            r[f'{per} sd'] = float(y.std()) if len(y) > 1 else np.nan
            r[f'{per} days'] = int(len(y))
        rows.append(r)
    T = pd.DataFrame(rows).set_index('candidate')
    T['finalist'] = [c in lock['finalists'] for c in T.index]
    os.makedirs(RES, exist_ok=True)
    T.to_csv(os.path.join(RES, 'allcand_wf_reference.csv'), encoding='utf-8-sig')
    A.to_csv(os.path.join(RES, 'allcand_wf_daily.csv'), encoding='utf-8-sig')

    fold_names = [f[0] for f in FOLD_COLS]
    hdr = f'{"candidate":12s} ' + ' '.join(f'{n:>10s}' for n in fold_names) + f' {"D mean":>7s} {"D t":>5s} {"V mean":>7s} {"V t":>5s}'
    out = ['REFERENCE ONLY -- annual walk-forward (expanding window from 2020-03-02, refit each fold), AR in bp per day, decision SD',
           '(Top25 of Top50 + ex-dividend rule; part 2 variants use their own rules). D = 2022-2024, V = 2025-2026/11/09.',
           'V exists only for the 8 finalists (V was used once, 11/15 23:55); blanks were never run on V and will not be.',
           'Not a basis for changing the race version: the V numbers of the finalists are already the selection data.',
           f'written {time.strftime("%Y-%m-%d %H:%M:%S")} from race_std\\results\\stage1_D_daily.csv, stage2_V_daily.csv, LOCK.json and',
           'race_std\\part2\\results\\part2_D_daily.csv and race_live\\results\\wf_daily_daily.csv', '']
    for title, sel in (('part 1 candidates (sorted by D mean)', [c for c in T.index if not c.startswith(('p2:', 'p3:')) and not c.startswith('all ')]),
                       ('baselines', ['all 1', 'all 0']),
                       ('part 2 variants on D (built on C.LU)', [c for c in T.index if c.startswith('p2:')]),
                       ('part 3 daily refit on D (race_live walk-forward; C.LU, one model per day)', [c for c in T.index if c.startswith('p3:')])):
        if not sel:
            continue
        out.append(title); out.append(hdr)
        sub = T.loc[sel]
        if title.startswith('part 1'):
            sub = sub.sort_values('D mean', ascending=False)
        for c, r in sub.iterrows():
            f = lambda v: f'{v:10.2f}' if pd.notna(v) else f'{"":10s}'
            g = lambda v, w=7: f'{v:{w}.2f}' if pd.notna(v) else ' ' * w
            out.append(f'{c[:12]:12s} ' + ' '.join(f(r[n]) for n in fold_names) +
                       f' {g(r["D mean"])} {g(r["D NW t"], 5)} {g(r["V mean"])} {g(r["V NW t"], 5)}' + ('  *' if r['finalist'] else ''))
        out.append('')
    out.append('* = finalist (run on V). Columns 2022..2026-06.. are the fold means; D t / V t are Newey-West t (5 lags).')
    txt = '\n'.join(out)
    with open(os.path.join(RES, 'allcand_wf_reference.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()

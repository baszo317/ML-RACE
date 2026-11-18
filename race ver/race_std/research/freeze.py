r"""
freeze.py -- RACE-STD v1: the finalists for V by the mechanical rule of research\PREREG_race_std.md 2.4, written to
results\LOCK.json together with the SHA-256 of the code, the data snapshot and the PREREG.

  rule: the 32 Stage-1 candidates ranked by mean AR on D (ties: fewer units); F1, F2, F3 = the top three;
        ENS_TOP3 = mean of their daily z; plus the four natives (U-native, P.LP, C.LC, M.LM) not already in the list.

  python freeze.py            write results\LOCK.json (refuses if results\SEALED_V.json exists)
  python freeze.py --replace  rewrite an existing LOCK.json (only before V is opened; the old one is kept as LOCK_<time>.json)
  --test                      from the quick-test Stage 1 (results\LOCK_test.json)
"""
import os, sys, json, time, argparse, shutil
import rslib as L

NATIVES = ['U-native', 'P.LP', 'C.LC', 'M.LM']


def closure(names, cands):
    """all candidate definitions needed to compute the scores of `names` (z-averages refer to other candidates)"""
    out = {}
    def add(n):
        if n in out:
            return
        c = cands[n]
        out[n] = dict(kind=c['kind'], units=c['units'], of=c.get('of'))
        for m in c.get('of') or []:
            add(m)
    for n in names:
        add(n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replace', action='store_true')
    ap.add_argument('--test', action='store_true')
    a = ap.parse_args()
    tag = '_test' if a.test else ''
    lock_f = os.path.join(L.RESULTS, f'LOCK{tag}.json')
    if os.path.exists(os.path.join(L.RESULTS, f'SEALED_V{tag}.json')):
        sys.exit('SEALED_V exists: V has been opened, the finalists can no longer change (PREREG 3)')
    if os.path.exists(lock_f):
        if not a.replace:
            sys.exit(f'{lock_f} exists (use --replace only before V is opened, e.g. after fixing a bug in Stage 1)')
        shutil.copy(lock_f, lock_f.replace('.json', time.strftime('_%Y%m%d_%H%M%S.json')))
    S = json.load(open(os.path.join(L.RESULTS, f'stage1_D{tag}.json'), encoding='utf-8'))
    cands = S['candidates']
    rank = sorted(cands, key=lambda n: (-cands[n]['mean'], len(cands[n]['units'])))
    top3 = rank[:3]
    defs = closure(top3 + NATIVES, cands)
    units = sorted({u for n in top3 for u in cands[n]['units']})
    defs['ENS_TOP3'] = dict(kind='zavg', units=units, of=top3)
    finalists = list(dict.fromkeys(top3 + ['ENS_TOP3'] + NATIVES))
    F = L.features()
    for n in finalists:
        defs[n]['complexity'] = list(L.complexity(defs[n], F))
    snap = json.load(open(os.path.join(L.RESULTS, 'snapshot.json'), encoding='utf-8'))
    lock = dict(written=time.strftime('%Y-%m-%d %H:%M:%S'), rule='PREREG 2.4: top 3 on D + ENS_TOP3 + 4 natives',
                ranking_D=[(n, round(cands[n]['mean'], 4), len(cands[n]['units'])) for n in rank],
                top3=top3, finalists=finalists, defs=defs, finalist_units=sorted({u for n in finalists for u in defs[n]['units']}),
                D={n: {k: cands[n][k] for k in ('mean', 'nw_t', 'sd', 'ic', 'days')} for n in finalists if n in cands},
                prereg_sha256=L.sha256(L.PREREG), code=L.code_hashes(), data_sha256=snap['data_sha256'],
                lstar=S['lstar'], test=a.test)
    with open(lock_f, 'w', encoding='utf-8') as fh:
        json.dump(lock, fh, indent=1, ensure_ascii=False, default=float)
    print(f'LOCK written {lock["written"]}: finalists {finalists}')
    print(f'units to train on V: {len(lock["finalist_units"])} -> {lock["finalist_units"]}')


if __name__ == '__main__':
    main()

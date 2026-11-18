r"""
p2_eval.py -- RACE-STD v1 part 2 (part2\PREREG_part2.md): every add-on on C.LU, D only, and the rule of section 3.

  python p2_eval.py            F1-F4 (and F5 when its units are complete); results\part2_D.txt, part2_D.json, part2_D_daily.csv
Nothing here reads a V prediction: C.LU is loaded for the D folds only and every series is cut to 2022-01-01 .. 2024-12-31.
"""
import os, sys, json, time, math
import numpy as np, pandas as pd
import p2lib as Q
L = Q.L

YEARS = [('2022', '2022-01-01', '2023-01-01'), ('2023', '2023-01-01', '2024-01-01'), ('2024', '2024-01-01', '2025-01-01')]
FAMILY = {'H75': 'F1', 'H50': 'F1', 'T3': 'F2', 'TH': 'F3', 'W0D': 'F3', 'MV25': 'F4', 'MV50': 'F4', 'MV50T2': 'F4', 'V11D': 'F4',
          'QR': 'F5', 'MR': 'F5'}


def cut_D(s):
    dts = s.index.get_level_values('date')
    return s[(dts >= Q.D_A) & (dts < Q.D_B)]


def load_refit(freq):
    import p2_train as PT
    parts = []
    for fa, _ in PT.refit_days(freq):
        ss = []
        for sd in PT.SEEDS:
            f = PT.unit_file(freq, fa, sd)
            if not os.path.exists(f):
                return None
            ss.append(L.load_npz(f)[0])
        parts.append(pd.concat(ss, axis=1).mean(axis=1))
    return pd.concat(parts).sort_index()


def stats(D, base):
    d = (D.AR - base.AR).dropna()
    yrs = {y: float(d[(d.index >= a) & (d.index < b)].mean()) for y, a, b in YEARS}
    t = float(L.nw_t(d.values))
    roll = D.AR.rolling(19).mean().dropna()
    return dict(mean=float(D.AR.mean()), diff=float(d.mean()), t=t, years=yrs, pos_years=int(sum(v > 0 for v in yrs.values())),
                sd=float(D.AR.std()), p19_neg=float((roll < 0).mean()), ones=float(D.ones.mean()), timing=float(D.timing.mean()),
                xsec=float(D.xsec.mean()), changed=float(D.changed.mean()),
                passes=bool(d.mean() > 0 and t >= 2.0 and sum(v > 0 for v in yrs.values()) >= 2))


def holm(pvals):
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); out = {}; run = 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p)); out[k] = run
    return out


def main():
    t0 = time.time()
    os.makedirs(Q.RES2, exist_ok=True)
    INFO = Q.info_D()
    p = cut_D(L.load_unit('C.LU', L.D_FOLDS))
    pc = Q.center(p)
    dates = INFO.index.get_level_values('date')
    ms = Q.month_start_days(dates)
    pb = Q.pbar_user(); z = Q.night_z(); pb2 = Q.pbar_t2(pb, z)
    pdates = pc.index.get_level_values('date')
    mv = lambda w, pbar: pd.Series(w * (pbar.reindex(pdates).to_numpy() - 0.5) + pc.to_numpy(), index=pc.index)
    res = {}
    res['SD'] = Q.decide_topk(pc, INFO)
    sd_stage1 = json.load(open(os.path.join(L.RESULTS, 'stage1_D.json'), encoding='utf-8'))['candidates']['C.LU']['mean']
    assert abs(res['SD'].AR.mean() - sd_stage1) < 1e-9, (res['SD'].AR.mean(), sd_stage1)
    res['H75'] = Q.decide_topk(pc, INFO, lam=0.75)
    res['H50'] = Q.decide_topk(pc, INFO, lam=0.5)
    res['T3'] = Q.decide_topk(pc, INFO, k_days={d: 30 for d in ms})
    res['TH'] = Q.decide_thresh(pc, INFO)
    res['W0D'] = Q.decide_thresh(pc, INFO, lam=0.5)
    res['MV25'] = Q.decide_thresh(mv(0.25, pb), INFO)
    res['MV50'] = Q.decide_thresh(mv(0.5, pb), INFO)
    res['MV50T2'] = Q.decide_thresh(mv(0.5, pb2), INFO)
    res['V11D'] = Q.decide_thresh(mv(0.5, pb2), INFO, month_start=ms, lam=0.75)
    missing_f5 = []
    for nm, fq in (('QR', 'Q'), ('MR', 'M')):
        s = load_refit(fq)
        if s is None:
            missing_f5.append(nm)
        else:
            res[nm] = Q.decide_topk(Q.center(cut_D(s)), INFO)
    base = res['SD']
    st = {k: stats(v, base) for k, v in res.items() if k != 'SD'}
    st['SD'] = stats(base, base); st['SD'].update(diff=0.0, t=float('nan'), passes=False)
    pv = {k: 1 - 0.5 * (1 + math.erf(v['t'] / math.sqrt(2))) for k, v in st.items() if k != 'SD' and v['t'] == v['t']}
    hp = holm(pv)
    # ---------------- rule (PREREG part 2, section 3)
    fam_win = {}
    for f in ('F1', 'F2', 'F3', 'F4'):
        ok = [k for k in st if FAMILY.get(k) == f and st[k]['passes']]
        if ok:
            fam_win[f] = max(ok, key=lambda k: st[k]['diff'])
    combo = None
    if len(fam_win) >= 2:
        dl = fam_win.get('F4') or fam_win.get('F3')
        t3 = 'F2' in fam_win
        lam = {'H75': 0.75, 'H50': 0.5}.get(fam_win.get('F1'))
        if dl is None:
            D = Q.decide_topk(pc, INFO, k_days=({d: 30 for d in ms} if t3 else None), lam=lam)
        else:
            score = {'TH': pc, 'W0D': pc, 'MV25': mv(0.25, pb), 'MV50': mv(0.5, pb), 'MV50T2': mv(0.5, pb2), 'V11D': mv(0.5, pb2)}[dl]
            own = {'W0D': 0.5, 'V11D': 0.75}.get(dl)
            D = Q.decide_thresh(score, INFO, month_start=(ms if (t3 or dl == 'V11D') else None),
                                lam=(own if own is not None else (lam if lam is not None else 1.0)))
        res['COMBO'] = D; st['COMBO'] = stats(D, base)
        combo = dict(parts=list(fam_win.values()), passes=st['COMBO']['passes'])
    if not fam_win:
        adopted = []
    elif combo is not None and combo['passes']:
        adopted = combo['parts']
    else:
        adopted = [max(fam_win.values(), key=lambda k: st[k]['diff'])]
    f5 = {k: st[k] for k in ('QR', 'MR') if k in st}
    f5_text = ('missing units: ' + ', '.join(missing_f5)) if missing_f5 else \
        ('fresher models help: refit live every day' if any(v['passes'] for v in f5.values()) else
         'no evidence that fresher models help: the refit frequency is an operational choice')
    out = dict(written=time.strftime('%Y-%m-%d %H:%M:%S'), period=[Q.D_A, Q.D_B], days=int(len(base)), stats=st, holm=hp,
               family_winners=fam_win, combo=combo, adopted=adopted, f5=f5_text, missing_f5=missing_f5)
    with open(os.path.join(Q.RES2, 'part2_D.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=float)
    pd.DataFrame({k: v.AR for k, v in res.items()}).to_csv(os.path.join(Q.RES2, 'part2_D_daily.csv'))
    lab = {'SD': 'Top25 + ex-div (race, part 1)', 'H75': 'hedge lambda 0.75', 'H50': 'hedge lambda 0.5', 'T3': 'month start Top30',
           'TH': 'threshold p_c > 0', 'W0D': 'W0 layer (threshold + hedge 0.5)', 'MV25': 'Baszo pbar, w_T 0.25',
           'MV50': 'Baszo pbar, w_T 0.5', 'MV50T2': 'pbar w_T 0.5 + T2', 'V11D': 'v11 layer (w_T 0.5, T2, T3, hedge 0.75)',
           'QR': 'refit every quarter', 'MR': 'refit every month', 'COMBO': 'passing parts combined'}
    lines = [f'RACE-STD v1 part 2 -- add-ons on C.LU, D only ({Q.D_A} .. 2024-12-31, {len(base)} days); V not opened',
             f'written {out["written"]}; SD reproduces part 1 (C.LU on D {sd_stage1:+.4f})', '',
             'variant  description                               D mean   diff   NW t  2022   2023   2024  pass  Holm p    sd   P19<0  1s    changed/day']
    for k in ['SD', 'H75', 'H50', 'T3', 'TH', 'W0D', 'MV25', 'MV50', 'MV50T2', 'V11D', 'QR', 'MR', 'COMBO']:
        if k not in st:
            continue
        v = st[k]; y = v['years']
        lines.append(f'{k:<8} {lab[k]:<40.40} {v["mean"]:+7.2f} {v["diff"]:+6.2f} {v["t"]:6.2f} {y["2022"]:+6.2f} {y["2023"]:+6.2f} '
                     f'{y["2024"]:+6.2f}  {"yes" if v["passes"] else "no ":<4} {hp.get(k, float("nan")):6.3f} {v["sd"]:5.1f} {v["p19_neg"]:5.2f} '
                     f'{50 * v["ones"]:5.1f} {v["changed"]:5.1f}')
    lines += ['', f'family winners (mean > 0, NW t >= 2, >= 2 of 3 years > 0): {fam_win if fam_win else "none"}',
              f'combination: {combo}', f'ADOPTED into the race version: {adopted if adopted else "nothing (race version stays SD)"}',
              f'F5: {f5_text}', '',
              'timing / selection decomposition on D (bp/day): ' + ', '.join(f'{k} {st[k]["timing"]:+.2f} / {st[k]["xsec"]:+.2f}' for k in st)]
    txt = '\n'.join(lines)
    with open(os.path.join(Q.RES2, 'part2_D.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    print(txt)
    print(f'({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()

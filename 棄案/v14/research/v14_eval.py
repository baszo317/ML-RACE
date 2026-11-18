r"""
v14_eval.py -- v14 candidates judged by the rules written before their results (research\PREREG_v14.md). numpy and pandas only.
Run v14_signals.py first (it writes ..\cache\signals.pkl, ..\cache\calendar.pkl).

Part 0  rebuild v9, v11, W0 from the cached walk-forward predictions and check them against the published numbers
Part 1  (report) each signal's own daily rank IC, correlation with the v11 stock view, two-signal diagnostics (README_v14 3.1);
        E5: every family's IC, the correlation matrix and the n-signal bound (README_v14 theorem 4)
Part 2  (the rules) v11 + candidate - v11: val > 0 -> selected; selected together: hold > 0 and post > 0 -> passed
Part 3  (report only) W0 versions, the v13 placement, variants, E1 (upper bound with the real open), by year, October, power, table

usage (in v14\research):  python v14_eval.py  > ..\results\v14_eval.txt      (about 3-6 minutes)
"""
import os, sys, json, math, time, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import v14lib as L
from v14lib import (c, sdd, per, w, ic_daily, nw_t, zday, spread_match, with_families, neutralize_top50, diff, fmt, power,
                    PER, YEARS, OCT, PUBLISHED, ROOT, V14)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)

E5_FAMILIES = [  # name, folder under ROOT, kwargs of load_member
    ('SVM (reg_svm exact)', ('reg_svm', 'cache', 'svm_exact'), dict(no_seed=True)),
    ('LSTM-R (v13)', ('v13', 'cache', 'seq_raw', 'lstm'), {}),
    ('CNN-R (v13)', ('v13', 'cache', 'seq_raw', 'tcn'), {}),
    ('F-LOAM (v13)', ('v13', 'cache', 'seq_raw', 'floam'), {}),
    ('lgb_sign (v12)', ('v12', 'cache', 'screen', 'lgb_sign'), {}),
    ('lgb_dec_l2 (v12)', ('v12', 'cache', 'screen', 'lgb_dec_l2'), {}),
    ('lgb_dec_pw (v12)', ('v12', 'cache', 'screen', 'lgb_dec_pw'), {}),
    ('lgb_size_l2 (v12)', ('v12', 'cache', 'screen', 'lgb_size_l2'), {}),
    ('lgb_size_pw (v12)', ('v12', 'cache', 'screen', 'lgb_size_pw'), {}),
    ('lgb_size_rank (v12)', ('v12', 'cache', 'screen', 'lgb_size_rank'), {}),
    ('X2 lgb_res_soft (v11)', ('v11', 'cache', 'screen', 'x', 'lgb_res_soft', 'lgb_res_soft'), {}),
    ('X5 lgb_rank (v11)', ('v11', 'cache', 'screen', 'x', 'lgb_rank', 'lgb_rank'), {}),
]
CLOUD = {'LSTM-R (v13)': ('v13', 'cache', 'seq_raw_cloud', 'lstm'), 'CNN-R (v13)': ('v13', 'cache', 'seq_raw_cloud', 'tcn')}


def two_signal(x_ic, z_ic, rho):
    """README_v14 theorem 2: gain condition IC_z - rho IC_x, best weight w*, best IC, and w0 = the largest weight that still does
    not lower the IC of (1 - w) x + w z (the mix helps for 0 < w < w0; v14 puts new families at w = 1/3)"""
    a_, b_ = x_ic - rho * z_ic, z_ic - rho * x_ic
    wstar = b_ / (a_ + b_) if (a_ + b_) != 0 else np.nan
    best = math.sqrt(max(x_ic ** 2 + z_ic ** 2 - 2 * rho * x_ic * z_ic, 0) / (1 - rho ** 2)) if abs(rho) < 1 else np.nan
    den = 2 * x_ic ** 2 * (1 - rho) - (z_ic - x_ic) ** 2
    if b_ <= 0:
        w0 = 0.0                                   # no positive weight helps
    elif den <= 0:
        w0 = 1.0                                   # then IC_z >= IC_x: every weight in (0, 1] helps
    else:
        w0 = min(2 * x_ic * b_ / den, 1.0)
    return b_, wstar, best, w0


def corr_matrix(series, idx, mask_days=None):
    """mean over days of the within-day Spearman correlation matrix of the given series (dict name -> Series on idx)"""
    names = list(series)
    df = pd.DataFrame({k: v.reindex(idx) for k, v in series.items()})
    g = df.groupby(level='date')
    Rk = g.rank()
    Rk = Rk - Rk.groupby(level='date').transform('mean')
    Rk = Rk / np.sqrt((Rk ** 2).groupby(level='date').transform('sum'))
    Rk = Rk.fillna(0.0)
    dates = Rk.index.get_level_values('date')
    starts = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1]
    days = dates[starts]
    U = Rk.values
    out = {}
    for i, k in enumerate(names):
        prod = np.add.reduceat(U[:, i:i + 1] * U, starts, axis=0)          # n_days x n
        out[k] = pd.DataFrame(prod, index=days, columns=names)
    return out, days


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='v14')
    a = ap.parse_args()
    t0 = time.time()
    print(f'v14_eval.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  root: {ROOT}')
    sp = os.path.join(V14, 'cache', 'signals.pkl')
    if not os.path.exists(sp):
        print('cache\\signals.pkl is missing: run python v14_signals.py first'); return
    S = pd.read_pickle(sp); CAL = pd.read_pickle(os.path.join(V14, 'cache', 'calendar.pkl'))
    d = L.Data()
    idx = d.idx
    sig = S.reindex(idx)
    print(f'  data rebuilt in {time.time() - t0:.0f}s: {len(idx):,} stock-days (100 stocks), {len(d.INFO):,} Top50 stock-days with a label')

    # ------------------------------------------------------------ part 0
    pe = d.view([d.LGB, d.x1])                                              # v11 stock view (centred)
    D = {'v9': d.run(d.view([d.LGB]), False), 'v11': d.run(pe, True), 'W0': d.run(pe, True, w_t=0.0, lam=0.5)}
    print('\nPART 0 -- rebuild check (val / hold / post / all, bp/day)')
    ok = True
    for k, ref in PUBLISHED.items():
        got = tuple(round(per(D[k].AR)[p], 2) for p in PER)
        same = all(abs(x - y) <= 0.005 for x, y in zip(got, ref)); ok &= same
        print(f'  {k:4s} rebuilt {" / ".join(f"{x:.2f}" for x in got)}   published {" / ".join(f"{x:.2f}" for x in ref)}   '
              + ('same' if same else 'DIFFERENT'))
    if not ok:
        print('  WARNING: the rebuild differs from the published numbers: find the cause before deciding')

    # ------------------------------------------------------------ signals and placements
    q20 = CAL.q_last20; q3 = CAL.q_first3
    Z = {'E2a': zday(sig.al_k5), 'E2b': zday(sig.rr_h5), 'E4': zday(sig.trust60), 'E1': zday(-sig.gap)}
    groups = pd.get_dummies(d.group.reindex(idx).fillna(-1).astype(int), prefix='g').astype(float)
    betas = sig[[f'beta{j}' for j in range(1, 6)]]
    top = pd.Series(True, index=d.INFO.index)
    views = {
        'E2a': with_families(pe, [(Z['E2a'], None)]),
        'E2b': with_families(pe, [(Z['E2b'], None)]),
        'E3a': neutralize_top50(pe, top, groups),
        'E3b': neutralize_top50(pe, top, betas),
        'E4a': with_families(pe, [(Z['E4'], q20)]),
        'E4b': with_families(pe, [(-Z['E4'], q3)]),
    }
    # E5: 1/N over 14 families
    fam = {'stock MLPs (4)': pd.concat(d.mlp, axis=1).mean(axis=1), 'LightGBM part (lgb_res + X1)': pd.concat([d.LGB, d.x1], axis=1).mean(axis=1)}
    missing = []
    for name, parts, kw in E5_FAMILIES:
        s = L.load_member(os.path.join(ROOT, *parts), **kw)
        if s is None and name in CLOUD:
            s = L.load_member(os.path.join(ROOT, *CLOUD[name]), **kw)
        if s is None:
            missing.append(name); continue
        s = s.reindex(idx)
        nmiss = int(s.isna().sum())
        if nmiss:
            s = s.fillna(s.groupby(level='date').transform('mean'))
            print(f'  {name}: {nmiss} of {len(idx):,} rows missing (filled with the day mean)')
        fam[name] = s
    have_e5 = not missing
    if have_e5:
        zbar = pd.concat([zday(v, idx) for v in fam.values()], axis=1).mean(axis=1)
        views['E5'] = spread_match(zbar, pe)
    else:
        print('  E5 not judged: missing ' + ', '.join(missing))
    CANDS = ['E2a', 'E2b', 'E3a', 'E3b', 'E4a', 'E4b'] + (['E5'] if have_e5 else [])
    LABEL = {'E2a': 'E2a Avellaneda-Lee s-score (k 5)', 'E2b': 'E2b residual reversal (5 days)', 'E3a': 'E3a industry-neutral',
             'E3b': 'E3b factor-neutral (5 PCA betas)', 'E4a': 'E4a quarter-end trust buying (last 20 days)',
             'E4b': 'E4b quarter-start reversal (first 3 days)', 'E5': 'E5 1/N of 14 stock-view families'}
    for k in CANDS:
        L.run_both(d, views[k], k, D)
    print(f'  candidate runs done ({time.time() - t0:.0f}s)')

    # ------------------------------------------------------------ part 1
    print('\nPART 1 -- signals (report only): own daily rank IC with the excess return (100 stocks), val / hold / post / all')
    ICv11 = ic_daily(pe, d.resid)
    rows = [{'signal': 'v11 stock view p_e', **per(ICv11)}]
    sigs = {'E1 -gap (open of D!)': -sig.gap, 'E2a AL s-score k5': sig.al_k5, 'E2a k1': sig.al_k1, 'E2a k3': sig.al_k3, 'E2a k10': sig.al_k10,
            'E2b resid reversal 5d': sig.rr_h5, 'E2b 1d': sig.rr_h1, 'E2b 10d': sig.rr_h10, 'E2b 20d': sig.rr_h20, 'E4 trust60 (all days)': sig.trust60}
    ICs = {k: ic_daily(v, d.resid) for k, v in sigs.items()}
    for k, v in ICs.items():
        rows.append({'signal': k, **per(v)})
    days_q20 = set(CAL.index[CAL.q_last20]); days_q3 = set(CAL.index[CAL.q_first3])
    for lab, dd in (('E4 trust60, last 20 days of a quarter', days_q20), ('E4 trust60, first 3 days of a quarter', days_q3)):
        x = ICs['E4 trust60 (all days)']
        rows.append({'signal': lab, **per(x[x.index.isin(dd)])})
    x = ICs['E4 trust60 (all days)']
    rows.append({'signal': 'E4 trust60, other days', **per(x[~x.index.isin(days_q20 | days_q3)])})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    for lab, dd in (('last 20 days', days_q20), ('first 3 days', days_q3)):
        inn = x[x.index.isin(dd)]; out_ = x[~x.index.isin(dd)]
        print(f'  E4 trust60 IC, {lab} minus other days: {inn.mean() - out_.mean():+.4f} (t {(inn.mean() - out_.mean()) / math.sqrt(inn.var() / len(inn) + out_.var() / len(out_)):.2f}, '
              f'{len(inn)} vs {len(out_)} days)')
    print('\n  two-signal diagnostics (x = v11 stock view, z = signal; within-day rank correlation rho; README_v14 3.1):')
    print('    adding a little of z raises the IC iff IC_z > rho * IC_x;  w* = best weight of z, best = best IC of a mix (now = IC_x);')
    print('    w0 = the largest weight that does not lower the IC (v14 places a new family at w = 1/3)')
    diag = {}
    for k in ['E1 -gap (open of D!)', 'E2a AL s-score k5', 'E2b resid reversal 5d', 'E4 trust60 (all days)']:
        rho_d = ic_daily(sigs[k], pe)
        parts_ = []
        for p in ('val', 'hold', 'post', 'all'):
            ix, iz, rho = per(ICv11)[p], per(ICs[k])[p], float(rho_d[w(rho_d.index, p)].mean())
            g_, ws, best, w0 = two_signal(ix, iz, rho)
            diag[(k, p)] = dict(IC_x=ix, IC_z=iz, rho=rho, gain=g_, w_star=ws, best=best, w0=w0)
            parts_.append(f'{p}: rho {rho:+.3f} gain {g_:+.4f} w* {ws:.2f} w0 {w0:.2f} best {best:.4f} (now {ix:.4f})')
        print(f'    {k[:24]:24s} ' + '  '.join(parts_))

    if have_e5:
        print('\n  E5 families: own rank IC (val / hold / post / all) and mean within-day rank correlation with the v11 view:')
        rowsf = []
        for k, v in fam.items():
            r_ = per(ic_daily(v, d.resid)); rho_ = float(ic_daily(v, pe).mean())
            rowsf.append({'family': k, **r_, 'rho_v11': rho_})
        print(pd.DataFrame(rowsf).round(4).to_string(index=False))
        allser = dict(fam); allser['__resid'] = d.resid
        CM, cdays = corr_matrix(allser, idx)
        names = list(fam)
        print('\n  E5 n-signal bound (README_v14 theorem 4), on within-day ranks: IC of a weighted mix <= sqrt(c\' S^-1 c)')
        e5 = {}
        mval = w(cdays, 'val')
        for p in ('val', 'hold', 'post', 'all'):
            mp = w(cdays, p)
            Sg = np.array([[CM[i][j][mp].mean() for j in names] for i in names])
            cv = np.array([CM[i]['__resid'][mp].mean() for i in names])
            inv = np.linalg.pinv(Sg)
            bound = float(np.sqrt(max(cv @ inv @ cv, 0)))
            ones = np.ones(len(names))
            eq = float(ones @ cv / np.sqrt(ones @ Sg @ ones))
            best1 = float(cv.max())
            wv = inv @ cv
            e5[p] = dict(bound=bound, one_over_n=eq, best_single=best1, w=(wv / np.abs(wv).sum()).tolist())
            print(f'    {p:4s}: best single family {best1:.4f}, 1/N mix {eq:.4f}, best linear mix (in-sample, this period) {bound:.4f}')
        Sv = np.array([[CM[i][j][mval].mean() for j in names] for i in names]); cvv = np.array([CM[i]['__resid'][mval].mean() for i in names])
        wv = np.linalg.pinv(Sv) @ cvv
        zz = {k: zday(v, idx) for k, v in fam.items()}
        mix_val = sum(wv[i] * zz[k] for i, k in enumerate(names))
        ic_mix = per(ic_daily(mix_val, d.resid)); ic_eq = per(ic_daily(zbar, d.resid))
        print('    weights of the val-best mix (sum |w| = 1): ' + ', '.join(f'{k.split(" (")[0]} {x:+.2f}' for k, x in zip(names, wv / np.abs(wv).sum())))
        print('    realised rank IC (val / hold / post / all): val-best weights ' + ' / '.join(f'{ic_mix[p]:.4f}' for p in PER)
              + ';  1/N ' + ' / '.join(f'{ic_eq[p]:.4f}' for p in PER) + ';  v11 view ' + ' / '.join(f'{per(ICv11)[p]:.4f}' for p in PER))
        e5['realised'] = {'val_best_weights': ic_mix, 'one_over_n': ic_eq, 'v11': per(ICv11)}
        views['E5 val-best weights (report)'] = spread_match(mix_val, pe)
        L.run_both(d, views['E5 val-best weights (report)'], 'E5 val-best weights (report)', D)
    else:
        e5 = {}

    # ------------------------------------------------------------ part 2
    print('\nPART 2 -- the rules (PREREG_v14.md section 3), v11 + candidate - v11, bp/day (NW t)')
    res, sel = {}, []
    for k in CANDS:
        r = diff(D, f'v11 + {k}', 'v11'); res[k] = r
        s_ = r['val'][0] > 0
        if s_:
            sel.append(k)
        print(f'  {LABEL[k][:44]:44s}: {fmt(r)}   -> ' + ('val > 0: selected' if s_ else 'not selected'))
    passed = False; rsel = None
    if sel:
        base, bw = (views['E5'], 14) if 'E5' in sel else (pe, 2)
        fams = []
        if 'E2a' in sel: fams.append((Z['E2a'], None))
        if 'E2b' in sel: fams.append((Z['E2b'], None))
        if 'E4a' in sel: fams.append((Z['E4'], q20))
        if 'E4b' in sel: fams.append((-Z['E4'], q3))
        comb = with_families(base, fams, base_w=bw, ref=pe) if fams else base
        des = []
        if 'E3a' in sel: des.append(groups)
        if 'E3b' in sel: des.append(betas)
        if des:
            comb = neutralize_top50(comb, top, pd.concat(des, axis=1))
        views['selected together'] = comb
        L.run_both(d, comb, 'selected together', D)
        rsel = diff(D, 'v11 + selected together', 'v11')
        passed = rsel['hold'][0] > 0 and rsel['post'][0] > 0
        print(f'\n  selected: {", ".join(sel)}')
        print(f'  together  v11 + selected - v11: {fmt(rsel)}')
        print('  ' + ('hold and post both > 0 -> PASSED (research result only; the daily programs are not changed: PREREG 3.3)' if passed
                      else 'hold or post <= 0 -> NOT PASSED: v9, v11 and W0 stay as they are'))
    else:
        print('\n  nothing selected -> NOT PASSED: v9, v11 and W0 stay as they are')

    # ------------------------------------------------------------ part 3
    print('\nPART 3 -- report only, each minus its base (bp/day, NW t)')
    print('  W0 (w_T 0, lambda 0.5; the file uploaded now):')
    for k in CANDS + (['selected together'] if sel else []):
        print(f'    W0 + {k[:40]:40s} - W0 : {fmt(diff(D, f"W0 + {k}", "W0"))}')
    if have_e5:
        print(f'    v11 + E5 val-best weights - v11 : {fmt(diff(D, "v11 + E5 val-best weights (report)", "v11"))}')
        print(f'    W0  + E5 val-best weights - W0  : {fmt(diff(D, "W0 + E5 val-best weights (report)", "W0"))}')

    print('\n  the v13 placement (candidate scaled to lgb_res, then averaged as a new family; p_e shrinks, so w_T weighs more):')
    scale = lambda y: d.LGB.groupby(level='date').transform('mean') + c(y) * sdd(c(d.LGB)) / sdd(c(y))
    def v13_view(z, active=None):
        v = d.view([d.LGB, d.x1], extras=[scale(z)])
        if active is None:
            return v
        on = active.reindex(v.index.get_level_values('date'), fill_value=False).values.astype(bool)
        return pd.Series(np.where(on, v.values, pe.values), index=v.index)
    for k, z_, act in (('E2a', Z['E2a'], None), ('E2b', Z['E2b'], None), ('E4a', Z['E4'], q20), ('E4b', -Z['E4'], q3)):
        tag = f'{k} v13 placement'
        L.run_both(d, v13_view(z_, act), tag, D)
        print(f'    v11 + {tag:22s} - v11: {fmt(diff(D, f"v11 + {tag}", "v11"))}')
        print(f'    W0  + {tag:22s} - W0 : {fmt(diff(D, f"W0 + {tag}", "W0"))}')

    print('\n  variants (spread-matched new family, v11 + variant - v11):')
    VAR = [('E2a k 1 (market only)', zday(sig.al_k1), None), ('E2a k 3', zday(sig.al_k3), None), ('E2a k 10', zday(sig.al_k10), None),
           ('E2b 1 day', zday(sig.rr_h1), None), ('E2b 10 days', zday(sig.rr_h10), None), ('E2b 20 days', zday(sig.rr_h20), None),
           ('E4a last 5 days', Z['E4'], CAL.q_last5), ('E4a last 10 days', Z['E4'], CAL.q_last10), ('E4a last 40 days', Z['E4'], CAL.q_last40),
           ('E4 trust60 every day (placebo)', Z['E4'], None)]
    for lab, z_, act in VAR:
        L.run_both(d, with_families(pe, [(z_, act)]), lab, D)
        print(f'    {lab:32s}: v11 {fmt(diff(D, f"v11 + {lab}", "v11"))}')
        print(f'    {"":32s}  W0  {fmt(diff(D, f"W0 + {lab}", "W0"))}')
    for lab, des in (('E3a + E3b together', pd.concat([groups, betas], axis=1)),):
        L.run_both(d, neutralize_top50(pe, top, des), lab, D)
        print(f'    {lab:32s}: v11 {fmt(diff(D, f"v11 + {lab}", "v11"))}')
        print(f'    {"":32s}  W0  {fmt(diff(D, f"W0 + {lab}", "W0"))}')

    # E1: the real open of day D (upper bound of pre-open trial-matching prices)
    print('\n  E1 -- if the opening price of day D were known before deciding (it is not: static forbids it; live needs the 08:59 trial price)')
    print('    rules without a model, Top50 bp/day (val / hold / post / all):')
    I = d.INFO.join(sig.gap.rename('gap'), how='left')
    g_ = I.gap.values; r_ = I.target_r.values
    dates_I = I.index.get_level_values('date')
    gm = I.gap.groupby(level='date').transform('mean').values
    rules = {'perfect foresight (sign of the return)': np.sign(r_), 'each stock: opposite of its own gap': np.where(g_ > 0, -1.0, 1.0),
             'relative: opposite of (gap - day mean)': np.where(g_ - gm > 0, -1.0, 1.0), 'market: opposite of the mean gap': np.where(gm > 0, -1.0, 1.0)}
    e1rules = {}
    for lab, s_ in rules.items():
        ar = pd.Series(s_ * r_, index=I.index).groupby(level='date').mean() * 1e4
        e1rules[lab] = per(ar)
        print(f'      {lab:42s} ' + ' / '.join(f'{e1rules[lab][p]:7.2f}' for p in PER))
    L.run_both(d, views.setdefault('E1a', with_families(pe, [(Z['E1'], None)])), 'E1a gap family', D)
    print(f'    E1a gap as a stock-view family : v11 {fmt(diff(D, "v11 + E1a gap family", "v11"))}')
    print(f'    {"":31s}  W0  {fmt(diff(D, "W0 + E1a gap family", "W0"))}')
    rng = np.random.default_rng(1)
    noise = {}
    for sgm in (0.001, 0.002, 0.003, 0.005):
        acc = {'v11': [], 'W0': []}
        for seed in range(5):
            gn = sig.gap + rng.normal(0.0, sgm, len(sig))
            v = with_families(pe, [(zday(-gn), None)])
            for base, wt, lam in (('v11', 0.5, 0.75), ('W0', 0.0, 0.5)):
                R_ = d.run(v, True, w_t=wt, lam=lam)
                acc[base].append(R_.AR - D[base].AR)
        for base in acc:
            dd = pd.concat(acc[base], axis=1).mean(axis=1)
            noise[(sgm, base)] = {p: (float(dd[w(dd.index, p)].mean()), float(nw_t(dd[w(dd.index, p)].values))) for p in PER}
            print(f'    E1b open known within N(0, {sgm * 100:.1f}%^2), 5 seeds: {base:3s} {fmt(noise[(sgm, base)])}')
    mg = sig.gap.groupby(level='date').mean()
    zg = (mg / mg.rolling(250, min_periods=60).std().shift(1)).clip(-3, 3)
    zn_keep = d.z_night
    d.z_night = zg
    D['v11 + E1c market gap'] = d.run(pe, True)
    D['v11 + E1d both'] = d.run(views['E1a'], True)
    d.z_night = zn_keep
    print(f'    E1c night reversal -> real mean gap (v11 only; W0 has no market view): {fmt(diff(D, "v11 + E1c market gap", "v11"))}')
    print(f'    E1d E1a + E1c                                                         : {fmt(diff(D, "v11 + E1d both", "v11"))}')

    print('\n  by year, v11 + candidate - v11 (bp/day, NW t):')
    for k in CANDS + (['selected together'] if sel else []):
        print(f'    {k[:18]:18s} ' + fmt(diff(D, f'v11 + {k}', 'v11', YEARS)))
    print('\n  2026-10-01 .. 10-22 (15 days, information only):')
    for k in CANDS:
        print(f'    {k:4s} v11 {fmt(diff(D, f"v11 + {k}", "v11", OCT))}   W0 {fmt(diff(D, f"W0 + {k}", "W0", OCT))}')
    print('\n  power of the rules (normal approximation, daily difference sd and days of each period, v11 + candidate - v11):')
    pw = {}
    for k in CANDS:
        sds, pr = power(D, f'v11 + {k}', 'v11')
        pw[k] = {'sd': sds, 'p_pass': pr}
        print(f'    {k:4s}: daily sd ' + ', '.join(f'{p} {sds[p][0]:.1f} bp ({sds[p][1]} days)' for p in sds)
              + ';  P(pass | true effect) ' + ', '.join(f'{dl:+.1f} -> {pv:.2f}' for dl, pv in pr.items()))

    rows = []
    for k, Dk in D.items():
        rr = per(Dk.AR); r19 = Dk.AR.rolling(19).mean().dropna()
        rows.append({'config': k, **rr, 'sd': Dk.AR.std(), 'timing': Dk.timing.mean(), 'xsec': Dk.xsec.mean(), 'share1': (Dk.sbar.mean() + 1) / 2,
                     'P19>0': (r19 > 0).mean()})
    print('\nTop50 bp/day (walk-forward 2022-01 .. 2026-10)')
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    out = os.path.join(V14, 'results'); os.makedirs(out, exist_ok=True)
    pd.DataFrame({k: v.AR for k, v in D.items()}).to_csv(os.path.join(out, f'{a.tag}_daily.csv'), float_format='%.4f')
    json.dump({'rebuild_ok': bool(ok), 'candidates': CANDS, 'selected': sel, 'passed': bool(passed), 'each_minus_v11': res,
               'selected_minus_v11': rsel, 'power': {k: {'p_pass': {str(a_): b_ for a_, b_ in v['p_pass'].items()}} for k, v in pw.items()},
               'diagnostics': {f'{k[0]} | {k[1]}': v for k, v in diag.items()}, 'e5': e5, 'e1_rules': e1rules,
               'e1_noise': {f'{k[0]} {k[1]}': v for k, v in noise.items()}, 'e5_missing': missing},
              open(os.path.join(out, f'{a.tag}_selection.json'), 'w'), indent=1)
    print(f'\nsaved results\\{a.tag}_daily.csv, results\\{a.tag}_selection.json  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()

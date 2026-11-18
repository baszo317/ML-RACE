r"""
p2_eval.py -- race_mkt part 2 (part2\PREREG_mkt_part2.md): CF.LU (C + 37 futures features) vs C.LU on D, decision SD.
  python p2_eval.py      -> part2\results\p2_results.txt / .json / p2_daily.csv
"""
import os, sys, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); MKT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(MKT, 'research'))
import mktlib as K                                                    # noqa: E402
L = K.L
UNITS = os.path.join(MKT, 'cache', 'units')
RES2 = os.path.join(HERE, 'results')


def load_cf():
    parts, metas = [], []
    for fa, _ in L.D_FOLDS:
        ss = []
        for s in (1, 2, 3):
            f = os.path.join(UNITS, 'CF.LU', f'{fa}_seed{s}.npz')
            if not os.path.exists(f):
                raise SystemExit(f'missing {f}: run p2_train.py')
            p, m = L.load_npz(f); ss.append(p); metas.append(m)
        parts.append(pd.concat(ss, axis=1).mean(axis=1))
    return pd.concat(parts).sort_index(), metas


def main():
    X = L.load_info()
    INFO = L.info_frame(X[(X.date >= K.D_A) & (X.date < K.D_B)])
    base = L.load_unit('C.LU', L.D_FOLDS)
    cf, metas = load_cf()
    Db = L.decide_sd(base, INFO); Dc = L.decide_sd(cf, INFO)
    j = Db.index.intersection(Dc.index)
    d = (Dc.AR - Db.AR).reindex(j)
    yrs = {str(y): float(g.mean()) for y, g in d.groupby(pd.DatetimeIndex(d.index).year)}
    t = float(L.nw_t(d.to_numpy()))
    ok = bool(d.mean() > 0 and t >= 2 and sum(v > 0 for v in yrs.values()) >= 2)
    imp = pd.DataFrame([m['importance'] for m in metas]).fillna(0.0)
    share = imp.div(imp.sum(axis=1), axis=0).mean()
    fut_share = float(share[[c for c in share.index if c.startswith('FUT__')]].sum())
    top = share.sort_values(ascending=False)
    top_fut = top[[c for c in top.index if c.startswith('FUT__')]].head(10)
    corr = float(np.corrcoef(base.reindex(cf.index).dropna().groupby(level='date').rank().to_numpy(),
                             cf.reindex(base.reindex(cf.index).dropna().index).groupby(level='date').rank().to_numpy())[0, 1])
    L_ = [f'race_mkt part 2, D (2022-2024), decision SD (Top25 + ex-dividend), {len(j)} days',
          f'  C.LU  (race_std cache, 169 features)   AR {Db.AR.mean():6.2f}  NW t {L.nw_t(Db.AR.to_numpy()):5.2f}  IC {Db.ic.mean():.4f}  '
          + ' '.join(f'{y}:{v:.2f}' for y, v in L.by_fold(Db, L.D_FOLDS).items()),
          f'  CF.LU (C + 37 futures features)        AR {Dc.AR.mean():6.2f}  NW t {L.nw_t(Dc.AR.to_numpy()):5.2f}  IC {Dc.ic.mean():.4f}  '
          + ' '.join(f'{y}:{v:.2f}' for y, v in L.by_fold(Dc, L.D_FOLDS).items()),
          f'  CF.LU - C.LU: {d.mean():+.2f} bp/day, NW t {t:.2f}, years ' + ' '.join(f'{y}:{v:+.2f}' for y, v in yrs.items())
          + f'  -> {"PASS" if ok else "not passed"} (mean > 0, t >= 2, >= 2 of 3 years > 0)',
          f'  within-day rank correlation of the two scores: {corr:.3f}; best_iter CF.LU ' + ', '.join(str(m['best_iter']) for m in metas),
          f'  gain importance share of the 37 futures features: {fut_share:.3f} (mean over the 9 models)',
          '  top futures features by gain share: ' + ', '.join(f'{k[5:]} {v:.4f}' for k, v in top_fut.items()),
          '  top 10 overall: ' + ', '.join(f'{k} {v:.3f}' for k, v in top.head(10).items())]
    txt = '\n'.join(L_)
    os.makedirs(RES2, exist_ok=True)
    with open(os.path.join(RES2, 'p2_results.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    pd.DataFrame({'C.LU': Db.AR, 'CF.LU': Dc.AR, 'ic_C': Db.ic, 'ic_CF': Dc.ic}).to_csv(os.path.join(RES2, 'p2_daily.csv'))
    with open(os.path.join(RES2, 'p2_results.json'), 'w', encoding='utf-8') as fh:
        json.dump({'diff': float(d.mean()), 't': t, 'years': yrs, 'pass': ok, 'AR_C': float(Db.AR.mean()),
                   'AR_CF': float(Dc.AR.mean()), 'fut_gain_share': fut_share, 'prereg_sha256': K.sha256(os.path.join(HERE, 'PREREG_mkt_part2.md'))},
                  fh, indent=1)
    print(txt)


if __name__ == '__main__':
    main()

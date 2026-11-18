r"""
mis_ceiling.py -- how much could TWSE MIS 試撮 (indicative opening price, 08:30-09:00) add to the race version at most?
Report only (no decision): MIS has no history, so the real 09:00 open is used as a stand-in for a perfect 08:59 snapshot
(an upper bound), with noisy versions for a realistic snapshot. D only (2022-2024), race version WF3 scores, decision
Top25 + ex-dividend rule (= DL0 of race_mkt).

  python mis_ceiling.py   -> ..\results\src_mis_ceiling.txt
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); MKT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(MKT, 'research'))
import mktlib as K                                                    # noqa: E402
L = K.L


def stock_gaps(T50):
    rows = []
    for sid in sorted(set(T50.index.get_level_values('stock_id'))):
        m = pd.read_csv(os.path.join(K.ROOT, 'data', 'market', f'{sid}.csv'), parse_dates=['date'])
        m = m[(m.date <= K.LAST_LABEL) & (m['etl:adj_open'] > 0) & (m['etl:adj_close'] > 0)].sort_values('date')
        g = m['etl:adj_open'].to_numpy() / m['etl:adj_close'].shift(1).to_numpy() - 1
        rows.append(pd.DataFrame({'date': pd.to_datetime(m.date.values).astype('datetime64[ns]'), 'stock_id': sid, 'g': g}))
    G = pd.concat(rows, ignore_index=True).set_index(['date', 'stock_id']).g
    return G.reindex(T50.index)


def zday(s):
    g = s.groupby(level='date')
    sd = g.transform('std')
    return ((s - g.transform('mean')) / sd.where(sd > 0)).clip(-3, 3).fillna(0.0)


def run_top25(T50, score):
    DF = K.day_frame(T50, score)
    rows = []
    for d, f in DF.items():
        rows.append((d, K.ar_of(K.topk(f, 25), f)))
    return pd.Series(dict(rows)).sort_index()


def main():
    days = K.all_days()
    T50 = K.load_top50()
    T50 = T50[(T50.index.get_level_values('date') >= K.D_A) & (T50.index.get_level_values('date') < K.D_B)]
    sc = K.wf3_scores(days).reindex(T50.index)
    g = stock_gaps(T50)
    r = T50.target_r
    base = run_top25(T50, sc)
    out = [f'MIS 試撮 ceiling on D ({len(base)} days), race version WF3, Top25 + ex-dividend; the real open stands in for 試撮',
           f'  DL0 (no gap)  AR {base.mean():.2f}']
    # descriptive: IC of -gap with the intraday return, correlation with the race score
    X = pd.DataFrame({'r': r, 'g': g, 's': sc}).dropna()
    ic = X.groupby(level='date').apply(lambda x: x['r'].rank().corr((-x['g']).rank())).mean()
    ic_s = X.groupby(level='date').apply(lambda x: x['r'].rank().corr(x['s'].rank())).mean()
    cs = X.groupby(level='date').apply(lambda x: x['s'].rank().corr((-x['g']).rank())).mean()
    out.append(f'  daily Spearman IC: race score {ic_s:.4f}; -gap (reversal) {ic:.4f}; corr(race score, -gap) {cs:.3f}')
    zs = zday(sc); zg = zday(-g)
    rng = np.random.default_rng(1117)
    for err in (0.0, 0.001, 0.003):
        gn = g + (rng.standard_normal(len(g)) * err if err > 0 else 0.0)
        zg = zday(-pd.Series(np.asarray(gn), index=g.index))
        line = f'  snapshot error sd {err * 100:.1f}%:'
        for w in (0.1, 0.2, 0.3, 0.5, 1.0):
            ar = run_top25(T50, zs + w * zg)
            d = (ar - base).dropna()
            yrs = ' / '.join(f'{v:+.2f}' for v in d.groupby(d.index.year).mean())
            line += f'  w {w}: {d.mean():+.2f} (t {K.nw_t(d.to_numpy()):.2f}; {yrs})'
        out.append(line)
    out.append('  (w = weight of the within-day z of -gap next to the z of the race score; in-sample on D, the best w is an upper bound)')
    txt = '\n'.join(out)
    with open(os.path.join(MKT, 'results', 'src_mis_ceiling.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()

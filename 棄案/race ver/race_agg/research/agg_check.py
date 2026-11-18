r"""
agg_check.py -- race_agg independent checks (no rslib decision code):
  1. recompute the AR of SD and of every candidate from results\agg_D_daily.csv with a separate Top25 implementation
     (scores rebuilt by agg_eval.scores) and compare day by day;
  2. timing: every trained unit's training rows end before its fold (meta train_last < fold) and only D folds exist;
  3. the blending rows of A3 / A4 lie before each fold.
  python agg_check.py   -> results\agg_check.txt
"""
import os, sys, json, glob
import numpy as np, pandas as pd
import agglib as A
import agg_eval as E
L = A.L


def ar_independent(score, X):
    """Top50 rows of D, Top25 by score (ties by stock id), ex-dividend -> -1; AR in bp per day"""
    df = X.join(score.rename('sc'), how='left').reset_index()
    df['sc'] = df.sc.fillna(-np.inf)
    df = df.sort_values(['date', 'sc', 'stock_id'], ascending=[True, False, True])
    df['rk'] = df.groupby('date').cumcount()
    df['s'] = np.where(df.rk < 25, 1.0, -1.0)
    ex = np.zeros(len(df), bool)
    for c in L.EX4:
        ex |= df[c].to_numpy() == 1
    df.loc[ex, 's'] = -1.0
    df = df[df.target_r.notna()]
    return (df.s * df.target_r).groupby(df.date).mean() * 1e4


def main():
    out = []
    P = A.load_panel()
    X = L.load_info()
    X = X[X.top50.astype(bool) & (X.date >= A.D_A) & (X.date < A.D_B)].set_index(['date', 'stock_id']).sort_index()
    S, mem, binfo, missing = E.scores(P)
    daily = pd.read_csv(os.path.join(A.RES, 'agg_D_daily.csv'), index_col=0, parse_dates=True)
    worst = 0.0
    for k, s in S.items():
        a = ar_independent(s, X)
        b = daily[k].reindex(a.index)
        dmax = float((a - b).abs().max())
        worst = max(worst, dmax)
        out.append(f'{k:<6} independent mean {a.mean():+.4f}  agg_eval {b.mean():+.4f}  max daily |diff| {dmax:.2e}')
    out.append(f'worst daily difference {worst:.2e}')
    bad = []
    for f in glob.glob(os.path.join(A.UNITS, '*', '*.npz')):
        z = np.load(f, allow_pickle=False); m = json.loads(str(z['meta']))
        fold = os.path.basename(f).split('_seed')[0]
        if fold not in [fa for fa, _ in A.D_FOLDS]:
            bad.append(f'not a D fold: {f}')
        if pd.Timestamp(m['train_last']) >= pd.Timestamp(fold):
            bad.append(f'training reaches the fold: {f}')
        dts = pd.to_datetime(z['date'])
        if m['cand'].startswith('GM.'):
            if dts.max() >= pd.Timestamp(fold):
                bad.append(f'GM rows reach the fold: {f}')
        elif dts.min() < pd.Timestamp(fold) or dts.max() >= pd.Timestamp(A.D_B):
            bad.append(f'prediction rows outside the fold: {f}')
    n_units = len(glob.glob(os.path.join(A.UNITS, '*', '*.npz')))
    out.append(f'{n_units} unit files checked; problems: {bad if bad else "none"}')
    if isinstance(binfo, dict) and 'missing' not in binfo:
        for fa, v in binfo.items():
            out.append(f'blending rows of fold {fa}: {v["days"][0]} .. {v["days"][1]} (before the fold: {pd.Timestamp(v["days"][1]) < pd.Timestamp(fa)})')
    txt = '\n'.join(out)
    with open(os.path.join(A.RES, 'agg_check.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()

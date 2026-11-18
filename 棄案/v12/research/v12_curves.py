r"""
v12_curves.py -- C4 of research\PREREG_v12.md: is X5's validation NDCG flat in the number of rounds?
For each of the 6 folds (seed 1): X5 (lambdarank, decile label, as v11e_train) and lgb_res (binary, weights min(|r|, 5%))
are trained for 1000 rounds on the training period without its last 120 days (no early stopping). Recorded on those 120 days:
X5's NDCG@100 every round (LightGBM's metric, checked against our own per-day NDCG), and the daily rank IC of both models at
rounds 10, 25, 50, 100, 200, 300, 500, 750, 1000.
C4 holds if, in at least 4 folds, max - min of X5's NDCG over rounds 10..1000 < 2 standard errors (sd of the daily NDCG at the
best round / sqrt(120)); does not hold if at least 4 folds are >= 2 standard errors.
run (in v12\research): python v12_curves.py > ..\results\v12_curves.txt
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v12lib as L
import v11e_train as E
V = L.V
CHECK = [10, 25, 50, 100, 200, 300, 500, 750, 1000]

def day_ndcg(p, lab, starts, gains):
    """NDCG@all per day with the given gains (log2 discount), as LightGBM's ndcg@100 for days of <= 100 stocks"""
    out = []
    for k in range(len(starts) - 1):
        a, b = starts[k], starts[k + 1]
        g = gains[lab[a:b]]; o = np.argsort(-p[a:b], kind='stable')
        disc = 1.0 / np.log2(np.arange(2, b - a + 2))
        dcg = (g[o] * disc).sum(); idcg = (np.sort(g)[::-1] * disc).sum()
        out.append(dcg / idcg if idcg > 0 else 1.0)
    return np.array(out)

def day_ic(p, r, starts):
    return np.array([np.corrcoef(pd.Series(p[a:b]).rank().values, pd.Series(r[a:b]).rank().values)[0, 1]
                     for a, b in zip(starts[:-1], starts[1:])])

def main():
    import lightgbm as lgb
    t0 = time.time()
    P = V.panel(log=lambda m: None); feats = list(V.base_features()['lgb_res'])
    rows, summ = [], []
    for fa, _ in V.FOLDS:
        te = pd.Timestamp(fa) - pd.Timedelta(days=1)
        Q = P[(P.date >= V.TRAIN_START) & (P.date <= te) & P.resid_r.notna()].sort_values(['date', 'stock_id'])
        X = Q[feats].to_numpy(np.float64); dates = Q.date.values; r = Q.resid_r.values
        ud = np.sort(np.unique(dates)); cut = ud[-120] if len(ud) > 240 else ud[-1]; tr = dates < cut; va = ~tr
        gs = lambda m: pd.Series(dates[m]).groupby(dates[m]).size().values
        st = L.day_starts(dates[va]); rv = r[va]
        # X5
        yq = E.decile_label(Q)
        ev = {}
        m = lgb.train({**E.RANK_PRM, 'seed': 1}, lgb.Dataset(X[tr], yq[tr], group=gs(tr)), 1000,
                      valid_sets=[lgb.Dataset(X[va], yq[va], group=gs(va))], valid_names=['va'],
                      callbacks=[lgb.record_evaluation(ev)])
        nd = np.array(ev['va']['ndcg@100'])
        best = int(np.argmax(nd)) + 1
        pb = m.predict(X[va], num_iteration=best)
        mine = day_ndcg(pb, yq[va], st, np.arange(10.0))
        se = mine.std(ddof=1) / np.sqrt(len(mine))
        rng = nd[9:].max() - nd[9:].min()
        ic5 = {k: day_ic(m.predict(X[va], num_iteration=k), rv, st).mean() for k in CHECK}
        # lgb_res
        yb = Q.y_resid.values.astype(float); wt = np.minimum(np.abs(r), 0.05); wt = wt / wt.mean()
        evb = {}
        mb = lgb.train({**V.LGB_PRM, 'seed': 1}, lgb.Dataset(X[tr], yb[tr], weight=wt[tr]), 1000,
                       valid_sets=[lgb.Dataset(X[va], yb[va], weight=wt[va])], valid_names=['va'], callbacks=[lgb.record_evaluation(evb)])
        ll = np.array(evb['va']['binary_logloss'])
        icb = {k: day_ic(mb.predict(X[va], num_iteration=k), rv, st).mean() for k in CHECK}
        summ.append({'fold': fa, 'X5 best round': best, 'NDCG best': nd[best - 1], 'our NDCG at best': mine.mean(),
                     'NDCG range r10-1000': rng, '2 x SE': 2 * se, 'flat (range < 2SE)': bool(rng < 2 * se),
                     'NDCG r10 / r100 / r300 / r1000': ' / '.join(f'{nd[k - 1]:.4f}' for k in (10, 100, 300, 1000)),
                     'lgb_res best logloss round': int(np.argmin(ll)) + 1})
        for k in CHECK:
            rows.append({'fold': fa, 'round': k, 'X5 val IC': ic5[k], 'lgb_res val IC': icb[k], 'X5 NDCG': nd[k - 1]})
        print(f'{fa}: done ({time.time() - t0:.0f}s)', flush=True)
    S = pd.DataFrame(summ)
    print('\nC4: X5 validation NDCG@100 over rounds (last 120 training days, seed 1, 1000 rounds, no early stopping)')
    print(S.round(4).to_string(index=False))
    n_flat = int(S['flat (range < 2SE)'].sum())
    verdict = 'holds' if n_flat >= 4 else ('does not hold' if 6 - n_flat >= 4 else 'uncertain')
    print(f'\nfolds with range < 2 SE: {n_flat} of 6 -> C4 {verdict}')
    R = pd.DataFrame(rows)
    print('\nvalidation rank IC by round (both models) and X5\'s NDCG')
    for col in ('X5 val IC', 'lgb_res val IC', 'X5 NDCG'):
        print(f'\n{col}'); print(R.pivot(index='fold', columns='round', values=col).round(4).to_string())
    R.to_csv(os.path.join(L.V12, 'results', 'v12_curves.csv'), index=False)
    print(f'\ndone in {time.time() - t0:.0f}s')

if __name__ == '__main__':
    main()

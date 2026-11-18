r"""
research\screen_mlp.py -- reproduces the screening numbers quoted in README_v2.md (MLP only, quarterly retraining,
seeds 1-5, test folds 2022Q1 .. 2026Q2 + 2026-06..08 + 2026-09..last day). About 40 minutes on an 8-thread CPU.
  python research\screen_mlp.py            (from the v2 folder)
Output: research\screen_results.txt, research\screen_*.pkl (predictions; rerunning skips finished configurations)
"""
import os, sys, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.dirname(HERE); sys.path.insert(0, V2)
import ens_lib as E

CUT = pd.Timestamp('2025-01-01'); POST = '2026-06-01'
CONFIGS = {'q_base': ('B', 'raw', None), 'q_w2': ('B', 'raw', 2), 'q_w4': ('B', 'raw', 4), 'q_xr': ('B+XR', 'raw', None),
           'q_cal': ('B+CAL', 'raw', None), 'q_resid': ('B', 'resid', None), 'q_resid_xr': ('B+XR', 'resid', None)}

def main():
    import torch
    from tfmodel import predict
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    P = E.build_panel(E.DATA); last = P[P.target_r.notna()].date.max(); fs = E.feature_sets()
    q = list(pd.date_range('2022-01-01', '2026-04-01', freq='QS')) + [pd.Timestamp(POST)]
    folds = list(zip(q[:-1], q[1:])) + [(pd.Timestamp(POST), pd.Timestamp('2026-09-01')), (pd.Timestamp('2026-09-01'), last + pd.Timedelta(days=1))]
    for name, (feats, target, window) in CONFIGS.items():
        f = os.path.join(HERE, f'screen_{name}.pkl')
        if os.path.exists(f):
            continue
        m = E.member(name, 'mlp', window=window, feats=feats, target=target); out = []; t0 = time.time()
        for a, b in folds:
            te = a - pd.Timedelta(days=1); ps = []
            for s in range(1, 6):
                bd = E.train_member(P, m, s, te, fs[feats], start=E.window_start(te, window))
                ps.append(predict(bd, P, lambda d: (d.date >= a) & (d.date < b) & d.target_r.notna()).set_index(['date', 'stock_id']).p)
            out.append(pd.concat(ps, axis=1).mean(axis=1).rename('p'))
        pd.concat(out).reset_index().to_pickle(f); print(f'{name}: {time.time() - t0:.0f}s', flush=True)
    report(P)

def report(P):
    K = P[['date', 'stock_id', 'top50', 'target_r', 'ex_today']]
    ld = lambda n: pd.read_pickle(os.path.join(HERE, f'screen_{n}.pkl')).set_index(['date', 'stock_id']).p
    def stats(S, label):
        r = {'config': label, 'val': S[S.index < CUT].mean(), 'hold': S[(S.index >= CUT) & (S.index < POST)].mean(), 'post': S[S.index >= POST].mean(), 'all': S.mean(), 'sd': S.std()}
        for y in range(2022, 2027):
            r[str(y)] = S[(S.index.year == y) & (S.index < POST)].mean()
        return r
    def score(out, exdiv):
        x = out.reset_index().merge(K, on=['date', 'stock_id']); x = x[x.target_r.notna()]
        x['pred'] = (x.score > 0).astype(int)
        if exdiv:
            x['pred'] = E.apply_exdiv(x.pred, x.ex_today)
        return E.daily_score(x)
    rows = []
    for n in CONFIGS:
        mm = [E.member(n, 'mlp')]
        rows.append(stats(score(E.combine({n: ld(n)}, mm), False), f'{n} alone, v1 rule'))
    rows.append(stats(score(E.combine({'q_base': ld('q_base')}, [E.member('q_base', 'mlp')]), True), 'q_base, v1 rule + exdiv'))
    for combo in (['q_base', 'q_xr'], ['q_base', 'q_w4'], ['q_base', 'q_w2', 'q_w4']):
        probs = {n: ld(n) for n in combo}; mm = [E.member(n, 'mlp') for n in combo]
        rows.append(stats(score(E.combine(probs, mm), False), '+'.join(combo) + ', v1 rule'))
    members = [E.member('q_base', 'mlp'), E.member('q_xr', 'mlp'), E.member('q_resid', 'mlp', target='resid'), E.member('q_resid_xr', 'mlp', target='resid')]
    probs = {m['name']: ld(m['name']) for m in members}
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        rows.append(stats(score(E.combine(probs, members, w_t=w), True), f'v2 rule w_T={w} + exdiv'))
    T = pd.DataFrame(rows); pd.set_option('display.width', 250)
    txt = T.round(2).to_string(index=False); print(txt)
    open(os.path.join(HERE, 'screen_results.txt'), 'w', encoding='utf-8').write(txt + '\n')

if __name__ == '__main__':
    main()

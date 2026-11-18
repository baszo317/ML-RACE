"""v9_eval: the evaluation of PREREG_v9.md. v9 (all 11 changes) vs the v4 rule with the current members (= v5 live), plus
ablations that need no retraining. The 11 changes were picked on val only, so val is biased upward; hold and post are the check.
uses ../cache/wf9/ if present (and writes ../cache/wf9_preds.csv.gz), otherwise ../cache/wf9_preds.csv.gz
run: python v9_eval.py > ../results/v9_eval.txt"""
import os, sys, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v9lib as V
L = V.L
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)
CSV = os.path.join(V.CACHE, 'wf9_preds.csv.gz')
NAMES = list(V.members())
CGRID = np.round(np.arange(-0.3, 0.3 + 1e-9, 0.0025), 6)

def load(name):
    parts = {}
    for f in sorted(glob.glob(os.path.join(V.CACHE, 'wf9', name, '*.pkl'))):
        x = pd.read_pickle(f); x['date'] = pd.to_datetime(x.date).astype('datetime64[ns]'); x['stock_id'] = x.stock_id.astype(str)
        parts.setdefault(os.path.basename(f).split('_seed')[0], []).append(x.set_index(['date', 'stock_id']).p)
    assert len(parts) == 6 and all(len(v) == 10 for v in parts.values()), (name, {k: len(v) for k, v in parts.items()})
    return pd.concat([pd.concat(v, axis=1).mean(axis=1) for v in parts.values()]).sort_index()

def read_csv(f):
    x = pd.read_csv(f, dtype={'stock_id': str}, parse_dates=['date']); x['date'] = x.date.astype('datetime64[ns]')
    return x.set_index(['date', 'stock_id']).sort_index()

if glob.glob(os.path.join(V.CACHE, 'wf9', '*', '*.pkl')):
    W9 = pd.DataFrame({n: load(n) for n in NAMES}); W9.reset_index().to_csv(CSV, index=False, float_format='%.7f')
else:
    W9 = read_csv(CSV)
M = L.members()
LGB = read_csv(os.path.join(L.CACHE, 'tab_preds.csv.gz')).lgb_resid
idx = M.index.intersection(W9.dropna().index).intersection(LGB.index)
print(f'rows: current members {len(M)}, v9 members {len(W9.dropna())}, common {len(idx)}')
M, W9, LGB = M.loc[idx], W9.loc[idx], LGB.loc[idx]

P = V.core_panel().set_index(['date', 'stock_id'])
B = pd.read_pickle(os.path.join(L.CACHE, 'beta_panel.pkl')).set_index(['date', 'stock_id']).beta
INFO = pd.DataFrame({'top50': P.top50, 'target_r': P.target_r, 'ex_today': P.ex_today, 'ex_next1': P.ex_next1, 'ex_prev1': P.ex_prev1,
                     'ex_prev2': P.ex_prev2}).reindex(idx)
INFO['beta'] = B.reindex(idx).clip(0.2, 2.5)
INFO['beta'] = INFO.beta.fillna(INFO.groupby(level='date').beta.transform('median')).fillna(1.0)
INFO = INFO[INFO.top50.astype(bool) & INFO.target_r.notna()]

def centre(p):
    return p - p.groupby(level='date').transform('mean')
def views(raw_parts, res_parts):
    """family means: raw_parts / res_parts = list of lists (families) of member Series"""
    p_raw = pd.concat([pd.concat(f, axis=1).mean(axis=1) for f in raw_parts], axis=1).mean(axis=1)
    p_e = pd.concat([pd.concat(f, axis=1).mean(axis=1) for f in res_parts], axis=1).mean(axis=1)
    return 0.5 * (p_raw.groupby(level='date').transform('mean') - 0.5) + centre(p_e)

def decide(score, ex_cols=('ex_today',), lam=1.0):
    """daily AR/timing/xsec on the Top50: 0 on the ex-dividend flags in ex_cols, then (lam < 1) beta-proportional hedge"""
    X = INFO.join(score.rename('score'), how='inner')
    rows = []
    for d, g in X.groupby(level='date'):
        sc, b, r = g.score.values, g.beta.values, g.target_r.values
        ex = np.zeros(len(g), bool)
        for c in ex_cols:
            ex |= g[c].values == 1
        s = np.where(ex, -1, np.where(sc > 0, 1, -1))
        if lam < 1.0:
            sb, bb = s.mean(), b.mean(); B0 = ((s - sb) * (b - bb)).mean(); Et = sb * bb + lam * B0
            S = (sc[None, :] + CGRID[:, None] * b[None, :] > 0) & ~ex[None, :]
            E = ((2 * S - 1) @ b) / len(b)
            dd = np.abs(E - Et); cand = np.flatnonzero(dd <= dd.min() + 1e-12); j = cand[np.argmin(np.abs(CGRID[cand]))]
            s = np.where(S[j], 1, -1)
        m = r.mean()
        rows.append((d, (s * r).mean() * 1e4, s.mean() * m * 1e4, m * 1e4))
    D = pd.DataFrame(rows, columns=['date', 'AR', 'timing', 'm']).set_index('date'); D['xsec'] = D.AR - D.timing
    return D

EX4 = ('ex_today', 'ex_next1', 'ex_prev1', 'ex_prev2')
mlp_raw = [M.mlp_raw_B, M.mlp_raw_xr]; tf = [M.tf_raw_B]; mlp_res = [M.mlp_res_B, M.mlp_res_xr]
raw9 = [W9.mlp_raw_B9, W9.mlp_raw_xr9, W9.mlp_raw_w4_9]; res9 = [W9.mlp_res_B9, W9.mlp_res_xr9, W9.mlp_res_BN9, W9.mlp_res_xrN9]
s_v4 = views([mlp_raw, tf], [mlp_res])
s_v9 = views([raw9, tf], [res9, [LGB]])
configs = {
    'control: v4 rule, current members (= v5 live)': (s_v4, ('ex_today',), 1.0),
    'v9: all 11 changes': (s_v9, EX4, 0.75),
    'ablation: v9 without the hedge': (s_v9, EX4, 1.0),
    'ablation: v9 without the ex-dividend neighbour rules': (s_v9, ('ex_today',), 0.75),
    'ablation: v9 without LightGBM': (views([raw9, tf], [res9]), EX4, 0.75),
    'ablation: v9 without the 4-year MLP': (views([raw9[:2], tf], [res9, [LGB]]), EX4, 0.75),
    'ablation: v9 without the v3 members': (views([raw9, tf], [res9[:2], [LGB]]), EX4, 0.75),
    'split: v9 market view + current stock view, v4 rules': (views([raw9, tf], [mlp_res]), ('ex_today',), 1.0),
    'split: current market view + v9 stock view, v4 rules': (views([mlp_raw, tf], [res9, [LGB]]), ('ex_today',), 1.0),
    'split: current members + v9 rules (ex neighbours, hedge)': (s_v4, EX4, 0.75),
    'split: v9 members, v4 rules': (s_v9, ('ex_today',), 1.0),
}
R = {k: decide(*v) for k, v in configs.items()}
base = R['control: v4 rule, current members (= v5 live)']
chk = L.daily(s_v4)
print(f'check: control from this script {base.AR.mean():.4f} vs v8 mlib.daily {chk.AR.mean():.4f}')
rows = []
for k, D in R.items():
    pr = L.periods(D.index); r19 = D.AR.rolling(19).mean().dropna(); dd = D.AR - base.AR
    st = (D.index >= '2026-06-01') & (D.index <= '2026-09-30'); oc = D.index >= '2026-10-01'
    r = {'config': k, 'val': D.AR[pr['val']].mean(), 'hold': D.AR[pr['hold']].mean(), 'post': D.AR[pr['post']].mean(), 'all': D.AR.mean(),
         'sd': D.AR.std(), 'timing': D.timing.mean(), 'xsec': D.xsec.mean(), 'P19>0': (r19 > 0).mean(), 'P19>20': (r19 > 20).mean(),
         'static': D.AR[st].mean(), 'Oct': D.AR[oc].mean()}
    for p in ('val', 'hold', 'post'):
        r['d_' + p] = dd[pr[p]].mean(); r['t_' + p] = L.plain_t(dd[pr[p]])
    rows.append(r)
T = pd.DataFrame(rows)
print('\nTop50 bp/day, walk-forward 2022-01..2026-10 (yearly retrain; post = models frozen at 2026-05-29); d_/t_ = paired difference to the control')
print(T[['config', 'val', 'hold', 'post', 'all', 'sd', 'timing', 'xsec', 'P19>0', 'P19>20', 'static', 'Oct']].round(2).to_string(index=False))
print()
print(T[['config', 'd_val', 't_val', 'd_hold', 't_hold', 'd_post', 't_post']].round(2).to_string(index=False))
r = T.iloc[1]
print(f'\npre-registered check (selection used val only): hold diff {r.d_hold:+.2f} (t {r.t_hold:.2f}), post diff {r.d_post:+.2f} (t {r.t_post:.2f}) -> '
      + ('both > 0: the mix generalised to the periods not used for picking' if (r.d_hold > 0 and r.d_post > 0) else 'not both > 0: the mix did not generalise'))
pd.to_pickle(R, os.path.join(V.CACHE, 'v9_daily.pkl'))

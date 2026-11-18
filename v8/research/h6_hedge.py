"""h6: the pre-registered test of the third round (PREREG_m.md): remove the stock view's beta tilt B by hedging the book's
beta exposure back to E* = sbar_v4 * mean(beta), keeping the market view's own net exposure.
  H6a  hedge with the number of 1s: the v4 ranking is unchanged, only k (how many 1s) moves
  H6b  hedge in proportion to beta: score_i + c_t * beta_i (the Lagrange form of "max sum s_i*score_i subject to E = E*")
Also (report only): the frontier E* = sbar_v4*mean(beta) + lambda*B_v4, and the score on big up / big down days.
run: python h6_hedge.py > ../results/h6_hedge.txt"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mlib as L
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)

P = L.panel(); V = L.views(L.members())
Bp = pd.read_pickle(os.path.join(L.CACHE, 'beta_panel.pkl'))[['date', 'stock_id', 'beta']]
F = V[['pbar', 'p_e_c']].reset_index().merge(Bp, on=['date', 'stock_id'], how='left')
F['beta'] = F.beta.clip(0.2, 2.5); F['beta'] = F.beta.fillna(F.groupby('date').beta.transform('median')).fillna(1.0)
F['score'] = 0.5 * (F.pbar - 0.5) + F.p_e_c
F = F.merge(P[['date', 'stock_id', 'top50', 'target_r', 'ex_today']], on=['date', 'stock_id'])
F = F[F.top50 & F.target_r.notna()].sort_values(['date', 'stock_id'])
CGRID = np.round(np.arange(-0.3, 0.3 + 1e-9, 0.0025), 6)
LAMS = [1.0, 0.75, 0.5, 0.25, 0.0]

def books(g):
    sc, b, ex, r = g.score.values, g.beta.values, g.ex_today.values == 1, g.target_r.values
    n = len(sc)
    s0 = np.where(ex, -1, np.where(sc > 0, 1, -1))
    sbar0 = s0.mean(); bbar = b.mean()
    B0 = ((s0 - sbar0) * (b - bbar)).mean()
    out = {('v4', 1.0): s0}
    # H6a: k = number of 1s among the eligible (non ex-dividend) stocks, v4 ranking unchanged
    el = np.where(~ex)[0]; order = el[np.argsort(-sc[el], kind='stable')]
    cum = np.r_[0.0, np.cumsum(b[order])]
    Ek = (-b.sum() + 2 * cum) / n                                    # E(k) for k = 0..len(el)
    k0 = int((sc[el] > 0).sum())
    # H6b: score + c * beta on a grid of c
    S = (sc[None, :] + CGRID[:, None] * b[None, :] > 0) & ~ex[None, :]
    Ec = ((2 * S - 1) @ b) / n
    for lam in LAMS[1:]:
        Et = sbar0 * bbar + lam * B0
        da = np.abs(Ek - Et); ka = np.flatnonzero(da <= da.min() + 1e-12)
        k = ka[np.argmin(np.abs(ka - k0))]
        sa = -np.ones(n, int); sa[order[:k]] = 1
        out[('H6a', lam)] = sa
        dc = np.abs(Ec - Et); ca = np.flatnonzero(dc <= dc.min() + 1e-12)
        j = ca[np.argmin(np.abs(CGRID[ca]))]
        out[('H6b', lam)] = np.where(S[j], 1, -1)
    rows = {}
    m = r.mean()
    for key, s in out.items():
        sb = s.mean(); Bk = ((s - sb) * (b - bbar)).mean(); AR = (s * r).mean()
        rows[key] = (AR * 1e4, sb * m * 1e4, Bk * m * 1e4, Bk, sb, (s * b).mean(), m * 1e4)
    return rows

res = {}
for d, g in F.groupby('date'):
    for key, v in books(g).items():
        res.setdefault(key, []).append((d,) + v)
D = {k: pd.DataFrame(v, columns=['date', 'AR', 'timing', 'beta_bet', 'B', 'sbar', 'E', 'm']).set_index('date') for k, v in res.items()}
for k in D:
    D[k]['xsec'] = D[k].AR - D[k].timing; D[k]['rest'] = D[k].xsec - D[k].beta_bet
base = D[('v4', 1.0)]
chk = L.daily(F.set_index(['date', 'stock_id']).score)
print(f'check: v4 AR from this script {base.AR.mean():.4f} vs mlib.daily {chk.AR.mean():.4f}; days {len(base)}')

def row(lab, X, b=None):
    pr = L.periods(X.index); r19 = X.AR.rolling(19).mean().dropna()
    r = {'config': lab, 'val': X.AR[pr['val']].mean(), 'hold': X.AR[pr['hold']].mean(), 'post': X.AR[pr['post']].mean(), 'all': X.AR.mean(),
         'sd': X.AR.std(), 'timing': X.timing.mean(), 'beta_bet': X.beta_bet.mean(), 'rest': X.rest.mean(), 'mean B': X.B.mean(),
         'mean sbar': X.sbar.mean(), 'mean E': X.E.mean(), 'corr(AR,m)': L.corr(X.AR, X.m),
         'P19>0': (r19 > 0).mean(), 'P19>20': (r19 > 20).mean()}
    if b is not None:
        dd = X.AR - b.AR
        for k in ('val', 'hold', 'post'):
            r['d_' + k] = dd[pr[k]].mean(); r['t_' + k] = L.plain_t(dd[pr[k]])
    return r
T = pd.DataFrame([row('v4 rule (control)', base), row('H6a hedge with the number of 1s', D[('H6a', 0.0)], base),
                  row('H6b hedge in proportion to beta', D[('H6b', 0.0)], base)])
print('\npre-registered test (target: B = 0, i.e. E* = sbar_v4 * mean beta), Top50 bp/day, walk-forward 2022-01..2026-10')
print(T.round(3).to_string(index=False))
tr = T.iloc[1:].sort_values('t_val', ascending=False).reset_index(drop=True); thr = [1.96, 1.645]; alive = True
for i, r in tr.iterrows():
    ok = alive and r.t_val >= thr[i] and r.d_hold > 0; alive = alive and r.t_val >= thr[i]
    print(f'  {r.config:34s} val t {r.t_val:+.2f} (needs {thr[i]:.2f}), hold diff {r.d_hold:+.2f} -> {"ADOPT" if ok else "do not adopt"}')

print('\nreport only: partial hedges, E* = sbar_v4*mean(beta) + lambda*B_v4 (lambda 1 = v4, 0 = fully hedged)')
rows = [row(f'{meth} lambda {lam}', D[(meth, lam)], base) for meth in ('H6a', 'H6b') for lam in LAMS[1:]]
print(pd.DataFrame(rows)[['config', 'all', 'val', 'hold', 'post', 'sd', 'mean B', 'mean sbar', 'beta_bet', 'rest', 'corr(AR,m)', 'P19>0', 'P19>20',
                          'd_val', 't_val']].round(3).to_string(index=False))

print('\nscore by market day type (deciles of m within 2022-01..2026-10), bp/day:')
q = pd.qcut(base.m.rank(method='first'), 10, labels=False)
for lab, X in (('v4', base), ('H6a full', D[('H6a', 0.0)]), ('H6b full', D[('H6b', 0.0)]), ('H6b lambda 0.5', D[('H6b', 0.5)])):
    print(f'  {lab:15s} biggest-down 10% {X.AR[q == 0].mean():7.2f} | middle 80% {X.AR[(q > 0) & (q < 9)].mean():6.2f} | biggest-up 10% {X.AR[q == 9].mean():7.2f}')
print(f'  (mean m in those groups: {base.m[q == 0].mean():.0f}, {base.m[(q > 0) & (q < 9)].mean():.1f}, {base.m[q == 9].mean():.0f} bp)')
for lab, a, b in (('static 2026-06-01..09-30', '2026-06-01', '2026-09-30'), ('live 2026-10-01..10-22', '2026-10-01', '2026-10-22')):
    print(f'\nfor information, {lab}:')
    for key in (('v4', 1.0), ('H6a', 0.0), ('H6b', 0.0), ('H6b', 0.5)):
        X = D[key]; m_ = (X.index >= a) & (X.index <= b); dd = (X.AR - base.AR)[m_]
        print(f'  {key[0]} lambda {key[1]:.2f}: AR {X.AR[m_].mean():7.2f}  diff {dd.mean():+6.2f} (t {L.plain_t(dd):.2f}, {m_.sum()} days)')
pd.to_pickle(D, os.path.join(L.CACHE, 'h6_daily.pkl'))

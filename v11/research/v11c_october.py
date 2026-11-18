r"""
v11c_october.py -- can v11's parameters be changed so that October 2026 looks better, and what does that do elsewhere?
(report only; nothing here changes v11)

Grid over v11's parameters (v11 = w_T 0.5, T2 weight 0.5, T3 on, X1 on, hedge lambda 0.75):
  w_T (market-view weight) 0 / 0.25 / 0.5 / 0.75 / 1.0; T2 night weight 0 / 0.25 / 0.5 / 1.0; T3 off / on; X1 off / on;
  lambda 0.5 / 0.75 / 1.0 (1 = no hedge)  -> 240 settings, every one scored on the same 1,164 walk-forward days.
Reports: v11's own numbers, the October-best settings and what they do in the other periods, how well a setting's October
predicts its other periods, and the honest version of "tune on the recent past": pick the best setting on the 15 trading days
before October (2026-09-09..09-30) and look at October.
run (in v11\research): python v11c_october.py > ..\results\v11c_october.txt
"""
import os, sys, time, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v11b_lib as L
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)

PER = {'val': ('2022-01-01', '2025-01-01'), 'hold': ('2025-01-01', '2026-06-01'), 'static': ('2026-06-01', '2026-10-01'),
       'Oct': ('2026-10-01', '2100-01-01'), 'all': ('2022-01-01', '2100-01-01'), 'pre15': ('2026-09-09', '2026-10-01')}
GRID = dict(w_t=[0.0, 0.25, 0.5, 0.75, 1.0], night=[0.0, 0.25, 0.5, 1.0], t3=[0, 1], x1=[0, 1], lam=[0.5, 0.75, 1.0])
_G = {}

def prep():
    d = L.Data()
    pb = d.pbar()
    sd = pb.rolling(250, min_periods=20).std().shift(1)
    adj1 = (sd * d.z_night.reindex(pb.index)).fillna(0.0)          # overlay per unit of T2 weight
    pe = {0: d.stock_view(), 1: d.stock_view(x1=True)}
    _G.update(d=d, pb=pb, adj1=adj1, pe=pe)

def run(cfg):
    w_t, night, t3, x1, lam = cfg
    d, pb, adj1, pe = _G['d'], _G['pb'], _G['adj1'], _G['pe'][x1]
    p = (pb - night * adj1)
    s = w_t * (p.reindex(pe.index.get_level_values('date')).values - 0.5) + pe
    D = d.decide(s, month_start=bool(t3), lam=lam)
    out = {'w_t': w_t, 'night': night, 't3': t3, 'x1': x1, 'lam': lam}
    ix = pd.DatetimeIndex(D.index)
    for k, (a, b) in PER.items():
        m = (ix >= pd.Timestamp(a)) & (ix < pd.Timestamp(b))
        out[k] = D.AR[m].mean()
    out['sd'] = D.AR.std()
    return out, D.AR

def main():
    t0 = time.time()
    prep()
    cfgs = list(itertools.product(*GRID.values()))
    import multiprocessing as mp
    with mp.get_context('fork').Pool(2) as pool:
        res = pool.map(run, cfgs, chunksize=4)
    T = pd.DataFrame([r for r, _ in res])
    AR = pd.DataFrame({i: a for i, (_, a) in enumerate(res)})
    key = lambda w, n, t, x, l: T[(T.w_t == w) & (T.night == n) & (T.t3 == t) & (T.x1 == x) & (T.lam == l)].index[0]
    i11, i9 = key(0.5, 0.5, 1, 1, 0.75), key(0.5, 0.0, 0, 0, 0.75)
    cols = ['w_t', 'night', 't3', 'x1', 'lam', 'val', 'hold', 'static', 'Oct', 'all', 'sd']
    print(f'{len(T)} settings scored on {AR.shape[0]} days in {time.time() - t0:.0f}s (Top50 bp/day; static = 6/1-9/30, Oct = 10/1-10/22, 15 days)')
    print('\nv9 and v11 as they are:')
    print(T.loc[[i9, i11], cols].round(2).to_string())
    for k in ('val', 'hold', 'static', 'Oct', 'all'):
        print(f'  v11 rank in {k}: {int((T[k] > T.loc[i11, k]).sum()) + 1} of {len(T)}; v9 rank {int((T[k] > T.loc[i9, k]).sum()) + 1}')
    print('\nthe 10 settings with the best October, and what they did in the other periods:')
    top = T.sort_values('Oct', ascending=False).head(10)
    print(top[cols].round(2).to_string())
    print(f'  mean of these 10: val {top.val.mean():.2f}, hold {top.hold.mean():.2f}, static {top.static.mean():.2f}, Oct {top.Oct.mean():.2f}, all {top["all"].mean():.2f} '
          f'(v11: {T.loc[i11, "val"]:.2f} / {T.loc[i11, "hold"]:.2f} / {T.loc[i11, "static"]:.2f} / {T.loc[i11, "Oct"]:.2f} / {T.loc[i11, "all"]:.2f})')
    print('\nOctober-best within one-parameter changes of v11 (everything else as v11):')
    for p, vals in GRID.items():
        rows = []
        for v in vals:
            c = {'w_t': 0.5, 'night': 0.5, 't3': 1, 'x1': 1, 'lam': 0.75}; c[p] = v
            i = key(c['w_t'], c['night'], c['t3'], c['x1'], c['lam']); rows.append(T.loc[i, cols])
        print(f'  {p}:'); print(pd.DataFrame(rows)[cols].round(2).to_string(index=False))
    print('\nacross the 240 settings: correlation of October with the other periods (does a good October predict a good setting?)')
    for k in ('val', 'hold', 'static', 'all'):
        print(f'  corr(Oct, {k}) {np.corrcoef(T.Oct, T[k])[0, 1]:+.2f}; rank corr {T.Oct.rank().corr(T[k].rank()):+.2f}')
    print('\n"tune on the recent past" done honestly: best setting on the 15 trading days before October (9/9-9/30) -> October')
    b = T.sort_values('pre15', ascending=False).head(5)
    print(b[cols + ['pre15']].round(2).to_string())
    print(f'  best pre-October setting: October {b.Oct.iloc[0]:.2f} (v11 {T.loc[i11, "Oct"]:.2f}, v9 {T.loc[i9, "Oct"]:.2f}); '
          f'rank of that setting in October {int((T.Oct > b.Oct.iloc[0]).sum()) + 1} of {len(T)}')
    # rolling version over the whole walk-forward: each 15-day block picks the best setting of the previous 15-day block
    blk = (np.arange(AR.shape[0]) // 15)
    picks = []
    for k in range(1, blk.max() + 1):
        prev = AR[blk == k - 1].mean(); cur = AR[blk == k].mean()
        picks.append((cur[prev.idxmax()], cur[i11], cur.mean(), cur.max()))
    P = pd.DataFrame(picks, columns=['picked', 'v11', 'average setting', 'best in hindsight'])
    print(f'\nevery 15-day block of 2022-01..2026-10 ({len(P)} blocks), using the setting that was best in the previous 15-day block:')
    print(f'  picked {P.picked.mean():.2f}, v11 fixed {P.v11.mean():.2f}, average of all 240 settings {P["average setting"].mean():.2f}, '
          f'best in hindsight {P["best in hindsight"].mean():.2f} bp/day; picked - v11 t {((P.picked - P.v11).mean() / (P.picked - P.v11).std() * np.sqrt(len(P))):.2f}')
    T.to_csv(os.path.join(L.V11, 'results', 'v11c_october_grid.csv'), index=False, float_format='%.4f')
    print(f'\ngrid written to ..\\results\\v11c_october_grid.csv; done in {time.time() - t0:.0f}s')

if __name__ == '__main__':
    main()

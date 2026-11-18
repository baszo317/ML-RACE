"""s7: risk 1 -- how much of the chosen design's validation score is selection luck?
family = every configuration that was actually compared during development and can be rebuilt from saved OOS predictions
(v1-era files in 棄案, overnight runs, ablations, the bp rule). Daily AR on 2022-01..2026-05 (val + hold).
(1) CSCV / PBO (Bailey, Borwein, Lopez de Prado & Zhu 2017): 16 blocks, all C(16,8) IS/OOS splits.
(2) nested selection: each year pick the config with the best AR on all earlier years, record the next year.
(3) max-t reality check vs the v1 baseline (stationary bootstrap, Politis-Romano), White (2000) / Romano-Wolf style."""
import os, itertools, numpy as np, pandas as pd, rlib as L
E = L.E
S10, S5, S3 = range(1, 11), range(1, 6), range(1, 4)
pr = {}
for n, s in [('mlp_raw_B', S10), ('mlp_raw_xr', S10), ('tf_raw_B', S5), ('mlp_res_B', S10), ('mlp_res_xr', S10), ('tf_res_B', S3),
             ('mlp_raw_w4', S10), ('tf_raw_w4', S3)]:
    pr[n] = L.load_wf(n, s)
pr3 = {'mlp_raw_B': L.load_wf('mlp_raw_B', S3), 'tf_raw_B': L.load_wf('tf_raw_B', S3)}
cfg = {}
def add(label, probs, names, w=None, exdiv=True):
    mem = [L.member_def(n) for n in names]
    idx = None
    for n in names:
        idx = probs[n].index if idx is None else idx.intersection(probs[n].index)
    pp = {n: probs[n].reindex(idx) for n in names}
    if w is None:
        o = E.combine(pp, [m for m in mem if m['target'] == 'raw'])
    else:
        o = E.combine(pp, mem, w_t=w)
    cfg[label] = L.daily(o[['score']], exdiv=exdiv).AR
for ex in (False, True):
    t = ' +exdiv' if ex else ''
    add('v1-like mlp3+tf3' + t, pr3, ['mlp_raw_B', 'tf_raw_B'], None, ex)
    add('mlp10+tf5 v1 rule' + t, pr, ['mlp_raw_B', 'tf_raw_B'], None, ex)
    add('+w4 mlp,tf v1 rule' + t, pr, ['mlp_raw_B', 'mlp_raw_w4', 'tf_raw_B', 'tf_raw_w4'], None, ex)
    add('+w4 mlp v1 rule' + t, pr, ['mlp_raw_B', 'mlp_raw_w4', 'tf_raw_B'], None, ex)
    add('mlp_raw_B alone v1 rule' + t, pr, ['mlp_raw_B'], None, ex)
    add('mlp_raw_B+w4 v1 rule' + t, pr, ['mlp_raw_B', 'mlp_raw_w4'], None, ex)
for w in (0.0, 0.25, 0.5, 0.75, 1.0):
    add(f'v2 MLP-only w{w}', pr, ['mlp_raw_B', 'mlp_raw_xr', 'mlp_res_B', 'mlp_res_xr'], w)
    add(f'v2 default w{w}', pr, ['mlp_raw_B', 'mlp_raw_xr', 'tf_raw_B', 'mlp_res_B', 'mlp_res_xr'], w)
    add(f'v2 +tf_res w{w}', pr, ['mlp_raw_B', 'mlp_raw_xr', 'tf_raw_B', 'mlp_res_B', 'mlp_res_xr', 'tf_res_B'], w)
add('v2 default no exdiv', pr, ['mlp_raw_B', 'mlp_raw_xr', 'tf_raw_B', 'mlp_res_B', 'mlp_res_xr'], 0.5, False)
add('v2 B only', pr, ['mlp_raw_B', 'tf_raw_B', 'mlp_res_B'], 0.5)
add('v2 res_B only', pr, ['mlp_raw_B', 'mlp_raw_xr', 'tf_raw_B', 'mlp_res_B'], 0.5)
add('v2 res_xr only', pr, ['mlp_raw_B', 'mlp_raw_xr', 'tf_raw_B', 'mlp_res_xr'], 0.5)
add('v2 +w4 market', pr, ['mlp_raw_B', 'mlp_raw_xr', 'tf_raw_B', 'mlp_raw_w4', 'mlp_res_B', 'mlp_res_xr'], 0.5)
for f in ['wf_mlp_B+margin', 'wf_mlp_B_recency', 'wf_mlp_B_quarterly', 'wf_mlp_F', 'wf_transformer_F', 'wf_mlp_B+breadth_taiex']:
    x = L._read(os.path.join(L.ROOT, '棄案', 'transformer', 'results', f + '.pkl'))
    o = pd.DataFrame({'score': x - 0.5}); cfg['v1-era ' + f[3:]] = L.daily(o, exdiv=False).AR
x = L._read(os.path.join(L.ROOT, 'transformer', 'results', 'wf_mlp_B.pkl'))
cfg['v1-era mlp_B'] = L.daily(pd.DataFrame({'score': x - 0.5}), exdiv=False).AR
cfg['v1-era transformer_B'] = L.daily(pd.DataFrame({'score': L._read(L.V1_TF) - 0.5}), exdiv=False).AR
for k in ('v5_volscale', ):
    pass
M = pd.DataFrame(cfg)
M = M[M.index < L.POST_START].dropna()
print(f'family: {M.shape[1]} configurations x {M.shape[0]} days (2022-01..2026-05)')
# ---------- (1) CSCV
S = 16; blocks = np.array_split(np.arange(len(M)), S); A = M.values; N = A.shape[1]
lam, deg = [], []
for c in itertools.combinations(range(S), S // 2):
    isx = np.concatenate([blocks[i] for i in c]); oos = np.concatenate([blocks[i] for i in range(S) if i not in c])
    ism, oom = A[isx].mean(0), A[oos].mean(0)
    n = ism.argmax(); rank = (oom < oom[n]).sum() + 0.5 * ((oom == oom[n]).sum() - 1) + 1
    w = rank / (N + 1); lam.append(np.log(w / (1 - w))); deg.append((ism[n], oom[n], np.median(oom)))
lam = np.array(lam); deg = np.array(deg)
print(f'(1) CSCV: PBO = {np.mean(lam <= 0):.3f}  (share of splits where the IS-best config is at or below the OOS median); '
      f'median logit {np.median(lam):.2f}; IS-best: IS mean {deg[:,0].mean():.2f} -> OOS mean {deg[:,1].mean():.2f} bp '
      f'(OOS median config {deg[:,2].mean():.2f})')
# ---------- (2) nested selection by year
years = [('2023', '2023-01-01', '2024-01-01'), ('2024', '2024-01-01', '2025-01-01'), ('2025', '2025-01-01', '2026-01-01'),
         ('2026-01..05', '2026-01-01', '2026-06-01')]
rows = []
for lab, a, b in years:
    past = M[M.index < a].mean(); best = past.idxmax()
    nxt = M[(M.index >= a) & (M.index < b)]
    rows.append({'year': lab, 'picked (best on all earlier years)': best, 'its earlier AR': past.max(), 'its AR this year': nxt[best].mean(),
                 'v2 default w0.5 this year': nxt['v2 default w0.5'].mean(), 'family median this year': nxt.mean().median(), 'days': len(nxt)})
T = pd.DataFrame(rows); pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 40)
print('(2) nested selection'); print(T.round(2).to_string(index=False))
w = T.days.values
print('    day-weighted: picked %.2f, v2 default %.2f, family median %.2f' % (np.average(T['its AR this year'], weights=w),
      np.average(T['v2 default w0.5 this year'], weights=w), np.average(T['family median this year'], weights=w)))
# ---------- (3) max-t reality check vs v1-like (+exdiv excluded from baseline)
base = M['v1-like mlp3+tf3']
Dd = M.drop(columns=['v1-like mlp3+tf3']).sub(base, axis=0)
tobs = Dd.mean() / Dd.std() * np.sqrt(len(Dd))
rng = np.random.default_rng(1); B = 2000; Tn = len(Dd); pblk = 1 / 10; mx = []
C = (Dd - Dd.mean()).values; sd = Dd.std().values
for _ in range(B):
    idx = np.empty(Tn, int); i = rng.integers(Tn)
    for j in range(Tn):
        idx[j] = i; i = rng.integers(Tn) if rng.random() < pblk else (i + 1) % Tn
    mx.append((C[idx].mean(0) / sd * np.sqrt(Tn)).max())
mx = np.array(mx)
top = tobs.sort_values(ascending=False).head(6)
print('(3) paired t vs v1-like (2022-01..2026-05) and the reality-check p-value (max-t over %d configs, stationary bootstrap):' % len(tobs))
for k, v in top.items():
    print(f'    {k:32s} t = {v:5.2f}   p_adj = {np.mean(mx >= v):.3f}')
print('    95%% critical value of max-t = %.2f' % np.quantile(mx, 0.95))
M.to_pickle(os.path.join(L.CACHE, 'family_daily_AR.pkl'))

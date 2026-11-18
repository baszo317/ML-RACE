"""Compare the 11/09 training point retrained here with the laptop's (ver_train_removed/ = the laptop's model files),
and the 11/10 files predicted with it with the laptop's 11/10 files (base/)."""
import glob, os, numpy as np, pandas as pd, torch
W = '/home/claude/w1111'
print('models (laptop vs retrained here):')
for f in sorted(glob.glob(f'{W}/ver_train_removed/*/2026-11-09_*.pt')):
    m = f.split('/')[-2]; g = f'{W}/ver_train/' + ('v11/models_v11/' if m == 'lgb_res_xrN' else 'v9/models_v9/') + m + '/' + os.path.basename(f)
    a = torch.load(f, map_location='cpu', weights_only=False); b = torch.load(g, map_location='cpu', weights_only=False)
    if isinstance(a, dict) and 'state' in a and isinstance(a['state'], dict):
        sa, sb = a['state'], b['state']
    elif isinstance(a, dict) and 'state_dict' in a:
        sa, sb = a['state_dict'], b['state_dict']
    else:
        sa, sb = a, b
    diffs = []
    def walk(x, y, path=''):
        if isinstance(x, dict):
            for k in x:
                walk(x[k], y[k], path + '/' + str(k))
        elif torch.is_tensor(x):
            diffs.append(float((x.double() - y.double()).abs().max()) if x.numel() else 0.0)
        elif isinstance(x, np.ndarray) and x.dtype.kind in 'fiu':
            diffs.append(float(np.abs(x.astype(float) - y.astype(float)).max()) if x.size else 0.0)
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            diffs.append(abs(float(x) - float(y)))
        elif isinstance(x, (str, bytes)):
            diffs.append(0.0 if x == y else float('inf'))
        elif isinstance(x, (list, tuple)) and len(x) == len(y):
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, path + f'[{i}]')
    walk(sa, sb)
    print(f'  {m:14s} {os.path.basename(f):24s} values compared {len(diffs):4d}, largest difference {max(diffs) if diffs else float("nan"):.2e}')
print('11/10 files (laptop vs here):')
for f in ['v9/out/submission_v9_1110.csv', 'v11/out/submission_v11_1110.csv', 'v11/out/submission_v11_1110_wt0_lam0.5.csv']:
    a = open(f'{W}/base/{f}', 'rb').read().replace(b'\r\n', b'\n'); b = open(f'{W}/ver_train/{f}', 'rb').read().replace(b'\r\n', b'\n')
    print(f'  {f}: 0/1 identical {a == b}')
for f in ['v9/out/probabilities_v9_1110.csv', 'v11/out/probabilities_v11_1110.csv', 'v11/out/probabilities_v11_1110_wt0_lam0.5.csv']:
    a = pd.read_csv(f'{W}/base/{f}', dtype={'stock_id': str}); b = pd.read_csv(f'{W}/ver_train/{f}', dtype={'stock_id': str})
    num = [c for c in a.columns if c not in ('date', 'stock_id')]
    d = (a[num].astype(float) - b[num].astype(float)).abs().max()
    print(f'  {f}: largest difference {d.max():.1e} (score {d.get("score", float("nan")):.1e}, p_e {d.get("p_e", float("nan")):.1e})')

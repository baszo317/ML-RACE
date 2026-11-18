r"""
v16_smoother_posthoc.py -- written AFTER the S3 judgment of v16_smoother.py (hindsight, report only; the verdict of
PREREG_v16b is not changed). S3 passed the rule on AR while its own IC fell; this asks where the AR difference comes from.

  1. stock view p_e: daily rank IC of v11 and of v11 + S3x (S3x in the LightGBM part), paired differences
  2. within-day rank correlation of each S3x with the v11 stock view, and README_v14 theorem 2 (IC_z - rho IC_x)
  3. the AR differences split into timing (share of 1s x Top50 mean) and cross-section, and the share of 1s
  4. shrinkage: posterior mean of the true effect with the overfit study's prior sd tau = 0.185 bp/day
usage (in v16\research): python v16_smoother_posthoc.py > ..\results\v16_smoother_posthoc.txt
"""
import os, sys, time, math
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v16lib as L
import v16_smoother as SM
from v16lib import per, w, nw_t, diff, fmt, PER

TAU = 0.185


def main():
    t0 = time.time()
    print('v16_smoother_posthoc.py -- after the S3 judgment (report only)')
    print(f'  {time.strftime("%Y-%m-%d %H:%M")}, pandas {pd.__version__}, numpy {np.__version__}')
    d = L.Data(); idx = d.idx
    pr, coefs, r2lag, resid = SM.s3_predictions(lambda s: None)
    r_ = resid.reindex(idx)
    sc = {}
    for k, p in pr.items():
        p = p.reindex(idx); p = p.fillna(p.groupby(level='date').transform('mean'))
        sc[k] = SM.scale_to(p, d.LGB)
    sc['S3 selected'] = pd.concat([sc['S3a'], sc['S3b']], axis=1).mean(axis=1)
    pe = {'v11': d.view([d.LGB, d.x1])}
    for k in ('S3-0', 'S3a', 'S3b', 'S3f', 'S3 selected'):
        pe[f'v11 + {k}'] = d.view([d.LGB, d.x1, sc[k]])
    IC = {k: SM.ic_daily(v, r_) for k, v in pe.items()}
    print('\n1. stock view p_e, daily rank IC with the excess return (val / hold / post / all), and minus v11 (NW t):')
    for k, v in IC.items():
        line = f'    {k:18s} ' + ' / '.join(f'{per(v)[p]:.4f}' for p in PER)
        if k != 'v11':
            dd = (v - IC['v11']).dropna()
            line += '   ' + '  '.join(f'{p} {dd[w(dd.index, p)].mean():+.4f} ({nw_t(dd[w(dd.index, p)].values):.2f})' for p in PER)
        print(line)
    print('\n2. each S3x against the v11 stock view (all periods): rank correlation rho, own IC, gain condition IC_z - rho IC_x (README_v14 theorem 2)')
    icx = per(IC['v11'])['all']
    for k in ('S3-0', 'S3a', 'S3b', 'S3f'):
        rho = float(SM.ic_daily(SM.c(pr[k].reindex(idx)), pe['v11']).mean())
        icz = per(SM.ic_daily(SM.c(pr[k].reindex(idx)), r_))['all']
        print(f'    {k:5s} rho {rho:.3f}, IC_z {icz:.4f}, IC_z - rho IC_x {icz - rho * icx:+.4f}')
    D = {'v11': d.run(pe['v11'], True)}
    for k in ('S3-0', 'S3a', 'S3b', 'S3 selected'):
        D[k] = d.run(pe[f'v11 + {k}'], True)
    print('\n3. AR difference split into timing (share of 1s x Top50 mean) and cross-section (bp/day, all periods), share of 1s:')
    for x, y in (('S3a', 'S3-0'), ('S3b', 'S3-0'), ('S3 selected', 'S3-0'), ('S3 selected', 'v11'), ('S3-0', 'v11')):
        dt = (D[x].timing - D[y].timing); dx = (D[x].xsec - D[y].xsec)
        print(f'    {x:11s} - {y:4s}: AR ' + ' / '.join(f'{v:+.2f}' for v in per(D[x].AR - D[y].AR).values())
              + '   timing ' + ' / '.join(f'{v:+.2f}' for v in per(dt).values())
              + '   xsec ' + ' / '.join(f'{v:+.2f}' for v in per(dx).values())
              + f'   share of 1s {(D[x].sbar.mean() + 1) / 2:.3f} vs {(D[y].sbar.mean() + 1) / 2:.3f}')
    print('\n4. shrinkage of the all-period difference (prior N(0, tau^2), tau 0.185 bp/day from the overfit study):')
    for x, y in (('S3 selected', 'S3-0'), ('S3 selected', 'v11')):
        dd = (D[x].AR - D[y].AR).dropna()
        mean, t = float(dd.mean()), float(nw_t(dd.values))
        se = abs(mean / t) if t and np.isfinite(t) and t != 0 else float('nan')
        post = mean * TAU ** 2 / (TAU ** 2 + se ** 2)
        print(f'    {x} - {y}: observed {mean:+.2f} (se {se:.2f}) -> posterior mean {post:+.3f} bp/day')
    print(f'\n({time.time() - t0:.0f}s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())

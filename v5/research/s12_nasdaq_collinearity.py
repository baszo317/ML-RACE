"""s12: why do Nasdaq and S&P get opposite signs in a linear market model? (context for the feature decision, 2.2)
Logistic regression of "Top100 mean intraday return > 0" on the previous U.S. session's returns (standardised),
each index alone and both together, by period. Plain numpy IRLS (no statsmodels / sklearn needed)."""
import numpy as np, pandas as pd, rlib as L

def logit(X, y, iters=50):
    """maximum-likelihood logistic regression with intercept -> coefficients, z-values (inverse Fisher information)"""
    Xd = np.column_stack([np.ones(len(X)), X]); b = np.zeros(Xd.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xd @ b)); W = p * (1 - p)
        H = Xd.T @ (Xd * W[:, None]); step = np.linalg.solve(H, Xd.T @ (y - p)); b += step
        if np.abs(step).max() < 1e-10:
            break
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return b[1:], b[1:] / se[1:]

P = L.panel()
d = P[P.target_r.notna()].groupby('date').agg(m=('target_r', 'mean'), ix=('us_ixic_ret1', 'first'),
                                              sp=('us_gspc_ret1', 'first')).dropna()
d['y'] = (d.m > 0).astype(float)
print(f'{len(d)} days; corr(Nasdaq 1-day return, S&P 1-day return) = {d.ix.corr(d.sp):.3f}')
rows = []
for per, msk in [('2020-03..2021', d.index < '2022-01-01'), ('2022-24 (val)', (d.index >= '2022-01-01') & (d.index < '2025-01-01')),
                 ('2025-26', d.index >= '2025-01-01'), ('all', np.ones(len(d), bool))]:
    x = d[msk]; Z = ((x[['ix', 'sp']] - x[['ix', 'sp']].mean()) / x[['ix', 'sp']].std()).values
    bj, zj = logit(Z, x.y.values); bi, zi = logit(Z[:, :1], x.y.values); bs, zs = logit(Z[:, 1:], x.y.values)
    rows.append({'period': per, 'days': len(x), 'joint: Nasdaq': f'{bj[0]:+.2f} (z {zj[0]:+.1f})', 'joint: S&P': f'{bj[1]:+.2f} (z {zj[1]:+.1f})',
                 'alone: Nasdaq': f'{bi[0]:+.2f} (z {zi[0]:+.1f})', 'alone: S&P': f'{bs[0]:+.2f} (z {zs[0]:+.1f})'})
pd.set_option('display.width', 200)
print(pd.DataFrame(rows).to_string(index=False))
d['spread'] = d.ix - d.sp
print(f'correlation with the Top100 mean intraday return: S&P {d.sp.corr(d.m):+.3f}, Nasdaq {d.ix.corr(d.m):+.3f}, '
      f'Nasdaq - S&P spread {d.spread.corr(d.m):+.3f}')

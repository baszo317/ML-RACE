r"""
v11_metrics.py -- daily metrics of every v11 config, cached in ..\cache\metrics.pkl (computed once per config):
  ic    : daily rank IC of the family prediction on all labelled stocks (Spearman with target_r)
  icp   : the same with Pearson (report only)
  ar    : Top50 bp/day in v9's system with only this family replaced (v9's other members, rules and hedge)
  pe_sd : daily sd of the centred stock view on the Top50 (scale of the stock view against the market view)
Keys: (family, config_name, block) with block 'b1'..'b4' for the baseline blocks and 's1-5' for configs (seed 1..5).
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v11lib as X

F = os.path.join(X.CACHE, 'metrics.pkl')
_S = {}

def store():
    if 'M' not in _S:
        try:
            _S['M'] = pd.read_pickle(F) if os.path.exists(F) else {}
        except Exception as e:                   # e.g. written by another pandas version: recompute from the unit files
            print(f'cannot read {F} ({e!r}); the metrics are recomputed')
            _S['M'] = {}
    return _S['M']

def save():
    pd.to_pickle(store(), F + '.tmp'); os.replace(F + '.tmp', F)

def context():
    if 'P' not in _S:
        P = X.V.panel()
        _S['P'] = P
        _S['tr'] = P.set_index(['date', 'stock_id']).target_r
        _S['AR'] = X.ARSystem(P)
    return _S

def compute(family, d, with_ar=True):
    c = context()
    p = X.family_pred(family, d)
    out = {'ic': X.daily_ic(p, c['tr']), 'icp': X.daily_ic(p, c['tr'], 'pearson')}
    if with_ar:
        kw = {'L': d} if family == 'lgb' else {'M': d}
        out['ar'] = c['AR'].ar(**kw)
        out['pe_sd'] = c['AR'].pe_sd(**kw)
    return out

def get(family, name, block='s1-5', d=None, with_ar=True, recompute=False):
    M = store(); k = (family, name, block)
    if k in M and not recompute and (not with_ar or 'ar' in M[k]):
        return M[k]
    if d is None:
        if block.startswith('b'):
            d = X.base_blocks(family)[int(block[1:]) - 1] if name == 'base' else None
        else:
            d = X.config_preds(family, [] if name == 'base' else name.split('+'))
        if d is None:
            raise RuntimeError(f'predictions of {k} not available')
    M[k] = compute(family, d, with_ar)
    return M[k]

def base(family, what='ic'):
    """mean over the four 5-seed blocks of the baseline (daily series)"""
    return pd.concat([get(family, 'base', f'b{i}')[what] for i in range(1, 5)], axis=1).mean(axis=1)

def delta(family, name, what='ic', ref='blocks'):
    """daily difference of a config to the baseline (ref 'blocks' = mean of the 4 blocks; 'b1' = seed 1-5 only, v10's baseline)"""
    x = get(family, name)[what]
    b = base(family, what) if ref == 'blocks' else get(family, 'base', ref)[what]
    return (x - b.reindex(x.index)).dropna()

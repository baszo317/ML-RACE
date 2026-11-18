"""
ens_lib.py -- shared code of v2 (used by train_and_predict_v2.py and overnight_experiments.py)

  * panel      : comp_features (67 features, unchanged) + features_extra (ranks, calendar, ex-dividend, residual target)
  * members    : MLP / Transformer models that differ by seed, training window and (optionally) target
  * ensemble   : market view (average P(close > open) of the 'raw' members over the 100 stocks of the day) +
                 stock view ('resid' members: P(stock beats the same-day average), centred per day), see combine()
  * ex-dividend: prediction forced to 0 on announced ex-dividend days (see fetch_exdiv.py)
Everything uses information published before 09:00 of the predicted day; scaling statistics come from the training
rows only (tfmodel.Scaler), as in v1.
"""
import os, sys, json, time, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
TF_DIR = os.path.join(ROOT, 'transformer')
for p in (HERE, TF_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
from comp_features import build_comp_features, placeholder_dates          # v1 code, unchanged
import features_extra as fe

DATA = os.path.join(ROOT, 'data')
TRAIN_START = pd.Timestamp('2020-03-02')
HP = dict(d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.2, lr=1e-3, wd=1e-2, bs=1024, epochs=10, patience=2)   # = v1 HP

def feature_sets():
    B = json.load(open(os.path.join(TF_DIR, 'final_feature_lists.json')))['B_competition_only']
    return {'B': B, 'B+XR': B + fe.XR, 'B+CAL': B + fe.CAL + fe.EXDIV, 'B+ALL': B + fe.XR + fe.CAL + fe.EXDIV + fe.MKT_EXTRA}

def build_panel(data_dir=DATA, pred_dates=()):
    P = build_comp_features(data_dir, pred_dates)
    return fe.add_all(P, data_dir)

# ------------------------------------------------------------------ members
def member(name, kind, window=None, feats='B', target='raw', seeds=5):
    return dict(name=name, kind=kind, seq_len=20 if kind == 'transformer' else 1, window=window, feats=feats, target=target, seeds=seeds)

def window_start(train_end, years):
    if years is None:
        return TRAIN_START
    return max(TRAIN_START, pd.Timestamp(train_end) + pd.Timedelta(days=1) - pd.DateOffset(years=years))

def target_panel(P, target):
    if target == 'raw':
        return P
    Q = P.copy(); Q['target_r'] = Q['resid_r']; Q['y'] = Q['y_resid']          # 'resid': beat the same-day average
    return Q

def train_member(P, m, seed, train_end, feats, start=None):
    """train one seed of member m on labelled rows in [window start, train_end] (train_end inclusive)"""
    from tfmodel import train_model
    start = window_start(train_end, m['window']) if start is None else start
    Q = target_panel(P, m['target']); te = pd.Timestamp(train_end)
    return train_model(m['kind'], Q, feats, lambda d: (d.date >= start) & (d.date <= te), m['seq_len'], seed=seed, hp=HP)

def save_bundle(b, m, seed, train_end, path):
    import torch
    torch.save(dict(state_dict=b['model'].state_dict(), scaler=b['scaler'].state(), kind=b['kind'], seq_len=b['seq_len'],
                    features=b['features'], hp=b['hp'], train_end=str(pd.Timestamp(train_end).date()), best_epoch=b['best_epoch'],
                    member=m, seed=seed), path)

def load_bundle(path):
    import torch
    from tfmodel import make_model, Scaler
    try:
        ck = torch.load(path, weights_only=False)
    except TypeError:
        ck = torch.load(path)
    keys = ('d_model', 'n_heads', 'n_layers', 'd_ff', 'dropout') if ck['kind'] == 'transformer' else ('dropout',)
    mdl = make_model(ck['kind'], len(ck['features']), ck['seq_len'], **{k: ck['hp'][k] for k in keys})
    mdl.load_state_dict(ck['state_dict']); mdl.eval()
    return dict(model=mdl, scaler=Scaler.from_state(ck['scaler']), kind=ck['kind'], seq_len=ck['seq_len'], features=ck['features'],
                hp=ck['hp'], train_end=ck['train_end'], member=ck.get('member'), seed=ck.get('seed'), best_epoch=ck.get('best_epoch'))

def predict_rows(b, P, mask_fn):
    from tfmodel import predict
    return predict(b, P, mask_fn).set_index(['date', 'stock_id']).p

# ------------------------------------------------------------------ ensemble
W_T = 0.5     # weight of the market view (see README_v2.md: calibration slopes, walk-forward grid)

def family_mean(member_probs, members, target):
    """average over the members with this target: first within each family (mlp / transformer), then over families"""
    fam = {}
    for m in members:
        if m['name'] in member_probs and m['target'] == target:
            fam.setdefault(m['kind'], []).append(member_probs[m['name']])
    if not fam:
        return None, {}
    parts = {k: pd.concat(v, axis=1).mean(axis=1) for k, v in fam.items()}
    return pd.concat(list(parts.values()), axis=1).mean(axis=1), parts

def combine(member_probs, members, w_t=W_T):
    """two views (indexed by date, stock_id; all 100 stocks of a day must be present):
      market view : pbar  = mean over the 100 stocks of P(close > open) of the 'raw' members
      stock view  : p_e   = P(stock beats the same-day average) of the 'resid' members, centred per day
      score = w_t * (pbar - 0.5) + (p_e - mean_day(p_e));  prediction = 1 if score > 0
    without 'resid' members it falls back to the v1 rule on the raw ensemble (score = p_raw - 0.5)."""
    p_raw, raw_parts = family_mean(member_probs, members, 'raw')
    p_res, res_parts = family_mean(member_probs, members, 'resid')
    out = pd.DataFrame({'p_raw': p_raw})
    for k, v in raw_parts.items():
        out[f'p_raw_{k}'] = v
    out['pbar'] = out.groupby(level='date').p_raw.transform('mean')
    if p_res is not None:
        out['p_e'] = p_res.reindex(out.index)
        for k, v in res_parts.items():
            out[f'p_e_{k}'] = v.reindex(out.index)
        out['p_e_c'] = out.p_e - out.groupby(level='date').p_e.transform('mean')
        out['score'] = w_t * (out.pbar - 0.5) + out.p_e_c
    else:
        out['score'] = out.p_raw - 0.5
    return out

def apply_exdiv(pred, ex_flag):
    """pred: 0/1 array; ex_flag: 1 on announced ex-dividend days -> 0"""
    return np.where(np.asarray(ex_flag) == 1, 0, np.asarray(pred)).astype(int)

# ------------------------------------------------------------------ scoring (competition metric)
def daily_score(df, pred_col='pred'):
    """df with columns date, top50, target_r, pred (0/1) -> Series bp/day of the Top50 average of (2p-1)*r"""
    q = df[df.top50]
    return ((2 * q[pred_col] - 1) * q.target_r).groupby(q.date).mean() * 1e4

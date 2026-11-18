r"""
seqlib.py (v13) -- LSTM and 1D-CNN stock-view members that read RAW daily price / volume / flow sequences
(v13\research\PREREG_v13.md).  Research only: no daily program is touched.

Why a new input: v8 (README_v8 sections 12, 14) already trained LSTM / GRU / CNN / Transformer on 20-day windows of the
B features (67 engineered columns); all were worse than the MLP members.  The engineered features already summarise the
history (ret20, RSI, intra_bias_250, ...), so a sequence model on top of them has little left to learn.  Here the
sequence models get the raw daily series instead (Qlib Alpha360-like, plus Jiang-Kelly-Xiu "image scaling") and must
build their own features.  Everything else (label, weights, scaler, optimiser, early stopping, refit) is v8's.

Input for predicting day D = the stock's last L = 60 trading days D-60 .. D-1 (all known before 09:00 of D):
  per-day channels (standardised with transformer\tfmodel.Scaler fitted on the training rows), RAW_FEATS:
    intra   log(C/O)                      gap     log(O / C_prev)
    upper   log(H / max(O, C))            lower   log(min(O, C) / L)
    vol     log(V / 20-day mean V)        trades  log(N / 20-day mean N)       size  log((V/N) / 20-day mean of V/N)
    fnet tnet dsnet dhnet  foreign / investment-trust / dealer-own / dealer-hedge net buy (shares) / 20-day mean V
    fhchg   change of the foreign holding ratio (percentage points)
    mkt     mean intra of the 100 stocks that day           rel   intra - mkt
  window channels (computed per window in RawSeqData.batch, already in [-0.5, 0.5]), IMG = 5:
    O, H, L, C scaled to the window: (log X - min log L) / (max log H - min log L) - 0.5     (Jiang, Kelly, Xiu 2023)
    V / max V of the window - 0.5
  adjusted prices: only ratios inside a window are used, so a later re-adjustment of the whole series changes nothing.
Members (architectures fixed before any result, PREREG_v13.md 1):
  lstm  = v8's LSTMClassifier (v8\research\lstm_lib.py): 2-layer LSTM, hidden 32, dropout 0.2, head on [last, mean]
  tcn   = 1D CNN: 1x1 input conv to 32 channels, 4 residual conv layers (kernel 3, dilation 1, 2, 4, 8; receptive field
          31 days), GELU, dropout 0.2, head on [last day, window mean]
Needs torch (CPU is enough), numpy, pandas.
"""
import os, sys, json, math, warnings
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); V13 = os.path.dirname(HERE)
ROOT = os.environ.get('MLRACE_ROOT', os.path.dirname(V13))
for p in (os.path.join(ROOT, 'transformer'), os.path.join(ROOT, 'v8', 'research')):
    if p not in sys.path:
        sys.path.insert(0, p)

CACHE = os.path.join(V13, 'cache'); RESULTS = os.path.join(V13, 'results')
TRAIN_START = pd.Timestamp('2020-03-02')
FOLDS = [('2022-01-01', '2023-01-01'), ('2023-01-01', '2024-01-01'), ('2024-01-01', '2025-01-01'), ('2025-01-01', '2026-01-01'),
         ('2026-01-01', '2026-06-01'), ('2026-06-01', '2100-01-01')]
SEQ_LEN = 60
RAW_FEATS = ['intra', 'gap', 'upper', 'lower', 'vol', 'trades', 'size', 'fnet', 'tnet', 'dsnet', 'dhnet', 'fhchg', 'mkt', 'rel']
LEVELS = ['lo', 'lh', 'll', 'lc', 'lv']            # log O, H, L, C, V of the day (unscaled; only used inside a window)
N_IMG = 5
N_IN = len(RAW_FEATS) + N_IMG
# = ens_lib.HP (v1 .. v12) except epochs / patience: raw inputs may need more passes, so the cap must not bind
HP = dict(dropout=0.2, lr=1e-3, wd=1e-2, bs=1024, epochs=30, patience=3)
KINDS = {'lstm': 'LSTM-R', 'tcn': 'CNN-R'}


# ---------------------------------------------------------------------------------------------------- the panel
def _read_market(path):
    r"""= transformer\comp_features.read_market (copied so that building the panel does not need torch)"""
    df = pd.read_csv(path, parse_dates=['date']).sort_values('date').drop_duplicates('date', keep='last').reset_index(drop=True)
    vals = df.columns.drop('date')
    df.loc[(df[vals] == -1).all(axis=1), vals] = np.nan
    return df


def _log(x):
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.log(x)
    out[~np.isfinite(out)] = np.nan
    return out


def _stock_raw(df):
    """day-t raw channels of one stock (NOT shifted yet)"""
    cols = list(df.columns)
    assert '成交股數' in cols[7] and '持股比率' in cols[8] and '外陸資' in cols[9] and '投信' in cols[10] \
        and '自行買賣' in cols[11] and '避險' in cols[12], f'unexpected market columns: {cols[:13]}'
    o, h, l, c = [df[f'etl:adj_{k}'].astype(float) for k in ('open', 'high', 'low', 'close')]
    v, n = df[cols[7]].astype(float), df[cols[5]].astype(float)
    F = pd.DataFrame({'date': df.date.values})
    F['intra'] = _log(c / o); F['gap'] = _log(o / c.shift())
    F['upper'] = _log(h / np.maximum(o, c)); F['lower'] = _log(np.minimum(o, c) / l)
    vm = v.rolling(20, min_periods=5).mean(); nm = n.rolling(20, min_periods=5).mean()
    sz = v / n.where(n > 0); szm = sz.rolling(20, min_periods=5).mean()
    F['vol'] = _log(v / vm); F['trades'] = _log(n / nm); F['size'] = _log(sz / szm)
    adv = vm.where(vm > 0)
    for i, nm_ in ((9, 'fnet'), (10, 'tnet'), (11, 'dsnet'), (12, 'dhnet')):
        F[nm_] = (df[cols[i]].astype(float) / adv).values
    F['fhchg'] = df[cols[8]].astype(float).diff().values
    F['lo'], F['lh'], F['ll'], F['lc'], F['lv'] = _log(o), _log(h), _log(l), _log(c), _log(v)
    return F


def build_panel(root=ROOT, end=None):
    r"""one row per (date, stock) of v9\cache\core9.pkl (the 100 stocks, 2020-01-02 .. its last day) with
    label columns (target_r, resid_r, y_resid, top50) and the raw channels of the PREVIOUS trading day (shift 1)"""
    core = pd.read_pickle(os.path.join(root, 'v9', 'cache', 'core9.pkl'))
    core = core[['date', 'stock_id', 'top50', 'target_r', 'resid_r', 'y_resid']].copy()
    core['date'] = pd.to_datetime(core.date).astype('datetime64[ns]')
    core['stock_id'] = core.stock_id.astype(str)
    end = core.date.max() if end is None else pd.Timestamp(end)
    info = json.load(open(os.path.join(root, 'data', 'info.json'), encoding='utf-8'))
    R = []
    for s in info['Top100@Y']:
        df = _read_market(os.path.join(root, 'data', 'market', f'{s}.csv'))
        df = df[df.date <= end].reset_index(drop=True)
        F = _stock_raw(df); F['stock_id'] = s; R.append(F)
    R = pd.concat(R, ignore_index=True)
    R['date'] = pd.to_datetime(R.date).astype('datetime64[ns]')
    R['mkt'] = R.groupby('date').intra.transform('mean'); R['rel'] = R.intra - R.mkt
    R = R.sort_values(['stock_id', 'date']).reset_index(drop=True)
    ch = RAW_FEATS + LEVELS
    R[ch] = R.groupby('stock_id')[ch].shift(1)                         # known only after the close -> next trading day
    P = core.merge(R[['date', 'stock_id'] + ch], on=['date', 'stock_id'], how='left')
    P = P[P.date <= end].sort_values(['date', 'stock_id']).reset_index(drop=True)
    for k in ch:
        P[k] = P[k].astype(np.float64)
    return P


def load_panel(rebuild=False):
    r"""cached in v13\cache\raw_panel.pkl (rebuilt when missing or unreadable, e.g. after a pandas upgrade)"""
    f = os.path.join(CACHE, 'raw_panel.pkl')
    if not rebuild and os.path.exists(f):
        try:
            P = pd.read_pickle(f)
            if all(k in P for k in RAW_FEATS + LEVELS):
                return P
        except Exception:
            pass
    P = build_panel()
    os.makedirs(CACHE, exist_ok=True); P.to_pickle(f + '.tmp'); os.replace(f + '.tmp', f)
    return P


# ---------------------------------------------------------------------------------------------------- torch part
def torch_parts():
    """imported lazily so that the panel / evaluation code runs without torch"""
    import torch, torch.nn as nn
    import tfmodel as TM                                   # transformer\tfmodel.py: Scaler, SeqData, _fit, _weights
    from lstm_lib import LSTMClassifier                    # v8\research\lstm_lib.py (the v8 LSTM, unchanged)

    class TCNClassifier(nn.Module):
        """x: [batch, L, n_in] -> 1x1 conv -> 4 residual dilated convs (k 3, d 1/2/4/8) -> [last day, window mean] -> logit"""
        def __init__(self, n_in, seq_len, ch=32, dropout=0.2, dilations=(1, 2, 4, 8)):
            super().__init__()
            self.inp = nn.Conv1d(n_in, ch, kernel_size=1)
            self.convs = nn.ModuleList([nn.Conv1d(ch, ch, kernel_size=3, padding=d, dilation=d) for d in dilations])
            self.act = nn.GELU(); self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(2 * ch, 1)

        def forward(self, x):
            h = self.inp(x.transpose(1, 2))                                     # [batch, ch, L]
            for cv in self.convs:
                h = h + self.drop(self.act(cv(h)))                              # same length L (padding = dilation)
            return self.head(torch.cat([h[:, :, -1], h.mean(2)], dim=1)).squeeze(-1)

    class RawSeqData(TM.SeqData):
        """tfmodel.SeqData (rows sorted by stock, date; window = rows j-L+1 .. j of the same stock) plus the 5 window
        channels of the module docstring, computed per batch from the unscaled log prices / volume"""
        def __init__(self, P, features, seq_len):
            super().__init__(P, features, seq_len)
            self.lev = self.P[LEVELS].to_numpy(dtype=np.float64)

        def window_arrays(self, idx):
            w = idx[:, None] - (self.L - 1) + np.arange(self.L)[None, :]
            Xs = self.X[w]                                                       # [B, L, F] float32, scaled
            Lv = self.lev[w]                                                     # [B, L, 5] log O H L C V
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                lo = np.nanmin(Lv[:, :, 2], axis=1); hi = np.nanmax(Lv[:, :, 1], axis=1)
                rng = np.maximum(hi - lo, 1e-4)
                img = (Lv[:, :, :4] - lo[:, None, None]) / rng[:, None, None] - 0.5
                vmax = np.nanmax(Lv[:, :, 4], axis=1)
                vv = np.exp(Lv[:, :, 4] - vmax[:, None]) - 0.5
            Z = np.concatenate([Xs, img.astype(np.float32), vv[:, :, None].astype(np.float32)], axis=2)
            return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        def batch(self, idx):
            return torch.from_numpy(self.window_arrays(idx))

    def make(kind, seq_len=SEQ_LEN, dropout=0.2):
        if kind == 'lstm':
            return LSTMClassifier(N_IN, seq_len, dropout=dropout)
        if kind == 'tcn':
            return TCNClassifier(N_IN, seq_len, dropout=dropout)
        raise ValueError(kind)

    def train(kind, P, train_mask, seed, hp=None, val_days=120, max_rows=None):
        """= tfmodel.train_model with RawSeqData: early stopping on the last val_days trading days of the window
        (weighted log-loss), then refit on the whole window for the chosen number of epochs.
        P must hold target_r / y of the wanted label (resid: target_r = resid_r, y = y_resid)."""
        hp = {**HP, **(hp or {})}
        torch.manual_seed(seed); np.random.seed(seed)
        D = RawSeqData(P, RAW_FEATS, SEQ_LEN)
        tm = np.asarray(train_mask(D.P), dtype=bool) & D.P.target_r.notna().values
        scaler = TM.Scaler().fit(D.raw[tm]); D.scale(scaler)
        idx_all = D.valid(tm)
        if max_rows:                                                            # test mode: the most recent rows only
            idx_all = np.sort(idx_all[np.argsort(D.P.date.values[idx_all], kind='stable')[-max_rows:]])
        dates = D.P.date.values[idx_all]
        ud = np.sort(np.unique(dates)); cut = ud[-val_days] if len(ud) > val_days * 2 else ud[-1]
        y = D.P.y.values.astype(np.float32); w = TM._weights(D.P.target_r.fillna(0).values)
        tr, va = idx_all[dates < cut], idx_all[dates >= cut]
        model = make(kind, dropout=hp['dropout'])
        model, best_ep = TM._fit(model, D, tr, y[tr], w[tr], hp['epochs'], hp['lr'], hp['wd'], hp['bs'], seed,
                                 val=(va, y[va], w[va]), patience=hp['patience'])
        best_ep = max(best_ep, 1)
        torch.manual_seed(seed)
        model = make(kind, dropout=hp['dropout'])
        model, _ = TM._fit(model, D, idx_all, y[idx_all], w[idx_all], best_ep, hp['lr'], hp['wd'], hp['bs'], seed)
        n_par = int(sum(p.numel() for p in model.parameters()))
        return dict(model=model.eval(), scaler=scaler, kind=kind, best_epoch=int(best_ep), rows=int(len(idx_all)),
                    val_rows=int(len(va)), n_params=n_par, hp=hp)

    def predict(bundle, P, pred_mask, bs=4096):
        D = RawSeqData(P, RAW_FEATS, SEQ_LEN); D.scale(bundle['scaler'])
        pm = np.asarray(pred_mask(D.P), dtype=bool)
        idx = D.valid(pm); out = D.P.loc[idx, ['date', 'stock_id']].copy()
        m = bundle['model'].eval()
        with torch.no_grad():
            out['p'] = np.concatenate([torch.sigmoid(m(D.batch(idx[s:s + bs]))).numpy() for s in range(0, len(idx), bs)]) \
                if len(idx) else np.zeros(0)
        return out

    return dict(torch=torch, TM=TM, RawSeqData=RawSeqData, make=make, train=train, predict=predict, TCNClassifier=TCNClassifier)


# ---------------------------------------------------------------------------------------------------- files
def unit_file(kind, fold, seed, test=False):
    return os.path.join(CACHE, 'seq_raw_test' if test else 'seq_raw', kind, f'{fold}_seed{seed}.npz')


def save_unit(f, out, meta):
    os.makedirs(os.path.dirname(f), exist_ok=True)
    tmp = f[:-4] + '.tmp'
    with open(tmp, 'wb') as fh:
        np.savez_compressed(fh, date=np.asarray(pd.to_datetime(out.date), dtype='datetime64[ns]').astype(np.int64),
                            sid=np.asarray(out.stock_id.astype(str).tolist(), dtype='<U8'), p=out.p.to_numpy(np.float64),
                            meta=np.array(json.dumps(meta)))
    os.replace(tmp, f)


def load_npz(f):
    z = np.load(f, allow_pickle=True)
    idx = pd.MultiIndex.from_arrays([pd.to_datetime(z['date']).astype('datetime64[ns]'),
                                     np.asarray(z['sid']).astype(str)], names=['date', 'stock_id'])
    return pd.Series(np.asarray(z['p'], dtype=np.float64), index=idx, name='p')


def load_meta(f):
    z = np.load(f, allow_pickle=True)
    return json.loads(str(z['meta'])) if 'meta' in z else {}


def load_member(kind, seeds=(1, 2, 3, 4, 5), pdir=None, test=False):
    """seed-averaged walk-forward predictions of one member (all six folds), or None if a unit is missing"""
    parts = []
    for fa, _ in FOLDS:
        fs = [os.path.join(pdir, kind, f'{fa}_seed{s}.npz') if pdir else unit_file(kind, fa, s, test) for s in seeds]
        fs = [f for f in fs if os.path.exists(f)]
        if not fs:
            return None
        parts.append(pd.concat([load_npz(f) for f in fs], axis=1).mean(axis=1))
    return pd.concat(parts).sort_index()

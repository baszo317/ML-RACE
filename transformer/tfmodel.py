"""
tfmodel.py -- Transformer (and MLP baseline) for "close > open" prediction, plus data utilities.

Sample = (stock, day D). Input = the stock's feature rows of the last L trading days up to and including D
(each row already uses only information published before 09:00 of its own day, so the whole window is past info).
Loss = binary cross-entropy weighted by |intraday return| (clipped at 5%), which is exactly the competition score
(2p-1)(close-open)/open turned into a weighted classification problem.
"""
import math, copy
import numpy as np, pandas as pd, torch, torch.nn as nn

# ------------------------------------------------------------------ scaling (fit on training rows only)
class Scaler:
    def fit(self, X):
        self.lo = np.nanquantile(X, 0.01, axis=0); self.hi = np.nanquantile(X, 0.99, axis=0)
        Xc = np.clip(X, self.lo, self.hi)
        self.mu = np.nanmean(Xc, axis=0); self.sd = np.nanstd(Xc, axis=0); self.sd[~(self.sd > 1e-12)] = 1.0
        self.mu = np.nan_to_num(self.mu); return self
    def transform(self, X):
        Z = (np.clip(X, self.lo, self.hi) - self.mu) / self.sd
        return np.nan_to_num(Z, nan=0.0).astype(np.float32)          # missing -> training mean
    def state(self): return {k: getattr(self, k) for k in ('lo', 'hi', 'mu', 'sd')}
    @classmethod
    def from_state(cls, s):
        o = cls(); [setattr(o, k, np.asarray(v)) for k, v in s.items()]; return o

# ------------------------------------------------------------------ models
class TransformerClassifier(nn.Module):
    def __init__(self, n_feat, seq_len, d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.2):
        super().__init__()
        self.inp = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)          # learnable positional embedding
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * d_model, 1))
    def forward(self, x):                      # x: [batch, L, n_feat]
        h = self.norm(self.enc(self.inp(x) + self.pos))
        return self.head(torch.cat([h[:, -1], h.mean(1)], dim=1)).squeeze(-1)    # last day + average over the window

class MLPClassifier(nn.Module):                # baseline recommended by the course: uses only day D's row
    def __init__(self, n_feat, seq_len=1, hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_feat, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))
    def forward(self, x): return self.net(x[:, -1]).squeeze(-1)

def make_model(kind, n_feat, seq_len, **kw):
    return TransformerClassifier(n_feat, seq_len, **kw) if kind == 'transformer' else MLPClassifier(n_feat, seq_len, **kw)

# ------------------------------------------------------------------ sequence data
class SeqData:
    """Rows sorted by (stock, date); a sample is identified by its row index j; window = rows j-L+1..j of the same stock."""
    def __init__(self, P, features, seq_len):
        P = P.sort_values(['stock_id', 'date']).reset_index(drop=True)
        self.P, self.features, self.L = P, features, seq_len
        self.pos = P.groupby('stock_id').cumcount().values                      # position of the row within its stock
        self.raw = P[features].to_numpy(dtype=np.float64)
    def scale(self, scaler): self.X = scaler.transform(self.raw)
    def valid(self, mask):  return np.where(mask & (self.pos >= self.L - 1))[0]   # enough history for a full window
    def batch(self, idx):
        w = idx[:, None] - (self.L - 1) + np.arange(self.L)[None, :]
        return torch.from_numpy(self.X[w])

# ------------------------------------------------------------------ training
def _weights(r): w = np.minimum(np.abs(r), 0.05); return (w / w.mean()).astype(np.float32)

def _epoch_loss(model, D, idx, y, w, bs=4096):
    model.eval(); tot = 0.0
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            lg = model(D.batch(idx[s:s + bs]))
            tot += float(nn.functional.binary_cross_entropy_with_logits(lg, torch.from_numpy(y[s:s + bs]), weight=torch.from_numpy(w[s:s + bs]), reduction='sum'))
    return tot / w.sum()

def _fit(model, D, idx, y, w, epochs, lr, wd, bs, seed, val=None, patience=3):
    g = torch.Generator().manual_seed(seed); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best, best_ep, best_state, bad = math.inf, 0, None, 0
    for ep in range(1, epochs + 1):
        model.train(); order = torch.randperm(len(idx), generator=g).numpy()
        for s in range(0, len(idx), bs):
            o = order[s:s + bs]
            lg = model(D.batch(idx[o]))
            loss = nn.functional.binary_cross_entropy_with_logits(lg, torch.from_numpy(y[o]), weight=torch.from_numpy(w[o]))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if val is not None:
            vl = _epoch_loss(model, D, *val)
            if vl < best - 1e-5: best, best_ep, best_state, bad = vl, ep, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience: break
    if val is not None and best_state is not None: model.load_state_dict(best_state)
    return model, best_ep

def train_model(kind, P, features, train_mask, seq_len, seed=0, hp=None, val_days=120, refit=True, verbose=False, half_life=None):
    """Fit scaler + model on the rows selected by train_mask(df) -> bool array (rows need a label).
    1) early stopping: train on all but the last `val_days` trading days, monitor weighted log-loss on those days;
    2) refit on the whole training window for the selected number of epochs (keeps the most recent data).
    half_life (trading days, optional): recency weighting, a sample k trading days before the last training day
    gets weight x 0.5**(k/half_life)."""
    hp = {**dict(d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.2, lr=1e-3, wd=1e-2, bs=512, epochs=15, patience=3), **(hp or {})}
    torch.manual_seed(seed); np.random.seed(seed)
    D = SeqData(P, features, seq_len)
    tm = np.asarray(train_mask(D.P), dtype=bool) & D.P.target_r.notna().values
    scaler = Scaler().fit(D.raw[tm]); D.scale(scaler)
    idx_all = D.valid(tm); dates = D.P.date.values[idx_all]
    ud = np.sort(np.unique(dates)); cut = ud[-val_days] if len(ud) > val_days * 2 else ud[-1]
    y = D.P.y.values.astype(np.float32); w = _weights(D.P.target_r.fillna(0).values)
    if half_life:
        tdays = np.unique(dates)                                                   # sorted training days
        age = (len(tdays) - 1) - np.searchsorted(tdays, D.P.date.values)            # trading days before the last training day
        dec = np.where(age >= 0, 0.5 ** (np.clip(age, 0, None) / float(half_life)), 1.0)
        w0 = w[idx_all].mean(); w = (w * dec).astype(np.float32); w *= np.float32(w0 / w[idx_all].mean())
    mkw = {k: hp[k] for k in (('d_model', 'n_heads', 'n_layers', 'd_ff', 'dropout') if kind == 'transformer' else ('dropout',))}
    tr, va = idx_all[dates < cut], idx_all[dates >= cut]
    model = make_model(kind, len(features), seq_len, **mkw)
    model, best_ep = _fit(model, D, tr, y[tr], w[tr], hp['epochs'], hp['lr'], hp['wd'], hp['bs'], seed, val=(va, y[va], w[va]), patience=hp['patience'])
    best_ep = max(best_ep, 1)
    if refit:
        torch.manual_seed(seed)
        model = make_model(kind, len(features), seq_len, **mkw)
        model, _ = _fit(model, D, idx_all, y[idx_all], w[idx_all], best_ep, hp['lr'], hp['wd'], hp['bs'], seed)
    if verbose: print(f'  {kind} seed {seed}: best epoch {best_ep}, train rows {len(idx_all)}')
    return dict(model=model.eval(), scaler=scaler, kind=kind, seq_len=seq_len, features=features, hp=hp, best_epoch=best_ep)

def predict(bundle, P, pred_mask):
    """Probability that close > open for rows where pred_mask is True (needs the previous seq_len-1 rows in P)."""
    D = SeqData(P, bundle['features'], bundle['seq_len']); D.scale(bundle['scaler'])
    pm = np.asarray(pred_mask(D.P), dtype=bool)
    idx = D.valid(pm); out = D.P.loc[idx, ['date', 'stock_id']].copy()
    with torch.no_grad():
        out['p'] = np.concatenate([torch.sigmoid(bundle['model'](D.batch(idx[s:s + 4096]))).numpy() for s in range(0, len(idx), 4096)])
    return out

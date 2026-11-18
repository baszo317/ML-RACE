r"""
npmlp.py -- numpy re-implementation of the MLP members' training (transformer\tfmodel.py MLPClassifier + _fit, and
v9\research\v9_train.train_custom), used only by o1_gap.py where torch is not installed.

Same model and procedure, not bit-identical to torch (different random streams):
  * MLPClassifier: Linear(d, 64) - GELU - Dropout(0.2) - Linear(64, 32) - GELU - Dropout(0.2) - Linear(32, 1);
    PyTorch's default init U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for weights and biases; exact (erf) GELU
  * loss: binary cross-entropy with logits, sample weights, mean over the batch (torch reduction='mean' with weight)
  * AdamW (lr 1e-3, betas 0.9 / 0.999, eps 1e-8, decoupled weight decay 1e-2 on every parameter), gradient-norm clip 1.0,
    batch 1024, random order every epoch
  * early stopping on the last 120 training days (weighted log-loss, eval mode, patience 2, at most 10 epochs, min delta 1e-5),
    then a fresh model with the same seed refit on the whole window for the selected number of epochs
"""
import math
import numpy as np
try:
    from scipy.special import erf as _erf
except Exception:                                   # slow fallback
    _erf = np.vectorize(math.erf)

SQ2 = math.sqrt(2.0); ISQ2PI = 1.0 / math.sqrt(2 * math.pi)


def _gelu(z):
    Phi = 0.5 * (1.0 + _erf(z / SQ2))
    return z * Phi, Phi


def _gelu_grad(z, Phi):
    return Phi + z * np.exp(-0.5 * z * z) * ISQ2PI


class MLP:
    def __init__(self, d, rng, hidden=64, dropout=0.2):
        self.p = dropout
        dims = [(d, hidden), (hidden, hidden // 2), (hidden // 2, 1)]
        self.W, self.b = [], []
        for fi, fo in dims:
            bd = 1.0 / math.sqrt(fi)
            self.W.append(rng.uniform(-bd, bd, (fi, fo)).astype(np.float32))
            self.b.append(rng.uniform(-bd, bd, fo).astype(np.float32))
        self.params = self.W + self.b
        self.m = [np.zeros_like(q) for q in self.params]; self.v = [np.zeros_like(q) for q in self.params]; self.t = 0

    def copy_state(self):
        return [q.copy() for q in self.params]

    def load_state(self, st):
        for q, s in zip(self.params, st):
            q[...] = s

    def logits(self, X, bs=65536):
        out = np.empty(len(X), np.float32)
        for s in range(0, len(X), bs):
            h = X[s:s + bs]
            for i in range(2):
                h, _ = _gelu(h @ self.W[i] + self.b[i])
            out[s:s + bs] = (h @ self.W[2] + self.b[2])[:, 0]
        return out

    def step(self, X, y, w, rng, lr=1e-3, wd=1e-2, b1=0.9, b2=0.999, eps=1e-8, clip=1.0):
        n = len(X)
        z1 = X @ self.W[0] + self.b[0]; a1, P1 = _gelu(z1)
        m1 = (rng.random(a1.shape, dtype=np.float32) >= self.p).astype(np.float32) / (1 - self.p); h1 = a1 * m1
        z2 = h1 @ self.W[1] + self.b[1]; a2, P2 = _gelu(z2)
        m2 = (rng.random(a2.shape, dtype=np.float32) >= self.p).astype(np.float32) / (1 - self.p); h2 = a2 * m2
        z3 = (h2 @ self.W[2] + self.b[2])[:, 0]
        sig = 1.0 / (1.0 + np.exp(-z3))
        dz3 = (w * (sig - y) / n).astype(np.float32)[:, None]
        gW3 = h2.T @ dz3; gb3 = dz3.sum(0)
        dz2 = (dz3 @ self.W[2].T) * m2 * _gelu_grad(z2, P2)
        gW2 = h1.T @ dz2; gb2 = dz2.sum(0)
        dz1 = (dz2 @ self.W[1].T) * m1 * _gelu_grad(z1, P1)
        gW1 = X.T @ dz1; gb1 = dz1.sum(0)
        grads = [gW1, gW2, gW3, gb1, gb2, gb3]
        tot = math.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads))
        if tot > clip:
            sc = clip / (tot + 1e-6); grads = [g * sc for g in grads]
        self.t += 1
        bc1 = 1 - b1 ** self.t; bc2 = 1 - b2 ** self.t
        for q, g, m, v in zip(self.params, grads, self.m, self.v):
            q *= (1 - lr * wd)
            m *= b1; m += (1 - b1) * g
            v *= b2; v += (1 - b2) * g * g
            q -= (lr * (m / bc1) / (np.sqrt(v / bc2) + eps)).astype(np.float32)


def wlogloss_logits(z, y, w):
    z = z.astype(np.float64)
    l = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
    return float((w * l).sum() / w.sum())


def fit(X, y, w, epochs, seed, val=None, patience=2, bs=1024, on_epoch=None):
    """returns (model, best_epoch).  val = (Xv, yv, wv) -> early stopping as tfmodel._fit;
    on_epoch(model, epoch) is called after every epoch (used for the learning curves)"""
    rng = np.random.default_rng(seed)
    model = MLP(X.shape[1], rng)
    order_rng = np.random.default_rng(seed + 7919)
    best, best_ep, best_state, bad = math.inf, 0, None, 0
    for ep in range(1, epochs + 1):
        order = order_rng.permutation(len(X))
        for s in range(0, len(X), bs):
            o = order[s:s + bs]
            model.step(X[o], y[o], w[o], rng)
        if on_epoch is not None:
            on_epoch(model, ep)
        if val is not None:
            vl = wlogloss_logits(model.logits(val[0]), val[1], val[2])
            if vl < best - 1e-5:
                best, best_ep, best_state, bad = vl, ep, model.copy_state(), 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if val is not None and best_state is not None:
        model.load_state(best_state)
    return model, best_ep

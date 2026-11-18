r"""
floamlib.py (v13) -- a leak-free version of F-LOAM for this competition, as a stock-view member (PREREG_v13b.md).

Paper: Liao & Lin (2025), "F-LOAM: an efficient hybrid model for stock price prediction based on SVMD denoising",
Discover Artificial Intelligence 5:347.  F-LOAM = Feature-enhanced LSTM-Optuna-Augmented Model:
  SVMD (VMD K 7, alpha 2000, keep IMFs with |Spearman| > 0.5) denoises 8 inputs (USD/CNY, market cap, PE, PB, PS, PCF ...)
  -> 2-layer LSTM with 50 units -> its hidden state h_t joined to the normalised inputs -> LightGBM,
  Optuna tunes both on the MSE of the TEST set; random 80/20 split; target = closing price (level).
Here, every change is needed either for the competition or to remove a look-ahead (README_v13.md section 7):
  target    beats the same-day average of the 100 stocks (y_resid), weight min(|excess return|, 5%)   (= lgb_res, LSTM-R)
  inputs    the competition has no valuation / FX data: the LSTM reads the B features (67) of the last 20 trading days
            (as every sequence member since v1); LightGBM reads B of day D (unscaled) + h_t (50)
  SVMD      not used: on the whole series it looks into the future (floam_checks.py A3 measures it); a past-only version
            for 67 features x 165,000 stock-days would be about 11 million decompositions (days on a laptop)
  LSTM      2 layers x 50 units (paper), dropout 0.2, head = Linear on the last hidden state h_t; trained like every
            sequence member here (AdamW lr 1e-3, weight decay 1e-2, batch 1024, weighted BCE); early stopping on the last
            120 trading days of the window (max 30 epochs, patience 3); the early-stopped model is KEPT (no refit), so h on
            those 120 days is out of sample, as it will be on the fold
  LightGBM  Optuna (TPE, 30 trials, seed = the unit's seed) over the paper's search space:
              num_leaves {20, 30, 40, 50}, learning_rate [0.01, 0.1], feature_fraction [0.8, 1.0],
              bagging_fraction [0.8, 1.0], bagging_freq {1, 5, 10}
            objective = weighted log-loss on the last 120 trading days of the window (never the fold that is predicted);
            not in the paper, so = lgb_res: min_data_in_leaf 500, lambda_l2 1.0, up to 1000 rounds, early stopping 100;
            then refit on the whole window with the chosen parameters and number of rounds, predict the fold
  saved too (report only): p_lstm = the LSTM's own output; p_fixed = LightGBM on B + h with lgb_res's parameters (no Optuna)
Needs torch, lightgbm, optuna, numpy, pandas.
"""
import os, sys, json, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); V13 = os.path.dirname(HERE)
ROOT = os.environ.get('MLRACE_ROOT', os.path.dirname(V13))
for p in (HERE, os.path.join(ROOT, 'transformer')):
    if p not in sys.path:
        sys.path.insert(0, p)
import seqlib as S                                               # FOLDS, TRAIN_START, unit_file, load_npz

SEQ_B = 20
HP_LSTM = dict(hidden=50, layers=2, dropout=0.2, lr=1e-3, wd=1e-2, bs=1024, epochs=30, patience=3)
N_TRIALS = 30
LGB_BASE = dict(objective='binary', metric='binary_logloss', min_data_in_leaf=500, lambda_l2=1.0, verbose=-1)
LGB_FIXED = dict(LGB_BASE, learning_rate=0.02, num_leaves=15, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1)  # = lgb_res


def b_features():
    return json.load(open(os.path.join(ROOT, 'transformer', 'final_feature_lists.json'), encoding='utf-8'))['B_competition_only']


def load_b_panel():
    r"""v9\cache\core9.pkl: the 100 stocks, labels and the B features (built by v9 from the competition data only)"""
    B = b_features()
    P = pd.read_pickle(os.path.join(ROOT, 'v9', 'cache', 'core9.pkl'))
    P = P[['date', 'stock_id', 'top50', 'target_r', 'resid_r', 'y_resid'] + B].copy()
    P['date'] = pd.to_datetime(P.date).astype('datetime64[ns]'); P['stock_id'] = P.stock_id.astype(str)
    for k in B:
        P[k] = P[k].astype(np.float64)
    return P.sort_values(['date', 'stock_id']).reset_index(drop=True)


def _model_class():
    import torch.nn as nn

    class LSTMh(nn.Module):
        """2-layer LSTM; h_t = top layer's hidden state on the last day; logit = Linear(dropout(h_t))"""
        def __init__(self, n_in, hidden=50, layers=2, dropout=0.2):
            super().__init__()
            self.rnn = nn.LSTM(n_in, hidden, num_layers=layers, batch_first=True, dropout=dropout)
            self.drop = nn.Dropout(dropout); self.head = nn.Linear(hidden, 1)

        def hidden(self, x):
            h, _ = self.rnn(x)
            return h[:, -1]

        def forward(self, x):
            return self.head(self.drop(self.hidden(x))).squeeze(-1)
    return LSTMh


def _cut(dates):
    ud = np.sort(np.unique(dates))
    return ud[-120] if len(ud) > 240 else ud[-1]


def run_unit(P, fa, fb, seed, test=False, threads=1, n_trials=N_TRIALS):
    """one walk-forward unit: fold [fa, fb), seed -> DataFrame(date, stock_id, p, p_fixed, p_lstm), meta"""
    import torch, lightgbm as lgb, optuna
    import tfmodel as TM
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    B = b_features(); LSTMh = _model_class()
    t0 = time.time()
    te = pd.Timestamp(fa) - pd.Timedelta(days=1); a_, b_ = pd.Timestamp(fa), pd.Timestamp(fb)
    Q = P[P.date <= max(b_, te)]
    Q = Q.assign(target_r=Q.resid_r, y=Q.y_resid)                  # 'resid' label (= ens_lib.target_panel)
    max_rows = 20000 if test else None
    hp = dict(HP_LSTM, **(dict(epochs=1, patience=1) if test else {}))

    # ---- 1) LSTM, early stopping on the last 120 trading days of the window; the early-stopped model is kept
    torch.manual_seed(seed); np.random.seed(seed)
    Dt = TM.SeqData(Q[Q.date <= te], B, SEQ_B)                     # training rows only (= tfmodel.train_model)
    tm = ((Dt.P.date >= S.TRAIN_START) & (Dt.P.date <= te) & Dt.P.target_r.notna()).values
    scaler = TM.Scaler().fit(Dt.raw[tm]); Dt.scale(scaler)
    idx = Dt.valid(tm)
    if max_rows:
        idx = np.sort(idx[np.argsort(Dt.P.date.values[idx], kind='stable')[-max_rows:]])
    cut = _cut(Dt.P.date.values[idx])
    y = Dt.P.y.values.astype(np.float32); w = TM._weights(Dt.P.target_r.fillna(0).values)
    tr, va = idx[Dt.P.date.values[idx] < cut], idx[Dt.P.date.values[idx] >= cut]
    model = LSTMh(len(B), hp['hidden'], hp['layers'], hp['dropout'])
    model, best_ep = TM._fit(model, Dt, tr, y[tr], w[tr], hp['epochs'], hp['lr'], hp['wd'], hp['bs'], seed,
                             val=(va, y[va], w[va]), patience=hp['patience'])
    model.eval(); t_lstm = time.time() - t0

    # ---- 2) h_t and the LSTM's own output for the training rows and the fold rows (same scaler, windows look back only)
    Da = TM.SeqData(Q, B, SEQ_B); Da.scale(scaler)
    tma = ((Da.P.date >= S.TRAIN_START) & (Da.P.date <= te) & Da.P.target_r.notna()).values
    fma = ((Da.P.date >= a_) & (Da.P.date < b_) & Da.P.target_r.notna()).values
    i_tr, i_f = Da.valid(tma), Da.valid(fma)
    if max_rows:
        i_tr = np.sort(i_tr[np.argsort(Da.P.date.values[i_tr], kind='stable')[-max_rows:]])

    def hid(ii, bs=4096):
        H, L = [], []
        with torch.no_grad():
            for s in range(0, len(ii), bs):
                h = model.hidden(Da.batch(ii[s:s + bs]))
                H.append(h.numpy()); L.append(torch.sigmoid(model.head(h).squeeze(-1)).numpy())
        return (np.concatenate(H) if H else np.zeros((0, hp['hidden']), np.float32)), (np.concatenate(L) if L else np.zeros(0))
    H_tr, _ = hid(i_tr); H_f, p_lstm = hid(i_f)
    X_tr = np.hstack([Da.P[B].to_numpy(np.float64)[i_tr], H_tr.astype(np.float64)])
    X_f = np.hstack([Da.P[B].to_numpy(np.float64)[i_f], H_f.astype(np.float64)])
    y_tr = Da.P.y.values[i_tr].astype(float)
    w_tr = np.minimum(np.abs(Da.P.target_r.values[i_tr]), 0.05); w_tr = w_tr / w_tr.mean()      # = lgb_res
    trm = Da.P.date.values[i_tr] < _cut(Da.P.date.values[i_tr])
    names = list(B) + [f'h{k}' for k in range(hp['hidden'])]
    dtr = lgb.Dataset(X_tr[trm], y_tr[trm], weight=w_tr[trm], feature_name=names, free_raw_data=False)
    dva = lgb.Dataset(X_tr[~trm], y_tr[~trm], weight=w_tr[~trm], reference=dtr, free_raw_data=False)
    dall = lgb.Dataset(X_tr, y_tr, weight=w_tr, feature_name=names, free_raw_data=False)

    # ---- 3) LightGBM with lgb_res's parameters (report only: the effect of h without Optuna)
    t1 = time.time()
    prm = dict(LGB_FIXED, seed=seed, num_threads=threads)
    m = lgb.train(prm, dtr, 1000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    fixed_rounds = max(int(m.best_iteration), 1)
    p_fixed = lgb.train(prm, dall, fixed_rounds).predict(X_f)

    # ---- 4) F-LOAM: Optuna over the paper's search space, objective on the last 120 days of the window
    def objective(trial):
        q = dict(LGB_BASE, seed=seed, num_threads=threads,
                 num_leaves=trial.suggest_categorical('num_leaves', [20, 30, 40, 50]),
                 learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1),
                 feature_fraction=trial.suggest_float('feature_fraction', 0.8, 1.0),
                 bagging_fraction=trial.suggest_float('bagging_fraction', 0.8, 1.0),
                 bagging_freq=trial.suggest_categorical('bagging_freq', [1, 5, 10]))
        mm = lgb.train(q, dtr, 1000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        trial.set_user_attr('rounds', max(int(mm.best_iteration), 1))
        return float(mm.best_score['valid_0']['binary_logloss'])
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=3 if test else n_trials)
    bp = dict(study.best_params); rounds = int(study.best_trial.user_attrs['rounds'])
    p = lgb.train(dict(LGB_BASE, seed=seed, num_threads=threads, **bp), dall, rounds).predict(X_f)
    t_lgb = time.time() - t1

    out = pd.DataFrame({'date': Da.P.date.values[i_f], 'stock_id': Da.P.stock_id.values[i_f], 'p': p, 'p_fixed': p_fixed, 'p_lstm': p_lstm})
    meta = dict(kind='floam', fold=fa, seed=seed, best_epoch=int(max(best_ep, 1)), rows=int(len(i_tr)), val_rows=int((~trm).sum()),
                n_params=int(sum(q.numel() for q in model.parameters())), lstm_sec=round(t_lstm, 1), lgb_sec=round(t_lgb, 1),
                train_sec=round(time.time() - t0, 1), pred_sec=0.0, pred_rows=int(len(out)), best_params=bp, best_rounds=rounds,
                best_val_logloss=float(study.best_value), fixed_rounds=fixed_rounds,
                fixed_val_logloss=float(m.best_score['valid_0']['binary_logloss']), n_trials=len(study.trials),
                trial_values=[float(t.value) for t in study.trials if t.value is not None], hp=hp,
                threads=int(threads), torch=torch.__version__, lightgbm=lgb.__version__, optuna=optuna.__version__,
                pandas=pd.__version__, test=test, finished=time.strftime('%Y-%m-%d %H:%M:%S'))
    return out, meta


def save_unit(f, out, meta):
    os.makedirs(os.path.dirname(f), exist_ok=True)
    tmp = f[:-4] + '.tmp'
    with open(tmp, 'wb') as fh:
        np.savez_compressed(fh, date=np.asarray(pd.to_datetime(out.date), dtype='datetime64[ns]').astype(np.int64),
                            sid=np.asarray(out.stock_id.astype(str).tolist(), dtype='<U8'),
                            p=out.p.to_numpy(np.float64), p_fixed=out.p_fixed.to_numpy(np.float64), p_lstm=out.p_lstm.to_numpy(np.float64),
                            meta=np.array(json.dumps(meta)))
    os.replace(tmp, f)


def load_member(col='p', seeds=(1, 2, 3, 4, 5), pdir=None, test=False):
    """seed-averaged walk-forward predictions of F-LOAM (col p), F-LOAM without Optuna (p_fixed) or its LSTM (p_lstm)"""
    parts = []
    for fa, _ in S.FOLDS:
        fs = [os.path.join(pdir, 'floam', f'{fa}_seed{s}.npz') if pdir else S.unit_file('floam', fa, s, test) for s in seeds]
        fs = [f for f in fs if os.path.exists(f)]
        if not fs:
            return None
        cols = []
        for f in fs:
            z = np.load(f, allow_pickle=True)
            idx = pd.MultiIndex.from_arrays([pd.to_datetime(z['date']).astype('datetime64[ns]'), np.asarray(z['sid']).astype(str)],
                                            names=['date', 'stock_id'])
            cols.append(pd.Series(np.asarray(z[col], np.float64), index=idx))
        parts.append(pd.concat(cols, axis=1).mean(axis=1))
    return pd.concat(parts).sort_index()

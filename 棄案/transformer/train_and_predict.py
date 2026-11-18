r"""
train_and_predict.py -- train the models and write submissions (layout: stock ids of sample_submission.csv as rows)

Static leaderboard (Jun-Sep 2026; models frozen at 2026-05-29, kept in models\):
  python train_and_predict.py                       # train (about 25 min) + predict all dates of sample_submission.csv
  python train_and_predict.py --use-saved           # reuse models\ (seconds), e.g. after update_data.py added new days
Live / daily leaderboard (separate models in models_live\, retrained on the newest data):
  python train_and_predict.py --models-dir models_live --train-end latest --dates auto      # weekly: retrain + predict
  python train_and_predict.py --models-dir models_live --use-saved --dates auto              # other days: predict only
  (--dates auto = the day(s) whose rows are all -1 in the downloaded data = the day to predict)
Check what would happen without torch:  python train_and_predict.py --dry-run [--dates auto]

Outputs (in this folder; --tag or, for --dates auto, the date is added to the file names)
  submission_ensemble*.csv      recommended: average probability of Transformer and MLP (best on 2022-2024 walk-forward)
  submission_transformer*.csv   Transformer only
  submission_sources*.csv       which dates were predicted by the model and which by the fallback rule
  submission_probabilities*.csv P(close > open) of each model
Steps
 1. features  : comp_features.build_comp_features() -> 67 features from the competition files only
 2. training  : Transformer (20-day window) and MLP, each an ensemble of --seeds models, trained on labelled rows
                2020-03-02 .. --train-end; scaling statistics from those rows only (no look-ahead)
 3. prediction: every requested date D whose previous trading day is in the data -> 1 if P(close > open) > 0.5
 4. fallback  : requested dates with no data before them -> 1 if the stock's mean intraday return over its
                last 250 trading days (up to the last data date) > 0, else 0   (best rule on 2022-2024)
Settings are the ones evaluated in walkforward_nn.py (no tuning on 2025-2026 or on the leaderboard period).
"""
import os, sys, json, argparse, time, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from comp_features import build_comp_features, placeholder_dates

HP = dict(d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.2, lr=1e-3, wd=1e-2, bs=1024, epochs=10, patience=2)
MODELS = {'transformer': dict(kind='transformer', seq_len=20), 'mlp': dict(kind='mlp', seq_len=1)}
TRAIN_START = pd.Timestamp('2020-03-02')

def load_bundle(f):
    import torch
    from tfmodel import make_model, Scaler
    try:
        ck = torch.load(f, weights_only=False)
    except TypeError:                                  # older torch without the weights_only argument
        ck = torch.load(f)
    keys = ('d_model', 'n_heads', 'n_layers', 'd_ff', 'dropout') if ck['kind'] == 'transformer' else ('dropout',)
    m = make_model(ck['kind'], len(ck['features']), ck['seq_len'], **{k: ck['hp'][k] for k in keys}); m.load_state_dict(ck['state_dict']); m.eval()
    return dict(model=m, scaler=Scaler.from_state(ck['scaler']), kind=ck['kind'], seq_len=ck['seq_len'], features=ck['features'], hp=ck['hp'], train_end=ck['train_end'])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(os.path.dirname(HERE), 'data'))
    ap.add_argument('--template', default=os.path.join(os.path.dirname(HERE), 'sample_submission.csv'))
    ap.add_argument('--models-dir', default='models', help='models (static, frozen at 2026-05-29) or models_live')
    ap.add_argument('--train-end', default='2026-05-29', help="last training date, or 'latest' = last complete day in the data")
    ap.add_argument('--dates', default=None, help="comma-separated dates, 'auto' (= placeholder day in the data), default: template columns")
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--use-saved', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--tag', default='', help='suffix for output files')
    ap.add_argument('--threads', type=int, default=0, help='torch threads (0 = all cores)')
    a = ap.parse_args(); t0 = time.time()

    tpl = pd.read_csv(a.template, dtype={'id': str}); stocks = tpl['id'].tolist()
    if a.dates == 'auto':
        dates = placeholder_dates(a.data)
        if not dates:
            sys.exit('no placeholder (-1) day in the data: run update_data.py with today\'s download first, or pass --dates')
        a.tag = a.tag or dates[-1].strftime('%m%d')
    else:
        dates = [pd.Timestamp(x) for x in (a.dates.split(',') if a.dates else [c for c in tpl.columns if c != 'id'])]
    feats = json.load(open(os.path.join(HERE, 'final_feature_lists.json')))['B_competition_only']
    P = build_comp_features(a.data, dates)
    last_data = P[P.target_r.notna()].date.max()
    mdir = os.path.join(HERE, a.models_dir); os.makedirs(mdir, exist_ok=True)
    if a.use_saved:
        f0 = sorted(glob.glob(os.path.join(mdir, '*_seed*.pt')))
        if not f0:
            sys.exit(f'no saved models in {mdir}: run once without --use-saved')
        import torch
        try:
            ends = {torch.load(f, weights_only=False)['train_end'] for f in f0}
        except TypeError:
            ends = {torch.load(f)['train_end'] for f in f0}
        if len(ends) != 1:
            sys.exit(f'models in {mdir} were trained up to different dates {ends}: retrain')
        train_end = pd.Timestamp(ends.pop())
    else:
        train_end = last_data if a.train_end == 'latest' else pd.Timestamp(a.train_end)
    if train_end > last_data:
        sys.exit(f'train end {train_end.date()} is after the last labelled date {last_data.date()}')
    have = set(P.date.unique()); model_dates = [d for d in dates if d in have and d > train_end]
    print(f'[{time.time()-t0:.0f}s] features: {P.shape[0]} rows; labelled data up to {last_data.date()}; models trained up to {train_end.date()}; '
          f'{len(model_dates)} of {len(dates)} requested dates can use the model', flush=True)
    if a.dry_run:
        fb_dates = [d.strftime('%Y-%m-%d') for d in dates if d not in model_dates]
        print('dry run: model dates', [d.strftime('%Y-%m-%d') for d in model_dates][:5], '...' if len(model_dates) > 5 else '',
              '| fallback dates', fb_dates[:10], '...' if len(fb_dates) > 10 else '')
        return

    import torch
    from tfmodel import train_model, predict
    torch.set_num_threads(a.threads if a.threads > 0 else max(1, os.cpu_count() or 1))
    B = {}
    for name, cfg in MODELS.items():
        B[name] = []
        for s in range(1, a.seeds + 1):
            f = os.path.join(mdir, f'{name}_seed{s}.pt')
            if a.use_saved:
                b = load_bundle(f)
            else:
                b = train_model(cfg['kind'], P, feats, lambda d: (d.date >= TRAIN_START) & (d.date <= train_end), cfg['seq_len'], seed=s, hp=HP)
                torch.save(dict(state_dict=b['model'].state_dict(), scaler=b['scaler'].state(), kind=b['kind'], seq_len=b['seq_len'],
                                features=b['features'], hp=b['hp'], train_end=str(train_end.date()), best_epoch=b['best_epoch']), f)
                print(f'[{time.time()-t0:.0f}s] trained {name} seed {s} (epochs {b["best_epoch"]})', flush=True)
            B[name].append(b)

    probs = pd.DataFrame(columns=['date', 'stock_id'])
    if model_dates:
        sel = lambda d: d.date.isin(model_dates) & d.stock_id.isin(stocks)
        for name, bs in B.items():
            p = pd.concat([predict(b, P, sel).set_index(['date', 'stock_id']).p for b in bs], axis=1).mean(axis=1).rename(f'p_{name}')
            probs = p.reset_index() if probs.empty else probs.merge(p.reset_index(), on=['date', 'stock_id'], how='outer')
        probs['p_ensemble'] = (probs.p_transformer + probs.p_mlp) / 2
    lab = P[P.target_r.notna()].sort_values('date')
    fb = (lab.groupby('stock_id').target_r.apply(lambda s: s.tail(250).mean()) > 0).astype(int).reindex(stocks).fillna(0).astype(int)

    suf = f'_{a.tag}' if a.tag else ''
    src = {d.strftime('%Y-%m-%d'): ('model' if d in model_dates else 'fallback: last-250-day mean intraday return > 0') for d in dates}
    for name in ['ensemble', 'transformer']:
        sub = pd.DataFrame({'id': stocks})
        for d in dates:
            col = d.strftime('%Y-%m-%d')
            if d in model_dates:
                p = probs[probs.date == d].set_index('stock_id')[f'p_{name}'].reindex(stocks)
                sub[col] = np.where(p.isna(), fb.values, (p > 0.5).astype(int).values).astype(int)
            else:
                sub[col] = fb.values
        out = os.path.join(HERE, f'submission_{name}{suf}.csv'); sub.to_csv(out, index=False)
        print(f'wrote {out}: {sub.shape[0]} stocks x {sub.shape[1]-1} dates, share of 1 = {sub.iloc[:, 1:].values.mean():.3f}')
    pd.DataFrame({'date': list(src), 'source': list(src.values())}).to_csv(os.path.join(HERE, f'submission_sources{suf}.csv'), index=False)
    if not probs.empty:
        probs.sort_values(['date', 'stock_id']).to_csv(os.path.join(HERE, f'submission_probabilities{suf}.csv'), index=False)
    n = sum(v == 'model' for v in src.values())
    print(f'[{time.time()-t0:.0f}s] dates predicted by the model: {n}; by the fallback rule (no data yet): {len(src) - n}')

if __name__ == '__main__':
    main()

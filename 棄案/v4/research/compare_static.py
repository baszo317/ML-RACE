r"""
research\compare_static.py -- what the walk-forward changes on the static months (reference only, NOT for tuning)

  python research\compare_static.py            (from the v4 folder, after train_and_predict_v4.py --static)

Uses the models and predictions already in models_v4_static\ (nothing is trained) and the labels in data\ to score:
  frozen       all members = the 2026-05-29 models only (= v2 static) [with / without the new ex-dividend dates]
  rolling      v4 default (every day uses models trained up to the previous trading day)
  variants     rolling MLP + frozen Transformer, MLP only, v1 rule, other w_T
AR = competition score in bp/day (Top50 mean of (2p-1) * (close-open)/open), averaged over the dates.
timing = part of AR from the share of 1s times the market move; xsec = the rest (stock picking).
t = paired t-stat of the daily difference against the first row.
The static months were also looked at when v2 was designed (README_v2.md, 6.), so none of these numbers is a clean
out-of-sample test; the v4 settings were fixed before this table was computed and are not chosen from it.
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); V4 = os.path.dirname(HERE); sys.path.insert(0, V4)
import ens_lib as E
import rolling as R
import train_and_predict_v4 as T

def score(P, ens, exdiv=True):
    x = ens.reset_index().merge(P[['date', 'stock_id', 'top50', 'target_r', 'ex_today']], on=['date', 'stock_id'])
    x = x[x.top50 & x.target_r.notna()]
    pred = (x.score > 0).astype(int) if 'score' in x else x.pred
    if exdiv:
        pred = E.apply_exdiv(pred, x.ex_today)
    s = 2 * np.asarray(pred) - 1
    x = x.assign(s=s, v=s * x.target_r)
    S = x.groupby('date').v.mean() * 1e4
    Tm = x.groupby('date').s.mean() * x.groupby('date').target_r.mean() * 1e4
    return S, Tm, x.groupby('date').s.mean().add(1).div(2)

def main():
    P = E.build_panel(E.DATA)
    P['date'] = P['date'].astype('datetime64[ns]')            # one datetime resolution everywhere (pandas 2 and 3)
    tpl = pd.read_csv(os.path.join(E.ROOT, 'sample_submission.csv'), dtype={'id': str})
    dates = [pd.Timestamp(c) for c in tpl.columns if c != 'id']
    root = os.path.join(V4, 'models_v4_static'); sched = {k: dict(v) for k, v in R.DEFAULT['static'].items()}
    members = [dict(m) for m in T.MEMBERS]
    units, use, prev = R.static_plan(P, members, dates, sched)
    missing = [u for u in units if not os.path.exists(R.pred_file(root, u['m'], u['te'], u['seed']))]
    if missing:
        sys.exit(f'{len(missing)} models not finished yet: run train_and_predict_v4.py --static first')
    roll = R.static_probs(root, members, units, use, dates)
    froz = R.static_probs(root, members, units, use, dates, frozen=True)
    mlp = [m for m in members if m['kind'] == 'mlp']
    mixed = dict(roll); mixed['tf_raw_B'] = froz['tf_raw_B']
    plain = lambda pr, mm: E.combine(pr, [m for m in mm if m['target'] == 'raw'])
    rows = [('frozen 2026-05-29 (= v2 static), old ex-dividend table', E.combine(froz, members), 'old'),
            ('frozen 2026-05-29, ex-dividend incl. 2026/08-09', E.combine(froz, members), True),
            ('v4 rolling (default)', E.combine(roll, members), True),
            ('v4 rolling MLP + frozen Transformer', E.combine(mixed, members), True),
            ('v4 rolling, MLP members only', E.combine({k: v for k, v in roll.items() if k != 'tf_raw_B'}, mlp), True),
            ('v4 rolling, v1 rule (raw members, p > 0.5)', plain(roll, members), False),
            ('frozen, v1 rule (raw members, p > 0.5)', plain(froz, members), False)]
    for w in (0.0, 0.25, 0.75, 1.0):
        rows.append((f'v4 rolling, w_T = {w}', E.combine(roll, members, w_t=w), True))
    old_ex = P.copy()
    hist = pd.read_csv(os.path.join(E.DATA, 'ref', 'exdiv_history.csv'), dtype={'stock_id': str}, parse_dates=['date'])
    new_only = hist[hist.date > '2026-07-31'][['stock_id', 'date']].assign(_n=1)
    old_ex = old_ex.merge(new_only, on=['stock_id', 'date'], how='left')
    old_ex.loc[old_ex._n == 1, 'ex_today'] = 0.0
    out, base = [], None
    for label, ens, ex in rows:
        S, Tm, sh = score(old_ex if ex == 'old' else P, ens, exdiv=(ex is not False))
        if base is None:
            base = S
        d = (S - base).dropna()
        r = {'config': label, 'AR': S.mean(), 'timing': Tm.mean(), 'xsec': (S - Tm).mean(), 'share1': sh.mean(), 'sd_day': S.std(),
             't_vs_first': d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else np.nan}
        for mo, v in S.groupby(S.index.month).mean().items():
            r[f'{mo:02d}'] = v
        out.append(r)
    Tb = pd.DataFrame(out); pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    txt = ['AR in bp/day, Top50, static dates 2026-06-01..09-30 (84 days). Reference only (see the docstring).', '',
           Tb.round(2).to_string(index=False)]
    open(os.path.join(HERE, 'compare_static.txt'), 'w', encoding='utf-8').write('\n'.join(txt) + '\n')
    print('\n'.join(txt))

if __name__ == '__main__':
    main()

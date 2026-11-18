r"""
guards.py -- checks that stop the morning run before a bad file is uploaded (v5)

One wrong day costs about as much as a whole model improvement (README_v5, 4.3): the average day earns ~13 bp, and a
day predicted from stale data or without the ex-dividend table can lose that and more. Every check below compares the
run with facts that are known before 09:00; none of them looks at returns that are not yet known.

  check_inputs   : (1) the newest labelled day in data\ is the trading day right before the day to predict
                       (official calendar data\ref\calendar.parquet); (2) every Top100 stock has a row for that day;
                   (3) the U.S. session used by the features is not older than one U.S. session;
                   (4) no B feature is missing for more than 20 % of the stocks on the day to predict
  check_exdiv    : fetch_exdiv.py succeeded today (data\ref\exdiv_fetch_log.csv)
  check_pool     : the newest model of every member was trained up to the previous trading day
  check_output   : share of 1 among the Top50 inside the historical range of the same rule
                   (walk-forward 2022-01..2026-10: 0.5 % quantile 0.30, 99.5 % quantile 0.98; static 06-09 min 0.22),
                   and pbar inside [0.20, 0.80] (walk-forward min 0.28, max 0.74)
Failures are collected; FATAL ones stop the run unless --force is given (use it only after you have looked).
"""
import os, json
import numpy as np, pandas as pd

SHARE1_RANGE = (0.20, 1.00)        # walk-forward min 0.22 (1 day), 0.5 % quantile 0.30; widened so only broken runs trip it
PBAR_RANGE = (0.20, 0.80)          # walk-forward min 0.278, max 0.739; wide on purpose (catches broken runs only)
MAX_NAN_SHARE = 0.20

class Report:
    def __init__(self):
        self.items = []
    def add(self, level, msg):
        self.items.append((level, msg))
    def fatal(self):
        return [m for l, m in self.items if l == 'FATAL']
    def show(self, log=print):
        for l, m in self.items:
            log(f'[{l}] {m}')

def trading_calendar(data_dir):
    f = os.path.join(data_dir, 'ref', 'calendar.parquet')
    try:
        return pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(f).date)).sort_values()
    except Exception:
        return None

def check_inputs(P, pred_dates, feats, data_dir, rep):
    lab = pd.DatetimeIndex(np.sort(P.loc[P.target_r.notna(), 'date'].unique()))
    cal = trading_calendar(data_dir)
    info = json.load(open(os.path.join(data_dir, 'info.json')))
    for d in pred_dates:
        d = pd.Timestamp(d)
        if cal is not None and d not in set(cal):
            rep.add('FATAL', f'{d.date()} is not a trading day in data\\ref\\calendar.parquet')
            continue
        prev = cal[cal < d][-1] if cal is not None and (cal < d).any() else None
        last = lab[lab < d][-1] if (lab < d).any() else None
        if last is None:
            rep.add('FATAL', f'day to predict {d.date()}: no labelled day before it')
        elif cal is None:                               # calendar unreadable (pyarrow missing): weaker check
            gap = (d - last).days
            if gap > 5:
                rep.add('FATAL', f'day to predict {d.date()}: newest labelled day is {last.date()} ({gap} calendar days earlier) '
                                 f'-> the newest zip is probably not merged (run update_data.py)')
            else:
                rep.add('WARN', f'{d.date()}: cannot read data\\ref\\calendar.parquet (pip install pyarrow); newest labelled day '
                                f'{last.date()} is {gap} calendar day(s) earlier -- check it is the previous trading day')
        elif last != prev:
            rep.add('FATAL', f'day to predict {d.date()}: previous trading day is {prev.date()} but the newest labelled day '
                             f'in data\\ is {last.date()} -> the newest zip is not merged (run update_data.py)')
        else:
            rep.add('OK', f'{d.date()}: newest labelled day {last.date()} = previous trading day')
        if last is not None:
            n = P[(P.date == last) & P.target_r.notna()].stock_id.nunique()
            if n < len(info['Top100@Y']):
                rep.add('FATAL', f'{last.date()}: only {n} of {len(info["Top100@Y"])} stocks have a result')
        row = P[P.date == d]
        if row.empty:
            rep.add('FATAL', f'{d.date()}: no feature rows (placeholder day missing in data\\market)')
            continue
        if bool(row.get('stale_us', pd.Series(False)).any()):
            rep.add('WARN', f'{d.date()}: U.S. features are one session old (global_data.csv lacks the last U.S. close)')
        miss = row[feats].isna().mean()
        bad = miss[miss > MAX_NAN_SHARE]
        if len(bad):
            rep.add('FATAL', f'{d.date()}: features missing for > {MAX_NAN_SHARE:.0%} of stocks: ' +
                             ', '.join(f'{k} {v:.0%}' for k, v in bad.head(8).items()))
        else:
            rep.add('OK', f'{d.date()}: {len(feats)} input features present (largest missing share {miss.max():.0%})')
    return rep

def check_exdiv(data_dir, rep, today=None):
    f = os.path.join(data_dir, 'ref', 'exdiv_fetch_log.csv')
    today = pd.Timestamp.now().normalize() if today is None else pd.Timestamp(today)
    if not os.path.exists(f):
        rep.add('FATAL', 'data\\ref\\exdiv_fetch_log.csv missing: run fetch_exdiv.py (or add today\'s ex-dates to exdiv_manual.csv and use --force)')
        return rep
    L = pd.read_csv(f, parse_dates=['run_at'])
    ok = L[(L.status == 'ok') & (L.run_at.dt.normalize() == today)]
    if ok.empty:
        last = L[L.status == 'ok'].run_at.max()
        rep.add('FATAL', f'ex-dividend announcements not fetched today (last success {last}); run fetch_exdiv.py')
    else:
        rep.add('OK', f'ex-dividend table fetched today ({int(ok.n_top100.iloc[-1])} Top100 announcements)')
    return rep

def check_pool(scan, members, prev_day, rep):
    """scan(m) -> {train_end: [seeds]}"""
    for m in members:
        pts = scan(m)
        if not pts:
            rep.add('FATAL', f"{m['name']}: no model in the pool"); continue
        newest = max(pts)
        if pd.Timestamp(newest) >= pd.Timestamp(prev_day):
            rep.add('OK', f"{m['name']}: newest model trained up to {pd.Timestamp(newest).date()}")
        else:
            rep.add('WARN', f"{m['name']}: newest model trained up to {pd.Timestamp(newest).date()}, previous trading day is "
                            f"{pd.Timestamp(prev_day).date()} (fine with --retrain none; otherwise training failed)")
    return rep

def check_output(ens, top50, rep):
    for d, g in ens[ens.stock_id.isin(top50)].groupby('date'):
        s1 = g.pred.mean(); pb = g.pbar.iloc[0]; n0 = len(rep.fatal())
        if not (SHARE1_RANGE[0] <= s1 <= SHARE1_RANGE[1]):
            rep.add('FATAL', f'{pd.Timestamp(d).date()}: share of 1 = {s1:.2f}, outside the historical range {SHARE1_RANGE}')
        if not (PBAR_RANGE[0] <= pb <= PBAR_RANGE[1]):
            rep.add('FATAL', f'{pd.Timestamp(d).date()}: market view pbar = {pb:.3f}, outside the historical range {PBAR_RANGE}')
        if len(g) != len(top50):
            rep.add('FATAL', f'{pd.Timestamp(d).date()}: {len(g)} of {len(top50)} Top50 stocks predicted')
        rep.add('OK' if len(rep.fatal()) == n0 else 'INFO', f'{pd.Timestamp(d).date()}: share of 1 = {s1:.2f}, pbar = {pb:.3f}, {len(g)} Top50 stocks')
    return rep

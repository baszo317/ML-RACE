r"""
v11f_eval.py -- v11 part F (research\PREREG_v11f.md, written 2026-10-27 23:00 before any result): the two changes suggested
by the 10/27 diagnosis, on the same walk-forward as v11 part B (2022-01 .. 2026-10-22, 1,164 days, Top50 bp/day).

  F1  no T2 after a break: if any weekday between the previous trading day and D had no session (holiday, typhoon day),
      the T2 night-session term of D is 0 (the pre-break night session does not cover the U.S. sessions of the break)
  F2  Top50 half and half: ex-dividend-rule stocks 0 as before; among the others the floor(n/2) highest stock-view scores
      (p_e with X1) get 1, the rest 0 -> net exposure 0, timing 0; no beta hedge, no T3

Settings: v9; v11 (defaults w_T 0.5, lambda 0.75, T2 0.5, T3 5, X1); W0 = the setting uploaded on 10/27 (v11 with w_T 0,
lambda 0.5). Rules (PREREG_v11f.md 3): F1 goes into the daily v11 if its mean difference to v11 over the affected days is
> 0; F2 becomes a --balanced option in any case and is recommended over W0 only if its AR >= W0 in val, hold and post.

  python v11f_eval.py            -> ..\results\v11f_eval.txt (also printed), ..\cache\v11f_daily.pkl
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import v11b_lib as L
V = L.V
V11 = os.path.dirname(HERE)
OUT = os.path.join(V11, 'results', 'v11f_eval.txt'); DAILY = os.path.join(V11, 'cache', 'v11f_daily.pkl')
PER = {'val 2022-24': ('2022-01-01', '2025-01-01'), 'hold 2025-26/05': ('2025-01-01', '2026-06-01'),
       'post 2026-06..10-22': ('2026-06-01', '2026-10-23')}
REP = {'Jun-Sep 2026': ('2026-06-01', '2026-10-01'), 'Oct 2026': ('2026-10-01', '2026-10-23'), 'all': ('2022-01-01', '2026-10-23')}


def stale_days(cal):
    """trading days D with at least one weekday (Mon-Fri) without a session between the previous trading day and D"""
    cal = pd.DatetimeIndex(sorted(cal)); out = set()
    for a, b in zip(cal[:-1], cal[1:]):
        between = pd.date_range(a + pd.Timedelta(days=1), b - pd.Timedelta(days=1))
        if (between.dayofweek < 5).any():
            out.add(b)
    return out


def run(data, adj, pe, w_t, night_w, extra, lam):
    """= v11_try.run: pbar' = pbar - night_w * adj; score = w_T (pbar' - 0.5) + p_e; v11 decision (ex-dividend, T3, hedge)"""
    pb = data.pbar() - night_w * adj
    s = w_t * (pb.reindex(pe.index.get_level_values('date')).values - 0.5) + pe
    L.EXTRA_ONES = int(extra)
    return data.decide(s, month_start=extra > 0, lam=lam)


def balanced(data, pe):
    """F2: per day, among the Top50 rows outside the ex-dividend rule, the floor(n/2) highest scores get 1"""
    X = data.INFO.join(pe.rename('score'), how='inner')
    rows = []
    for d, g in X.groupby(level='date'):
        sc, r = g.score.values, g.target_r.values
        ex = np.zeros(len(g), bool)
        for col in V.EX4:
            ex |= g[col].values == 1
        k = len(g) // 2
        s = -np.ones(len(g))
        ok = np.flatnonzero(~ex)
        top = ok[np.argsort(-sc[ok], kind='stable')[:k]]
        s[top] = 1
        m = r.mean()
        rows.append((d, (s * r).mean() * 1e4, s.mean() * m * 1e4, m * 1e4, s.mean(), len(g), int(ex.sum())))
    D = pd.DataFrame(rows, columns=['date', 'AR', 'timing', 'm', 'sbar', 'n', 'n_ex']).set_index('date')
    D['xsec'] = D.AR - D.timing
    return D


def cut(x, a, b):
    return x[(x.index >= a) & (x.index < b)]


def stats(D, a, b):
    x = cut(D.AR, a, b)
    r19 = cut(D.AR.rolling(19).mean(), a, b).dropna()
    q1, q5 = x.quantile(0.01), x.quantile(0.05)
    return {'days': len(x), 'AR': x.mean(), 'sd': x.std(), 'AR/sd': x.mean() / x.std(), 'worst1%': x[x <= q1].mean(),
            'worst5%': x[x <= q5].mean(), 'P(19d>0)': (r19 > 0).mean(), 'P(19d>20)': (r19 > 20).mean(),
            'timing': cut(D.timing, a, b).mean(), 'xsec': cut(D.xsec, a, b).mean()}


def main():
    t0 = time.time(); lines = []
    say = lambda s='': (print(s, flush=True), lines.append(s))
    data = L.Data()
    pb = data.pbar(); sd = pb.rolling(250, min_periods=20).std().shift(1)
    adj = (sd * data.z_night.reindex(pb.index)).fillna(0.0)
    cal = sorted(set(pd.to_datetime(data.P.date.unique())))
    stale = stale_days(cal)
    adj_f1 = adj.where(~adj.index.isin(stale), 0.0)
    pe = {False: data.stock_view(), True: data.stock_view(x1=True)}
    runs = {'v9': run(data, adj, pe[False], 0.5, 0.0, 0, 0.75),
            'v11': run(data, adj, pe[True], 0.5, 0.5, 5, 0.75),
            'v11 + F1': run(data, adj_f1, pe[True], 0.5, 0.5, 5, 0.75),
            'W0 (10/27 upload)': run(data, adj, pe[True], 0.0, 0.5, 5, 0.5),
            'W0 + F1': run(data, adj_f1, pe[True], 0.0, 0.5, 5, 0.5),
            'F2 (half and half)': balanced(data, pe[True])}
    days = runs['v11'].index
    aff = sorted(d for d in days if d in stale)
    say('v11 part F (research\\PREREG_v11f.md, 2026-10-27 23:00); walk-forward of v11 part B, Top50 bp/day')
    say(f'days {len(days)} ({days.min().date()} .. {days.max().date()}); F1 affects {len(aff)} days: ' +
        ', '.join(f'{k} {sum(1 for d in aff if a <= str(d.date()) < b)}' for k, (a, b) in PER.items()))
    nz = adj.reindex(aff).fillna(0.0)
    say(f'  of them T2 was non-zero (night session present) on {int((nz != 0).sum())}; '
        f'mean |T2 shift of pbar| there {0.5 * nz.abs()[nz != 0].mean():.4f} (all days {0.5 * adj.reindex(days).abs().mean():.4f})')
    say('')

    # ------------------------------------------------------------------ 1. every setting
    say('1. each setting (bp/day; worst = mean of the worst 1 % / 5 % days; P(19d) = share of 19-day windows)')
    for pk, (a, b) in {**PER, **REP}.items():
        rows = []
        for nm, D in runs.items():
            r = {'setting': nm}; r.update(stats(D, a, b)); rows.append(r)
        T = pd.DataFrame(rows).set_index('setting')
        say(f'\n  {pk}'); say(T.round(2).to_string())
    say('')

    # ------------------------------------------------------------------ 2. F1
    d = (runs['v11 + F1'].AR - runs['v11'].AR).reindex(days).fillna(0.0)
    da = d.reindex(aff)
    se = da.std() / np.sqrt(len(da)) if len(da) > 1 else np.nan
    say('2. F1 minus v11 (bp/day)')
    say(f'  over the {len(aff)} affected days (the rule): mean {da.mean():+.2f}, sd {da.std():.2f}, iid t {da.mean() / se:.2f}, '
        f'positive on {int((da > 0).sum())}, negative on {int((da < 0).sum())}, zero on {int((da == 0).sum())}')
    for k, (a, b) in PER.items():
        x = cut(da, a, b)
        say(f'  {k}: affected days {len(x)}, mean {x.mean():+.2f}; per day over the whole period {cut(d, a, b).mean():+.3f}')
    say(f'  whole walk-forward, per day: {d.mean():+.3f} (NW t {V.nw_t(d.values):.2f})')
    w0 = (runs['W0 + F1'].AR - runs['W0 (10/27 upload)'].AR).abs().max()
    say(f'  check: W0 + F1 equals W0 (w_T 0 has no T2): largest daily difference {w0:.2e}')
    tm = (runs['v11 + F1'].timing - runs['v11'].timing).reindex(aff)
    say(f'  affected days: timing change {tm.mean():+.2f}, xsec change {(da - tm).mean():+.2f}')
    pos = da.mean() > 0
    say(f'  RULE: mean over the affected days > 0 -> {"F1 goes into the daily v11" if pos else "F1 is not added; v11 unchanged"}')
    say('')

    # ------------------------------------------------------------------ 3. F2
    say('3. F2 (half and half) minus W0 and minus v11 (bp/day, NW t)')
    ok = True
    for ref in ('W0 (10/27 upload)', 'v11', 'v9'):
        dd = (runs['F2 (half and half)'].AR - runs[ref].AR).dropna()
        parts = []
        for k, (a, b) in {**PER, **REP}.items():
            x = cut(dd, a, b)
            parts.append(f'{k} {x.mean():+.2f} ({V.nw_t(x.values):.2f})')
            if ref.startswith('W0') and k in PER and x.mean() < 0:
                ok = False
        say(f'  minus {ref}: ' + '; '.join(parts))
    F2 = runs['F2 (half and half)']
    say(f'  F2 check: largest |net exposure| {F2.sbar.abs().max():.3f} (0 when n is even), days with n != 50: {int((F2.n != 50).sum())}, '
        f'days with ex-dividend-rule stocks: {int((F2.n_ex > 0).sum())}')
    corr = np.corrcoef(runs['W0 (10/27 upload)'].AR.reindex(days), F2.AR.reindex(days))[0, 1]
    say(f'  daily correlation F2 vs W0 {corr:.2f}; vs v11 {np.corrcoef(runs["v11"].AR.reindex(days), F2.AR.reindex(days))[0, 1]:.2f}')
    say(f'  RULE: AR >= W0 in val, hold and post -> {"recommend F2 instead of W0 as the low-risk upload" if ok else "not recommended over W0; F2 stays an option (--balanced)"}')
    say('')

    # ------------------------------------------------------------------ 4. market-direction days
    say('4. by the day\'s market move m (Top50 mean intraday return), all 1,164 days (report only)')
    m = runs['v11'].m.reindex(days)
    bins = [(-1e9, -100), (-100, -50), (-50, 0), (0, 50), (50, 100), (100, 1e9)]
    rows = []
    for lo, hi in bins:
        sel = (m > lo) & (m <= hi)
        r = {'m bin (bp)': f'{lo if lo > -1e8 else "-inf"}..{hi if hi < 1e8 else "inf"}', 'days': int(sel.sum())}
        for nm in ('v11', 'W0 (10/27 upload)', 'F2 (half and half)'):
            r[nm] = runs[nm].AR.reindex(days)[sel].mean()
        rows.append(r)
    say(pd.DataFrame(rows).round(2).to_string(index=False))
    say('')
    say(f'done in {time.time() - t0:.0f}s')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    pd.to_pickle({k: v for k, v in runs.items()}, DAILY)
    print(f'written {OUT}')


if __name__ == '__main__':
    main()

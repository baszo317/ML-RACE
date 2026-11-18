"""s13: Taiwan 50 index review days (step 1 of section 3 of the previous plan: no new data needed).
FTSE TWSE Taiwan 50 changes take effect after the close of the third Friday of March, June, September and December, so
0050 and other trackers trade the changes in that day's closing auction. Question: is the Top50 intraday return on
those days systematically different (sign in every year)? Only then would a rule be worth the stock-level work
(historical add / delete lists). The 2026 December review day is 12/18, inside the private test period.
Also: how did the v4 rule do on those days (walk-forward 2022-01..2026-10, wf_v4_frame.pkl from s4)?"""
import os, numpy as np, pandas as pd, rlib as L
P = L.panel()
lab = P[P.target_r.notna() & P.top50]
days = pd.DatetimeIndex(np.sort(lab.date.unique()))
ev = []
for y in range(2020, 2027):
    for m in (3, 6, 9, 12):
        fr = pd.date_range(f'{y}-{m:02d}-01', periods=31, freq='D')
        fr = fr[(fr.month == m) & (fr.weekday == 4)]
        if len(fr) >= 3 and fr[2] in days:
            ev.append(fr[2])
ev = pd.DatetimeIndex(ev)
g = lab.groupby('date').agg(m=('target_r', 'mean'), p1=('y', 'mean')) if 'y' in lab else lab.groupby('date').agg(m=('target_r', 'mean'))
g['m'] *= 1e4
is_ev = g.index.isin(ev)
print(f'{len(ev)} review days with data ({ev[0].date()} .. {ev[-1].date()})')
print(f"Top50 mean intraday return: review days {g.m[is_ev].mean():+.1f} bp (sd {g.m[is_ev].std():.1f}), other days {g.m[~is_ev].mean():+.1f} bp; "
      f"t (review days vs 0) = {L.plain_t(g.m[is_ev].values):+.2f}; positive on {int((g.m[is_ev] > 0).sum())} of {is_ev.sum()}")
if 'p1' in g:
    print(f"share of Top50 stocks with close > open: review days {g.p1[is_ev].mean():.3f}, other days {g.p1[~is_ev].mean():.3f}")
print('by year (mean of the review days, bp):')
print(g.m[is_ev].groupby(g.index[is_ev].year).agg(['mean', 'count']).round(1).to_string())
X = pd.read_pickle(os.path.join(L.CACHE, 'wf_v4_frame.pkl'))
D = L.daily(X.set_index(['date', 'stock_id'])[['score']])
e = D.index.isin(ev)
print(f"\nv4 rule (walk-forward 2022-01..2026-10): review days AR {D.AR[e].mean():+.2f} (n = {e.sum()}, timing {D.timing[e].mean():+.2f}, "
      f"xsec {D.xsec[e].mean():+.2f}); other days AR {D.AR[~e].mean():+.2f}; difference t = "
      f"{(D.AR[e].mean() - D.AR[~e].mean()) / np.sqrt(D.AR[e].var() / e.sum() + D.AR[~e].var() / (~e).sum()):+.2f}")

"""W0 (v11, w_T 0, lambda 0.5) daily AR 2026-10-01..11-05 from the upload-format files, prices = adj_open / adj_close of
today's course zip (data (9).zip); same formula as v6\\phase2_check.py and rerun_1030\\scripts\\ar_check.py.

  python final_table.py OUTDIR
"""
import io, os, sys, zipfile
import numpy as np, pandas as pd
W = '/tmp/claude-0/-home-claude/6a2b61c3-80a1-5e4b-95c2-bb732ce3f64e/scratchpad'
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
z = zipfile.ZipFile(f'{W}/base/downloads/data (9).zip')
stocks = pd.read_csv(f'{W}/base/sample_submission.csv', dtype={'id': str})['id'].tolist()
R, Y = {}, {}
for s in stocks:
    df = pd.read_csv(io.BytesIO(z.read(f'data/market/{s}.csv')))
    df = df[df['etl:adj_open'] != -1].set_index('date')
    R[s] = (df['etl:adj_close'] - df['etl:adj_open']) / df['etl:adj_open']; Y[s] = df['y']
R = pd.DataFrame(R); Y = pd.DataFrame(Y)

src = {}
S = pd.read_csv(f'{W}/runs/oct_w0/submission_w0_october_asof.csv', dtype={'id': str}).set_index('id')
for c in S.columns:
    src[c] = ('as-of rebuild (W0 had no file yet)', 'submission_w0_october_asof.csv', S[c])
live = {'2026-10-27': 'v11/out/submission_v9.4_1027_wt0_lam0.5.csv', '2026-10-28': 'v11/out/submission_v11_1028_wt0_lam0.5.csv',
        '2026-10-29': 'v11/out/submission_v11_1029_wt0_lam0.5.csv', '2026-10-30': 'v11/out/submission_v11_1030_wt0_lam0.5.csv',
        '2026-11-02': 'v11/out/submission_v11_1102_wt0_lam0.5.csv', '2026-11-03': 'v11/out/submission_v11_1103_wt0_lam0.5.csv'}
for d, f in live.items():
    x = pd.read_csv(f'{W}/base/{f}', dtype={'id': str}).set_index('id'); assert list(x.columns) == [d], f
    src[d] = ('your file' + (' (made 11/1 for 10/30)' if d == '2026-10-30' else ''), f, x[d])
for d, t in (('2026-11-04', '1104'), ('2026-11-05', '1105')):
    f = f'v11/out/submission_v11_{t}_wt0_lam0.5.csv'
    x = pd.read_csv(f'{W}/asof/{f}', dtype={'id': str}).set_index('id'); assert list(x.columns) == [d], f
    src[d] = ('as-of rerun of daily_v11 + custom_v11 (no file that day)', f'submission_v11_{t}_wt0_lam0.5.csv', x[d])

rows = []
for d in sorted(src):
    how, f, p = src[d]
    p = p.reindex(stocks).astype(float); r = R.loc[d, stocks]; s = 2 * p - 1
    prev = Y.loc[:d].iloc[-2, :].reindex(stocks)
    ar = (s * r).mean(); tim = s.mean() * r.mean()
    rows.append(dict(date=d, source=how, file=f, AR=ar, AR_bp=ar * 1e4, timing_bp=tim * 1e4, xsec_bp=(ar - tim) * 1e4, share1=p.mean(),
                     hits=int(((p == 1) == (r > 0)).sum()), mkt_bp=r.mean() * 1e4, always_down_bp=-r.mean() * 1e4,
                     prev_day_bp=((2 * prev - 1) * r).mean() * 1e4))
T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, 'w0_daily_ar.csv'), index=False, float_format='%.6f')
pd.set_option('display.width', 250)
print(T.drop(columns=['file', 'AR']).round(2).to_string(index=False))

def per(name, m):
    x = T[m]
    print(f'{name:<34s} {len(x):2d} d  W0 total {x.AR.sum():+.6f} ({x.AR_bp.sum():+8.2f} bp)  avg {x.AR_bp.mean():+7.2f}  sd {x.AR_bp.std():6.2f}  '
          f'| Always down avg {x.always_down_bp.mean():+7.2f}  Previous day avg {x.prev_day_bp.mean():+7.2f}  | days > 0: {(x.AR > 0).sum()}')
d = T.date
print()
per('10/01-10/22 (as-of rebuild)', d <= '2026-10-22')
per('10/27-11/05 (after W0 was chosen)', d >= '2026-10-27')
per('  of which your own files', T.source.str.startswith('your'))
per('October 10/01-10/30', d <= '2026-10-31')
per('November 11/02-11/05', d >= '2026-11-01')
per('all 10/01-11/05', d >= '2026-10-01')

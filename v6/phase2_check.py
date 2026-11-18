r"""
phase2_check.py -- Phase 2 (dynamic leaderboard, scored every trading day from 2026-10-12): fetch the day's open and close
of the 50 stocks by itself and score live submissions with the competition metric, next to the board's two baselines,
so that the AR can be checked against the daily board and the accumulated board without waiting for the course zip.

  python phase2_check.py                       newest day with prices: every one-day submission file in v9\out, v11\out,
                                               the other version folders' out\ and v6\phase2\files\
  python phase2_check.py A.csv [B.csv ...]     these files, each on the day(s) in its header (drag files onto phase2.bat)
  python phase2_check.py --date 2026-10-27     that day
  python phase2_check.py --uploaded A.csv      score A.csv and remember it as the file uploaded for its day
  python phase2_check.py --all                 every live day that has files and prices (rebuilds the accumulated table)
  python phase2_check.py --board               only the accumulated table from the saved scores (no download)
  more: --source auto|course|finmind|twse   --check (download even when the course data has the day, and compare)
        --refetch (ignore the saved prices)

Metric (course slides; this script reproduces the board's baselines to all 6 digits: 2026-10-22 Always down 0.015102,
Previous day 0.015561; accumulated 10/12-10/22 Always down -0.014105, Previous day 0.005455):
  R = (2p - 1)(close - open) / open for each stock; AR score = mean of R over the 50 stocks of sample_submission.csv
  accumulated board: total = sum of the daily AR scores, average = total / days
  baselines: Always down = every p is 0; Previous day = p is the stock's y on the previous trading day (close > open)
  timing = mean(2p - 1) * mean(r) and xsec = AR - timing (bp), as in compare_static.py

Prices: for each stock, the course data if it has the day, otherwise downloaded.
  1. course data: data\market\<id>.csv (merged by update_data.py), or the newest data*.zip in downloads\ or in your
     Downloads folder (read only: nothing in data\ is changed)
  2. FinMind TaiwanStockPrice (raw open/close, one request per stock, about 20 s). The course data comes from the same
     prices: the ratio close/open equals adj_close/adj_open to 1e-15 (checked 2026-10-27 on 8069, 3293 and 3661 for
     10/20, 10/21, 10/22, 10/27). set FINMIND_TOKEN=... is optional, as for the night session.
  3. if FinMind fails: TWSE MI_INDEX (listed stocks) and TPEx daily close quotes (OTC stocks: 8069, 3293)
  Complete downloads are saved in v6\phase2\prices\prices_YYYY-MM-DD.csv and reused; when the course data has the day
  as well, the two are compared and every stock whose return differs is listed.
  The prices come out after the 13:30 close (FinMind / TWSE usually within an hour or two); before that the newest day
  is reported as not available yet and the day before is shown.
Output: the console; v6\phase2\report_YYYYMMDD.txt (the table and every stock of every file); v6\phase2\scores.csv (one
row per day and file, used for the accumulated table); v6\phase2\uploaded.csv (date,file: what was uploaded each day).
Only the Python standard library, numpy and pandas are needed.
"""
import os, sys, re, ssl, json, time, glob, zipfile, argparse, datetime as dt
import urllib.request, urllib.parse, urllib.error
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); DATA = os.path.join(ROOT, 'data')
OUTD = os.path.join(HERE, 'phase2'); PRICES = os.path.join(OUTD, 'prices'); DROP = os.path.join(OUTD, 'files')
SCORES = os.path.join(OUTD, 'scores.csv'); UPLOADED = os.path.join(OUTD, 'uploaded.csv')
TEMPLATE = os.path.join(ROOT, 'sample_submission.csv')
LIVE_START = pd.Timestamp('2026-10-12')
FINMIND = 'https://api.finmindtrade.com/api/v4/data'
TWSE = 'https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX'
TPEX = 'https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes'
BL_DOWN, BL_PREV = '(baseline) Always down', '(baseline) Previous day'
SKIP_DIRS = ('棄案', '__pycache__', 'cache')


def say(*a):
    print(*a, flush=True)


def now():
    return dt.datetime.now()


# ----------------------------------------------------------------------------------------------------------- downloads
def get_json(url, params=None, timeout=40, tries=3, headers=None):
    """GET -> parsed JSON; system certificates first, certifi as a fallback (as fetch_night_session.py); no retry on
    HTTP errors (a 4xx answer will not change by asking again)"""
    full = url + ('?' + urllib.parse.urlencode(params) if params else '')
    req = urllib.request.Request(full, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json', **(headers or {})})
    ctxs = [ssl.create_default_context()]
    try:
        import certifi
        ctxs.append(ssl.create_default_context(cafile=certifi.where()))
    except ImportError:
        pass
    last = None
    for attempt in range(tries):
        for ctx in ctxs:
            try:
                with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                    return json.loads(r.read().decode('utf-8-sig'))
            except urllib.error.HTTPError as e:
                try:
                    body = e.read().decode('utf-8', 'replace')[:200]
                except Exception:
                    body = ''
                raise RuntimeError(f'HTTP {e.code} {body}'.strip())
            except Exception as e:
                last = e
        time.sleep(2 * (attempt + 1))
    raise last


def num(x):
    """'1,045.00' -> 1045.0; '--', '', 'X' -> nan"""
    try:
        return float(str(x).replace(',', '').strip())
    except ValueError:
        return np.nan


def fetch_finmind(ids, day):
    """raw open/close from FinMind for the 14 calendar days up to `day` (so the previous trading day comes too);
    -> DataFrame(date, id, open, close, source), list of stocks that failed, last error"""
    tok = os.environ.get('FINMIND_TOKEN', '').strip()
    hdr = {'Authorization': f'Bearer {tok}'} if tok else None
    start, end = (day - pd.Timedelta(days=14)).strftime('%Y-%m-%d'), day.strftime('%Y-%m-%d')
    rows, failed, err, run = [], [], None, 0
    print(f'  FinMind TaiwanStockPrice {start}..{end}, {len(ids)} stocks ', end='', flush=True)
    for k, sid in enumerate(ids):
        p = {'dataset': 'TaiwanStockPrice', 'data_id': sid, 'start_date': start, 'end_date': end}
        if tok:
            p['token'] = tok
        try:
            j = get_json(FINMIND, p, headers=hdr)
            if j.get('status') not in (200, None):
                raise RuntimeError(f"FinMind status {j.get('status')} {j.get('msg')}")
            for x in j.get('data') or []:
                o, c = num(x.get('open')), num(x.get('close'))
                if o > 0 and c > 0:
                    rows.append((pd.Timestamp(str(x['date'])[:10]), str(x.get('stock_id', sid)).strip(), o, c, 'FinMind'))
            run = 0
        except Exception as e:
            failed.append(sid); err = e; run += 1
            stop = '402' in str(e) or 'upper limit' in str(e).lower()
            if stop or k == 0 or run >= 3:                         # limit reached, FinMind down, or three failures in a row
                failed += [s for s in ids[k + 1:]]
                print(' request limit reached' if stop else ' FinMind does not answer, stopping', end='')
                break
            continue
        if k == 0 and not any(r[0] == day for r in rows):
            print(f' the first stock ({sid}) has no price for {end} yet, stopping', end='')   # not published: save 49 requests
            failed += [s for s in ids[1:]]
            break
        if k % 10 == 9:
            print('.', end='', flush=True)
        time.sleep(0.15)
    print(flush=True)
    return pd.DataFrame(rows, columns=['date', 'id', 'open', 'close', 'source']), failed, err


def roc_date(s):
    """'1151027', '115/10/27', '115年10月27日' -> Timestamp('2026-10-27'); anything else -> None"""
    m = re.search(r'(\d{2,3})\D?(\d{2})\D?(\d{2})', str(s))
    if not m:
        return None
    try:
        return pd.Timestamp(int(m.group(1)) + 1911, int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def fetch_twse(day):
    """TWSE MI_INDEX (every listed stock, one request) -> {id: (open, close)}"""
    j = get_json(TWSE, {'date': day.strftime('%Y%m%d'), 'type': 'ALLBUT0999', 'response': 'json'})
    tables = list(j.get('tables') or [])
    for k in list(j):                                              # older layout: fields9 / data9
        if k.startswith('fields') and 'data' + k[6:] in j:
            tables.append({'title': j.get('title', ''), 'fields': j[k], 'data': j['data' + k[6:]]})
    for t in tables:
        f = [str(x).strip() for x in (t.get('fields') or [])]
        if '證券代號' in f and '開盤價' in f and '收盤價' in f:
            td = roc_date(t.get('title', '')) or (pd.Timestamp(str(j['date'])) if str(j.get('date', '')).isdigit() else None)
            if td is not None and td != day:
                raise RuntimeError(f'TWSE answered for {td.date()}, not {day.date()}')
            i, io, ic = f.index('證券代號'), f.index('開盤價'), f.index('收盤價')
            out = {}
            for row in t.get('data') or []:
                o, c = num(row[io]), num(row[ic])
                if o > 0 and c > 0:
                    out[str(row[i]).strip()] = (o, c)
            return out
    raise RuntimeError(f"TWSE: no daily quote table for {day.date()} ({j.get('stat', 'no stat')})")


def fetch_tpex(day):
    """TPEx OpenAPI daily close quotes (newest trading day only) -> {id: (open, close)} if that day is `day`"""
    j = get_json(TPEX)
    if not isinstance(j, list) or not j:
        raise RuntimeError('TPEx: empty answer')

    def key(d, *names):
        for n in names:
            for k in d:
                if k.lower() == n.lower():
                    return k
        return None
    k0 = j[0]
    kd, ks = key(k0, 'Date', '資料日期'), key(k0, 'SecuritiesCompanyCode', 'Code', '代號', '股票代號')
    ko, kc = key(k0, 'Open', '開盤'), key(k0, 'Close', '收盤')
    if not (ks and ko and kc):
        raise RuntimeError(f'TPEx: unexpected fields {list(k0)[:8]}')
    out, dates = {}, set()
    for d in j:
        dd = roc_date(d.get(kd, '')) if kd else None
        dates.add(dd)
        if dd is not None and dd != day:
            continue
        o, c = num(d.get(ko)), num(d.get(kc))
        if o > 0 and c > 0:
            out[str(d.get(ks)).strip()] = (o, c)
    if not out:
        raise RuntimeError(f'TPEx: no quotes for {day.date()} (the API has {sorted(str(x.date()) for x in dates if x is not None)[:3]})')
    return out


# --------------------------------------------------------------------------------------------------------- course data
def newest_zip():
    c = glob.glob(os.path.join(ROOT, 'downloads', 'data*.zip')) + glob.glob(os.path.join(os.path.expanduser('~'), 'Downloads', 'data*.zip'))
    return max(c, key=os.path.getmtime) if c else None


def _course_rows(m, sid, source):
    m = m[['date', 'etl:adj_open', 'etl:adj_close']].copy()
    m.columns = ['date', 'open', 'close']
    m['date'] = pd.to_datetime(m['date'])
    m = m[(m.open > 0) & (m.close > 0)]
    return m.assign(id=sid, source=source)[['date', 'id', 'open', 'close', 'source']]


def course_prices(ids):
    """every valid (date, stock) of the course data: data\\market first, then the newest zip for days data\\ lacks"""
    parts = []
    for sid in ids:
        f = os.path.join(DATA, 'market', f'{sid}.csv')
        if os.path.exists(f):
            parts.append(_course_rows(pd.read_csv(f), sid, 'course data'))
    have = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['date', 'id', 'open', 'close', 'source'])
    z = newest_zip()
    if z:
        try:
            zparts = []
            with zipfile.ZipFile(z) as zf:
                names = {os.path.basename(n): n for n in zf.namelist() if '/market/' in n.replace('\\', '/') and n.endswith('.csv')}
                for sid in ids:
                    if f'{sid}.csv' in names:
                        with zf.open(names[f'{sid}.csv']) as fh:
                            zparts.append(_course_rows(pd.read_csv(fh), sid, 'course zip ' + os.path.basename(z)))
            if zparts:
                zz = pd.concat(zparts, ignore_index=True)
                zz = zz[~zz.date.isin(set(have.date))]
                have = pd.concat([have, zz], ignore_index=True)
        except (OSError, zipfile.BadZipFile, KeyError, ValueError) as e:
            say(f'note: could not read {z} ({e})')
    return have


# --------------------------------------------------------------------------------------------------------- submissions
def template_ids():
    return pd.read_csv(TEMPLATE, dtype={'id': str}, encoding='utf-8-sig')['id'].astype(str).str.strip().tolist()


def read_sub(path, ids):
    """-> DataFrame index = the 50 ids, columns = dates (Timestamp), values 0/1; raises ValueError if unusable"""
    s = pd.read_csv(path, dtype={'id': str}, encoding='utf-8-sig')
    if 'id' not in s.columns:
        raise ValueError('no id column')
    s['id'] = s['id'].astype(str).str.strip()
    if s['id'].duplicated().any():
        raise ValueError(f"stock listed twice: {s.loc[s['id'].duplicated(), 'id'].tolist()[:5]}")
    s = s.set_index('id')
    cols = {}
    for c in s.columns:
        try:
            cols[c] = pd.Timestamp(str(c).strip())
        except ValueError:
            raise ValueError(f'column {c!r} is not a date')
    s = s.rename(columns=cols)
    miss, extra = sorted(set(ids) - set(s.index)), sorted(set(s.index) - set(ids))
    if miss:
        raise ValueError(f'missing stocks {miss[:5]}{"..." if len(miss) > 5 else ""}')
    s = s.loc[ids]
    v = s.apply(pd.to_numeric, errors='coerce')
    if v.isna().any().any() or not v.isin([0, 1]).all().all():
        raise ValueError('values are not all 0/1')
    if extra:
        say(f'note: {os.path.basename(path)} has stocks outside the 50 (ignored): {extra[:5]}')
    return v.astype(int)


def fkey(path):
    """comparable form of a file name (uploaded.csv may be edited by hand: / or \\, upper or lower case)"""
    return os.path.normcase(os.path.normpath(str(path).replace('/', os.sep).replace('\\', os.sep)))


def rel(path):
    """path relative to D:\\ML RACE with / (the same text on Windows and Linux, so scores.csv adds up across runs)"""
    a = os.path.abspath(path)
    try:
        r = os.path.relpath(a, ROOT)
        r = a if r.startswith('..') else r
    except ValueError:                                             # another drive on Windows
        r = a
    return r.replace('\\', '/')


def discover():
    """one-day submission files: <version>\\out\\submission*.csv for every version folder, and v6\\phase2\\files\\*.csv"""
    c = glob.glob(os.path.join(ROOT, '*', 'out', 'submission*.csv')) + glob.glob(os.path.join(DROP, '*.csv'))
    out = []
    for f in sorted(set(c)):
        if any(p in SKIP_DIRS for p in rel(f).replace('\\', '/').split('/')):
            continue
        try:
            head = pd.read_csv(f, nrows=0, encoding='utf-8-sig').columns
        except Exception:
            continue
        if len(head) == 2 and head[0].strip() == 'id':
            out.append(f)
    return out


def series_of(relpath, day):
    """file name with its MMDD replaced, so that the same setting on different days adds up (submission_v9_MMDD.csv)"""
    mmdd = day.strftime('%m%d')
    return re.sub(rf'(?<=[_\-.]){mmdd}(?=[_\-.])', 'MMDD', relpath, count=1)


# -------------------------------------------------------------------------------------------------------------- prices
def load_cache(day):
    f = os.path.join(PRICES, f'prices_{day:%Y-%m-%d}.csv')
    if not os.path.exists(f):
        return None
    c = pd.read_csv(f, dtype={'id': str}, parse_dates=['date'])
    return c


def save_cache(day, df):
    os.makedirs(PRICES, exist_ok=True)
    f = os.path.join(PRICES, f'prices_{day:%Y-%m-%d}.csv')
    df.sort_values(['date', 'id']).to_csv(f + '.tmp', index=False)
    os.replace(f + '.tmp', f)


def download(ids, day, how):
    """FinMind (unless how == 'twse'); TWSE + TPEx for the stocks FinMind did not give"""
    cols = ['date', 'id', 'open', 'close', 'source']
    got = pd.DataFrame(columns=cols)
    missing = list(ids)
    if how in ('auto', 'finmind'):
        fm, failed, err = fetch_finmind(ids, day)
        got = fm
        on_day = set(fm.loc[fm.date == day, 'id'])
        missing = [s for s in ids if s not in on_day]
        if err is not None:
            say(f'  FinMind: {len(failed)} stocks failed, last error: {str(err)[:160]}')
            if '402' in str(err) or 'limit' in str(err).lower():
                say('  (request limit: get a free token at finmindtrade.com, then  set FINMIND_TOKEN=...  in this window)')
    if missing and how in ('auto', 'twse'):
        extra = []
        for name, fn in (('TWSE', fetch_twse), ('TPEx', fetch_tpex)):
            try:
                q = fn(day)
                hit = [s for s in missing if s in q]
                extra += [(day, s, q[s][0], q[s][1], name) for s in hit]
                missing = [s for s in missing if s not in q]
                say(f'  {name}: {len(hit)} stocks')
            except Exception as e:
                say(f'  {name}: {str(e)[:160]}')
            if not missing:
                break
        if extra:
            ex = pd.DataFrame(extra, columns=cols)
            got = ex if got.empty else pd.concat([got, ex], ignore_index=True)
    return got, missing


def prices_for(ids, day, course, a):
    """-> (r: Series id -> (close-open)/open on `day`, src: Series id -> source, prev_day, y_prev: Series, notes)"""
    notes = []
    cd = course[course.date == day]
    fetched = None if a.refetch else load_cache(day)
    complete_course = set(ids) <= set(cd.id)
    want_dl = a.source in ('finmind', 'twse') or a.check or not complete_course
    if a.source == 'course':
        want_dl = False
    t = now(); today = pd.Timestamp(t.date())
    if day > today or (day == today and t.time() < dt.time(13, 30)):
        if want_dl:
            notes.append(f'the market has not closed yet on {day.date()} (13:30), nothing to download')
        want_dl = False
    if want_dl and (fetched is None or not set(ids) <= set(fetched.loc[fetched.date == day, 'id'])):
        say(f'  {day.date()}: ' + ('course data incomplete, ' if not complete_course else '') + 'downloading prices')
        got, missing = download(ids, day, 'auto' if a.source in ('auto', 'course') else a.source)
        if not got.empty:
            fetched = got
            if not missing:
                save_cache(day, got)
        if missing:
            notes.append(f'no downloaded price on {day.date()} for {len(missing)} stocks: {" ".join(missing[:10])}'
                         + ('' if len(missing) < len(ids) else ' (not published yet? prices come out after the close)'))
    fd = fetched[fetched.date == day] if fetched is not None else cd.iloc[0:0]

    use_dl = a.source in ('finmind', 'twse')
    first, second = (fd, cd) if use_dl else (cd, fd)
    pick = pd.concat([first, second[~second.id.isin(set(first.id))]], ignore_index=True)
    pick = pick[pick.id.isin(set(ids))].drop_duplicates('id').set_index('id')
    r = (pick.close / pick.open - 1).reindex([s for s in ids if s in pick.index])
    src = pick.source.reindex(r.index)

    # cross-check course data and downloaded prices wherever both have the stock
    if not cd.empty and not fd.empty:
        a1 = cd.drop_duplicates('id').set_index('id'); b1 = fd.drop_duplicates('id').set_index('id')
        com = a1.index.intersection(b1.index)
        if len(com):
            d = ((a1.close / a1.open)[com] - (b1.close / b1.open)[com]).abs()
            bad = d[d > 1e-6]                                     # one price tick is at least 5e-4
            if len(bad):
                notes.append(f'WARNING: course data and downloaded prices differ for {len(bad)} of {len(com)} stocks '
                             f'(largest {bad.max() * 1e4:.3f} bp: {" ".join(bad.sort_values(ascending=False).index[:8])}); '
                             'the AR uses ' + ('the download' if use_dl else 'the course data (= the board)'))
            else:
                notes.append(f'course data and downloaded prices agree on all {len(com)} stocks (largest difference {d.max():.1e})')

    # previous trading day: newest earlier day that most stocks have (closures such as 10/23 have no valid rows)
    allp = pd.concat([course, fetched if fetched is not None else course.iloc[0:0]], ignore_index=True)
    allp = allp[(allp.date < day) & allp.id.isin(set(ids))]
    cnt = allp.groupby('date').id.nunique()
    cnt = cnt[cnt >= 0.8 * len(ids)]
    prev, y_prev = None, None
    if len(cnt):
        prev = cnt.index.max()
        pp = allp[allp.date == prev].drop_duplicates('id').set_index('id')
        y_prev = (pp.close > pp.open).astype(int).reindex(ids)
    return r, src, prev, y_prev, notes


# ------------------------------------------------------------------------------------------------------------- scoring
def score(p, r):
    """p: Series id -> 0/1, r: Series id -> return (decimal); stocks without a return are left out (and reported)"""
    com = [s for s in p.index if s in r.index and pd.notna(r[s]) and pd.notna(p[s])]
    s = 2 * p[com].astype(float) - 1
    rr = r[com].astype(float)
    AR = float((s * rr).mean())
    tm = float(s.mean() * rr.mean())
    return {'AR': AR, 'AR_bp': AR * 1e4, 'timing_bp': tm * 1e4, 'xsec_bp': (AR - tm) * 1e4,
            'share1': float(p[com].mean()), 'hits': int(((rr > 0).astype(int) == p[com].astype(int)).sum()), 'n_stocks': len(com)}


def read_uploaded():
    if os.path.exists(UPLOADED):
        u = pd.read_csv(UPLOADED, dtype={'file': str}, encoding='utf-8-sig')
        u['date'] = pd.to_datetime(u['date'])
        return u.dropna(subset=['date', 'file']).drop_duplicates('date', keep='last')
    return pd.DataFrame({'date': pd.to_datetime(pd.Series([], dtype=str)), 'file': pd.Series([], dtype=str)})


def write_uploaded(u):
    os.makedirs(OUTD, exist_ok=True)
    u = u.assign(date=pd.to_datetime(u['date'])).drop_duplicates('date', keep='last').sort_values('date')
    u.assign(date=u.date.dt.strftime('%Y-%m-%d')).to_csv(UPLOADED, index=False, encoding='utf-8')


def read_scores():
    if os.path.exists(SCORES):
        S = pd.read_csv(SCORES, dtype={'file': str, 'series': str}, encoding='utf-8')
        S['date'] = pd.to_datetime(S['date'])
        return S
    return pd.DataFrame()


def write_scores(new):
    old = read_scores()
    allr = pd.concat([old, new], ignore_index=True) if not old.empty else new
    allr['date'] = pd.to_datetime(allr['date'])
    allr = allr.drop_duplicates(['date', 'file'], keep='last').sort_values(['date', 'AR'], ascending=[True, False])
    os.makedirs(OUTD, exist_ok=True)
    allr.assign(date=allr.date.dt.strftime('%Y-%m-%d')).to_csv(SCORES + '.tmp', index=False, encoding='utf-8')
    os.replace(SCORES + '.tmp', SCORES)
    return allr


def fmt_row(x, label, mark=''):
    if x.get('AR') is None or pd.isna(x.get('AR')):
        return f"  {'':>9s}  {'':>8s}  {'':>7s}  {'':>7s}  {'':>4s}  {'':>5s}  {label}{mark}"
    t = '' if pd.isna(x.get('timing_bp')) else f"{x['timing_bp']:7.2f}"
    xs = '' if pd.isna(x.get('xsec_bp')) else f"{x['xsec_bp']:7.2f}"
    sh = '' if pd.isna(x.get('share1')) else f"{x['share1']:4.2f}"
    hits = '' if pd.isna(x.get('hits')) else f"{int(x['hits'])}/{int(x['n_stocks'])}"
    return f"  {x['AR']:9.6f}  {x['AR_bp']:8.2f}  {t:>7s}  {xs:>7s}  {sh:>4s}  {hits:>5s}  {label}{mark}"


def one_day(day, ids, subs, course, uploaded, a, report):
    r, src, prev, y_prev, notes = prices_for(ids, day, course, a)
    if len(r) == 0:
        return None, notes
    srcs = ', '.join(f'{k} {v}' for k, v in src.value_counts().items())
    head = [f'Phase 2 check  {day:%Y-%m-%d} ({day:%a})   prices: {srcs}   ({len(r)}/{len(ids)} stocks)']
    head += ['  ' + n for n in notes]
    head.append(f'  market: Top50 mean (close - open)/open {r.mean() * 1e4:.2f} bp, {int((r > 0).sum())} of {len(r)} up'
                + (f'; previous trading day {prev:%Y-%m-%d}' if prev is not None else ''))
    if len(r) < len(ids):
        head.append(f'  WARNING: only {len(r)} of {len(ids)} stocks have a price; the AR below is not comparable to the board')
    rows = []
    b0 = score(pd.Series(0, index=ids), r); b0.update(file=BL_DOWN); rows.append(b0)
    if y_prev is not None and y_prev.notna().all():
        b1 = score(y_prev.astype(int), r); b1.update(file=BL_PREV); rows.append(b1)
    up = uploaded.loc[uploaded.date == day, 'file']
    up = up.iloc[0] if len(up) else None
    detail = {}
    for f, s in subs.items():
        if day not in s.columns:
            continue
        x = score(s[day], r); x.update(file=rel(f)); rows.append(x)
        detail[rel(f)] = s[day]
    T = pd.DataFrame(rows)
    T['date'] = pd.to_datetime(pd.Series([day] * len(T))); T['series'] = [series_of(f, day) if not f.startswith('(') else f for f in T.file]
    T['prices'] = srcs; T['scored_at'] = now().strftime('%Y-%m-%d %H:%M')
    bl = T[T.file.str.startswith('(')]; fl = T[~T.file.str.startswith('(')].sort_values('AR', ascending=False)
    lines = head + ['', f"  {'AR score':>9s}  {'AR bp':>8s}  {'timing':>7s}  {'xsec':>7s}  {'1s':>4s}  {'hits':>5s}  file"]
    for _, x in pd.concat([bl, fl]).iterrows():
        lines.append(fmt_row(x, x.file, '   <- uploaded' if up is not None and fkey(x.file) == fkey(up) else ''))
    if fl.empty:
        lines.append('  (no submission file for this day)')
    say('\n'.join(lines))
    report += lines + ['']
    # per-stock detail (report file only)
    d = pd.DataFrame({'r_bp': (r * 1e4).round(2), 'y': (r > 0).astype(int)})
    if y_prev is not None:
        d['prev_p'] = y_prev
    legend = []
    for n, (k, v) in enumerate(detail.items(), 1):
        d[f'p{n}'] = v
        d[f'bp{n}'] = ((2 * v.reindex(d.index) - 1) * r / len(r) * 1e4).round(2)
        legend.append(f'    p{n}, bp{n}: {k}')
    d['source'] = src
    report += ['  per stock: r_bp = (close - open)/open in bp, y = close > open, prev_p = the Previous day baseline,',
               '  pN = prediction of file N, bpN = its contribution (2p - 1) * r / 50 in bp (the column sums to the AR in bp)']
    report += legend + [d.to_string(), '']
    return T, notes


def accumulated(S, uploaded):
    S = S[S.date >= LIVE_START]
    if S.empty:
        return ['accumulated: nothing scored yet']
    lines = [f'Accumulated from {rel(SCORES)} (board: total = sum of the daily AR scores, average = total / days)',
             f"  {'days':>4s}  {'total':>9s}  {'average':>9s}  {'first':>10s}  {'last':>10s}  series"]
    ser = []
    if len(uploaded):
        m = uploaded.assign(k=uploaded.file.map(fkey)).merge(S.assign(k=S.file.map(fkey)).drop(columns='file'),
                                                             on=['date', 'k'], how='left')
        ok = m.dropna(subset=['AR'])
        if len(ok):
            ser.append(('uploaded (v6/phase2/uploaded.csv)', ok))
        lost = m[m.AR.isna()]
        if len(lost):
            lines.append(f"  note: uploaded days without a score yet: {' '.join(lost.date.dt.strftime('%m/%d'))}")
    for k, g in S[~S.file.str.startswith('(')].groupby('series'):
        ser.append((k, g))
    for k in (BL_DOWN, BL_PREV):
        g = S[S.file == k]
        if len(g):
            ser.append((k + ' (every scored day)', g))
    for k, g in ser:
        lines.append(f"  {g.date.nunique():4d}  {g.AR.sum():9.6f}  {g.AR.mean():9.6f}  {g.date.min():%Y-%m-%d}  {g.date.max():%Y-%m-%d}  {k}")
    if len(ser) and ser[0][0].startswith('uploaded'):
        days = set(ser[0][1].date)
        for k in (BL_DOWN, BL_PREV):
            g = S[(S.file == k) & S.date.isin(days)]
            if len(g):
                lines.append(f"        same {len(g)} days as uploaded: {k[11:]} total {g.AR.sum():.6f}, average {g.AR.mean():.6f}")
    return lines


def main():
    try:
        sys.stdout.reconfigure(errors='replace')
    except Exception:
        pass
    ap = argparse.ArgumentParser(description='Phase 2 (live board) local check: download prices, score submissions')
    ap.add_argument('files', nargs='*')
    ap.add_argument('--date', default=None)
    ap.add_argument('--uploaded', nargs='+', default=[], help='these files were uploaded (their day is in the header)')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--board', action='store_true')
    ap.add_argument('--source', default='auto', choices=['auto', 'course', 'finmind', 'twse'])
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--refetch', action='store_true')
    a = ap.parse_args()

    ids = template_ids()
    uploaded = read_uploaded()
    if a.board:
        say('\n'.join(accumulated(read_scores(), uploaded))); return

    def uniq(fs):                                                  # the same file given twice (relative / absolute) counts once
        seen, out = set(), []
        for f in fs:
            k = os.path.normcase(os.path.abspath(f))
            if k not in seen:
                seen.add(k); out.append(f)
        return out
    explicit = uniq(a.files + a.uploaded)
    files = uniq(explicit + (discover() if not a.files else []))  # --uploaded alone: the day's other files too
    subs = {}
    for f in files:
        try:
            subs[f] = read_sub(f, ids)
        except Exception as e:
            say(f'skipped {rel(f)}: {e}')
    for f in a.uploaded:
        if f in subs:
            for d in subs[f].columns:
                uploaded = pd.concat([uploaded, pd.DataFrame({'date': [d], 'file': [rel(f)]})], ignore_index=True)
    if a.uploaded:
        write_uploaded(uploaded); uploaded = read_uploaded()

    today = pd.Timestamp(now().date())
    dates = sorted({d for s in subs.values() for d in s.columns})
    edates = sorted({d for f in explicit if f in subs for d in subs[f].columns})
    if a.date:
        targets = [pd.Timestamp(a.date)]
    elif a.all:
        targets = sorted({d for d in dates + list(uploaded.date) if LIVE_START <= d <= today})
    elif explicit:
        targets = [d for d in edates if d <= today]
        later = [d for d in edates if d > today]
        if later:
            say(f"not scored (in the future): {' '.join(str(d.date()) for d in later)}")
    else:
        targets = [d for d in dates if LIVE_START <= d <= today][-1:]
    if not targets:
        say('nothing to score among the given files' if explicit else
            'no day to score: no one-day submission file up to today was found (drag a file onto phase2.bat, or use --date)')
        say('\n'.join(accumulated(read_scores(), uploaded))); return

    course = course_prices(ids)
    report, done = [], []
    fallback = not (a.date or a.all or explicit)
    tried = 0
    while targets:
        day = targets.pop(0); tried += 1
        T, notes = one_day(day, ids, subs, course, uploaded, a, report)
        if T is None:
            say(f'{day:%Y-%m-%d}: no prices yet. ' + ' '.join(notes))
            if fallback and tried < 6:
                earlier = [d for d in dates if LIVE_START <= d < day]
                if earlier:
                    say(f'showing the day before that has files: {earlier[-1]:%Y-%m-%d}\n')
                    targets.insert(0, earlier[-1])
            continue
        done.append(T)
        say('')
    if not done:
        say('Nothing new scored. Prices come out after the 13:30 close (usually within one or two hours); try again later.\n')
        S = read_scores()
        if not S.empty:
            say('\n'.join(accumulated(S, uploaded)))
    if done:
        S = write_scores(pd.concat(done, ignore_index=True))
        acc = accumulated(S, uploaded)
        say('\n'.join(acc))
        report += [''] + acc
        os.makedirs(OUTD, exist_ok=True)
        rp = os.path.join(OUTD, f"report_{done[-1].date.iloc[0]:%Y%m%d}.txt")
        with open(rp, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(report) + '\n')
        say(f'\nreport: {rel(rp)}   scores: {rel(SCORES)}')
        say('Compare the AR score column with the daily board, and the uploaded line with the accumulated board.')


if __name__ == '__main__':
    main()

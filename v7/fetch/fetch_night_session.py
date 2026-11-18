r"""
fetch_night_session.py -- daily TAIEX futures (TX) prices incl. the night session (夜盤, 15:00-05:00) from FinMind,
2020-01 to today, for the v7 timing test (README_v7 section 4). The night session closes at 05:00, before the 09:00
deadline, so it is allowed for the static period and for live.

  python fetch_night_session.py                  -> ..\data\tx_daily.csv  (7 requests, one per year)
  set FINMIND_TOKEN=...  (optional; free account token, higher request limit)

FinMind API v4: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanFuturesDaily&data_id=TX&start_date=..&end_date=..
Rows: one per contract month and session (trading_session = 'position' regular / 'after_market' night); every row is
kept as downloaded, the front-month night return is derived later (how FinMind dates the night rows is checked there).
2026-10-25: the file is replaced only if the new download is not smaller than the old file (rows within 1 %, newest day
and night session not older); it is written to a temporary file first, and a gzip copy per day goes to
..\data\archive\tx_daily_YYYYMMDD.csv.gz so that the inputs of every live day can be reproduced. A refused or failed
download exits with code 1 and keeps the old file (daily.bat goes on; v9 then treats the night session as missing).
"""
import os, sys, ssl, json, time, datetime as dt, urllib.request, urllib.parse, csv

HERE = os.path.dirname(os.path.abspath(__file__)); V7 = os.path.dirname(HERE); OUT = os.path.join(V7, 'data', 'tx_daily.csv')
API = 'https://api.finmindtrade.com/api/v4/data'

def get(params):
    ctxs = [ssl.create_default_context()]
    try:
        import certifi
        ctxs.append(ssl.create_default_context(cafile=certifi.where()))
    except ImportError:
        pass
    url = API + '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    last = None
    for attempt in range(3):
        for ctx in ctxs:
            try:
                with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
                    j = json.loads(r.read().decode('utf-8'))
                if j.get('status') != 200:
                    raise RuntimeError(f"FinMind: {j.get('status')} {j.get('msg')}")
                return j['data']
            except Exception as e:
                last = e
        time.sleep(5 * (attempt + 1))
    raise last

ARCHIVE = os.path.join(V7, 'data', 'archive')

def stats_of(rows):
    """number of rows and newest date per trading session ('position' = day, 'after_market' = night)"""
    s = {'n': len(rows), 'position': '', 'after_market': ''}
    for r in rows:
        k, d = r.get('trading_session'), str(r.get('date', ''))
        if k in ('position', 'after_market') and d > s[k]:
            s[k] = d
    return s

def stats_of_file(path):
    if not os.path.exists(path):
        return None
    with open(path, newline='', encoding='utf-8') as f:
        return stats_of(list(csv.DictReader(f)))

def compare_stats(old, new):
    """the history only grows: fewer rows (beyond 1 %) or an older newest day means an incomplete download"""
    if not old:
        return []
    bad = []
    if new['n'] < 0.99 * old['n']:
        bad.append(f"{new['n']} rows < {old['n']} before")
    for k in ('position', 'after_market'):
        if new[k] < old[k]:
            bad.append(f"newest {k} day {new[k] or 'none'} is older than {old[k]} before")
    return bad

def snapshot(path):
    """one gzip copy per day in ..\\data\\archive (reproducibility of the live inputs)"""
    try:
        import gzip, shutil
        os.makedirs(ARCHIVE, exist_ok=True)
        dst = os.path.join(ARCHIVE, f'tx_daily_{dt.date.today():%Y%m%d}.csv.gz')
        with open(path, 'rb') as fi, gzip.open(dst, 'wb') as fo:
            shutil.copyfileobj(fi, fo)
        return dst
    except OSError as e:
        print(f'note: no snapshot ({e})')
        return None

def main():
    tok = os.environ.get('FINMIND_TOKEN', '')
    rows = []
    for y in range(2020, dt.date.today().year + 1):
        p = {'dataset': 'TaiwanFuturesDaily', 'data_id': 'TX', 'start_date': f'{y}-01-01', 'end_date': min(dt.date(y, 12, 31), dt.date.today()).isoformat()}
        if tok:
            p['token'] = tok
        try:
            d = get(p)
        except Exception as e:
            sys.exit(f'FinMind download failed for {y}: {e!r}\n'
                     'If the message mentions a token, a limit or status 402: get a free token at finmindtrade.com, '
                     'then  set FINMIND_TOKEN=...  and run again from the same window.')
        rows += d
        print(f'{y}: {len(d)} rows', flush=True); time.sleep(2)
    if not rows:
        sys.exit('no rows')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    keys = list(rows[0].keys())
    new, old = stats_of(rows), stats_of_file(OUT)
    bad = compare_stats(old, new)
    if bad:
        sys.exit(f'NOT written, {OUT} kept as it was: ' + '; '.join(bad) + ' (FinMind returned less than before; try again later)')
    tmp = OUT + '.tmp'
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    try:
        os.replace(tmp, OUT)
    except OSError as e:
        sys.exit(f'could not replace {OUT} ({e}); is it open in another program? The old file is unchanged.')
    snap = snapshot(OUT)
    print(f"written {OUT}: {len(rows)} rows, newest day session {new['position']}, newest night session {new['after_market']}"
          + (f'; snapshot {os.path.basename(snap)}' if snap else ''))

if __name__ == '__main__':
    main()

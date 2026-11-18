r"""
fetch_news.py -- race_news (2026-11-17): download 鉅亨網 台股新聞 (cnyes category tw_stock) with publish time and the
stock tags of every article, 2019-10 .. yesterday, into race_news\ext\cnyes\YYYY-MM.csv. Runs on the user's Windows
Python (the Linux VM of the Claude app has no internet). Standard library + pandas only.

  python fetch_news.py                 download the months that are missing (finished months are kept; run again to continue)
  python fetch_news.py --since 2026-11-01   re-download from this date (live update; overwrites those months)

Source: https://api.cnyes.com/media/api/v1/newslist/category/tw_stock?startAt=..&endAt=..&limit=30&page=..
(the JSON API behind news.cnyes.com; startAt / endAt are read as whole Taipei calendar days, 30 items per page).
Kept per article: newsId, publishAt (unix seconds) and its Taipei time, categoryName, title, summary (first 300
characters), stock (Taiwan codes the site tagged), market (all tagged symbols), keyword. The article body is not kept.
Output: ext\cnyes\YYYY-MM.csv, ext\fetch_log.txt, ext\fetch_status.json, ext\FETCH_DONE.txt
About 5,000 requests (one week = about 13 pages); 0.3 s apart: roughly 40-80 minutes.
"""
import os, sys, ssl, json, time, argparse, datetime as dt, urllib.request, urllib.parse, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, 'ext'); OUT = os.path.join(EXT, 'cnyes')
API = 'https://api.cnyes.com/media/api/v1/newslist/category/tw_stock'
START = dt.date(2019, 10, 1)
UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36',
      'Accept': 'application/json', 'Origin': 'https://news.cnyes.com', 'Referer': 'https://news.cnyes.com/'}
TPE = dt.timezone(dt.timedelta(hours=8))


def log(*a):
    s = time.strftime('%H:%M:%S') + ' ' + ' '.join(str(x) for x in a)
    print(s, flush=True)
    os.makedirs(EXT, exist_ok=True)
    with open(os.path.join(EXT, 'fetch_log.txt'), 'a', encoding='utf-8') as fh:
        fh.write(s + '\n')


def contexts():
    c = [ssl.create_default_context()]
    try:
        import certifi
        c.append(ssl.create_default_context(cafile=certifi.where()))
    except ImportError:
        pass
    return c


def get_json(url):
    last = None
    for attempt in range(5):
        for ctx in contexts():
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
                    return json.loads(r.read().decode('utf-8'))
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (429, 422):                              # 422 'unknown' seen 11/17 13:28 and 13:52; a rerun went through
                    log(f'HTTP {e.code}: waiting 30 s and retrying'); time.sleep(30); break
                if e.code in (400, 401, 403, 404):
                    raise
            except Exception as e:
                last = e
        time.sleep(3 * (attempt + 1))
    raise last


def tpe_day_start(d):
    return int(dt.datetime(d.year, d.month, d.day, tzinfo=TPE).timestamp())


def fetch_window(a, b):
    """all articles of the Taipei days a..b (inclusive)"""
    items, page, last = [], 1, 1
    while page <= last:
        q = urllib.parse.urlencode({'startAt': tpe_day_start(a), 'endAt': tpe_day_start(b), 'limit': 30, 'page': page})
        j = get_json(API + '?' + q)
        it = j.get('items') or {}
        last = int(it.get('last_page') or 1)
        items += it.get('data') or []
        page += 1
        time.sleep(0.3)
        if page > 200:
            raise RuntimeError(f'more than 200 pages for {a}..{b}')
    return items


def rows_of(items):
    out = []
    for x in items:
        t = int(x.get('publishAt') or 0)
        mk = x.get('market') or []
        out.append({'newsId': x.get('newsId'), 'publishAt': t,
                    'publish_tpe': dt.datetime.fromtimestamp(t, TPE).strftime('%Y-%m-%d %H:%M:%S') if t else '',
                    'categoryName': x.get('categoryName') or '', 'title': (x.get('title') or '').replace('\n', ' '),
                    'summary': (x.get('summary') or '').replace('\n', ' ')[:300],
                    'stock': '|'.join(str(s) for s in (x.get('stock') or [])),
                    'market': '|'.join(str(m.get('symbol', '')) for m in mk if isinstance(m, dict)),
                    'keyword': '|'.join(str(k) for k in (x.get('keyword') or []))})
    return out


def month_iter(a, b):
    d = dt.date(a.year, a.month, 1)
    while d <= b:
        nxt = dt.date(d.year + (d.month == 12), d.month % 12 + 1, 1)
        yield d, min(nxt - dt.timedelta(days=1), b)
        d = nxt


def main():
    import pandas as pd
    ap = argparse.ArgumentParser(); ap.add_argument('--since', default=''); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    yday = dt.datetime.now(TPE).date() - dt.timedelta(days=1)
    since = dt.date.fromisoformat(a.since) if a.since else None
    sf = os.path.join(EXT, 'fetch_status.json')
    status = json.load(open(sf, encoding='utf-8')) if os.path.exists(sf) else {}
    # connection test
    try:
        t = fetch_window(yday, yday)
        log(f'connection ok: {len(t)} articles on {yday}')
    except Exception as e:
        log(f'connection test FAILED: {e!r}  -- check the internet connection / proxy and run again'); sys.exit(1)
    n_req0 = 0
    for m0, m1 in month_iter(START, yday):
        key = m0.strftime('%Y-%m'); f = os.path.join(OUT, key + '.csv')
        complete_month = m1 >= dt.date(m0.year + (m0.month == 12), m0.month % 12 + 1, 1) - dt.timedelta(days=1)
        if os.path.exists(f) and status.get(key, {}).get('complete') and not (since and m1 >= since):
            continue
        t0 = time.time(); items = []
        d = m0
        failed_days = []
        while d <= m1:
            e = min(d + dt.timedelta(days=6), m1)
            try:
                items += fetch_window(d, e)
            except Exception as ex:                               # fall back to single days; skip a day that still fails
                log(f'window {d}..{e} failed ({ex!r}); trying day by day')
                x = d
                while x <= e:
                    try:
                        items += fetch_window(x, x)
                    except Exception as ex2:
                        failed_days.append(str(x)); log(f'  {x}: FAILED ({ex2!r}), skipped')
                    x += dt.timedelta(days=1)
            d = e + dt.timedelta(days=1)
        rows = rows_of(items)
        df = pd.DataFrame(rows).drop_duplicates('newsId') if rows else pd.DataFrame(columns=['newsId'])
        tmp = f + '.tmp'
        df.to_csv(tmp, index=False, encoding='utf-8'); os.replace(tmp, f)
        status[key] = {'rows': int(len(df)), 'tagged': int((df['stock'].astype(str) != '').sum()) if len(df) else 0,
                       'complete': bool(complete_month) and not failed_days, 'failed_days': failed_days,
                       'last_day': str(m1), 'at': time.strftime('%Y-%m-%d %H:%M')}
        with open(sf, 'w', encoding='utf-8') as fh:
            json.dump(status, fh, ensure_ascii=False, indent=1)
        log(f'{key}: {len(df)} articles ({status[key]["tagged"]} with stock tags), {time.time() - t0:.0f}s')
    total = sum(v.get('rows', 0) for v in status.values())
    log(f'done: {len(status)} months, {total} articles')
    with open(os.path.join(EXT, 'FETCH_DONE.txt'), 'w', encoding='utf-8') as fh:
        fh.write(time.strftime('%Y-%m-%d %H:%M:%S') + f'  months {len(status)}  articles {total}\n')


if __name__ == '__main__':
    main()

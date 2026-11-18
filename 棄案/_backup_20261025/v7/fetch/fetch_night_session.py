r"""
fetch_night_session.py -- daily TAIEX futures (TX) prices incl. the night session (夜盤, 15:00-05:00) from FinMind,
2020-01 to today, for the v7 timing test (README_v7 section 4). The night session closes at 05:00, before the 09:00
deadline, so it is allowed for the static period and for live.

  python fetch_night_session.py                  -> ..\data\tx_daily.csv  (7 requests, one per year)
  set FINMIND_TOKEN=...  (optional; free account token, higher request limit)

FinMind API v4: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanFuturesDaily&data_id=TX&start_date=..&end_date=..
Rows: one per contract month and session (trading_session = 'position' regular / 'after_market' night); every row is
kept as downloaded, the front-month night return is derived later (how FinMind dates the night rows is checked there).
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
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f'written {OUT}: {len(rows)} rows, columns {keys}')

if __name__ == '__main__':
    main()

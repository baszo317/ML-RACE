r"""
record_mis.py -- record TWSE MIS pre-open snapshots (試撮 / indicative prices, 08:30-09:00) of the competition's 100
stocks and the TAIEX, so that we can measure how close the 08:5x indicative price is to the real open, and how much of
the same-day gap signal it carries (v7 README, section 3). Recording only: nothing here feeds a submission.

  python record_mis.py              poll every 20 s from 08:28 to 09:03 (start it before 08:28, e.g. Task Scheduler)
  python record_mis.py --once       one snapshot now (connection test; outside trading hours it shows the last close)
  python record_mis.py --start 08:28 --end 09:03 --every 20

Output: ..\data\mis\YYYYMMDD.jsonl, one line per request: {"t_local": ..., "symbols": [...], "status": ..., "body": {...}}
The raw response is kept as is (field names of the MIS JSON are not documented; they are parsed later).
Only the Python standard library is used (certifi is tried for certificates, as in v5\fetch_exdiv.py).
"""
import os, sys, ssl, json, time, csv, argparse, datetime as dt, urllib.request, http.cookiejar

HERE = os.path.dirname(os.path.abspath(__file__)); V7 = os.path.dirname(HERE)
OUT = os.path.join(V7, 'data', 'mis'); SYMS = os.path.join(V7, 'data', 'mis_symbols.csv')
BASE = 'https://mis.twse.com.tw/stock/'

def opener():
    ctxs = [ssl.create_default_context()]
    try:
        import certifi
        ctxs.append(ssl.create_default_context(cafile=certifi.where()))
    except ImportError:
        pass
    jar = http.cookiejar.CookieJar()
    for ctx in ctxs:
        op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar), urllib.request.HTTPSHandler(context=ctx))
        op.addheaders = [('User-Agent', 'Mozilla/5.0'), ('Accept', 'application/json, text/javascript, */*'),
                         ('Referer', BASE + 'index.jsp?lang=zh_tw')]
        try:
            op.open(BASE + 'index.jsp?lang=zh_tw', timeout=20).read()          # session cookie
            return op
        except Exception as e:
            last = e
    raise last

def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

def snapshot(op, symbols, f):
    for part in chunks(symbols, 40):
        url = BASE + 'api/getStockInfo.jsp?ex_ch=' + '|'.join(part) + f'&json=1&delay=0&_={int(time.time() * 1000)}'
        rec = {'t_local': dt.datetime.now().isoformat(timespec='milliseconds'), 'symbols': part}
        try:
            with op.open(url, timeout=15) as r:
                rec['status'] = r.status; rec['body'] = json.loads(r.read().decode('utf-8'))
        except Exception as e:
            rec['status'] = 'error'; rec['error'] = repr(e)
        f.write(json.dumps(rec, ensure_ascii=False) + '\n'); f.flush()
        n = len(rec.get('body', {}).get('msgArray', [])) if isinstance(rec.get('body'), dict) else 0
        print(f"{rec['t_local'][11:19]}  {rec['status']}  {n}/{len(part)} quotes", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--once', action='store_true'); ap.add_argument('--start', default='08:28'); ap.add_argument('--end', default='09:03')
    ap.add_argument('--every', type=float, default=20.0)
    a = ap.parse_args()
    with open(SYMS, encoding='utf-8') as fh:
        symbols = [f"{r['market']}_{r['stock_id']}.tw" for r in csv.DictReader(fh)] + ['tse_t00.tw']   # + TAIEX
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, dt.date.today().strftime('%Y%m%d') + ('_test' if a.once else '') + '.jsonl')
    op = opener()
    with open(path, 'a', encoding='utf-8') as f:
        if a.once:
            snapshot(op, symbols, f); print('written', path); return
        today = dt.date.today()
        t0 = dt.datetime.combine(today, dt.datetime.strptime(a.start, '%H:%M').time())
        t1 = dt.datetime.combine(today, dt.datetime.strptime(a.end, '%H:%M').time())
        while dt.datetime.now() < t0:
            time.sleep(5)
        while dt.datetime.now() < t1:
            s = time.time()
            try:
                snapshot(op, symbols, f)
            except Exception as e:
                print('error', repr(e), flush=True)
                try:
                    op = opener()
                except Exception:
                    pass
            time.sleep(max(1.0, a.every - (time.time() - s)))
    print('written', path)

if __name__ == '__main__':
    main()

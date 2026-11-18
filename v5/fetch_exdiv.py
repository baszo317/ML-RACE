r"""
fetch_exdiv.py -- record the ex-dividend / ex-rights dates of the Top100 stocks (run every morning, ~2 s)

  python fetch_exdiv.py                          announced (future) ex-dates, TWSE TWT48U_ALL -> data\ref\exdiv_upcoming.csv
  python fetch_exdiv.py --history 2026-06-01     past ex-dates from START to today, TWSE TWT49U -> data\ref\exdiv_history.csv
  python fetch_exdiv.py --history 2026-06-01 2026-09-30

Sources (public, no token):
  TWT48U_ALL  https://openapi.twse.com.tw/v1/exchangeReport/TWT48U_ALL                   除權除息預告表 (announcements)
  TWT49U      https://www.twse.com.tw/rwd/zh/exRight/TWT49U?response=json&startDate=YYYYMMDD&endDate=YYYYMMDD
                                                                                     除權除息計算結果表 (what happened)
  Dates are in ROC format (1151211 or 115年12月11日 = 2026-12-11).
Why --history: data\ref\corporate_actions.parquet ends on 2026-07-31, so the static leaderboard months 2026/08-09
  had no ex-dividend dates in v2 (the rule "predict 0 on ex-dividend days" was silently off). Every date in TWT49U was
  announced in TWT48U weeks before it happened, so using it for those past days adds no information that was not
  public before 09:00 of the day.
Rows are only ever added (union with the existing file). Manual fallback: data\ref\exdiv_manual.csv (stock_id,date).
v5: every run appends one line to data\ref\exdiv_fetch_log.csv (run_at, status, n_events, n_top100); guards.py
  stops the morning run when there is no successful fetch from today (an empty announcement table is a valid result,
  a failed download is not -- the file alone cannot tell the two apart).
Why: on ex-dividend days the intraday return of these stocks averaged -62 bp in 2020-2026 (P(close > open) = 0.34,
     negative in every single year), and the models do not know about it -> the prediction is set to 0 on those days.
"""
import os, sys, json, ssl, time, argparse, urllib.request
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data'); OUT = os.path.join(DATA, 'ref', 'exdiv_upcoming.csv'); HIST = os.path.join(DATA, 'ref', 'exdiv_history.csv')
FETCH_LOG = os.path.join(DATA, 'ref', 'exdiv_fetch_log.csv')
URL = 'https://openapi.twse.com.tw/v1/exchangeReport/TWT48U_ALL'
URL49 = 'https://www.twse.com.tw/rwd/zh/exRight/TWT49U?response=json&startDate={a}&endDate={b}'

def roc_to_date(s):
    s = str(s).strip()
    if '年' in s:                                            # 115年08月03日
        y, r = s.split('年'); m, r = r.split('月'); d = r.replace('日', '')
        return pd.Timestamp(year=int(y) + 1911, month=int(m), day=int(d))
    s = s.replace('/', '')
    if not s.isdigit() or len(s) < 6:
        return None
    y, m, d = int(s[:-4]) + 1911, int(s[-4:-2]), int(s[-2:])
    return pd.Timestamp(year=y, month=m, day=d)

def download(url):
    """certificate verification stays on; if the Windows certificate store lacks TWSE's chain, certifi's bundle is tried"""
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'})
    contexts = [ssl.create_default_context()]
    try:
        import certifi
        contexts.append(ssl.create_default_context(cafile=certifi.where()))
    except ImportError:
        pass
    last = None
    for attempt in range(2):
        for ctx in contexts:
            try:
                with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
                    return json.loads(r.read().decode('utf-8'))
            except Exception as e:
                last = e
        time.sleep(2)
    hint = '' if len(contexts) > 1 else ' (pip install certifi may fix a certificate error)'
    raise RuntimeError(f'download failed: {last}{hint}')

def log_fetch(status, n_events=0, n_top100=0):
    row = pd.DataFrame([{'run_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), 'status': status,
                         'n_events': n_events, 'n_top100': n_top100}])
    row.to_csv(FETCH_LOG, mode='a', index=False, header=not os.path.exists(FETCH_LOG))

def union(path, new, key=('stock_id', 'date')):
    if os.path.exists(path):
        old = pd.read_csv(path, dtype=str)
        allr = pd.concat([old, new.astype(str)], ignore_index=True).drop_duplicates(list(key), keep='first')
    else:
        allr = new.astype(str)
    allr = allr.sort_values(['date', 'stock_id']); os.makedirs(os.path.dirname(path), exist_ok=True)
    allr.to_csv(path, index=False, encoding='utf-8')
    return allr

def history(start, end, top100, top50):
    """TWT49U month by month (one request per month, 3 s apart)"""
    rows, a = [], pd.Timestamp(start)
    end = pd.Timestamp(end) if end else pd.Timestamp.now().normalize()
    while a <= end:
        b = min(a + pd.offsets.MonthEnd(0), end)
        js = download(URL49.format(a=a.strftime('%Y%m%d'), b=b.strftime('%Y%m%d')))
        data = js.get('data') or []
        n = 0
        for r in data:
            if len(r) < 7 or str(r[1]).strip() not in top100:
                continue
            d = roc_to_date(r[0])
            if d is None:
                continue
            rows.append({'stock_id': str(r[1]).strip(), 'date': d.strftime('%Y-%m-%d'), 'kind': str(r[6]).strip(), 'name': str(r[2]).strip(),
                         'value': str(r[5]).replace(',', ''), 'prev_close': str(r[3]).replace(',', ''), 'source': 'TWT49U',
                         'fetched': pd.Timestamp.now().strftime('%Y-%m-%d')})
            n += 1
        print(f'{a.date()}..{b.date()}: {len(data)} events in TWT49U ({js.get("stat", "")}), {n} for Top100', flush=True)
        a = b + pd.Timedelta(days=1); time.sleep(3)
    new = pd.DataFrame(rows, columns=['stock_id', 'date', 'kind', 'name', 'value', 'prev_close', 'source', 'fetched'])
    allr = union(HIST, new)
    x = allr[(pd.to_datetime(allr.date) >= pd.Timestamp(start)) & allr.stock_id.isin(top50)]
    print(f'{len(new)} rows downloaded; {HIST} now has {len(allr)} rows; Top50 ex-dates since {pd.Timestamp(start).date()}:')
    print(x[['date', 'stock_id', 'name', 'kind']].to_string(index=False) if len(x) else '  none')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--history', nargs='*', metavar='DATE', help='START [END]: download past ex-dates (TWT49U)')
    a = ap.parse_args()
    info = json.load(open(os.path.join(DATA, 'info.json'))); top100 = set(info['Top100@Y']); top50 = set(info['Top50@Y'])
    if a.history is not None:
        if not a.history:
            sys.exit('give a start date, e.g. python fetch_exdiv.py --history 2026-06-01')
        try:
            history(a.history[0], a.history[1] if len(a.history) > 1 else None, top100, top50)
        except Exception as e:
            sys.exit(f'{e}\n-> add the ex-dates by hand to {os.path.join(DATA, "ref", "exdiv_manual.csv")} (columns stock_id,date)')
        return
    try:
        recs = download(URL)
    except Exception as e:
        log_fetch('failed')
        sys.exit(f'{e}\n-> add the ex-dates by hand to {os.path.join(DATA, "ref", "exdiv_manual.csv")} (columns stock_id,date)')
    rows = []
    for r in recs:
        code = str(r.get('Code', '')).strip()
        if code not in top100:
            continue
        d = roc_to_date(r.get('Date', ''))
        if d is None:
            continue
        rows.append({'stock_id': code, 'date': d.strftime('%Y-%m-%d'), 'kind': str(r.get('Exdividend', '')).strip(),
                     'name': str(r.get('Name', '')).strip(), 'cash_dividend': r.get('CashDividend', ''),
                     'first_seen': pd.Timestamp.now().strftime('%Y-%m-%d')})
    new = pd.DataFrame(rows, columns=['stock_id', 'date', 'kind', 'name', 'cash_dividend', 'first_seen'])
    allr = union(OUT, new)
    log_fetch('ok', len(recs), len(new))
    today = pd.Timestamp.now().normalize()
    up = allr[pd.to_datetime(allr.date) >= today]
    print(f'{len(recs)} announced events in TWT48U, {len(new)} for Top100; file now has {len(allr)} rows -> {OUT}')
    if len(up):
        up = up.assign(top50=up.stock_id.isin(top50))
        print('upcoming ex-dates (Top100):'); print(up[['date', 'stock_id', 'name', 'kind', 'cash_dividend', 'top50']].to_string(index=False))
    else:
        print('no upcoming ex-date for the Top100 stocks in the current TWSE announcement table')

if __name__ == '__main__':
    main()

r"""
fetch_exdiv.py -- record the announced ex-dividend / ex-rights dates of the Top100 stocks (run every morning, ~2 s)

  python fetch_exdiv.py

Source: TWSE OpenAPI "除權除息預告表" (TWT48U_ALL), public, no token:
        https://openapi.twse.com.tw/v1/exchangeReport/TWT48U_ALL
        It lists the ex-dates announced for the coming weeks (dates in ROC format, e.g. 1151211 = 2026-12-11).
Output: ..\data\ref\exdiv_upcoming.csv  (stock_id, date, kind, name, cash_dividend, first_seen)
        Rows are only ever added (union with the existing file), so dates seen earlier stay recorded.
Manual fallback: if the download fails, add lines "stock_id,date" (e.g. 2330,2026-12-10) to ..\data\ref\exdiv_manual.csv;
        train_and_predict_v2.py reads both files plus data\ref\corporate_actions.parquet (history up to 2026-07-31).
Why: on ex-dividend days the intraday return of these stocks averaged -62 bp in 2020-2026 (P(close > open) = 0.34,
     negative in every single year), and the models did not know about it -> the prediction is set to 0 on those days.
"""
import os, sys, json, ssl, time, urllib.request
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data'); OUT = os.path.join(DATA, 'ref', 'exdiv_upcoming.csv')
URL = 'https://openapi.twse.com.tw/v1/exchangeReport/TWT48U_ALL'

def roc_to_date(s):
    s = str(s).strip().replace('/', '')
    if not s.isdigit() or len(s) < 6:
        return None
    y, m, d = int(s[:-4]) + 1911, int(s[-4:-2]), int(s[-2:])
    return pd.Timestamp(year=y, month=m, day=d)

def download():
    """certificate verification stays on; if the Windows certificate store lacks TWSE's chain, certifi's bundle is tried"""
    req = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'})
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

def main():
    info = json.load(open(os.path.join(DATA, 'info.json'))); top100 = set(info['Top100@Y']); top50 = set(info['Top50@Y'])
    try:
        recs = download()
    except Exception as e:
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
    if os.path.exists(OUT):
        old = pd.read_csv(OUT, dtype=str)
        allr = pd.concat([old, new.astype(str)], ignore_index=True).drop_duplicates(['stock_id', 'date'], keep='first')
    else:
        allr = new.astype(str)
    allr = allr.sort_values(['date', 'stock_id']); os.makedirs(os.path.dirname(OUT), exist_ok=True)
    allr.to_csv(OUT, index=False, encoding='utf-8')
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

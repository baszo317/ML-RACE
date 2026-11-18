r"""
fetch_ext.py -- RACE-STD v1 amendment 1 (PREREG section 6): download the data behind Chen's crawled V5 feature groups
(K_pre, K_pre_stk, T1_chip, L_macro) into race_std\ext\. Runs on the user's Windows Python (the VM has no internet).
Standard library + pandas only.

  python fetch_ext.py            download what is missing (finished files are kept; run again to continue)
  python fetch_ext.py --force    download everything again

Sources (public):
  Yahoo Finance chart API v8   ADRs TSM UMC ASX CHT; their local shares 2330 2303 3711 2412 (unadjusted close); ^SOX ^N225
                               ^KS11 ^IXIC ^GSPC; TWD=X ^TNX DX-Y.NYB; U.S. sector ETFs SPY SOXX XLK XLF XLB XLI XLP XLY XLE
                               XLV XLC IYT.  Fallback for ADRs / ETFs / TWD: stooq.com daily csv.
  FinMind API v4               TaiwanStockMarginPurchaseShortSale, TaiwanDailyShortSaleBalances (SBL), TaiwanStockDayTrading
                               for the 100 competition stocks; TaiwanStockInfo (industry, for the sector-ETF map).
                               Without a token FinMind allows about 300 requests an hour: the script waits and continues.
                               set FINMIND_TOKEN=...  (free account at finmindtrade.com) for 600 an hour.
Output: ext\yahoo\<ticker>.csv, ext\finmind\<dataset>\<stock>.csv, ext\finmind\TaiwanStockInfo.csv, ext\fetch_log.txt,
        ext\fetch_status.json (ok / failed per file).
"""
import os, sys, ssl, json, time, argparse, datetime as dt, urllib.request, urllib.parse, urllib.error, io

HERE = os.path.dirname(os.path.abspath(__file__)); STD = os.path.dirname(HERE)
ROOT = os.environ.get('MLRACE_ROOT', os.path.dirname(os.path.dirname(STD)))
EXT = os.path.join(STD, 'ext')
START = '2019-10-01'
YAHOO = ['TSM', 'UMC', 'ASX', 'CHT', '2330.TW', '2303.TW', '3711.TW', '2412.TW', '^SOX', '^N225', '^KS11', '^IXIC', '^GSPC',
         'TWD=X', '^TNX', 'DX-Y.NYB', 'SPY', 'SOXX', 'XLK', 'XLF', 'XLB', 'XLI', 'XLP', 'XLY', 'XLE', 'XLV', 'XLC', 'IYT']
STOOQ = {'TSM': 'tsm.us', 'UMC': 'umc.us', 'ASX': 'asx.us', 'CHT': 'cht.us', 'TWD=X': 'usdtwd', 'SPY': 'spy.us',
         'SOXX': 'soxx.us', 'XLK': 'xlk.us', 'XLF': 'xlf.us', 'XLB': 'xlb.us', 'XLI': 'xli.us', 'XLP': 'xlp.us', 'XLY': 'xly.us',
         'XLE': 'xle.us', 'XLV': 'xlv.us', 'XLC': 'xlc.us', 'IYT': 'iyt.us'}
FINMIND = ['TaiwanStockMarginPurchaseShortSale', 'TaiwanDailyShortSaleBalances', 'TaiwanStockDayTrading']
API = 'https://api.finmindtrade.com/api/v4/data'
UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'}


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


def http(url, timeout=60):
    last = None
    for ctx in contexts():
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (401, 402, 403, 404, 429):
                raise
        except Exception as e:
            last = e
    raise last


def write_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False, encoding='utf-8')
    os.replace(tmp, path)


# ------------------------------------------------------------------ Yahoo
def yahoo(sym):
    import pandas as pd
    p1 = int(dt.datetime.strptime(START, '%Y-%m-%d').replace(tzinfo=dt.timezone.utc).timestamp())
    p2 = int(time.time()) + 86400
    last = None
    for host in ('query1', 'query2'):
        url = (f'https://{host}.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(sym)}?period1={p1}&period2={p2}'
               f'&interval=1d&includeAdjustedClose=true&events=div%2Csplit')
        try:
            j = json.loads(http(url).decode('utf-8'))
            res = j['chart']['result'][0]
            ts = res.get('timestamp') or []
            q = res['indicators']['quote'][0]
            adj = (res['indicators'].get('adjclose') or [{}])[0].get('adjclose') or [None] * len(ts)
            off = int(res['meta'].get('gmtoffset') or 0)
            d = [dt.datetime.fromtimestamp(t + off, dt.timezone.utc).date().isoformat() for t in ts]
            df = pd.DataFrame({'date': d, 'open': q.get('open'), 'high': q.get('high'), 'low': q.get('low'), 'close': q.get('close'),
                               'adjclose': adj, 'volume': q.get('volume')})
            df = df.dropna(subset=['close']).drop_duplicates('date', keep='last').sort_values('date')
            if len(df) < 100:
                raise RuntimeError(f'only {len(df)} rows')
            df['source'] = 'yahoo'; df['tz'] = res['meta'].get('exchangeTimezoneName', '')
            return df
        except Exception as e:
            last = e
            time.sleep(2)
    if sym in STOOQ:
        try:
            raw = http(f'https://stooq.com/q/d/l/?s={STOOQ[sym]}&i=d').decode('utf-8')
            df = pd.read_csv(io.StringIO(raw))
            df.columns = [c.lower() for c in df.columns]
            df = df[df.date >= START].rename(columns={'vol': 'volume'})
            df['adjclose'] = df['close']; df['source'] = 'stooq'; df['tz'] = ''
            if len(df) >= 100:
                return df[['date', 'open', 'high', 'low', 'close', 'adjclose', 'volume', 'source', 'tz']]
        except Exception as e:
            last = e
    raise RuntimeError(f'{sym}: {last!r}')


# ------------------------------------------------------------------ FinMind
def finmind(params, tok):
    p = dict(params)
    if tok:
        p['token'] = tok
    url = API + '?' + urllib.parse.urlencode(p)
    waits = 0
    while True:
        try:
            j = json.loads(http(url, timeout=120).decode('utf-8'))
        except urllib.error.HTTPError as e:
            j = {'status': e.code, 'msg': str(e)}
        except Exception as e:
            j = {'status': -1, 'msg': repr(e)}
        st = j.get('status')
        if st == 200:
            return j.get('data') or []
        msg = str(j.get('msg', ''))
        if st == 402 or 'limit' in msg.lower() or 'upper' in msg.lower():
            waits += 1
            if waits > 14:
                raise RuntimeError(f'FinMind limit: {msg}')
            log(f'FinMind request limit reached ({msg[:80]}); waiting 5 minutes (wait {waits}). '
                f'A free token (set FINMIND_TOKEN=...) doubles the limit.')
            time.sleep(300)
            continue
        if st == -1 and waits < 3:
            waits += 1; time.sleep(10); continue
        raise RuntimeError(f'FinMind {st}: {msg[:200]}')


def main():
    import pandas as pd
    ap = argparse.ArgumentParser(); ap.add_argument('--force', action='store_true'); a = ap.parse_args()
    os.makedirs(EXT, exist_ok=True)
    with open(os.path.join(ROOT, 'data', 'info.json'), encoding='utf-8') as fh:
        ids = [str(s) for s in json.load(fh)['Top100@Y']]
    today = dt.date.today().isoformat()
    status = {}
    sf = os.path.join(EXT, 'fetch_status.json')
    if os.path.exists(sf) and not a.force:
        try:
            status = json.load(open(sf, encoding='utf-8'))
        except Exception:
            status = {}

    def save_status():
        with open(sf, 'w', encoding='utf-8') as fh:
            json.dump(status, fh, ensure_ascii=False, indent=1)

    log(f'start: {len(YAHOO)} Yahoo tickers, FinMind {len(FINMIND)} datasets x {len(ids)} stocks + stock info; output {EXT}')
    for sym in YAHOO:
        f = os.path.join(EXT, 'yahoo', sym.replace('^', 'IDX_').replace('=', '_') + '.csv')
        if os.path.exists(f) and status.get('yahoo:' + sym, {}).get('ok') and not a.force:
            continue
        try:
            df = yahoo(sym)
            write_csv(df, f)
            status['yahoo:' + sym] = {'ok': True, 'rows': len(df), 'first': df.date.min(), 'last': df.date.max(),
                                      'source': df.source.iloc[0], 'at': time.strftime('%Y-%m-%d %H:%M')}
            log(f'yahoo {sym}: {len(df)} rows {df.date.min()} .. {df.date.max()} ({df.source.iloc[0]})')
        except Exception as e:
            status['yahoo:' + sym] = {'ok': False, 'error': repr(e)[:300]}
            log(f'yahoo {sym}: FAILED {e!r}')
        save_status(); time.sleep(1.0)
    tok = os.environ.get('FINMIND_TOKEN', '').strip()
    log('FinMind token: ' + ('set' if tok else 'not set (about 300 requests an hour)'))
    f = os.path.join(EXT, 'finmind', 'TaiwanStockInfo.csv')
    if not (os.path.exists(f) and status.get('finmind:TaiwanStockInfo', {}).get('ok')) or a.force:
        try:
            d = pd.DataFrame(finmind({'dataset': 'TaiwanStockInfo'}, tok))
            write_csv(d, f)
            status['finmind:TaiwanStockInfo'] = {'ok': True, 'rows': len(d)}
            log(f'finmind TaiwanStockInfo: {len(d)} rows')
        except Exception as e:
            status['finmind:TaiwanStockInfo'] = {'ok': False, 'error': repr(e)[:300]}
            log(f'finmind TaiwanStockInfo: FAILED {e!r}')
        save_status()
    n = 0
    for ds in FINMIND:
        for s in ids:
            key = f'finmind:{ds}:{s}'
            f = os.path.join(EXT, 'finmind', ds, f'{s}.csv')
            if os.path.exists(f) and status.get(key, {}).get('ok') and not a.force:
                continue
            try:
                d = pd.DataFrame(finmind({'dataset': ds, 'data_id': s, 'start_date': START, 'end_date': today}, tok))
                write_csv(d, f)
                status[key] = {'ok': True, 'rows': len(d), 'last': (str(d.date.max()) if len(d) else '')}
                n += 1
                if n % 10 == 0:
                    log(f'finmind {ds}: {s} ({n} files this run)')
            except Exception as e:
                status[key] = {'ok': False, 'error': repr(e)[:300]}
                log(f'finmind {ds} {s}: FAILED {e!r}')
            save_status(); time.sleep(0.6)
    ok = sum(1 for v in status.values() if v.get('ok')); bad = [k for k, v in status.items() if not v.get('ok')]
    log(f'done: {ok} files ok, {len(bad)} failed' + (': ' + ', '.join(bad[:20]) if bad else ''))
    with open(os.path.join(EXT, 'FETCH_DONE.txt'), 'w', encoding='utf-8') as fh:
        fh.write(time.strftime('%Y-%m-%d %H:%M:%S') + f'  ok {ok}  failed {len(bad)}\n' + '\n'.join(bad) + '\n')


if __name__ == '__main__':
    main()

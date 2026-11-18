r"""
fetch_mkt.py -- race_mkt (2026-11-17): download the futures / options positioning data behind the market-view experiments
into race_mkt\ext\. Runs on the user's Windows Python (the Linux VM of the Claude app has no internet). Standard library +
pandas only. Every source is independent: a failed source is logged and the others continue.

  python fetch_mkt.py          download everything (about 3-6 minutes)
  python fetch_mkt.py --only taifex_lt,taifex_pc     only some sources (names below)

Sources (public, daily, all published after the 13:45 / 15:00 close of day t-1, i.e. before 09:00 of day t):
  finmind_fut_inst   FinMind TaiwanFuturesInstitutionalInvestors  TX MTX TMF   三大法人 long / short trading and open interest
  finmind_opt_inst   FinMind TaiwanOptionInstitutionalInvestors   TXO          三大法人 call / put trading and open interest
  finmind_fut_daily  FinMind TaiwanFuturesDaily                   TX MTX TMF   OHLC, volume, settlement, OI per contract month and session
  finmind_total      FinMind TaiwanStockTotalInstitutionalInvestors, TaiwanStockTotalMarginPurchaseShortSale (whole market)
  taifex_lt          TAIFEX 期貨大額交易人未沖銷部位 (dlLargeTraderFutDown, <= 3 months per request), only the TX composite
                     (TX + MTX/4 + TMF/20) rows are kept
  taifex_pc          TAIFEX 臺指選擇權 Put/Call 比 (dlPcRatioDown, <= 30 days per request)
  taifex_inst        TAIFEX 三大法人 區分各期貨契約 (dlFutContractsDateDown; the site only serves the last 3 years) for TX MTX TMF,
                     used to cross-check FinMind (and to fill TMF if FinMind does not have it)
Output: ext\finmind\*.csv, ext\taifex\*.csv, ext\fetch_log.txt, ext\fetch_status.json, ext\FETCH_DONE.txt
FinMind token (optional): set FINMIND_TOKEN=... in the same cmd window before running (the race_live .bat files use it too).
"""
import os, sys, ssl, json, time, argparse, datetime as dt, urllib.request, urllib.parse, urllib.error, io

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, 'ext')
START = '2019-10-01'
API = 'https://api.finmindtrade.com/api/v4/data'
TAIFEX = 'https://www.taifex.com.tw/cht/3/'
UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'}
SOURCES = ['finmind_fut_inst', 'finmind_opt_inst', 'finmind_fut_daily', 'finmind_total', 'taifex_lt', 'taifex_pc', 'taifex_inst']


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


def http(url, data=None, timeout=120, referer=None):
    """GET (data None) or POST form data; returns (bytes, headers)"""
    last = None
    hdr = dict(UA)
    if data is not None:
        hdr['Content-Type'] = 'application/x-www-form-urlencoded'
    if referer:
        hdr['Referer'] = referer
    body = urllib.parse.urlencode(data).encode('ascii') if data is not None else None
    for attempt in range(3):
        for ctx in contexts():
            try:
                req = urllib.request.Request(url, data=body, headers=hdr)
                with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                    return r.read(), dict(r.headers)
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (401, 402, 403, 404):
                    raise
            except Exception as e:
                last = e
        time.sleep(3 * (attempt + 1))
    raise last


def write_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False, encoding='utf-8')
    os.replace(tmp, path)


# ------------------------------------------------------------------ FinMind
def finmind(params, tok):
    p = dict(params)
    if tok:
        p['token'] = tok
    url = API + '?' + urllib.parse.urlencode(p)
    waits = 0
    while True:
        try:
            j = json.loads(http(url, timeout=180)[0].decode('utf-8'))
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
            log(f'FinMind request limit reached ({msg[:80]}); waiting 5 minutes (wait {waits}).')
            time.sleep(300)
            continue
        if st == -1 and waits < 3:
            waits += 1; time.sleep(10); continue
        raise RuntimeError(f'FinMind {st}: {msg[:200]}')


def finmind_range(ds, data_id, tok, today):
    """one request for the whole range; if it fails or looks short, one request per calendar year"""
    import pandas as pd
    base = {'dataset': ds, 'start_date': START, 'end_date': today}
    if data_id:
        base['data_id'] = data_id
    try:
        d = pd.DataFrame(finmind(base, tok))
        recent = (dt.date.fromisoformat(today) - dt.timedelta(days=12)).isoformat()
        if len(d) and 'date' in d and str(d.date.max())[:10] >= recent:
            return d
        log(f'  {ds} {data_id}: whole range gave {len(d)} rows (last {d.date.max() if len(d) and "date" in d else "-"}); '
            f'trying year by year')
    except Exception as e:
        log(f'  {ds} {data_id}: whole range failed ({e!r}); trying year by year')
    parts = []
    y0, y1 = int(START[:4]), int(today[:4])
    for y in range(y0, y1 + 1):
        a = max(START, f'{y}-01-01'); b = min(today, f'{y}-12-31')
        p = dict(base); p['start_date'] = a; p['end_date'] = b
        parts.append(pd.DataFrame(finmind(p, tok)))
        time.sleep(0.5)
    parts = [x for x in parts if len(x)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ------------------------------------------------------------------ TAIFEX
def chunks(a, b, days):
    a = dt.date.fromisoformat(a); b = dt.date.fromisoformat(b)
    while a <= b:
        e = min(b, a + dt.timedelta(days=days - 1))
        yield a, e
        a = e + dt.timedelta(days=1)


def taifex_csv(endpoint, a, b, extra=None, referer=None):
    import pandas as pd
    data = {'queryStartDate': a.strftime('%Y/%m/%d'), 'queryEndDate': b.strftime('%Y/%m/%d')}
    data.update(extra or {})
    raw, hdr = http(TAIFEX + endpoint, data=data, referer=referer)
    disp = {k.lower(): v for k, v in hdr.items()}.get('content-disposition', '') or ''
    txt = raw.decode('cp950', errors='replace')
    if 'attachment' not in disp.lower():
        if '<html' in txt[:400].lower() or '<!doctype' in txt[:400].lower():
            raise RuntimeError(f'{endpoint} {a}..{b}: HTML instead of CSV ({len(raw)} bytes)')
    if not txt.strip():
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(txt), dtype=str, index_col=False)
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    df = df.loc[:, [c for c in df.columns if c and not c.startswith('Unnamed')]]
    return df


def taifex_series(name, endpoint, start, end, days, keep=None, extra=None, referer=None):
    import pandas as pd
    parts, failed = [], []
    for a, b in chunks(start, end, days):
        for attempt in range(3):
            try:
                df = taifex_csv(endpoint, a, b, extra, referer)
                if keep is not None and len(df):
                    df = keep(df)
                parts.append(df)
                break
            except Exception as e:
                if attempt == 2:
                    failed.append(f'{a}..{b}: {e!r}'[:200])
                    log(f'  {name} {a}..{b}: FAILED {e!r}')
                time.sleep(2 + 3 * attempt)
        time.sleep(0.6)
    parts = [x for x in parts if len(x)]
    out = pd.concat(parts, ignore_index=True).drop_duplicates() if parts else pd.DataFrame()
    return out, failed


def main():
    import pandas as pd
    ap = argparse.ArgumentParser(); ap.add_argument('--only', default=''); a = ap.parse_args()
    only = [s.strip() for s in a.only.split(',') if s.strip()] or SOURCES
    os.makedirs(os.path.join(EXT, 'finmind'), exist_ok=True); os.makedirs(os.path.join(EXT, 'taifex'), exist_ok=True)
    today = dt.date.today().isoformat()
    sf = os.path.join(EXT, 'fetch_status.json')
    status = json.load(open(sf, encoding='utf-8')) if os.path.exists(sf) else {}

    def done(key, df=None, err=None, extra=None):
        if err is None:
            st = {'ok': True, 'rows': int(len(df)), 'at': time.strftime('%Y-%m-%d %H:%M')}
            if df is not None and len(df):
                dc = 'date' if 'date' in df else ('日期' if '日期' in df else None)
                if dc:
                    st['first'] = str(df[dc].min()); st['last'] = str(df[dc].max())
            st.update(extra or {})
            log(f'{key}: {st}')
        else:
            st = {'ok': False, 'error': str(err)[:300], 'at': time.strftime('%Y-%m-%d %H:%M')}
            log(f'{key}: FAILED {err}')
        status[key] = st
        with open(sf, 'w', encoding='utf-8') as fh:
            json.dump(status, fh, ensure_ascii=False, indent=1)

    yday = (dt.date.today() - dt.timedelta(days=1)).isoformat()      # TAIFEX: end at yesterday (today's file is not out yet)
    tok = os.environ.get('FINMIND_TOKEN', '').strip()
    log(f'start {today}; sources {only}; FinMind token ' + ('set' if tok else 'not set'))
    jobs = []
    if 'finmind_fut_inst' in only:
        jobs += [('TaiwanFuturesInstitutionalInvestors', i, f'fut_inst_{i}.csv') for i in ('TX', 'MTX', 'TMF')]
    if 'finmind_opt_inst' in only:
        jobs += [('TaiwanOptionInstitutionalInvestors', 'TXO', 'opt_inst_TXO.csv')]
    if 'finmind_fut_daily' in only:
        jobs += [('TaiwanFuturesDaily', i, f'fut_daily_{i}.csv') for i in ('TX', 'MTX', 'TMF')]
    if 'finmind_total' in only:
        jobs += [('TaiwanStockTotalInstitutionalInvestors', None, 'total_inst.csv'),
                 ('TaiwanStockTotalMarginPurchaseShortSale', None, 'total_margin.csv')]
    for ds, did, fn in jobs:
        key = f'finmind:{ds}:{did or "-"}'
        try:
            d = finmind_range(ds, did, tok, today)
            if not len(d):
                raise RuntimeError('no rows')
            write_csv(d, os.path.join(EXT, 'finmind', fn))
            done(key, d)
        except Exception as e:
            done(key, err=repr(e))
        time.sleep(0.6)

    if 'taifex_lt' in only:
        def keep_tx(df):
            c = [x for x in df.columns if x.startswith('商品(契約)')]
            return df[df[c[0]].str.strip() == 'TX'] if c else df.iloc[0:0]
        try:
            df, failed = taifex_series('taifex_lt', 'dlLargeTraderFutDown', '2020-01-01', yday, 85, keep=keep_tx,
                                       referer=TAIFEX + 'dlLargeTraderFutView')
            if not len(df):
                raise RuntimeError('no rows; ' + '; '.join(failed[:3]))
            write_csv(df, os.path.join(EXT, 'taifex', 'large_trader_TX.csv'))
            done('taifex:large_trader_TX', df, extra={'failed_chunks': failed})
        except Exception as e:
            done('taifex:large_trader_TX', err=repr(e))

    if 'taifex_pc' in only:
        try:
            df, failed = taifex_series('taifex_pc', 'dlPcRatioDown', '2020-01-01', yday, 28, extra={'down_type': ''},
                                       referer=TAIFEX + 'dlPcRatio')
            if not len(df):
                raise RuntimeError('no rows; ' + '; '.join(failed[:3]))
            write_csv(df, os.path.join(EXT, 'taifex', 'pc_ratio.csv'))
            done('taifex:pc_ratio', df, extra={'failed_chunks': failed})
        except Exception as e:
            done('taifex:pc_ratio', err=repr(e))

    if 'taifex_inst' in only:
        names = {'臺股期貨', '小型臺指期貨', '微型臺指期貨'}

        def keep_idx(df):
            c = [x for x in df.columns if x.startswith('商品名稱')]
            return df[df[c[0]].isin(names)] if c else df.iloc[0:0]
        start3 = (dt.date.today() - dt.timedelta(days=3 * 365 - 5)).isoformat()
        try:
            df, failed = taifex_series('taifex_inst', 'dlFutContractsDateDown', start3, yday, 60, keep=keep_idx,
                                       extra={'commodityId': ''}, referer=TAIFEX + 'dlFutContractsDateView')
            if not len(df):
                raise RuntimeError('no rows; ' + '; '.join(failed[:3]))
            write_csv(df, os.path.join(EXT, 'taifex', 'fut_inst_taifex.csv'))
            done('taifex:fut_inst', df, extra={'failed_chunks': failed})
        except Exception as e:
            done('taifex:fut_inst', err=repr(e))

    ok = sum(1 for v in status.values() if v.get('ok')); bad = [k for k, v in status.items() if not v.get('ok')]
    log(f'done: {ok} sources ok, {len(bad)} failed' + (': ' + ', '.join(bad) if bad else ''))
    with open(os.path.join(EXT, 'FETCH_DONE.txt'), 'w', encoding='utf-8') as fh:
        fh.write(time.strftime('%Y-%m-%d %H:%M:%S') + f'  ok {ok}  failed {len(bad)}\n' + '\n'.join(bad) + '\n')


if __name__ == '__main__':
    main()

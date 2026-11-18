r"""
fetch_live.py -- daily update of the crawled data of the race version (race ver\race_live\ext). Windows Python with internet.
The download functions are those of race_std\research\fetch_ext.py (frozen); only the output folder differs, so race_std\ext
(the data of the experiment) never changes.

  python fetch_live.py --finmind     evening, after about 21:30: the last 14 days of the 3 FinMind datasets for the 100 stocks
                                     (margin / short sale, SBL, day trading), merged into the files; about 300 requests
  python fetch_live.py --yahoo       morning, after 08:01: the 28 Yahoo series (whole history; U.S. closes of last night,
                                     Nikkei / KOSPI opening of today)
  python fetch_live.py --finmind --stale   morning: only the files that do not reach the last trading day in ..\..\data yet
  python fetch_live.py --check       how fresh race_live\ext is (no download)
FINMIND_TOKEN is read from the environment (set FINMIND_TOKEN=... in the cmd window or in Windows).
"""
import os, sys, json, time, argparse, datetime as dt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); RACEVER = os.path.dirname(HERE)
STD = os.path.join(RACEVER, 'race_std')
sys.path.insert(0, os.path.join(STD, 'research'))
import fetch_ext as FE                                               # noqa: E402  (frozen, read only)

EXT = os.environ.get('RACE_LIVE_EXT', os.path.join(HERE, 'ext'))
FE.EXT = EXT                                                         # FE.log and FE.write_csv now write here
STATUS = os.path.join(EXT, 'live_status.json')
BACK_DAYS = 14


def ensure_ext():
    import shutil
    if not os.path.isdir(os.path.join(EXT, 'yahoo')):
        shutil.copytree(os.path.join(STD, 'ext'), EXT, dirs_exist_ok=True)
        FE.log(f'copied {os.path.join(STD, "ext")} -> {EXT}')


def load_status():
    try:
        return json.load(open(STATUS, encoding='utf-8'))
    except Exception:
        return {}


def save_status(st):
    with open(STATUS + '.tmp', 'w', encoding='utf-8') as fh:
        json.dump(st, fh, ensure_ascii=False, indent=1)
    os.replace(STATUS + '.tmp', STATUS)


def yahoo_file(sym):
    return os.path.join(EXT, 'yahoo', sym.replace('^', 'IDX_').replace('=', '_') + '.csv')


def fetch_yahoo(st):
    ok = bad = 0
    for sym in FE.YAHOO:
        f = yahoo_file(sym)
        old = pd.read_csv(f, dtype=str) if os.path.exists(f) else None
        try:
            df = FE.yahoo(sym)
            if old is not None and len(df) < len(old) - 5:
                raise RuntimeError(f'only {len(df)} rows, the file has {len(old)}')
            FE.write_csv(df, f)
            st['yahoo:' + sym] = dict(ok=True, last=str(df.date.max()), rows=len(df), source=str(df.source.iloc[0]),
                                      at=time.strftime('%Y-%m-%d %H:%M'))
            ok += 1
        except Exception as e:
            st['yahoo:' + sym] = dict(ok=False, error=repr(e)[:200], kept=(str(old.date.max()) if old is not None else ''),
                                      at=time.strftime('%Y-%m-%d %H:%M'))
            FE.log(f'yahoo {sym}: FAILED {e!r} (the old file is kept)')
            bad += 1
        save_status(st); time.sleep(0.5)
    FE.log(f'yahoo: {ok} ok, {bad} failed')
    return bad


def last_trading_day():
    """the newest day of ..\\..\\data\\market\\2330.csv that is not the -1 placeholder"""
    x = pd.read_csv(os.path.join(FE.ROOT, 'data', 'market', '2330.csv'), encoding='utf-8')
    x = x[pd.to_numeric(x['etl:adj_close'], errors='coerce') > 0]
    return pd.to_datetime(x['date']).max()


def fetch_finmind(st, ids, stale_only=False):
    tok = os.environ.get('FINMIND_TOKEN', '').strip()
    need = last_trading_day() if stale_only else None
    if need is not None:
        FE.log(f'finmind --stale: only files that end before {need.date()}')
    FE.log('FinMind token: ' + ('set' if tok else 'not set (about 300 requests an hour; the script waits when it hits the limit)'))
    today = dt.date.today().isoformat()
    ok = bad = 0
    for ds in FE.FINMIND:
        for s in ids:
            f = os.path.join(EXT, 'finmind', ds, f'{s}.csv')
            old = None
            if os.path.exists(f):
                try:
                    old = pd.read_csv(f, dtype=str)
                except Exception:
                    old = None
            if need is not None and old is not None and len(old) and 'date' in old.columns and pd.to_datetime(old.date).max() >= need:
                continue
            if old is not None and len(old) and 'date' in old.columns:
                start = (pd.to_datetime(old.date).max() - pd.Timedelta(days=BACK_DAYS)).date().isoformat()
            else:
                start = FE.START
            try:
                new = pd.DataFrame(FE.finmind({'dataset': ds, 'data_id': s, 'start_date': start, 'end_date': today}, tok))
                if len(new):
                    new = new.astype(str)
                    df = new if old is None or not len(old) else pd.concat([old, new], ignore_index=True, sort=False)
                    df = df.drop_duplicates('date', keep='last').sort_values('date')
                    FE.write_csv(df, f)
                last = str(pd.to_datetime((new if len(new) else old).date).max().date()) if (len(new) or old is not None) else ''
                st[f'finmind:{ds}:{s}'] = dict(ok=True, last=last, new_rows=int(len(new)), at=time.strftime('%Y-%m-%d %H:%M'))
                ok += 1
            except Exception as e:
                st[f'finmind:{ds}:{s}'] = dict(ok=False, error=repr(e)[:200], at=time.strftime('%Y-%m-%d %H:%M'))
                FE.log(f'finmind {ds} {s}: FAILED {e!r} (the old file is kept)')
                bad += 1
            if (ok + bad) % 50 == 0:
                save_status(st); FE.log(f'finmind: {ok + bad} requests ({bad} failed)')
            time.sleep(0.6)
    save_status(st)
    FE.log(f'finmind: {ok} ok, {bad} failed')
    return bad


def check():
    rows = []
    for sym in FE.YAHOO:
        f = yahoo_file(sym)
        last = pd.to_datetime(pd.read_csv(f, usecols=['date']).date).max().date() if os.path.exists(f) else None
        rows.append(('yahoo', sym, str(last)))
    for ds in FE.FINMIND:
        d = os.path.join(EXT, 'finmind', ds)
        lasts = []
        for x in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            try:
                lasts.append(pd.to_datetime(pd.read_csv(os.path.join(d, x), usecols=['date']).date).max())
            except Exception:
                pass
        s = pd.Series(lasts).value_counts().sort_index()
        rows.append(('finmind', ds, ', '.join(f'{k.date()} x{v}' for k, v in s.items())))
    for r in rows:
        print(f'{r[0]:8s} {r[1]:38s} {r[2]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--yahoo', action='store_true'); ap.add_argument('--finmind', action='store_true')
    ap.add_argument('--check', action='store_true'); ap.add_argument('--stale', action='store_true')
    a = ap.parse_args()
    ensure_ext()
    st = load_status()
    bad = 0
    if a.yahoo:
        bad += fetch_yahoo(st)
    if a.finmind:
        with open(os.path.join(FE.ROOT, 'data', 'info.json'), encoding='utf-8') as fh:
            ids = [str(s) for s in json.load(fh)['Top100@Y']]
        bad += fetch_finmind(st, ids, a.stale)
    if a.check or not (a.yahoo or a.finmind):
        check()
    sys.exit(0)                                                      # failures are reported; race_live.py predict warns


if __name__ == '__main__':
    main()

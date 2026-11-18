r"""
update_data.py -- merge the daily data.zip downloaded from the course server into ..\data (run every morning)

  python update_data.py                       # newest data*.zip in ..\downloads or in your Downloads folder
  python update_data.py C:\path\to\data.zip   # or a specific zip / an already extracted folder

What it does
  1. first run only: copies ..\data to ..\data_backup_original (an untouched copy of the original competition data)
  2. for data\market\*.csv and data\global\global_data.csv: old rows + rows from the zip; if a date appears in both,
     the zip wins (so yesterday's "-1" placeholder is replaced by the real values); sorted by date
  3. checks (2026-10-25: a problem now stops daily.bat / evening.bat with exit code 1, unless --accept is given)
       stop - the download changes adj_open / adj_close of days already in data\ (the server changed the price-adjustment
              base, old and new rows would no longer fit together), y != (adj_close > adj_open) in new rows, a Top100
              stock file missing in the zip
       note - SMA(30) recomputed from adj_close differs from the server's SMA on the newest day (normal for about 30 days
              after a trading halt), market dates without a global row (normal after U.S. holidays), a possible gap of days
  4. prints the last complete trading day and the day to predict (the date whose rows are all -1)
     closed days (2026-10-25): a date in data\ref\closures_manual.csv (date,reason) is a day the market did not open although
     the course server's calendar has it (typhoon day 2026-10-23): its rows are dropped, it is removed from
     data\ref\calendar.parquet (the official file is kept as calendar_official.parquet), and a -1 placeholder on it becomes
     the placeholder of the next trading day (the day the course server should have asked for)
  5. keeps a copy of the zip in ..\downloads\archive\data_<day to predict>.zip (reproducibility)
  python update_data.py --accept              # merge and exit 0 even if a check fails (only after reading the message)
Nothing is downloaded by this script; it only uses the zip you downloaded from the course server.
"""
import os, sys, glob, json, shutil, zipfile, tempfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data'); BACKUP = os.path.join(ROOT, 'data_backup_original')
CLOSURES = os.path.join(DATA, 'ref', 'closures_manual.csv')
TRADING_DAYS = []     # official calendar without the closed days, 'YYYY-MM-DD' (filled in main when closed days exist)
CLOSURE_LOG = []      # (file, closed day, day its placeholder moved to or None)

def find_zip(arg):
    if arg:
        return arg
    cands = glob.glob(os.path.join(ROOT, 'downloads', 'data*.zip')) + glob.glob(os.path.join(os.path.expanduser('~'), 'Downloads', 'data*.zip'))
    if not cands:
        sys.exit('no data*.zip found in ..\\downloads or in your Downloads folder; pass the path: python update_data.py <zip>')
    return max(cands, key=os.path.getmtime)

def is_placeholder(df):
    vals = df.columns.drop('date')
    return (df[vals] == -1).all(axis=1)

def load_closures():
    """days the market did not open although the calendar published in advance has them (typhoon day 2026-10-23):
    hand-made file data\\ref\\closures_manual.csv with columns date,reason (2026-10-25)"""
    if not os.path.exists(CLOSURES):
        return set()
    c = pd.read_csv(CLOSURES, dtype=str, encoding='utf-8-sig', encoding_errors='replace')
    return {pd.Timestamp(x.strip()).strftime('%Y-%m-%d') for x in c['date'].dropna() if x.strip()}

def fix_calendar(closed, notes):
    """remove the closed days from data\\ref\\calendar.parquet (the official file is kept once as calendar_official.parquet);
    returns the trading days as 'YYYY-MM-DD' strings, [] if the calendar cannot be read"""
    f = os.path.join(DATA, 'ref', 'calendar.parquet')
    try:
        import pyarrow as pa, pyarrow.parquet as pq
        cal = pq.read_table(f).to_pandas()
        days = pd.to_datetime(cal['date']).dt.strftime('%Y-%m-%d')
        drop = days.isin(closed).values
        if drop.any():
            off = os.path.join(DATA, 'ref', 'calendar_official.parquet')
            if not os.path.exists(off):
                shutil.copy2(f, off)
            cal = cal[~drop].reset_index(drop=True)
            if 'pos' in cal:
                cal['pos'] = np.arange(len(cal), dtype='int64')
            pq.write_table(pa.Table.from_pandas(cal, preserve_index=False), f + '.tmp')
            os.replace(f + '.tmp', f)
            notes.append(f'calendar.parquet: removed the closed day(s) {sorted(days[drop])} (official file kept as calendar_official.parquet)')
        return sorted(days[~drop])
    except Exception as e:
        notes.append(f'cannot read or update {f} ({e}): closed days are not removed from the calendar')
        return []

def _placeholder_line(line, head):
    """text-level -1 placeholder test: market files every value -1; global_data.csv ^TWII -1 (as comp_features.read_global)"""
    cols, vals = head.split(','), line.split(',')
    try:
        if '^TWII' in cols:
            return float(vals[cols.index('^TWII')]) == -1
        return len(vals) > 1 and all(float(v) == -1 for v in vals[1:])
    except (ValueError, IndexError):
        return False

def apply_closures(rows, head, closed, name):
    """rows {date: line} of one file: the rows of closed days are dropped; a -1 placeholder on a closed day that is still the
    day to predict (no complete row after it) becomes the row of the next trading day, if the file has no row for that day
    yet (2026-10-25: the course server kept asking for the typhoon day 10/23; the next trading day is 10/27)"""
    hit = sorted(d for d in rows if d in closed)
    if not hit:
        return
    complete = [d for d, ln in rows.items() if d not in closed and not _placeholder_line(ln, head)]
    last = max(complete) if complete else ''
    for d in hit:
        ln = rows.pop(d)
        later = [t for t in TRADING_DAYS if t > d]
        nd = later[0] if later else None
        if nd and nd > last and nd not in rows and _placeholder_line(ln, head):
            rows[nd] = nd + ln[len(d):]
            CLOSURE_LOG.append((name, d, nd))
        else:
            CLOSURE_LOG.append((name, d, None))

def _lines(path):
    with open(path, encoding='utf-8-sig') as f:
        rows = [ln.rstrip('\r\n') for ln in f if ln.strip()]
    return rows[0], {ln.split(',', 1)[0]: ln for ln in rows[1:]}       # header, {date: original text line}

def merge_file(old_path, new_path, closed=()):
    """Merge at the text level: untouched old rows keep their exact original text (no float re-formatting);
    dates present in the download are taken from the download (this replaces yesterday's -1 placeholder); closed days
    (load_closures) are applied before the file is written (apply_closures)."""
    new_head, new_rows = _lines(new_path)
    old_head, rows = _lines(old_path) if os.path.exists(old_path) else (new_head, {})
    if old_head.split(',') != new_head.split(','):
        sys.exit(f'column mismatch in {os.path.basename(new_path)}:\n  {old_head}\n  {new_head}')
    n_old = len(rows); rows.update(new_rows)
    if closed:
        apply_closures(rows, old_head, closed, os.path.basename(old_path))
    tmp = old_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
        f.write(old_head + '\n')
        for d in sorted(rows):
            f.write(rows[d] + '\n')
    os.replace(tmp, old_path)
    return pd.read_csv(old_path), n_old, len(new_rows)

def _md5(path):
    import hashlib
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def archive_zip(src, tag):
    """keep a copy of every zip that was merged: ..\\downloads\\archive\\data_<day to predict>.zip; a different zip for the
    same day is kept next to it as data_<day>_2.zip, _3, ... (2026-10-25)"""
    try:
        if not os.path.isfile(src):
            return None
        dst_dir = os.path.join(ROOT, 'downloads', 'archive'); os.makedirs(dst_dir, exist_ok=True)
        h, k = _md5(src), 1
        while True:
            dst = os.path.join(dst_dir, f'data_{tag}.zip' if k == 1 else f'data_{tag}_{k}.zip')
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                return dst
            if os.path.abspath(dst) == os.path.abspath(src) or _md5(dst) == h:
                return dst
            k += 1
    except OSError as e:
        print(f'note: could not keep a copy of the zip ({e})')
        return None

def main():
    accept = '--accept' in sys.argv[1:]
    args = [x for x in sys.argv[1:] if x != '--accept']
    src = find_zip(args[0] if args else None)
    print(f'source: {src}', flush=True)
    if not os.path.exists(os.path.join(BACKUP, 'info.json')):          # back up only the competition files (not raw/, pit/, ...)
        os.makedirs(BACKUP, exist_ok=True)
        for sub in ('market', 'global'):
            shutil.copytree(os.path.join(DATA, sub), os.path.join(BACKUP, sub), dirs_exist_ok=True)
        shutil.copy2(os.path.join(DATA, 'info.json'), os.path.join(BACKUP, 'info.json'))   # written last = backup complete
        print(f'backup of the original competition data (market, global, info.json) -> {BACKUP}', flush=True)
    work = tempfile.mkdtemp(prefix='mlrace_')
    if os.path.isdir(src):
        newroot = src
    else:
        zipfile.ZipFile(src).extractall(work); newroot = work
    hits = glob.glob(os.path.join(newroot, '**', 'global_data.csv'), recursive=True)
    if not hits:
        sys.exit('global_data.csv not found in the download')
    nd = os.path.dirname(os.path.dirname(hits[0]))                       # folder that contains market\ and global\
    info = json.load(open(os.path.join(DATA, 'info.json')))
    problems, warns = [], []          # problems stop the morning / evening run (exit code 1); warns are notes
    closed = load_closures()
    if closed:
        TRADING_DAYS[:] = fix_calendar(closed, warns)
    before = pd.read_csv(os.path.join(DATA, 'market', f"{info['Top50@Y'][0]}.csv"))
    last_complete_before = before.loc[~is_placeholder(before), 'date'].max()
    g, n_old_g, n_new_g = merge_file(os.path.join(DATA, 'global', 'global_data.csv'), hits[0], closed)
    first_new = pd.read_csv(hits[0]).date.min()
    if (pd.Timestamp(first_new) - pd.Timestamp(last_complete_before)).days > 4:
        warns.append(f'possible gap: your data ended {last_complete_before} but the download starts {first_new} '
                        f'(fine if only holidays lie in between; otherwise download a zip that covers the missing days)')
    files = sorted(glob.glob(os.path.join(nd, 'market', '*.csv')))
    for f in files:
        sid = os.path.basename(f)[:-4]
        old_path = os.path.join(DATA, 'market', f'{sid}.csv')
        old = pd.read_csv(old_path) if os.path.exists(old_path) else None
        new = pd.read_csv(f)
        m, n_old, n_new = merge_file(old_path, f, closed)
        if old is not None:                                   # the same day in both versions: the adjustment base must not move
            o = old[~is_placeholder(old)].drop_duplicates('date', keep='last').set_index('date')
            n_ = new[~is_placeholder(new)].drop_duplicates('date', keep='last').set_index('date')
            common = o.index.intersection(n_.index)
            if len(common):
                rel = max(float((n_.loc[common, k] / o.loc[common, k] - 1).abs().max()) for k in ('etl:adj_close', 'etl:adj_open'))
                if rel > 1e-6:
                    problems.append(f'{sid}: the download changes adj prices of days already in data\\ ({len(common)} days, max {rel:.1e}) '
                                    f'-> the server changed the price-adjustment base, old rows no longer fit: ask the TA or re-download the full history')
        comp = m[~is_placeholder(m)].reset_index(drop=True)
        c = comp['etl:adj_close']
        if len(comp) >= 30 and abs(c.tail(30).mean() / comp.SMA.iloc[-1] - 1) > 1e-6:
            warns.append(f'{sid}: SMA(30) from adj_close differs from the server SMA on the newest day (normal for about 30 days after a trading halt)')
        if ((comp['etl:adj_close'] > comp['etl:adj_open']).astype(int) != comp.y).tail(len(new)).any():
            problems.append(f'{sid}: y inconsistent with adj_close > adj_open in new rows')
    for d in sorted({x[1] for x in CLOSURE_LOG}):
        mv = [x for x in CLOSURE_LOG if x[1] == d]; n_mv = sum(1 for x in mv if x[2])
        to = sorted({x[2] for x in mv if x[2]})
        warns.append(f'closed day {d} (data\\ref\\closures_manual.csv): rows dropped in {len(mv) - n_mv} file(s)'
                     + (f'; its -1 placeholder became the placeholder of {", ".join(to)} in {n_mv} file(s)' if n_mv else ''))
    if CLOSURE_LOG and not TRADING_DAYS:
        problems.append('closed days are listed but data\\ref\\calendar.parquet cannot be read, so the day to predict was not '
                        'moved to the next trading day (pip install pyarrow, then run again)')
    missing_sids = set(info['Top100@Y']) - {os.path.basename(f)[:-4] for f in files}
    if missing_sids:
        problems.append(f'zip has no file for: {sorted(missing_sids)}')
    ref = pd.read_csv(os.path.join(DATA, 'market', f"{info['Top50@Y'][0]}.csv"))
    ph = ref.loc[is_placeholder(ref), 'date'].tolist(); complete = ref.loc[~is_placeholder(ref), 'date']
    miss_g = sorted(set(ref.date) - set(g.date))
    if [d for d in miss_g if d >= first_new]:
        warns.append(f'market dates without a global row: {[d for d in miss_g if d >= first_new][:5]} (normal after U.S. holidays)')
    shutil.rmtree(work, ignore_errors=True)
    print(f'merged {len(files)} stock files + global_data.csv; data now covers {ref.date.min()} .. {ref.date.max()}')
    print(f'last complete trading day: {complete.max()}')
    print(f'day(s) to predict (placeholder rows): {ph if ph else "none in this download"}')
    kept = archive_zip(src, ph[0] if ph else complete.max())
    if kept:
        print(f'zip kept as {os.path.relpath(kept, ROOT)}')
    print('checks: OK' if not problems else 'CHECK THESE (they stop daily.bat / evening.bat):\n  ' + '\n  '.join(problems))
    if warns:
        print('notes (nothing to do if expected):\n  ' + '\n  '.join(warns))
    if problems and not accept:
        sys.exit('update_data.py: stopped with exit code 1 because of the problems above (the zip is merged already). '
                 'If you have checked that they are false alarms, rerun with --accept (daily.bat accept / evening.bat accept).')

if __name__ == '__main__':
    main()

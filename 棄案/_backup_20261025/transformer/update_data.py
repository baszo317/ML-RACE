r"""
update_data.py -- merge the daily data.zip downloaded from the course server into ..\data (run every morning)

  python update_data.py                       # newest data*.zip in ..\downloads or in your Downloads folder
  python update_data.py C:\path\to\data.zip   # or a specific zip / an already extracted folder

What it does
  1. first run only: copies ..\data to ..\data_backup_original (an untouched copy of the original competition data)
  2. for data\market\*.csv and data\global\global_data.csv: old rows + rows from the zip; if a date appears in both,
     the zip wins (so yesterday's "-1" placeholder is replaced by the real values); sorted by date
  3. checks: SMA(30) recomputed from adj_close equals the server's SMA on the newest complete day (same price
     adjustment base), y == (adj_close > adj_open) on complete rows, every market date has a global row,
     no complete trading day missing between the old and the new rows
  4. prints the last complete trading day and the day to predict (the date whose rows are all -1)
Nothing is downloaded by this script; it only uses the zip you downloaded from the course server.
"""
import os, sys, glob, json, shutil, zipfile, tempfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data'); BACKUP = os.path.join(ROOT, 'data_backup_original')

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

def _lines(path):
    with open(path, encoding='utf-8-sig') as f:
        rows = [ln.rstrip('\r\n') for ln in f if ln.strip()]
    return rows[0], {ln.split(',', 1)[0]: ln for ln in rows[1:]}       # header, {date: original text line}

def merge_file(old_path, new_path):
    """Merge at the text level: untouched old rows keep their exact original text (no float re-formatting);
    dates present in the download are taken from the download (this replaces yesterday's -1 placeholder)."""
    new_head, new_rows = _lines(new_path)
    old_head, rows = _lines(old_path) if os.path.exists(old_path) else (new_head, {})
    if old_head.split(',') != new_head.split(','):
        sys.exit(f'column mismatch in {os.path.basename(new_path)}:\n  {old_head}\n  {new_head}')
    n_old = len(rows); rows.update(new_rows)
    tmp = old_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
        f.write(old_head + '\n')
        for d in sorted(rows):
            f.write(rows[d] + '\n')
    os.replace(tmp, old_path)
    return pd.read_csv(old_path), n_old, len(new_rows)

def main():
    src = find_zip(sys.argv[1] if len(sys.argv) > 1 else None)
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
    problems = []
    before = pd.read_csv(os.path.join(DATA, 'market', f"{info['Top50@Y'][0]}.csv"))
    last_complete_before = before.loc[~is_placeholder(before), 'date'].max()
    g, n_old_g, n_new_g = merge_file(os.path.join(DATA, 'global', 'global_data.csv'), hits[0])
    first_new = pd.read_csv(hits[0]).date.min()
    if (pd.Timestamp(first_new) - pd.Timestamp(last_complete_before)).days > 4:
        problems.append(f'possible gap: your data ended {last_complete_before} but the download starts {first_new} '
                        f'(fine if only holidays lie in between; otherwise download a zip that covers the missing days)')
    files = sorted(glob.glob(os.path.join(nd, 'market', '*.csv')))
    for f in files:
        sid = os.path.basename(f)[:-4]
        m, n_old, n_new = merge_file(os.path.join(DATA, 'market', f'{sid}.csv'), f)
        comp = m[~is_placeholder(m)].reset_index(drop=True)
        c = comp['etl:adj_close']
        if len(comp) >= 30 and abs(c.tail(30).mean() / comp.SMA.iloc[-1] - 1) > 1e-6:
            problems.append(f'{sid}: SMA(30) from adj_close differs from the server SMA -> price adjustment base changed? re-download full history')
        if ((comp['etl:adj_close'] > comp['etl:adj_open']).astype(int) != comp.y).tail(len(pd.read_csv(f))).any():
            problems.append(f'{sid}: y inconsistent with adj_close > adj_open in new rows')
    missing_sids = set(info['Top100@Y']) - {os.path.basename(f)[:-4] for f in files}
    if missing_sids:
        problems.append(f'zip has no file for: {sorted(missing_sids)}')
    ref = pd.read_csv(os.path.join(DATA, 'market', f"{info['Top50@Y'][0]}.csv"))
    ph = ref.loc[is_placeholder(ref), 'date'].tolist(); complete = ref.loc[~is_placeholder(ref), 'date']
    miss_g = sorted(set(ref.date) - set(g.date))
    if [d for d in miss_g if d >= first_new]:
        problems.append(f'market dates without a global row: {[d for d in miss_g if d >= first_new][:5]}')
    shutil.rmtree(work, ignore_errors=True)
    print(f'merged {len(files)} stock files + global_data.csv; data now covers {ref.date.min()} .. {ref.date.max()}')
    print(f'last complete trading day: {complete.max()}')
    print(f'day(s) to predict (placeholder rows): {ph if ph else "none in this download"}')
    print('checks: OK' if not problems else 'CHECK THESE:\n  ' + '\n  '.join(problems))

if __name__ == '__main__':
    main()

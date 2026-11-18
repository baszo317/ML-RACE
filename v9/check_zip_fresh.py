r"""
check_zip_fresh.py -- does the day to predict have last night's U.S. close? (2026-10-27; only warns, changes nothing)

The row of the day to predict D carries the U.S. close of the night before D (^VIX, ^IXIC, ^GSPC; the U.S. session ends at
04:00-05:00 Taipei time). The course server packs a zip after the Taiwan close (10/23 18:18, 10/27 15:52) and in the morning
(10/23 07:00); only a zip packed on D after 05:00 can carry last night's U.S. close. On 10/27 the data came from the zip packed
10/23 18:18 (the typhoon day's row moved to 10/27), so the U.S. numbers were those of 10/22, two U.S. sessions old
(v11\results\rerun_1027_us\README.txt: about 4 bp that day).

  python check_zip_fresh.py          daily.bat and ..\v11\daily_v11.bat run it at the end; evening runs do not (the
                                     preliminary file is expected to miss the night session and last night's U.S. close)
Which zip: the newest copy of the day in ..\downloads\archive\data_<day>[_N].zip (update_data.py keeps one per merged zip),
otherwise the newest data*.zip in ..\downloads or in your Downloads folder. Its packing time is the newest time stamp
inside the zip (course server time = Taipei time). The line is also appended to the day's checks file(s) of v9 and v11.
"""
import os, sys, glob, zipfile, datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data'); DL = os.path.join(ROOT, 'downloads')


def day_to_predict():
    """newest date whose global_data.csv row has ^TWII = -1 (the placeholder of the day to predict), or None"""
    f = os.path.join(DATA, 'global', 'global_data.csv')
    last = None
    with open(f, encoding='utf-8-sig') as fh:
        head = fh.readline().strip().split(',')
        i = head.index('^TWII')
        for line in fh:
            p = line.strip().split(',')
            if len(p) > i and p[i].strip() in ('-1', '-1.0'):
                last = p[0]
    return dt.datetime.strptime(last, '%Y-%m-%d') if last else None


def zip_for(day):
    arch = glob.glob(os.path.join(DL, 'archive', f'data_{day:%Y-%m-%d}*.zip'))
    if arch:
        return max(arch, key=os.path.getmtime)
    c = glob.glob(os.path.join(DL, 'data*.zip')) + glob.glob(os.path.join(os.path.expanduser('~'), 'Downloads', 'data*.zip'))
    return max(c, key=os.path.getmtime) if c else None


def packed(z):
    """newest time stamp inside the zip, and whether its global_data.csv has a row for the day (filled in by the caller)"""
    with zipfile.ZipFile(z) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        t = max(dt.datetime(*i.date_time) for i in infos)
        g = [i for i in infos if i.filename.replace('\\', '/').endswith('global/global_data.csv')]
        rows = zf.read(g[0]).decode('utf-8-sig').splitlines() if g else []
    return t, rows


def main():
    try:
        sys.stdout.reconfigure(errors='replace')
    except Exception:
        pass
    d = day_to_predict()
    if d is None:
        print('[INFO] U.S. data check: no day to predict (-1 row) in the data')
        return
    z = zip_for(d)
    if z is None:
        msg = f'[WARN] U.S. data check: no zip found for {d:%Y-%m-%d} in ..\\downloads; cannot tell how old the U.S. numbers are'
    else:
        t, rows = packed(z)
        has_row = any(r.startswith(f'{d:%Y-%m-%d},') for r in rows)
        cutoff = d.replace(hour=5, minute=0)
        if t >= cutoff and has_row:
            msg = (f'[OK] U.S. data for {d:%Y-%m-%d}: zip {os.path.basename(z)} was packed {t:%Y-%m-%d %H:%M}, after the U.S. close '
                   f'of the night before')
        else:
            why = (f'was packed {t:%Y-%m-%d %H:%M}, before the U.S. close of the night before {d:%m/%d}' if t < cutoff else
                   f'has no row for {d:%Y-%m-%d} (the row was moved from a closed day)')
            msg = (f'[WARN] U.S. data for {d:%Y-%m-%d}: zip {os.path.basename(z)} {why}, so ^VIX ^IXIC ^GSPC are older than last night. '
                   f'If the course site has a newer zip (it also packs one around 07:00), download it into ..\\downloads and run '
                   f'daily again. (10/27: about 4 bp; v11\\results\\rerun_1027_us)')
    print('\n' + msg)
    tag = f'{d:%m%d}'
    for f in (os.path.join(HERE, 'out', f'checks_v9_{tag}.txt'), os.path.join(ROOT, 'v11', 'out', f'checks_v11_{tag}.txt')):
        if os.path.exists(f):
            try:
                with open(f, 'a', encoding='utf-8') as fh:
                    fh.write(msg + '\n')
            except OSError:
                pass


if __name__ == '__main__':
    main()

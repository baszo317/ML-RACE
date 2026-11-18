"""Build the course zip as it would have been on the morning of day D (packed 07:00), from a later zip.

  python make_zip_asof.py SRC.zip DST.zip 2026-10-30

Every file keeps its header and its rows dated before D byte for byte; the row of D becomes the placeholder the course
server writes in the morning zip:
  market\\*.csv           every value -1
  global\\global_data.csv  ^VIX ^IXIC ^GSPC as in SRC (the U.S. close of the night before D), ^TWII ^KS11 TX_OI -1
Rows after D are dropped. Entry names, order, compression and file attributes are kept; the time stamps are D 07:00.
"""
import sys, zipfile

src, dst, day = sys.argv[1], sys.argv[2], sys.argv[3]
stamp = tuple(int(x) for x in day.split('-')) + (7, 0, 0)
US = ('^VIX', '^IXIC', '^GSPC')

zin = zipfile.ZipFile(src)
n_market = 0
with zipfile.ZipFile(dst, 'w') as zout:
    for info in zin.infolist():
        raw = zin.read(info.filename)
        if info.is_dir() or not info.filename.endswith('.csv'):
            out = raw
        else:
            text = raw.decode('utf-8')
            assert '\r\n' in text and not text.startswith('﻿'), info.filename
            lines = text.split('\r\n')
            assert lines[-1] == '', info.filename          # file ends with CRLF
            head, rows = lines[0], [l for l in lines[1:-1] if l]
            cols = head.split(',')
            keep = [l for l in rows if l[:10] < day]
            today = [l for l in rows if l[:10] == day]
            assert len(today) == 1, (info.filename, len(today))
            vals = today[0].split(',')
            assert len(vals) == len(cols), info.filename
            if info.filename.endswith('global/global_data.csv'):
                ph = [day] + [vals[i] if c in US else '-1' for i, c in enumerate(cols) if i > 0]
                assert all(float(vals[cols.index(c)]) > 0 for c in US), 'no U.S. close in the source row'
            else:
                ph = [day] + ['-1'] * (len(cols) - 1)
                n_market += 1
            out = ('\r\n'.join([head] + keep + [','.join(ph)]) + '\r\n').encode('utf-8')
        zi = zipfile.ZipInfo(info.filename, date_time=stamp)
        zi.compress_type = info.compress_type
        zi.create_system = info.create_system
        zi.external_attr = info.external_attr
        zout.writestr(zi, out)
print(f'{dst}: {len(zin.infolist())} entries ({n_market} market files), rows before {day} kept, {day} = morning placeholder, packed {stamp}')

r"""
backup_v9.py -- copy D:\ML RACE to a second place; only new or changed files are copied, nothing is ever deleted there

  python backup_v9.py                       -> %USERPROFILE%\ML_RACE_backup (C: drive, a different disk than D:)
  python backup_v9.py E:\ML_RACE_backup     -> e.g. a USB stick or a private cloud-synced folder
  python backup_v9.py --dry                 -> only count what would be copied

evening.bat runs it at the end. What is copied: everything except the research caches and the unused archive
(棄案, v7\cache, v8\cache, v9\cache, v2\results_v2\wf, the dry-run test folders, __pycache__). The first run copies
about 0.2 GB, later runs only the files of the day (seconds). Keep the backup private: the course data may not be
shared with anyone outside the course.
"""
import os, sys, shutil, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
SKIP = {os.path.normcase(os.path.join(ROOT, *p)) for p in
        (('棄案',), ('v7', 'cache'), ('v8', 'cache'), ('v9', 'cache'), ('v2', 'results_v2', 'wf'),
         ('v9', 'models_dry'), ('v9', 'out_dryrun'))}

def main():
    args = [x for x in sys.argv[1:] if x != '--dry']; dry = '--dry' in sys.argv[1:]
    dst_root = os.path.abspath(args[0] if args else os.path.join(os.path.expanduser('~'), 'ML_RACE_backup'))
    if os.path.normcase(dst_root).startswith(os.path.normcase(ROOT + os.sep)) or os.path.normcase(dst_root) == os.path.normcase(ROOT):
        sys.exit(f'the backup folder must be outside {ROOT}')
    n = nbytes = skipped = 0; t0 = time.time()
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x != '__pycache__' and os.path.normcase(os.path.join(d, x)) not in SKIP]
        out = os.path.join(dst_root, os.path.relpath(d, ROOT))
        for fn in files:
            if fn.endswith('.tmp'):
                continue
            s, t = os.path.join(d, fn), os.path.join(out, fn)
            try:
                st = os.stat(s)
                if os.path.exists(t):
                    tt = os.stat(t)
                    if tt.st_size == st.st_size and int(tt.st_mtime) >= int(st.st_mtime):
                        skipped += 1
                        continue
                if not dry:
                    os.makedirs(out, exist_ok=True)
                    shutil.copy2(s, t)
                n += 1; nbytes += st.st_size
            except OSError as e:
                print(f'could not copy {s}: {e}')
    verb = 'would copy' if dry else 'copied'
    print(f'backup {dst_root}: {verb} {n} file(s), {nbytes / 1e6:.1f} MB ({skipped} unchanged) in {time.time() - t0:.0f}s')

if __name__ == '__main__':
    main()

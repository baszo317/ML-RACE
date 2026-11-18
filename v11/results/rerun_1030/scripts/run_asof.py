"""Run the user's daily program on a copied ML RACE tree as if today were ASOF (a past trading day).

  python run_asof.py ASOF v9|v11 ROOT [arguments of train_and_predict_v9.py / train_and_predict_v11.py]

Only the two date checks that compare with the clock get ASOF as "today":
  check_not_past      (otherwise every past day is FATAL "is in the past")
  check_exdiv_recent  (age of the newest ex-dividend download, counted up to ASOF)
Everything else is the unchanged program in ROOT (paths are relative to ROOT, so nothing outside the copy is read or written).
"""
import os, sys

asof, which, root = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])
args = sys.argv[4:]
v9 = os.path.join(root, 'v9')
sys.path.insert(0, v9)
import train_and_predict_v9 as T9          # noqa: E402

assert os.path.dirname(os.path.abspath(T9.__file__)) == v9, T9.__file__
_cnp, _cer = T9.check_not_past, T9.check_exdiv_recent

def check_not_past(dates, rep, today=None):
    return _cnp(dates, rep, today=asof)

def check_exdiv_recent(data_dir, rep, max_days=3, today=None):
    return _cer(data_dir, rep, max_days=max_days, today=asof)

T9.check_not_past, T9.check_exdiv_recent = check_not_past, check_exdiv_recent
print(f'[run_asof] {which} on {root} as of {asof}: {" ".join(args)}', flush=True)
if which == 'v9':
    sys.argv = [os.path.join(v9, 'train_and_predict_v9.py')] + args
    T9.main()
elif which == 'v11':
    v11 = os.path.join(root, 'v11')
    sys.path.insert(0, v11)
    import train_and_predict_v11 as T11    # noqa: E402
    assert os.path.dirname(os.path.abspath(T11.__file__)) == v11 and T11.T9 is T9
    sys.argv = [os.path.join(v11, 'train_and_predict_v11.py')] + args
    T11.main()
else:
    sys.exit('which must be v9 or v11')

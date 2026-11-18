r"""
monitor_v9.py -- live results of v9 so far (v8 monitor_market.py on v9\out\history_predictions.csv)

  python monitor_v9.py                     every live day predicted by train_and_predict_v9.py whose close is known
  python monitor_v9.py --sub out\submission_v9_1027.csv ..\v5\out\submission_v5_1027.csv     any submission files

Per day (Top50, bp): AR (competition score), timing (net exposure x market move), beta_bet (hidden market bet of the
stock view), rest (stock picking not tied to the market). Run it after update_data.py has merged the newest zip.
"""
import os, sys, glob
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'v8'))
import monitor_market as MM

_orig = MM.read_predictions

def read_predictions(subs):
    if subs:
        files = sorted(set(f for s in subs for f in (glob.glob(s) or [s])))          # cmd.exe does not expand *
        return _orig(files)
    f = os.path.join(HERE, 'out', 'history_predictions.csv')
    if not os.path.exists(f):
        sys.exit('no v9\\out\\history_predictions.csv yet: run v9\\daily.bat first, or pass --sub <submission files>')
    H = pd.read_csv(f, dtype={'stock_id': str}, parse_dates=['date'])
    return H[['date', 'stock_id', 'pred']]

MM.read_predictions = read_predictions

if __name__ == '__main__':
    MM.main()

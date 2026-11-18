"""
check_submission.py -- verify a submission against the template before uploading
  python check_submission.py submission_ensemble.csv            (template defaults to ../sample_submission.csv)
Checks: same stock ids in the same order, same date columns in the same order, every value is 0 or 1, no blanks.
"""
import os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sub_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'submission_ensemble.csv')
tpl_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(HERE), 'sample_submission.csv')
s = pd.read_csv(sub_path, dtype=str); t = pd.read_csv(tpl_path, dtype=str)
s.columns = [c.lstrip('﻿') for c in s.columns]; t.columns = [c.lstrip('﻿') for c in t.columns]
problems = []
if list(s.columns) != list(t.columns): problems.append(f'columns differ: {set(s.columns) ^ set(t.columns) or "same set, different order"}')
if list(s['id']) != list(t['id']): problems.append('stock ids differ from the template (set or order)')
vals = s.drop(columns='id').stack(future_stack=True) if 'future_stack' in pd.DataFrame.stack.__code__.co_varnames else s.drop(columns='id').stack()
if s.drop(columns='id').isna().any().any(): problems.append('blank cells')
if not set(vals.unique()) <= {'0', '1'}: problems.append(f'values other than 0/1: {sorted(set(vals.unique()) - {"0", "1"})[:5]}')
print(f'{sub_path}: {s.shape[0]} rows x {s.shape[1] - 1} dates, share of 1 = {(vals == "1").mean():.3f}')
print('OK' if not problems else 'PROBLEMS:\n  ' + '\n  '.join(problems))
sys.exit(1 if problems else 0)

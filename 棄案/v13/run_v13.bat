@echo off
rem v13: three stock-view candidates trained and judged together on this laptop (v13\README_v13.md). Research only:
rem no daily file (v9, v11, W0) is changed.
rem   LSTM-R, CNN-R : LSTM / dilated 1D CNN on 60 days of raw price-volume-flow sequences   (research\PREREG_v13.md)
rem   F-LOAM        : 2x50 LSTM -> hidden state + B features -> LightGBM tuned by Optuna     (research\PREREG_v13b.md)
rem plus floam_checks.py: what the F-LOAM paper's protocol does on our data (price-level R2, random split, SVMD look-ahead).
rem Each model: 6 walk-forward folds x 5 seeds = 30 units (90 in all). Finished units are kept in cache\seq_raw:
rem close this window any time, double-click again and it continues where it stopped.
rem Time on the i5-10210U (estimates): checks 5-15 min; LSTM-R + CNN-R 1.5-3 hours; F-LOAM 1-2 hours.
rem Runs at below-normal priority and keeps Windows awake while it runs; plug in the charger and do not close the lid.
rem   double-click            -> checks, train all three, judge -> results\floam_checks.txt, results\v13_local_eval.txt
rem   run_v13.bat floam       -> checks, train F-LOAM only (LSTM-R / CNN-R from the cloud run), judge
rem   run_v13.bat test        -> 3-6 minute check that everything runs (tiny models, cache\seq_raw_test)
rem   run_v13.bat eval        -> only judge (whatever units are finished)
rem   run_v13.bat checks      -> only floam_checks.py
rem   run_v13.bat floam 1     -> same with 1 worker (less memory, slower); the number works for the default run too: run_v13.bat all 1
rem Progress: results\v13_train.log. This file stays ASCII so that cmd reads it on any code page.
cd /d "%~dp0research"
python -c "import torch" 2>nul
if errorlevel 1 (
  echo torch is missing. Install the CPU version with:  python -m pip install torch
  pause
  exit /b 1
)
python -c "import lightgbm" 2>nul || python -m pip install lightgbm
python -c "import optuna" 2>nul || python -m pip install optuna
python -c "import vmdpy" 2>nul || python -m pip install vmdpy
set "MODE=%~1"
if "%MODE%"=="" set "MODE=all"
set "W=2"
if not "%~2"=="" set "W=%~2"
if /i "%MODE%"=="test" goto :test
if /i "%MODE%"=="eval" goto :eval
if /i "%MODE%"=="checks" goto :checks
set "KINDS=lstm,tcn,floam"
if /i "%MODE%"=="floam" set "KINDS=floam"
echo Running floam_checks.py (about 5-15 minutes) ...
python floam_checks.py --workers %W% > ..\results\floam_checks.txt
if errorlevel 1 echo   floam_checks.py FAILED: see results\floam_checks.txt (the training continues)
python v13_train.py --kinds %KINDS% --workers %W% || goto :fail
:eval
python v13_eval.py > ..\results\v13_local_eval.txt
type ..\results\v13_local_eval.txt
echo.
echo Saved: v13\results\v13_local_eval.txt and v13\results\floam_checks.txt
pause
exit /b 0
:checks
python floam_checks.py --workers %W% > ..\results\floam_checks.txt
type ..\results\floam_checks.txt
pause
exit /b 0
:test
python v13_train.py --test --kinds lstm,tcn,floam --workers 2 || goto :fail
python v13_eval.py --test > ..\results\test_eval.txt
type ..\results\test_eval.txt
python floam_checks.py --start 2026-10-01 --workers 2
echo.
echo Test finished. If there is no error above, double-click run_v13.bat (or run_v13.bat floam) for the real run.
pause
exit /b 0
:fail
echo.
echo Stopped before all units were done (finished units are kept in v13\cache\seq_raw; see results\v13_train.log).
echo Double-click run_v13.bat again (or the same command) to continue.
pause
exit /b 1

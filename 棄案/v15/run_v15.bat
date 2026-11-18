@echo off
rem v15: de-noising methods and Random Forest - v15\README_v15.md. Research only: no daily file of v9, v11, W0 is changed.
rem   A  Random Forest, ExtraTrees, plus the sklearn-default RF for the report                   scikit-learn
rem   B  lgb_res de-noising: sigma weights, Huber weights, 3-day / 5-day target, no ex-dividend
rem      rows, DoubleEnsemble from Qlib; B0 = lgb_res retrained here as the control               lightgbm
rem   C  MLP de-noising: multi-task D, D+1, D+2 and supervised autoencoder + MLP; C0 = control    torch
rem   D  Kalman filter, causal wavelet - no training.   E  Vasicek beta shrinkage for the hedge
rem Rules written before any result: research\PREREG_v15.md
rem Finished units are kept in cache\units: close this window any time, double-click again and it continues.
rem Time on the i5-10210U, estimate: 40-60 minutes in all.
rem Runs at below-normal priority and keeps Windows awake while it runs; plug in the charger and do not close the lid.
rem   double-click            = quick test, train everything, signals, judge; report in results\v15_eval.txt
rem   run_v15.bat test        = only the quick test: tiny models in cache\units_test, the numbers mean nothing
rem   run_v15.bat eval        = only signals + judge, with whatever units are finished
rem   run_v15.bat all 1       = the default run with 1 worker: less memory, slower
rem Progress: results\v15_train.log. This file uses goto only, no parenthesized blocks, and stays ASCII.
cd /d "%~dp0research"
python -c "import lightgbm" 2>nul || python -m pip install lightgbm
python -c "import sklearn" 2>nul || python -m pip install scikit-learn
python -c "import torch" 2>nul
if errorlevel 1 goto :notorch
set "MODE=%~1"
if "%MODE%"=="" set "MODE=all"
set "W=2"
if not "%~2"=="" set "W=%~2"
if /i "%MODE%"=="test" goto :test
if /i "%MODE%"=="eval" goto :eval
echo [1/4] quick test of every model, 1-3 minutes ...
python v15_train.py --test --workers %W%
if errorlevel 1 goto :testfail
echo.
echo [2/4] training, about 40-60 minutes. Progress is also written to v15\results\v15_train.log
python v15_train.py --workers %W%
if errorlevel 1 goto :fail
:eval
echo.
echo [3/4] signals: Kalman, wavelet, Vasicek betas, about 1 minute ...
python v15_signals.py > ..\results\v15_signals.txt 2>&1
if errorlevel 1 goto :sigfail
echo [4/4] judging, about 1-3 minutes ...
python v15_eval.py > ..\results\v15_eval.txt 2> ..\results\v15_eval.err
if errorlevel 1 goto :evalfail
type ..\results\v15_eval.txt
echo.
echo Saved: v15\results\v15_eval.txt, v15_daily.csv, v15_selection.json, v15_signals.txt
pause
exit /b 0

:test
python v15_train.py --test --workers %W%
if errorlevel 1 goto :testfail
python v15_signals.py > ..\results\v15_signals.txt 2>&1
python v15_eval.py --test > ..\results\v15_test_eval.txt 2>&1
type ..\results\v15_test_eval.txt
echo.
echo Test finished. If there is no error above, double-click run_v15.bat for the real run.
pause
exit /b 0

:notorch
echo torch is missing. Install the CPU version with:  python -m pip install torch
pause
exit /b 1

:testfail
echo.
echo The quick test FAILED: see the messages above.
pause
exit /b 1

:sigfail
type ..\results\v15_signals.txt
echo.
echo v15_signals.py FAILED: see the messages above.
pause
exit /b 1

:evalfail
type ..\results\v15_eval.txt
type ..\results\v15_eval.err
echo.
echo v15_eval.py FAILED: see the messages above.
pause
exit /b 1

:fail
echo.
echo Stopped before all units were done. Finished units are kept in v15\cache\units, see results\v15_train.log
echo Double-click run_v15.bat again to continue.
pause
exit /b 1

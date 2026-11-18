@echo off
rem v14: quant methods not tried in v1-v13, judged on the cached walk-forward predictions (v14\README_v14.md). Research only:
rem no daily file (v9, v11, W0) is changed. Nothing is trained: only numpy and pandas are needed.
rem   E1  the real opening price of day D (upper bound of the 08:59 pre-open trial price; report only)
rem   E2  PCA statistical arbitrage (Avellaneda-Lee s-score, residual reversal)
rem   E3  industry / statistical-factor neutral stock view
rem   E4  investment-trust window dressing at quarter ends
rem   E5  1/N of all 14 cached stock-view models
rem Rules: research\PREREG_v14.md (written before the results).
rem   double-click          -> build the signals (about 30 s) and judge (about 2-4 minutes)
rem   run_v14.bat eval      -> only judge (cache\signals.pkl must exist)
rem   run_v14.bat signals   -> only build the signals and the leak check
rem Output: results\v14_signals.txt, results\v14_eval.txt, results\v14_daily.csv, results\v14_selection.json
rem This file stays ASCII so that cmd reads it on any code page.
cd /d "%~dp0research"
if not exist ..\results mkdir ..\results
if not exist ..\cache mkdir ..\cache
set "MODE=%~1"
if /i "%MODE%"=="eval" goto :eval
echo Building the signals (about 30 seconds) ...
python v14_signals.py > ..\results\v14_signals.txt || goto :fail
type ..\results\v14_signals.txt
if /i "%MODE%"=="signals" goto :done
:eval
echo.
echo Judging the candidates (about 2-4 minutes) ...
python v14_eval.py > ..\results\v14_eval.txt || goto :fail
type ..\results\v14_eval.txt
:done
echo.
echo Saved in v14\results\
pause
exit /b 0
:fail
echo.
echo FAILED: see the message above (results\v14_signals.txt or results\v14_eval.txt has the output so far)
pause
exit /b 1

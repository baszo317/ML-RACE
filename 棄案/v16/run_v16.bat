@echo off
rem v16: does a Kalman filter work in this competition? - v16\README_v16.md. Research only: no daily file of v9, v11, W0 is changed.
rem   K1  Kalman beta for the hedge (random-walk or mean-reverting beta; parameters by the filter's own likelihood)
rem   K2  time-varying coefficients of the timing view (TVP regression of the Top50 intraday mean on pbar')
rem   K3  each day v11 or W0, by a local-level filter on their daily AR difference
rem Rules written before any result: research\PREREG_v16.md
rem Nothing is trained: numpy and pandas only, about 1-3 minutes in all.
rem   double-click          = K1 betas + leak check, then judge; report in results\v16_eval.txt
rem   run_v16.bat eval      = only judge (cache\signals16.pkl must exist)
rem   run_v16.bat signals   = only the K1 betas and the leak check
rem   run_v16.bat posthoc   = the hindsight checks after the judgment (about 1 minute), results\v16_posthoc.txt
rem   run_v16.bat smoother  = part 2, the Kalman smoother (research\PREREG_v16b.md), about 1-2 minutes, results\v16_smoother.txt
rem This file uses goto only, no parenthesized blocks, and stays ASCII.
cd /d "%~dp0research"
if not exist ..\results mkdir ..\results
if not exist ..\cache mkdir ..\cache
set "MODE=%~1"
if /i "%MODE%"=="eval" goto :eval
if /i "%MODE%"=="posthoc" goto :posthoc
if /i "%MODE%"=="smoother" goto :smoother
echo [1/2] K1 Kalman betas and the leak check, about 30-60 seconds ...
python v16_signals.py > ..\results\v16_signals.txt 2>&1
if errorlevel 1 goto :sigfail
type ..\results\v16_signals.txt
if /i "%MODE%"=="signals" goto :done
:eval
echo.
echo [2/2] K2, K3 and the judgment, about 1-2 minutes ...
python v16_eval.py > ..\results\v16_eval.txt 2> ..\results\v16_eval.err
if errorlevel 1 goto :evalfail
type ..\results\v16_eval.txt
goto :done

:posthoc
echo Hindsight checks (report only), about 1 minute ...
python v16_posthoc.py > ..\results\v16_posthoc.txt 2> ..\results\v16_posthoc.err
if errorlevel 1 goto :phfail
type ..\results\v16_posthoc.txt
goto :done

:smoother
echo Part 2, the Kalman smoother (needs cache\signals16.pkl from the default run), about 1-2 minutes ...
python v16_smoother.py > ..\results\v16_smoother.txt 2> ..\results\v16_smoother.err
if errorlevel 1 goto :smfail
python v16_smoother_posthoc.py > ..\results\v16_smoother_posthoc.txt 2> ..\results\v16_smoother_posthoc.err
if errorlevel 1 goto :smfail
type ..\results\v16_smoother.txt
type ..\results\v16_smoother_posthoc.txt
goto :done

:done
echo.
echo Saved in v16\results\
pause
exit /b 0

:sigfail
type ..\results\v16_signals.txt
echo.
echo v16_signals.py FAILED: see the messages above.
pause
exit /b 1

:evalfail
type ..\results\v16_eval.txt
type ..\results\v16_eval.err
echo.
echo v16_eval.py FAILED: see the messages above.
pause
exit /b 1

:phfail
type ..\results\v16_posthoc.err
echo.
echo v16_posthoc.py FAILED: see the messages above.
pause
exit /b 1

:smfail
type ..\results\v16_smoother.err
type ..\results\v16_smoother_posthoc.err
echo.
echo v16_smoother.py or v16_smoother_posthoc.py FAILED: see the messages above.
pause
exit /b 1

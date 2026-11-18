@echo off
rem RACE-STD v1 (research\PREREG_race_std.md): one validation standard for merging the four team systems.
rem The official run was done in the Linux VM of the Claude app on this computer (results\, cache\).
rem   double-click                 = reproduce everything on Windows into results_win\ and cache_win\ (does NOT touch the
rem                                  official results; about 1.5-3 hours on the i5-10210U with 2 workers)
rem   run_race_std.bat test        = quick test of every step (tiny models, numbers mean nothing), a few minutes
rem   run_race_std.bat audit       = recompute the official V result from the official units and compare (no training)
rem Needs: python with numpy, pandas, lightgbm, scikit-learn, scipy, pyarrow; the crawled data in race_std\ext
rem (run_fetch.bat). Finished units are kept: close the window any time, double-click again and it continues.
rem This file stays ASCII and uses goto only.
cd /d "%~dp0research"
python -c "import lightgbm, sklearn, scipy, pandas" 2>nul || goto :nomods
set "MODE=%~1"
if "%MODE%"=="" set "MODE=all"
if /i "%MODE%"=="audit" goto :audit
if /i "%MODE%"=="test" goto :test
set "RACE_TAG=win"
echo [1/8] panel and leak checks ...
python build_panel.py || goto :fail
python build_panel.py --ic || goto :fail
python build_panel.py --leak 0 9 || goto :fail
python build_panel.py --leak-u 0 9 || goto :fail
echo [2/8] Stage 1 training on D ...
python train.py --stage 1 --workers 2 || goto :fail
echo [3/8] grid on D, L* ...
python stage1_eval.py || goto :fail
echo [4/8] ablations on D ...
python train.py --stage 1b --workers 2 || goto :fail
echo [5/8] all 32 candidates on D ...
python stage1_eval.py --final > ..\results_win\stage1_D_console.txt || goto :fail
echo [6/8] freeze the finalists ...
python freeze.py || goto :fail
echo [7/8] finalists on V ...
python train.py --stage 2 --workers 2 || goto :fail
echo [8/8] the single validation ...
python stage2_validate.py || goto :fail
type ..\results_win\stage2_V.txt
echo.
echo Saved in race_std\results_win. Compare with the official race_std\results\stage2_V.txt
pause
exit /b 0

:test
set "RACE_TAG=win"
python build_panel.py || goto :fail
python train.py --stage 1 --test --workers 2 || goto :fail
python stage1_eval.py --test || goto :fail
python train.py --stage 1b --test --workers 2 || goto :fail
python stage1_eval.py --final --test || goto :fail
python freeze.py --test || goto :fail
python train.py --stage 2 --test --workers 2 || goto :fail
python stage2_validate.py --test || goto :fail
echo.
echo Test finished. If there is no error above, double-click run_race_std.bat for the real reproduction.
pause
exit /b 0

:audit
set "RACE_TAG="
python stage2_validate.py --audit || goto :fail
pause
exit /b 0

:nomods
echo A module is missing:  python -m pip install lightgbm scikit-learn scipy pandas pyarrow
pause
exit /b 1

:fail
echo.
echo Stopped with an error: see the messages above. Finished units are kept; double-click again to continue.
pause
exit /b 1

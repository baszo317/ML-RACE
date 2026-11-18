@echo off
rem v11 with your own parameters (README_v11.md section 8). Press Enter to keep the value in [brackets].
rem  1) shows how these parameters did in the walk-forward 2022-01..2026-10 next to v11 and v9 (research\v11_try.py, about a minute)
rem  2) writes out\submission_v11_MMDD_<parameters>.csv for the next trading day with the models already trained.
rem     The normal out\submission_v11_MMDD.csv is never replaced. Run daily_v11.bat or evening_v11.bat first on that day.
cd /d "%~dp0"
set "WT=0.5"
set /p "WT=w_T, market-view weight (0 = off)                     [0.5]: "
set "LAM=0.75"
set /p "LAM=lambda, beta hedge (1 = off, smaller = hedge more)    [0.75]: "
set "NW=0.5"
set /p "NW=T2 night-session reversal weight (0 = off)           [0.5]: "
set "EX=5"
set /p "EX=T3 extra 1s on the first 3 days of a month (0 = off) [5]: "
set "X1=y"
set /p "X1=X1 second LightGBM, y or n                           [y]: "
set "X1ARG="
if /i "%X1%"=="n" set "X1ARG=--no-x1"
set "ARGS=--w-t %WT% --lam %LAM% --night-w %NW% --extra-ones %EX% %X1ARG%"
echo.
echo === how these parameters did in 2022-01..2026-10 (walk-forward) ===
python research\v11_try.py %ARGS%
echo.
set "GO=y"
set /p "GO=write the file for the next trading day with these parameters? y or n [y]: "
if /i not "%GO%"=="y" goto :end
python train_and_predict_v11.py --dates auto --retrain none %ARGS%
echo.
echo The file just written is the one on the "wrote ..." line above: its name has the predicted date and the parameters.
:end
pause

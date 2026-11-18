@echo off
rem v9 morning routine (after 05:00, before 09:00). If evening.bat ran last evening the models are already trained and this
rem takes about 2 minutes; otherwise it trains them first (about 7 minutes on the laptop, measured 2026-10-25).
rem Steps: merge the newest zip, ex-dividend table, TX night session (it closes 05:00), input checks, predict,
rem ex-dividend rules, beta hedge, output checks, live results so far.
rem If update_data.py reports CHECK THESE and you have checked that they are false alarms, run from a cmd window:  daily.bat accept
cd /d "%~dp0"
set "UDARG="
if /i "%~1"=="accept" set "UDARG=--accept"
python -c "import lightgbm, pyarrow" 2>nul || python -m pip install lightgbm pyarrow
python ..\transformer\update_data.py %UDARG% || goto :fail
python ..\v5\fetch_exdiv.py
python ..\v7\fetch\fetch_night_session.py
python train_and_predict_v9.py --dates auto || goto :fail
python monitor_v9.py
echo.
echo Upload the newest out\submission_v9_MMDD.csv  (checks: out\checks_v9_MMDD.txt)
pause
exit /b 0
:fail
echo.
echo Stopped - read the [FATAL] or CHECK THESE lines above (also in out\checks_v9_MMDD.txt). No final file was written.
echo  - short of time: upload last evening's out\submission_v9_MMDD_prelim.csv (same models, only the night session missing)
echo  - "is in the past" or zip not merged: download the newest zip (released about 17:00) into ..\downloads and run again
echo  - update_data.py printed CHECK THESE: read them; if they are false alarms run  daily.bat accept  from a cmd window
echo  - ex-dividend table not fetched for several days: run  python ..\v5\fetch_exdiv.py  again; if TWSE is down, put known ex-dates in ..\data\ref\exdiv_manual.csv
echo  - after checking that it is a false alarm:  python train_and_predict_v9.py --dates auto --force
echo  - predict with the models already trained (about 1 minute):  python train_and_predict_v9.py --dates auto --retrain none
echo  - emergency fallback: ..\v5\daily.bat (never run on this laptop; its first run also trains a Transformer)
pause
exit /b 1

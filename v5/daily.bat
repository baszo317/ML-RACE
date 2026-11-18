@echo off
rem v5 morning routine (before 09:00): merge today's zip, ex-dividend table, input checks, newest models, predict, output checks, monitor
cd /d "%~dp0"
python ..\transformer\update_data.py || goto :fail
python fetch_exdiv.py
python train_and_predict_v5.py --dates auto || goto :fail
python monitor_live.py
echo.
echo Upload the newest out\submission_v5_MMDD.csv  (checks: out\checks_v5_MMDD.txt)
pause
exit /b 0
:fail
echo.
echo Stopped - read the [FATAL] lines above (also in out\checks_v5_MMDD.txt). Nothing was written for today.
echo  - zip not merged / not a trading day: download today's zip into ..\downloads and run daily.bat again
echo  - ex-dividend fetch failed: run  python fetch_exdiv.py  again; if TWSE is down, put known ex-dates in ..\data\ref\exdiv_manual.csv
echo  - after checking that it is a false alarm:  python train_and_predict_v5.py --dates auto --force
echo  - if time is short:                         python train_and_predict_v5.py --dates auto --retrain none
pause
exit /b 1

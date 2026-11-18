@echo off
rem v3 morning routine (before 09:00): merge today's zip, ex-dividend table, retrain what is stale, predict, monitor
cd /d "%~dp0"
python ..\transformer\update_data.py || goto :fail
python ..\v2\fetch_exdiv.py
python train_and_predict_v3.py --dates auto || goto :fail
python monitor_live.py
echo.
echo Upload the newest out\submission_v3_MMDD.csv
pause
exit /b 0
:fail
echo.
echo Something failed above - read the message. If time is short:  python train_and_predict_v3.py --dates auto --retrain none
pause
exit /b 1

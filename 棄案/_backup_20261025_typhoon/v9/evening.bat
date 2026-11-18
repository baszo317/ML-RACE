@echo off
rem v9 evening routine, after the course zip of the day is out (about 17:00, after the close):
rem merge the zip, ex-dividend table, input checks, train the new models (about 7 minutes on the laptop) and write the
rem PRELIMINARY file for the next trading day. The TX night session of that day closes at 05:00, so it counts as missing
rem here; next morning daily.bat adds it and only predicts (about 2 minutes). At the end: backup (backup_v9.py).
rem If update_data.py reports CHECK THESE and you have checked that they are false alarms, run from a cmd window:  evening.bat accept
cd /d "%~dp0"
set "UDARG="
if /i "%~1"=="accept" set "UDARG=--accept"
python -c "import lightgbm, pyarrow" 2>nul || python -m pip install lightgbm pyarrow
python ..\transformer\update_data.py %UDARG% || goto :fail
python ..\v5\fetch_exdiv.py
python train_and_predict_v9.py --dates auto --prelim || goto :fail
python backup_v9.py
echo.
echo Preliminary file for the next trading day: out\submission_v9_MMDD_prelim.csv  (checks: out\checks_v9_MMDD_prelim.txt)
echo  - if the course site lets you replace an upload, you can upload it now as insurance
echo  - next morning (after 05:00, before 09:00): run daily.bat and upload out\submission_v9_MMDD.csv
pause
exit /b 0
:fail
echo.
echo Stopped - read the [FATAL] or CHECK THESE lines above. No preliminary file was written.
echo  - "is in the past": the zip is an old one - download today's zip (released about 17:00) into ..\downloads and run again
echo  - update_data.py printed CHECK THESE: read them; if they are false alarms run  evening.bat accept  from a cmd window
echo  - otherwise just run daily.bat tomorrow morning as usual (it then trains the models itself, about 7 minutes)
pause
exit /b 1

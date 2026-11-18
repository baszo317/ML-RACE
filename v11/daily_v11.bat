@echo off
rem v11 morning routine (after 05:00, before 09:00) = v9's morning routine + v11 (README_v11.md part B).
rem v9 runs first: it merges the zip, fetches the ex-dividend table and the TX night session, trains v9's models if needed and
rem writes ..\v9\out\submission_v9_MMDD.csv. Then v11 adds its three changes (night-session reversal, month start, a second
rem LightGBM, about 1 minute) and writes out\submission_v11_MMDD.csv. Upload ONE of the two files.
rem If update_data.py reports CHECK THESE and you have checked that they are false alarms, run from a cmd window:  daily_v11.bat accept
cd /d "%~dp0"
set "UDARG="
if /i "%~1"=="accept" set "UDARG=--accept"
python -c "import lightgbm, pyarrow" 2>nul || python -m pip install lightgbm pyarrow
python ..\transformer\update_data.py %UDARG% || goto :fail
python ..\v5\fetch_exdiv.py
python ..\v7\fetch\fetch_night_session.py
python ..\v9\train_and_predict_v9.py --dates auto || goto :fail
python ..\v9\monitor_v9.py
python train_and_predict_v11.py --dates auto || goto :fail11
rem 2026-10-27: warn if the zip was packed before last night's U.S. close (only prints)
python ..\v9\check_zip_fresh.py
echo.
echo v9 : ..\v9\out\submission_v9_MMDD.csv
echo v11: out\submission_v11_MMDD.csv  (checks: out\checks_v11_MMDD.txt, last line = how many predictions differ from v9)
echo Upload ONE of the two.
pause
exit /b 0
:fail11
echo.
echo v9 finished (..\v9\out\submission_v9_MMDD.csv was written) but v11 stopped - read the lines above.
echo  - you can upload v9's file as usual
echo  - v11 with the models already trained (no training):  python train_and_predict_v11.py --dates auto --retrain none
pause
exit /b 1
:fail
echo.
echo Stopped in the v9 part: no v9 or v11 file was written. What to do is the same as for ..\v9\daily.bat:
echo  - short of time: upload last evening's ..\v9\out\submission_v9_MMDD_prelim.csv or out\submission_v11_MMDD_prelim.csv
echo  - "is in the past" or zip not merged: download the newest zip into ..\downloads and run again
echo  - typhoon day still asked for by the newest zip: add the date to ..\data\ref\closures_manual.csv and run again
echo  - update_data.py printed CHECK THESE: if they are false alarms run  daily_v11.bat accept  from a cmd window
echo  - FATAL "no U.S. close for this day yet": the zip is the one packed at midnight - download the morning zip (about 07:00)
echo    into ..\downloads and run again
pause
exit /b 1

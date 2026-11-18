@echo off
rem race version (race ver\race_live), every morning after 08:01 -- PREREG_race_live.md section 3
rem   1. download this morning's zip from the course server (data*.zip) into your Downloads folder or ..\..\downloads
rem   2. run this file; the upload file is out\submission_race_MMDD.csv (report: out\report_race_MMDD.txt)
rem If update_data.py reports CHECK THESE and you have checked that they are false alarms, run from a cmd window:
rem   race_daily.bat accept
rem FinMind token: set FINMIND_TOKEN=... once in Windows (or in the cmd window) before running.
cd /d "%~dp0"
set UDARG=
if /i "%~1"=="accept" set UDARG=--accept
set RACE_THREADS=%NUMBER_OF_PROCESSORS%
python -c "import lightgbm, pyarrow" 2>nul || python -m pip install lightgbm pyarrow
python ..\..\transformer\update_data.py %UDARG% || goto :fail
python ..\..\v5\fetch_exdiv.py
python ..\..\v7\fetch\fetch_night_session.py
python fetch_live.py --yahoo
python fetch_live.py --finmind --stale
python race_live.py predict || goto :fail
echo.
echo  done: upload out\submission_race_MMDD.csv  (read the WARN lines above, if any)
pause
exit /b 0
:fail
echo.
echo  STOPPED. What to do:
echo  - update_data.py printed CHECK THESE: read them; if they are false alarms run  race_daily.bat accept  from a cmd window
echo  - FATAL no -1 placeholder day: this morning's zip is not merged yet (download it, run again)
echo  - FATAL ... missing: the crawled data failed; run  python fetch_live.py --yahoo  again, then  python race_live.py predict
echo  - after checking a FATAL is a false alarm:  python race_live.py predict --force
pause
exit /b 1

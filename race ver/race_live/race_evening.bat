@echo off
rem race version (race ver\race_live), every evening after about 21:30 -- PREREG_race_live.md section 3
rem merges the evening zip, downloads the FinMind data of today (margin / short sale, SBL, day trading; published about
rem 21:30) and trains today's model, so the morning run only has to predict.
rem If update_data.py reports CHECK THESE and you have checked that they are false alarms:  race_evening.bat accept
cd /d "%~dp0"
set UDARG=
if /i "%~1"=="accept" set UDARG=--accept
set RACE_THREADS=%NUMBER_OF_PROCESSORS%
python -c "import lightgbm, pyarrow" 2>nul || python -m pip install lightgbm pyarrow
python ..\..\transformer\update_data.py %UDARG% || goto :fail
python ..\..\v5\fetch_exdiv.py
python ..\..\v7\fetch\fetch_night_session.py
python fetch_live.py --finmind
python race_live.py train || goto :fail
echo.
echo  done: today's model is trained; tomorrow morning run race_daily.bat
pause
exit /b 0
:fail
echo.
echo  STOPPED. update_data.py printed CHECK THESE: read them; if they are false alarms run  race_evening.bat accept
pause
exit /b 1

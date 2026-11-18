@echo off
rem v11 evening routine (after the course zip of the day, about 17:00) = v9's evening routine + v11 (README_v11.md part B):
rem merge the zip, ex-dividend table, v9 trains its new models (about 7 minutes) and writes ..\v9\out\submission_v9_MMDD_prelim.csv,
rem then v11 trains its second LightGBM (about 1 minute) and writes out\submission_v11_MMDD_prelim.csv, then the backup.
rem The TX night session closes at 05:00, so tonight's files have no night session (v11's night-session change adds nothing
rem tonight); next morning daily_v11.bat adds it.
rem 2026-10-28: the evening zips have no usable day to predict (no -1 row, or a -1 row without the U.S. close; the course
rem fills that row in the morning zip, about 07:00). Then v9 and v11 only train their models and write no files.
cd /d "%~dp0"
set "UDARG="
if /i "%~1"=="accept" set "UDARG=--accept"
python -c "import lightgbm, pyarrow" 2>nul || python -m pip install lightgbm pyarrow
python ..\transformer\update_data.py %UDARG% || goto :fail
python ..\v5\fetch_exdiv.py
python ..\v9\train_and_predict_v9.py --dates auto --prelim || goto :fail
python train_and_predict_v11.py --dates auto --prelim || goto :fail11
python ..\v9\backup_v9.py
echo.
echo If the zip had a usable day to predict: ..\v9\out\submission_v9_MMDD_prelim.csv and out\submission_v11_MMDD_prelim.csv.
echo Usually (since 10/27) it has none: then only the models were trained tonight (lines above) and there are no files.
echo  - next morning after about 07:00: download the newest zip into ..\downloads, run daily_v11.bat (only predicts,
echo    a few minutes) and upload ONE of the two final files
pause
exit /b 0
:fail11
echo.
echo v9's preliminary file was written, but v11 stopped - read the lines above; tomorrow morning daily_v11.bat tries again.
python ..\v9\backup_v9.py
pause
exit /b 1
:fail
echo.
echo Stopped in the v9 part (same messages as ..\v9\evening.bat); no preliminary file was written.
echo  - otherwise just run daily_v11.bat tomorrow morning as usual (it then trains the models itself)
pause
exit /b 1

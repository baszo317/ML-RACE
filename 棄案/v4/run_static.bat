@echo off
rem v4 static leaderboard (2026-06-01..09-30): every day is predicted only with models trained up to the previous trading day
rem models_v4_static\ already holds all 725 models and their predictions, so this only recombines them (about 1 minute);
rem to retrain everything from scratch, rename models_v4_static\ first (about 2-4 hours with 2 workers)
cd /d "%~dp0"
python fetch_exdiv.py --history 2026-06-01
python train_and_predict_v4.py --static --workers 2 || goto :fail
python research\compare_static.py
echo.
echo Upload out\submission_v4_static.csv
pause
exit /b 0
:fail
echo.
echo Something failed above - read the message. Finished models are kept: run run_static.bat again to continue.
pause
exit /b 1

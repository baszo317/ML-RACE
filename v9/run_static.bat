@echo off
rem v9 static leaderboard (2026-06-01..09-30) and its October extension: every day is predicted only with models trained
rem up to the previous trading day. out\members_v9_static.csv.gz holds the pooled probabilities of every member, so this
rem only recombines them (seconds). To retrain everything from scratch: python train_and_predict_v9.py --static
rem (Linux / WSL with --workers 2: about 1.5 hours; one process on Windows: several hours)
cd /d "%~dp0"
python train_and_predict_v9.py --static --from-members out\members_v9_static.csv.gz || goto :fail
echo.
echo Upload out\submission_v9_static.csv   (scores: out\static_report_v9.txt)
pause
exit /b 0
:fail
echo.
echo Something failed above - read the message.
pause
exit /b 1

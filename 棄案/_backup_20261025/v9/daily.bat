@echo off
rem v9 morning routine (before 09:00): merge today's zip, ex-dividend table, TX night session, input checks,
rem newest models (2 MLP seeds x 7 members, 1 Transformer, 1 LightGBM), predict, ex-dividend rules, beta hedge, output checks
cd /d "%~dp0"
python -c "import lightgbm, pyarrow" 2>nul || python -m pip install lightgbm pyarrow
python ..\transformer\update_data.py || goto :fail
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
echo Stopped - read the [FATAL] lines above (also in out\checks_v9_MMDD.txt). Nothing was written for today.
echo  - zip not merged / not a trading day: download today's zip into ..\downloads and run daily.bat again
echo  - ex-dividend fetch failed: run  python ..\v5\fetch_exdiv.py  again; if TWSE is down, put known ex-dates in ..\data\ref\exdiv_manual.csv
echo  - after checking that it is a false alarm:  python train_and_predict_v9.py --dates auto --force
echo  - if time is short (predict with the models already trained):  python train_and_predict_v9.py --dates auto --retrain none
echo  - emergency fallback: ..\v5\daily.bat (the v5 pipeline is unchanged)
pause
exit /b 1

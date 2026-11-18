@echo off
rem RACE-STD v1, amendment 1 (research\PREREG_race_std.md section 6): download the data of Chen's crawled V5 features
rem into race_std\ext. Only this step needs the internet; everything else runs in the Linux VM of the Claude app.
rem   Yahoo Finance: ADRs, U.S. sector ETFs, SOX, Nikkei, KOSPI, TWD, U.S. 10y yield, dollar index  (about 1 minute)
rem   FinMind: margin / short, SBL, day trading of the 100 stocks  (301 requests; without a token FinMind allows about
rem            300 an hour, the script then waits by itself and continues)
rem Faster, optional: open cmd, type  set FINMIND_TOKEN=your_token  (free account at finmindtrade.com), then run this
rem file from the same window.
rem Finished files are kept: close the window any time, double-click again and it continues.
rem This file stays ASCII and uses goto only.
cd /d "%~dp0research"
python -c "import pandas" 2>nul || goto :nopandas
python fetch_ext.py %*
echo.
echo Finished. Results: race_std\ext  (fetch_log.txt, fetch_status.json, FETCH_DONE.txt)
pause
exit /b 0

:nopandas
echo pandas is missing in this Python: python -m pip install pandas
pause
exit /b 1

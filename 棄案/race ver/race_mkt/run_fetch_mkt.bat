@echo off
rem race_mkt (2026-11-17): download futures / options positioning data (FinMind + TAIFEX) into race_mkt\ext.
rem Only this step needs the internet; the experiments run in the Linux VM of the Claude app.
rem About 3-6 minutes. Optional: set FINMIND_TOKEN=your_token in a cmd window first, then run this file from it.
rem This file stays ASCII and uses goto only.
cd /d "%~dp0"
python -c "import pandas" 2>nul || goto :nopandas
python fetch_mkt.py %*
echo.
echo Finished. Results: race ver\race_mkt\ext  (fetch_log.txt, fetch_status.json, FETCH_DONE.txt)
pause
exit /b 0

:nopandas
echo pandas is missing in this Python: python -m pip install pandas
pause
exit /b 1

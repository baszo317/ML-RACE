@echo off
rem race_news (2026-11-17): download cnyes Taiwan stock news (publish time + stock tags), 2019-10 .. yesterday, into race_news\ext\cnyes.
rem About 40-80 minutes. Finished months are kept: close the window any time, double-click again and it continues.
rem This file stays ASCII and uses goto only.
cd /d "%~dp0"
python -c "import pandas" 2>nul || goto :nopandas
python fetch_news.py %*
echo.
echo Finished. Results: race ver\race_news\ext  (fetch_log.txt, fetch_status.json, FETCH_DONE.txt)
pause
exit /b 0

:nopandas
echo pandas is missing in this Python: python -m pip install pandas
pause
exit /b 1

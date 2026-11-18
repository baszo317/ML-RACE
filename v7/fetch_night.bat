@echo off
rem v7 (live only): TAIEX futures (TX) daily prices incl. the night session from FinMind -> data\tx_daily.csv (about 1 minute)
rem optional free FinMind token (higher request limit): in a cmd window run  set FINMIND_TOKEN=your_token  then this file
cd /d "%~dp0fetch"
python fetch_night_session.py || goto :fail
echo.
echo Done: v7\data\tx_daily.csv  - the night-session test (README_v7 section 6) runs on this file.
pause
exit /b 0
:fail
echo.
echo Download failed - read the message above. If FinMind asks for a token or a limit is reached:
echo   1. register (free) at finmindtrade.com and copy the API token
echo   2. open cmd, run:  set FINMIND_TOKEN=your_token
echo   3. in the same cmd window run:  "D:\ML RACE\v7\fetch_night.bat"
pause
exit /b 1

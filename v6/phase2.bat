@echo off
rem v6 Phase 2 (dynamic leaderboard, scored daily from 2026-10-12): download the day's open and close of the 50 stocks
rem (FinMind; TWSE / TPEx if FinMind fails) and score live submissions with the competition metric, next to the board's
rem two baselines (Always down, Previous day). When the course data already has the day it is used instead (and the two
rem are compared if a download was saved). Writes phase2\report_YYYYMMDD.txt and phase2\scores.csv.
rem   double-click            -> newest day with prices: every one-day submission in v9\out, v11\out, phase2\files
rem   drag file(s) onto it    -> those files (each on the day in its header)
rem   from cmd:  phase2.bat --uploaded FILE.csv    score FILE.csv and remember it as the file uploaded for its day
rem              phase2.bat --date 2026-10-27      a given day
rem              phase2.bat --all                  every live day again (rebuilds the accumulated table)
rem              phase2.bat --board                accumulated table only (no download)
rem              phase2.bat --check                download even if the course data has the day, and compare
rem Prices come out after the 13:30 close (usually within one or two hours). Needs numpy and pandas only.
rem This file stays ASCII so that cmd reads it on any code page.
python "%~dp0phase2_check.py" %*
echo.
pause

@echo off
rem v6 research: recompute the static-leaderboard calibration table (README_v6 section 2), a few seconds
cd /d "%~dp0"
python s1_static_leaks.py > results\s1_static_leaks.txt || goto :fail
type results\s1_static_leaks.txt
pause
exit /b 0
:fail
echo Failed - read the message above.
pause
exit /b 1

@echo off
rem v10 stage 2 (research\PREREG_v10.md), after results\v10_selection.json exists: trains V10's changed members with the
rem selected Alpha158 groups (LightGBM seed 1-5, the three MLPs seed 1-10) and the v9 replica of the MLPs (seed 6-10), then
rem writes results\v10_eval.txt (V10 vs v9 replica: the pre-registered hold / post check).
rem Can be stopped and started again. Optional: stage2.bat --workers 6
cd /d "%~dp0research"
if not exist ..\results\v10_selection.json (
  echo results\v10_selection.json is missing: run the screen and a158_eval.py first.
  pause
  exit /b 1
)
python a158_screen.py --family v10 %* || goto :fail
python v10_eval.py > ..\results\v10_eval.txt || goto :fail
type ..\results\v10_eval.txt
pause
exit /b 0
:fail
echo.
echo Stopped - see the message above.
pause
exit /b 1

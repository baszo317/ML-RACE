@echo off
rem walk-forward of the race version (PREREG_race_live.md section 2) on this Windows PC, next to the VM run.
rem It fits the daily models of 2023 and 2022 with half of the logical processors and stops starting new work at 07:55
rem (the stopping rule is 08:00). Close the window to stop early; finished models are kept and nothing is redone.
cd /d "%~dp0"
set /a W=%NUMBER_OF_PROCESSORS%/2
if %W% LSS 1 set W=1
if %W% GTR 6 set W=6
set RACE_THREADS=1
python walkforward.py --prep || goto :fail
python walkforward.py --run2 --years 2023,2022 --workers %W% --until 07:55
python walkforward.py --status
pause
exit /b 0
:fail
echo  STOPPED: python walkforward.py --prep failed (read the message above)
pause
exit /b 1

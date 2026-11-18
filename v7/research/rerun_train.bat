@echo off
rem v7 research, optional: rebuild both panels and retrain all 270 MLPs on this laptop (one at a time, about 2 hours),
rem then recompute the tables. The saved results in ..\cache were trained in the cloud with the same code.
cd /d "%~dp0"
python e1_build_panel.py || goto :fail
python e2_train_ext.py --workers 1 || goto :fail
python e4_build_gap_panel.py || goto :fail
python e5_train_gap.py --workers 1 || goto :fail
python e7_build_night_panel.py > results\e7_build_night_panel.txt || goto :fail
python e8_train_night.py --workers 1 || goto :fail
call rerun_eval.bat
exit /b 0
:fail
echo Failed - read the message above. Finished models are kept in ..\cache: run this file again to continue.
pause
exit /b 1

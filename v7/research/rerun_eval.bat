@echo off
rem v7 research: recompute the result tables from the saved predictions in ..\cache (no training, a few minutes)
cd /d "%~dp0"
python e0_check_rebuild.py > results\e0_check_rebuild.txt || goto :fail
python e3_eval_h1.py > results\e3_eval_h1.txt || goto :fail
python e6_eval_gap.py > results\e6_eval_gap.txt || goto :fail
python r1_risk_and_exdiv.py > results\r1_risk_and_exdiv.txt || goto :fail
python e9_eval_night.py > results\e9_eval_night.txt || goto :fail
echo Done: results\e0_check_rebuild.txt, e3_eval_h1.txt, e6_eval_gap.txt, r1_risk_and_exdiv.txt, e9_eval_night.txt
pause
exit /b 0
:fail
echo Failed - read the message above (missing pyarrow: pip install pyarrow).
pause
exit /b 1

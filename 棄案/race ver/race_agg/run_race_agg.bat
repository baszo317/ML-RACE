@echo off
rem race_agg (race ver\race_agg): the aggregation methods of topic 6 / W12 on the race version. Research only; live is not changed.
rem   run_race_agg.bat        recompute the evaluation from the saved units (agg_eval, agg_posthoc, agg_check; about 3 minutes)
rem   run_race_agg.bat all    first train any missing unit (all 132 are saved; deleting cache\units_agg retrains all, about 1 hour)
rem   run_race_agg.bat diag   rerun the E3 out-of-bag diagnostic (about 30 minutes)
rem Results: results\agg_D.txt (rule), agg_posthoc.txt, agg_check.txt, agg_E3.txt. LightGBM 4.6.0 reproduces the saved units exactly.
cd /d "%~dp0research"
python -c "import lightgbm, sklearn, scipy" 2>nul || python -m pip install lightgbm==4.6.0 scikit-learn scipy
if /i "%~1"=="diag" goto diag
if /i "%~1"=="all" goto train
goto eval
:train
python agg_train.py --what all --workers 4
if errorlevel 1 goto fail
:eval
python agg_eval.py
if errorlevel 1 goto fail
python agg_posthoc.py
if errorlevel 1 goto fail
python agg_check.py
if errorlevel 1 goto fail
echo.
echo  done: results\agg_D.txt, results\agg_posthoc.txt, results\agg_check.txt
pause
exit /b 0
:diag
python agg_diag.py
if errorlevel 1 goto fail
echo.
echo  done: results\agg_E3.txt
pause
exit /b 0
:fail
echo.
echo  STOPPED: read the error above
pause
exit /b 1

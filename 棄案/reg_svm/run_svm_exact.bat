@echo off
rem Exact RBF SVM as an extra stock-view member (reg_svm\README_reg_svm.md section 1). Research only: no daily file changes.
rem Trains the 6 walk-forward folds, then scores the SVM as an extra member of v11 / v9 (results\svm_exact_eval.txt).
rem Time on the i5-10210U: about 3-6 hours with 2 folds at a time (default), 6-12 hours with 1; the last fold alone 2-4 hours.
rem Finished folds are kept in cache\svm_exact: close this window any time, double-click again and it continues.
rem Runs at below-normal priority (daily.bat / evening.bat can run at the same time) and keeps Windows awake while it runs;
rem plug in the charger and do not close the lid (closing the lid still puts the laptop to sleep).
rem   double-click                 -> 2 folds at a time
rem   run_svm_exact.bat 1          -> 1 fold at a time (less memory, about twice as long)
rem   run_svm_exact.bat test       -> 1-2 minute check that everything runs (tiny models, writes cache\svm_exact_test)
rem   run_svm_exact.bat eval       -> only the scoring (after all 6 folds are done)
rem Progress: results\svm_exact.log. This file stays ASCII so that cmd reads it on any code page.
cd /d "%~dp0research"
python -c "import sklearn" 2>nul || python -m pip install scikit-learn
if /i "%~1"=="eval" goto :eval
if /i "%~1"=="test" goto :test
set "W=2"
if not "%~1"=="" set "W=%~1"
python svm_exact.py --workers %W% || goto :fail
:eval
python svm_exact_eval.py > ..\results\svm_exact_eval.txt
type ..\results\svm_exact_eval.txt
echo.
echo Saved: reg_svm\results\svm_exact_eval.txt
pause
exit /b 0
:test
python svm_exact.py --test --workers 2 || goto :fail
python svm_exact_eval.py --test
echo.
echo Test finished. If there is no error above, double-click run_svm_exact.bat to start the real run.
pause
exit /b 0
:fail
echo.
echo Stopped before all folds were done (finished folds are kept in reg_svm\cache\svm_exact; see results\svm_exact.log).
echo Double-click run_svm_exact.bat again to continue.
pause
exit /b 1

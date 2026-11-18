@echo off
rem v6 (static only): compare static submissions with the competition metric (paired differences, 95% intervals, MDE)
rem   double-click            -> v4 vs v2 vs v2 plain (files already in this project)
rem   drag file(s) onto it    -> v4 vs those files, e.g. a group member's static upload (0/1 csv, same format as sample_submission)
rem   from cmd:  compare.bat member.csv --tried member.csv=20   (that file was picked among 20 settings by its static score)
if "%~1"=="" goto :default
python "%~dp0compare_static.py" "%~dp0..\v4\out\submission_v4_static.csv" %*
goto :end
:default
python "%~dp0compare_static.py" "%~dp0..\v4\out\submission_v4_static.csv" "%~dp0..\v2\out\submission_v2_static.csv" "%~dp0..\v2\out\submission_v2_plain_static.csv"
:end
echo.
pause

@echo off
rem v6 (static only): leak check for static uploads - format, answer-key segments (in-sample / copied y),
rem date shift, same-row (same-day) information, market timing. FATAL / WARN / OK per file.
rem   double-click            -> check every csv in v6\others
rem   drag file(s) onto it    -> check those files
rem   from cmd:  leak_check.bat member.csv --out report.txt
if "%~1"=="" goto :default
python "%~dp0leak_check.py" %*
goto :end
:default
python "%~dp0leak_check.py" "%~dp0others\*.csv"
:end
echo.
pause

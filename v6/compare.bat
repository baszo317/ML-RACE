@echo off
rem v6 (static only): compare static submissions with the competition metric (paired differences, 95% intervals, MDE)
rem   double-click            -> v4 vs v2 vs v2 plain (files already in this project)
rem   drag file(s) onto it    -> v4 vs those files, e.g. a group member's static upload (0/1 csv, same format as sample_submission)
rem   from cmd:  compare.bat member.csv --tried member.csv=20   (that file was picked among 20 settings by its static score)
rem 2026-10-26: v2 and v4 (unused) were moved into the abandoned-files folder; their static files are looked up in the
rem old place first, then in every folder next to v6 (this file stays ASCII so that cmd reads it on any code page)
setlocal
set "V4OUT="
set "V2OUT="
for /d %%D in ("%~dp0..\*") do (
  if exist "%%~fD\v4\out\submission_v4_static.csv" set "V4OUT=%%~fD\v4\out"
  if exist "%%~fD\v2\out\submission_v2_static.csv" set "V2OUT=%%~fD\v2\out"
)
if exist "%~dp0..\v4\out\submission_v4_static.csv" set "V4OUT=%~dp0..\v4\out"
if exist "%~dp0..\v2\out\submission_v2_static.csv" set "V2OUT=%~dp0..\v2\out"
if not defined V4OUT (
  echo cannot find v4\out\submission_v4_static.csv next to v6 or in the abandoned-files folder
  goto :end
)
if "%~1"=="" goto :default
python "%~dp0compare_static.py" "%V4OUT%\submission_v4_static.csv" %*
goto :end
:default
python "%~dp0compare_static.py" "%V4OUT%\submission_v4_static.csv" "%V2OUT%\submission_v2_static.csv" "%V2OUT%\submission_v2_plain_static.csv"
:end
echo.
pause

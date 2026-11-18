@echo off
rem v10 Alpha158 screen, family M: the three MLP stock members (mlp_res_xr9, mlp_res_BN9, mlp_res_xrN9) with one Alpha158
rem group added at a time, walk-forward 2022..2026-10, seed 1-5 (research\PREREG_v10.md). 2250 small trainings.
rem Can be stopped (close the window) and started again: finished units are kept and skipped.
rem Progress: results\a158_screen_mlp.log.  Optional: screen_mlp.bat --workers 6   (default = half the logical CPUs, at most 4)
cd /d "%~dp0research"
python -c "import torch, lightgbm, pandas, numpy, pyarrow; print('torch', torch.__version__, '| lightgbm', lightgbm.__version__, '| pandas', pandas.__version__, '| numpy', numpy.__version__)" || goto :fail
python a158_screen.py --family mlp %*
echo.
echo Finished or stopped. Log: v10\results\a158_screen_mlp.log  (run this file again to continue if it stopped)
pause
exit /b 0
:fail
echo.
echo A Python package is missing (see above). v9\daily.bat installs lightgbm and pyarrow; torch is the one v9 uses.
pause
exit /b 1

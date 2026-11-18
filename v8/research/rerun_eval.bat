@echo off
rem v8: recompute every result table from the caches (no training). Run from anywhere.
cd /d "%~dp0"
if not exist ..\results mkdir ..\results
for %%s in (d1_market_diag d2_hidden_beta d3_what_is_pbar d4_pbar_sources m1_market_model m2_eval e1_explore e2_gap_upper_bound e3_october_in_context t5_eval h6_hedge h6_check e4_full_neutral c2_cnn_eval r2_rnn_eval g2_tab_eval k2_seq_eval) do (
  echo running %%s ...
  python %%s.py > ..\results\%%s.txt 2>&1
  if errorlevel 1 echo   FAILED: see ..\results\%%s.txt
)
echo done. results in ..\results\
pause

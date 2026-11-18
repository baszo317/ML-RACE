@echo off
rem v2 overnight run: ex-dividend table, production models (about 1 h), walk-forward experiments (up to 6.5 h)
cd /d "%~dp0"
python fetch_exdiv.py
python train_and_predict_v2.py --train-only --retrain all --tf-resid
python overnight_experiments.py --hours 6.5
echo.
echo Done. Results: results_v2\summary_v2.txt
pause

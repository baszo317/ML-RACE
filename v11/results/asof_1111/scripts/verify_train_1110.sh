#!/bin/bash
# Training reproducibility: remove the laptop's 11/09 training point from a copy of the laptop tree, retrain it in this
# environment (same seeds 26110901/02, same data = the laptop's data as of the 11/10 morning) and predict 11/10 again.
W=/home/claude/w1111; T=$W/ver_train; RA=$W/scripts/run_asof.py; L=$W/logs
set -o pipefail
(cd $T/v9 && python3 $RA 2026-11-10 v9 $T --dates auto) > $L/vt1110_v9.log 2>&1 || { echo "v9 failed"; exit 1; }
(cd $T/v11 && python3 $RA 2026-11-10 v11 $T --dates auto) > $L/vt1110_v11.log 2>&1 || { echo "v11 failed"; exit 1; }
(cd $T/v11 && python3 $RA 2026-11-10 v11 $T --dates auto --retrain none --w-t 0 --lam 0.5) > $L/vt1110_w0.log 2>&1 || { echo "w0 failed"; exit 1; }
echo ALL DONE

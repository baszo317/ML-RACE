#!/bin/bash
# As-of runs of the user's own daily programs on a copied tree (nothing on the laptop is touched):
#   11/04 morning: data = rows up to 11/03 + the 11/04 placeholder (US close of 11/03 night)  [already merged]
#   11/05 morning: data = rows up to 11/04 + the 11/05 placeholder (US close of 11/04 night)
# Each morning = daily_v11.bat without the two network fetches (ex-dividend table stays the 11/03 one, 0 Top100 events;
# TX night session missing -> only affects the market view, which W0 does not use), then custom_v11 with w_T 0, lambda 0.5.
W=/tmp/claude-0/-home-claude/6a2b61c3-80a1-5e4b-95c2-bb732ce3f64e/scratchpad
T=$W/asof; RA=$W/base/v11/results/rerun_1030/scripts/run_asof.py; L=$W/runs/nov
mkdir -p $L
set -o pipefail
for D in 2026-11-04 2026-11-05; do
  t=${D:5:2}${D:8:2}
  if [ "$D" = 2026-11-05 ]; then
    (cd $T/transformer && python3 update_data.py $W/zips/data_asof_1105.zip) > $L/update_$t.log 2>&1 || { echo "update $D failed"; exit 1; }
  fi
  (cd $T/v9 && python3 $RA $D v9 $T --dates auto) > $L/v9_$t.log 2>&1 || { echo "v9 $D failed"; exit 1; }
  (cd $T/v11 && python3 $RA $D v11 $T --dates auto) > $L/v11_$t.log 2>&1 || { echo "v11 $D failed"; exit 1; }
  (cd $T/v11 && python3 $RA $D v11 $T --dates auto --retrain none --w-t 0 --lam 0.5) > $L/w0_$t.log 2>&1 || { echo "w0 $D failed"; exit 1; }
  echo "$D done $(date +%T)"
done
echo ALL DONE

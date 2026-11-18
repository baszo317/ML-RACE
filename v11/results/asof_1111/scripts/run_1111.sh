#!/bin/bash
# As-of run of the user's own 11/11 morning routine on a copied tree (nothing on the laptop is touched):
#   data = rows up to 11/10 + the 11/11 placeholder of the 07:00 zip (U.S. close of 11/10 night), built by make_zip_asof.py
#          from data (13).zip (packed 11/11 18:18; rows before 11/11 equal to data (12).zip byte for byte)
#   daily_v11.bat = update_data.py -> fetch_exdiv.py (not rerun: the 11/10 table, 1 Top100 announcement, 2344 on 11/18)
#                   -> fetch_night_session.py (rows the 11/11 morning fetch adds: 11/10 day session + 11/10 15:00-11/11 05:00
#                   night session, appended to v7\data\tx_daily.csv) -> v9 -> v11 ; then custom_v11 with w_T 0, lambda 0.5
W=/home/claude/w1111; T=$W/asof; RA=$W/scripts/run_asof.py; L=$W/logs
set -o pipefail
D=2026-11-11
(cd $T/transformer && python3 update_data.py $W/zips/data_asof_1111.zip) > $L/r1111_update.log 2>&1 || { echo "update failed"; exit 1; }
echo "update done $(date +%T)"
(cd $T/v9 && python3 $RA $D v9 $T --dates auto) > $L/r1111_v9.log 2>&1 || { echo "v9 failed"; exit 1; }
echo "v9 done $(date +%T)"
(cd $T/v11 && python3 $RA $D v11 $T --dates auto) > $L/r1111_v11.log 2>&1 || { echo "v11 failed"; exit 1; }
echo "v11 done $(date +%T)"
(cd $T/v11 && python3 $RA $D v11 $T --dates auto --retrain none --w-t 0 --lam 0.5) > $L/r1111_w0.log 2>&1 || { echo "w0 failed"; exit 1; }
echo "w0 done $(date +%T)"
echo ALL DONE

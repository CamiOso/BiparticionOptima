#!/bin/bash
# Circuit 15B — filas 6-55, 3 en paralelo
PROJECT="/home/cami/Desktop/AnalisisDiseñoAlgoritmos/Proyecto/ProyectoAnalisis2026"
cd "$PROJECT"

FILAS=(6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
       26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44
       45 46 47 48 49 50 51 52 53 54 55)

N=${#FILAS[@]}
i=0
while [ $i -lt $N ]; do
    BATCH=()
    for b in 0 1 2; do
        idx=$((i + b))
        [ $idx -lt $N ] && BATCH+=("${FILAS[$idx]}")
    done
    echo "=== [CIRCUIT-15B] $(date) — filas ${BATCH[*]} ==="
    PIDS=""
    for F in "${BATCH[@]}"; do
        python3 -u scripts/run_circuito_single.py $F --sistema 15B > /tmp/circ15B_${F}.log 2>&1 &
        PIDS="$PIDS $!"
    done
    wait $PIDS
    echo "=== [CIRCUIT-15B] batch ${BATCH[*]} listo ==="
    i=$((i + 3))
done
echo "[CIRCUIT-15B] TODAS LAS FILAS COMPLETADAS — $(date)"

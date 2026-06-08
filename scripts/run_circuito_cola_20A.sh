#!/bin/bash
# Circuit 20A — filas 6-55, 2 en paralelo
PROJECT="/home/cami/Desktop/AnalisisDiseñoAlgoritmos/Proyecto/ProyectoAnalisis2026"
cd "$PROJECT"

FILAS=(6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
       26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44
       45 46 47 48 49 50 51 52 53 54 55)

N=${#FILAS[@]}
i=0
while [ $i -lt $N ]; do
    F1="${FILAS[$i]}"
    F2="${FILAS[$((i+1))]-}"
    echo "=== [CIRCUIT-20A] $(date) — fila $F1${F2:+ y $F2} ==="
    python3 -u scripts/run_circuito_single.py $F1 --sistema 20A > /tmp/circ20A_${F1}.log 2>&1 &
    PID1=$!
    if [ -n "$F2" ]; then
        python3 -u scripts/run_circuito_single.py $F2 --sistema 20A > /tmp/circ20A_${F2}.log 2>&1 &
        wait $PID1 $!
    else
        wait $PID1
    fi
    echo "=== [CIRCUIT-20A] batch listo ==="
    i=$((i + 2))
done
echo "[CIRCUIT-20A] TODAS LAS FILAS COMPLETADAS — $(date)"

#!/bin/bash
# Geometric 22A — de a 2 en paralelo, menor mec primero
PROJECT="/home/cami/Desktop/AnalisisDiseñoAlgoritmos/Proyecto/ProyectoAnalisis2026"
cd "$PROJECT"

COLA=(
    # 46,54 completadas; 47,53 interrumpidas sin guardar nada → reinician desde k=2
    "47:2"  "53:2"
    "39:2"  "40:2"
    "45:2"  "52:2"
    "38:2"  "55:2"
    "11:2"  "12:2"
    "18:2"  "19:2"
    "25:2"  "26:2"
    "32:2"  "33:2"
    "17:2"  "10:2"
    "24:2"  "31:2"
    "44:2"  "51:2"
    "16:2"  "23:2"
    "30:2"  "37:2"
    "9:2"   "43:2"
    "50:2"  "42:2"
    "49:2"  "36:2"
    "29:2"  "35:2"
    "22:2"  "15:2"
    "28:2"  "21:2"
    "14:2"  "41:2"
    "48:2"  "34:2"
    "27:2"  "20:2"
    "13:2"  "8:2"
    "7:2"   "6:2"
)

run_par() {
    local ITEM1=$1 ITEM2=$2
    local FILA1=${ITEM1%%:*} SK1=${ITEM1##*:}
    local FILA2=${ITEM2%%:*} SK2=${ITEM2##*:}
    echo "========================================"
    echo "[GEO-22A] Lanzando fila $FILA1 ${FILA2:+y fila $FILA2} — $(date)"
    echo "========================================"
    python3 -u run_geo_single_22A.py $FILA1 --start-k $SK1 > /tmp/geo22_${FILA1}.log 2>&1 &
    local PID1=$!
    if [ -n "$FILA2" ]; then
        python3 -u run_geo_single_22A.py $FILA2 --start-k $SK2 > /tmp/geo22_${FILA2}.log 2>&1 &
        wait $PID1 $!
    else
        wait $PID1
    fi
    echo "[GEO-22A] Par terminado — $(date)"
}

i=0
while [ $i -lt ${#COLA[@]} ]; do
    run_par "${COLA[$i]}" "${COLA[$((i+1))]}"
    i=$((i+2))
done

echo "[GEO-22A] TODAS LAS FILAS COMPLETADAS — $(date)"

echo "[GEO-22A] Lanzando cola 25A..."
bash run_geo_cola_25A.sh > /tmp/geo25_cola_master.log 2>&1

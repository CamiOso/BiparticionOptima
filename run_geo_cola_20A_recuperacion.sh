#!/bin/bash
# Recuperacion Geometric 20A — filas sin datos ni parciales, menor n_max primero
PROJECT="/home/cami/Desktop/AnalisisDiseñoAlgoritmos/Proyecto/ProyectoAnalisis2026"
cd "$PROJECT"

COLA=(
    # n_max=18 (mas rapidas)
    "51:5"  "44:2"  "37:2"
    "30:2"
    # n_max=19
    "24:2"  "35:2"  "36:2"
    "42:2"  "43:2"  "49:2"
    "50:2"  "16:2"  "23:2"
    "14:2"  "15:2"  "21:2"
    "22:2"  "28:2"  "29:2"
    # n_max=20 (mas lentas)
    "9:2"   "34:2"  "41:2"
    "48:2"  "7:2"   "8:2"
    "13:2"  "20:2"  "27:2"
    "6:2"
)

run_trio() {
    local ITEM1=$1 ITEM2=${2:-} ITEM3=${3:-}
    local FILA1=${ITEM1%%:*} SK1=${ITEM1##*:}
    echo "========================================"
    echo "[GEO-20A-REC] Lanzando fila $FILA1${ITEM2:+ y ${ITEM2%%:*}}${ITEM3:+ y ${ITEM3%%:*}} — $(date)"
    echo "========================================"
    python3 -u run_geo_single.py $FILA1 --start-k $SK1 > /tmp/geo20rec_${FILA1}.log 2>&1 &
    local PIDS="$!"
    if [ -n "$ITEM2" ]; then
        local FILA2=${ITEM2%%:*} SK2=${ITEM2##*:}
        python3 -u run_geo_single.py $FILA2 --start-k $SK2 > /tmp/geo20rec_${FILA2}.log 2>&1 &
        PIDS="$PIDS $!"
    fi
    if [ -n "$ITEM3" ]; then
        local FILA3=${ITEM3%%:*} SK3=${ITEM3##*:}
        python3 -u run_geo_single.py $FILA3 --start-k $SK3 > /tmp/geo20rec_${FILA3}.log 2>&1 &
        PIDS="$PIDS $!"
    fi
    wait $PIDS
    echo "[GEO-20A-REC] Trio terminado — $(date)"
}

i=0
while [ $i -lt ${#COLA[@]} ]; do
    run_trio "${COLA[$i]}" "${COLA[$((i+1))]-}" "${COLA[$((i+2))]-}"
    i=$((i+3))
done

echo "[GEO-20A-REC] RECUPERACION COMPLETA — $(date)"

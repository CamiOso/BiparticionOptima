#!/bin/bash
# Corre las filas QNodos pendientes de 20A de a 2 en paralelo
PROJECT="/home/cami/Desktop/AnalisisDiseñoAlgoritmos/Proyecto/ProyectoAnalisis2026"
cd "$PROJECT"

# Orden: mec menor primero, mec=20 al final
# Formato: "fila:start_k"
COLA=(
    "42:2"  "49:2"
    "43:2"  "50:2"
    "35:2"  "36:2"
    "28:2"  "29:2"
    "14:2"  "15:2"
    "21:2"  "22:2"
    "41:2"  "48:2"
    "34:2"  "27:2"
    "13:2"  "20:2"
    "7:2"   "8:2"
    "6:2"
)

run_par() {
    local ITEM1=$1 ITEM2=$2
    local FILA1=${ITEM1%%:*} SK1=${ITEM1##*:}
    local FILA2=${ITEM2%%:*} SK2=${ITEM2##*:}
    echo "========================================"
    echo "[QNODOS-COLA] Lanzando fila $FILA1 (start_k=$SK1) ${FILA2:+y fila $FILA2} — $(date)"
    echo "========================================"
    python3 -u run_qnodos_single.py $FILA1 --start-k $SK1 > /tmp/qnodos_${FILA1}cola.log 2>&1 &
    local PID1=$!
    if [ -n "$FILA2" ]; then
        python3 -u run_qnodos_single.py $FILA2 --start-k $SK2 > /tmp/qnodos_${FILA2}cola.log 2>&1 &
        wait $PID1 $!
    else
        wait $PID1
    fi
    echo "[QNODOS-COLA] Par terminado — $(date)"
}

i=0
while [ $i -lt ${#COLA[@]} ]; do
    run_par "${COLA[$i]}" "${COLA[$((i+1))]}"
    i=$((i+2))
done

echo "[QNODOS-COLA] TODAS LAS FILAS COMPLETADAS — $(date)"

echo "[QNODOS-COLA] Lanzando cola 22A..."
bash run_qnodos_cola_22A.sh > /tmp/qnodos22_cola_master.log 2>&1

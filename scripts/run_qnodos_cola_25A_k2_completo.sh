#!/bin/bash
# QNodos 25A — filas mec<=17 con k=2 solo
# Orden: primero alc>mec (poblan cache), luego cache hits, luego alc<mec
# Tiempo estimado: ~2h

PROJECT="/home/cami/Desktop/AnalisisDiseñoAlgoritmos/Proyecto/ProyectoAnalisis2026"
cd "$PROJECT"

run_trio() {
    local ITEM1=$1 ITEM2=${2:-} ITEM3=${3:-}
    local F1=${ITEM1%%:*}
    local R1=${ITEM1#*:}
    local SK1=${R1%%:*}
    local EK1=${R1##*:}
    echo "========================================"
    echo "[Q-K2] Lanzando fila $F1${ITEM2:+ y ${ITEM2%%:*}}${ITEM3:+ y ${ITEM3%%:*}} — $(date)"
    echo "========================================"
    python3 -u run_qnodos_single_25A.py $F1 --start-k $SK1 --end-k $EK1 > /tmp/qnodos25_${F1}.log 2>&1 &
    local PIDS="$!"
    if [ -n "$ITEM2" ]; then
        local F2=${ITEM2%%:*}
        local R2=${ITEM2#*:}
        local SK2=${R2%%:*}
        local EK2=${R2##*:}
        python3 -u run_qnodos_single_25A.py $F2 --start-k $SK2 --end-k $EK2 > /tmp/qnodos25_${F2}.log 2>&1 &
        PIDS="$PIDS $!"
    fi
    if [ -n "$ITEM3" ]; then
        local F3=${ITEM3%%:*}
        local R3=${ITEM3#*:}
        local SK3=${R3%%:*}
        local EK3=${R3##*:}
        python3 -u run_qnodos_single_25A.py $F3 --start-k $SK3 --end-k $EK3 > /tmp/qnodos25_${F3}.log 2>&1 &
        PIDS="$PIDS $!"
    fi
    wait $PIDS
    echo "[Q-K2] Trío terminado — $(date)"
}

# Trío 1: poblan cache (mec=17, 13, 12)
run_trio "10:2:2" "11:2:2" "12:2:2"

# Cache hits (~0s): mec=17, 13, 12
run_trio "17:2:2" "18:2:2" "19:2:2"
run_trio "24:2:2" "25:2:2" "26:2:2"
run_trio "31:2:2" "32:2:2" "33:2:2"

# alc<mec: sin cache, computan normalmente
run_trio "45:2:2" "52:2:2" "53:2:2"

echo "[Q-K2] COMPLETO — $(date)"

#!/bin/bash
# Geo 25A — filas mec<=17 con k=2 solo
# Sin cache: agrupadas por mec similar
# Tiempo estimado: ~2.5h

PROJECT="/home/cami/Desktop/AnalisisDiseñoAlgoritmos/Proyecto/ProyectoAnalisis2026"
cd "$PROJECT"

run_trio() {
    local ITEM1=$1 ITEM2=${2:-} ITEM3=${3:-}
    local F1=${ITEM1%%:*}
    local R1=${ITEM1#*:}
    local SK1=${R1%%:*}
    local EK1=${R1##*:}
    echo "========================================"
    echo "[GEO-K2] Lanzando fila $F1${ITEM2:+ y ${ITEM2%%:*}}${ITEM3:+ y ${ITEM3%%:*}} — $(date)"
    echo "========================================"
    python3 -u scripts/run_geo_single_25A.py $F1 --start-k $SK1 --end-k $EK1 > /tmp/geo25_${F1}.log 2>&1 &
    local PIDS="$!"
    if [ -n "$ITEM2" ]; then
        local F2=${ITEM2%%:*}
        local R2=${ITEM2#*:}
        local SK2=${R2%%:*}
        local EK2=${R2##*:}
        python3 -u scripts/run_geo_single_25A.py $F2 --start-k $SK2 --end-k $EK2 > /tmp/geo25_${F2}.log 2>&1 &
        PIDS="$PIDS $!"
    fi
    if [ -n "$ITEM3" ]; then
        local F3=${ITEM3%%:*}
        local R3=${ITEM3#*:}
        local SK3=${R3%%:*}
        local EK3=${R3##*:}
        python3 -u scripts/run_geo_single_25A.py $F3 --start-k $SK3 --end-k $EK3 > /tmp/geo25_${F3}.log 2>&1 &
        PIDS="$PIDS $!"
    fi
    wait $PIDS
    echo "[GEO-K2] Trío terminado — $(date)"
}

# mec=12 (~5-10 min por fila)
run_trio "12:2:2" "19:2:2" "26:2:2"
run_trio "33:2:2" "40:2:2" "47:2:2"

# mec=13 (~15-20 min por fila)
run_trio "11:2:2" "18:2:2" "25:2:2"
run_trio "32:2:2" "39:2:2" "53:2:2"

# mec=17 (~45 min por fila)
run_trio "10:2:2" "17:2:2" "24:2:2"
run_trio "31:2:2" "45:2:2" "52:2:2"

echo "[GEO-K2] COMPLETO — $(date)"

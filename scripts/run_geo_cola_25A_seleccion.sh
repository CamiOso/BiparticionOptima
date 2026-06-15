#!/bin/bash
# Geometric 25A — seleccion representativa (n_nodos <= 17)
# 54: alc=mec=BDFHJLNPRTVX    n_nodos=12
# 46: alc=mec=ACEGIKMOQSUWY   n_nodos=13
# 38: alc=mec=ABDEGHJKMNPQSTVWY n_nodos=17
PROJECT="/home/cami/Desktop/AnalisisDiseñoAlgoritmos/Proyecto/ProyectoAnalisis2026"
cd "$PROJECT"

COLA=(
    "54:2"  "46:2"
    "38:2"
)

run_trio() {
    local ITEM1=$1 ITEM2=${2:-} ITEM3=${3:-}
    local FILA1=${ITEM1%%:*} SK1=${ITEM1##*:}
    echo "========================================"
    echo "[GEO-25A-SEL] Lanzando fila $FILA1${ITEM2:+ y ${ITEM2%%:*}}${ITEM3:+ y ${ITEM3%%:*}} — $(date)"
    echo "========================================"
    python3 -u run_geo_single_25A.py $FILA1 --start-k $SK1 > /tmp/geo25_${FILA1}.log 2>&1 &
    local PIDS="$!"
    if [ -n "$ITEM2" ]; then
        local FILA2=${ITEM2%%:*} SK2=${ITEM2##*:}
        python3 -u run_geo_single_25A.py $FILA2 --start-k $SK2 > /tmp/geo25_${FILA2}.log 2>&1 &
        PIDS="$PIDS $!"
    fi
    if [ -n "$ITEM3" ]; then
        local FILA3=${ITEM3%%:*} SK3=${ITEM3##*:}
        python3 -u run_geo_single_25A.py $FILA3 --start-k $SK3 > /tmp/geo25_${FILA3}.log 2>&1 &
        PIDS="$PIDS $!"
    fi
    wait $PIDS
    echo "[GEO-25A-SEL] Trio terminado — $(date)"
}

i=0
while [ $i -lt ${#COLA[@]} ]; do
    run_trio "${COLA[$i]}" "${COLA[$((i+1))]-}" "${COLA[$((i+2))]-}"
    i=$((i+3))
done

echo "[GEO-25A-SEL] SELECCION COMPLETA — $(date)"

#!/bin/bash
# QNodos 25A — seleccion representativa (n_max <= 21)
# 54: alc=mec=BDFHJLNPRTVX           n_max=12 simetrico
# 46: alc=mec=ACEGIKMOQSUWY          n_max=13 simetrico
# 47: alc=ACEGIKMOQSUWY mec=BDFHJLNPRTVX  n_max=13 asimetrico
# 38: alc=mec=ABDEGHJKMNPQSTVWY      n_max=17 simetrico
# 40: alc=ABDEGHJKMNPQSTVWY mec=BDFHJLNPRTVX n_max=17 asimetrico
# 39: alc=ABDEGHJKMNPQSTVWY mec=ACEGIKMOQSUWY n_max=17 asimetrico
# 55: alc=mec=ACDEFGHIJKLMNOPQRSTVX  n_max=21 simetrico — dos pasadas
PROJECT="/home/cami/Desktop/AnalisisDiseñoAlgoritmos/Proyecto/ProyectoAnalisis2026"
cd "$PROJECT"

# Formato: "fila:start_k:end_k"  (end_k opcional, default 5)
COLA=(
    "54:2:5"  "46:2:5"  "47:2:5"
    "38:2:5"  "40:2:5"  "39:2:5"
)

run_trio() {
    local ITEM1=$1 ITEM2=${2:-} ITEM3=${3:-}
    local FILA1=${ITEM1%%:*} REST1=${ITEM1#*:} SK1=${REST1%%:*} EK1=${REST1##*:}
    echo "========================================"
    echo "[QNODOS-25A-SEL] Lanzando fila $FILA1${ITEM2:+ y ${ITEM2%%:*}}${ITEM3:+ y ${ITEM3%%:*}} — $(date)"
    echo "========================================"
    python3 -u run_qnodos_single_25A.py $FILA1 --start-k $SK1 --end-k $EK1 > /tmp/qnodos25_${FILA1}.log 2>&1 &
    local PIDS="$!"
    if [ -n "$ITEM2" ]; then
        local FILA2=${ITEM2%%:*} REST2=${ITEM2#*:} SK2=${REST2%%:*} EK2=${REST2##*:}
        python3 -u run_qnodos_single_25A.py $FILA2 --start-k $SK2 --end-k $EK2 > /tmp/qnodos25_${FILA2}.log 2>&1 &
        PIDS="$PIDS $!"
    fi
    if [ -n "$ITEM3" ]; then
        local FILA3=${ITEM3%%:*} REST3=${ITEM3#*:} SK3=${REST3%%:*} EK3=${REST3##*:}
        python3 -u run_qnodos_single_25A.py $FILA3 --start-k $SK3 --end-k $EK3 > /tmp/qnodos25_${FILA3}.log 2>&1 &
        PIDS="$PIDS $!"
    fi
    wait $PIDS
    echo "[QNODOS-25A-SEL] Trio terminado — $(date)"
}

i=0
while [ $i -lt ${#COLA[@]} ]; do
    run_trio "${COLA[$i]}" "${COLA[$((i+1))]-}" "${COLA[$((i+2))]-}"
    i=$((i+3))
done

# Fila 55 (mec=21): primera pasada solo k=2 → minimo phi disponible rapido
echo "========================================"
echo "[QNODOS-25A-SEL] Fila 55 — pasada 1: solo k=2 — $(date)"
echo "========================================"
python3 -u run_qnodos_single_25A.py 55 --start-k 2 --end-k 2 >> /tmp/qnodos25_55.log 2>&1
echo "[QNODOS-25A-SEL] Fila 55 k=2 COMPLETO — $(date)"

# Fila 55: segunda pasada k=3,4,5 (puede cancelarse si no hay tiempo)
echo "========================================"
echo "[QNODOS-25A-SEL] Fila 55 — pasada 2: k=3,4,5 — $(date)"
echo "========================================"
python3 -u run_qnodos_single_25A.py 55 --start-k 3 --end-k 5 >> /tmp/qnodos25_55.log 2>&1
echo "[QNODOS-25A-SEL] Fila 55 COMPLETO — $(date)"

echo "[QNODOS-25A-SEL] SELECCION COMPLETA — $(date)"

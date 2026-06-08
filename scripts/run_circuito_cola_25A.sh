#!/bin/bash
# Circuit 25A — filas 6-55, secuencial (25A usa memmap, mejor no paralelizar)
PROJECT="/home/cami/Desktop/AnalisisDiseñoAlgoritmos/Proyecto/ProyectoAnalisis2026"
cd "$PROJECT"

FILAS=(6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
       26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44
       45 46 47 48 49 50 51 52 53 54 55)

for F in "${FILAS[@]}"; do
    echo "=== [CIRCUIT-25A] $(date) — fila $F ==="
    python3 -u scripts/run_circuito_single.py $F --sistema 25A 2>&1 | tee /tmp/circ25A_${F}.log
    echo "=== [CIRCUIT-25A] fila $F listo ==="
done
echo "[CIRCUIT-25A] TODAS LAS FILAS COMPLETADAS — $(date)"

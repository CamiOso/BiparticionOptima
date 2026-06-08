#!/usr/bin/env bash
# Cola BranchBound para filas INTERESANTES de 20A (donde Q ≠ Geo).
# BB sirve como árbitro: confirma o mejora el mejor resultado de QNodos.
set -e
PROJECT="/home/cami/Desktop/AnalisisDiseñoAlgoritmos/Proyecto/ProyectoAnalisis2026"
source "$PROJECT/.venv/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Filas donde Q ≠ Geo k=2 (árbitro crítico)
for fila in 10 17 31 55; do
    echo "=== [Q≠G] 20A fila $fila ==="
    python3 "$PROJECT/run_bb_single.py" 20A "$fila" || echo "  [WARN] fila $fila falló"
    sleep 0.3
done

# Filas representativas adicionales
for fila in 6 7 8 9 11 12 13 14; do
    echo "=== [extra] 20A fila $fila ==="
    python3 "$PROJECT/run_bb_single.py" 20A "$fila" || echo "  [WARN] fila $fila falló"
    sleep 0.3
done

echo "=== Cola 20A interesantes COMPLETA ==="

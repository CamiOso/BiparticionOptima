#!/bin/bash
# Cola completa Geo 22A + 25A — encadenamiento automático
# Espera a que terminen las filas 10 y 17 antes de continuar

PROJECT="/home/cami/Desktop/AnalisisDiseñoAlgoritmos/Proyecto/ProyectoAnalisis2026"
cd "$PROJECT" || exit 1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

LOG_MASTER="/tmp/geo_cola_completa_master.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_MASTER"
}

run22() {
    local fila=$1 sk=${2:-2}
    log "Iniciando 22A fila=$fila start_k=$sk"
    python3 -u scripts/run_geo_single_22A.py "$fila" --start-k "$sk" >> "/tmp/geo22_${fila}.log" 2>&1
    log "COMPLETO 22A fila=$fila"
}

run25() {
    local fila=$1 sk=${2:-2}
    log "Iniciando 25A fila=$fila start_k=$sk"
    python3 -u scripts/run_geo_single_25A.py "$fila" --start-k "$sk" >> "/tmp/geo25_${fila}.log" 2>&1
    log "COMPLETO 25A fila=$fila"
}

# ── Esperar filas activas 10 y 17 ──────────────────────────────────────────
log "Esperando filas 22A 10 y 17 (ya corriendo)..."
wait $(pgrep -f "run_geo_single_22A.py 10") 2>/dev/null
wait $(pgrep -f "run_geo_single_22A.py 17") 2>/dev/null
log "Filas 10 y 17 terminadas."

# ── 22A mec=15 ─────────────────────────────────────────────────────────────
log "=== 22A mec=15: filas 24 y 31 ==="
run22 24 & run22 31 & wait
log "=== 22A mec=15 completo ==="

# ── 22A mec=19 (fila 55 — k=2 ya existe) ──────────────────────────────────
log "=== 22A mec=19: fila 55 start_k=3 ==="
run22 55 3
log "=== 22A mec=19 completo ==="

# ── 22A mec=20 ─────────────────────────────────────────────────────────────
log "=== 22A mec=20: filas 9 y 16 ==="
run22 9 & run22 16 & wait

log "=== 22A mec=20: filas 23 y 30 ==="
run22 23 & run22 30 & wait

log "=== 22A mec=20: filas 37 y 44 ==="
run22 37 & run22 44 & wait

log "=== 22A mec=20: fila 51 ==="
run22 51
log "=== 22A mec=20 completo ==="

# ── 25A mec=12 ─────────────────────────────────────────────────────────────
log "=== 25A mec=12: filas 12 y 19 ==="
run25 12 & run25 19 & wait

log "=== 25A mec=12: filas 47 y 40 (40 start_k=4) ==="
run25 47 & run25 40 4 & wait
log "=== 25A mec=12 completo ==="

# ── 25A mec=13 ─────────────────────────────────────────────────────────────
log "=== 25A mec=13: filas 11 y 39 (39 start_k=3) ==="
run25 11 & run25 39 3 & wait
log "=== 25A mec=13 completo ==="

# ── 25A mec=17 ─────────────────────────────────────────────────────────────
log "=== 25A mec=17: filas 10 y 38 (38 start_k=4) ==="
run25 10 & run25 38 4 & wait
log "=== 25A mec=17 completo ==="

log "=============================="
log "COLA COMPLETA"
log "=============================="

# ── Sección ampliada: filas adicionales ────────────────────────────────────

runq25() {
    local fila=$1 sk=${2:-2}
    log "Iniciando 25A QNodos fila=$fila start_k=$sk"
    python3 -u scripts/run_qnodos_single_25A.py "$fila" --start-k "$sk" >> "/tmp/qnodos25_${fila}.log" 2>&1
    log "COMPLETO 25A QNodos fila=$fila"
}

# ── 22A mec=21 Geo ─────────────────────────────────────────────────────────
log "=== 22A mec=21: filas 42 y 43 ==="
run22 42 & run22 43 & wait

log "=== 22A mec=21: filas 49 y 50 ==="
run22 49 & run22 50 & wait
log "=== 22A mec=21 completo ==="

# ── 25A mec=17: fila 10 QNodos sk=3 ───────────────────────────────────────
log "=== 25A mec=17: fila 10 QNodos sk=3 ==="
runq25 10 3
log "=== 25A fila 10 QNodos completo ==="

# ── 25A mec=12 filas nuevas ────────────────────────────────────────────────
log "=== 25A mec=12: fila 26 (Q+G) ==="
runq25 26 & run25 26 & wait

log "=== 25A mec=12: fila 33 (Q+G) ==="
runq25 33 & run25 33 & wait
log "=== 25A mec=12 filas nuevas completo ==="

# ── 25A mec=13 filas nuevas ────────────────────────────────────────────────
log "=== 25A mec=13: fila 18 (Q+G) ==="
runq25 18 & run25 18 & wait

log "=== 25A mec=13: fila 25 (Q+G) ==="
runq25 25 & run25 25 & wait

log "=== 25A mec=13: fila 32 (Q+G) ==="
runq25 32 & run25 32 & wait

log "=== 25A mec=13: fila 53 (Q+G) ==="
runq25 53 & run25 53 & wait
log "=== 25A mec=13 filas nuevas completo ==="

# ── 25A mec=17 filas nuevas ────────────────────────────────────────────────
log "=== 25A mec=17: fila 17 (Q+G) ==="
runq25 17 & run25 17 & wait

log "=== 25A mec=17: fila 24 (Q+G) ==="
runq25 24 & run25 24 & wait

log "=== 25A mec=17: fila 31 (Q+G) ==="
runq25 31 & run25 31 & wait

log "=== 25A mec=17: fila 45 (Q+G) ==="
runq25 45 & run25 45 & wait

log "=== 25A mec=17: fila 52 (Q+G) ==="
runq25 52 & run25 52 & wait
log "=== 25A mec=17 filas nuevas completo ==="

log "=============================="
log "COLA AMPLIADA COMPLETA"
log "=============================="

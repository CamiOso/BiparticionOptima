#!/bin/bash
# Watchdog de carga — monitorea load y mata el proceso mas reciente
# si el sistema se degrada sostenidamente con 3+ procesos corriendo.
#
# Uso: bash scripts/watchdog_carga.sh &
# Log: /tmp/watchdog.log

LOG="/tmp/watchdog.log"
UMBRAL_ALTO=4.3        # load 1-min para considerar degradacion
CHECKS_PARA_MATAR=3    # checks consecutivos sobre umbral antes de matar
INTERVALO=300          # segundos entre checks (5 min)

checks_altos=0

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"
}

matar_proceso_extra() {
    # Mata el proceso run_*_single mas reciente (mayor PID = mas nuevo = menor progreso)
    # Nunca mata procesos que llevan mas de 1 hora corriendo (tienen progreso valioso)
    local ahora=$(date +%s)
    local pid_matar=""
    local pid_mayor=0

    while IFS= read -r linea; do
        local pid=$(echo "$linea" | awk '{print $2}')
        local start=$(echo "$linea" | awk '{print $9}')
        # Calcular tiempo corriendo aproximado via /proc
        local ticks=$(cat /proc/$pid/stat 2>/dev/null | awk '{print $22}')
        local hz=100
        local uptime_s=$(awk '{print int($1)}' /proc/uptime)
        if [ -n "$ticks" ]; then
            local start_s=$(( uptime_s - ticks / hz ))
            local corriendo=$(( ahora - ($(date +%s) - uptime_s + start_s) ))
            # Solo candidato si lleva menos de 3600s (1h)
            if [ "$corriendo" -lt 3600 ] && [ "$pid" -gt "$pid_mayor" ]; then
                pid_mayor=$pid
                pid_matar=$pid
            fi
        fi
    done < <(ps aux | grep -E "run_(qnodos|geo)_single" | grep -v grep)

    if [ -n "$pid_matar" ]; then
        local cmd=$(ps -p $pid_matar -o cmd= 2>/dev/null)
        log "MATANDO PID=$pid_matar ($cmd) — carga sostenida sobre $UMBRAL_ALTO"
        kill $pid_matar
        checks_altos=0
    else
        log "Todos los procesos llevan >1h — no se mata ningun proceso con progreso"
        checks_altos=0
    fi
}

log "=== Watchdog iniciado. Umbral=$UMBRAL_ALTO checks=$CHECKS_PARA_MATAR intervalo=${INTERVALO}s ==="

while true; do
    n_proc=$(ps aux | grep -E "run_(qnodos|geo)_single" | grep -v grep | wc -l)
    load=$(awk '{print $1}' /proc/loadavg)
    es_alto=$(awk "BEGIN {print ($load > $UMBRAL_ALTO) ? 1 : 0}")

    log "procesos=$n_proc load1m=$load"

    if [ "$n_proc" -ge 3 ] && [ "$es_alto" -eq 1 ]; then
        checks_altos=$((checks_altos + 1))
        log "Load alto ($load > $UMBRAL_ALTO) — check $checks_altos/$CHECKS_PARA_MATAR"
        if [ "$checks_altos" -ge "$CHECKS_PARA_MATAR" ]; then
            matar_proceso_extra
        fi
    else
        if [ "$checks_altos" -gt 0 ]; then
            log "Load normalizado ($load) — reset contador"
        fi
        checks_altos=0
    fi

    sleep $INTERVALO
done

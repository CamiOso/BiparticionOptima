#!/bin/bash
# Corre Circuit para 20A, 22A y 25A en secuencia.
# Cada sistema usa su propio script de cola.
# Log global: /tmp/circ_todos.log
PROJECT="/home/cami/Desktop/AnalisisDiseñoAlgoritmos/Proyecto/ProyectoAnalisis2026"
cd "$PROJECT"

LOG=/tmp/circ_todos.log
echo "[TODOS] Inicio — $(date)" | tee -a "$LOG"

echo "[TODOS] === 20A ===" | tee -a "$LOG"
bash scripts/run_circuito_cola_20A.sh 2>&1 | tee -a "$LOG"
echo "[TODOS] 20A terminado — $(date)" | tee -a "$LOG"

echo "[TODOS] === 22A ===" | tee -a "$LOG"
bash scripts/run_circuito_cola_22A.sh 2>&1 | tee -a "$LOG"
echo "[TODOS] 22A terminado — $(date)" | tee -a "$LOG"

echo "[TODOS] === 25A ===" | tee -a "$LOG"
bash scripts/run_circuito_cola_25A.sh 2>&1 | tee -a "$LOG"
echo "[TODOS] 25A terminado — $(date)" | tee -a "$LOG"

echo "[TODOS] TODOS LOS SISTEMAS COMPLETADOS — $(date)" | tee -a "$LOG"

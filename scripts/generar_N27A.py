"""Genera la TPM N27A.npy para n=27 nodos.

Formato idéntico a N26A: float32, shape=(2^27, 27), valores en (0,1).
Cada celda = P(nodo_i=1 en t+1 | estado del sistema en t).

Generación en bloques de 2^20 filas (~113 MB c/u) para no saturar la RAM.
Semilla fija (42) para reproducibilidad.

Uso:
    python scripts/generar_N27A.py
"""

import time
from pathlib import Path

import numpy as np

N       = 27
ROWS    = 1 << N            # 134_217_728
COLS    = N
DTYPE   = np.float32
SEED    = 42
CHUNK   = 1 << 20           # 1_048_576 filas por bloque ≈ 113 MB
OUT     = Path("src/.samples/N27A.npy")

OUT.parent.mkdir(parents=True, exist_ok=True)

total_gb = ROWS * COLS * np.dtype(DTYPE).itemsize / 1e9
print(f"Generando N27A.npy — shape=({ROWS:,}, {COLS}), {total_gb:.2f} GB")
print(f"Bloques: {ROWS // CHUNK} × {CHUNK:,} filas  (~{CHUNK*COLS*4/1e6:.0f} MB c/u)")
print(f"Destino: {OUT.resolve()}\n")

fp = np.lib.format.open_memmap(OUT, mode="w+", dtype=DTYPE, shape=(ROWS, COLS))

rng = np.random.default_rng(SEED)
t0  = time.perf_counter()

for bloque in range(ROWS // CHUNK):
    inicio = bloque * CHUNK
    fin    = inicio + CHUNK
    fp[inicio:fin] = rng.random((CHUNK, COLS), dtype=DTYPE)
    elapsed = time.perf_counter() - t0
    pct     = (bloque + 1) / (ROWS // CHUNK) * 100
    speed   = (fin * COLS * 4) / elapsed / 1e6
    print(f"  [{bloque+1:3d}/{ROWS//CHUNK}]  {pct:5.1f}%  {elapsed:5.1f}s  {speed:.0f} MB/s",
          flush=True)

del fp  # flush a disco

elapsed = time.perf_counter() - t0
print(f"\nListo en {elapsed:.1f}s  ({total_gb/elapsed*1000:.0f} MB/s promedio)")
print(f"Archivo: {OUT.resolve()}  ({OUT.stat().st_size/1e9:.2f} GB)")

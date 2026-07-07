"""Convierte archivos TPM .npy de Float32 a Float16 en chunks.

Float32 → Float16 reduce el tamaño a la mitad:
  N26A: 6.6 GB → 3.3 GB
  N27A: 14  GB → 7   GB
  N28A: 30  GB → 15  GB

Con N28A_f16.npy (15 GB) cabiendo casi completa en los 15 GB de RAM,
se elimina el I/O-bound en precompute_lut y el speedup de Threads.@threads
pasa de 1.6x a ~3-4x real.

Precisión: Float16 tiene ~3.3 dígitos sig. Para probabilidades en [0,1]
el error relativo es ≤0.1%. La LUT ya se guarda en Float16, así que
este cambio no degrada la calidad de los resultados.

Uso:
    python scripts/convertir_tpm_float16.py           # convierte N28A
    python scripts/convertir_tpm_float16.py N27A N28A  # convierte N27A y N28A
"""

import sys
import time
import os
from pathlib import Path
import numpy as np

SAMPLES_DIR = Path(__file__).parent.parent / "src" / ".samples"
CHUNK_ROWS  = 1 << 20   # 1M filas por chunk (~112 MB de lectura por iteración)


def convertir(nombre: str) -> None:
    src_path = SAMPLES_DIR / f"{nombre}.npy"
    dst_path = SAMPLES_DIR / f"{nombre}_f16.npy"

    if not src_path.exists():
        print(f"[ERROR] No existe: {src_path}")
        return

    if dst_path.exists():
        print(f"[SKIP] Ya existe: {dst_path} ({dst_path.stat().st_size / 1e9:.1f} GB)")
        return

    src = np.load(src_path, mmap_mode="r")
    n_rows, n_cols = src.shape
    src_gb  = src.nbytes / 1e9
    dst_gb  = src_gb / 2

    print(f"\n{'='*60}")
    print(f"  {nombre}.npy  →  {nombre}_f16.npy")
    print(f"  Shape: {src.shape}  |  {src_gb:.1f} GB → {dst_gb:.1f} GB")
    print(f"{'='*60}")

    dst = np.lib.format.open_memmap(
        dst_path, mode="w+", dtype=np.float16, shape=(n_rows, n_cols)
    )

    n_chunks = (n_rows + CHUNK_ROWS - 1) // CHUNK_ROWS
    t0 = time.time()

    for i in range(n_chunks):
        start = i * CHUNK_ROWS
        end   = min(start + CHUNK_ROWS, n_rows)
        dst[start:end] = src[start:end].astype(np.float16)

        elapsed = time.time() - t0
        pct     = (end / n_rows) * 100
        eta     = (elapsed / pct * (100 - pct)) if pct > 0 else 0
        print(f"  {pct:5.1f}%  |  {elapsed/60:4.1f} min transcurridos  |  ETA {eta/60:.1f} min   ", end="\r", flush=True)

    dst.flush()
    elapsed_total = time.time() - t0
    print(f"\n  Listo en {elapsed_total/60:.1f} min  →  {dst_path}  ({os.path.getsize(dst_path)/1e9:.1f} GB)")


def main() -> None:
    nombres = sys.argv[1:] if len(sys.argv) > 1 else ["N28A"]
    for nombre in nombres:
        convertir(nombre)
    print("\nHecho. Reinicia Julia con:")
    print("  julia -t auto --project=. implementacion_julia/main_ibqnodos28.jl")


if __name__ == "__main__":
    main()

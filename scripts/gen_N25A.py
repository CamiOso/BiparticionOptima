"""Genera src/.samples/N25A.npy con seed=44 (mismo que N20A y N22A).
Usa memmap para no necesitar 3.36 GB en RAM de golpe."""
import numpy as np, time, sys, os

OUT = "src/.samples/N25A.npy"
N   = 25
ROWS = 1 << N          # 33_554_432
CHUNK = 1 << 20        # 1_048_576 filas por bloque (~100 MB)

print(f"Generando N25A: {ROWS} filas x {N} cols = {ROWS*N*4/1e9:.2f} GB")
print(f"Destino: {OUT}")

t0 = time.perf_counter()
fp = np.memmap(OUT, dtype="float32", mode="w+", shape=(ROWS, N))
rng = np.random.default_rng(44)

n_chunks = ROWS // CHUNK
for i in range(n_chunks):
    start = i * CHUNK
    fp[start : start + CHUNK] = rng.random((CHUNK, N), dtype=np.float32)
    elapsed = time.perf_counter() - t0
    eta = elapsed / (i + 1) * (n_chunks - i - 1)
    print(f"  chunk {i+1}/{n_chunks}  {elapsed:.0f}s  ETA {eta:.0f}s", flush=True)

del fp  # flush to disk
elapsed = time.perf_counter() - t0
print(f"Listo en {elapsed:.1f}s  ({elapsed/60:.1f} min)")
print(f"Tamaño: {os.path.getsize(OUT)/1e9:.2f} GB")

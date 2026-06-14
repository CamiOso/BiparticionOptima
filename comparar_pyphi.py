"""Compara IBQNodos vs PyPhi (referencia oficial IIT) en redes N4A–N8A.

PyPhi es el software de referencia de Tononi para calcular φ exacto (IIT 3.0).
Solo es práctico hasta ~8 nodos. Los archivos N4A-N8A son redes deterministas
(0/1) compatibles con PyPhi.

Uso:
    pip install pyphi          # instalar primero
    python comparar_pyphi.py
"""

import sys
import time
from pathlib import Path

import numpy as np

try:
    import pyphi
    import pyphi.compute
    HAS_PYPHI = True
    print(f"PyPhi {pyphi.__version__} disponible.")
except ImportError:
    HAS_PYPHI = False
    print("ERROR: PyPhi no instalado. Ejecuta: pip install pyphi")
    sys.exit(1)

PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))
from src.estrategias.ib_qnodos import IBQNodos

# ── Redes a comparar ──────────────────────────────────────────────────
REDES = [
    {"nombre": "N4A", "n": 4,  "tpm": PROJECT / "src/.samples/N4A.csv"},
    {"nombre": "N5A", "n": 5,  "tpm": PROJECT / "src/.samples/N5A.csv"},
    {"nombre": "N6A", "n": 6,  "tpm": PROJECT / "src/.samples/N6A.csv"},
    {"nombre": "N7A", "n": 7,  "tpm": PROJECT / "src/.samples/N7A.csv"},
    {"nombre": "N8A", "n": 8,  "tpm": PROJECT / "src/.samples/N8A.csv"},
]

# Silenciar logs de PyPhi
import logging
logging.getLogger("pyphi").setLevel(logging.WARNING)

resultados = []

for red in REDES:
    n       = red["n"]
    tpm_arr = np.genfromtxt(red["tpm"], delimiter=",").astype(np.float32)
    sistema = "".join(chr(ord("A") + i) for i in range(n))
    estado  = "1" + "0" * (n - 1)
    estado_tuple = tuple(int(b) for b in estado)

    condicion = "1" * n
    alc_bin   = "1" * n
    mec_bin   = "1" * n

    print(f"\n{'='*55}")
    print(f"  {red['nombre']}  n={n}  sistema={sistema}")
    print(f"  estado={estado}")

    # ── PyPhi φ exacto ──────────────────────────────────────────────
    t0  = time.perf_counter()
    net = pyphi.Network(tpm_arr)
    sub = pyphi.Subsystem(net, estado_tuple)
    phi_pyphi = float(pyphi.compute.phi(sub))
    t_pyphi   = time.perf_counter() - t0
    print(f"  PyPhi    φ={phi_pyphi:.6f}  t={t_pyphi:.2f}s")

    # ── IBQNodos ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    estrategia = IBQNodos(tpm_arr)
    res = estrategia.aplicar_estrategia(
        estado_inicial=estado,
        condicion=condicion,
        alcance=alc_bin,
        mecanismo=mec_bin,
        k=2,
    )
    t_ib  = time.perf_counter() - t0
    phi_ib = float(res.perdida)
    print(f"  IBQNodos φ={phi_ib:.6f}  t={t_ib:.2f}s")

    diff  = abs(phi_ib - phi_pyphi)
    match = "✓ exacto" if diff < 1e-5 else (f"↓mejor" if phi_ib < phi_pyphi else f"✗ diff={diff:.6f}")
    speedup = round(t_pyphi / t_ib, 1) if t_ib > 0 else "—"
    print(f"  Resultado: {match}  speedup=×{speedup}")

    resultados.append({
        "red": red["nombre"], "n": n,
        "phi_pyphi": phi_pyphi, "t_pyphi": round(t_pyphi, 3),
        "phi_ib": phi_ib,       "t_ib":    round(t_ib, 3),
        "match": match, "speedup": speedup,
    })

# ── Resumen ──────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("  RESUMEN")
print(f"{'='*55}")
print(f"  {'Red':<6} {'n':>2}  {'PyPhi_φ':>10}  {'IBQ_φ':>10}  {'Resultado':<20}  {'Speedup':>8}")
for r in resultados:
    print(f"  {r['red']:<6} {r['n']:>2}  {r['phi_pyphi']:>10.6f}  {r['phi_ib']:>10.6f}  {r['match']:<20}  ×{r['speedup']:>6}")

exactos = sum(1 for r in resultados if "exacto" in r["match"])
print(f"\n  Match exacto: {exactos}/{len(resultados)}")

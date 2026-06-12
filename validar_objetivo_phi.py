"""Valida que las estrategias modificadas optimizan el objetivo φ correcto.

Compara los resultados antes/después de las correcciones:
  - Louvain e IB: corrección del espacio de búsqueda k=2 (bipartir) + reinicios
  - Hiperbolica y Variacional: reinicios aleatorios con φ

Ejecutar:  python validar_objetivo_phi.py
"""
from __future__ import annotations

import time
import numpy as np

from src.modelos.base.aplicacion import aplicacion
from src.controladores.gestor import Gestor
from src.estrategias.q_nodos import QNodos
from src.estrategias.louvain import Louvain
from src.estrategias.informacion_bottleneck import InformacionBottleneck
from src.estrategias.hiperbolica import ParticionHiperbolica
from src.estrategias.variacional import ParticionVariacional
from src.estrategias.remcmc import REMCMC

# ── Sistema 4A ──────────────────────────────────────────────────────────
aplicacion.set_pagina_red_muestra("A")
tpm_4a = Gestor("1000").cargar_red()

# ── Sistema 10A ─────────────────────────────────────────────────────────
tpm_10a = np.genfromtxt("src/.samples/N10A.csv", delimiter=",").astype(np.float32)

ESTRATEGIAS = [
    ("QNodos",      QNodos),
    ("Louvain",     Louvain),
    ("Info Bottleneck", InformacionBottleneck),
    ("Hiperbolica", ParticionHiperbolica),
    ("Variacional", ParticionVariacional),
    ("REMCMC",      REMCMC),
]

CASOS = [
    ("4A",  tpm_4a,  "1000",          "1111", "1111", "1111"),
    ("4A",  tpm_4a,  "1000",          "1111", "0111", "1111"),
    ("10A", tpm_10a, "1000000000",    "1111111111", "1111111111", "1111111111"),
]

print(f"\n{'─'*72}")
print(f"{'Sistema':<6} {'Alcance/Mec':<12} {'Estrategia':<20} {'k=2 φ':>10} {'Tiempo':>8} {'vs QN':>6}")
print(f"{'─'*72}")

for sistema, tpm, estado, condicion, alcance, mecanismo in CASOS:
    ref_phi = None
    for nombre, ClaseEst in ESTRATEGIAS:
        try:
            inst = ClaseEst(tpm)
            t0 = time.perf_counter()
            res = inst.aplicar_estrategia(
                estado_inicial=estado,
                condicion=condicion,
                alcance=alcance,
                mecanismo=mecanismo,
                k=2,
            )
            elapsed = time.perf_counter() - t0
            phi = res.perdida

            if nombre == "QNodos":
                ref_phi = phi

            match = ""
            if ref_phi is not None and nombre != "QNodos":
                diff = phi - ref_phi
                match = "✓" if abs(diff) < 1e-5 else f"Δ+{diff:.4f}"

            alc_mec = f"{alcance[:5]}/{mecanismo[:5]}"
            print(f"{sistema:<6} {alc_mec:<12} {nombre:<20} {phi:>10.4f} {elapsed:>7.2f}s {match:>6}")
        except Exception as e:
            print(f"{sistema:<6} {'':12} {nombre:<20} {'ERROR':>10}  {str(e)[:40]}")

    print(f"{'─'*72}")

print()

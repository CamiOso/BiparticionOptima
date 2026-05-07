"""Benchmark de SA con n grande: 1 cadena vs 3 cadenas.

QNodos con n=7 (14 vertices de pares temporales) es el caso de prueba
porque supera el umbral_exacto=8 y el SA corre en serio.
Geometric usa DP para n<=12, por lo que el SA solo entraría en sistemas
de 13+ nodos (demasiado grandes para un benchmark practico).

Compara:
  - 1 cadena SA (comportamiento anterior a n_cadenas=3)
  - 3 cadenas SA (implementacion actual)

Metrica: phi obtenido (menor es mejor) y cuantos casos mejora la cadena extra.
"""

from __future__ import annotations

import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.estrategias.q_nodos import QNodos
from src.funciones import k_particion_buscador as _mod

K_VALS = (3, 4)
N_NODOS = 7
SEMILLAS = [3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79]
OUT_DIR = Path("review/benchmarks")


@dataclass
class FilaSA:
    k: int
    semilla: int
    phi_1_cadena: float
    phi_3_cadenas: float
    mejora: float
    tiempo_1_cadena: float
    tiempo_3_cadenas: float


def _random_tpm(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random((1 << n, n), dtype=np.float32)


def _run_qnodos(tpm: np.ndarray, k: int, n_cadenas: int) -> tuple[float, float]:
    orig = _mod.BuscadorKRecocido._multi_recocido

    if n_cadenas == 1:
        def _single(self, k_, semilla):
            return self._recocido(k_, semilla)
        _mod.BuscadorKRecocido._multi_recocido = _single

    t0 = time.perf_counter()
    q = QNodos(tpm)
    res = q.aplicar_estrategia(
        "0" * N_NODOS, "1" * N_NODOS, "1" * N_NODOS, "1" * N_NODOS, k=k
    )
    dt = time.perf_counter() - t0

    if n_cadenas == 1:
        _mod.BuscadorKRecocido._multi_recocido = orig

    return float(res.perdida), dt


def ejecutar() -> list[FilaSA]:
    filas: list[FilaSA] = []
    for k in K_VALS:
        for semilla in SEMILLAS:
            tpm = _random_tpm(N_NODOS, semilla)
            phi1, t1 = _run_qnodos(tpm, k, n_cadenas=1)
            phi3, t3 = _run_qnodos(tpm, k, n_cadenas=3)
            filas.append(FilaSA(
                k=k,
                semilla=semilla,
                phi_1_cadena=phi1,
                phi_3_cadenas=phi3,
                mejora=phi1 - phi3,
                tiempo_1_cadena=t1,
                tiempo_3_cadenas=t3,
            ))
    return filas


def _guardar_csv(filas: list[FilaSA]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / "sa_n_grande_detalle.csv"
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k","semilla","phi_1_cadena","phi_3_cadenas","mejora","t_1_cadena_s","t_3_cadenas_s"])
        for r in filas:
            w.writerow([r.k, r.semilla,
                        f"{r.phi_1_cadena:.6f}", f"{r.phi_3_cadenas:.6f}",
                        f"{r.mejora:.6f}", f"{r.tiempo_1_cadena:.4f}", f"{r.tiempo_3_cadenas:.4f}"])
    return dest


def _imprimir(filas: list[FilaSA]) -> None:
    print("\n" + "=" * 72)
    print(f"SA 1 cadena vs 3 cadenas — QNodos n={N_NODOS} ({N_NODOS*2} vertices, SA activo)")
    print("=" * 72)

    for k in K_VALS:
        grupo = [r for r in filas if r.k == k]
        phi1 = [r.phi_1_cadena for r in grupo]
        phi3 = [r.phi_3_cadenas for r in grupo]
        mejoras = [r.mejora for r in grupo]
        t1 = [r.tiempo_1_cadena for r in grupo]
        t3 = [r.tiempo_3_cadenas for r in grupo]
        mejoro = sum(1 for m in mejoras if m > 1e-9)
        empeoro = sum(1 for m in mejoras if m < -1e-9)
        igual = len(mejoras) - mejoro - empeoro

        print(f"\nk={k}:")
        print(f"  {'':25s} {'1 cadena':>10} {'3 cadenas':>10} {'diferencia':>10}")
        print(f"  {'phi promedio':25s} {statistics.fmean(phi1):>10.6f} {statistics.fmean(phi3):>10.6f} {statistics.fmean(mejoras):>+10.6f}")
        print(f"  {'phi mediana':25s} {statistics.median(phi1):>10.6f} {statistics.median(phi3):>10.6f}")
        print(f"  {'phi minimo':25s} {min(phi1):>10.6f} {min(phi3):>10.6f}")
        print(f"  {'phi maximo':25s} {max(phi1):>10.6f} {max(phi3):>10.6f}")
        print(f"  {'tiempo promedio (s)':25s} {statistics.fmean(t1):>10.4f} {statistics.fmean(t3):>10.4f}")
        print(f"  Casos mejorados: {mejoro}/{len(grupo)}  |  Iguales: {igual}/{len(grupo)}  |  Peores: {empeoro}/{len(grupo)}")


if __name__ == "__main__":
    filas = ejecutar()
    _imprimir(filas)
    ruta = _guardar_csv(filas)
    print(f"\nDetalle guardado en: {ruta}")

"""Benchmark Circuito, Geometric y QNodos en sistemas grandes (n=6 y n=7).

Extiende el benchmark de todas las estrategias a sistemas mas grandes para
evaluar si la mejora de la estrategia Circuito (Laplaciano de hipergrafo)
se sostiene fuera de n=4,5. Se usa k=2 para mantener FuerzaBruta como
referencia exacta factible (63 biparticiones en n=7).

Para reproducir:
    source .venv/bin/activate
    PYTHONPATH=. python review/benchmarks/benchmark_n_grande_circuito.py
"""

from __future__ import annotations

import csv
import importlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OUT_DIR = Path("review/benchmarks")

ESTRATEGIAS: list[tuple[str, str, str]] = [
    ("FuerzaBruta", "src.estrategias.fuerza_bruta",   "FuerzaBruta"),
    ("QNodos",      "src.estrategias.q_nodos",        "QNodos"),
    ("Geometric",   "src.strategies.geometric",       "Geometric"),
    ("Circuito",    "src.estrategias.circuito",        "Circuito"),
]

CONFIGS: list[tuple[int, int]] = [(6, 2), (7, 2)]
SEMILLAS = [11, 23, 37, 53, 71]


@dataclass
class Resultado:
    nombre: str
    n: int
    k: int
    semilla: int
    phi: float
    tiempo: float
    error: str = ""


def _random_tpm(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random((1 << n, n), dtype=np.float32)


def _medir(cls, tpm: np.ndarray, n: int, k: int) -> tuple[float, float, str]:
    try:
        obj = cls(tpm)
        estado = "0" * n
        mascara = "1" * n
        t0 = time.perf_counter()
        res = obj.aplicar_estrategia(estado, mascara, mascara, mascara, k=k)
        dt = time.perf_counter() - t0
        return float(res.perdida), dt, ""
    except Exception as e:
        return float("inf"), 0.0, str(e)


def ejecutar() -> list[Resultado]:
    clases = {}
    for nombre, modulo, clase in ESTRATEGIAS:
        try:
            mod = importlib.import_module(modulo)
            clases[nombre] = getattr(mod, clase)
        except Exception as e:
            print(f"  No se pudo cargar {nombre}: {e}")

    resultados: list[Resultado] = []
    for n, k in CONFIGS:
        print(f"\n=== n={n}, k={k} ===")
        print(f"  {'Estrategia':12s}  " + "  ".join(f"s={s:>3}" for s in SEMILLAS))
        for nombre, cls in clases.items():
            phis_str = []
            for semilla in SEMILLAS:
                tpm = _random_tpm(n, semilla)
                phi, dt, err = _medir(cls, tpm, n, k)
                resultados.append(Resultado(nombre=nombre, n=n, k=k,
                                             semilla=semilla, phi=phi,
                                             tiempo=dt, error=err))
                phis_str.append(f"{phi:.4f}" if phi < float("inf") else " ERR")
            print(f"  {nombre:12s}  " + "  ".join(f"{s:>7}" for s in phis_str))
    return resultados


def resumir_e_imprimir(resultados: list[Resultado]) -> None:
    exacto: dict[tuple[int, int, int], float] = {}
    for r in resultados:
        if r.nombre == "FuerzaBruta" and not r.error and r.phi < float("inf"):
            exacto[(r.n, r.k, r.semilla)] = r.phi

    print("\n" + "=" * 78)
    print("RESUMEN — brecha respecto a FuerzaBruta (menor phi = mejor)")
    print("=" * 78)

    for n, k in CONFIGS:
        print(f"\nn={n}, k={k}")
        print(f"  {'Estrategia':12s}  {'phi prom':>9}  {'brecha %':>9}  {'t prom s':>9}  {'ok/tot':>6}")
        print("  " + "-" * 54)

        nombres_orden = [e[0] for e in ESTRATEGIAS]
        for nombre in nombres_orden:
            grupo = [r for r in resultados if r.nombre == nombre and r.n == n and r.k == k]
            validos = [r for r in grupo if not r.error and r.phi < float("inf")]
            if not validos:
                continue
            phis = [r.phi for r in validos]
            tiempos = [r.tiempo for r in validos]
            brechas = []
            for r in validos:
                opt = exacto.get((r.n, r.k, r.semilla))
                if opt is not None and opt > 1e-12:
                    brechas.append((r.phi - opt) / opt * 100)
                elif opt is not None:
                    brechas.append(0.0)
            phi_prom = sum(phis) / len(phis)
            t_prom = sum(tiempos) / len(tiempos)
            brecha_str = f"{sum(brechas)/len(brechas):>+9.1f}%" if brechas else "      —"
            print(f"  {nombre:12s}  {phi_prom:>9.4f}  {brecha_str:>9}  {t_prom:>9.4f}  "
                  f"{len(validos):>3}/{len(grupo)}")


def guardar_csv(resultados: list[Resultado]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / "n_grande_circuito_detalle.csv"
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["estrategia", "n", "k", "semilla", "phi", "tiempo_s", "error"])
        for r in resultados:
            w.writerow([r.nombre, r.n, r.k, r.semilla,
                        f"{r.phi:.6f}", f"{r.tiempo:.4f}", r.error])
    return dest


if __name__ == "__main__":
    print("Benchmark Circuito en sistemas grandes (n=6, n=7) — k=2")
    print("(FuerzaBruta como referencia exacta: 31 y 63 biparticiones respectivamente)\n")
    resultados = ejecutar()
    resumir_e_imprimir(resultados)
    ruta = guardar_csv(resultados)
    print(f"\nDatos guardados en: {ruta}")

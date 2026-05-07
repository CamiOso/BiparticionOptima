"""Benchmark comparativo de todas las estrategias disponibles.

Ejecuta cada estrategia sobre los mismos sistemas aleatorios y compara:
  - phi obtenido (perdida de informacion, menor es mejor)
  - tiempo de ejecucion
  - brecha con el optimo exacto (FuerzaBruta como referencia)

Configuracion: n=4 y n=5 nodos, k=2 y k=3, 5 semillas por caso.
ParticionILP se excluye por requerir scipy (no instalado en este entorno).
"""

from __future__ import annotations

import csv
import importlib
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

OUT_DIR = Path("review/benchmarks")

ESTRATEGIAS: list[tuple[str, str, str]] = [
    ("FuerzaBruta",      "src.estrategias.fuerza_bruta",            "FuerzaBruta"),
    ("QNodos",           "src.estrategias.q_nodos",                 "QNodos"),
    ("Geometric",        "src.strategies.geometric",                "Geometric"),
    ("Circuito",         "src.estrategias.circuito",                "Circuito"),
    ("InfoBottleneck",   "src.estrategias.informacion_bottleneck",  "InformacionBottleneck"),
    ("Louvain",          "src.estrategias.louvain",                 "Louvain"),
    ("Genetico",         "src.estrategias.genetico",                "AlgoritmoGenetico"),
    ("BeliefProp",       "src.estrategias.belief_propagation",      "BeliefPropagation"),
]

CONFIGS: list[tuple[int, int]] = [(4, 2), (4, 3), (5, 2), (5, 3)]
SEMILLAS = [11, 23, 37, 53, 71]


@dataclass
class ResultadoEstrategia:
    nombre: str
    n: int
    k: int
    semilla: int
    phi: float
    tiempo: float
    error: str = ""


@dataclass
class ResumenEstrategia:
    nombre: str
    n: int
    k: int
    phi_prom: float
    phi_min: float
    phi_max: float
    tiempo_prom: float
    brecha_optimo_prom: float
    victorias: int
    muestras: int
    errores: int


def _random_tpm(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random((1 << n, n), dtype=np.float32)


def _cargar_estrategia(modulo: str, clase: str):
    mod = importlib.import_module(modulo)
    return getattr(mod, clase)


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


def ejecutar() -> list[ResultadoEstrategia]:
    clases = {}
    for nombre, modulo, clase in ESTRATEGIAS:
        try:
            clases[nombre] = _cargar_estrategia(modulo, clase)
        except Exception as e:
            print(f"  No se pudo cargar {nombre}: {e}")

    resultados: list[ResultadoEstrategia] = []
    for n, k in CONFIGS:
        for semilla in SEMILLAS:
            tpm = _random_tpm(n, semilla)
            for nombre, cls in clases.items():
                phi, dt, err = _medir(cls, tpm, n, k)
                resultados.append(ResultadoEstrategia(
                    nombre=nombre, n=n, k=k, semilla=semilla,
                    phi=phi, tiempo=dt, error=err,
                ))
    return resultados


def resumir(resultados: list[ResultadoEstrategia]) -> list[ResumenEstrategia]:
    # Calcular phi exacto (FuerzaBruta) por (n, k, semilla)
    exacto: dict[tuple[int, int, int], float] = {}
    for r in resultados:
        if r.nombre == "FuerzaBruta" and not r.error:
            exacto[(r.n, r.k, r.semilla)] = r.phi

    resumenes: list[ResumenEstrategia] = []
    nombres = [n for n, _, _ in ESTRATEGIAS]

    for nombre in nombres:
        for n, k in CONFIGS:
            grupo = [r for r in resultados if r.nombre == nombre and r.n == n and r.k == k]
            validos = [r for r in grupo if not r.error and r.phi < float("inf")]
            errores = len(grupo) - len(validos)

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

            # Victorias: cuantas veces esta estrategia tiene el menor phi del grupo
            victorias = 0
            for r in validos:
                todos_para_caso = [
                    x.phi for x in resultados
                    if x.n == r.n and x.k == r.k and x.semilla == r.semilla and not x.error
                ]
                if todos_para_caso and r.phi <= min(todos_para_caso) + 1e-9:
                    victorias += 1

            resumenes.append(ResumenEstrategia(
                nombre=nombre, n=n, k=k,
                phi_prom=statistics.fmean(phis),
                phi_min=min(phis),
                phi_max=max(phis),
                tiempo_prom=statistics.fmean(tiempos),
                brecha_optimo_prom=statistics.fmean(brechas) if brechas else float("nan"),
                victorias=victorias,
                muestras=len(validos),
                errores=errores,
            ))

    return resumenes


def _imprimir(resumenes: list[ResumenEstrategia]) -> None:
    print("\n" + "=" * 88)
    print("BENCHMARK: Todas las estrategias comparadas")
    print("Referencia de optimo: FuerzaBruta (busqueda exacta exhaustiva)")
    print("=" * 88)

    for n, k in CONFIGS:
        grupo = [r for r in resumenes if r.n == n and r.k == k]
        if not grupo:
            continue
        grupo.sort(key=lambda r: r.phi_prom)

        print(f"\nn={n} nodos, k={k} grupos  ({len(SEMILLAS)} semillas)")
        print(f"  {'Estrategia':16s} {'phi prom':>9} {'phi min':>9} {'phi max':>9} {'brecha %':>9} {'t prom (s)':>10} {'victorias':>9}")
        print("  " + "-" * 74)
        for r in grupo:
            brecha = f"{r.brecha_optimo_prom:+.1f}%" if not (r.brecha_optimo_prom != r.brecha_optimo_prom) else "  —"
            print(
                f"  {r.nombre:16s} {r.phi_prom:>9.4f} {r.phi_min:>9.4f} {r.phi_max:>9.4f} "
                f"{brecha:>9} {r.tiempo_prom:>10.4f} {r.victorias:>5}/{r.muestras}"
            )


def _guardar_csv(resumenes: list[ResumenEstrategia]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / "todas_estrategias_resumen.csv"
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["estrategia","n","k","phi_prom","phi_min","phi_max",
                    "tiempo_prom_s","brecha_optimo_pct","victorias","muestras","errores"])
        for r in resumenes:
            w.writerow([r.nombre, r.n, r.k,
                        f"{r.phi_prom:.6f}", f"{r.phi_min:.6f}", f"{r.phi_max:.6f}",
                        f"{r.tiempo_prom:.4f}",
                        f"{r.brecha_optimo_prom:.2f}" if r.brecha_optimo_prom == r.brecha_optimo_prom else "nan",
                        r.victorias, r.muestras, r.errores])
    return dest


if __name__ == "__main__":
    print("Ejecutando benchmark... (esto puede tardar 1-2 minutos)")
    resultados = ejecutar()
    resumenes = resumir(resultados)
    _imprimir(resumenes)
    ruta = _guardar_csv(resumenes)
    print(f"\nResumen guardado en: {ruta}")

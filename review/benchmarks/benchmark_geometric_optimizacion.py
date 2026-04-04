from __future__ import annotations

import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.modelos.enumeraciones.geometric_mode import GeometricMode
from src.strategies.geometric import Geometric


@dataclass
class FilaOpt:
    nodos: int
    semilla: int
    tiempo_base_s: float
    tiempo_opt_s: float
    speedup_opt_x: float
    phi_base: float
    phi_opt: float
    delta_phi_abs: float


@dataclass
class ResumenOpt:
    nodos: int
    muestras: int
    tiempo_base_prom_s: float
    tiempo_opt_prom_s: float
    speedup_prom_x: float
    speedup_mediana_x: float
    delta_phi_prom_abs: float
    delta_phi_mediana_abs: float


def _random_tpm(num_nodes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((1 << num_nodes, num_nodes), dtype=np.float32)


def _medir(solver: Geometric, estado: str, mascara: str) -> tuple[float, float]:
    inicio = time.perf_counter()
    resultado = solver.aplicar_estrategia(
        estado_inicial=estado,
        condicion=mascara,
        alcance=mascara,
        mecanismo=mascara,
    )
    duracion = time.perf_counter() - inicio
    return duracion, float(resultado.perdida)


def _solver_base(tpm: np.ndarray) -> Geometric:
    solver = Geometric(tpm, mode=GeometricMode.REFINED)
    solver._usar_optimizacion_grandes = False
    solver._usar_paralelizacion_costos = False
    solver._usar_simetrias_hipercubo = False
    return solver


def _solver_opt(tpm: np.ndarray) -> Geometric:
    solver = Geometric(tpm, mode=GeometricMode.REFINED)
    solver._usar_optimizacion_grandes = True
    solver._usar_paralelizacion_costos = True
    solver._usar_simetrias_hipercubo = True
    return solver


def ejecutar_benchmark_opt() -> list[FilaOpt]:
    configuraciones: dict[int, list[int]] = {
        9: [73, 89, 107],
        10: [79, 97],
    }

    filas: list[FilaOpt] = []

    for nodos, semillas in configuraciones.items():
        for semilla in semillas:
            tpm = _random_tpm(nodos, semilla)
            estado = "0" * nodos
            mascara = "1" * nodos

            solver_base = _solver_base(tpm)
            solver_opt = _solver_opt(tpm)

            tiempo_base, phi_base = _medir(solver_base, estado, mascara)
            tiempo_opt, phi_opt = _medir(solver_opt, estado, mascara)

            speedup = tiempo_base / tiempo_opt if tiempo_opt > 0 else float("inf")
            filas.append(
                FilaOpt(
                    nodos=nodos,
                    semilla=semilla,
                    tiempo_base_s=tiempo_base,
                    tiempo_opt_s=tiempo_opt,
                    speedup_opt_x=speedup,
                    phi_base=phi_base,
                    phi_opt=phi_opt,
                    delta_phi_abs=abs(phi_base - phi_opt),
                )
            )

    return filas


def resumir(filas: list[FilaOpt]) -> list[ResumenOpt]:
    agrupado: dict[int, list[FilaOpt]] = {}
    for fila in filas:
        agrupado.setdefault(fila.nodos, []).append(fila)

    salida: list[ResumenOpt] = []
    for nodos, grupo in sorted(agrupado.items()):
        tiempos_base = [f.tiempo_base_s for f in grupo]
        tiempos_opt = [f.tiempo_opt_s for f in grupo]
        speedups = [f.speedup_opt_x for f in grupo]
        deltas = [f.delta_phi_abs for f in grupo]
        salida.append(
            ResumenOpt(
                nodos=nodos,
                muestras=len(grupo),
                tiempo_base_prom_s=statistics.fmean(tiempos_base),
                tiempo_opt_prom_s=statistics.fmean(tiempos_opt),
                speedup_prom_x=statistics.fmean(speedups),
                speedup_mediana_x=statistics.median(speedups),
                delta_phi_prom_abs=statistics.fmean(deltas),
                delta_phi_mediana_abs=statistics.median(deltas),
            )
        )
    return salida


def guardar_detalle(filas: list[FilaOpt], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    destino = out_dir / "geometric_optimizacion_detalle.csv"

    with destino.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "nodos",
                "semilla",
                "tiempo_base_s",
                "tiempo_opt_s",
                "speedup_opt_x",
                "phi_base",
                "phi_opt",
                "delta_phi_abs",
            ]
        )
        for fila in filas:
            writer.writerow(
                [
                    fila.nodos,
                    fila.semilla,
                    f"{fila.tiempo_base_s:.6f}",
                    f"{fila.tiempo_opt_s:.6f}",
                    f"{fila.speedup_opt_x:.2f}",
                    f"{fila.phi_base:.6f}",
                    f"{fila.phi_opt:.6f}",
                    f"{fila.delta_phi_abs:.6f}",
                ]
            )

    return destino


def guardar_resumen(resumenes: list[ResumenOpt], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    destino = out_dir / "geometric_optimizacion_resumen.csv"

    with destino.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "nodos",
                "muestras",
                "tiempo_base_prom_s",
                "tiempo_opt_prom_s",
                "speedup_opt_prom_x",
                "speedup_opt_mediana_x",
                "delta_phi_prom_abs",
                "delta_phi_mediana_abs",
            ]
        )
        for item in resumenes:
            writer.writerow(
                [
                    item.nodos,
                    item.muestras,
                    f"{item.tiempo_base_prom_s:.6f}",
                    f"{item.tiempo_opt_prom_s:.6f}",
                    f"{item.speedup_prom_x:.2f}",
                    f"{item.speedup_mediana_x:.2f}",
                    f"{item.delta_phi_prom_abs:.6f}",
                    f"{item.delta_phi_mediana_abs:.6f}",
                ]
            )

    return destino


def imprimir(resumenes: list[ResumenOpt]) -> None:
    print("nodos | muestras | t_base_prom | t_opt_prom | speedup_opt | delta_phi_prom")
    print("-" * 78)
    for item in resumenes:
        print(
            f"{item.nodos:>5} | "
            f"{item.muestras:>7} | "
            f"{item.tiempo_base_prom_s:>11.3f} | "
            f"{item.tiempo_opt_prom_s:>10.3f} | "
            f"{item.speedup_prom_x:>10.2f}x | "
            f"{item.delta_phi_prom_abs:>13.6f}"
        )


if __name__ == "__main__":
    filas = ejecutar_benchmark_opt()
    resumen = resumir(filas)

    out = Path("review") / "benchmarks"
    ruta_detalle = guardar_detalle(filas, out)
    ruta_resumen = guardar_resumen(resumen, out)

    imprimir(resumen)
    print(f"\nDetalle guardado en: {ruta_detalle}")
    print(f"Resumen guardado en: {ruta_resumen}")

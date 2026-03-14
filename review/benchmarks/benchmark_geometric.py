from __future__ import annotations

import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.estrategias.fuerza_bruta import FuerzaBruta
from src.strategies.geometric import Geometric


@dataclass
class FilaBenchmark:
    nodos: int
    semilla: int
    tiempo_fuerza_bruta: float
    tiempo_geometric: float
    speedup: float
    phi_fuerza_bruta: float
    phi_geometric: float
    diferencia_phi: float


@dataclass
class ResumenBenchmark:
    nodos: int
    muestras: int
    tiempo_fuerza_bruta_prom: float
    tiempo_geometric_prom: float
    speedup_prom: float
    speedup_mediana: float
    diferencia_phi_prom: float
    diferencia_phi_mediana: float


def _random_tpm(num_nodes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((1 << num_nodes, num_nodes), dtype=np.float32)


def _medir_estrategia(
    estrategia,
    estado_inicial: str,
    mascara: str,
) -> tuple[float, float]:
    inicio = time.perf_counter()
    resultado = estrategia.aplicar_estrategia(
        estado_inicial=estado_inicial,
        condicion=mascara,
        alcance=mascara,
        mecanismo=mascara,
    )
    duracion = time.perf_counter() - inicio
    return duracion, float(resultado.perdida)


def ejecutar_benchmark() -> list[FilaBenchmark]:
    configuraciones: dict[int, list[int]] = {
        5: [11, 29, 47],
        6: [13, 31, 53],
        7: [17, 37, 59],
        8: [19, 41, 61],
    }

    filas: list[FilaBenchmark] = []

    for nodos, semillas in configuraciones.items():
        for semilla in semillas:
            tpm = _random_tpm(nodos, semilla)
            estado = "0" * nodos
            mascara = "1" * nodos

            fuerza_bruta = FuerzaBruta(tpm)
            geometric = Geometric(tpm)

            tiempo_fb, phi_fb = _medir_estrategia(fuerza_bruta, estado, mascara)
            tiempo_geo, phi_geo = _medir_estrategia(geometric, estado, mascara)

            speedup = (tiempo_fb / tiempo_geo) if tiempo_geo > 0 else float("inf")

            filas.append(
                FilaBenchmark(
                    nodos=nodos,
                    semilla=semilla,
                    tiempo_fuerza_bruta=tiempo_fb,
                    tiempo_geometric=tiempo_geo,
                    speedup=speedup,
                    phi_fuerza_bruta=phi_fb,
                    phi_geometric=phi_geo,
                    diferencia_phi=abs(phi_fb - phi_geo),
                )
            )

    return filas


def resumir_benchmark(filas: list[FilaBenchmark]) -> list[ResumenBenchmark]:
    agrupado: dict[int, list[FilaBenchmark]] = {}
    for fila in filas:
        agrupado.setdefault(fila.nodos, []).append(fila)

    resumenes: list[ResumenBenchmark] = []
    for nodos, grupo in sorted(agrupado.items()):
        tiempos_fb = [fila.tiempo_fuerza_bruta for fila in grupo]
        tiempos_geo = [fila.tiempo_geometric for fila in grupo]
        speedups = [fila.speedup for fila in grupo]
        deltas_phi = [fila.diferencia_phi for fila in grupo]

        resumenes.append(
            ResumenBenchmark(
                nodos=nodos,
                muestras=len(grupo),
                tiempo_fuerza_bruta_prom=statistics.fmean(tiempos_fb),
                tiempo_geometric_prom=statistics.fmean(tiempos_geo),
                speedup_prom=statistics.fmean(speedups),
                speedup_mediana=statistics.median(speedups),
                diferencia_phi_prom=statistics.fmean(deltas_phi),
                diferencia_phi_mediana=statistics.median(deltas_phi),
            )
        )

    return resumenes


def guardar_reporte(filas: list[FilaBenchmark], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ruta_csv = out_dir / "geometric_vs_fuerza_bruta.csv"

    with ruta_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "nodos",
                "semilla",
                "tiempo_fuerza_bruta_s",
                "tiempo_geometric_s",
                "speedup_x",
                "phi_fuerza_bruta",
                "phi_geometric",
                "diferencia_phi_abs",
            ]
        )

        for fila in filas:
            writer.writerow(
                [
                    fila.nodos,
                    fila.semilla,
                    f"{fila.tiempo_fuerza_bruta:.6f}",
                    f"{fila.tiempo_geometric:.6f}",
                    f"{fila.speedup:.2f}",
                    f"{fila.phi_fuerza_bruta:.6f}",
                    f"{fila.phi_geometric:.6f}",
                    f"{fila.diferencia_phi:.6f}",
                ]
            )

    return ruta_csv


def guardar_resumen(resumenes: list[ResumenBenchmark], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ruta_csv = out_dir / "geometric_vs_fuerza_bruta_resumen.csv"

    with ruta_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "nodos",
                "muestras",
                "tiempo_fuerza_bruta_prom_s",
                "tiempo_geometric_prom_s",
                "speedup_prom_x",
                "speedup_mediana_x",
                "diferencia_phi_prom_abs",
                "diferencia_phi_mediana_abs",
            ]
        )
        for item in resumenes:
            writer.writerow(
                [
                    item.nodos,
                    item.muestras,
                    f"{item.tiempo_fuerza_bruta_prom:.6f}",
                    f"{item.tiempo_geometric_prom:.6f}",
                    f"{item.speedup_prom:.2f}",
                    f"{item.speedup_mediana:.2f}",
                    f"{item.diferencia_phi_prom:.6f}",
                    f"{item.diferencia_phi_mediana:.6f}",
                ]
            )

    return ruta_csv


def imprimir_resumen(filas: list[FilaBenchmark]) -> None:
    print("nodos | t_fb(s) | t_geo(s) | speedup | phi_fb | phi_geo | |delta_phi|")
    print("-" * 74)
    for fila in filas:
        print(
            f"{fila.nodos:>5} | "
            f"{fila.tiempo_fuerza_bruta:>7.4f} | "
            f"{fila.tiempo_geometric:>8.4f} | "
            f"{fila.speedup:>7.2f}x | "
            f"{fila.phi_fuerza_bruta:>6.4f} | "
            f"{fila.phi_geometric:>7.4f} | "
            f"{fila.diferencia_phi:>10.4f}"
        )


def imprimir_tabla_agregada(resumenes: list[ResumenBenchmark]) -> None:
    print("\nResumen agregado por nodos")
    print("nodos | muestras | speedup_prom | speedup_mediana | delta_phi_prom | delta_phi_mediana")
    print("-" * 89)
    for item in resumenes:
        print(
            f"{item.nodos:>5} | "
            f"{item.muestras:>7} | "
            f"{item.speedup_prom:>11.2f}x | "
            f"{item.speedup_mediana:>14.2f}x | "
            f"{item.diferencia_phi_prom:>14.4f} | "
            f"{item.diferencia_phi_mediana:>17.4f}"
        )


if __name__ == "__main__":
    filas_benchmark = ejecutar_benchmark()
    resumenes = resumir_benchmark(filas_benchmark)

    ruta_detalle = guardar_reporte(
        filas_benchmark,
        Path("review") / "benchmarks",
    )
    ruta_resumen = guardar_resumen(
        resumenes,
        Path("review") / "benchmarks",
    )

    imprimir_resumen(filas_benchmark)
    imprimir_tabla_agregada(resumenes)
    print(f"\nReporte detalle guardado en: {ruta_detalle}")
    print(f"Reporte resumen guardado en: {ruta_resumen}")

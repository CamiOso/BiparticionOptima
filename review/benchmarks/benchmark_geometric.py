from __future__ import annotations

import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.estrategias.fuerza_bruta import FuerzaBruta
from src.modelos.enumeraciones.geometric_mode import GeometricMode
from src.strategies.geometric import Geometric


@dataclass
class FilaBenchmark:
    nodos: int
    semilla: int
    tiempo_fuerza_bruta: float
    tiempo_geometric_estricto: float
    tiempo_geometric_refinado: float
    speedup_estricto: float
    speedup_refinado: float
    phi_fuerza_bruta: float
    phi_geometric_estricto: float
    phi_geometric_refinado: float
    diferencia_phi_estricto: float
    diferencia_phi_refinado: float


@dataclass
class ResumenBenchmark:
    nodos: int
    muestras: int
    tiempo_fuerza_bruta_prom: float
    tiempo_geometric_estricto_prom: float
    tiempo_geometric_refinado_prom: float
    speedup_estricto_prom: float
    speedup_estricto_mediana: float
    speedup_refinado_prom: float
    speedup_refinado_mediana: float
    diferencia_phi_estricto_prom: float
    diferencia_phi_estricto_mediana: float
    diferencia_phi_refinado_prom: float
    diferencia_phi_refinado_mediana: float


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
            geometric_estricto = Geometric(tpm, mode=GeometricMode.STRICT)
            geometric_refinado = Geometric(tpm, mode=GeometricMode.REFINED)

            tiempo_fb, phi_fb = _medir_estrategia(fuerza_bruta, estado, mascara)
            tiempo_geo_estricto, phi_geo_estricto = _medir_estrategia(
                geometric_estricto,
                estado,
                mascara,
            )
            tiempo_geo_refinado, phi_geo_refinado = _medir_estrategia(
                geometric_refinado,
                estado,
                mascara,
            )

            speedup_estricto = (
                tiempo_fb / tiempo_geo_estricto if tiempo_geo_estricto > 0 else float("inf")
            )
            speedup_refinado = (
                tiempo_fb / tiempo_geo_refinado if tiempo_geo_refinado > 0 else float("inf")
            )

            filas.append(
                FilaBenchmark(
                    nodos=nodos,
                    semilla=semilla,
                    tiempo_fuerza_bruta=tiempo_fb,
                    tiempo_geometric_estricto=tiempo_geo_estricto,
                    tiempo_geometric_refinado=tiempo_geo_refinado,
                    speedup_estricto=speedup_estricto,
                    speedup_refinado=speedup_refinado,
                    phi_fuerza_bruta=phi_fb,
                    phi_geometric_estricto=phi_geo_estricto,
                    phi_geometric_refinado=phi_geo_refinado,
                    diferencia_phi_estricto=abs(phi_fb - phi_geo_estricto),
                    diferencia_phi_refinado=abs(phi_fb - phi_geo_refinado),
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
        tiempos_geo_estricto = [fila.tiempo_geometric_estricto for fila in grupo]
        tiempos_geo_refinado = [fila.tiempo_geometric_refinado for fila in grupo]
        speedups_estricto = [fila.speedup_estricto for fila in grupo]
        speedups_refinado = [fila.speedup_refinado for fila in grupo]
        deltas_phi_estricto = [fila.diferencia_phi_estricto for fila in grupo]
        deltas_phi_refinado = [fila.diferencia_phi_refinado for fila in grupo]

        resumenes.append(
            ResumenBenchmark(
                nodos=nodos,
                muestras=len(grupo),
                tiempo_fuerza_bruta_prom=statistics.fmean(tiempos_fb),
                tiempo_geometric_estricto_prom=statistics.fmean(tiempos_geo_estricto),
                tiempo_geometric_refinado_prom=statistics.fmean(tiempos_geo_refinado),
                speedup_estricto_prom=statistics.fmean(speedups_estricto),
                speedup_estricto_mediana=statistics.median(speedups_estricto),
                speedup_refinado_prom=statistics.fmean(speedups_refinado),
                speedup_refinado_mediana=statistics.median(speedups_refinado),
                diferencia_phi_estricto_prom=statistics.fmean(deltas_phi_estricto),
                diferencia_phi_estricto_mediana=statistics.median(deltas_phi_estricto),
                diferencia_phi_refinado_prom=statistics.fmean(deltas_phi_refinado),
                diferencia_phi_refinado_mediana=statistics.median(deltas_phi_refinado),
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
                "tiempo_geometric_estricto_s",
                "tiempo_geometric_refinado_s",
                "speedup_estricto_x",
                "speedup_refinado_x",
                "phi_fuerza_bruta",
                "phi_geometric_estricto",
                "phi_geometric_refinado",
                "diferencia_phi_estricto_abs",
                "diferencia_phi_refinado_abs",
            ]
        )

        for fila in filas:
            writer.writerow(
                [
                    fila.nodos,
                    fila.semilla,
                    f"{fila.tiempo_fuerza_bruta:.6f}",
                    f"{fila.tiempo_geometric_estricto:.6f}",
                    f"{fila.tiempo_geometric_refinado:.6f}",
                    f"{fila.speedup_estricto:.2f}",
                    f"{fila.speedup_refinado:.2f}",
                    f"{fila.phi_fuerza_bruta:.6f}",
                    f"{fila.phi_geometric_estricto:.6f}",
                    f"{fila.phi_geometric_refinado:.6f}",
                    f"{fila.diferencia_phi_estricto:.6f}",
                    f"{fila.diferencia_phi_refinado:.6f}",
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
                "tiempo_geometric_estricto_prom_s",
                "tiempo_geometric_refinado_prom_s",
                "speedup_estricto_prom_x",
                "speedup_estricto_mediana_x",
                "speedup_refinado_prom_x",
                "speedup_refinado_mediana_x",
                "diferencia_phi_estricto_prom_abs",
                "diferencia_phi_estricto_mediana_abs",
                "diferencia_phi_refinado_prom_abs",
                "diferencia_phi_refinado_mediana_abs",
            ]
        )
        for item in resumenes:
            writer.writerow(
                [
                    item.nodos,
                    item.muestras,
                    f"{item.tiempo_fuerza_bruta_prom:.6f}",
                    f"{item.tiempo_geometric_estricto_prom:.6f}",
                    f"{item.tiempo_geometric_refinado_prom:.6f}",
                    f"{item.speedup_estricto_prom:.2f}",
                    f"{item.speedup_estricto_mediana:.2f}",
                    f"{item.speedup_refinado_prom:.2f}",
                    f"{item.speedup_refinado_mediana:.2f}",
                    f"{item.diferencia_phi_estricto_prom:.6f}",
                    f"{item.diferencia_phi_estricto_mediana:.6f}",
                    f"{item.diferencia_phi_refinado_prom:.6f}",
                    f"{item.diferencia_phi_refinado_mediana:.6f}",
                ]
            )

    return ruta_csv


def imprimir_resumen(filas: list[FilaBenchmark]) -> None:
    print("nodos | t_fb | t_geo_e | t_geo_r | sp_e | sp_r | phi_fb | phi_e | phi_r | d_e | d_r")
    print("-" * 108)
    for fila in filas:
        print(
            f"{fila.nodos:>5} | "
            f"{fila.tiempo_fuerza_bruta:>5.2f} | "
            f"{fila.tiempo_geometric_estricto:>7.2f} | "
            f"{fila.tiempo_geometric_refinado:>7.2f} | "
            f"{fila.speedup_estricto:>4.1f}x | "
            f"{fila.speedup_refinado:>4.1f}x | "
            f"{fila.phi_fuerza_bruta:>6.4f} | "
            f"{fila.phi_geometric_estricto:>5.4f} | "
            f"{fila.phi_geometric_refinado:>5.4f} | "
            f"{fila.diferencia_phi_estricto:>5.4f} | "
            f"{fila.diferencia_phi_refinado:>5.4f}"
        )


def imprimir_tabla_agregada(resumenes: list[ResumenBenchmark]) -> None:
    print("\nResumen agregado por nodos")
    print("nodos | muestras | sp_e_prom | sp_e_med | sp_r_prom | sp_r_med | dphi_e_prom | dphi_r_prom")
    print("-" * 104)
    for item in resumenes:
        print(
            f"{item.nodos:>5} | "
            f"{item.muestras:>7} | "
            f"{item.speedup_estricto_prom:>9.2f}x | "
            f"{item.speedup_estricto_mediana:>8.2f}x | "
            f"{item.speedup_refinado_prom:>9.2f}x | "
            f"{item.speedup_refinado_mediana:>8.2f}x | "
            f"{item.diferencia_phi_estricto_prom:>11.4f} | "
            f"{item.diferencia_phi_refinado_prom:>11.4f}"
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

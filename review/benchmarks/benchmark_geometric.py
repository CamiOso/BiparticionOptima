from __future__ import annotations

import csv
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
    configuraciones = [
        (5, 11),
        (6, 13),
        (7, 17),
        (8, 19),
    ]

    filas: list[FilaBenchmark] = []

    for nodos, semilla in configuraciones:
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


if __name__ == "__main__":
    filas_benchmark = ejecutar_benchmark()
    ruta = guardar_reporte(
        filas_benchmark,
        Path("review") / "benchmarks",
    )
    imprimir_resumen(filas_benchmark)
    print(f"\nReporte guardado en: {ruta}")

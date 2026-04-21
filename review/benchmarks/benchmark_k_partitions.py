from __future__ import annotations

import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.estrategias.q_nodos import QNodos
from src.strategies.geometric import Geometric


K_PARTICIONES = (3, 4)
CONFIGURACIONES: dict[int, list[int]] = {
    4: [11, 19, 29, 37, 47],
    5: [13, 23, 31, 41, 53],
    6: [17, 27, 37, 47, 59],
}


@dataclass
class FilaBenchmarkK:
    k: int
    nodos: int
    semilla: int
    tiempo_qnodos: float
    tiempo_geometric: float
    speedup_geometric_sobre_qnodos: float
    speedup_qnodos_sobre_geometric: float
    phi_qnodos: float
    phi_geometric: float
    diferencia_phi_abs: float
    ganador_phi: str
    ganador_tiempo: str


@dataclass
class ResumenBenchmarkK:
    k: int
    nodos: int
    muestras: int
    tiempo_qnodos_prom: float
    tiempo_geometric_prom: float
    speedup_geometric_sobre_qnodos_prom: float
    speedup_qnodos_sobre_geometric_prom: float
    diferencia_phi_prom: float
    diferencia_phi_mediana: float
    victorias_phi_qnodos: int
    victorias_phi_geometric: int
    empates_phi: int
    victorias_tiempo_qnodos: int
    victorias_tiempo_geometric: int
    empates_tiempo: int


def _random_tpm(num_nodes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((1 << num_nodes, num_nodes), dtype=np.float32)


def _medir_estrategia(
    estrategia,
    estado_inicial: str,
    mascara: str,
    k: int,
) -> tuple[float, float]:
    inicio = time.perf_counter()
    resultado = estrategia.aplicar_estrategia(
        estado_inicial=estado_inicial,
        condicion=mascara,
        alcance=mascara,
        mecanismo=mascara,
        k=k,
    )
    duracion = time.perf_counter() - inicio
    return duracion, float(resultado.perdida)


def _ganador_phi(phi_qnodos: float, phi_geometric: float) -> str:
    if abs(phi_qnodos - phi_geometric) <= 1e-12:
        return "empate"
    return "qnodos" if phi_qnodos < phi_geometric else "geometric"


def _ganador_tiempo(tiempo_qnodos: float, tiempo_geometric: float) -> str:
    if abs(tiempo_qnodos - tiempo_geometric) <= 1e-12:
        return "empate"
    return "qnodos" if tiempo_qnodos < tiempo_geometric else "geometric"


def ejecutar_benchmark_k() -> list[FilaBenchmarkK]:
    filas: list[FilaBenchmarkK] = []

    for k in K_PARTICIONES:
        for nodos, semillas in CONFIGURACIONES.items():
            for semilla in semillas:
                tpm = _random_tpm(nodos, semilla)
                estado = "0" * nodos
                mascara = "1" * nodos

                qnodos = QNodos(tpm)
                geometric = Geometric(tpm)

                tiempo_qnodos, phi_qnodos = _medir_estrategia(qnodos, estado, mascara, k)
                tiempo_geometric, phi_geometric = _medir_estrategia(
                    geometric,
                    estado,
                    mascara,
                    k,
                )

                filas.append(
                    FilaBenchmarkK(
                        k=k,
                        nodos=nodos,
                        semilla=semilla,
                        tiempo_qnodos=tiempo_qnodos,
                        tiempo_geometric=tiempo_geometric,
                        speedup_geometric_sobre_qnodos=(
                            tiempo_qnodos / tiempo_geometric if tiempo_geometric > 0 else float("inf")
                        ),
                        speedup_qnodos_sobre_geometric=(
                            tiempo_geometric / tiempo_qnodos if tiempo_qnodos > 0 else float("inf")
                        ),
                        phi_qnodos=phi_qnodos,
                        phi_geometric=phi_geometric,
                        diferencia_phi_abs=abs(phi_qnodos - phi_geometric),
                        ganador_phi=_ganador_phi(phi_qnodos, phi_geometric),
                        ganador_tiempo=_ganador_tiempo(tiempo_qnodos, tiempo_geometric),
                    )
                )

    return filas


def resumir_benchmark_k(filas: list[FilaBenchmarkK]) -> list[ResumenBenchmarkK]:
    agrupado: dict[tuple[int, int], list[FilaBenchmarkK]] = {}
    for fila in filas:
        agrupado.setdefault((fila.k, fila.nodos), []).append(fila)

    resumenes: list[ResumenBenchmarkK] = []
    for (k, nodos), grupo in sorted(agrupado.items()):
        deltas = [fila.diferencia_phi_abs for fila in grupo]
        resumenes.append(
            ResumenBenchmarkK(
                k=k,
                nodos=nodos,
                muestras=len(grupo),
                tiempo_qnodos_prom=statistics.fmean(fila.tiempo_qnodos for fila in grupo),
                tiempo_geometric_prom=statistics.fmean(fila.tiempo_geometric for fila in grupo),
                speedup_geometric_sobre_qnodos_prom=statistics.fmean(
                    fila.speedup_geometric_sobre_qnodos for fila in grupo
                ),
                speedup_qnodos_sobre_geometric_prom=statistics.fmean(
                    fila.speedup_qnodos_sobre_geometric for fila in grupo
                ),
                diferencia_phi_prom=statistics.fmean(deltas),
                diferencia_phi_mediana=statistics.median(deltas),
                victorias_phi_qnodos=sum(1 for fila in grupo if fila.ganador_phi == "qnodos"),
                victorias_phi_geometric=sum(1 for fila in grupo if fila.ganador_phi == "geometric"),
                empates_phi=sum(1 for fila in grupo if fila.ganador_phi == "empate"),
                victorias_tiempo_qnodos=sum(1 for fila in grupo if fila.ganador_tiempo == "qnodos"),
                victorias_tiempo_geometric=sum(
                    1 for fila in grupo if fila.ganador_tiempo == "geometric"
                ),
                empates_tiempo=sum(1 for fila in grupo if fila.ganador_tiempo == "empate"),
            )
        )
    return resumenes


def guardar_detalle(filas: list[FilaBenchmarkK], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    destino = out_dir / "k_particiones_qnodos_vs_geometric.csv"

    with destino.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "k",
                "nodos",
                "semilla",
                "tiempo_qnodos_s",
                "tiempo_geometric_s",
                "speedup_geometric_sobre_qnodos_x",
                "speedup_qnodos_sobre_geometric_x",
                "phi_qnodos",
                "phi_geometric",
                "diferencia_phi_abs",
                "ganador_phi",
                "ganador_tiempo",
            ]
        )
        for fila in filas:
            writer.writerow(
                [
                    fila.k,
                    fila.nodos,
                    fila.semilla,
                    f"{fila.tiempo_qnodos:.6f}",
                    f"{fila.tiempo_geometric:.6f}",
                    f"{fila.speedup_geometric_sobre_qnodos:.2f}",
                    f"{fila.speedup_qnodos_sobre_geometric:.2f}",
                    f"{fila.phi_qnodos:.6f}",
                    f"{fila.phi_geometric:.6f}",
                    f"{fila.diferencia_phi_abs:.6f}",
                    fila.ganador_phi,
                    fila.ganador_tiempo,
                ]
            )

    return destino


def guardar_resumen(resumenes: list[ResumenBenchmarkK], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    destino = out_dir / "k_particiones_qnodos_vs_geometric_resumen.csv"

    with destino.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "k",
                "nodos",
                "muestras",
                "tiempo_qnodos_prom_s",
                "tiempo_geometric_prom_s",
                "speedup_geometric_sobre_qnodos_prom_x",
                "speedup_qnodos_sobre_geometric_prom_x",
                "diferencia_phi_prom_abs",
                "diferencia_phi_mediana_abs",
                "victorias_phi_qnodos",
                "victorias_phi_geometric",
                "empates_phi",
                "victorias_tiempo_qnodos",
                "victorias_tiempo_geometric",
                "empates_tiempo",
            ]
        )
        for item in resumenes:
            writer.writerow(
                [
                    item.k,
                    item.nodos,
                    item.muestras,
                    f"{item.tiempo_qnodos_prom:.6f}",
                    f"{item.tiempo_geometric_prom:.6f}",
                    f"{item.speedup_geometric_sobre_qnodos_prom:.2f}",
                    f"{item.speedup_qnodos_sobre_geometric_prom:.2f}",
                    f"{item.diferencia_phi_prom:.6f}",
                    f"{item.diferencia_phi_mediana:.6f}",
                    item.victorias_phi_qnodos,
                    item.victorias_phi_geometric,
                    item.empates_phi,
                    item.victorias_tiempo_qnodos,
                    item.victorias_tiempo_geometric,
                    item.empates_tiempo,
                ]
            )

    return destino


def guardar_resumen_markdown(resumenes: list[ResumenBenchmarkK], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    destino = out_dir / "k_particiones_qnodos_vs_geometric_resumen.md"

    lineas = [
        "# Resumen benchmark QNodos-k vs Geometric-k",
        "",
        "| k | nodos | muestras | t_qnodos_prom (s) | t_geometric_prom (s) | dphi_prom | dphi_mediana | phi_q_wins | phi_g_wins | t_q_wins | t_g_wins |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in resumenes:
        lineas.append(
            f"| {item.k} | {item.nodos} | {item.muestras} | "
            f"{item.tiempo_qnodos_prom:.4f} | {item.tiempo_geometric_prom:.4f} | "
            f"{item.diferencia_phi_prom:.6f} | {item.diferencia_phi_mediana:.6f} | "
            f"{item.victorias_phi_qnodos} | {item.victorias_phi_geometric} | "
            f"{item.victorias_tiempo_qnodos} | {item.victorias_tiempo_geometric} |"
        )

    destino.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return destino


def imprimir_detalle(filas: list[FilaBenchmarkK]) -> None:
    print("k | nodos | semilla | t_qnodos | t_geo | phi_qnodos | phi_geo | ganador_phi | ganador_tiempo")
    print("-" * 104)
    for fila in filas:
        print(
            f"{fila.k:>1} | "
            f"{fila.nodos:>5} | "
            f"{fila.semilla:>7} | "
            f"{fila.tiempo_qnodos:>8.4f} | "
            f"{fila.tiempo_geometric:>5.4f} | "
            f"{fila.phi_qnodos:>10.6f} | "
            f"{fila.phi_geometric:>7.6f} | "
            f"{fila.ganador_phi:>11} | "
            f"{fila.ganador_tiempo:>14}"
        )


def imprimir_resumen(resumenes: list[ResumenBenchmarkK]) -> None:
    print("\nResumen agregado por k y nodos")
    print("k | nodos | muestras | dphi_prom | dphi_med | phi_q_wins | phi_g_wins | t_q_wins | t_g_wins")
    print("-" * 104)
    for item in resumenes:
        print(
            f"{item.k:>1} | "
            f"{item.nodos:>5} | "
            f"{item.muestras:>7} | "
            f"{item.diferencia_phi_prom:>9.6f} | "
            f"{item.diferencia_phi_mediana:>8.6f} | "
            f"{item.victorias_phi_qnodos:>10} | "
            f"{item.victorias_phi_geometric:>10} | "
            f"{item.victorias_tiempo_qnodos:>8} | "
            f"{item.victorias_tiempo_geometric:>8}"
        )


if __name__ == "__main__":
    filas_benchmark = ejecutar_benchmark_k()
    resumenes = resumir_benchmark_k(filas_benchmark)

    out_dir = Path("review") / "benchmarks"
    ruta_detalle = guardar_detalle(filas_benchmark, out_dir)
    ruta_resumen = guardar_resumen(resumenes, out_dir)
    ruta_resumen_md = guardar_resumen_markdown(resumenes, out_dir)

    imprimir_detalle(filas_benchmark)
    imprimir_resumen(resumenes)
    print(f"\nDetalle guardado en: {ruta_detalle}")
    print(f"Resumen CSV guardado en: {ruta_resumen}")
    print(f"Resumen Markdown guardado en: {ruta_resumen_md}")

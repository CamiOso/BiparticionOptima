"""Benchmark de impacto de swap moves y multiples cadenas SA.

Compara los resultados de Geometric y QNodos ANTES de las mejoras
(datos del CSV historico generado con el codigo anterior) contra
DESPUES (corrida actual con swap moves + n_cadenas=3).

Metrica principal: reduccion en phi (perdida de informacion).
Metrica secundaria: cambio en tiempo de ejecucion.
"""

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
CSV_HISTORICO = Path("review/benchmarks/k_particiones_qnodos_vs_geometric.csv")


@dataclass
class FilaMejora:
    k: int
    nodos: int
    semilla: int
    phi_qnodos_antes: float
    phi_qnodos_ahora: float
    phi_geometric_antes: float
    phi_geometric_ahora: float
    mejora_qnodos: float
    mejora_geometric: float
    tiempo_qnodos_s: float
    tiempo_geometric_s: float


def _random_tpm(num_nodes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((1 << num_nodes, num_nodes), dtype=np.float32)


def _medir(estrategia, estado: str, mascara: str, k: int) -> tuple[float, float]:
    t0 = time.perf_counter()
    res = estrategia.aplicar_estrategia(estado, mascara, mascara, mascara, k=k)
    return time.perf_counter() - t0, float(res.perdida)


def _cargar_historico(path: Path) -> dict[tuple[int, int, int], tuple[float, float]]:
    datos: dict[tuple[int, int, int], tuple[float, float]] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            clave = (int(row["k"]), int(row["nodos"]), int(row["semilla"]))
            datos[clave] = (float(row["phi_qnodos"]), float(row["phi_geometric"]))
    return datos


def ejecutar_benchmark() -> list[FilaMejora]:
    historico = _cargar_historico(CSV_HISTORICO)
    filas: list[FilaMejora] = []

    for k in K_PARTICIONES:
        for nodos, semillas in CONFIGURACIONES.items():
            for semilla in semillas:
                tpm = _random_tpm(nodos, semilla)
                estado = "0" * nodos
                mascara = "1" * nodos

                t_q, phi_q_nuevo = _medir(QNodos(tpm), estado, mascara, k)
                t_g, phi_g_nuevo = _medir(Geometric(tpm), estado, mascara, k)

                phi_q_viejo, phi_g_viejo = historico[(k, nodos, semilla)]

                filas.append(FilaMejora(
                    k=k,
                    nodos=nodos,
                    semilla=semilla,
                    phi_qnodos_antes=phi_q_viejo,
                    phi_qnodos_ahora=phi_q_nuevo,
                    phi_geometric_antes=phi_g_viejo,
                    phi_geometric_ahora=phi_g_nuevo,
                    mejora_qnodos=phi_q_viejo - phi_q_nuevo,
                    mejora_geometric=phi_g_viejo - phi_g_nuevo,
                    tiempo_qnodos_s=t_q,
                    tiempo_geometric_s=t_g,
                ))

    return filas


def _guardar_csv(filas: list[FilaMejora], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    destino = out_dir / "sa_mejoras_detalle.csv"
    with destino.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "k", "nodos", "semilla",
            "phi_qnodos_antes", "phi_qnodos_ahora", "mejora_qnodos",
            "phi_geometric_antes", "phi_geometric_ahora", "mejora_geometric",
            "tiempo_qnodos_s", "tiempo_geometric_s",
        ])
        for r in filas:
            writer.writerow([
                r.k, r.nodos, r.semilla,
                f"{r.phi_qnodos_antes:.6f}", f"{r.phi_qnodos_ahora:.6f}", f"{r.mejora_qnodos:.6f}",
                f"{r.phi_geometric_antes:.6f}", f"{r.phi_geometric_ahora:.6f}", f"{r.mejora_geometric:.6f}",
                f"{r.tiempo_qnodos_s:.4f}", f"{r.tiempo_geometric_s:.4f}",
            ])
    return destino


def _imprimir_resumen(filas: list[FilaMejora]) -> dict:
    agrupado: dict[tuple[int, int], list[FilaMejora]] = {}
    for f in filas:
        agrupado.setdefault((f.k, f.nodos), []).append(f)

    resumen_global: dict = {}

    print("\n" + "=" * 80)
    print("BENCHMARK: Impacto de swap moves + 3 cadenas SA")
    print("Comparacion contra CSV historico (codigo anterior)")
    print("=" * 80)

    for estrategia, campo_antes, campo_ahora, campo_mejora in [
        ("QNodos",   "phi_qnodos_antes",   "phi_qnodos_ahora",   "mejora_qnodos"),
        ("Geometric","phi_geometric_antes", "phi_geometric_ahora","mejora_geometric"),
    ]:
        print(f"\n--- {estrategia} ---")
        print(f"{'k':>2} | {'nodos':>5} | {'phi antes':>10} | {'phi ahora':>10} | {'mejora':>9} | {'tiempo (s)':>10}")
        print("-" * 60)

        mejoras_total = []
        for (k, nodos), grupo in sorted(agrupado.items()):
            antes_vals = [getattr(r, campo_antes) for r in grupo]
            ahora_vals = [getattr(r, campo_ahora) for r in grupo]
            mejora_vals = [getattr(r, campo_mejora) for r in grupo]
            if estrategia == "QNodos":
                tiempos = [r.tiempo_qnodos_s for r in grupo]
            else:
                tiempos = [r.tiempo_geometric_s for r in grupo]

            prom_antes = statistics.fmean(antes_vals)
            prom_ahora = statistics.fmean(ahora_vals)
            prom_mejora = statistics.fmean(mejora_vals)
            prom_tiempo = statistics.fmean(tiempos)
            mejoras_total.extend(mejora_vals)

            print(
                f"{k:>2} | {nodos:>5} | {prom_antes:>10.6f} | {prom_ahora:>10.6f} | "
                f"{prom_mejora:>+9.6f} | {prom_tiempo:>10.4f}"
            )

        mejora_global = statistics.fmean(mejoras_total)
        casos_mejoraron = sum(1 for r in filas if getattr(r, campo_mejora) > 1e-9)
        casos_empeoraron = sum(1 for r in filas if getattr(r, campo_mejora) < -1e-9)
        resumen_global[estrategia] = {
            "mejora_promedio": mejora_global,
            "casos_mejoraron": casos_mejoraron,
            "casos_empeoraron": casos_empeoraron,
        }
        print(f"\n  Mejora promedio global en phi: {mejora_global:+.6f}")
        print(f"  Casos que mejoraron: {casos_mejoraron}/{len(filas)}")
        print(f"  Casos que empeoraron: {casos_empeoraron}/{len(filas)}")

    return resumen_global


if __name__ == "__main__":
    filas = ejecutar_benchmark()
    resumen = _imprimir_resumen(filas)

    out_dir = Path("review/benchmarks")
    ruta = _guardar_csv(filas, out_dir)
    print(f"\nDetalle guardado en: {ruta}")

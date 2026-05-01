"""Benchmark comparativo de estrategias de biparticion.

Ejecuta todas las estrategias disponibles sobre una o varias redes y
consolida los resultados en una tabla con perdida, tiempo de ejecucion
y diferencia porcentual respecto a la fuerza bruta (linea base exacta).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.estrategias.fuerza_bruta import FuerzaBruta
from src.estrategias.geometrica import Geometric
from src.estrategias.phi import Phi
from src.estrategias.q_nodos import QNodos
from src.modelos.enumeraciones.geometric_mode import GeometricMode
from src.modelos.base.aplicacion import aplicacion


@dataclass
class FilaBenchmark:
    """Una fila de la tabla de resultados."""

    estrategia: str
    estado_inicial: str
    perdida: float
    tiempo_seg: float
    diferencia_vs_bruta: float | None = None


@dataclass
class ResultadoBenchmark:
    """Resultado completo de una corrida de benchmark."""

    filas: list[FilaBenchmark] = field(default_factory=list)

    def agregar(self, fila: FilaBenchmark) -> None:
        self.filas.append(fila)

    def perdida_bruta(self, estado_inicial: str) -> float | None:
        for fila in self.filas:
            if fila.estrategia == "FuerzaBruta" and fila.estado_inicial == estado_inicial:
                return fila.perdida
        return None

    def calcular_diferencias(self) -> None:
        """Calcula la diferencia porcentual de cada estrategia vs fuerza bruta."""
        for fila in self.filas:
            if fila.estrategia == "FuerzaBruta":
                fila.diferencia_vs_bruta = 0.0
                continue
            ref = self.perdida_bruta(fila.estado_inicial)
            if ref is None:
                continue
            if ref == 0.0:
                fila.diferencia_vs_bruta = 0.0 if fila.perdida == 0.0 else float("inf")
            else:
                fila.diferencia_vs_bruta = 100.0 * (fila.perdida - ref) / ref

    def imprimir(self) -> None:
        """Imprime la tabla de resultados en consola."""
        ancho = 72
        print("=" * ancho)
        print(f"{'BENCHMARK DE ESTRATEGIAS':^{ancho}}")
        print("=" * ancho)
        encabezado = f"{'Estrategia':<22} {'Estado':>8} {'Perdida':>10} {'Tiempo(s)':>10} {'Dif%':>8}"
        print(encabezado)
        print("-" * ancho)
        for fila in self.filas:
            dif = f"{fila.diferencia_vs_bruta:+.2f}" if fila.diferencia_vs_bruta is not None else "  N/A"
            print(
                f"{fila.estrategia:<22} "
                f"{fila.estado_inicial:>8} "
                f"{fila.perdida:>10.4f} "
                f"{fila.tiempo_seg:>10.4f} "
                f"{dif:>8}"
            )
        print("=" * ancho)

    def resumen(self) -> dict[str, dict]:
        """Devuelve estadisticas por estrategia: media, min, max de perdida y tiempo."""
        por_estrategia: dict[str, list[FilaBenchmark]] = {}
        for fila in self.filas:
            por_estrategia.setdefault(fila.estrategia, []).append(fila)

        stats = {}
        for nombre, filas in por_estrategia.items():
            perdidas = [f.perdida for f in filas]
            tiempos = [f.tiempo_seg for f in filas]
            stats[nombre] = {
                "perdida_media": float(np.mean(perdidas)),
                "perdida_min": float(np.min(perdidas)),
                "perdida_max": float(np.max(perdidas)),
                "tiempo_medio": float(np.mean(tiempos)),
                "tiempo_total": float(np.sum(tiempos)),
            }
        return stats


class Benchmark:
    """Ejecuta y compara estrategias de biparticion sobre una TPM dada.

    Uso basico:
        bench = Benchmark(tpm)
        resultado = bench.ejecutar(estados=["1000", "0100"])
        resultado.imprimir()
    """

    def __init__(self, tpm: np.ndarray, k: int = 2) -> None:
        self.tpm = tpm
        self.k = k
        self._n = tpm.shape[1]

    def _mascara_completa(self, estado: str) -> str:
        return "1" * len(estado)

    def _medir(self, fn: Callable) -> tuple[float, float]:
        """Ejecuta fn y devuelve (perdida, segundos)."""
        inicio = time.perf_counter()
        resultado = fn()
        elapsed = time.perf_counter() - inicio
        return float(resultado.perdida), elapsed

    def ejecutar(
        self,
        estados: list[str] | None = None,
        incluir_geometric_estricto: bool = True,
        incluir_geometric_refinado: bool = True,
        incluir_phi: bool = True,
        incluir_qnodos: bool = True,
    ) -> ResultadoBenchmark:
        """Corre el benchmark sobre los estados dados y devuelve la tabla de resultados."""
        if estados is None:
            estados = ["1" + "0" * (self._n - 1)]

        for estado in estados:
            if len(estado) != self._n:
                raise ValueError(
                    f"Estado '{estado}' tiene longitud {len(estado)}, "
                    f"se esperaba {self._n}."
                )
            if any(c not in {"0", "1"} for c in estado):
                raise ValueError(f"Estado invalido: '{estado}'. Solo se permiten 0 y 1.")

        tabla = ResultadoBenchmark()

        for estado in estados:
            mascara = self._mascara_completa(estado)
            kwargs = dict(
                estado_inicial=estado,
                condicion=mascara,
                alcance=mascara,
                mecanismo=mascara,
            )

            # Fuerza bruta (referencia exacta).
            solver_fb = FuerzaBruta(self.tpm)
            perdida, tiempo = self._medir(lambda s=solver_fb, kw=kwargs: s.aplicar_estrategia(**kw))
            tabla.agregar(FilaBenchmark("FuerzaBruta", estado, perdida, tiempo))

            if incluir_phi:
                solver_phi = Phi(self.tpm)
                perdida, tiempo = self._medir(
                    lambda s=solver_phi, kw=kwargs: s.aplicar_estrategia(**kw)
                )
                tabla.agregar(FilaBenchmark("Phi", estado, perdida, tiempo))

            if incluir_qnodos:
                solver_q = QNodos(self.tpm)
                kwargs_k = {**kwargs, "k": self.k}
                perdida, tiempo = self._medir(
                    lambda s=solver_q, kw=kwargs_k: s.aplicar_estrategia(**kw)
                )
                etiqueta = f"QNodos(k={self.k})"
                tabla.agregar(FilaBenchmark(etiqueta, estado, perdida, tiempo))

            if incluir_geometric_estricto:
                modo_anterior = aplicacion.modo_geometrico
                aplicacion.set_modo_geometrico(GeometricMode.STRICT)
                solver_ge = Geometric(self.tpm, mode=GeometricMode.STRICT)
                kwargs_k = {**kwargs, "k": self.k}
                perdida, tiempo = self._medir(
                    lambda s=solver_ge, kw=kwargs_k: s.aplicar_estrategia(**kw)
                )
                aplicacion.modo_geometrico = modo_anterior
                etiqueta = f"Geometric-estricto(k={self.k})"
                tabla.agregar(FilaBenchmark(etiqueta, estado, perdida, tiempo))

            if incluir_geometric_refinado:
                modo_anterior = aplicacion.modo_geometrico
                aplicacion.set_modo_geometrico(GeometricMode.REFINED)
                solver_gr = Geometric(self.tpm, mode=GeometricMode.REFINED)
                kwargs_k = {**kwargs, "k": self.k}
                perdida, tiempo = self._medir(
                    lambda s=solver_gr, kw=kwargs_k: s.aplicar_estrategia(**kw)
                )
                aplicacion.modo_geometrico = modo_anterior
                etiqueta = f"Geometric-refinado(k={self.k})"
                tabla.agregar(FilaBenchmark(etiqueta, estado, perdida, tiempo))

        tabla.calcular_diferencias()
        return tabla

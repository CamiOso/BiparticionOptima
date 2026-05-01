from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.funciones.particiones import k_particiones_asignacion


@dataclass(frozen=True)
class ResultadoKParticion:
    """Resultado de una busqueda de k-particion."""

    perdida: float
    distribucion: np.ndarray
    asignacion: tuple[int, ...]


class BuscadorKParticion(ABC):
    """Algoritmo de busqueda de k-particion optima (patron Template Method).

    Define la estructura del algoritmo: busqueda exacta para sistemas pequenos
    y busqueda local con reinicios para sistemas grandes. Las subclases concretas
    implementan evaluar_asignacion() segun como construyen la particion del sistema.
    """

    def __init__(
        self,
        umbral_exacto: int = 8,
        max_iter_refinamiento: int = 24,
        max_restarts: int = 16,
    ) -> None:
        self.umbral_exacto = umbral_exacto
        self.max_iter_refinamiento = max_iter_refinamiento
        self.max_restarts = max_restarts

    @abstractmethod
    def evaluar_asignacion(self, asignacion: tuple[int, ...]) -> tuple[float, np.ndarray]:
        """Calcula la perdida y la distribucion para una asignacion de grupos."""

    @abstractmethod
    def total_elementos(self) -> int:
        """Numero de elementos (nodos o vertices) que se van a particionar."""

    def canonicalizar(self, asignacion: tuple[int, ...]) -> tuple[int, ...]:
        mapa: dict[int, int] = {}
        siguiente = 0
        canon = []
        for grupo in asignacion:
            if grupo not in mapa:
                mapa[grupo] = siguiente
                siguiente += 1
            canon.append(mapa[grupo])
        return tuple(canon)

    def vecinos(self, asignacion: tuple[int, ...], k: int) -> list[tuple[int, ...]]:
        resultado: list[tuple[int, ...]] = []
        vistos: set[tuple[int, ...]] = set()
        for idx in range(len(asignacion)):
            for nuevo_grupo in range(k):
                if nuevo_grupo == asignacion[idx]:
                    continue
                nueva = list(asignacion)
                nueva[idx] = nuevo_grupo
                canon = self.canonicalizar(tuple(nueva))
                if len(set(canon)) < 2 or canon in vistos:
                    continue
                vistos.add(canon)
                resultado.append(canon)
        return resultado

    def refinar_local(self, inicio: ResultadoKParticion, k: int) -> ResultadoKParticion:
        actual = inicio
        mejor = inicio
        for _ in range(self.max_iter_refinamiento):
            candidatos = self.vecinos(actual.asignacion, k)
            if not candidatos:
                break
            mejor_vecino = actual
            for asig in candidatos:
                perdida, dist = self.evaluar_asignacion(asig)
                if perdida < mejor_vecino.perdida:
                    mejor_vecino = ResultadoKParticion(
                        perdida=perdida,
                        distribucion=dist,
                        asignacion=asig,
                    )
            if mejor_vecino.perdida + 1e-12 >= actual.perdida:
                break
            actual = mejor_vecino
            if actual.perdida < mejor.perdida:
                mejor = actual
        return mejor

    def buscar(self, k: int, semilla: int = 42) -> ResultadoKParticion:
        n = self.total_elementos()
        k_eff = min(k, n)
        if n <= self.umbral_exacto:
            return self._buscar_exacto(k_eff)
        return self._buscar_local(k_eff, semilla)

    def _buscar_exacto(self, k: int) -> ResultadoKParticion:
        n = self.total_elementos()
        asig_base = tuple([0] * (n - 1) + [1])
        perdida_base, dist_base = self.evaluar_asignacion(asig_base)
        mejor = ResultadoKParticion(
            perdida=perdida_base,
            distribucion=dist_base,
            asignacion=asig_base,
        )
        for asig in k_particiones_asignacion(n, k):
            perdida, dist = self.evaluar_asignacion(asig)
            if perdida < mejor.perdida:
                mejor = ResultadoKParticion(perdida=perdida, distribucion=dist, asignacion=asig)
        return mejor

    def _buscar_local(self, k: int, semilla: int) -> ResultadoKParticion:
        n = self.total_elementos()
        rng = np.random.default_rng(semilla)

        asig_inicial = self.canonicalizar(
            tuple(list(range(k)) + [int(rng.integers(0, k)) for _ in range(max(0, n - k))])
        )
        perdida_ini, dist_ini = self.evaluar_asignacion(asig_inicial)
        mejor = self.refinar_local(
            ResultadoKParticion(perdida=perdida_ini, distribucion=dist_ini, asignacion=asig_inicial),
            k,
        )

        for _ in range(self.max_restarts):
            perm = list(range(k)) + [int(rng.integers(0, k)) for _ in range(max(0, n - k))]
            rng.shuffle(perm)
            asig = self.canonicalizar(tuple(int(v) for v in perm))
            perdida, dist = self.evaluar_asignacion(asig)
            semilla_local = ResultadoKParticion(perdida=perdida, distribucion=dist, asignacion=asig)
            refinado = self.refinar_local(semilla_local, k)
            if refinado.perdida < mejor.perdida:
                mejor = refinado

        return mejor

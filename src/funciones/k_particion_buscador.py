from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Sequence

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


class BuscadorKRecocido(BuscadorKParticion):
    """Busqueda de k-particion usando recocido simulado (Simulated Annealing).

    A diferencia de la busqueda local codiciosa, acepta soluciones peores con
    probabilidad exp(-delta/T), donde T disminuye geometricamente. Esto permite
    escapar de minimos locales que atrapan a los buscadores voraces.

    Parametros de enfriamiento:
        temp_inicial: temperatura de arranque (controla la exploracion inicial).
        temp_final: criterio de parada por temperatura.
        factor_enfriamiento: multiplicador por ciclo (0 < factor < 1).
        pasos_por_temp: evaluaciones por nivel de temperatura.
    """

    def __init__(
        self,
        temp_inicial: float = 1.0,
        temp_final: float = 0.001,
        factor_enfriamiento: float = 0.92,
        pasos_por_temp: int = 30,
        n_cadenas: int = 3,
    ) -> None:
        super().__init__(umbral_exacto=6, max_iter_refinamiento=0, max_restarts=0)
        self.temp_inicial = temp_inicial
        self.temp_final = temp_final
        self.factor_enfriamiento = factor_enfriamiento
        self.pasos_por_temp = pasos_por_temp
        self.n_cadenas = n_cadenas

    def buscar(self, k: int, semilla: int = 42) -> ResultadoKParticion:
        n = self.total_elementos()
        k_eff = min(k, n)
        if n <= self.umbral_exacto:
            return self._buscar_exacto(k_eff)
        return self._multi_recocido(k_eff, semilla)

    def _recocido(self, k: int, semilla: int) -> ResultadoKParticion:
        n = self.total_elementos()
        rng = np.random.default_rng(semilla)

        asig_actual = self.canonicalizar(
            tuple(list(range(k)) + [int(rng.integers(0, k)) for _ in range(max(0, n - k))])
        )
        perm = list(asig_actual)
        rng.shuffle(perm)
        asig_actual = self.canonicalizar(tuple(perm))

        perdida_actual, dist_actual = self.evaluar_asignacion(asig_actual)
        mejor = ResultadoKParticion(
            perdida=perdida_actual,
            distribucion=dist_actual,
            asignacion=asig_actual,
        )

        temp = self.temp_inicial
        while temp > self.temp_final:
            for _ in range(self.pasos_por_temp):
                nueva = list(asig_actual)
                if n >= 2 and rng.random() < 0.5:
                    # Swap: intercambiar grupos de dos nodos distintos.
                    # Permite llegar en un paso a vecinos que el movimiento
                    # individual necesita dos pasos aceptados para alcanzar.
                    i = int(rng.integers(0, n))
                    j = int(rng.integers(0, n - 1))
                    if j >= i:
                        j += 1
                    nueva[i], nueva[j] = nueva[j], nueva[i]
                else:
                    idx = int(rng.integers(0, n))
                    nuevo_grupo = int(rng.integers(0, k))
                    nueva[idx] = nuevo_grupo

                asig_vecina = self.canonicalizar(tuple(nueva))

                if len(set(asig_vecina)) < 2:
                    continue

                perdida_vecina, dist_vecina = self.evaluar_asignacion(asig_vecina)
                delta = perdida_vecina - perdida_actual

                if delta < 0 or rng.random() < math.exp(-delta / temp):
                    asig_actual = asig_vecina
                    perdida_actual = perdida_vecina
                    dist_actual = dist_vecina

                    if perdida_actual < mejor.perdida:
                        mejor = ResultadoKParticion(
                            perdida=perdida_actual,
                            distribucion=dist_actual,
                            asignacion=asig_actual,
                        )

            temp *= self.factor_enfriamiento

        return mejor

    def buscar_con_semilla(
        self, k: int, semilla_asig: tuple[int, ...], semilla: int = 42
    ) -> ResultadoKParticion:
        """Combina refinamiento local de la semilla con SA para no quedar atrapado.

        Evalua la semilla, la refina con busqueda local codiciosa y compara
        contra una corrida SA independiente. Retorna el mejor de ambos.
        """
        perdida_s, dist_s = self.evaluar_asignacion(semilla_asig)
        inicio = ResultadoKParticion(perdida=perdida_s, distribucion=dist_s, asignacion=semilla_asig)
        refinado = self.refinar_local(inicio, k)
        resultado_sa = self._multi_recocido(k, semilla)
        return refinado if refinado.perdida <= resultado_sa.perdida else resultado_sa

    def _multi_recocido(self, k: int, semilla: int) -> ResultadoKParticion:
        """Corre n_cadenas corridas SA independientes y retorna la mejor.

        Cada cadena arranca desde un punto aleatorio distinto (semillas
        separadas por 1009 para evitar correlaciones). El cache compartido
        hace que las cadenas posteriores sean mas rapidas que la primera.
        """
        mejor = self._recocido(k, semilla)
        for i in range(1, self.n_cadenas):
            candidato = self._recocido(k, semilla + i * 1009)
            if candidato.perdida < mejor.perdida:
                mejor = candidato
        return mejor


class BuscadorKDP(BuscadorKRecocido):
    """DP de subconjuntos para inicializacion + recocido simulado para refinamiento.

    Fase 1 (n <= umbral_dp): precomputa el costo de biparticion de cada
    subconjunto de elementos y aplica DP de subconjuntos O(3^n * k) para
    encontrar la asignacion inicial de minimo costo estimado.

    Fase 2: refina la semilla DP con recocido simulado (heredado).

    Si se pasan costos_subconjuntos precalculados (e.g. costos_locales del
    hipercubo geometrico), la Fase 1 no requiere ninguna evaluacion adicional.

    Para n > umbral_dp cae directamente al recocido puro del padre.
    """

    def __init__(
        self,
        costos_subconjuntos: np.ndarray | None = None,
        umbral_dp: int = 12,
        temp_inicial: float = 1.0,
        temp_final: float = 0.001,
        factor_enfriamiento: float = 0.92,
        pasos_por_temp: int = 30,
    ) -> None:
        super().__init__(
            temp_inicial=temp_inicial,
            temp_final=temp_final,
            factor_enfriamiento=factor_enfriamiento,
            pasos_por_temp=pasos_por_temp,
        )
        self._costos_subconjuntos = costos_subconjuntos
        self._umbral_dp = umbral_dp

    def buscar(self, k: int, semilla: int = 42) -> ResultadoKParticion:
        n = self.total_elementos()
        k_eff = min(k, n)
        if n <= self.umbral_exacto:
            return self._buscar_exacto(k_eff)
        if n <= self._umbral_dp:
            return self._buscar_dp_sa(k_eff, semilla)
        return self._multi_recocido(k_eff, semilla)

    def _obtener_costos_sub(self, n: int) -> np.ndarray:
        total = 1 << n
        if self._costos_subconjuntos is not None and len(self._costos_subconjuntos) == total:
            return self._costos_subconjuntos.astype(np.float64)
        full_mask = total - 1
        costos = np.full(total, np.inf, dtype=np.float64)
        costos[0] = 0.0
        costos[full_mask] = 0.0
        for mask in range(1, full_mask):
            asig = tuple(0 if (mask >> i) & 1 else 1 for i in range(n))
            perdida, _ = self.evaluar_asignacion(asig)
            costos[mask] = perdida
        return costos

    def _buscar_dp_sa(self, k: int, semilla: int) -> ResultadoKParticion:
        n = self.total_elementos()
        total = 1 << n
        full_mask = total - 1

        costos_sub = self._obtener_costos_sub(n)
        INF = float("inf")

        # dp_costo[mask][j] = min costo estimado de j-particion de elementos en mask
        dp_costo = np.full((total, k + 1), INF, dtype=np.float64)
        dp_division = np.full((total, k + 1), -1, dtype=np.int32)

        for mask in range(total):
            dp_costo[mask, 0] = 0.0
            if mask > 0:
                dp_costo[mask, 1] = float(costos_sub[mask])
                dp_division[mask, 1] = mask

        for mask in range(1, total):
            submask = (mask - 1) & mask
            while submask > 0:
                costo_s = float(costos_sub[submask])
                if np.isfinite(costo_s):
                    resto = mask ^ submask
                    for j in range(2, k + 1):
                        nuevo = dp_costo[resto, j - 1] + costo_s
                        if nuevo < dp_costo[mask, j]:
                            dp_costo[mask, j] = nuevo
                            dp_division[mask, j] = submask
                submask = (submask - 1) & mask

        asig_dp = self._reconstruir_dp(n, k, full_mask, dp_division)
        asig_dp = self.canonicalizar(asig_dp)
        perdida_dp, dist_dp = self.evaluar_asignacion(asig_dp)
        mejor = ResultadoKParticion(perdida=perdida_dp, distribucion=dist_dp, asignacion=asig_dp)

        # Refinar la semilla DP con SA
        rng = np.random.default_rng(semilla)
        asig_actual = asig_dp
        perdida_actual = perdida_dp
        temp = self.temp_inicial
        while temp > self.temp_final:
            for _ in range(self.pasos_por_temp):
                nueva = list(asig_actual)
                if n >= 2 and rng.random() < 0.5:
                    i = int(rng.integers(0, n))
                    j = int(rng.integers(0, n - 1))
                    if j >= i:
                        j += 1
                    nueva[i], nueva[j] = nueva[j], nueva[i]
                else:
                    idx = int(rng.integers(0, n))
                    nuevo_grupo = int(rng.integers(0, k))
                    nueva[idx] = nuevo_grupo
                asig_v = self.canonicalizar(tuple(nueva))
                if len(set(asig_v)) < 2:
                    continue
                perdida_v, dist_v = self.evaluar_asignacion(asig_v)
                delta = perdida_v - perdida_actual
                if delta < 0 or rng.random() < math.exp(-delta / temp):
                    asig_actual = asig_v
                    perdida_actual = perdida_v
                    if perdida_actual < mejor.perdida:
                        mejor = ResultadoKParticion(
                            perdida=perdida_actual,
                            distribucion=dist_v,
                            asignacion=asig_actual,
                        )
            temp *= self.factor_enfriamiento

        for i in range(1, self.n_cadenas):
            candidato = self._recocido(k, semilla + i * 1009)
            if candidato.perdida < mejor.perdida:
                mejor = candidato
        return mejor

    def _reconstruir_dp(
        self,
        n: int,
        k: int,
        full_mask: int,
        dp_division: np.ndarray,
    ) -> tuple[int, ...]:
        asig = [0] * n
        mask = full_mask
        for grupo in range(k):
            j = k - grupo
            if mask <= 0 or j <= 0:
                break
            submask = int(dp_division[mask, j])
            if submask < 0:
                for i in range(n):
                    if (mask >> i) & 1:
                        asig[i] = grupo
                break
            for i in range(n):
                if (submask >> i) & 1:
                    asig[i] = grupo
            mask ^= submask
        return tuple(asig)

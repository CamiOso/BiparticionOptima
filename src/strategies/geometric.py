from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
import heapq
import os

import numpy as np

from src.constantes.models import GEOMETRIC_LABEL
from src.funciones.formato import fmt_biparticion, fmt_k_particion_asignacion
from src.funciones.iit import seleccionar_emd
from src.funciones.k_particion_buscador import BuscadorKDP, ResultadoKParticion
from src.funciones.particiones import biparticiones, k_particiones_asignacion
from src.modelos.base.aplicacion import aplicacion
from src.modelos.base.sia import SIA
from src.modelos.enumeraciones.geometric_mode import GeometricMode
from src.modelos.nucleo.sistema import Sistema
from src.modelos.nucleo.solucion import Solucion


@dataclass(frozen=True)
class _ResultadoParticion:
    perdida: float
    distribucion: np.ndarray
    subalcance: tuple[int, ...]
    submecanismo: tuple[int, ...]


class _BuscadorKGeometric(BuscadorKDP):
    """Buscador de k-particion para la estrategia geometrica.

    Evalua cada asignacion usando k_bipartir sobre el sistema espacial,
    donde los nodos son indices enteros (no pares temporales).

    Hereda de BuscadorKDP: usa costos del hipercubo geometrico precalculados
    para inicializacion DP y refina con recocido simulado.
    """

    def __init__(
        self,
        nodos: list[int],
        sistema: Sistema,
        dists: np.ndarray,
        distancia_metrica,
        cache: dict[tuple[int, ...], tuple[float, np.ndarray]],
        max_restarts: int = 20,
        costos_subconjuntos: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            costos_subconjuntos=costos_subconjuntos,
            umbral_dp=12,
            temp_inicial=1.0,
            temp_final=0.001,
            factor_enfriamiento=0.92,
            pasos_por_temp=30,
        )
        self.umbral_exacto = 5
        self.max_restarts = max_restarts
        self._nodos = nodos
        self._sistema = sistema
        self._dists = dists
        self._distancia_metrica = distancia_metrica
        self._cache = cache

    def total_elementos(self) -> int:
        return len(self._nodos)

    def evaluar_asignacion(self, asignacion: tuple[int, ...]) -> tuple[float, np.ndarray]:
        en_cache = self._cache.get(asignacion)
        if en_cache is not None:
            return en_cache
        sistema_partido = self._sistema.k_bipartir(self._nodos, asignacion)
        dist = _alinear_distribucion(sistema_partido.distribucion_marginal(), self._dists)
        perdida = float(self._distancia_metrica(self._dists, dist))
        self._cache[asignacion] = (perdida, dist)
        return perdida, dist



def _alinear_distribucion(distribucion: np.ndarray, referencia: np.ndarray) -> np.ndarray:
    if distribucion.size == referencia.size:
        return distribucion
    salida = np.zeros_like(referencia)
    salida[: distribucion.size] = distribucion
    return salida


class Geometric(SIA):
    """Estrategia geometrica sobre hipercubo para aproximar la MIP en O(n*2^n)."""

    def __init__(self, tpm: np.ndarray, mode: GeometricMode | str | None = None, config=None) -> None:
        super().__init__(tpm, config)
        self.distancia_metrica = seleccionar_emd(config)
        self.mode = self._resolver_modo(mode)
        self._beam_top_k = 12
        self._max_candidatos_costo_cero = 32
        self._max_seeds_refinamiento = 6
        self._max_iter_refinamiento = 24
        self._beam_top_k_adaptativo = 20
        self._max_iter_refinamiento_adaptativo = 40
        self._umbral_incertidumbre = 0.10
        self._random_restarts = 20
        self._umbral_restarts = 0.05
        self._usar_optimizacion_grandes = True
        self._umbral_nodos_optimizacion = 9
        self._usar_simetrias_hipercubo = True
        self._fraccion_muestreo_mascaras = 0.35
        self._min_muestras_mascaras = 128
        self._usar_paralelizacion_costos = True
        self._umbral_paralelizacion_mascaras = 96
        self._max_workers_costos = max(1, (os.cpu_count() or 2) - 1)
        self._cache_particiones: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            tuple[float, np.ndarray],
        ] = {}
        self._cache_k_particiones: dict[
            tuple[int, ...],
            tuple[float, np.ndarray],
        ] = {}

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
        k: int = 2,
    ) -> Solucion:
        if k < 2:
            raise ValueError(f"k debe ser >= 2, se recibio {k}.")

        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None

        self._cache_particiones.clear()
        self._cache_k_particiones.clear()
        _ = self._tpm_a_tensores_elementales()

        alcance_total = tuple(int(v) for v in self.sia_subsistema.indices_ncubos.tolist())
        mecanismo_total = tuple(int(v) for v in self.sia_subsistema.dims_ncubos.tolist())

        if not alcance_total and not mecanismo_total:
            return Solucion(
                estrategia=GEOMETRIC_LABEL,
                perdida=0.0,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=self.sia_dists_marginales.copy(),
                estado_inicial=estado_inicial,
                particion="NO-PARTITION",
            )

        if k > 2:
            nodos = sorted(set(alcance_total) | set(mecanismo_total))
            _, _, costos_locales, _ = self._precalcular_busqueda_geometrica(
                alcance_total, mecanismo_total
            )
            # Warm-start 1: dendrograma divisivo — jerarquia de cortes optimos.
            semilla_k = self._resolver_k_dendrograma(
                nodos, alcance_total, mecanismo_total, k
            )
            # Warm-start 2 (fallback): mascara de mejor biparticion del hipercubo.
            if semilla_k is None:
                semilla_k = self._semilla_desde_biparticion(
                    nodos, alcance_total, mecanismo_total, costos_locales
                )
            buscador = _BuscadorKGeometric(
                nodos=nodos,
                sistema=self.sia_subsistema,
                dists=self.sia_dists_marginales,
                distancia_metrica=self.distancia_metrica,
                cache=self._cache_k_particiones,
                max_restarts=self._random_restarts,
                costos_subconjuntos=costos_locales,
            )
            if semilla_k is not None:
                resultado_k = buscador.buscar_con_semilla(
                    k, semilla_k, semilla=aplicacion.semilla_numpy
                )
            else:
                resultado_k = buscador.buscar(k, semilla=aplicacion.semilla_numpy)
            return Solucion(
                estrategia=GEOMETRIC_LABEL,
                perdida=resultado_k.perdida,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=resultado_k.distribucion,
                estado_inicial=estado_inicial,
                particion=fmt_k_particion_asignacion(
                    nodos,
                    resultado_k.asignacion,
                    alcance_total,
                    mecanismo_total,
                ),
            )

        n_nodos = len(set(alcance_total) | set(mecanismo_total))
        if self.mode == GeometricMode.STRICT.value:
            mejor = self._resolver_geometrico_estricto(alcance_total, mecanismo_total)
        else:
            if n_nodos <= 5:
                mejor = self._resolver_exacto(alcance_total, mecanismo_total)
            else:
                mejor = self._resolver_geometrico_refinado(alcance_total, mecanismo_total)

        return Solucion(
            estrategia=GEOMETRIC_LABEL,
            perdida=mejor.perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=mejor.distribucion,
            estado_inicial=estado_inicial,
            particion=fmt_biparticion(
                mejor.subalcance,
                mejor.submecanismo,
                alcance_total,
                mecanismo_total,
            ),
        )

    def _resolver_modo(self, mode: GeometricMode | str | None) -> str:
        if isinstance(mode, GeometricMode):
            return mode.value
        if isinstance(mode, str):
            if mode not in {GeometricMode.STRICT.value, GeometricMode.REFINED.value}:
                raise ValueError(f"Modo geometrico invalido: {mode}")
            return mode
        return aplicacion.modo_geometrico

    def _tpm_a_tensores_elementales(self) -> tuple[np.ndarray, ...]:
        """Representa cada n-cubo del subsistema como tensor elemental."""
        assert self.sia_subsistema is not None
        return tuple(np.array(cubo.data, dtype=np.float32, copy=True) for cubo in self.sia_subsistema.ncubos)

    def _resolver_exacto(
        self,
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
    ) -> _ResultadoParticion:
        mejor = _ResultadoParticion(
            perdida=float("inf"),
            distribucion=self.sia_dists_marginales.copy(),
            subalcance=(),
            submecanismo=(),
        )
        assert self.sia_subsistema is not None

        for subalcance, submecanismo in biparticiones(
            self.sia_subsistema.indices_ncubos,
            self.sia_subsistema.dims_ncubos,
        ):
            perdida, distribucion = self._evaluar_particion(subalcance, submecanismo)
            if perdida < mejor.perdida:
                mejor = _ResultadoParticion(
                    perdida=perdida,
                    distribucion=distribucion,
                    subalcance=subalcance,
                    submecanismo=submecanismo,
                )
        return mejor

    def _resolver_geometrico_estricto(
        self,
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
    ) -> _ResultadoParticion:
        nodos, total_mascaras, costos_locales, candidatos = self._precalcular_busqueda_geometrica(
            alcance_total,
            mecanismo_total,
        )

        mejor_resultado = _ResultadoParticion(
            perdida=float("inf"),
            distribucion=self.sia_dists_marginales.copy(),
            subalcance=(),
            submecanismo=(),
        )

        for subalcance, submecanismo in candidatos:
            perdida, distribucion = self._evaluar_particion(subalcance, submecanismo)
            if perdida < mejor_resultado.perdida:
                mejor_resultado = _ResultadoParticion(
                    perdida=perdida,
                    distribucion=distribucion,
                    subalcance=subalcance,
                    submecanismo=submecanismo,
                )

        return mejor_resultado

    def _precalcular_busqueda_geometrica(
        self,
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
    ) -> tuple[list[int], int, np.ndarray, list[tuple[tuple[int, ...], tuple[int, ...]]]]:
        nodos = sorted(set(alcance_total) | set(mecanismo_total))
        n_nodos = len(nodos)
        total_mascaras = 1 << len(nodos)
        mascara_total = total_mascaras - 1
        optimizacion_grande = self._debe_optimizar_grandes(n_nodos, total_mascaras)

        costos = np.full(total_mascaras, np.inf, dtype=np.float64)
        costos_locales = np.full(total_mascaras, np.inf, dtype=np.float64)

        costos[0] = 0.0
        candidatos_costo_cero: set[int] = set()

        mascaras_evaluacion = self._seleccionar_mascaras_evaluacion(
            n_nodos=n_nodos,
            total_mascaras=total_mascaras,
            optimizacion_grande=optimizacion_grande,
        )
        resultados = self._evaluar_mascaras_locales(
            mascaras=mascaras_evaluacion,
            nodos=nodos,
            alcance_total=alcance_total,
            mecanismo_total=mecanismo_total,
        )

        for mascara, perdida_local in resultados.items():
            costos_locales[mascara] = perdida_local
            if optimizacion_grande and self._usar_simetrias_hipercubo:
                mascara_comp = mascara_total ^ mascara
                if 0 < mascara_comp < mascara_total and not np.isfinite(costos_locales[mascara_comp]):
                    costos_locales[mascara_comp] = perdida_local

        # candidatos_costo_cero via numpy
        _fin_lc = np.isfinite(costos_locales)
        _internal = np.zeros(total_mascaras, dtype=bool)
        if mascara_total > 1:
            _internal[1:mascara_total] = True
        candidatos_costo_cero.update(
            np.where(_fin_lc & (costos_locales <= 1e-12) & _internal)[0].tolist()
        )

        # DP vectorizado por nivel de popcount (orden topologico garantizado).
        # Semantica identica al bucle Python original: sin perdida de calidad.
        _gamma = np.float64(0.5)
        _all_idx = np.arange(total_mascaras, dtype=np.int64)
        _pop = np.zeros(total_mascaras, dtype=np.int32)
        for _b in range(n_nodos):
            _pop += ((_all_idx >> _b) & 1).astype(np.int32)

        for _level in range(1, n_nodos + 1):
            _lm = _all_idx[_pop == _level]
            _lc = costos_locales[_lm]
            _valid = np.isfinite(_lc)
            if not _valid.any():
                continue
            _lm_v = _lm[_valid]
            _lc_v = _lc[_valid]
            _min_pred = np.full(len(_lm_v), np.inf, dtype=np.float64)
            for _b in range(n_nodos):
                _bv = np.int64(1 << _b)
                _has = (_lm_v & _bv).astype(bool)
                if not _has.any():
                    continue
                _pred_costs = costos[(_lm_v[_has] ^ _bv).astype(np.intp)]
                np.minimum(_min_pred[_has], _pred_costs, out=_min_pred[_has])
            _candidates = _min_pred + _gamma * _lc_v
            np.minimum(costos[_lm_v.astype(np.intp)], _candidates, out=costos[_lm_v.astype(np.intp)])

        if not candidatos_costo_cero:
            internas_finitas = [
                mascara
                for mascara in range(1, total_mascaras - 1)
                if np.isfinite(costos[mascara])
            ]
            if internas_finitas:
                mejor_costo = float(min(costos[mascara] for mascara in internas_finitas))
                candidatos_costo_cero = {
                    mascara
                    for mascara in internas_finitas
                    if costos[mascara] <= mejor_costo + 1e-12
                }

        if not candidatos_costo_cero:
            internas_locales_finitas = [
                mascara
                for mascara in range(1, total_mascaras - 1)
                if np.isfinite(costos_locales[mascara])
            ]
            if internas_locales_finitas:
                candidatos_costo_cero = {
                    min(internas_locales_finitas, key=lambda mascara: float(costos_locales[mascara]))
                }
            else:
                candidatos_costo_cero = {1}

        candidatos_base = self._seleccionar_mascaras_base(
            costos=costos,
            costos_locales=costos_locales,
            candidatos_costo_cero=candidatos_costo_cero,
            total_mascaras=total_mascaras,
        )
        if optimizacion_grande and self._usar_simetrias_hipercubo:
            candidatos_base = self._incluir_complementos(candidatos_base, total_mascaras)

        candidatos = self._expandir_candidatos_vecindad(
            mascaras_base=candidatos_base,
            nodos=nodos,
            alcance_total=alcance_total,
            mecanismo_total=mecanismo_total,
            total_mascaras=total_mascaras,
        )
        return nodos, total_mascaras, costos_locales, candidatos

    def _resolver_geometrico_refinado(
        self,
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
    ) -> _ResultadoParticion:
        nodos, total_mascaras, costos_locales, candidatos = self._precalcular_busqueda_geometrica(
            alcance_total,
            mecanismo_total,
        )
        if len(nodos) >= 6:
            vistos = set(candidatos)
            for c in self._candidatos_fiedler(nodos, alcance_total, mecanismo_total):
                if c not in vistos:
                    candidatos.append(c)
                    vistos.add(c)
        mejor_resultado = _ResultadoParticion(
            perdida=float("inf"),
            distribucion=self.sia_dists_marginales.copy(),
            subalcance=(),
            submecanismo=(),
        )
        for subalcance, submecanismo in candidatos:
            perdida, distribucion = self._evaluar_particion(subalcance, submecanismo)
            if perdida < mejor_resultado.perdida:
                mejor_resultado = _ResultadoParticion(
                    perdida=perdida,
                    distribucion=distribucion,
                    subalcance=subalcance,
                    submecanismo=submecanismo,
                )
        ranking_inicial = self._ranking_desde_cache()

        ranking_inicial.sort(key=lambda item: item.perdida)
        semillas_refinar = ranking_inicial[: self._max_seeds_refinamiento]
        for semilla in semillas_refinar:
            refinado = self._refinar_local_desacoplado(
                semilla,
                alcance_total,
                mecanismo_total,
                max_iter=self._max_iter_refinamiento,
            )
            if refinado.perdida < mejor_resultado.perdida:
                mejor_resultado = refinado

        # Refinamiento adaptativo: se activa solo si hay alta incertidumbre.
        if self._debe_refinar_adaptativo(
            mejor_resultado=mejor_resultado,
            costos_locales=costos_locales,
            total_mascaras=total_mascaras,
            n_nodos=len(nodos),
        ):
            mejores_locales = sorted(
                range(1, total_mascaras - 1),
                key=lambda mascara: float(costos_locales[mascara]),
            )[: self._beam_top_k_adaptativo]
            candidatos_adaptativos = self._expandir_candidatos_adaptativos(
                mascaras_base=mejores_locales,
                nodos=nodos,
                alcance_total=alcance_total,
                mecanismo_total=mecanismo_total,
                total_mascaras=total_mascaras,
            )

            ranking_adaptativo: list[_ResultadoParticion] = []
            for subalcance, submecanismo in candidatos_adaptativos:
                perdida, distribucion = self._evaluar_particion(subalcance, submecanismo)
                ranking_adaptativo.append(
                    _ResultadoParticion(
                        perdida=perdida,
                        distribucion=distribucion,
                        subalcance=subalcance,
                        submecanismo=submecanismo,
                    )
                )

            ranking_adaptativo.sort(key=lambda item: item.perdida)
            for semilla in ranking_adaptativo[: self._max_seeds_refinamiento]:
                refinado = self._refinar_local_desacoplado(
                    semilla,
                    alcance_total,
                    mecanismo_total,
                    max_iter=self._max_iter_refinamiento_adaptativo,
                )
                if refinado.perdida < mejor_resultado.perdida:
                    mejor_resultado = refinado

        # Restarts deterministas para escapar minimos locales en sistemas grandes.
        if len(nodos) >= 6 and mejor_resultado.perdida > self._umbral_restarts:
            semillas = self._generar_semillas_aleatorias(
                total_mascaras=total_mascaras,
                cantidad=self._random_restarts,
            )
            for mascara in semillas:
                subalcance, submecanismo = self._particion_desde_mascara(
                    mascara,
                    nodos,
                    alcance_total,
                    mecanismo_total,
                )
                perdida, distribucion = self._evaluar_particion(subalcance, submecanismo)
                semilla = _ResultadoParticion(
                    perdida=perdida,
                    distribucion=distribucion,
                    subalcance=subalcance,
                    submecanismo=submecanismo,
                )
                refinado = self._refinar_local_desacoplado(
                    semilla,
                    alcance_total,
                    mecanismo_total,
                    max_iter=self._max_iter_refinamiento_adaptativo,
                )
                if refinado.perdida < mejor_resultado.perdida:
                    mejor_resultado = refinado

        return mejor_resultado

    def _ranking_desde_cache(self) -> list[_ResultadoParticion]:
        ranking: list[_ResultadoParticion] = []
        assert self.sia_subsistema is not None
        alcance_total = tuple(int(v) for v in self.sia_subsistema.indices_ncubos.tolist())
        mecanismo_total = tuple(int(v) for v in self.sia_subsistema.dims_ncubos.tolist())
        for (subalcance, submecanismo), (perdida, distribucion) in self._cache_particiones.items():
            if not subalcance and not submecanismo:
                continue
            if subalcance == alcance_total and submecanismo == mecanismo_total:
                continue
            ranking.append(
                _ResultadoParticion(
                    perdida=perdida,
                    distribucion=distribucion,
                    subalcance=subalcance,
                    submecanismo=submecanismo,
                )
            )
        return ranking

    def _generar_semillas_aleatorias(self, total_mascaras: int, cantidad: int) -> list[int]:
        if total_mascaras <= 2 or cantidad <= 0:
            return []
        rng = np.random.default_rng(total_mascaras)
        semillas = set()
        while len(semillas) < cantidad:
            mascara = int(rng.integers(1, total_mascaras - 1))
            semillas.add(mascara)
            if len(semillas) >= (total_mascaras - 2):
                break
        return sorted(semillas)

    def _refinar_local_desacoplado(
        self,
        inicio: _ResultadoParticion,
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
        max_iter: int,
    ) -> _ResultadoParticion:
        actual = inicio
        mejor_global = inicio

        for _ in range(max_iter):
            vecinos = self._vecinos_desacoplados(
                actual.subalcance,
                actual.submecanismo,
                alcance_total,
                mecanismo_total,
            )
            if not vecinos:
                break

            mejor_vecino = actual
            for subalcance, submecanismo in vecinos:
                perdida, distribucion = self._evaluar_particion(subalcance, submecanismo)
                if perdida < mejor_vecino.perdida:
                    mejor_vecino = _ResultadoParticion(
                        perdida=perdida,
                        distribucion=distribucion,
                        subalcance=subalcance,
                        submecanismo=submecanismo,
                    )

            if mejor_vecino.perdida + 1e-12 >= actual.perdida:
                break

            actual = mejor_vecino
            if actual.perdida < mejor_global.perdida:
                mejor_global = actual

        return mejor_global

    def _debe_refinar_adaptativo(
        self,
        mejor_resultado: _ResultadoParticion,
        costos_locales: np.ndarray,
        total_mascaras: int,
        n_nodos: int,
    ) -> bool:
        if n_nodos < 7:
            return False
        if total_mascaras <= 2:
            return False
        mejor_local = float(np.min(costos_locales[1: total_mascaras - 1]))
        brecha = max(0.0, mejor_resultado.perdida - mejor_local)
        return brecha >= self._umbral_incertidumbre

    def _expandir_candidatos_adaptativos(
        self,
        mascaras_base: list[int],
        nodos: list[int],
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
        total_mascaras: int,
    ) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        vistos: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        candidatos: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        def agregar(subalcance: tuple[int, ...], submecanismo: tuple[int, ...]) -> None:
            if not subalcance and not submecanismo:
                return
            if subalcance == alcance_total and submecanismo == mecanismo_total:
                return
            clave = (subalcance, submecanismo)
            if clave in vistos:
                return
            vistos.add(clave)
            candidatos.append(clave)

        for mascara in mascaras_base:
            base = self._particion_desde_mascara(
                mascara,
                nodos,
                alcance_total,
                mecanismo_total,
            )
            agregar(*base)

            mascara_comp = (total_mascaras - 1) ^ mascara
            if 0 < mascara_comp < (total_mascaras - 1):
                comp = self._particion_desde_mascara(
                    mascara_comp,
                    nodos,
                    alcance_total,
                    mecanismo_total,
                )
                agregar(*comp)

            # Vecindad de radio 1 y 2 (bit flips) para mejorar robustez ante outliers.
            for bit_i in range(len(nodos)):
                m1 = mascara ^ (1 << bit_i)
                if 0 < m1 < (total_mascaras - 1):
                    p1 = self._particion_desde_mascara(
                        m1,
                        nodos,
                        alcance_total,
                        mecanismo_total,
                    )
                    agregar(*p1)

                for bit_j in range(bit_i + 1, len(nodos)):
                    m2 = mascara ^ (1 << bit_i) ^ (1 << bit_j)
                    if 0 < m2 < (total_mascaras - 1):
                        p2 = self._particion_desde_mascara(
                            m2,
                            nodos,
                            alcance_total,
                            mecanismo_total,
                        )
                        agregar(*p2)

        return candidatos

    def _debe_optimizar_grandes(self, n_nodos: int, total_mascaras: int) -> bool:
        if not self._usar_optimizacion_grandes:
            return False
        if n_nodos < self._umbral_nodos_optimizacion:
            return False
        return total_mascaras >= (1 << self._umbral_nodos_optimizacion)

    def _seleccionar_mascaras_evaluacion(
        self,
        n_nodos: int,
        total_mascaras: int,
        optimizacion_grande: bool,
    ) -> list[int]:
        internas = list(range(1, total_mascaras - 1))
        if not optimizacion_grande:
            return internas

        candidatas = internas
        if self._usar_simetrias_hipercubo:
            mascara_total = total_mascaras - 1
            candidatas = [
                mascara
                for mascara in internas
                if mascara <= (mascara_total ^ mascara)
            ]

        return self._muestrear_mascaras(candidatas, n_nodos, total_mascaras)

    def _muestrear_mascaras(
        self,
        candidatas: list[int],
        n_nodos: int,
        total_mascaras: int,
    ) -> list[int]:
        if not candidatas:
            return [1]

        if len(candidatas) <= self._min_muestras_mascaras:
            return sorted(candidatas)

        objetivo = max(
            self._min_muestras_mascaras,
            int(len(candidatas) * self._fraccion_muestreo_mascaras),
        )
        objetivo = min(objetivo, len(candidatas))

        esenciales = {
            mascara
            for mascara in candidatas
            if mascara.bit_count() in {1, max(1, n_nodos - 1), max(1, n_nodos // 2)}
        }

        if len(esenciales) >= objetivo:
            return sorted(list(esenciales)[:objetivo])

        restantes = [mascara for mascara in candidatas if mascara not in esenciales]
        faltan = objetivo - len(esenciales)
        if faltan <= 0 or not restantes:
            return sorted(esenciales)

        rng = np.random.default_rng(aplicacion.semilla_numpy + total_mascaras + n_nodos)
        seleccion_idx = rng.choice(len(restantes), size=min(faltan, len(restantes)), replace=False)
        muestreadas = {restantes[int(idx)] for idx in seleccion_idx.tolist()}
        return sorted(esenciales | muestreadas)

    def _incluir_complementos(self, mascaras: list[int], total_mascaras: int) -> list[int]:
        if not mascaras:
            return [1]
        mascara_total = total_mascaras - 1
        salida: set[int] = set()
        for mascara in mascaras:
            if 0 < mascara < mascara_total:
                salida.add(mascara)
            mascara_comp = mascara_total ^ mascara
            if 0 < mascara_comp < mascara_total:
                salida.add(mascara_comp)
        return sorted(salida) or [1]

    def _evaluar_mascara_local(
        self,
        mascara: int,
        nodos: list[int],
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
    ) -> tuple[int, float]:
        subalcance, submecanismo = self._particion_desde_mascara(
            mascara,
            nodos,
            alcance_total,
            mecanismo_total,
        )
        perdida_local, _ = self._evaluar_particion(subalcance, submecanismo)
        return mascara, float(perdida_local)

    def _evaluar_mascaras_locales(
        self,
        mascaras: list[int],
        nodos: list[int],
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
    ) -> dict[int, float]:
        if not mascaras:
            return {}

        worker = partial(
            self._evaluar_mascara_local,
            nodos=nodos,
            alcance_total=alcance_total,
            mecanismo_total=mecanismo_total,
        )

        if (
            self._usar_paralelizacion_costos
            and self._max_workers_costos > 1
            and len(mascaras) >= self._umbral_paralelizacion_mascaras
        ):
            with ThreadPoolExecutor(max_workers=self._max_workers_costos) as executor:
                pares = list(executor.map(worker, mascaras))
        else:
            pares = [worker(mascara) for mascara in mascaras]

        return {mascara: perdida for mascara, perdida in pares}

    def _vecinos_desacoplados(
        self,
        subalcance: tuple[int, ...],
        submecanismo: tuple[int, ...],
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
    ) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        vecinos: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        vistos: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

        alcance_set = set(subalcance)
        mecanismo_set = set(submecanismo)

        def agregar(cand_alcance: tuple[int, ...], cand_mecanismo: tuple[int, ...]) -> None:
            if not cand_alcance and not cand_mecanismo:
                return
            if cand_alcance == alcance_total and cand_mecanismo == mecanismo_total:
                return
            clave = (cand_alcance, cand_mecanismo)
            if clave in vistos:
                return
            vistos.add(clave)
            vecinos.append(clave)

        for nodo in alcance_total:
            nuevo_set = set(alcance_set)
            if nodo in nuevo_set:
                nuevo_set.remove(nodo)
            else:
                nuevo_set.add(nodo)
            cand_alcance = tuple(v for v in alcance_total if v in nuevo_set)
            agregar(cand_alcance, submecanismo)

        for nodo in mecanismo_total:
            nuevo_set = set(mecanismo_set)
            if nodo in nuevo_set:
                nuevo_set.remove(nodo)
            else:
                nuevo_set.add(nodo)
            cand_mecanismo = tuple(v for v in mecanismo_total if v in nuevo_set)
            agregar(subalcance, cand_mecanismo)

        return vecinos

    def _seleccionar_mascaras_base(
        self,
        costos: np.ndarray,
        costos_locales: np.ndarray,
        candidatos_costo_cero: set[int],
        total_mascaras: int,
    ) -> list[int]:
        internas = [
            mascara
            for mascara in range(1, total_mascaras - 1)
            if np.isfinite(costos[mascara]) or np.isfinite(costos_locales[mascara])
        ]
        if not internas:
            return [1]

        top_costos = sorted(internas, key=lambda mascara: float(costos[mascara]))[: self._beam_top_k]
        top_locales = sorted(
            internas,
            key=lambda mascara: float(costos_locales[mascara]),
        )[: self._beam_top_k]

        costo_cero_ordenadas = sorted(
            candidatos_costo_cero,
            key=lambda mascara: float(costos_locales[mascara]),
        )[: self._max_candidatos_costo_cero]

        combinadas: list[int] = []
        for mascara in (costo_cero_ordenadas + top_costos + top_locales):
            if mascara not in combinadas:
                combinadas.append(mascara)
        if combinadas:
            return combinadas
        return [min(internas, key=lambda mascara: float(costos_locales[mascara]))]

    def _expandir_candidatos_vecindad(
        self,
        mascaras_base: list[int],
        nodos: list[int],
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
        total_mascaras: int,
    ) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        vistos: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        candidatos: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        def agregar(subalcance: tuple[int, ...], submecanismo: tuple[int, ...]) -> None:
            if not subalcance and not submecanismo:
                return
            if subalcance == alcance_total and submecanismo == mecanismo_total:
                return
            clave = (subalcance, submecanismo)
            if clave in vistos:
                return
            vistos.add(clave)
            candidatos.append(clave)

        for mascara in mascaras_base:
            subalcance, submecanismo = self._particion_desde_mascara(
                mascara,
                nodos,
                alcance_total,
                mecanismo_total,
            )
            agregar(subalcance, submecanismo)

            for bit in range(len(nodos)):
                mascara_flip = mascara ^ (1 << bit)
                if mascara_flip <= 0 or mascara_flip >= (total_mascaras - 1):
                    continue

                alcance_flip, _ = self._particion_desde_mascara(
                    mascara_flip,
                    nodos,
                    alcance_total,
                    mecanismo_total,
                )
                _, mecanismo_flip = self._particion_desde_mascara(
                    mascara_flip,
                    nodos,
                    alcance_total,
                    mecanismo_total,
                )

                agregar(alcance_flip, submecanismo)
                agregar(subalcance, mecanismo_flip)

        return candidatos

    def _particion_desde_mascara(
        self,
        mascara: int,
        nodos: list[int],
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        seleccionados = {nodos[idx] for idx in range(len(nodos)) if mascara & (1 << idx)}
        subalcance = tuple(nodo for nodo in alcance_total if nodo in seleccionados)
        submecanismo = tuple(nodo for nodo in mecanismo_total if nodo in seleccionados)
        return subalcance, submecanismo

    def _evaluar_particion(
        self,
        subalcance: tuple[int, ...],
        submecanismo: tuple[int, ...],
    ) -> tuple[float, np.ndarray]:
        clave = (subalcance, submecanismo)
        en_cache = self._cache_particiones.get(clave)
        if en_cache is not None:
            return en_cache

        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None

        sistema_partido = self.sia_subsistema.bipartir(
            np.array(subalcance, dtype=np.int8),
            np.array(submecanismo, dtype=np.int8),
        )
        distribucion = sistema_partido.distribucion_marginal()
        distribucion = _alinear_distribucion(distribucion, self.sia_dists_marginales)

        perdida = float(self.distancia_metrica(self.sia_dists_marginales, distribucion))
        resultado = (perdida, distribucion)
        self._cache_particiones[clave] = resultado
        return resultado

    # ------------------------------------------------------------------
    # Candidatos espectrales (Fiedler) para sistemas n >= 6
    # ------------------------------------------------------------------

    def _conductancias_geometrica(self, nodos: list[int]) -> np.ndarray:
        """W[i][j] = sensibilidad del nodo i al estado del nodo j (diferencias finitas)."""
        assert self.sia_subsistema is not None
        n = len(nodos)
        idx = {v: i for i, v in enumerate(nodos)}
        W = np.zeros((n, n), dtype=np.float64)
        for cubo in self.sia_subsistema.ncubos:
            i_orig = int(cubo.indice)
            if i_orig not in idx:
                continue
            ii = idx[i_orig]
            dims = cubo.dims.tolist()
            data = cubo.data
            nd = len(dims)
            for pos, dim_j in enumerate(dims):
                j_orig = int(dim_j)
                if j_orig not in idx:
                    continue
                jj = idx[j_orig]
                otras = [d for d in range(nd) if d != pos]
                otras_sizes = [data.shape[d] for d in otras]
                n_otras = max(1, int(np.prod(otras_sizes)) if otras_sizes else 1)
                total = 0.0
                for estado_idx in range(n_otras):
                    idx_otras: list[int] = []
                    temp = estado_idx
                    for s in reversed(otras_sizes):
                        idx_otras.append(temp % s)
                        temp //= s
                    idx_otras = list(reversed(idx_otras))
                    idx_0 = [0] * nd
                    idx_1 = [0] * nd
                    idx_0[pos] = 0
                    idx_1[pos] = 1
                    for k_ot, d_ot in enumerate(otras):
                        idx_0[d_ot] = idx_otras[k_ot]
                        idx_1[d_ot] = idx_otras[k_ot]
                    total += abs(float(data[tuple(idx_0)]) - float(data[tuple(idx_1)]))
                W[ii][jj] = total / n_otras
        return W

    def _candidatos_fiedler(
        self,
        nodos: list[int],
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
    ) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """Particiones derivadas del vector de Fiedler del Laplaciano simetrizado."""
        n = len(nodos)
        if n < 3:
            return []
        try:
            W = self._conductancias_geometrica(nodos)
            W_sym = (W + W.T) / 2.0
            grado = W_sym.sum(axis=1)
            if grado.max() < 1e-12:
                return []
            L = np.diag(grado) - W_sym
            vals, vecs = np.linalg.eigh(L)
            fiedler = vecs[:, 1]
        except Exception:
            return []

        candidatos: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        vistos: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

        umbrales = [np.percentile(fiedler, p) for p in [0, 25, 50, 75]]
        for umbral in umbrales:
            seleccionados = {nodos[i] for i in range(n) if fiedler[i] >= umbral}
            if not seleccionados or seleccionados == set(nodos):
                continue
            subalcance = tuple(v for v in alcance_total if v in seleccionados)
            submecanismo = tuple(v for v in mecanismo_total if v in seleccionados)
            if not subalcance and not submecanismo:
                comp = set(nodos) - seleccionados
                subalcance = tuple(v for v in alcance_total if v in comp)
                submecanismo = tuple(v for v in mecanismo_total if v in comp)
            if not subalcance and not submecanismo:
                continue
            clave = (subalcance, submecanismo)
            if clave not in vistos:
                vistos.add(clave)
                candidatos.append(clave)
        return candidatos

    # ------------------------------------------------------------------
    # Dendrograma divisivo para k-particiones
    # ------------------------------------------------------------------

    def _bipartir_componente(
        self,
        comp_nodos: list[int],
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
    ) -> tuple[frozenset, frozenset, float] | None:
        """Mejor biparticion dentro de un componente del dendrograma.

        Enumera todas las biparticiones propias del subconjunto comp_nodos
        y evalua cada una con _evaluar_particion (en el contexto del sistema
        completo). Retorna (izq, der, perdida) o None si no es divisible.
        """
        n = len(comp_nodos)
        if n <= 1:
            return None

        comp_set = frozenset(comp_nodos)
        mejor_perdida = float("inf")
        mejor_izq: frozenset = comp_set

        # Para componentes grandes muestrear solo los bits dentro del componente.
        total_mascaras = 1 << n
        for mask in range(1, total_mascaras - 1):
            nodos_izq = frozenset(comp_nodos[i] for i in range(n) if mask & (1 << i))
            nodos_der = comp_set - nodos_izq
            if not nodos_izq or not nodos_der:
                continue
            subalcance = tuple(v for v in alcance_total if v in nodos_izq)
            submecanismo = tuple(v for v in mecanismo_total if v in nodos_izq)
            if not subalcance and not submecanismo:
                # Intentar con el otro lado
                subalcance = tuple(v for v in alcance_total if v in nodos_der)
                submecanismo = tuple(v for v in mecanismo_total if v in nodos_der)
                if not subalcance and not submecanismo:
                    continue
                nodos_izq, nodos_der = nodos_der, nodos_izq
            perdida, _ = self._evaluar_particion(subalcance, submecanismo)
            if perdida < mejor_perdida:
                mejor_perdida = perdida
                mejor_izq = nodos_izq

        nodos_der = comp_set - mejor_izq
        if not nodos_der:
            return None
        return mejor_izq, nodos_der, mejor_perdida

    def _resolver_k_dendrograma(
        self,
        nodos: list[int],
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
        k: int,
    ) -> tuple[int, ...] | None:
        """K-particion via dendrograma de cortes divisivos minimos.

        En cada paso se divide el componente cuya mejor biparticion tiene
        el menor EMD (division mas natural). La k-particion es la asignacion
        de los k componentes hoja resultantes.
        """
        if len(nodos) < 2:
            return None

        comp_raiz = frozenset(nodos)
        split = self._bipartir_componente(list(comp_raiz), alcance_total, mecanismo_total)
        if split is None:
            return None

        # Heap: (perdida_split, id_unico, frozenset_componente)
        # Los splits precalculados se guardan en un dict indexado por id.
        splits_info: dict[int, tuple[frozenset, frozenset, float]] = {}
        id_cnt = 0
        splits_info[id_cnt] = split
        heap: list[tuple[float, int]] = []
        heapq.heappush(heap, (split[2], id_cnt))
        id_cnt += 1

        hojas: set[frozenset] = {comp_raiz}
        hoja_a_split_id: dict[frozenset, int] = {comp_raiz: 0}

        while len(hojas) < min(k, len(nodos)):
            if not heap:
                break

            _, eid = heapq.heappop(heap)
            if eid not in splits_info:
                continue

            izq, der, _ = splits_info.pop(eid)

            # Identificar el componente padre (el que contiene izq U der)
            padre = izq | der
            if padre not in hojas:
                continue  # ya fue dividido por otra rama del heap

            hojas.discard(padre)
            hojas.add(izq)
            hojas.add(der)

            for hijo in (izq, der):
                if len(hijo) > 1:
                    s = self._bipartir_componente(list(hijo), alcance_total, mecanismo_total)
                    if s is not None:
                        splits_info[id_cnt] = s
                        heapq.heappush(heap, (s[2], id_cnt))
                        id_cnt += 1

        # Construir asignacion
        nodo_a_grupo: dict[int, int] = {}
        for grupo_idx, hoja in enumerate(hojas):
            for nodo in hoja:
                nodo_a_grupo[nodo] = grupo_idx

        asig = tuple(nodo_a_grupo.get(n, 0) for n in nodos)
        return self.canonicalizar(asig) if len(set(asig)) >= 2 else None

    def _semilla_desde_biparticion(
        self,
        nodos: list[int],
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
        costos_locales: np.ndarray,
    ) -> tuple[int, ...] | None:
        """Convierte la mejor biparticion del hipercubo en una asignacion k-grupos inicial.

        Toma la mascara con menor costo local como grupo 0 y el complemento como grupo 1.
        Sirve de warm-start para la busqueda k>2.
        """
        total = len(costos_locales)
        full_mask = total - 1
        internas_finitas = [
            m for m in range(1, full_mask) if np.isfinite(costos_locales[m])
        ]
        if not internas_finitas:
            return None
        mejor_mascara = min(internas_finitas, key=lambda m: float(costos_locales[m]))
        n = len(nodos)
        asig = tuple(0 if (mejor_mascara >> i) & 1 else 1 for i in range(n))
        return self.canonicalizar(asig) if hasattr(self, "canonicalizar") else asig

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


# Alias en espanol para conservar consistencia del proyecto.
Geometrica = Geometric

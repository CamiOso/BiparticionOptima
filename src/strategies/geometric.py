from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
import os

import numpy as np

from src.constantes.models import GEOMETRIC_LABEL
from src.funciones.formato import fmt_biparticion
from src.funciones.iit import seleccionar_emd
from src.funciones.particiones import biparticiones
from src.modelos.base.aplicacion import aplicacion
from src.modelos.base.sia import SIA
from src.modelos.enumeraciones.geometric_mode import GeometricMode
from src.modelos.nucleo.solucion import Solucion


@dataclass(frozen=True)
class _ResultadoParticion:
    perdida: float
    distribucion: np.ndarray
    subalcance: tuple[int, ...]
    submecanismo: tuple[int, ...]


class Geometric(SIA):
    """Estrategia geometrica sobre hipercubo para aproximar la MIP en O(n*2^n)."""

    def __init__(self, tpm: np.ndarray, mode: GeometricMode | str | None = None) -> None:
        super().__init__(tpm)
        self.distancia_metrica = seleccionar_emd()
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

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ) -> Solucion:
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None

        self._cache_particiones.clear()
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

        for mascara in range(1, total_mascaras):
            perdida_local = float(costos_locales[mascara])
            if not np.isfinite(perdida_local):
                continue

            if perdida_local <= 1e-12 and mascara != mascara_total:
                candidatos_costo_cero.add(mascara)

            bits = mascara
            while bits:
                lsb = bits & -bits
                anterior = mascara ^ lsb
                if not np.isfinite(costos[anterior]):
                    bits ^= lsb
                    continue
                gamma = 2.0 ** (-1)
                costo = costos[anterior] + gamma * perdida_local
                if costo < costos[mascara]:
                    costos[mascara] = costo
                bits ^= lsb

        if not candidatos_costo_cero:
            # Si no hay costo cero, usamos las mascaras con mejor costo recursivo.
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
        if len(nodos) >= 8 and mejor_resultado.perdida > self._umbral_restarts:
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
            # mascara base y su complemento.
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
        distribucion = self._alinear_distribucion(distribucion, self.sia_dists_marginales)

        perdida = float(self.distancia_metrica(self.sia_dists_marginales, distribucion))
        resultado = (perdida, distribucion)
        self._cache_particiones[clave] = resultado
        return resultado

    def _alinear_distribucion(
        self,
        distribucion: np.ndarray,
        referencia: np.ndarray,
    ) -> np.ndarray:
        if distribucion.size == referencia.size:
            return distribucion
        salida = np.zeros_like(referencia)
        salida[: distribucion.size] = distribucion
        return salida


# Alias en espanol para conservar consistencia del proyecto.
Geometrica = Geometric

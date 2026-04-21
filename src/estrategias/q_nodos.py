from dataclasses import dataclass

import numpy as np

from src.funciones.formato import fmt_biparticion_q, fmt_k_particion_q
from src.funciones.iit import seleccionar_emd
from src.funciones.particiones import k_particiones_asignacion
from src.modelos.base.aplicacion import aplicacion
from src.modelos.base.sia import SIA
from src.modelos.nucleo.solucion import Solucion


@dataclass(frozen=True)
class _ResultadoParticionQK:
    perdida: float
    distribucion: np.ndarray
    asignacion: tuple[int, ...]
    vertices: list[tuple[int, int]]


class QNodos(SIA):
    """Implementacion submodular de QNodos basada en deltas y omegas."""

    def __init__(self, tpm: np.ndarray) -> None:
        super().__init__(tpm)
        self.distancia_metrica = seleccionar_emd()
        self.memoria_delta: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[float, np.ndarray]] = {}
        self.memoria_grupo_candidato: dict[tuple[tuple[int, int], ...], tuple[float, np.ndarray | None]] = {}
        self.clave_submodular: list[list[int]] = [[], []]
        self.vertices: set[tuple[int, int]] = set()
        self._cache_k_particiones: dict[tuple[int, ...], tuple[float, np.ndarray]] = {}
        self._max_iter_refinamiento_k = 24
        self._random_restarts_k = 16

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

        assert self.sia_dists_marginales is not None
        distribucion_subsistema = self.sia_dists_marginales

        assert self.sia_subsistema is not None
        self.memoria_delta.clear()
        self.memoria_grupo_candidato.clear()

        futuro = [(1, int(indice)) for indice in self.sia_subsistema.indices_ncubos.tolist()]
        presente = [(0, int(indice)) for indice in self.sia_subsistema.dims_ncubos.tolist()]
        vertices = list(presente + futuro)
        self.vertices = set(vertices)

        if k > 2:
            self._cache_k_particiones.clear()
            mejor_k = self._resolver_k_particiones(vertices, k)
            grupos = self._grupos_desde_asignacion(mejor_k.asignacion, mejor_k.vertices)
            return Solucion(
                estrategia="QNodos",
                perdida=float(mejor_k.perdida),
                distribucion_subsistema=distribucion_subsistema,
                distribucion_particion=mejor_k.distribucion,
                estado_inicial=estado_inicial,
                particion=fmt_k_particion_q(grupos),
            )

        clave_mip = self.algoritmo_q(vertices)
        perdida_mip, distribucion_particion = self.memoria_grupo_candidato[clave_mip]
        assert distribucion_particion is not None
        biparticion = fmt_biparticion_q(
            list(clave_mip),
            self.nodos_complemento(list(clave_mip)),
        )

        return Solucion(
            estrategia="QNodos",
            perdida=float(perdida_mip),
            distribucion_subsistema=distribucion_subsistema,
            distribucion_particion=distribucion_particion,
            estado_inicial=estado_inicial,
            particion=biparticion,
        )

    def algoritmo_q(
        self,
        vertices: list[tuple[int, int] | list[tuple[int, int]]],
    ) -> tuple[tuple[int, int], ...]:
        for _ in range(len(vertices) - 1):
            omegas_ciclo = [vertices[0]]
            deltas_ciclo = vertices[1:]

            emd_particion_candidata = np.inf
            dist_particion_candidata: np.ndarray | None = None

            for _ in range(max(0, len(deltas_ciclo) - 1)):
                emd_local = np.inf
                indice_mip = 0

                for indice_delta, delta in enumerate(deltas_ciclo):
                    emd_union, emd_delta, dist_marginal_delta = self.funcion_submodular(
                        delta,
                        omegas_ciclo,
                    )

                    emd_iteracion = emd_union - emd_delta
                    if emd_iteracion < emd_local:
                        if emd_delta == 0.0:
                            clave = self._normalizar_grupo(delta)
                            self.memoria_grupo_candidato[clave] = (
                                emd_delta,
                                dist_marginal_delta,
                            )
                            return clave

                        emd_local = emd_iteracion
                        indice_mip = indice_delta
                        emd_particion_candidata = emd_delta
                        dist_particion_candidata = dist_marginal_delta

                omegas_ciclo.append(deltas_ciclo[indice_mip])
                deltas_ciclo.pop(indice_mip)

            if deltas_ciclo:
                clave_final = self._normalizar_grupo(deltas_ciclo[-1])
                self.memoria_grupo_candidato[clave_final] = (
                    float(emd_particion_candidata),
                    dist_particion_candidata,
                )

                ultimo_omega = omegas_ciclo.pop()
                nuevo_grupo = self._desplegar_nodos(ultimo_omega) + self._desplegar_nodos(
                    deltas_ciclo[-1]
                )
                omegas_ciclo.append(nuevo_grupo)
                vertices = omegas_ciclo

        return min(
            self.memoria_grupo_candidato,
            key=lambda clave: self.memoria_grupo_candidato[clave][0],
        )

    def funcion_submodular(
        self,
        delta: tuple[int, int] | list[tuple[int, int]],
        omegas: list[tuple[int, int] | list[tuple[int, int]]],
    ) -> tuple[float, float, np.ndarray]:
        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None

        self.clave_submodular = [[], []]
        mecanismo_delta, alcance_delta = self.definir_clave(delta)
        clave_delta = (tuple(mecanismo_delta), tuple(alcance_delta))

        if clave_delta not in self.memoria_delta:
            particion_delta = self.sia_subsistema.bipartir(
                np.array(alcance_delta, dtype=np.int8),
                np.array(mecanismo_delta, dtype=np.int8),
            )
            vector_delta = self._alinear_distribucion(
                particion_delta.distribucion_marginal(),
                self.sia_dists_marginales,
            )
            emd_delta = float(self.distancia_metrica(vector_delta, self.sia_dists_marginales))
            self.memoria_delta[clave_delta] = (emd_delta, vector_delta)
        else:
            emd_delta, vector_delta = self.memoria_delta[clave_delta]

        for omega in omegas:
            self.definir_clave(omega)

        mecanismos_union, alcances_union = self.clave_submodular[0], self.clave_submodular[1]
        particion_union = self.sia_subsistema.bipartir(
            np.array(alcances_union, dtype=np.int8),
            np.array(mecanismos_union, dtype=np.int8),
        )
        vector_union = self._alinear_distribucion(
            particion_union.distribucion_marginal(),
            self.sia_dists_marginales,
        )
        emd_union = float(self.distancia_metrica(vector_union, self.sia_dists_marginales))

        return emd_union, emd_delta, vector_delta

    def definir_clave(
        self,
        conjunto: tuple[int, int] | list[tuple[int, int]],
    ) -> tuple[list[int], list[int]]:
        for tiempo, indice in self._desplegar_nodos(conjunto):
            self.clave_submodular[tiempo].append(indice)
        self.clave_submodular[0] = sorted(set(self.clave_submodular[0]))
        self.clave_submodular[1] = sorted(set(self.clave_submodular[1]))
        return self.clave_submodular[0], self.clave_submodular[1]

    def nodos_complemento(self, nodos: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return sorted(list(self.vertices - set(nodos)), key=lambda v: (v[0], v[1]))

    def _normalizar_grupo(
        self,
        conjunto: tuple[int, int] | list[tuple[int, int]],
    ) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(self._desplegar_nodos(conjunto), key=lambda v: (v[0], v[1])))

    def _desplegar_nodos(
        self,
        conjunto: tuple[int, int] | list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        if isinstance(conjunto, tuple) and len(conjunto) == 2 and all(
            isinstance(v, int) for v in conjunto
        ):
            return [conjunto]
        return [(int(t), int(i)) for t, i in conjunto]

    def _alinear_distribucion(
        self,
        distribucion: np.ndarray,
        referencia: np.ndarray,
    ) -> np.ndarray:
        if distribucion.size == referencia.size:
            return distribucion
        distribucion_alineada = np.zeros_like(referencia)
        distribucion_alineada[: distribucion.size] = distribucion
        return distribucion_alineada

    @staticmethod
    def _canonicalizar_asignacion(asignacion: tuple[int, ...]) -> tuple[int, ...]:
        mapa: dict[int, int] = {}
        siguiente = 0
        resultado = []
        for grupo in asignacion:
            if grupo not in mapa:
                mapa[grupo] = siguiente
                siguiente += 1
            resultado.append(mapa[grupo])
        return tuple(resultado)

    def _grupos_desde_asignacion(
        self,
        asignacion: tuple[int, ...],
        vertices: list[tuple[int, int]],
    ) -> list[list[tuple[int, int]]]:
        if not asignacion:
            return []
        total_grupos = max(asignacion) + 1
        grupos: list[list[tuple[int, int]]] = [[] for _ in range(total_grupos)]
        for indice, grupo in enumerate(asignacion):
            grupos[grupo].append(vertices[indice])
        return grupos

    def _evaluar_k_particion(
        self,
        asignacion: tuple[int, ...],
        vertices: list[tuple[int, int]],
    ) -> tuple[float, np.ndarray]:
        en_cache = self._cache_k_particiones.get(asignacion)
        if en_cache is not None:
            return en_cache

        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None

        grupos = self._grupos_desde_asignacion(asignacion, vertices)
        grupos_mecanismo = [
            tuple(indice for tiempo, indice in grupo if tiempo == 0)
            for grupo in grupos
        ]
        grupos_alcance = [
            tuple(indice for tiempo, indice in grupo if tiempo == 1)
            for grupo in grupos
        ]

        sistema_partido = self.sia_subsistema.k_bipartir_temporal(
            grupos_mecanismo,
            grupos_alcance,
        )
        distribucion = self._alinear_distribucion(
            sistema_partido.distribucion_marginal(),
            self.sia_dists_marginales,
        )
        perdida = float(self.distancia_metrica(distribucion, self.sia_dists_marginales))
        resultado = (perdida, distribucion)
        self._cache_k_particiones[asignacion] = resultado
        return resultado

    def _resolver_k_particiones(
        self,
        vertices: list[tuple[int, int]],
        k: int,
    ) -> _ResultadoParticionQK:
        if len(vertices) <= 8:
            return self._resolver_k_exacto(vertices, k)
        return self._resolver_k_local(vertices, k)

    def _resolver_k_exacto(
        self,
        vertices: list[tuple[int, int]],
        k: int,
    ) -> _ResultadoParticionQK:
        total_vertices = len(vertices)
        asignacion_fallback = tuple([0] * (total_vertices - 1) + [1])
        perdida_fallback, distribucion_fallback = self._evaluar_k_particion(
            asignacion_fallback,
            vertices,
        )
        mejor = _ResultadoParticionQK(
            perdida=perdida_fallback,
            distribucion=distribucion_fallback,
            asignacion=asignacion_fallback,
            vertices=vertices,
        )

        for asignacion in k_particiones_asignacion(total_vertices, min(k, total_vertices)):
            perdida, distribucion = self._evaluar_k_particion(asignacion, vertices)
            if perdida < mejor.perdida:
                mejor = _ResultadoParticionQK(
                    perdida=perdida,
                    distribucion=distribucion,
                    asignacion=asignacion,
                    vertices=vertices,
                )
        return mejor

    def _vecinos_k(
        self,
        asignacion: tuple[int, ...],
        k: int,
    ) -> list[tuple[int, ...]]:
        vecinos: list[tuple[int, ...]] = []
        vistos: set[tuple[int, ...]] = set()
        for indice in range(len(asignacion)):
            for nuevo_grupo in range(k):
                if nuevo_grupo == asignacion[indice]:
                    continue
                nueva = list(asignacion)
                nueva[indice] = nuevo_grupo
                canon = self._canonicalizar_asignacion(tuple(nueva))
                if len(set(canon)) < 2:
                    continue
                if canon in vistos:
                    continue
                vistos.add(canon)
                vecinos.append(canon)
        return vecinos

    def _refinar_k_local(
        self,
        inicio: _ResultadoParticionQK,
        vertices: list[tuple[int, int]],
        k: int,
    ) -> _ResultadoParticionQK:
        actual = inicio
        mejor_global = inicio
        for _ in range(self._max_iter_refinamiento_k):
            vecinos = self._vecinos_k(actual.asignacion, k)
            if not vecinos:
                break
            mejor_vecino = actual
            for vecino in vecinos:
                perdida, distribucion = self._evaluar_k_particion(vecino, vertices)
                if perdida < mejor_vecino.perdida:
                    mejor_vecino = _ResultadoParticionQK(
                        perdida=perdida,
                        distribucion=distribucion,
                        asignacion=vecino,
                        vertices=vertices,
                    )
            if mejor_vecino.perdida + 1e-12 >= actual.perdida:
                break
            actual = mejor_vecino
            if actual.perdida < mejor_global.perdida:
                mejor_global = actual
        return mejor_global

    def _resolver_k_local(
        self,
        vertices: list[tuple[int, int]],
        k: int,
    ) -> _ResultadoParticionQK:
        total_vertices = len(vertices)
        k_eff = min(k, total_vertices)
        rng = np.random.default_rng(aplicacion.semilla_numpy + total_vertices)

        asignacion_inicial = self._canonicalizar_asignacion(
            tuple(list(range(k_eff)) + [int(rng.integers(0, k_eff)) for _ in range(max(0, total_vertices - k_eff))])
        )
        perdida_inicial, distribucion_inicial = self._evaluar_k_particion(
            asignacion_inicial,
            vertices,
        )
        mejor_global = _ResultadoParticionQK(
            perdida=perdida_inicial,
            distribucion=distribucion_inicial,
            asignacion=asignacion_inicial,
            vertices=vertices,
        )
        mejor_global = self._refinar_k_local(mejor_global, vertices, k_eff)

        for _ in range(self._random_restarts_k):
            permutacion = list(range(k_eff)) + [
                int(rng.integers(0, k_eff)) for _ in range(max(0, total_vertices - k_eff))
            ]
            rng.shuffle(permutacion)
            asignacion = self._canonicalizar_asignacion(tuple(permutacion))
            perdida, distribucion = self._evaluar_k_particion(asignacion, vertices)
            semilla = _ResultadoParticionQK(
                perdida=perdida,
                distribucion=distribucion,
                asignacion=asignacion,
                vertices=vertices,
            )
            refinado = self._refinar_k_local(semilla, vertices, k_eff)
            if refinado.perdida < mejor_global.perdida:
                mejor_global = refinado

        return mejor_global


# Alias retrocompatible.
QNodes = QNodos

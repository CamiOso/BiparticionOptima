from dataclasses import dataclass

import numpy as np

from src.funciones.formato import fmt_biparticion_q, fmt_k_particion_q
from src.funciones.iit import seleccionar_emd
from src.funciones.k_particion_buscador import BuscadorKRecocido, ResultadoKParticion
from src.modelos.base.aplicacion import aplicacion
from src.modelos.base.sia import SIA
from src.modelos.nucleo.sistema import Sistema
from src.modelos.nucleo.solucion import Solucion


def _alinear_distribucion(distribucion: np.ndarray, referencia: np.ndarray) -> np.ndarray:
    if distribucion.size == referencia.size:
        return distribucion
    distribucion_alineada = np.zeros_like(referencia)
    distribucion_alineada[: distribucion.size] = distribucion
    return distribucion_alineada


def _grupos_desde_asignacion(
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


class _BuscadorKQNodos(BuscadorKRecocido):
    """Buscador de k-particion para la estrategia QNodos.

    Evalua cada asignacion usando k_bipartir_temporal, que trabaja con pares
    (tiempo, indice) para separar el mecanismo (t=0) del alcance (t=1).

    Hereda de BuscadorKRecocido para escapar minimos locales con SA.
    """

    def __init__(
        self,
        vertices: list[tuple[int, int]],
        sistema: Sistema,
        dists: np.ndarray,
        distancia_metrica,
        cache: dict[tuple[int, ...], tuple[float, np.ndarray]],
    ) -> None:
        super().__init__(
            temp_inicial=1.0,
            temp_final=0.001,
            factor_enfriamiento=0.92,
            pasos_por_temp=30,
        )
        self.umbral_exacto = 8
        self._vertices = vertices
        self._sistema = sistema
        self._dists = dists
        self._distancia_metrica = distancia_metrica
        self._cache = cache

    def total_elementos(self) -> int:
        return len(self._vertices)

    def evaluar_asignacion(self, asignacion: tuple[int, ...]) -> tuple[float, np.ndarray]:
        en_cache = self._cache.get(asignacion)
        if en_cache is not None:
            return en_cache

        grupos = _grupos_desde_asignacion(asignacion, self._vertices)
        grupos_mecanismo = [
            tuple(indice for tiempo, indice in grupo if tiempo == 0)
            for grupo in grupos
        ]
        grupos_alcance = [
            tuple(indice for tiempo, indice in grupo if tiempo == 1)
            for grupo in grupos
        ]

        sistema_partido = self._sistema.k_bipartir_temporal(grupos_mecanismo, grupos_alcance)
        dist = _alinear_distribucion(sistema_partido.distribucion_marginal(), self._dists)
        perdida = float(self._distancia_metrica(dist, self._dists))
        self._cache[asignacion] = (perdida, dist)
        return perdida, dist


class QNodos(SIA):
    """Implementacion submodular de QNodos basada en deltas y omegas."""

    def __init__(self, tpm: np.ndarray, config=None) -> None:
        super().__init__(tpm, config)
        self.distancia_metrica = seleccionar_emd(config)
        self.memoria_delta: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[float, np.ndarray]] = {}
        self.memoria_grupo_candidato: dict[tuple[tuple[int, int], ...], tuple[float, np.ndarray | None]] = {}
        self.clave_submodular: list[list[int]] = [[], []]
        self.vertices: set[tuple[int, int]] = set()
        self._cache_k_particiones: dict[tuple[int, ...], tuple[float, np.ndarray]] = {}

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
            # Warm-start: particion recursiva Q con memoizacion DP.
            memo_q: dict[tuple, tuple[int, ...]] = {}
            semilla_asig = self._particionar_recursivo_q(vertices, k, memo_q)

            buscador = _BuscadorKQNodos(
                vertices=vertices,
                sistema=self.sia_subsistema,
                dists=self.sia_dists_marginales,
                distancia_metrica=self.distancia_metrica,
                cache=self._cache_k_particiones,
            )
            if semilla_asig is not None:
                resultado_k = buscador.buscar_con_semilla(
                    k, semilla_asig, semilla=aplicacion.semilla_numpy + len(vertices)
                )
            else:
                resultado_k = buscador.buscar(k, semilla=aplicacion.semilla_numpy + len(vertices))
            grupos = _grupos_desde_asignacion(resultado_k.asignacion, vertices)
            return Solucion(
                estrategia="QNodos",
                perdida=float(resultado_k.perdida),
                distribucion_subsistema=distribucion_subsistema,
                distribucion_particion=resultado_k.distribucion,
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
            vector_delta = _alinear_distribucion(
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
        vector_union = _alinear_distribucion(
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

    def _particionar_recursivo_q(
        self,
        vertices: list[tuple[int, int]],
        k: int,
        memo: dict[tuple, tuple[int, ...] | None],
    ) -> tuple[int, ...] | None:
        """Particion jerarquica con memoizacion DP usando el algoritmo Q.

        Aplica algoritmo_q recursivamente (divide y venceras) para obtener
        una asignacion inicial de k grupos sin evaluaciones redundantes.
        Memoiza por (subconjunto_ordenado, k) para evitar recomputo.
        Retorna una asignacion sobre `vertices` en orden de entrada, o None.
        """
        clave = (tuple(sorted(vertices, key=lambda v: (v[0], v[1]))), k)
        if clave in memo:
            return memo[clave]

        n = len(vertices)
        if k >= n or n <= 1:
            memo[clave] = None
            return None

        # Solo guardar/restaurar vertices y memoria_grupo_candidato.
        # memoria_delta se acumula entre llamadas (es independiente del subconjunto).
        vertices_prev = self.vertices
        mem_cand_prev = self.memoria_grupo_candidato.copy()

        self.vertices = set(vertices)
        self.memoria_grupo_candidato.clear()
        try:
            clave_mip = self.algoritmo_q(list(vertices))
        except Exception:
            self.vertices = vertices_prev
            self.memoria_grupo_candidato = mem_cand_prev
            memo[clave] = None
            return None

        grupo_a = list(clave_mip)
        grupo_b = [v for v in vertices if v not in set(clave_mip)]

        self.vertices = vertices_prev
        self.memoria_grupo_candidato = mem_cand_prev

        if k == 2:
            set_a = set(grupo_a)
            asig = tuple(0 if v in set_a else 1 for v in vertices)
            memo[clave] = asig
            return asig

        # Para k>2: partir el grupo mayor en k-1 subgrupos.
        mayor, menor = (grupo_a, grupo_b) if len(grupo_a) >= len(grupo_b) else (grupo_b, grupo_a)
        sub_asig = self._particionar_recursivo_q(mayor, k - 1, memo)

        if sub_asig is None:
            memo[clave] = None
            return None

        # Indice de cada vertice dentro de `mayor` para mapear sub_asig.
        pos_en_mayor = {v: i for i, v in enumerate(mayor)}
        menor_set = set(menor)

        asig_global = [0] * n
        for i, v in enumerate(vertices):
            if v in menor_set:
                asig_global[i] = k - 1
            else:
                asig_global[i] = sub_asig[pos_en_mayor[v]]

        # Canonicalizar grupos (0, 1, ...).
        mapa: dict[int, int] = {}
        sig = 0
        canon = []
        for g in asig_global:
            if g not in mapa:
                mapa[g] = sig
                sig += 1
            canon.append(mapa[g])
        asig_canon = tuple(canon)
        memo[clave] = asig_canon
        return asig_canon


# Alias retrocompatible.
QNodes = QNodos

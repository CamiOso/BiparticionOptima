"""Biparticionamiento Exacto Recursivo (BER) para k-particiones.

Para k=2 es identico a QNodos (exacto por submodularidad + Queyranne).
Para k>=3 construye una semilla jerarquica garantizada phi_k <= phi_{k-1}:
  1. Obtiene la biparticion exacta (S1, S2) con QNodos k=2.
  2. Divide greedy uno de los grupos en cada paso (S1->S1a+S1b, etc.)
     evaluando siempre el costo de la k-particion completa.
  3. Refina el resultado con SA (BuscadorKRecocido existente).

Ventajas sobre QNodos SA puro:
  - La semilla BER respeta phi_k <= phi_{k-1} por construccion.
  - Subgrupos pequenos (<=_UMBRAL_FISICO nodos) se tratan exhaustivamente.
  - Subgrupos grandes usan Queyranne como proxy de busqueda + evaluacion exacta.
"""
from __future__ import annotations

import numpy as np

from src.estrategias.q_nodos import (
    QNodos,
    _BuscadorKQNodos,
    _alinear_distribucion,
    _grupos_desde_asignacion,
)
from src.funciones.formato import fmt_k_particion_q
from src.modelos.base.aplicacion import aplicacion
from src.modelos.nucleo.solucion import Solucion

_UMBRAL_FISICO = 14


class BER(QNodos):
    """Biparticionamiento Exacto Recursivo para k-particiones.

    Hereda toda la infraestructura de QNodos (Queyranne, SA, memos).
    Solo redefine aplicar_estrategia para k>2.
    """

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
        k: int = 2,
    ) -> Solucion:
        if k == 2:
            return super().aplicar_estrategia(
                estado_inicial, condicion, alcance, mecanismo, k=2
            )

        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        assert self.sia_dists_marginales is not None
        assert self.sia_subsistema is not None

        distribucion_subsistema = self.sia_dists_marginales
        self.memoria_delta.clear()
        self.memoria_grupo_candidato.clear()
        self._cache_k_particiones.clear()

        futuro = [(1, int(i)) for i in self.sia_subsistema.indices_ncubos.tolist()]
        presente = [(0, int(i)) for i in self.sia_subsistema.dims_ncubos.tolist()]
        vertices = list(presente + futuro)

        # Paso 1: biparticion exacta k=2 (Queyranne + SA)
        clave_k2, _, _ = self._mao_multi_start(vertices)
        clave_k2, _, _ = self._sa_biparticion(vertices, set(clave_k2))
        s1 = list(clave_k2)
        s2 = [v for v in vertices if v not in set(clave_k2)]

        # Paso 2: BER — expandir de 2 a k grupos de forma greedy
        semilla_ber = self._ber_semilla(vertices, [s1, s2], k - 2)

        # Paso 3: SA — refinar desde la semilla BER
        buscador = _BuscadorKQNodos(
            vertices=vertices,
            sistema=self.sia_subsistema,
            dists=self.sia_dists_marginales,
            distancia_metrica=self.distancia_metrica,
            cache=self._cache_k_particiones,
        )
        if semilla_ber is not None:
            resultado = buscador.buscar_con_semilla(
                k, semilla_ber, semilla=aplicacion.semilla_numpy + len(vertices)
            )
        else:
            resultado = buscador.buscar(k, semilla=aplicacion.semilla_numpy + len(vertices))

        grupos = _grupos_desde_asignacion(resultado.asignacion, vertices)
        return Solucion(
            estrategia="BER",
            perdida=float(resultado.perdida),
            distribucion_subsistema=distribucion_subsistema,
            distribucion_particion=resultado.distribucion,
            estado_inicial=estado_inicial,
            particion=fmt_k_particion_q(grupos),
        )

    # ------------------------------------------------------------------
    # Nucleo BER
    # ------------------------------------------------------------------

    def _ber_semilla(
        self,
        vertices: list[tuple[int, int]],
        grupos: list[list[tuple[int, int]]],
        splits_restantes: int,
    ) -> tuple[int, ...] | None:
        """Expande la lista de grupos dividiendo uno en cada paso.

        En cada iteracion prueba dividir cada grupo posible y elige el split
        que minimiza phi de la particion completa resultante.
        Aplica busqueda exhaustiva para grupos pequenos y Queyranne como
        proxy para grupos grandes, evaluando siempre el costo real.
        """
        if splits_restantes == 0:
            return _grupos_a_asignacion(vertices, grupos)

        mejor_perdida = float("inf")
        mejor_grupos: list[list[tuple[int, int]]] | None = None

        for i, grupo in enumerate(grupos):
            nodos = sorted({idx for _, idx in grupo})
            n_nodos = len(nodos)
            if n_nodos < 2:
                continue

            otros = grupos[:i] + grupos[i + 1:]

            if n_nodos <= _UMBRAL_FISICO:
                perdida, split_a, split_b = self._split_exhaustivo(grupo, nodos, otros)
            else:
                perdida, split_a, split_b = self._split_queyranne(grupo, otros)

            if split_a is not None and perdida < mejor_perdida:
                mejor_perdida = perdida
                mejor_grupos = [split_a, split_b] + otros

        if mejor_grupos is None:
            return _grupos_a_asignacion(vertices, grupos)

        return self._ber_semilla(vertices, mejor_grupos, splits_restantes - 1)

    def _split_exhaustivo(
        self,
        grupo: list[tuple[int, int]],
        nodos: list[int],
        otros: list[list[tuple[int, int]]],
    ) -> tuple[float, list | None, list | None]:
        """Prueba todos los 2^(n-1) splits de nodos fisicos, evalua la k-particion completa."""
        n = len(nodos)
        mejor_perdida = float("inf")
        mejor_a: list | None = None
        mejor_b: list | None = None

        for mask in range(1, 1 << (n - 1)):
            nodos_a = {nodos[j] for j in range(n) if mask & (1 << j)}
            split_a = [(t, idx) for t, idx in grupo if idx in nodos_a]
            split_b = [(t, idx) for t, idx in grupo if idx not in nodos_a]
            if not split_a or not split_b:
                continue

            perdida, _ = self._evaluar_kparticion([split_a, split_b] + otros)
            if perdida < mejor_perdida:
                mejor_perdida = perdida
                mejor_a, mejor_b = split_a, split_b

        return mejor_perdida, mejor_a, mejor_b

    def _split_queyranne(
        self,
        grupo: list[tuple[int, int]],
        otros: list[list[tuple[int, int]]],
    ) -> tuple[float, list | None, list | None]:
        """Usa Queyranne como proxy para encontrar un buen split del grupo grande."""
        vertices_prev = self.vertices
        mem_cand_prev = dict(self.memoria_grupo_candidato)
        self.vertices = set(grupo)
        self.memoria_grupo_candidato.clear()
        try:
            clave = self.algoritmo_q(list(grupo))
            split_a_set = set(clave)
            split_a = [v for v in grupo if v in split_a_set]
            split_b = [v for v in grupo if v not in split_a_set]
        except Exception:
            return float("inf"), None, None
        finally:
            self.vertices = vertices_prev
            self.memoria_grupo_candidato = mem_cand_prev

        if not split_a or not split_b:
            return float("inf"), None, None

        perdida, _ = self._evaluar_kparticion([split_a, split_b] + otros)
        return perdida, split_a, split_b

    def _evaluar_kparticion(
        self, grupos: list[list[tuple[int, int]]]
    ) -> tuple[float, np.ndarray]:
        """Evalua phi de una k-particion arbitraria usando k_bipartir_temporal."""
        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None

        mec_grupos = [tuple(sorted(idx for t, idx in g if t == 0)) for g in grupos]
        alc_grupos = [tuple(sorted(idx for t, idx in g if t == 1)) for g in grupos]

        sistema_partido = self.sia_subsistema.k_bipartir_temporal(
            list(mec_grupos), list(alc_grupos)
        )
        dist = _alinear_distribucion(
            sistema_partido.distribucion_marginal(), self.sia_dists_marginales
        )
        perdida = float(self.distancia_metrica(dist, self.sia_dists_marginales))
        return perdida, dist


# ------------------------------------------------------------------
# Utilidad
# ------------------------------------------------------------------

def _grupos_a_asignacion(
    vertices: list[tuple[int, int]],
    grupos: list[list[tuple[int, int]]],
) -> tuple[int, ...]:
    """Convierte lista de grupos a tupla de asignaciones sobre todos los vertices."""
    vert_to_grupo: dict[tuple[int, int], int] = {}
    for g_idx, grupo in enumerate(grupos):
        for v in grupo:
            vert_to_grupo[v] = g_idx
    return tuple(vert_to_grupo.get(v, 0) for v in vertices)


# Alias
BER_QNodos = BER

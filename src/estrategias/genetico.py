"""Estrategia Algoritmo Genético para k-partición.

Metaheurística evolutiva que mantiene una **población de asignaciones**
(cromosomas) y las hace evolucionar mediante selección, cruce y mutación
hasta converger a una partición de baja pérdida.

Representación
--------------
Cromosoma: vector entero [g_0, g_1, …, g_{n-1}] con g_i ∈ {0, …, k−1},
donde n = número de nodos y g_i es el grupo al que pertenece el nodo i.
Fitness: f(c) = −φ(c)   (minimizar pérdida ↔ maximizar fitness).

Operadores genéticos
---------------------
Selección    — torneo binario: se eligen 2 individuos al azar; gana el mejor.
Cruce        — uniforme: cada gen se hereda del padre 1 o padre 2 con P=0.5.
Mutación     — punto: cada gen cambia de grupo con probabilidad p_mut = 1/n.
Élitismo     — los `n_elite` mejores individuos pasan intactos a la siguiente generación.
Canonicalización — normaliza cromosomas para evitar duplicados semánticamente
                   equivalentes (permutaciones de etiquetas de grupo).

Complejidad: O(generaciones × población × evaluación_particion)
"""

from __future__ import annotations

import numpy as np

from src.constantes.models import GENETICO_LABEL
from src.funciones.formato import fmt_biparticion, fmt_k_particion_asignacion
from src.funciones.iit import seleccionar_emd
from src.modelos.base.sia import SIA
from src.modelos.nucleo.solucion import Solucion


class AlgoritmoGenetico(SIA):
    """Búsqueda evolutiva de la k-partición óptima.

    Parámetros
    ----------
    tam_poblacion  : Número de individuos por generación.
    generaciones   : Número máximo de generaciones.
    n_elite        : Individuos que pasan directamente sin cambios.
    config         : AppConfig inyectada (None → usa singleton global).
    """

    def __init__(
        self,
        tpm: np.ndarray,
        config=None,
        tam_poblacion: int = 40,
        generaciones: int = 80,
        n_elite: int = 2,
    ) -> None:
        super().__init__(tpm, config)
        self.distancia_metrica = seleccionar_emd(config)
        self.tam_poblacion = tam_poblacion
        self.generaciones = generaciones
        self.n_elite = max(1, n_elite)

    # ------------------------------------------------------------------
    # Puerto SIA
    # ------------------------------------------------------------------

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
        k: int = 2,
        **_kwargs,
    ) -> Solucion:
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)
        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None

        nodos = sorted(
            set(int(v) for v in self.sia_subsistema.indices_ncubos.tolist())
            | set(int(v) for v in self.sia_subsistema.dims_ncubos.tolist())
        )
        alcance_total = tuple(int(v) for v in self.sia_subsistema.indices_ncubos.tolist())
        mecanismo_total = tuple(int(v) for v in self.sia_subsistema.dims_ncubos.tolist())
        n = len(nodos)
        k_eff = min(k, n)

        if n <= 1 or k_eff < 2:
            return Solucion(
                estrategia=GENETICO_LABEL,
                perdida=0.0,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=self.sia_dists_marginales.copy(),
                estado_inicial=estado_inicial,
                particion="NO-PARTITION",
            )

        mejor_asig, mejor_perdida, mejor_dist = self._evolucionar(nodos, k_eff)

        particion_str = self._formatear(
            mejor_asig, nodos, alcance_total, mecanismo_total, k_eff
        )
        return Solucion(
            estrategia=GENETICO_LABEL,
            perdida=mejor_perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=mejor_dist,
            estado_inicial=estado_inicial,
            particion=particion_str,
        )

    # ------------------------------------------------------------------
    # Bucle evolutivo
    # ------------------------------------------------------------------

    def _evolucionar(
        self, nodos: list[int], k: int
    ) -> tuple[tuple[int, ...], float, np.ndarray]:
        n = len(nodos)
        rng = np.random.default_rng(73)
        p_mut = 1.0 / n

        # Inicializar población garantizando al menos k grupos distintos
        poblacion: list[tuple[int, ...]] = []
        for _ in range(self.tam_poblacion):
            base = list(range(k))
            resto = [int(rng.integers(0, k)) for _ in range(max(0, n - k))]
            cromo = base + resto
            rng.shuffle(cromo)
            poblacion.append(self._canonicalizar(tuple(cromo)))

        # Evaluar población inicial
        scores: list[tuple[float, np.ndarray]] = [
            self._evaluar(nodos, c) for c in poblacion
        ]

        mejor_perdida, mejor_dist = min(scores, key=lambda s: s[0])
        mejor_asig = poblacion[scores.index((mejor_perdida, mejor_dist))]

        for _ in range(self.generaciones):
            # Élitismo: ordenar por pérdida ascendente
            orden = sorted(range(len(poblacion)), key=lambda i: scores[i][0])
            nueva_gen: list[tuple[int, ...]] = [poblacion[i] for i in orden[: self.n_elite]]

            while len(nueva_gen) < self.tam_poblacion:
                p1 = self._torneo(poblacion, scores, rng)
                p2 = self._torneo(poblacion, scores, rng)
                hijo = self._cruce_uniforme(p1, p2, rng)
                hijo = self._mutar(hijo, k, p_mut, rng)
                nuevo = self._canonicalizar(hijo)
                if len(set(nuevo)) >= 2:
                    nueva_gen.append(nuevo)

            poblacion = nueva_gen
            scores = [self._evaluar(nodos, c) for c in poblacion]

            gen_mejor_perdida, gen_mejor_dist = min(scores, key=lambda s: s[0])
            if gen_mejor_perdida < mejor_perdida - 1e-12:
                mejor_perdida = gen_mejor_perdida
                mejor_dist = gen_mejor_dist
                mejor_asig = poblacion[scores.index((gen_mejor_perdida, gen_mejor_dist))]

        return mejor_asig, mejor_perdida, mejor_dist

    # ------------------------------------------------------------------
    # Operadores genéticos
    # ------------------------------------------------------------------

    def _torneo(
        self,
        poblacion: list[tuple[int, ...]],
        scores: list[tuple[float, np.ndarray]],
        rng: np.random.Generator,
        tam: int = 3,
    ) -> tuple[int, ...]:
        indices = rng.integers(0, len(poblacion), size=tam)
        mejor_idx = min(indices, key=lambda i: scores[i][0])
        return poblacion[mejor_idx]

    def _cruce_uniforme(
        self,
        p1: tuple[int, ...],
        p2: tuple[int, ...],
        rng: np.random.Generator,
    ) -> tuple[int, ...]:
        mascara = rng.random(len(p1)) < 0.5
        return tuple(int(p1[i]) if mascara[i] else int(p2[i]) for i in range(len(p1)))

    def _mutar(
        self,
        cromo: tuple[int, ...],
        k: int,
        p_mut: float,
        rng: np.random.Generator,
    ) -> tuple[int, ...]:
        nueva = list(cromo)
        for i in range(len(nueva)):
            if rng.random() < p_mut:
                nueva[i] = int(rng.integers(0, k))
        return tuple(nueva)

    def _canonicalizar(self, asig: tuple[int, ...]) -> tuple[int, ...]:
        mapa: dict[int, int] = {}
        sig = 0
        canon = []
        for g in asig:
            if g not in mapa:
                mapa[g] = sig
                sig += 1
            canon.append(mapa[g])
        return tuple(canon)

    # ------------------------------------------------------------------
    # Evaluación
    # ------------------------------------------------------------------

    def _evaluar(
        self,
        nodos: list[int],
        asig: tuple[int, ...],
    ) -> tuple[float, np.ndarray]:
        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None
        if len(set(asig)) < 2:
            asig = tuple(i % 2 for i in range(len(nodos)))

        sistema_partido = self.sia_subsistema.k_bipartir(nodos, asig)
        dist = sistema_partido.distribucion_marginal()
        if dist.size != self.sia_dists_marginales.size:
            alineada = np.zeros_like(self.sia_dists_marginales)
            alineada[: dist.size] = dist
            dist = alineada
        return float(self.distancia_metrica(self.sia_dists_marginales, dist)), dist

    def _formatear(
        self,
        asig: tuple[int, ...],
        nodos: list[int],
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
        k: int,
    ) -> str:
        if k == 2 and len(set(asig)) <= 2:
            sub_alc = tuple(nodos[i] for i, g in enumerate(asig) if g == 0 and nodos[i] in alcance_total)
            sub_mec = tuple(nodos[i] for i, g in enumerate(asig) if g == 0 and nodos[i] in mecanismo_total)
            return fmt_biparticion(sub_alc, sub_mec, alcance_total, mecanismo_total)
        return fmt_k_particion_asignacion(nodos, asig, alcance_total, mecanismo_total)

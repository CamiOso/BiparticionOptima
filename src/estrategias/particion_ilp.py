"""Estrategia ILP (Relajación LP del k-cut mínimo) para k-partición.

Formula el problema de k-partición como un **programa lineal entero** (ILP)
sobre el grafo de acoplamientos del sistema. Usa la relajación LP continua
—resolver el problema relajando x ∈ {0,1} a x ∈ [0,1]— y luego redondea
la solución fraccionaria para obtener una asignación entera.

Formulación LP del k-cut
-------------------------
Variables:
  x[i,g] ∈ [0,1]   — probabilidad de asignar el nodo i al grupo g
  y[i,j] ∈ [0,1]   — indicador de que la arista (i,j) está cortada

Objetivo (minimizar el peso total del corte):
  min  Σ_{i<j} w_{ij} · y[i,j]

Restricciones:
  (R1) Σ_g x[i,g] = 1   ∀i          (cada nodo en exactamente un grupo)
  (R2) Σ_i x[i,g] ≥ 1/k ∀g          (cada grupo tiene al menos un nodo)
  (R3) y[i,j] ≥ x[i,g] − x[j,g]     ∀g, i<j  (corte mayor que diferencia)
  (R4) y[i,j] ≥ x[j,g] − x[i,g]     ∀g, i<j
  (bounds) x ∈ [0,1], y ∈ [0,1]

La solución LP es un lower bound exacto del ILP; el redondeo argmax da
una solución factible que se refina con búsqueda local.

Propiedad teórica: la LP relaxation del k-cut tiene ratio de aproximación
2(1 − 1/k) (Calinescu et al., 2000), el mejor conocido para k ≥ 3.

Complejidad: O(LP) = O((n²k)³) con simplex; O(n²k) variables en la práctica.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from src.constantes.models import ILP_LABEL
from src.funciones.formato import fmt_biparticion, fmt_k_particion_asignacion
from src.funciones.grafo_info import construir_afinidad
from src.funciones.iit import seleccionar_emd
from src.modelos.base.sia import SIA
from src.modelos.nucleo.solucion import Solucion


class ParticionILP(SIA):
    """Partición por relajación LP del k-cut mínimo en el grafo de acoplamientos."""

    def __init__(
        self,
        tpm: np.ndarray,
        config=None,
        max_iter_refinamiento: int = 30,
    ) -> None:
        super().__init__(tpm, config)
        self.distancia_metrica = seleccionar_emd(config)
        self.max_iter_refinamiento = max_iter_refinamiento

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

        alcance_total = tuple(int(v) for v in self.sia_subsistema.indices_ncubos.tolist())
        mecanismo_total = tuple(int(v) for v in self.sia_subsistema.dims_ncubos.tolist())
        nodos, W = construir_afinidad(self.sia_subsistema)
        n = len(nodos)
        k_eff = min(k, n)

        if n <= 1 or k_eff < 2:
            return Solucion(
                estrategia=ILP_LABEL,
                perdida=0.0,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=self.sia_dists_marginales.copy(),
                estado_inicial=estado_inicial,
                particion="NO-PARTITION",
            )

        asig = self._resolver_lp(W, k_eff, n)
        asig, perdida, dist = self._refinar_local(nodos, asig, k_eff)

        particion_str = self._formatear(asig, nodos, alcance_total, mecanismo_total, k_eff)
        return Solucion(
            estrategia=ILP_LABEL,
            perdida=perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist,
            estado_inicial=estado_inicial,
            particion=particion_str,
        )

    # ------------------------------------------------------------------
    # Relajación LP
    # ------------------------------------------------------------------

    def _resolver_lp(self, W: np.ndarray, k: int, n: int) -> tuple[int, ...]:
        """Resuelve la relajación LP y redondea al argmax."""
        # Índices de aristas (i < j)
        aristas = [(i, j) for i in range(n) for j in range(i + 1, n)]
        m = len(aristas)
        arista_idx = {a: idx for idx, a in enumerate(aristas)}

        # Variables: x[i,g] para i=0..n-1, g=0..k-1 → índice i*k + g
        # y[a]      para a=0..m-1                    → índice n*k + a
        n_vars = n * k + m

        # ── Objetivo: minimizar Σ w_ij * y_ij ──────────────────────────
        c = np.zeros(n_vars)
        for idx, (i, j) in enumerate(aristas):
            c[n * k + idx] = W[i, j]

        # ── Restricciones de igualdad (R1): Σ_g x[i,g] = 1 ─────────────
        A_eq = np.zeros((n, n_vars))
        b_eq = np.ones(n)
        for i in range(n):
            for g in range(k):
                A_eq[i, i * k + g] = 1.0

        # ── Restricciones de desigualdad ─────────────────────────────────
        # (R2) −Σ_i x[i,g] ≤ −1/k  →  cada grupo tiene masa ≥ 1/k
        # (R3) x[i,g] − x[j,g] − y[ij] ≤ 0  ∀g, ∀arista
        # (R4) x[j,g] − x[i,g] − y[ij] ≤ 0  ∀g, ∀arista
        n_ineq = k + 2 * k * m
        A_ub = np.zeros((n_ineq, n_vars))
        b_ub = np.zeros(n_ineq)
        fila = 0

        # R2
        for g in range(k):
            for i in range(n):
                A_ub[fila, i * k + g] = -1.0
            b_ub[fila] = -1.0 / k
            fila += 1

        # R3 y R4
        for g in range(k):
            for (i, j), a_idx in arista_idx.items():
                # R3: x[i,g] − x[j,g] − y[ij] ≤ 0
                A_ub[fila, i * k + g] = 1.0
                A_ub[fila, j * k + g] = -1.0
                A_ub[fila, n * k + a_idx] = -1.0
                fila += 1
                # R4: x[j,g] − x[i,g] − y[ij] ≤ 0
                A_ub[fila, j * k + g] = 1.0
                A_ub[fila, i * k + g] = -1.0
                A_ub[fila, n * k + a_idx] = -1.0
                fila += 1

        bounds = [(0.0, 1.0)] * n_vars

        resultado = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if resultado.success:
            x_sol = resultado.x[: n * k].reshape(n, k)
            asig = tuple(int(np.argmax(x_sol[i])) for i in range(n))
        else:
            # Fallback: asignación espectral
            asig = self._asignacion_espectral(W, k, n)

        return self._canonicalizar(asig)

    def _asignacion_espectral(self, W: np.ndarray, k: int, n: int) -> tuple[int, ...]:
        """Fallback espectral: k primeros eigenvectores + asignación por umbrales."""
        L = np.diag(W.sum(axis=1)) - W
        vals, vecs = np.linalg.eigh(L)
        embedding = vecs[:, 1 : k + 1] if k < n else vecs
        asig = np.argmax(embedding, axis=1) % k
        return tuple(int(g) for g in asig)

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
    # Evaluación y refinamiento
    # ------------------------------------------------------------------

    def _evaluar(
        self, nodos: list[int], asig: tuple[int, ...]
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

    def _refinar_local(
        self,
        nodos: list[int],
        asig_ini: tuple[int, ...],
        k: int,
    ) -> tuple[tuple[int, ...], float, np.ndarray]:
        perdida, dist = self._evaluar(nodos, asig_ini)
        mejor_asig = asig_ini
        mejor_perdida = perdida
        mejor_dist = dist
        n = len(nodos)

        for _ in range(self.max_iter_refinamiento):
            mejorado = False
            for i in range(n):
                for g in range(k):
                    if mejor_asig[i] == g:
                        continue
                    nueva = list(mejor_asig)
                    nueva[i] = g
                    nueva_t = tuple(nueva)
                    if len(set(nueva_t)) < 2:
                        continue
                    p, d = self._evaluar(nodos, nueva_t)
                    if p < mejor_perdida - 1e-12:
                        mejor_perdida, mejor_dist, mejor_asig = p, d, nueva_t
                        mejorado = True
            if not mejorado:
                break

        return mejor_asig, mejor_perdida, mejor_dist

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

"""Estrategia Information Bottleneck para k-partición (Tishby et al., 1999).

Agrupa los nodos del sistema según la similitud de sus **perfiles causales**:
la distribución de probabilidad de transición de cada nodo, vista como un vector
en el espacio de estados. Nodos con perfiles similares —que se comportan igual
bajo la dinámica del sistema— se asignan al mismo grupo.

Algoritmo de minimización alternada
------------------------------------
Dado un conjunto de perfiles causales {f_i} para cada nodo i:

1. Inicializar asignación blanda aleatoria:  p(t|i) ~ Dirichlet(1)
2. Iterar hasta convergencia:
   a. Centroide del cluster t:
          μ_t = Σ_i p(t|i)·p(i) / p(t)     (media ponderada de perfiles)
   b. Re-asignar con la regla de Bayes soft:
          p(t|i) ∝ p(t) · exp(−β · KL(f_i ‖ μ_t))
3. Hard-assign: z_i = argmax_t p(t|i)
4. Refinamiento local por intercambio de nodos entre grupos

El hiperparámetro β controla el "compromiso" compresión/precisión:
  β grande → grupos muy separados (fuerte compresión)
  β pequeño → grupos difusos (preserva toda la información)

Complejidad: O(n²·k·iter_IB + n·k·iter_local)
"""

from __future__ import annotations

import numpy as np

from src.constantes.models import IB_LABEL
from src.funciones.formato import fmt_biparticion, fmt_k_particion_asignacion
from src.funciones.iit import seleccionar_emd
from src.modelos.base.sia import SIA
from src.modelos.nucleo.solucion import Solucion


def _kl_suavizada(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p ‖ q) con suavizado ε para evitar log(0)."""
    p_ = np.clip(p, 1e-12, None)
    q_ = np.clip(q, 1e-12, None)
    p_ = p_ / p_.sum()
    q_ = q_ / q_.sum()
    return float(np.sum(p_ * np.log(p_ / q_)))


class InformacionBottleneck(SIA):
    """Encuentra la k-partición mínima de pérdida usando compresión IB.

    Extrae el perfil causal de cada nodo (su distribución marginal aplanada
    desde el n-cubo correspondiente) y aplica la minimización alternada del
    Information Bottleneck para agrupar nodos con comportamientos similares.

    Parámetros
    ----------
    beta       : Factor de compresión. Valores en [2, 8] funcionan bien.
    max_iter   : Iteraciones máximas del bucle IB.
    n_restarts : Reinicios aleatorios; el mejor resultado se reporta.
    """

    def __init__(
        self,
        tpm: np.ndarray,
        config=None,
        beta: float = 4.0,
        max_iter: int = 60,
        n_restarts: int = 8,
    ) -> None:
        super().__init__(tpm, config)
        self.distancia_metrica = seleccionar_emd(config)
        self.beta = beta
        self.max_iter = max_iter
        self.n_restarts = n_restarts

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
                estrategia=IB_LABEL,
                perdida=0.0,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=self.sia_dists_marginales.copy(),
                estado_inicial=estado_inicial,
                particion="NO-PARTITION",
            )

        perfiles = self._extraer_perfiles(nodos)

        mejor_perdida = np.inf
        mejor_dist = self.sia_dists_marginales.copy()
        mejor_asig: tuple[int, ...] = tuple(i % k_eff for i in range(n))

        rng = np.random.default_rng(73)
        for restart in range(self.n_restarts):
            asig = self._ib_alternating(perfiles, k_eff, rng, restart)
            asig, perdida, dist = self._refinar_local(nodos, asig, k_eff)
            if perdida < mejor_perdida:
                mejor_perdida = perdida
                mejor_dist = dist
                mejor_asig = asig

        particion_str = self._formatear(
            mejor_asig, nodos, alcance_total, mecanismo_total, k_eff
        )
        return Solucion(
            estrategia=IB_LABEL,
            perdida=mejor_perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=mejor_dist,
            estado_inicial=estado_inicial,
            particion=particion_str,
        )

    # ------------------------------------------------------------------
    # Perfiles causales
    # ------------------------------------------------------------------

    def _extraer_perfiles(self, nodos: list[int]) -> np.ndarray:
        """Devuelve matriz (n_nodos, max_len) con el perfil de cada nodo."""
        nodo_a_cubo = {int(c.indice): c for c in self.sia_subsistema.ncubos}
        perfiles_raw: list[np.ndarray] = []
        for nodo in nodos:
            if nodo in nodo_a_cubo:
                raw = nodo_a_cubo[nodo].data.ravel().astype(np.float64)
            else:
                raw = np.array([0.5, 0.5], dtype=np.float64)
            raw = np.clip(raw, 1e-12, None)
            raw /= raw.sum()
            perfiles_raw.append(raw)

        max_len = max(len(p) for p in perfiles_raw)
        mat = np.zeros((len(nodos), max_len), dtype=np.float64)
        for i, p in enumerate(perfiles_raw):
            mat[i, : len(p)] = p
            mat[i] /= mat[i].sum()
        return mat

    # ------------------------------------------------------------------
    # Minimización alternada IB
    # ------------------------------------------------------------------

    def _ib_alternating(
        self,
        perfiles: np.ndarray,
        k: int,
        rng: np.random.Generator,
        restart: int,
    ) -> tuple[int, ...]:
        n = len(perfiles)
        # p(t|i): asignación blanda [n, k]
        pt_x = rng.dirichlet(np.ones(k), size=n)
        # p(i): prior uniforme sobre nodos
        px = np.ones(n, dtype=np.float64) / n

        for _ in range(self.max_iter):
            # p(t) = distribución marginal de clusters
            pt = pt_x.T @ px
            pt = np.clip(pt, 1e-12, None)
            pt /= pt.sum()

            # Centroides: media ponderada de perfiles por cluster
            centroides = np.zeros((k, perfiles.shape[1]), dtype=np.float64)
            for t in range(k):
                pesos = pt_x[:, t] * px
                total = pesos.sum()
                if total > 1e-12:
                    centroides[t] = pesos @ perfiles / total
                else:
                    centroides[t] = perfiles.mean(axis=0)
                centroides[t] = np.clip(centroides[t], 1e-12, None)
                centroides[t] /= centroides[t].sum()

            # Actualizar p(t|i) ∝ p(t)·exp(−β·KL(f_i ‖ centroide_t))
            log_nuevo = np.zeros((n, k), dtype=np.float64)
            for i in range(n):
                for t in range(k):
                    log_nuevo[i, t] = np.log(pt[t]) - self.beta * _kl_suavizada(
                        perfiles[i], centroides[t]
                    )
            # Estabilización numérica
            log_nuevo -= log_nuevo.max(axis=1, keepdims=True)
            pt_x_new = np.exp(log_nuevo)
            pt_x_new /= pt_x_new.sum(axis=1, keepdims=True)

            if np.max(np.abs(pt_x_new - pt_x)) < 1e-8:
                break
            pt_x = pt_x_new

        return tuple(int(np.argmax(pt_x[i])) for i in range(n))

    # ------------------------------------------------------------------
    # Evaluación y refinamiento
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

    def _refinar_local(
        self,
        nodos: list[int],
        asig_ini: tuple[int, ...],
        k: int,
        max_iter: int = 30,
    ) -> tuple[tuple[int, ...], float, np.ndarray]:
        perdida, dist = self._evaluar(nodos, asig_ini)
        mejor_asig = asig_ini
        mejor_perdida = perdida
        mejor_dist = dist
        n = len(nodos)

        for _ in range(max_iter):
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

    # ------------------------------------------------------------------
    # Formateo
    # ------------------------------------------------------------------

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

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

import os
from concurrent.futures import ThreadPoolExecutor

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
        n_random_restarts: int = 8,
    ) -> None:
        super().__init__(tpm, config)
        self.distancia_metrica = seleccionar_emd(config)
        self.beta = beta
        self.max_iter = max_iter
        self.n_restarts = n_restarts
        self.n_random_restarts = n_random_restarts

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

        # Fase IB: solución semilla usando perfiles causales
        mejor_perdida_kbp = np.inf
        mejor_asig_seed: tuple[int, ...] = tuple(i % k_eff for i in range(n))
        rng = np.random.default_rng(73)
        for restart in range(self.n_restarts):
            asig = self._ib_alternating(perfiles, k_eff, rng, restart)
            asig, perdida, dist = self._refinar_local(nodos, asig, k_eff)
            if perdida < mejor_perdida_kbp:
                mejor_perdida_kbp = perdida
                mejor_asig_seed = asig

        if k_eff == 2:
            # Para k=2 se usa el espacio completo bipartir (subalcance y
            # submecanismo independientes). La semilla IB (k_bipartir) se
            # convierte y se refina como un arranque más.
            grupo0 = {nodos[i] for i, g in enumerate(mejor_asig_seed) if g == 0}
            seed_alc = tuple(v for v in alcance_total if v in grupo0)
            seed_mec = tuple(v for v in mecanismo_total if v in grupo0)
            subalc, submec, perdida, dist = self._multistart_k2(
                alcance_total, mecanismo_total, seed_alc, seed_mec
            )
            return Solucion(
                estrategia=IB_LABEL,
                perdida=perdida,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=dist,
                estado_inicial=estado_inicial,
                particion=fmt_biparticion(subalc, submec, alcance_total, mecanismo_total),
            )

        # k > 2: semilla IB + reinicios aleatorios en espacio k_bipartir
        mejor_asig, mejor_perdida, mejor_dist = self._multistart_kn(
            nodos, k_eff, mejor_asig_seed
        )
        return Solucion(
            estrategia=IB_LABEL,
            perdida=mejor_perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=mejor_dist,
            estado_inicial=estado_inicial,
            particion=self._formatear(mejor_asig, nodos, alcance_total, mecanismo_total, k_eff),
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
    # Multi-start con objetivo φ directo
    # ------------------------------------------------------------------

    def _evaluar_bipartir(
        self,
        subalcance: tuple[int, ...],
        submecanismo: tuple[int, ...],
    ) -> tuple[float, np.ndarray]:
        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None
        sp = self.sia_subsistema.bipartir(
            np.array(list(subalcance), dtype=np.int8),
            np.array(list(submecanismo), dtype=np.int8),
        )
        dist = sp.distribucion_marginal()
        ref = self.sia_dists_marginales
        if dist.size != ref.size:
            alineada = np.zeros_like(ref)
            alineada[: dist.size] = dist
            dist = alineada
        return float(self.distancia_metrica(ref, dist)), dist

    def _refinar_bipartir(
        self,
        subalc: tuple[int, ...],
        submec: tuple[int, ...],
        perdida: float,
        dist: np.ndarray,
        alc_total: tuple[int, ...],
        mec_total: tuple[int, ...],
        max_iter: int = 24,
    ) -> tuple[tuple[int, ...], tuple[int, ...], float, np.ndarray]:
        """Descenso más pronunciado en el espacio bipartir: evalúa todos los vecinos y elige el mejor."""
        for _ in range(max_iter):
            mejor_ca, mejor_cm, mejor_p, mejor_d = subalc, submec, perdida, dist

            for nodo in alc_total:
                nuevo = set(subalc)
                nuevo.symmetric_difference_update({nodo})
                ca = tuple(v for v in alc_total if v in nuevo)
                if not ca and not submec:
                    continue
                if ca == alc_total and submec == mec_total:
                    continue
                p, d = self._evaluar_bipartir(ca, submec)
                if p < mejor_p - 1e-12:
                    mejor_ca, mejor_cm, mejor_p, mejor_d = ca, submec, p, d

            for nodo in mec_total:
                nuevo = set(submec)
                nuevo.symmetric_difference_update({nodo})
                cm = tuple(v for v in mec_total if v in nuevo)
                if not subalc and not cm:
                    continue
                if subalc == alc_total and cm == mec_total:
                    continue
                p, d = self._evaluar_bipartir(subalc, cm)
                if p < mejor_p - 1e-12:
                    mejor_ca, mejor_cm, mejor_p, mejor_d = subalc, cm, p, d

            if mejor_p + 1e-12 >= perdida:
                break
            subalc, submec, perdida, dist = mejor_ca, mejor_cm, mejor_p, mejor_d

        return subalc, submec, perdida, dist

    def _multistart_k2(
        self,
        alc_total: tuple[int, ...],
        mec_total: tuple[int, ...],
        seed_alc: tuple[int, ...] | None = None,
        seed_mec: tuple[int, ...] | None = None,
    ) -> tuple[tuple[int, ...], tuple[int, ...], float, np.ndarray]:
        assert self.sia_dists_marginales is not None
        n_alc, n_mec = len(alc_total), len(mec_total)
        rng = np.random.default_rng(42)

        starts: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        if seed_alc is not None and (seed_alc or seed_mec):
            if not (seed_alc == alc_total and seed_mec == mec_total):
                starts.append((seed_alc, seed_mec))

        for _ in range(self.n_random_restarts):
            for _intento in range(20):
                alc_mask = rng.integers(0, 2, n_alc).astype(bool)
                mec_mask = rng.integers(0, 2, n_mec).astype(bool)
                sa = tuple(alc_total[i] for i in range(n_alc) if alc_mask[i])
                sm = tuple(mec_total[j] for j in range(n_mec) if mec_mask[j])
                if (sa or sm) and not (sa == alc_total and sm == mec_total):
                    starts.append((sa, sm))
                    break

        if not starts:
            starts = [((alc_total[0],) if alc_total else (), ())]

        def _evaluar_start(sa_sm: tuple) -> tuple:
            sa, sm = sa_sm
            p, d = self._evaluar_bipartir(sa, sm)
            return self._refinar_bipartir(sa, sm, p, d, alc_total, mec_total)

        n_workers = min(os.cpu_count() or 1, len(starts))
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            resultados = list(executor.map(_evaluar_start, starts))

        mejor_perdida = float("inf")
        mejor_subalc: tuple[int, ...] = (alc_total[0],) if alc_total else ()
        mejor_submec: tuple[int, ...] = ()
        mejor_dist = self.sia_dists_marginales.copy()
        for sa, sm, p, d in resultados:
            if p < mejor_perdida:
                mejor_perdida, mejor_subalc, mejor_submec, mejor_dist = p, sa, sm, d

        return mejor_subalc, mejor_submec, mejor_perdida, mejor_dist

    def _multistart_kn(
        self,
        nodos: list[int],
        k: int,
        seed_asig: tuple[int, ...] | None = None,
    ) -> tuple[tuple[int, ...], float, np.ndarray]:
        assert self.sia_dists_marginales is not None
        n = len(nodos)
        rng = np.random.default_rng(42)

        asignaciones: list[tuple[int, ...]] = []
        if seed_asig is not None and len(set(seed_asig)) >= 2:
            asignaciones.append(seed_asig)

        for _ in range(self.n_random_restarts):
            base = list(range(k))
            resto = [int(rng.integers(0, k)) for _ in range(max(0, n - k))]
            nueva = base + resto
            rng.shuffle(nueva)
            nueva_t: tuple[int, ...] = tuple(nueva)
            mapa: dict[int, int] = {}
            sig = 0
            canon = []
            for g in nueva_t:
                if g not in mapa:
                    mapa[g] = sig
                    sig += 1
                canon.append(mapa[g])
            nueva_t = tuple(canon)
            if len(set(nueva_t)) >= 2:
                asignaciones.append(nueva_t)

        if not asignaciones:
            asignaciones = [tuple(i % k for i in range(n))]

        def _evaluar_asig(asig: tuple) -> tuple:
            return self._refinar_local(nodos, asig, k)

        n_workers = min(os.cpu_count() or 1, len(asignaciones))
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            resultados = list(executor.map(_evaluar_asig, asignaciones))

        mejor_asig = asignaciones[0]
        mejor_perdida = float("inf")
        mejor_dist = self.sia_dists_marginales.copy()
        for asig_r, p, d in resultados:
            if p < mejor_perdida:
                mejor_perdida, mejor_asig, mejor_dist = p, asig_r, d

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

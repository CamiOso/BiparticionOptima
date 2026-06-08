"""Estrategia espectral causal v2.

Mejoras sobre v1:
  - Pesos MI: W[i][j] = I(Xi(t+1); Xj(t)) en lugar de sensibilidad finita.
    Captura dependencia estadística total; más rápido (slicing vs loop Python).
  - Laplacianos dirigidos: L_sal (D_out) y L_ent (D_in) en lugar del simétrico
    D-W. Sus espectros complementarios preservan asimetría causal W[i][j]≠W[j][i].
  - Multi-arranque: candidatos de L_sal, L_ent, hipergrafo y aleatorios, todos
    refinados con EMD exacto; retorna el mejor global.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.constantes.models import CIRCUITO_LABEL
from src.funciones.formato import fmt_biparticion, fmt_k_particion_asignacion
from src.funciones.iit import seleccionar_emd
from src.modelos.base.sia import SIA
from src.modelos.nucleo.solucion import Solucion


@dataclass(frozen=True)
class _ResultadoParticion:
    perdida: float
    distribucion: np.ndarray
    subalcance: tuple[int, ...]
    submecanismo: tuple[int, ...]


@dataclass(frozen=True)
class _ResultadoParticionK:
    perdida: float
    distribucion: np.ndarray
    asignacion: tuple[int, ...]
    nodos: list[int]


class Circuito(SIA):
    """Estrategia espectral causal con pesos MI y búsqueda multi-arranque.

    Construye W[i][j] = I(Xi(t+1); Xj(t)) (información mutua dirigida) y deriva
    dos Laplacianos complementarios: L_sal ponderado por grados de salida (D_out)
    y L_ent por grados de entrada (D_in). Sus eigenvectores generan particiones
    candidatas que cubren la asimetría causal. El multi-arranque refina los mejores
    candidatos con EMD exacto y retorna el mínimo global encontrado.
    """

    def __init__(self, tpm: np.ndarray, config=None) -> None:
        super().__init__(tpm, config)
        self.distancia_metrica = seleccionar_emd(config)
        self._max_iter_refinamiento = 24
        self._n_arranques = 4
        self._n_aleatorios = 4
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

        alcance_total = tuple(int(v) for v in self.sia_subsistema.indices_ncubos.tolist())
        mecanismo_total = tuple(int(v) for v in self.sia_subsistema.dims_ncubos.tolist())

        if not alcance_total and not mecanismo_total:
            return Solucion(
                estrategia=CIRCUITO_LABEL,
                perdida=0.0,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=self.sia_dists_marginales.copy(),
                estado_inicial=estado_inicial,
                particion="NO-PARTITION",
            )

        nodos = sorted(set(alcance_total) | set(mecanismo_total))

        if k > 2:
            mejor_k = self._resolver_k(nodos, alcance_total, mecanismo_total, k)
            return Solucion(
                estrategia=CIRCUITO_LABEL,
                perdida=mejor_k.perdida,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=mejor_k.distribucion,
                estado_inicial=estado_inicial,
                particion=fmt_k_particion_asignacion(
                    mejor_k.nodos,
                    mejor_k.asignacion,
                    alcance_total,
                    mecanismo_total,
                ),
            )

        mejor = self._resolver_biparticion(nodos, alcance_total, mecanismo_total)
        return Solucion(
            estrategia=CIRCUITO_LABEL,
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

    # ------------------------------------------------------------------
    # Pesos por Información Mutua Dirigida
    # ------------------------------------------------------------------

    def _construir_pesos(self, nodos: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Construye dos matrices de peso en un solo recorrido de ncubos.

        Returns
        -------
        W_mi  : W[i][j] = I(Xi(t+1); Xj(t)) — dirigida, sin simetrizar.
        W_sen : W[i][j] = |P(Xi=1|Xj=0) - P(Xi=1|Xj=1)| — sensibilidad
                simetrizada (como v1) para conservar cobertura original.
        """
        assert self.sia_subsistema is not None
        n = len(nodos)
        nodo_a_idx = {nodo: idx for idx, nodo in enumerate(nodos)}
        W_mi  = np.zeros((n, n), dtype=np.float64)
        W_sen = np.zeros((n, n), dtype=np.float64)

        for cubo in self.sia_subsistema.ncubos:
            i = int(cubo.indice)
            if i not in nodo_a_idx:
                continue
            idx_i = nodo_a_idx[i]
            for dim in cubo.dims.tolist():
                j = int(dim)
                if j not in nodo_a_idx:
                    continue
                idx_j = nodo_a_idx[j]
                mi, sen = self._pesos_dim(cubo, j)
                W_mi[idx_i][idx_j] += mi
                # Sensibilidad simetrizada — igual que v1
                W_sen[idx_i][idx_j] += sen
                W_sen[idx_j][idx_i] += sen

        return W_mi, W_sen

    def _pesos_dim(self, cubo, dim_nodo: int) -> tuple[float, float]:
        """Calcula MI y sensibilidad para un par (Xi, Xj) con slicing numpy."""
        dims = cubo.dims.tolist()
        if dim_nodo not in dims:
            return 0.0, 0.0

        pos = dims.index(dim_nodo)
        data = cubo.data

        idx_0 = [slice(None)] * len(dims)
        idx_0[pos] = 0
        idx_1 = [slice(None)] * len(dims)
        idx_1[pos] = 1

        d0 = np.asarray(data[tuple(idx_0)])
        d1 = np.asarray(data[tuple(idx_1)])
        p1_xj0 = float(d0.mean())
        p1_xj1 = float(d1.mean())

        # mean(|P(Xi=1|Xj=0,s) - P(Xi=1|Xj=1,s)|) — igual que v1 pero vectorizado
        sen = float(np.abs(d0 - d1).mean())

        # Información mutua I(Xi(t+1); Xj(t))
        p_xi1 = 0.5 * (p1_xj0 + p1_xj1)
        p_xi0 = 1.0 - p_xi1
        eps = 1e-15
        mi = 0.0
        for p_xj, p1_g in ((0.5, p1_xj0), (0.5, p1_xj1)):
            p0_g = 1.0 - p1_g
            p11 = p1_g * p_xj
            p01 = p0_g * p_xj
            if p11 > eps and p_xi1 > eps:
                mi += p11 * np.log2(p11 / (p_xi1 * p_xj + eps))
            if p01 > eps and p_xi0 > eps:
                mi += p01 * np.log2(p01 / (p_xi0 * p_xj + eps))

        return max(0.0, mi), sen

    def _construir_pesos_mi(self, nodos: list[int]) -> np.ndarray:
        """Conveniencia: devuelve solo W_mi (compatibilidad con _resolver_k)."""
        return self._construir_pesos(nodos)[0]

    def _info_mutua_dirigida(self, cubo, dim_nodo: int) -> float:
        """I(Xi(t+1); Xj(t)) bajo prior uniforme sobre los demás nodos.

        Usa slicing numpy: separa estados con Xj=0 y Xj=1, promedia sobre
        los demás dims. Más rápido que el loop Python de la versión anterior
        y captura dependencia estadística total (no solo diferencia marginal).
        """
        dims = cubo.dims.tolist()
        if dim_nodo not in dims:
            return 0.0

        pos = dims.index(dim_nodo)
        data = cubo.data

        idx_0 = [slice(None)] * len(dims)
        idx_0[pos] = 0
        idx_1 = [slice(None)] * len(dims)
        idx_1[pos] = 1

        p1_xj0 = float(np.asarray(data[tuple(idx_0)]).mean())
        p1_xj1 = float(np.asarray(data[tuple(idx_1)]).mean())

        p_xi1 = 0.5 * (p1_xj0 + p1_xj1)
        p_xi0 = 1.0 - p_xi1

        eps = 1e-15
        mi = 0.0
        for p_xj, p1_g in ((0.5, p1_xj0), (0.5, p1_xj1)):
            p0_g = 1.0 - p1_g
            p11 = p1_g * p_xj
            p01 = p0_g * p_xj
            if p11 > eps and p_xi1 > eps:
                mi += p11 * np.log2(p11 / (p_xi1 * p_xj + eps))
            if p01 > eps and p_xi0 > eps:
                mi += p01 * np.log2(p01 / (p_xi0 * p_xj + eps))

        return max(0.0, mi)

    # ------------------------------------------------------------------
    # Laplacianos dirigidos
    # ------------------------------------------------------------------

    def _laplaciano_salida(self, W: np.ndarray) -> np.ndarray:
        """L_sal = D_out - Wsym.  D_out[i] = influencia que i emite.

        Nodos con alto grado de salida (fuentes causales) tienen mayor peso
        diagonal. El espectro identifica particiones que aíslan fuentes.
        """
        W_sym = 0.5 * (W + W.T)
        return np.diag(W.sum(axis=1)) - W_sym

    def _laplaciano_entrada(self, W: np.ndarray) -> np.ndarray:
        """L_ent = D_in - Wsym.  D_in[i] = influencia que i recibe.

        Complementario a L_sal: cuando la red es asimétrica (caso general
        en IIT), L_ent genera eigenvectores distintos a L_sal, ampliando
        la diversidad de candidatos espectrales.
        """
        W_sym = 0.5 * (W + W.T)
        return np.diag(W.sum(axis=0)) - W_sym

    def _laplaciano(self, W: np.ndarray) -> np.ndarray:
        """Laplaciano simétrico estándar — usado por el hipergrafo."""
        W_sym = 0.5 * (W + W.T)
        return np.diag(W_sym.sum(axis=1)) - W_sym

    # ------------------------------------------------------------------
    # Circuitos causales (DFS limitado) y Laplaciano de hipergrafo
    # ------------------------------------------------------------------

    _MAX_CIRCUITOS = 4000

    def _encontrar_circuitos(
        self,
        n: int,
        W: np.ndarray,
        umbral: float = 0.0,
    ) -> list[tuple[list[int], float]]:
        """Circuitos simples del grafo dirigido W > umbral (Johnson simplificado).

        Limitado a n<=14 y _MAX_CIRCUITOS para evitar explosion combinatoria
        en grafos densos.
        """
        if n > 14:
            return []

        adj = [[W[i][j] if W[i][j] > umbral else 0.0 for j in range(n)] for i in range(n)]
        circuitos: list[tuple[list[int], float]] = []
        visitado = [False] * n

        def dfs(inicio: int, actual: int, camino: list[int], peso: float) -> None:
            if len(circuitos) >= self._MAX_CIRCUITOS:
                return
            for vecino in range(n):
                if len(circuitos) >= self._MAX_CIRCUITOS:
                    return
                w = adj[actual][vecino]
                if w == 0.0:
                    continue
                if vecino == inicio and len(camino) >= 2:
                    if peso * w > 0.0:
                        circuitos.append((list(camino), peso * w))
                elif vecino > inicio and not visitado[vecino]:
                    visitado[vecino] = True
                    camino.append(vecino)
                    dfs(inicio, vecino, camino, peso * w)
                    camino.pop()
                    visitado[vecino] = False

        for s in range(n):
            if len(circuitos) >= self._MAX_CIRCUITOS:
                break
            visitado[s] = True
            dfs(s, s, [s], 1.0)
            visitado[s] = False

        return circuitos

    def _laplaciano_hipergrafo(
        self,
        n: int,
        circuitos: list[tuple[list[int], float]],
    ) -> np.ndarray:
        """Laplaciano del hipergrafo de circuitos causales (L_H = D_v - H W H^T)."""
        m = len(circuitos)
        if m == 0:
            return np.zeros((n, n), dtype=np.float64)

        H = np.zeros((n, m), dtype=np.float64)
        for c_idx, (ciclo, fuerza) in enumerate(circuitos):
            for nodo_idx in ciclo:
                H[nodo_idx, c_idx] = 1.0

        W_diag = np.array([fuerza for _, fuerza in circuitos], dtype=np.float64)
        HW = H * W_diag[np.newaxis, :]
        HWHT = HW @ H.T
        return np.diag(HWHT.sum(axis=1)) - HWHT

    # ------------------------------------------------------------------
    # Bipartición espectral multi-arranque
    # ------------------------------------------------------------------

    def _resolver_biparticion(
        self,
        nodos: list[int],
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
    ) -> _ResultadoParticion:
        """Evalúa candidatos espectrales + aleatorios; refina los mejores con EMD."""
        assert self.sia_dists_marginales is not None

        candidatos = self._candidatos_espectrales(nodos, alcance_total, mecanismo_total)
        candidatos += self._candidatos_aleatorios(nodos, alcance_total, mecanismo_total)

        evaluados: list[tuple[float, np.ndarray, tuple[int, ...], tuple[int, ...]]] = []
        for sa, sm in candidatos:
            perd, dist = self._evaluar_particion(sa, sm)
            evaluados.append((perd, dist, sa, sm))
        evaluados.sort(key=lambda x: x[0])

        mejor = _ResultadoParticion(
            perdida=float("inf"),
            distribucion=self.sia_dists_marginales.copy(),
            subalcance=(),
            submecanismo=(),
        )
        for perd, dist, sa, sm in evaluados[: self._n_arranques]:
            inicio = _ResultadoParticion(perdida=perd, distribucion=dist, subalcance=sa, submecanismo=sm)
            res = self._refinar_local(inicio, alcance_total, mecanismo_total)
            if res.perdida < mejor.perdida:
                mejor = res

        # Always refine single-node fallback candidates regardless of their initial EMD,
        # so refinement can reach asymmetric cuts like (sa=(x,), sm=()) that the
        # spectral/random sweeps miss (eigenvectors generate symmetric node splits).
        for sa, sm in self._candidatos_fallback(nodos, alcance_total, mecanismo_total):
            perd, dist = self._evaluar_particion(sa, sm)
            inicio = _ResultadoParticion(perdida=perd, distribucion=dist, subalcance=sa, submecanismo=sm)
            res = self._refinar_local(inicio, alcance_total, mecanismo_total)
            if res.perdida < mejor.perdida:
                mejor = res

        return mejor

    def _candidatos_espectrales(
        self,
        nodos: list[int],
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
    ) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """Candidatos desde eigenvectores de L_sal, L_ent y L_hipergrafo."""
        n = len(nodos)
        if n == 1:
            nodo = nodos[0]
            return [
                (tuple(v for v in alcance_total if v == nodo), ()),
                ((), tuple(v for v in mecanismo_total if v == nodo)),
            ]

        W_mi, W_sen = self._construir_pesos(nodos)
        # Circuitos con W_sen (simétrica, densa) para reproducir cobertura de v1.
        circuitos = self._encontrar_circuitos(n, W_sen, umbral=0.0)

        candidatos: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        vistos: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

        def _sweep(L: np.ndarray) -> None:
            try:
                eigenvalores, eigenvectores = np.linalg.eigh(L)
            except np.linalg.LinAlgError:
                return
            orden = np.argsort(eigenvalores)
            for ev_idx in range(1, min(4, n)):
                ev = eigenvectores[:, orden[ev_idx]]
                valores_unicos = np.unique(ev)
                umbrales = [
                    (valores_unicos[i] + valores_unicos[i + 1]) / 2.0
                    for i in range(len(valores_unicos) - 1)
                ]
                if not umbrales:
                    umbrales = [0.0]
                for thr in umbrales:
                    for grupo_set in (
                        {idx for idx in range(n) if ev[idx] <= thr},
                        {idx for idx in range(n) if ev[idx] > thr},
                    ):
                        if not grupo_set or len(grupo_set) == n:
                            continue
                        grupo_nodos = {nodos[idx] for idx in grupo_set}
                        clave = (
                            tuple(v for v in alcance_total if v in grupo_nodos),
                            tuple(v for v in mecanismo_total if v in grupo_nodos),
                        )
                        if clave not in vistos and (clave[0] or clave[1]):
                            vistos.add(clave)
                            candidatos.append(clave)

        if circuitos:
            L_h = self._laplaciano_hipergrafo(n, circuitos)
            if not np.allclose(L_h, 0):
                _sweep(L_h)

        # Cuatro Laplacianos: sen_sim (cobertura v1), mi_sim, mi_sal, mi_ent.
        _sweep(self._laplaciano(W_sen))       # sensibilidad simetrizada — cobertura original
        _sweep(self._laplaciano(W_mi))        # MI simétrica
        _sweep(self._laplaciano_salida(W_mi)) # MI + grados de salida
        _sweep(self._laplaciano_entrada(W_mi)) # MI + grados de entrada

        return candidatos or self._candidatos_fallback(nodos, alcance_total, mecanismo_total)

    def _candidatos_aleatorios(
        self,
        nodos: list[int],
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
    ) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """Particiones aleatorias de tamaño variado para diversificar la búsqueda."""
        n = len(nodos)
        if n < 2:
            return []
        rng = np.random.default_rng(0xCAFE)
        candidatos: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        vistos: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        for _ in range(self._n_aleatorios):
            k = max(1, int(rng.integers(1, n)))
            grupo = set(rng.choice(nodos, size=k, replace=False).tolist())
            sa = tuple(v for v in alcance_total if v in grupo)
            sm = tuple(v for v in mecanismo_total if v in grupo)
            if not sa and not sm:
                continue
            if sa == alcance_total and sm == mecanismo_total:
                continue
            clave = (sa, sm)
            if clave not in vistos:
                vistos.add(clave)
                candidatos.append(clave)
        return candidatos

    def _candidatos_fallback(
        self,
        nodos: list[int],
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
    ) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """Cortes triviales de un nodo como emergencia."""
        candidatos = []
        for nodo in nodos:
            sa = tuple(v for v in alcance_total if v == nodo)
            sm = tuple(v for v in mecanismo_total if v == nodo)
            if sa or sm:
                candidatos.append((sa, sm))
        return candidatos or [((alcance_total[0],) if alcance_total else (), ())]

    # ------------------------------------------------------------------
    # K-partición espectral multi-arranque
    # ------------------------------------------------------------------

    def _resolver_k(
        self,
        nodos: list[int],
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
        k: int,
    ) -> _ResultadoParticionK:
        W_mi, W_sen = self._construir_pesos(nodos)
        n = len(nodos)
        k_eff = min(k, n)

        circuitos = self._encontrar_circuitos(n, W_mi, umbral=0.0)
        L_h = self._laplaciano_hipergrafo(n, circuitos) if circuitos else None

        assert self.sia_dists_marginales is not None
        mejor = _ResultadoParticionK(
            perdida=float("inf"),
            distribucion=self.sia_dists_marginales.copy(),
            asignacion=(0,) * n,
            nodos=nodos,
        )

        laplacianos = [
            self._laplaciano(W_sen),
            self._laplaciano(W_mi),
            self._laplaciano_salida(W_mi),
            self._laplaciano_entrada(W_mi),
        ]
        if L_h is not None and not np.allclose(L_h, 0):
            laplacianos.insert(0, L_h)

        for L in laplacianos:
            for seed in (42, 7):
                asig = self._embedding_k(L, n, k_eff, semilla=seed)
                perd, dist = self._evaluar_k_particion(asig, nodos)
                res = _ResultadoParticionK(perdida=perd, distribucion=dist, asignacion=asig, nodos=nodos)
                res = self._refinar_k_local(res, k_eff, nodos)
                if res.perdida < mejor.perdida:
                    mejor = res

        rng = np.random.default_rng(0xBEEF)
        for _ in range(3):
            asig = self._canonicalizar_asignacion(
                tuple(int(rng.integers(0, k_eff)) for _ in range(n))
            )
            if len(set(asig)) < 2:
                continue
            perd, dist = self._evaluar_k_particion(asig, nodos)
            res = _ResultadoParticionK(perdida=perd, distribucion=dist, asignacion=asig, nodos=nodos)
            res = self._refinar_k_local(res, k_eff, nodos)
            if res.perdida < mejor.perdida:
                mejor = res

        return mejor

    def _embedding_k(self, L: np.ndarray, n: int, k: int, semilla: int = 42) -> tuple[int, ...]:
        """Proyecta nodos en los k primeros eigenvectores y aplica k-means."""
        if n <= 1:
            return (0,) * n
        try:
            eigenvalores, eigenvectores = np.linalg.eigh(L)
        except np.linalg.LinAlgError:
            return self._canonicalizar_asignacion(tuple(i % k for i in range(n)))

        orden = np.argsort(eigenvalores)
        embedding = eigenvectores[:, orden[1:k]]

        asignacion = self._kmeans(embedding, k, semilla=semilla)
        asignacion = self._canonicalizar_asignacion(asignacion)
        if len(set(asignacion)) < 2:
            asignacion = self._canonicalizar_asignacion(tuple(i % k for i in range(n)))
        return asignacion

    def _kmeans(self, X: np.ndarray, k: int, max_iter: int = 60, semilla: int = 42) -> tuple[int, ...]:
        """K-means++ sobre las filas de X."""
        n = X.shape[0]
        if k >= n:
            return tuple(range(n))

        rng = np.random.default_rng(semilla)
        centros_idx = [int(rng.integers(0, n))]
        for _ in range(k - 1):
            dists_min = np.min(
                np.stack([np.sum((X - X[c]) ** 2, axis=1) for c in centros_idx]), axis=0
            )
            dists_min[centros_idx] = 0.0
            total = dists_min.sum()
            if total == 0.0:
                break
            centros_idx.append(int(rng.choice(n, p=dists_min / total)))

        centros = X[centros_idx].copy()
        etiquetas = np.zeros(n, dtype=np.int32)

        for _ in range(max_iter):
            dists = np.stack([np.sum((X - c) ** 2, axis=1) for c in centros])
            nuevas = np.argmin(dists, axis=0).astype(np.int32)
            if np.all(nuevas == etiquetas):
                break
            etiquetas = nuevas
            for cluster in range(len(centros_idx)):
                miembros = X[etiquetas == cluster]
                if len(miembros) > 0:
                    centros[cluster] = miembros.mean(axis=0)

        return tuple(int(e) for e in etiquetas.tolist())

    # ------------------------------------------------------------------
    # Evaluación de particiones
    # ------------------------------------------------------------------

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
        distribucion = self._alinear(sistema_partido.distribucion_marginal(), self.sia_dists_marginales)
        perdida = float(self.distancia_metrica(self.sia_dists_marginales, distribucion))
        self._cache_particiones[clave] = (perdida, distribucion)
        return perdida, distribucion

    def _evaluar_k_particion(
        self,
        asignacion: tuple[int, ...],
        nodos: list[int],
    ) -> tuple[float, np.ndarray]:
        en_cache = self._cache_k_particiones.get(asignacion)
        if en_cache is not None:
            return en_cache

        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None

        sistema_partido = self.sia_subsistema.k_bipartir(nodos, asignacion)
        distribucion = self._alinear(sistema_partido.distribucion_marginal(), self.sia_dists_marginales)
        perdida = float(self.distancia_metrica(self.sia_dists_marginales, distribucion))
        self._cache_k_particiones[asignacion] = (perdida, distribucion)
        return perdida, distribucion

    def _alinear(self, distribucion: np.ndarray, referencia: np.ndarray) -> np.ndarray:
        if distribucion.size == referencia.size:
            return distribucion
        salida = np.zeros_like(referencia)
        salida[: distribucion.size] = distribucion
        return salida

    # ------------------------------------------------------------------
    # Refinamiento local
    # ------------------------------------------------------------------

    def _refinar_local(
        self,
        inicio: _ResultadoParticion,
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
    ) -> _ResultadoParticion:
        actual = inicio
        for _ in range(self._max_iter_refinamiento):
            vecinos = self._vecinos(actual.subalcance, actual.submecanismo, alcance_total, mecanismo_total)
            if not vecinos:
                break
            mejor_vecino = actual
            for sa, sm in vecinos:
                perd, dist = self._evaluar_particion(sa, sm)
                if perd < mejor_vecino.perdida:
                    mejor_vecino = _ResultadoParticion(perdida=perd, distribucion=dist, subalcance=sa, submecanismo=sm)
            if mejor_vecino.perdida + 1e-12 >= actual.perdida:
                break
            actual = mejor_vecino
        return actual

    def _vecinos(
        self,
        subalcance: tuple[int, ...],
        submecanismo: tuple[int, ...],
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
    ) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        vecinos: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        vistos: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

        def agregar(ca: tuple[int, ...], cm: tuple[int, ...]) -> None:
            if not ca and not cm:
                return
            if ca == alcance_total and cm == mecanismo_total:
                return
            clave = (ca, cm)
            if clave not in vistos:
                vistos.add(clave)
                vecinos.append(clave)

        for nodo in alcance_total:
            nuevo = set(subalcance)
            nuevo.discard(nodo) if nodo in nuevo else nuevo.add(nodo)
            agregar(tuple(v for v in alcance_total if v in nuevo), submecanismo)

        for nodo in mecanismo_total:
            nuevo = set(submecanismo)
            nuevo.discard(nodo) if nodo in nuevo else nuevo.add(nodo)
            agregar(subalcance, tuple(v for v in mecanismo_total if v in nuevo))

        return vecinos

    def _refinar_k_local(
        self,
        inicio: _ResultadoParticionK,
        k: int,
        nodos: list[int],
    ) -> _ResultadoParticionK:
        actual = inicio
        mejor_global = inicio
        for _ in range(self._max_iter_refinamiento):
            vecinos = self._vecinos_k(actual.asignacion, k)
            if not vecinos:
                break
            mejor_vecino = actual
            for asig in vecinos:
                perd, dist = self._evaluar_k_particion(asig, nodos)
                if perd < mejor_vecino.perdida:
                    mejor_vecino = _ResultadoParticionK(perdida=perd, distribucion=dist, asignacion=asig, nodos=nodos)
            if mejor_vecino.perdida + 1e-12 >= actual.perdida:
                break
            actual = mejor_vecino
            if actual.perdida < mejor_global.perdida:
                mejor_global = actual
        return mejor_global

    def _vecinos_k(self, asignacion: tuple[int, ...], k: int) -> list[tuple[int, ...]]:
        vecinos: list[tuple[int, ...]] = []
        vistos: set[tuple[int, ...]] = set()
        for idx in range(len(asignacion)):
            for nuevo_grupo in range(k):
                if nuevo_grupo == asignacion[idx]:
                    continue
                nueva = list(asignacion)
                nueva[idx] = nuevo_grupo
                canon = self._canonicalizar_asignacion(tuple(nueva))
                if len(set(canon)) < 2 or canon in vistos:
                    continue
                vistos.add(canon)
                vecinos.append(canon)
        return vecinos

    @staticmethod
    def _canonicalizar_asignacion(asignacion: tuple[int, ...]) -> tuple[int, ...]:
        mapa: dict[int, int] = {}
        siguiente = 0
        resultado = []
        for g in asignacion:
            if g not in mapa:
                mapa[g] = siguiente
                siguiente += 1
            resultado.append(mapa[g])
        return tuple(resultado)


ElectricNetwork = Circuito

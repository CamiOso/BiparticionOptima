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
    """Estrategia espectral inspirada en redes electricas.

    Modela el sistema como un circuito donde las conductancias representan
    el acoplamiento entre nodos segun las probabilidades de transicion (TPM).
    Construye el Laplaciano del grafo y usa sus eigenvectores para proponer
    particiones de baja perdida en O(n^3):

    - k=2: vector de Fiedler (segundo eigenvector) con multiples umbrales.
    - k>2: embedding espectral con los k primeros eigenvectores + k-means.

    En ambos casos se aplica refinamiento local para mejorar el resultado.
    """

    def __init__(self, tpm: np.ndarray, config=None) -> None:
        super().__init__(tpm, config)
        self.distancia_metrica = seleccionar_emd(config)
        self._max_iter_refinamiento = 24
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
    # Red electrica: conductancias y Laplaciano
    # ------------------------------------------------------------------

    def _construir_conductancias(self, nodos: list[int]) -> np.ndarray:
        """W[i][j] = sensibilidad promedio de nodo i al estado del nodo j.

        Se calcula como la variacion media de P(X_i=1|X(t)) al cambiar el
        bit de X_j, promediada sobre todos los estados de los demas nodos.
        La simetrizacion W += W.T asegura un Laplaciano no dirigido.
        """
        assert self.sia_subsistema is not None
        n = len(nodos)
        nodo_a_idx = {nodo: idx for idx, nodo in enumerate(nodos)}
        W = np.zeros((n, n), dtype=np.float64)

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
                conductancia = self._sensibilidad(cubo, j)
                # Acumula en ambas direcciones para simetrizar.
                W[idx_i][idx_j] += conductancia
                W[idx_j][idx_i] += conductancia

        return W

    def _sensibilidad(self, cubo, dim_nodo: int) -> float:
        """Diferencia finita: |P(i=1|j=0,...) - P(i=1|j=1,...)| promediada."""
        dims = cubo.dims.tolist()
        if dim_nodo not in dims:
            return 0.0

        pos = dims.index(dim_nodo)
        data = cubo.data
        n_dims = len(dims)

        if n_dims == 1:
            return abs(float(data[0]) - float(data[1]))

        otras_dims = [d for d in range(n_dims) if d != pos]
        otras_sizes = [data.shape[d] for d in otras_dims]
        n_otras = 1
        for s in otras_sizes:
            n_otras *= s

        total = 0.0
        for estado_idx in range(n_otras):
            idx_otras = []
            temp = estado_idx
            for s in reversed(otras_sizes):
                idx_otras.append(temp % s)
                temp //= s
            idx_otras = list(reversed(idx_otras))

            idx_0 = [0] * n_dims
            idx_1 = [0] * n_dims
            idx_0[pos] = 0
            idx_1[pos] = 1
            for k_ot, d_ot in enumerate(otras_dims):
                idx_0[d_ot] = idx_otras[k_ot]
                idx_1[d_ot] = idx_otras[k_ot]

            total += abs(float(data[tuple(idx_0)]) - float(data[tuple(idx_1)]))

        return total / n_otras

    def _laplaciano(self, W: np.ndarray) -> np.ndarray:
        """L = D - W  (D diagonal de grados = suma de conductancias por fila)."""
        return np.diag(W.sum(axis=1)) - W

    # ------------------------------------------------------------------
    # Biparticion espectral (k = 2)
    # ------------------------------------------------------------------

    def _resolver_biparticion(
        self,
        nodos: list[int],
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
    ) -> _ResultadoParticion:
        candidatos = self._candidatos_fiedler(nodos, alcance_total, mecanismo_total)

        assert self.sia_dists_marginales is not None
        mejor = _ResultadoParticion(
            perdida=float("inf"),
            distribucion=self.sia_dists_marginales.copy(),
            subalcance=(),
            submecanismo=(),
        )

        for subalcance, submecanismo in candidatos:
            perdida, distribucion = self._evaluar_particion(subalcance, submecanismo)
            if perdida < mejor.perdida:
                mejor = _ResultadoParticion(
                    perdida=perdida,
                    distribucion=distribucion,
                    subalcance=subalcance,
                    submecanismo=submecanismo,
                )

        return self._refinar_local(mejor, alcance_total, mecanismo_total)

    def _candidatos_fiedler(
        self,
        nodos: list[int],
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
    ) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """Genera candidatos barriendo umbrales sobre el vector de Fiedler.

        Tambien explora los eigenvectores 3 y 4 para mayor robustez ante
        sistemas con multiples puntos de corte de costo similar.
        """
        n = len(nodos)
        if n == 1:
            nodo = nodos[0]
            return [
                (tuple(v for v in alcance_total if v == nodo), ()),
                ((), tuple(v for v in mecanismo_total if v == nodo)),
            ]

        W = self._construir_conductancias(nodos)
        L = self._laplaciano(W)

        try:
            eigenvalores, eigenvectores = np.linalg.eigh(L)
        except np.linalg.LinAlgError:
            return self._candidatos_fallback(nodos, alcance_total, mecanismo_total)

        orden = np.argsort(eigenvalores)
        candidatos: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        vistos: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

        def agregar_desde_vector(ev: np.ndarray) -> None:
            valores_unicos = np.unique(ev)
            umbrales = [
                (valores_unicos[i] + valores_unicos[i + 1]) / 2.0
                for i in range(len(valores_unicos) - 1)
            ]
            if not umbrales:
                umbrales = [0.0]

            for umbral in umbrales:
                for grupo0_set in (
                    {idx for idx in range(n) if ev[idx] <= umbral},
                    {idx for idx in range(n) if ev[idx] > umbral},
                ):
                    if not grupo0_set or len(grupo0_set) == n:
                        continue
                    grupo0_nodos = {nodos[idx] for idx in grupo0_set}
                    clave = (
                        tuple(v for v in alcance_total if v in grupo0_nodos),
                        tuple(v for v in mecanismo_total if v in grupo0_nodos),
                    )
                    if clave not in vistos and (clave[0] or clave[1]):
                        vistos.add(clave)
                        candidatos.append(clave)

        # Vector de Fiedler (indice 1) y los siguientes dos para robustez.
        for ev_idx in range(1, min(4, n)):
            agregar_desde_vector(eigenvectores[:, orden[ev_idx]])

        return candidatos or self._candidatos_fallback(nodos, alcance_total, mecanismo_total)

    def _candidatos_fallback(
        self,
        nodos: list[int],
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
    ) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """Cortes triviales de un nodo como emergencia."""
        candidatos = []
        for nodo in nodos:
            subalcance = tuple(v for v in alcance_total if v == nodo)
            submecanismo = tuple(v for v in mecanismo_total if v == nodo)
            if subalcance or submecanismo:
                candidatos.append((subalcance, submecanismo))
        return candidatos or [((alcance_total[0],) if alcance_total else (), ())]

    # ------------------------------------------------------------------
    # K-particion espectral (k >= 3)
    # ------------------------------------------------------------------

    def _resolver_k(
        self,
        nodos: list[int],
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
        k: int,
    ) -> _ResultadoParticionK:
        W = self._construir_conductancias(nodos)
        L = self._laplaciano(W)
        n = len(nodos)
        k_eff = min(k, n)

        asignacion = self._embedding_k(L, n, k_eff)
        perdida, distribucion = self._evaluar_k_particion(asignacion, nodos)

        assert self.sia_dists_marginales is not None
        mejor = _ResultadoParticionK(
            perdida=perdida,
            distribucion=distribucion,
            asignacion=asignacion,
            nodos=nodos,
        )
        return self._refinar_k_local(mejor, k_eff, nodos)

    def _embedding_k(self, L: np.ndarray, n: int, k: int) -> tuple[int, ...]:
        """Proyecta nodos en los k primeros eigenvectores y aplica k-means."""
        if n <= 1:
            return (0,) * n

        try:
            eigenvalores, eigenvectores = np.linalg.eigh(L)
        except np.linalg.LinAlgError:
            return self._canonicalizar_asignacion(tuple(i % k for i in range(n)))

        orden = np.argsort(eigenvalores)
        # Eigenvectores 1..k (ignora el trivial 0).
        embedding = eigenvectores[:, orden[1:k]]

        asignacion = self._kmeans(embedding, k)
        asignacion = self._canonicalizar_asignacion(asignacion)

        if len(set(asignacion)) < 2:
            asignacion = self._canonicalizar_asignacion(tuple(i % k for i in range(n)))

        return asignacion

    def _kmeans(self, X: np.ndarray, k: int, max_iter: int = 60) -> tuple[int, ...]:
        """K-means con inicializacion kmeans++ sobre las filas de X."""
        n = X.shape[0]
        if k >= n:
            return tuple(range(n))

        rng = np.random.default_rng(42)
        centros_idx = [int(rng.integers(0, n))]
        for _ in range(k - 1):
            dists_min = np.min(
                np.stack([np.sum((X - X[c]) ** 2, axis=1) for c in centros_idx]),
                axis=0,
            )
            dists_min[centros_idx] = 0.0
            total = dists_min.sum()
            if total == 0.0:
                break
            probs = dists_min / total
            centros_idx.append(int(rng.choice(n, p=probs)))

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
    # Evaluacion de particiones
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
            for asignacion_vec in vecinos:
                perdida, distribucion = self._evaluar_k_particion(asignacion_vec, nodos)
                if perdida < mejor_vecino.perdida:
                    mejor_vecino = _ResultadoParticionK(
                        perdida=perdida,
                        distribucion=distribucion,
                        asignacion=asignacion_vec,
                        nodos=nodos,
                    )
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


# Alias retrocompatible.
ElectricNetwork = Circuito

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.constantes.models import HIPERBOLICA_LABEL
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


class ParticionHiperbolica(SIA):
    """Partición usando geometría hiperbólica — analogía con la holografía.

    Conexión con la Relatividad General
    ------------------------------------
    La fórmula de Ryu-Takayanagi (2006) establece que en teoría de campos
    conforme con dual holográfico (AdS/CFT), la entropía de entrelazamiento
    de una región A es proporcional al ÁREA de la superficie geodésica mínima
    en el espacio AdS (Anti-de Sitter) que ancla en el borde de A:

        S(A) = Área(γ_A) / 4·G_N

    El espacio AdS en (2+1)D es isométrico al disco de Poincaré (espacio
    hiperbólico 2D). El MIP-IIT es el análogo discreto de este principio:
    encontrar la "frontera geodésica mínima" que separa el sistema en dos
    partes con mínima pérdida de información.

    Algoritmo
    ---------
    1. Construir conductancias W y Laplaciano L de conductancias.
    2. Usar los dos primeros eigenvectores no triviales de L como coordenadas
       (x, y) y proyectar al disco de Poincaré mediante:
           coords_h = coords · tanh(α·‖coords‖) / ‖coords‖
       La proyección concentra nodos fuertemente conectados cerca del
       centro y nodos periféricos hacia el borde — igual que en AdS.
    3. Barrer dos familias de geodésicas:
       a. Diametrales: líneas rectas a través del origen, parametrizadas
          por ángulo θ ∈ [0, π). Análogo al plano de corte euclídeo.
       b. Circulares: para cada par de nodos (zi, zj), calcular la
          geodésica única del disco de Poincaré que pasa por ambos
          (inversión de Möbius T_{z_i}: z → (z-z_i)/(1-z̄_i·z),
          luego rotar al eje real). El signo de Im[T(z_k)] clasifica
          los demás nodos a cada lado — exactamente la partición.
    4. Evaluar todos los candidatos con bipartir() y tomar el mínimo.
    5. Refinamiento local (flip de un nodo a la vez).
    """

    def __init__(
        self,
        tpm: np.ndarray,
        config=None,
        n_angulos: int = 32,
        n_random_restarts: int = 8,
    ) -> None:
        super().__init__(tpm, config)
        self.distancia_metrica = seleccionar_emd(config)
        self.n_angulos = n_angulos
        self.n_random_restarts = n_random_restarts
        self._max_iter_refinamiento = 24
        self._cache_particiones: dict[tuple, tuple[float, np.ndarray]] = {}
        self._cache_k_particiones: dict[tuple, tuple[float, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

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

        alc_total = tuple(int(v) for v in self.sia_subsistema.indices_ncubos.tolist())
        mec_total = tuple(int(v) for v in self.sia_subsistema.dims_ncubos.tolist())

        if not alc_total and not mec_total:
            return Solucion(
                estrategia=HIPERBOLICA_LABEL,
                perdida=0.0,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=self.sia_dists_marginales.copy(),
                estado_inicial=estado_inicial,
                particion="NO-PARTITION",
            )

        nodos = sorted(set(alc_total) | set(mec_total))

        if k > 2:
            mejor_k = self._resolver_k(nodos, alc_total, mec_total, k)
            return Solucion(
                estrategia=HIPERBOLICA_LABEL,
                perdida=mejor_k.perdida,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=mejor_k.distribucion,
                estado_inicial=estado_inicial,
                particion=fmt_k_particion_asignacion(
                    mejor_k.nodos, mejor_k.asignacion, alc_total, mec_total,
                ),
            )

        mejor = self._resolver_biparticion(nodos, alc_total, mec_total)
        # Reinicios aleatorios con objetivo φ para escapar mínimos locales
        mejor = self._multistart_k2(mejor, alc_total, mec_total)
        return Solucion(
            estrategia=HIPERBOLICA_LABEL,
            perdida=mejor.perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=mejor.distribucion,
            estado_inicial=estado_inicial,
            particion=fmt_biparticion(
                mejor.subalcance, mejor.submecanismo, alc_total, mec_total,
            ),
        )

    # ------------------------------------------------------------------
    # Conductancias (igual que Circuito)
    # ------------------------------------------------------------------

    def _construir_conductancias(self, nodos: list[int]) -> np.ndarray:
        assert self.sia_subsistema is not None
        n = len(nodos)
        idx = {nodo: i for i, nodo in enumerate(nodos)}
        W = np.zeros((n, n), dtype=np.float64)
        for cubo in self.sia_subsistema.ncubos:
            i = int(cubo.indice)
            if i not in idx:
                continue
            for dim in cubo.dims.tolist():
                j = int(dim)
                if j not in idx:
                    continue
                c = self._sensibilidad(cubo, j)
                W[idx[i]][idx[j]] += c
                W[idx[j]][idx[i]] += c
        return W

    def _sensibilidad(self, cubo, dim_nodo: int) -> float:
        dims = cubo.dims.tolist()
        if dim_nodo not in dims:
            return 0.0
        pos = dims.index(dim_nodo)
        data = cubo.data
        n_dims = len(dims)
        if n_dims == 1:
            return abs(float(data[0]) - float(data[1]))
        otras = [d for d in range(n_dims) if d != pos]
        sizes = [data.shape[d] for d in otras]
        n_otras = 1
        for s in sizes:
            n_otras *= s
        total = 0.0
        for ei in range(n_otras):
            idx_otras = []
            tmp = ei
            for s in reversed(sizes):
                idx_otras.append(tmp % s)
                tmp //= s
            idx_otras = list(reversed(idx_otras))
            i0 = [0] * n_dims
            i1 = [0] * n_dims
            i0[pos] = 0
            i1[pos] = 1
            for ko, do in enumerate(otras):
                i0[do] = idx_otras[ko]
                i1[do] = idx_otras[ko]
            total += abs(float(data[tuple(i0)]) - float(data[tuple(i1)]))
        return total / n_otras

    # ------------------------------------------------------------------
    # Embedding hiperbólico (disco de Poincaré)
    # ------------------------------------------------------------------

    def _embeber_poincare(
        self, nodos: list[int], W: np.ndarray
    ) -> np.ndarray:
        """Proyecta nodos en el disco de Poincaré.

        Pasos:
        1. Laplaciano L = D - W.
        2. Eigenvectores 1 y 2 (los dos primeros no triviales) dan
           coordenadas euclidianas (x, y) que capturan la estructura del grafo.
        3. Proyección hiperbólica: escala radial con tanh para que nodos
           fuertemente conectados queden cerca del origen (centro AdS) y
           nodos periféricos cerca del borde (frontera conforme).
        """
        n = len(nodos)
        L = np.diag(W.sum(axis=1)) - W

        try:
            vals, vecs = np.linalg.eigh(L)
        except np.linalg.LinAlgError:
            return np.zeros((n, 2))

        orden = np.argsort(vals)

        if n >= 3:
            x = vecs[:, orden[1]]
            y = vecs[:, orden[2]]
        elif n == 2:
            x = vecs[:, orden[1]]
            y = np.zeros(n)
        else:
            return np.zeros((n, 2))

        coords = np.column_stack([x, y])

        # Normalizar para que el máximo radio euclidiano sea 1
        r_max = np.linalg.norm(coords, axis=1).max()
        if r_max < 1e-12:
            return np.zeros((n, 2))
        coords = coords / r_max

        # Proyección hiperbólica: r → tanh(α·r) / r · coords
        # α=2 da buena separación sin acumular todo en el borde
        alpha = 2.0
        r = np.linalg.norm(coords, axis=1, keepdims=True)
        r_safe = np.where(r > 1e-12, r, 1.0)
        coords_h = coords / r_safe * np.tanh(alpha * r)

        # Recortar para garantizar |z| < 1 estrictamente
        r_h = np.linalg.norm(coords_h, axis=1, keepdims=True)
        mask = r_h >= 1.0
        if mask.any():
            coords_h = np.where(mask, coords_h / (r_h + 1e-9) * 0.99, coords_h)

        return coords_h

    # ------------------------------------------------------------------
    # Geodésicas en el disco de Poincaré
    # ------------------------------------------------------------------

    def _clasificar_por_geodesica(
        self,
        coords_h: np.ndarray,
        zi: np.ndarray,
        zj: np.ndarray,
    ) -> np.ndarray | None:
        """Para cada nodo, determina en qué lado de la geodésica (zi, zj) está.

        Usa la transformación de Möbius T_{zi}(z) = (z - zi) / (1 - z̄_i·z)
        que mapea zi → 0. Bajo T, la geodésica que pasa por zi y zj se convierte
        en la geodésica que pasa por 0 y T(zj) = un diámetro.
        El signo de Im[T(z_k) · conj(T(zj)/|T(zj)|)] da el lado de cada nodo.
        """
        zi_c = complex(zi[0], zi[1])
        zj_c = complex(zj[0], zj[1])
        zi_conj = zi_c.conjugate()

        def mobius(z: complex) -> complex:
            d = 1.0 - zi_conj * z
            if abs(d) < 1e-12:
                return complex(1e6, 0.0)
            return (z - zi_c) / d

        t_zj = mobius(zj_c)
        if abs(t_zj) < 1e-9:
            return None

        # Dirección de la geodésica desde el origen hacia t_zj
        direccion = t_zj / abs(t_zj)

        lados = np.empty(len(coords_h))
        for k, p in enumerate(coords_h):
            t_p = mobius(complex(p[0], p[1]))
            # Componente perpendicular a la dirección de la geodésica
            lados[k] = (t_p * direccion.conjugate()).imag

        return lados

    # ------------------------------------------------------------------
    # Generación de candidatos (geodésicas diametrales + circulares)
    # ------------------------------------------------------------------

    def _candidatos_geodesicos(
        self,
        nodos: list[int],
        alc_total: tuple[int, ...],
        mec_total: tuple[int, ...],
    ) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        n = len(nodos)
        if n == 1:
            nodo = nodos[0]
            return [
                (tuple(v for v in alc_total if v == nodo), ()),
                ((), tuple(v for v in mec_total if v == nodo)),
            ]

        W = self._construir_conductancias(nodos)
        coords_h = self._embeber_poincare(nodos, W)

        candidatos: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        vistos: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

        def registrar(grupo0_idx: set[int]) -> None:
            if not grupo0_idx or len(grupo0_idx) == n:
                return
            grupo0_nodos = {nodos[i] for i in grupo0_idx}
            clave = (
                tuple(v for v in alc_total if v in grupo0_nodos),
                tuple(v for v in mec_total if v in grupo0_nodos),
            )
            if clave not in vistos and (clave[0] or clave[1]):
                vistos.add(clave)
                candidatos.append(clave)

        # --- Familia 1: geodésicas DIAMETRALES (líneas a través del origen) ---
        # Parametrizadas por ángulo θ ∈ [0, π): separan z·e^{-iθ} > 0 vs ≤ 0
        for k in range(self.n_angulos):
            theta = k * np.pi / self.n_angulos
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            # Proyección de cada nodo sobre la dirección perpendicular al diámetro
            proj = coords_h[:, 0] * cos_t + coords_h[:, 1] * sin_t
            vals_unicos = np.unique(proj)
            umbrales = (vals_unicos[:-1] + vals_unicos[1:]) / 2.0
            if len(umbrales) == 0:
                umbrales = [0.0]
            for umbral in umbrales:
                registrar({i for i in range(n) if proj[i] <= umbral})
                registrar({i for i in range(n) if proj[i] > umbral})

        # --- Familia 2: geodésicas CIRCULARES (inversión de Möbius) ---
        # Para cada par (i, j) existe una única geodésica del disco de
        # Poincaré que los conecta. Clasificar los demás nodos por lado.
        for i in range(n):
            for j in range(i + 1, n):
                lados = self._clasificar_por_geodesica(
                    coords_h, coords_h[i], coords_h[j]
                )
                if lados is None:
                    continue
                registrar({k for k in range(n) if lados[k] >= 0.0})
                registrar({k for k in range(n) if lados[k] < 0.0})

        return candidatos or self._candidatos_fallback(nodos, alc_total, mec_total)

    def _candidatos_fallback(
        self,
        nodos: list[int],
        alc_total: tuple[int, ...],
        mec_total: tuple[int, ...],
    ) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        candidatos = []
        for nodo in nodos:
            alc = tuple(v for v in alc_total if v == nodo)
            mec = tuple(v for v in mec_total if v == nodo)
            if alc or mec:
                candidatos.append((alc, mec))
        return candidatos or [((alc_total[0],) if alc_total else (), ())]

    # ------------------------------------------------------------------
    # Bipartición espectral (k = 2)
    # ------------------------------------------------------------------

    def _resolver_biparticion(
        self,
        nodos: list[int],
        alc_total: tuple[int, ...],
        mec_total: tuple[int, ...],
    ) -> _ResultadoParticion:
        candidatos = self._candidatos_geodesicos(nodos, alc_total, mec_total)

        assert self.sia_dists_marginales is not None
        mejor = _ResultadoParticion(
            perdida=float("inf"),
            distribucion=self.sia_dists_marginales.copy(),
            subalcance=(),
            submecanismo=(),
        )
        for subalcance, submecanismo in candidatos:
            p, d = self._evaluar_particion(subalcance, submecanismo)
            if p < mejor.perdida:
                mejor = _ResultadoParticion(p, d, subalcance, submecanismo)

        return self._refinar_local(mejor, alc_total, mec_total)

    # ------------------------------------------------------------------
    # K-partición: embedding en R^{k-1} hiperbólico + k-means
    # ------------------------------------------------------------------

    def _resolver_k(
        self,
        nodos: list[int],
        alc_total: tuple[int, ...],
        mec_total: tuple[int, ...],
        k: int,
    ) -> _ResultadoParticionK:
        n = len(nodos)
        W = self._construir_conductancias(nodos)
        L = np.diag(W.sum(axis=1)) - W
        k_eff = min(k, n)

        try:
            vals, vecs = np.linalg.eigh(L)
            orden = np.argsort(vals)
            # Tomar k-1 eigenvectores no triviales
            embedding_eucl = vecs[:, orden[1:k_eff]]
        except np.linalg.LinAlgError:
            embedding_eucl = np.random.default_rng(42).random((n, k_eff - 1))

        # Proyección hiperbólica por filas
        embedding_h = self._proyectar_hiperbolico(embedding_eucl)
        asignacion = self._kmeans(embedding_h, k_eff)
        asignacion = self._canonicalizar(asignacion)
        if len(set(asignacion)) < 2:
            asignacion = self._canonicalizar(tuple(i % k_eff for i in range(n)))

        perdida, dist = self._evaluar_k(asignacion, nodos)
        assert self.sia_dists_marginales is not None
        mejor = _ResultadoParticionK(perdida, dist, asignacion, nodos)
        return self._refinar_k(mejor, k_eff, nodos)

    def _proyectar_hiperbolico(self, coords: np.ndarray, alpha: float = 2.0) -> np.ndarray:
        """Proyección hiperbólica fila a fila: r → tanh(α·r)/r · coords."""
        r = np.linalg.norm(coords, axis=1, keepdims=True)
        r_max = r.max()
        if r_max < 1e-12:
            return coords
        coords = coords / r_max
        r = r / r_max
        r_safe = np.where(r > 1e-12, r, 1.0)
        coords_h = coords / r_safe * np.tanh(alpha * r)
        r_h = np.linalg.norm(coords_h, axis=1, keepdims=True)
        mask = r_h >= 1.0
        if mask.any():
            coords_h = np.where(mask, coords_h / (r_h + 1e-9) * 0.99, coords_h)
        return coords_h

    def _kmeans(self, X: np.ndarray, k: int, max_iter: int = 60) -> tuple[int, ...]:
        n = X.shape[0]
        if k >= n:
            return tuple(range(n))
        rng = np.random.default_rng(42)
        centros_idx = [int(rng.integers(0, n))]
        for _ in range(k - 1):
            d_min = np.min(
                np.stack([np.sum((X - X[c]) ** 2, axis=1) for c in centros_idx]), axis=0,
            )
            d_min[centros_idx] = 0.0
            total = d_min.sum()
            if total == 0.0:
                break
            centros_idx.append(int(rng.choice(n, p=d_min / total)))
        centros = X[centros_idx].copy()
        etiquetas = np.zeros(n, dtype=np.int32)
        for _ in range(max_iter):
            nuevas = np.argmin(
                np.stack([np.sum((X - c) ** 2, axis=1) for c in centros]), axis=0,
            ).astype(np.int32)
            if np.all(nuevas == etiquetas):
                break
            etiquetas = nuevas
            for cl in range(len(centros_idx)):
                m = X[etiquetas == cl]
                if len(m) > 0:
                    centros[cl] = m.mean(axis=0)
        return tuple(int(e) for e in etiquetas.tolist())

    # ------------------------------------------------------------------
    # Evaluación de particiones
    # ------------------------------------------------------------------

    def _evaluar_particion(
        self, subalcance: tuple[int, ...], submecanismo: tuple[int, ...]
    ) -> tuple[float, np.ndarray]:
        clave = (subalcance, submecanismo)
        cached = self._cache_particiones.get(clave)
        if cached is not None:
            return cached
        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None
        sp = self.sia_subsistema.bipartir(
            np.array(subalcance, dtype=np.int8),
            np.array(submecanismo, dtype=np.int8),
        )
        d = sp.distribucion_marginal()
        if d.size != self.sia_dists_marginales.size:
            d2 = np.zeros_like(self.sia_dists_marginales)
            d2[: d.size] = d
            d = d2
        p = float(self.distancia_metrica(self.sia_dists_marginales, d))
        self._cache_particiones[clave] = (p, d)
        return p, d

    def _evaluar_k(
        self, asignacion: tuple[int, ...], nodos: list[int]
    ) -> tuple[float, np.ndarray]:
        cached = self._cache_k_particiones.get(asignacion)
        if cached is not None:
            return cached
        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None
        sp = self.sia_subsistema.k_bipartir(nodos, asignacion)
        d = sp.distribucion_marginal()
        if d.size != self.sia_dists_marginales.size:
            d2 = np.zeros_like(self.sia_dists_marginales)
            d2[: d.size] = d
            d = d2
        p = float(self.distancia_metrica(self.sia_dists_marginales, d))
        self._cache_k_particiones[asignacion] = (p, d)
        return p, d

    # ------------------------------------------------------------------
    # Multi-start con objetivo φ directo (reinicios aleatorios)
    # ------------------------------------------------------------------

    def _multistart_k2(
        self,
        mejor_hasta_ahora: _ResultadoParticion,
        alc_total: tuple[int, ...],
        mec_total: tuple[int, ...],
    ) -> _ResultadoParticion:
        """Prueba n_random_restarts puntos aleatorios en el espacio bipartir y refina cada uno."""
        assert self.sia_dists_marginales is not None
        n_alc, n_mec = len(alc_total), len(mec_total)
        rng = np.random.default_rng(42)
        mejor = mejor_hasta_ahora

        for _ in range(self.n_random_restarts):
            for _intento in range(20):
                alc_mask = rng.integers(0, 2, n_alc).astype(bool)
                mec_mask = rng.integers(0, 2, n_mec).astype(bool)
                sa = tuple(alc_total[i] for i in range(n_alc) if alc_mask[i])
                sm = tuple(mec_total[j] for j in range(n_mec) if mec_mask[j])
                if (sa or sm) and not (sa == alc_total and sm == mec_total):
                    break
            else:
                continue
            p, d = self._evaluar_particion(sa, sm)
            candidato = _ResultadoParticion(p, d, sa, sm)
            candidato = self._refinar_local(candidato, alc_total, mec_total)
            if candidato.perdida < mejor.perdida:
                mejor = candidato

        return mejor

    # ------------------------------------------------------------------
    # Refinamiento local
    # ------------------------------------------------------------------

    def _refinar_local(
        self,
        inicio: _ResultadoParticion,
        alc_total: tuple[int, ...],
        mec_total: tuple[int, ...],
    ) -> _ResultadoParticion:
        actual = inicio
        for _ in range(self._max_iter_refinamiento):
            vecinos = self._vecinos(actual.subalcance, actual.submecanismo, alc_total, mec_total)
            if not vecinos:
                break
            mejor_v = actual
            for alc, mec in vecinos:
                p, d = self._evaluar_particion(alc, mec)
                if p < mejor_v.perdida:
                    mejor_v = _ResultadoParticion(p, d, alc, mec)
            if mejor_v.perdida + 1e-12 >= actual.perdida:
                break
            actual = mejor_v
        return actual

    def _vecinos(
        self,
        subalc: tuple[int, ...],
        submec: tuple[int, ...],
        alc_total: tuple[int, ...],
        mec_total: tuple[int, ...],
    ) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        vecinos: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        vistos: set[tuple] = set()

        def agregar(ca: tuple[int, ...], cm: tuple[int, ...]) -> None:
            if not ca and not cm:
                return
            if ca == alc_total and cm == mec_total:
                return
            k = (ca, cm)
            if k not in vistos:
                vistos.add(k)
                vecinos.append(k)

        for nodo in alc_total:
            nuevo = set(subalc)
            nuevo.discard(nodo) if nodo in nuevo else nuevo.add(nodo)
            agregar(tuple(v for v in alc_total if v in nuevo), submec)
        for nodo in mec_total:
            nuevo = set(submec)
            nuevo.discard(nodo) if nodo in nuevo else nuevo.add(nodo)
            agregar(subalc, tuple(v for v in mec_total if v in nuevo))
        return vecinos

    def _refinar_k(
        self, inicio: _ResultadoParticionK, k: int, nodos: list[int]
    ) -> _ResultadoParticionK:
        actual = inicio
        mejor_global = inicio
        for _ in range(self._max_iter_refinamiento):
            vecinos = self._vecinos_k(actual.asignacion, k)
            if not vecinos:
                break
            mejor_v = actual
            for asig in vecinos:
                p, d = self._evaluar_k(asig, nodos)
                if p < mejor_v.perdida:
                    mejor_v = _ResultadoParticionK(p, d, asig, nodos)
            if mejor_v.perdida + 1e-12 >= actual.perdida:
                break
            actual = mejor_v
            if actual.perdida < mejor_global.perdida:
                mejor_global = actual
        return mejor_global

    def _vecinos_k(self, asig: tuple[int, ...], k: int) -> list[tuple[int, ...]]:
        vecinos: list[tuple[int, ...]] = []
        vistos: set[tuple[int, ...]] = set()
        for idx in range(len(asig)):
            for g in range(k):
                if g == asig[idx]:
                    continue
                nueva = list(asig)
                nueva[idx] = g
                canon = self._canonicalizar(tuple(nueva))
                if len(set(canon)) < 2 or canon in vistos:
                    continue
                vistos.add(canon)
                vecinos.append(canon)
        return vecinos

    @staticmethod
    def _canonicalizar(asig: tuple[int, ...]) -> tuple[int, ...]:
        mapa: dict[int, int] = {}
        sig = 0
        res = []
        for g in asig:
            if g not in mapa:
                mapa[g] = sig
                sig += 1
            res.append(mapa[g])
        return tuple(res)

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from src.constantes.models import BB_LABEL
from src.funciones.formato import fmt_biparticion, fmt_k_particion_q
from src.funciones.iit import seleccionar_emd
from src.modelos.base.sia import SIA
from src.modelos.nucleo.solucion import Solucion


@dataclass
class _Mejor:
    perdida: float
    distribucion: np.ndarray
    subalcance: tuple[int, ...]
    submecanismo: tuple[int, ...]


@dataclass
class _MejorK:
    perdida: float
    distribucion: np.ndarray
    asignacion: tuple[int, ...]  # etiqueta de grupo para cada vértice


class BranchBound(SIA):
    """Búsqueda exacta/cuasi-exacta para el MIP-IIT, k≥2.

    k=2 (bipartición)
    -----------------
    Fase exacta (n_total <= umbral_exacto, default 14):
        Enumera todas las biparticiones válidas → óptimo global garantizado.

    Fase heurística (n_total > umbral_exacto):
        SA multi-arranque + expansión Hamming exhaustiva.

    k>2 (k-partición)
    -----------------
    Fase exacta (S(n,k) <= max_exact_k, default 200_000):
        Genera todas las k-particiones válidas via cadenas de crecimiento
        restringido → óptimo global garantizado.
        Ejemplos de alcance exacto:
          k=2: n_total ≤ 18  (S(18,2)=131 071)
          k=3: n_total ≤ 12  (S(12,3)=86 526)
          k=4: n_total ≤ 10  (S(10,4)=34 105)
          k=5: n_total ≤ 10  (S(10,5)=42 525)

    Fase heurística (S(n,k) > max_exact_k):
        SA multi-arranque sobre asignaciones de k grupos + búsqueda
        exhaustiva del vecindario Hamming-1 desde el mejor SA.

    Ventaja sobre QNodos para k>2
    ------------------------------
    QNodos usa un único warm-start desde el árbol de Queyranne + SA.
    BranchBound usa n_sa_arranques arranques independientes y búsqueda
    local exhaustiva, capturando más mínimos no-submodulares.
    """

    def __init__(
        self,
        tpm: np.ndarray,
        config=None,
        umbral_exacto: int = 14,
        n_sa_arranques: int = 8,
        radio_hamming: int = 3,
        pasos_sa: int = 800,
        max_exact_k: int = 200_000,
    ) -> None:
        super().__init__(tpm, config)
        self.distancia_metrica = seleccionar_emd(config)
        self.umbral_exacto = umbral_exacto
        self.n_sa_arranques = n_sa_arranques
        self.radio_hamming = radio_hamming
        self.pasos_sa = pasos_sa
        self.max_exact_k = max_exact_k
        self._cache: dict[tuple, tuple[float, np.ndarray]] = {}
        self._cache_k: dict[tuple, tuple[float, np.ndarray]] = {}

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
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None
        self._cache.clear()
        self._cache_k.clear()

        alc_total = tuple(int(v) for v in self.sia_subsistema.indices_ncubos.tolist())
        mec_total = tuple(int(v) for v in self.sia_subsistema.dims_ncubos.tolist())

        if not alc_total and not mec_total:
            return Solucion(
                estrategia=BB_LABEL,
                perdida=0.0,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=self.sia_dists_marginales.copy(),
                estado_inicial=estado_inicial,
                particion="NO-PARTITION",
            )

        # ── k=2: código original (bipartición) ──────────────────────────
        if k == 2:
            n_total = len(alc_total) + len(mec_total)
            if n_total <= self.umbral_exacto:
                mejor = self._fase_exacta(alc_total, mec_total)
            else:
                mejor = self._fase_heuristica(alc_total, mec_total)

            return Solucion(
                estrategia=BB_LABEL,
                perdida=mejor.perdida,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=mejor.distribucion,
                estado_inicial=estado_inicial,
                particion=fmt_biparticion(
                    mejor.subalcance,
                    mejor.submecanismo,
                    alc_total,
                    mec_total,
                ),
            )

        # ── k>2: k-partición general ─────────────────────────────────────
        # vertices: (tiempo=0 → mecanismo, tiempo=1 → alcance)
        vertices: list[tuple[int, int]] = (
            [(0, int(v)) for v in mec_total]
            + [(1, int(v)) for v in alc_total]
        )
        n = len(vertices)
        s = self._stirling(n, k)

        if s <= self.max_exact_k:
            mejor_k = self._fase_exacta_k(vertices, k)
        else:
            mejor_k = self._fase_heuristica_k(vertices, k)

        grupos = [
            [vertices[i] for i in range(n) if mejor_k.asignacion[i] == g]
            for g in range(k)
        ]
        return Solucion(
            estrategia=BB_LABEL,
            perdida=mejor_k.perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=mejor_k.distribucion,
            estado_inicial=estado_inicial,
            particion=fmt_k_particion_q(grupos),
        )

    # ==================================================================
    # k=2: código original sin cambios
    # ==================================================================

    def _fase_exacta(
        self,
        alc_total: tuple[int, ...],
        mec_total: tuple[int, ...],
    ) -> _Mejor:
        n_a = len(alc_total)
        n_m = len(mec_total)

        assert self.sia_dists_marginales is not None
        mejor = _Mejor(
            perdida=float("inf"),
            distribucion=self.sia_dists_marginales.copy(),
            subalcance=(),
            submecanismo=(),
        )

        for alc_int in range(1 << n_a):
            alc_set = tuple(alc_total[i] for i in range(n_a) if (alc_int >> i) & 1)
            for mec_int in range(1 << n_m):
                mec_set = tuple(mec_total[j] for j in range(n_m) if (mec_int >> j) & 1)
                if not alc_set and not mec_set:
                    continue
                if len(alc_set) == n_a and len(mec_set) == n_m:
                    continue
                perdida, dist = self._evaluar(alc_set, mec_set)
                if perdida < mejor.perdida:
                    mejor = _Mejor(perdida, dist, alc_set, mec_set)

        return mejor

    def _fase_heuristica(
        self,
        alc_total: tuple[int, ...],
        mec_total: tuple[int, ...],
    ) -> _Mejor:
        assert self.sia_dists_marginales is not None
        mejor = _Mejor(
            perdida=float("inf"),
            distribucion=self.sia_dists_marginales.copy(),
            subalcance=(),
            submecanismo=(),
        )

        for arranque in range(self.n_sa_arranques):
            resultado = self._sa_run(
                alc_total, mec_total, arranque,
                temp_inicial=0.5 * (1.0 + arranque * 0.4),
            )
            if resultado.perdida < mejor.perdida:
                mejor = resultado

        for alc_set, mec_set in self._hamming_ball(
            mejor.subalcance, mejor.submecanismo, alc_total, mec_total
        ):
            perdida, dist = self._evaluar(alc_set, mec_set)
            if perdida < mejor.perdida:
                mejor = _Mejor(perdida, dist, alc_set, mec_set)

        if mejor.subalcance or mejor.submecanismo:
            for alc_set, mec_set in self._hamming_ball(
                mejor.subalcance, mejor.submecanismo, alc_total, mec_total
            ):
                perdida, dist = self._evaluar(alc_set, mec_set)
                if perdida < mejor.perdida:
                    mejor = _Mejor(perdida, dist, alc_set, mec_set)

        return mejor

    def _sa_run(
        self,
        alc_total: tuple[int, ...],
        mec_total: tuple[int, ...],
        semilla_offset: int = 0,
        temp_inicial: float = 1.0,
        temp_final: float = 0.001,
    ) -> _Mejor:
        n_a = len(alc_total)
        n_m = len(mec_total)
        n_total = n_a + n_m

        rng = np.random.default_rng(42 + semilla_offset)

        alc_mask = list(rng.integers(0, 2, size=n_a).tolist())
        mec_mask = list(rng.integers(0, 2, size=n_m).tolist())
        self._garantizar_valido(alc_mask, mec_mask, n_a, n_m, rng)

        alc_set = tuple(alc_total[i] for i in range(n_a) if alc_mask[i])
        mec_set = tuple(mec_total[j] for j in range(n_m) if mec_mask[j])
        perdida, dist = self._evaluar(alc_set, mec_set)

        mejor = _Mejor(perdida, dist, alc_set, mec_set)

        factor = (temp_final / temp_inicial) ** (1.0 / max(1, self.pasos_sa))
        temp = temp_inicial

        for _ in range(self.pasos_sa):
            idx = int(rng.integers(n_total))
            if idx < n_a:
                alc_mask[idx] ^= 1
            else:
                mec_mask[idx - n_a] ^= 1

            nueva_alc = tuple(alc_total[i] for i in range(n_a) if alc_mask[i])
            nueva_mec = tuple(mec_total[j] for j in range(n_m) if mec_mask[j])

            if not nueva_alc and not nueva_mec:
                if idx < n_a:
                    alc_mask[idx] ^= 1
                else:
                    mec_mask[idx - n_a] ^= 1
                temp *= factor
                continue
            if len(nueva_alc) == n_a and len(nueva_mec) == n_m:
                if idx < n_a:
                    alc_mask[idx] ^= 1
                else:
                    mec_mask[idx - n_a] ^= 1
                temp *= factor
                continue

            nueva_p, nueva_d = self._evaluar(nueva_alc, nueva_mec)
            delta = nueva_p - perdida

            if delta < 0 or rng.random() < math.exp(-delta / (temp + 1e-12)):
                perdida, dist, alc_set, mec_set = nueva_p, nueva_d, nueva_alc, nueva_mec
                if perdida < mejor.perdida:
                    mejor = _Mejor(perdida, dist, alc_set, mec_set)
            else:
                if idx < n_a:
                    alc_mask[idx] ^= 1
                else:
                    mec_mask[idx - n_a] ^= 1

            temp *= factor

        return mejor

    def _hamming_ball(
        self,
        alc_star: tuple[int, ...],
        mec_star: tuple[int, ...],
        alc_total: tuple[int, ...],
        mec_total: tuple[int, ...],
    ) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        n_a = len(alc_total)
        n_m = len(mec_total)
        n = n_a + n_m

        base_mask = [1 if a in alc_star else 0 for a in alc_total] + \
                    [1 if m in mec_star else 0 for m in mec_total]

        vistos: set[tuple] = set()
        resultado: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        for radio in range(1, self.radio_hamming + 1):
            for flips in combinations(range(n), radio):
                mask = base_mask.copy()
                for f in flips:
                    mask[f] ^= 1

                alc_set = tuple(alc_total[i] for i in range(n_a) if mask[i])
                mec_set = tuple(mec_total[j] for j in range(n_m) if mask[n_a + j])

                if not alc_set and not mec_set:
                    continue
                if len(alc_set) == n_a and len(mec_set) == n_m:
                    continue

                clave = (alc_set, mec_set)
                if clave not in vistos:
                    vistos.add(clave)
                    resultado.append(clave)

        return resultado

    def _evaluar(
        self,
        alc_set: tuple[int, ...],
        mec_set: tuple[int, ...],
    ) -> tuple[float, np.ndarray]:
        clave = (alc_set, mec_set)
        en_cache = self._cache.get(clave)
        if en_cache is not None:
            return en_cache

        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None

        sistema_partido = self.sia_subsistema.bipartir(
            np.array(alc_set, dtype=np.int8),
            np.array(mec_set, dtype=np.int8),
        )
        dist = sistema_partido.distribucion_marginal()
        if dist.size != self.sia_dists_marginales.size:
            dist_alineada = np.zeros_like(self.sia_dists_marginales)
            dist_alineada[: dist.size] = dist
            dist = dist_alineada

        perdida = float(self.distancia_metrica(self.sia_dists_marginales, dist))
        self._cache[clave] = (perdida, dist)
        return perdida, dist

    @staticmethod
    def _garantizar_valido(
        alc_mask: list[int],
        mec_mask: list[int],
        n_a: int,
        n_m: int,
        rng: np.random.Generator,
    ) -> None:
        if not any(alc_mask) and not any(mec_mask):
            alc_mask[int(rng.integers(n_a))] = 1
        if all(alc_mask) and all(mec_mask):
            alc_mask[int(rng.integers(n_a))] = 0

    # ==================================================================
    # k>2: k-partición general
    # ==================================================================

    @staticmethod
    def _stirling(n: int, k: int) -> int:
        """Número de Stirling de segunda especie S(n, k): particiones de n en k grupos."""
        if k > n or k < 1:
            return 0
        if k == 1 or k == n:
            return 1
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                dp[i][j] = j * dp[i - 1][j] + dp[i - 1][j - 1]
        return dp[n][k]

    def _enumerar_k_particiones(self, n: int, k: int):
        """Genera todas las asignaciones de n elementos a exactamente k grupos (RGS)."""
        def rec(i: int, current: list[int], max_usado: int):
            if i == n:
                if max_usado == k - 1:
                    yield tuple(current)
                return
            for g in range(min(max_usado + 2, k)):
                current.append(g)
                yield from rec(i + 1, current, max(max_usado, g))
                current.pop()

        yield from rec(0, [], -1)

    @staticmethod
    def _canonicalizar(asig: tuple[int, ...]) -> tuple[int, ...]:
        """Forma canónica de una asignación: renumera grupos en orden de primera aparición.

        (2,0,1,2,0) → (0,1,2,0,1)
        Permite que particiones equivalentes con grupos renombrados compartan caché,
        reduciendo evaluaciones hasta k! veces en la fase heurística.
        """
        mapa: dict[int, int] = {}
        sig = 0
        canon: list[int] = []
        for g in asig:
            if g not in mapa:
                mapa[g] = sig
                sig += 1
            canon.append(mapa[g])
        return tuple(canon)

    def _evaluar_k(
        self,
        asignacion: tuple[int, ...],
        vertices: list[tuple[int, int]],
    ) -> tuple[float, np.ndarray]:
        # Canonicalizar antes del lookup: (2,0,1) == (0,1,2) → misma partición
        canon = self._canonicalizar(asignacion)
        en_cache = self._cache_k.get(canon)
        if en_cache is not None:
            return en_cache

        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None

        n = len(vertices)
        k = max(canon) + 1

        grupos_mec = [
            tuple(
                vertices[i][1]
                for i in range(n)
                if canon[i] == g and vertices[i][0] == 0
            )
            for g in range(k)
        ]
        grupos_alc = [
            tuple(
                vertices[i][1]
                for i in range(n)
                if canon[i] == g and vertices[i][0] == 1
            )
            for g in range(k)
        ]

        sistema_partido = self.sia_subsistema.k_bipartir_temporal(grupos_mec, grupos_alc)
        dist = sistema_partido.distribucion_marginal()
        if dist.size != self.sia_dists_marginales.size:
            dist_a = np.zeros_like(self.sia_dists_marginales)
            dist_a[: dist.size] = dist
            dist = dist_a

        perdida = float(self.distancia_metrica(self.sia_dists_marginales, dist))
        self._cache_k[canon] = (perdida, dist)
        return perdida, dist

    def _fase_exacta_k(
        self,
        vertices: list[tuple[int, int]],
        k: int,
    ) -> _MejorK:
        assert self.sia_dists_marginales is not None
        n = len(vertices)
        mejor = _MejorK(
            perdida=float("inf"),
            distribucion=self.sia_dists_marginales.copy(),
            asignacion=(0,) * n,
        )
        for asig in self._enumerar_k_particiones(n, k):
            perdida, dist = self._evaluar_k(asig, vertices)
            if perdida < mejor.perdida:
                mejor = _MejorK(perdida, dist, asig)
        return mejor

    @staticmethod
    def _asig_estructurada(
        vertices: list[tuple[int, int]],
        k: int,
    ) -> list[int]:
        """Warm-start determinista: asigna vértices a grupos por orden (tiempo, índice).

        Ordena los vértices y los reparte en k grupos de tamaño ~igual.
        Garantiza todos los grupos no vacíos y separa vértices "similares".
        """
        n = len(vertices)
        orden = sorted(range(n), key=lambda i: (vertices[i][0], vertices[i][1]))
        asig = [0] * n
        for pos, idx in enumerate(orden):
            asig[idx] = pos % k
        return asig

    def _sa_run_k(
        self,
        vertices: list[tuple[int, int]],
        k: int,
        semilla_offset: int = 0,
        temp_inicial: float = 1.0,
        temp_final: float = 0.001,
        asig_inicial: list[int] | None = None,
    ) -> _MejorK:
        n = len(vertices)
        rng = np.random.default_rng(42 + semilla_offset)

        if asig_inicial is not None:
            asig = list(asig_inicial)
        else:
            # Inicialización aleatoria válida: todos los k grupos tienen ≥1 elemento
            base = list(range(k))
            resto = list(rng.integers(0, k, size=max(0, n - k)).tolist())
            asig = base + resto
            rng.shuffle(asig)

        # Contador de elementos por grupo — se mantiene incrementalmente
        conteo = [0] * k
        for g in asig:
            conteo[g] += 1

        perdida, dist = self._evaluar_k(tuple(asig), vertices)
        mejor = _MejorK(perdida, dist, tuple(asig))

        factor = (temp_final / temp_inicial) ** (1.0 / max(1, self.pasos_sa))
        temp = temp_inicial

        for _ in range(self.pasos_sa):
            # 30% swap (intercambio entre dos grupos), 70% reassign (cambio de grupo)
            if rng.random() < 0.3 and n >= 2:
                # Swap: intercambia dos elementos de grupos distintos
                i = int(rng.integers(n))
                j = int(rng.integers(n))
                if i == j or asig[i] == asig[j]:
                    temp *= factor
                    continue
                gi, gj = asig[i], asig[j]
                asig[i], asig[j] = gj, gi
                # conteo no cambia en un swap

                nueva_p, nueva_d = self._evaluar_k(tuple(asig), vertices)
                delta = nueva_p - perdida
                if delta < 0 or rng.random() < math.exp(-delta / (temp + 1e-12)):
                    perdida, dist = nueva_p, nueva_d
                    if perdida < mejor.perdida:
                        mejor = _MejorK(perdida, dist, tuple(asig))
                else:
                    asig[i], asig[j] = gi, gj  # revertir
            else:
                # Reassign: mueve un elemento a otro grupo
                elem = int(rng.integers(n))
                g_actual = asig[elem]

                if conteo[g_actual] == 1:
                    temp *= factor
                    continue

                opciones = [g for g in range(k) if g != g_actual]
                g_nuevo = int(opciones[int(rng.integers(len(opciones)))])
                asig[elem] = g_nuevo
                conteo[g_actual] -= 1
                conteo[g_nuevo] += 1

                nueva_p, nueva_d = self._evaluar_k(tuple(asig), vertices)
                delta = nueva_p - perdida
                if delta < 0 or rng.random() < math.exp(-delta / (temp + 1e-12)):
                    perdida, dist = nueva_p, nueva_d
                    if perdida < mejor.perdida:
                        mejor = _MejorK(perdida, dist, tuple(asig))
                else:
                    asig[elem] = g_actual
                    conteo[g_actual] += 1
                    conteo[g_nuevo] -= 1

            temp *= factor

        return mejor

    def _vecindad_1_k(
        self,
        asig: tuple[int, ...],
        n: int,
        k: int,
    ) -> list[tuple[int, ...]]:
        """Todos los vecinos a distancia Hamming 1: un elemento cambia de grupo."""
        conteo = [0] * k
        for g in asig:
            conteo[g] += 1

        vistos: set[tuple[int, ...]] = set()
        resultado: list[tuple[int, ...]] = []

        for elem in range(n):
            g_actual = asig[elem]
            if conteo[g_actual] == 1:
                continue  # vaciaria el grupo
            for g_nuevo in range(k):
                if g_nuevo == g_actual:
                    continue
                nueva = list(asig)
                nueva[elem] = g_nuevo
                clave = tuple(nueva)
                if clave not in vistos:
                    vistos.add(clave)
                    resultado.append(clave)

        return resultado

    def _fase_heuristica_k(
        self,
        vertices: list[tuple[int, int]],
        k: int,
    ) -> _MejorK:
        assert self.sia_dists_marginales is not None
        n = len(vertices)
        mejor = _MejorK(
            perdida=float("inf"),
            distribucion=self.sia_dists_marginales.copy(),
            asignacion=(0,) * n,
        )

        # Arranque 0: warm-start estructurado (orden tiempo/índice, round-robin)
        warm = self._asig_estructurada(vertices, k)
        resultado = self._sa_run_k(
            vertices, k, semilla_offset=0,
            temp_inicial=0.5,
            asig_inicial=warm,
        )
        if resultado.perdida < mejor.perdida:
            mejor = resultado

        # Arranques 1..n_sa_arranques-1: inicialización aleatoria con temperatura escalonada
        for arranque in range(1, self.n_sa_arranques):
            resultado = self._sa_run_k(
                vertices, k, semilla_offset=arranque,
                temp_inicial=0.5 * (1.0 + arranque * 0.4),
            )
            if resultado.perdida < mejor.perdida:
                mejor = resultado

        # Búsqueda exhaustiva del vecindario Hamming-1 desde el mejor SA
        for _ in range(2):
            mejorado = False
            for asig_vec in self._vecindad_1_k(mejor.asignacion, n, k):
                perdida, dist = self._evaluar_k(asig_vec, vertices)
                if perdida < mejor.perdida - 1e-12:
                    mejor = _MejorK(perdida, dist, asig_vec)
                    mejorado = True
            if not mejorado:
                break

        return mejor

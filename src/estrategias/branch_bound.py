from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from src.constantes.models import BB_LABEL
from src.funciones.formato import fmt_biparticion
from src.funciones.iit import seleccionar_emd
from src.modelos.base.sia import SIA
from src.modelos.nucleo.solucion import Solucion


@dataclass
class _Mejor:
    perdida: float
    distribucion: np.ndarray
    subalcance: tuple[int, ...]
    submecanismo: tuple[int, ...]


class BranchBound(SIA):
    """Búsqueda exacta/cuasi-exacta para el MIP-IIT.

    Fase exacta (n_total <= umbral_exacto, por defecto 14 bits = 7 nodos/lado):
        Enumera todas las biparticiones válidas y devuelve el óptimo global.
        Garantiza el mismo resultado que FuerzaBruta pero con cache compartida.

    Fase heurística (n_total > umbral_exacto):
        1. SA multi-arranque: n_sa_arranques cadenas independientes con
           temperaturas iniciales escalonadas, exploración más amplia que
           el SA de QNodos (que usa un solo arranque desde el resultado MAO).
        2. Expansión Hamming: para el mejor de los SA, evalúa exhaustivamente
           todos los vecinos dentro de distancia Hamming radio_hamming.
           Captura mínimos no submodulares a los que el SA de QNodos no llega.

    Cuándo supera a QNodos
    ----------------------
    QNodos es exacto para funciones submodulares (~88% de sistemas aleatorios).
    En el ~12% restante cae a SA desde un único arranque (resultado MAO), que
    puede quedar atrapado en un mínimo local lejano del óptimo.
    BranchBound:
      - n <= 7 nodos/lado: GARANTIZA el óptimo (exhaustivo).
      - n > 7 nodos/lado: reduce el gap con multi-arranque + Hamming exhaustivo.
    """

    def __init__(
        self,
        tpm: np.ndarray,
        config=None,
        umbral_exacto: int = 14,
        n_sa_arranques: int = 8,
        radio_hamming: int = 3,
        pasos_sa: int = 800,
    ) -> None:
        super().__init__(tpm, config)
        self.distancia_metrica = seleccionar_emd(config)
        self.umbral_exacto = umbral_exacto
        self.n_sa_arranques = n_sa_arranques
        self.radio_hamming = radio_hamming
        self.pasos_sa = pasos_sa
        self._cache: dict[tuple, tuple[float, np.ndarray]] = {}

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
        if k != 2:
            raise NotImplementedError("BranchBound solo soporta k=2 (bipartición).")

        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None
        self._cache.clear()

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

    # ------------------------------------------------------------------
    # Fase exacta: enumeración exhaustiva
    # ------------------------------------------------------------------

    def _fase_exacta(
        self,
        alc_total: tuple[int, ...],
        mec_total: tuple[int, ...],
    ) -> _Mejor:
        """Evalúa todas las biparticiones válidas y devuelve el mínimo global."""
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
                # Excluir los dos estados triviales
                if not alc_set and not mec_set:
                    continue
                if len(alc_set) == n_a and len(mec_set) == n_m:
                    continue
                perdida, dist = self._evaluar(alc_set, mec_set)
                if perdida < mejor.perdida:
                    mejor = _Mejor(perdida, dist, alc_set, mec_set)

        return mejor

    # ------------------------------------------------------------------
    # Fase heurística: SA multi-arranque + expansión Hamming
    # ------------------------------------------------------------------

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

        n_a = len(alc_total)
        n_m = len(mec_total)

        # SA desde n_sa_arranques inicializaciones distintas
        for arranque in range(self.n_sa_arranques):
            resultado = self._sa_run(
                alc_total, mec_total, arranque,
                # Temperatura inicial escalonada: arranques posteriores exploran más
                temp_inicial=0.5 * (1.0 + arranque * 0.4),
            )
            if resultado.perdida < mejor.perdida:
                mejor = resultado

        # Expansión Hamming exhaustiva desde el mejor SA
        for alc_set, mec_set in self._hamming_ball(
            mejor.subalcance, mejor.submecanismo, alc_total, mec_total
        ):
            perdida, dist = self._evaluar(alc_set, mec_set)
            if perdida < mejor.perdida:
                mejor = _Mejor(perdida, dist, alc_set, mec_set)

        # Si la expansión Hamming mejoró, re-expandir desde el nuevo mínimo
        if mejor.subalcance or mejor.submecanismo:
            for alc_set, mec_set in self._hamming_ball(
                mejor.subalcance, mejor.submecanismo, alc_total, mec_total
            ):
                perdida, dist = self._evaluar(alc_set, mec_set)
                if perdida < mejor.perdida:
                    mejor = _Mejor(perdida, dist, alc_set, mec_set)

        return mejor

    # ------------------------------------------------------------------
    # SA (Simulated Annealing) sobre (alc_mask, mec_mask)
    # ------------------------------------------------------------------

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

        # Inicialización: bit aleatorio garantizado válido
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
            # Movimiento: flip de un bit aleatorio (alc o mec)
            idx = int(rng.integers(n_total))
            if idx < n_a:
                alc_mask[idx] ^= 1
            else:
                mec_mask[idx - n_a] ^= 1

            nueva_alc = tuple(alc_total[i] for i in range(n_a) if alc_mask[i])
            nueva_mec = tuple(mec_total[j] for j in range(n_m) if mec_mask[j])

            if not nueva_alc and not nueva_mec:
                # Revertir
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
                # Revertir movimiento
                if idx < n_a:
                    alc_mask[idx] ^= 1
                else:
                    mec_mask[idx - n_a] ^= 1

            temp *= factor

        return mejor

    # ------------------------------------------------------------------
    # Expansión Hamming
    # ------------------------------------------------------------------

    def _hamming_ball(
        self,
        alc_star: tuple[int, ...],
        mec_star: tuple[int, ...],
        alc_total: tuple[int, ...],
        mec_total: tuple[int, ...],
    ) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """Genera todos los (alc, mec) a distancia Hamming ≤ radio_hamming de (alc_star, mec_star)."""
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

    # ------------------------------------------------------------------
    # Evaluador con cache
    # ------------------------------------------------------------------

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
        """Asegura que la máscara no es trivial (todo-cero o todo-uno)."""
        if not any(alc_mask) and not any(mec_mask):
            alc_mask[int(rng.integers(n_a))] = 1
        if all(alc_mask) and all(mec_mask):
            alc_mask[int(rng.integers(n_a))] = 0

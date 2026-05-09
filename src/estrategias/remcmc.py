"""Estrategia Replica Exchange Markov Chain Monte Carlo (REMCMC) para k-partición.

Parallel Tempering: R réplicas corren en paralelo a temperaturas fijas
T_0 < T_1 < ... < T_{R-1}. Cada réplica explora el espacio de particiones
con pasos Metropolis-Hastings. Periódicamente se intenta intercambiar los
estados de réplicas adyacentes i ↔ i+1 con probabilidad:

    A = min(1, exp((φ_i − φ_j) · (1/T_i − 1/T_j)))

donde φ_i es la pérdida (EMD) de la réplica i y T_i < T_j.

La cadena fría (T_0) explota mínimos; las calientes escapan de ellos.
Los intercambios transfieren configuraciones prometedoras desde cadenas
calientes a la fría sin violar balance detallado, combinando exploración
global con explotación local.

Semántica de bipartición
------------------------
k=2: busca sobre pares (subalcance, submecanismo) usando `bipartir`,
     idéntica al espacio que explora FuerzaBruta. El estado de cada
     réplica es (alc_mask, mec_mask) — vectores binarios de longitud n.
     Movimientos: flip de un elemento en alc_mask, flip en mec_mask, o swap.
k>2: asignación de nodos a grupos con `k_bipartir`.

Ventajas sobre SA estándar
--------------------------
- Las cadenas calientes no tienen restricción de temperatura descendente.
- Los swaps producen saltos de largo alcance en el espacio de particiones.
- Adecuado para funciones no submodulares (Hidaka y Oizumi, 2018).

Complejidad: O(n_rondas · n_replicas · pasos_por_ronda · evaluación)
"""

from __future__ import annotations

import math

import numpy as np

from src.constantes.models import REMCMC_LABEL
from src.funciones.formato import fmt_biparticion, fmt_k_particion_asignacion
from src.funciones.iit import seleccionar_emd
from src.modelos.base.sia import SIA
from src.modelos.nucleo.solucion import Solucion


class REMCMC(SIA):
    """k-partición por Replica Exchange MCMC (Parallel Tempering).

    Parámetros
    ----------
    n_replicas     : Número de cadenas (mín. 2).
    temp_min       : Temperatura de la cadena más fría (explotación).
    temp_max       : Temperatura de la cadena más caliente (exploración).
    pasos_por_ronda: Pasos MH que realiza cada réplica entre intentos de swap.
    n_rondas       : Número de rondas de intercambio (longitud total del run).
    config         : AppConfig inyectada (None → usa singleton global).
    """

    def __init__(
        self,
        tpm: np.ndarray,
        config=None,
        n_replicas: int = 6,
        temp_min: float = 0.001,
        temp_max: float = 2.0,
        pasos_por_ronda: int = 40,
        n_rondas: int = 60,
    ) -> None:
        super().__init__(tpm, config)
        self.distancia_metrica = seleccionar_emd(config)
        self.n_replicas = max(2, n_replicas)
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.pasos_por_ronda = pasos_por_ronda
        self.n_rondas = n_rondas

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
        n_alc = len(alcance_total)
        n_mec = len(mecanismo_total)

        if n_alc == 0 or n_mec == 0:
            return Solucion(
                estrategia=REMCMC_LABEL,
                perdida=0.0,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=self.sia_dists_marginales.copy(),
                estado_inicial=estado_inicial,
                particion="NO-PARTITION",
            )

        if k == 2:
            return self._resolver_k2(estado_inicial, alcance_total, mecanismo_total)

        # k > 2: búsqueda sobre asignaciones de nodos con k_bipartir
        nodos = sorted(set(alcance_total) | set(mecanismo_total))
        n = len(nodos)
        k_eff = min(k, n)
        if n <= 1 or k_eff < 2:
            return Solucion(
                estrategia=REMCMC_LABEL,
                perdida=0.0,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=self.sia_dists_marginales.copy(),
                estado_inicial=estado_inicial,
                particion="NO-PARTITION",
            )
        mejor_asig, mejor_perdida, mejor_dist = self._parallel_tempering_kn(nodos, k_eff)
        particion_str = fmt_k_particion_asignacion(nodos, mejor_asig, alcance_total, mecanismo_total)
        return Solucion(
            estrategia=REMCMC_LABEL,
            perdida=mejor_perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=mejor_dist,
            estado_inicial=estado_inicial,
            particion=particion_str,
        )

    # ------------------------------------------------------------------
    # k=2: bipartir sobre (subalcance, submecanismo) — mismo espacio que FB
    # ------------------------------------------------------------------

    def _resolver_k2(
        self,
        estado_inicial: str,
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
    ) -> Solucion:
        n_alc = len(alcance_total)
        n_mec = len(mecanismo_total)

        # Estado de cada réplica: (alc_mask, mec_mask) — arrays binarios
        # alc_mask[i]=1 → nodo alcance_total[i] está en subalcance
        # mec_mask[j]=1 → nodo mecanismo_total[j] está en submecanismo
        mejor_alc, mejor_mec, mejor_perdida, mejor_dist = self._parallel_tempering_k2(
            n_alc, n_mec, alcance_total, mecanismo_total
        )

        particion_str = fmt_biparticion(
            tuple(alcance_total[i] for i in range(n_alc) if mejor_alc[i]),
            tuple(mecanismo_total[j] for j in range(n_mec) if mejor_mec[j]),
            alcance_total,
            mecanismo_total,
        )
        return Solucion(
            estrategia=REMCMC_LABEL,
            perdida=mejor_perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=mejor_dist,
            estado_inicial=estado_inicial,
            particion=particion_str,
        )

    def _evaluar_k2(
        self,
        alc_mask: np.ndarray,
        mec_mask: np.ndarray,
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
    ) -> tuple[float, np.ndarray]:
        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None

        subalc = np.array(
            [alcance_total[i] for i in range(len(alcance_total)) if alc_mask[i]],
            dtype=np.int8,
        )
        submec = np.array(
            [mecanismo_total[j] for j in range(len(mecanismo_total)) if mec_mask[j]],
            dtype=np.int8,
        )
        sp = self.sia_subsistema.bipartir(subalc, submec)
        dist = sp.distribucion_marginal()
        ref = self.sia_dists_marginales
        if dist.size != ref.size:
            alineada = np.zeros_like(ref)
            alineada[: dist.size] = dist
            dist = alineada
        return float(self.distancia_metrica(ref, dist)), dist

    def _parallel_tempering_k2(
        self,
        n_alc: int,
        n_mec: int,
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        R = self.n_replicas
        rng = np.random.default_rng(73)

        ratio = (self.temp_max / self.temp_min) ** (1.0 / (R - 1))
        temps = [self.temp_min * (ratio ** i) for i in range(R)]

        # Inicializar réplicas con máscaras aleatorias
        alc_states: list[np.ndarray] = []
        mec_states: list[np.ndarray] = []
        perdidas: list[float] = []
        dists: list[np.ndarray] = []

        for _ in range(R):
            while True:
                alc = rng.integers(0, 2, size=n_alc).astype(np.int8)
                mec = rng.integers(0, 2, size=n_mec).astype(np.int8)
                if self._valido_k2(alc, mec, n_alc, n_mec):
                    break
            p, d = self._evaluar_k2(alc, mec, alcance_total, mecanismo_total)
            alc_states.append(alc.copy())
            mec_states.append(mec.copy())
            perdidas.append(p)
            dists.append(d)

        mejor_alc = alc_states[0].copy()
        mejor_mec = mec_states[0].copy()
        mejor_perdida = perdidas[0]
        mejor_dist = dists[0]
        for i in range(R):
            if perdidas[i] < mejor_perdida:
                mejor_perdida = perdidas[i]
                mejor_alc = alc_states[i].copy()
                mejor_mec = mec_states[i].copy()
                mejor_dist = dists[i]

        for ronda in range(self.n_rondas):
            # Fase 1: pasos MH a temperatura fija
            for i in range(R):
                alc_states[i], mec_states[i], perdidas[i], dists[i] = self._mcmc_pasos_k2(
                    alc_states[i], mec_states[i], perdidas[i], dists[i],
                    n_alc, n_mec, alcance_total, mecanismo_total, temps[i], rng,
                )
                if perdidas[i] < mejor_perdida:
                    mejor_perdida = perdidas[i]
                    mejor_alc = alc_states[i].copy()
                    mejor_mec = mec_states[i].copy()
                    mejor_dist = dists[i]

            # Fase 2: intentos de swap entre réplicas adyacentes (pares alternados)
            offset = ronda % 2
            for i in range(offset, R - 1, 2):
                j = i + 1
                delta_log = (perdidas[i] - perdidas[j]) * (1.0 / temps[i] - 1.0 / temps[j])
                if delta_log >= 0 or rng.random() < math.exp(delta_log):
                    alc_states[i], alc_states[j] = alc_states[j], alc_states[i]
                    mec_states[i], mec_states[j] = mec_states[j], mec_states[i]
                    perdidas[i], perdidas[j] = perdidas[j], perdidas[i]
                    dists[i], dists[j] = dists[j], dists[i]
                    if perdidas[i] < mejor_perdida:
                        mejor_perdida = perdidas[i]
                        mejor_alc = alc_states[i].copy()
                        mejor_mec = mec_states[i].copy()
                        mejor_dist = dists[i]

        return mejor_alc, mejor_mec, mejor_perdida, mejor_dist

    def _mcmc_pasos_k2(
        self,
        alc: np.ndarray,
        mec: np.ndarray,
        perdida: float,
        dist: np.ndarray,
        n_alc: int,
        n_mec: int,
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
        temp: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        for _ in range(self.pasos_por_ronda):
            nueva_alc = alc.copy()
            nueva_mec = mec.copy()

            # Tres tipos de movimiento con probabilidad igual
            mov = int(rng.integers(0, 3))
            if mov == 0:
                # Flip un elemento de subalcance
                idx = int(rng.integers(0, n_alc))
                nueva_alc[idx] ^= 1
            elif mov == 1:
                # Flip un elemento de submecanismo
                idx = int(rng.integers(0, n_mec))
                nueva_mec[idx] ^= 1
            else:
                # Flip simultáneo: uno de alc y uno de mec
                nueva_alc[int(rng.integers(0, n_alc))] ^= 1
                nueva_mec[int(rng.integers(0, n_mec))] ^= 1

            if not self._valido_k2(nueva_alc, nueva_mec, n_alc, n_mec):
                continue
            nueva_perdida, nueva_dist = self._evaluar_k2(
                nueva_alc, nueva_mec, alcance_total, mecanismo_total
            )
            delta = nueva_perdida - perdida
            if delta < 0 or rng.random() < math.exp(-delta / temp):
                alc = nueva_alc
                mec = nueva_mec
                perdida = nueva_perdida
                dist = nueva_dist

        return alc, mec, perdida, dist

    @staticmethod
    def _valido_k2(alc: np.ndarray, mec: np.ndarray, n_alc: int, n_mec: int) -> bool:
        """Descarta los dos estados triviales: (vacío,vacío) y (todo,todo).

        bipartir([],[]) y bipartir(all,all) dejan el sistema intacto → pérdida=0.
        Son los únicos casos excluidos por biparticiones().
        """
        todo_cero = int(alc.sum()) == 0 and int(mec.sum()) == 0
        todo_uno  = int(alc.sum()) == n_alc and int(mec.sum()) == n_mec
        return not todo_cero and not todo_uno

    # ------------------------------------------------------------------
    # k>2: bipartir sobre asignaciones de nodos con k_bipartir
    # ------------------------------------------------------------------

    def _parallel_tempering_kn(
        self, nodos: list[int], k: int
    ) -> tuple[tuple[int, ...], float, np.ndarray]:
        n = len(nodos)
        R = self.n_replicas
        rng = np.random.default_rng(73)

        ratio = (self.temp_max / self.temp_min) ** (1.0 / (R - 1))
        temps = [self.temp_min * (ratio ** i) for i in range(R)]

        estados: list[tuple[int, ...]] = []
        perdidas: list[float] = []
        dists: list[np.ndarray] = []

        for _ in range(R):
            base = list(range(k))
            resto = [int(rng.integers(0, k)) for _ in range(max(0, n - k))]
            asig = base + resto
            rng.shuffle(asig)
            asig_c = self._canonicalizar(tuple(asig))
            p, d = self._evaluar_kn(nodos, asig_c)
            estados.append(asig_c)
            perdidas.append(p)
            dists.append(d)

        mejor_asig = estados[0]
        mejor_perdida = perdidas[0]
        mejor_dist = dists[0]
        for i in range(R):
            if perdidas[i] < mejor_perdida:
                mejor_perdida = perdidas[i]
                mejor_asig = estados[i]
                mejor_dist = dists[i]

        for ronda in range(self.n_rondas):
            for i in range(R):
                estados[i], perdidas[i], dists[i] = self._mcmc_pasos_kn(
                    nodos, estados[i], perdidas[i], dists[i], k, temps[i], rng
                )
                if perdidas[i] < mejor_perdida:
                    mejor_perdida = perdidas[i]
                    mejor_asig = estados[i]
                    mejor_dist = dists[i]

            offset = ronda % 2
            for i in range(offset, R - 1, 2):
                j = i + 1
                delta_log = (perdidas[i] - perdidas[j]) * (1.0 / temps[i] - 1.0 / temps[j])
                if delta_log >= 0 or rng.random() < math.exp(delta_log):
                    estados[i], estados[j] = estados[j], estados[i]
                    perdidas[i], perdidas[j] = perdidas[j], perdidas[i]
                    dists[i], dists[j] = dists[j], dists[i]
                    if perdidas[i] < mejor_perdida:
                        mejor_perdida = perdidas[i]
                        mejor_asig = estados[i]
                        mejor_dist = dists[i]

        return mejor_asig, mejor_perdida, mejor_dist

    def _evaluar_kn(
        self, nodos: list[int], asig: tuple[int, ...]
    ) -> tuple[float, np.ndarray]:
        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None
        if len(set(asig)) < 2:
            asig = tuple(i % 2 for i in range(len(nodos)))
        sp = self.sia_subsistema.k_bipartir(nodos, asig)
        dist = sp.distribucion_marginal()
        ref = self.sia_dists_marginales
        if dist.size != ref.size:
            alineada = np.zeros_like(ref)
            alineada[: dist.size] = dist
            dist = alineada
        return float(self.distancia_metrica(ref, dist)), dist

    def _mcmc_pasos_kn(
        self,
        nodos: list[int],
        asig: tuple[int, ...],
        perdida: float,
        dist: np.ndarray,
        k: int,
        temp: float,
        rng: np.random.Generator,
    ) -> tuple[tuple[int, ...], float, np.ndarray]:
        n = len(nodos)
        for _ in range(self.pasos_por_ronda):
            nueva = list(asig)
            if n >= 2 and rng.random() < 0.5:
                i = int(rng.integers(0, n))
                j = int(rng.integers(0, n - 1))
                if j >= i:
                    j += 1
                nueva[i], nueva[j] = nueva[j], nueva[i]
            else:
                idx = int(rng.integers(0, n))
                nueva[idx] = int(rng.integers(0, k))
            asig_nueva = self._canonicalizar(tuple(nueva))
            if len(set(asig_nueva)) < 2:
                continue
            nueva_perdida, nueva_dist = self._evaluar_kn(nodos, asig_nueva)
            delta = nueva_perdida - perdida
            if delta < 0 or rng.random() < math.exp(-delta / temp):
                asig = asig_nueva
                perdida = nueva_perdida
                dist = nueva_dist
        return asig, perdida, dist

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

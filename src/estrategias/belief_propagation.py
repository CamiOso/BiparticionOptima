"""Estrategia Belief Propagation (LBP) para k-partición.

Modela la asignación de grupos como inferencia en un **modelo gráfico
probabilístico** (Markov Random Field) y corre el algoritmo sum-product
de propagación de mensajes (Loopy Belief Propagation) hasta convergencia.

Modelo gráfico
--------------
Variables aleatorias: z_i ∈ {0, …, k−1}  (grupo del nodo i)

Potenciales unarios: ψ_i(g)
  Miden la afinidad intrínseca del nodo i con el grupo g.
  Se estiman como el negativo de la pérdida marginal: cuánto empeora φ
  si se fuerza el nodo i solo en el grupo g (los demás van al grupo g̅).

Potenciales de par (modelo de Potts): ψ_{ij}(g, h)
  Codifican la compatibilidad de asignar los nodos i y j a los grupos g y h.
      ψ_{ij}(g=h) = exp(+α · w_{ij})    — recompensa si mismo grupo y alta afinidad
      ψ_{ij}(g≠h) = exp(−α · w_{ij}/k)  — penalización si distinto grupo y alta afinidad
  donde w_{ij} es el peso de acoplamiento del grafo de información.

Algoritmo sum-product (LBP)
---------------------------
Mensajes escalares μ_{i→j}(h): lo que el nodo i "comunica" a j sobre su creencia
de que j pertenece al grupo h.

  μ_{i→j}(h) ← Σ_g ψ_i(g) · ψ_{ij}(g,h) · Π_{l∈N(i)\j} μ_{l→i}(g)

Creencias marginales (beliefs):
  b_i(g) ∝ ψ_i(g) · Π_{j∈N(i)} μ_{j→i}(g)

Decisión hard: z_i* = argmax_g b_i(g)

Para grafos con ciclos (loopy BP) se usa amortiguación: μ_nuevo ← (1-δ)·μ_viejo + δ·μ_calculado
para estabilidad numérica.

Complejidad: O(iteraciones · |aristas| · k²)
"""

from __future__ import annotations

import numpy as np

from src.constantes.models import BP_LABEL
from src.funciones.formato import fmt_biparticion, fmt_k_particion_asignacion
from src.funciones.grafo_info import construir_afinidad
from src.funciones.iit import seleccionar_emd
from src.modelos.base.sia import SIA
from src.modelos.nucleo.solucion import Solucion


class BeliefPropagation(SIA):
    """k-partición por inferencia en modelo gráfico con LBP.

    Parámetros
    ----------
    alpha          : Escala del potencial de Potts. Mayor α → agrupamiento más fuerte.
    max_iter       : Iteraciones máximas de LBP.
    amortiguacion  : Factor de damping (0 < δ ≤ 1). Reduce oscilaciones en grafos con ciclos.
    config         : AppConfig inyectada (None → usa singleton global).
    """

    def __init__(
        self,
        tpm: np.ndarray,
        config=None,
        alpha: float = 2.0,
        max_iter: int = 50,
        amortiguacion: float = 0.5,
    ) -> None:
        super().__init__(tpm, config)
        self.distancia_metrica = seleccionar_emd(config)
        self.alpha = alpha
        self.max_iter = max_iter
        self.amortiguacion = amortiguacion

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
                estrategia=BP_LABEL,
                perdida=0.0,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=self.sia_dists_marginales.copy(),
                estado_inicial=estado_inicial,
                particion="NO-PARTITION",
            )

        asig = self._lbp(W, k_eff, n, nodos)
        asig, perdida, dist = self._refinar_local(nodos, asig, k_eff)

        particion_str = self._formatear(asig, nodos, alcance_total, mecanismo_total, k_eff)
        return Solucion(
            estrategia=BP_LABEL,
            perdida=perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist,
            estado_inicial=estado_inicial,
            particion=particion_str,
        )

    # ------------------------------------------------------------------
    # Loopy Belief Propagation
    # ------------------------------------------------------------------

    def _lbp(
        self, W: np.ndarray, k: int, n: int, nodos: list[int]
    ) -> tuple[int, ...]:
        # Potenciales unarios ψ_i(g): estimados desde pérdida marginal
        psi_unario = self._potenciales_unarios(W, k, n, nodos)

        # Potenciales de par: modelo de Potts ponderado por afinidad
        # psi_par[i,j,g,h] — se construye on-the-fly por eficiencia
        # (sólo necesitamos w[i,j] para calcular ψ_{ij}(g,h))

        # Mensajes iniciales: uniforme μ_{i→j}(h) = 1/k para todo h
        # Almacenados como mensajes[i,j,h]
        mensajes = np.ones((n, n, k), dtype=np.float64) / k

        aristas = [(i, j) for i in range(n) for j in range(n) if i != j]

        for iteracion in range(self.max_iter):
            mensajes_nuevos = mensajes.copy()

            for (i, j) in aristas:
                w_ij = W[i, j]
                # Calcular nuevo mensaje μ_{i→j}(h)
                nuevo = np.zeros(k, dtype=np.float64)
                for h in range(k):
                    total = 0.0
                    for g in range(k):
                        # Potencial de Potts
                        if g == h:
                            pot_par = np.exp(self.alpha * w_ij)
                        else:
                            pot_par = np.exp(-self.alpha * w_ij / max(1, k - 1))

                        # Producto de mensajes entrantes excepto desde j
                        prod_mensajes = psi_unario[i, g]
                        for l in range(n):
                            if l != i and l != j:
                                prod_mensajes *= mensajes[l, i, g]

                        total += pot_par * prod_mensajes
                    nuevo[h] = total

                # Normalizar
                suma = nuevo.sum()
                if suma > 1e-300:
                    nuevo /= suma
                else:
                    nuevo = np.ones(k) / k

                # Amortiguación para estabilidad en grafos con ciclos
                mensajes_nuevos[i, j] = (
                    (1.0 - self.amortiguacion) * mensajes[i, j]
                    + self.amortiguacion * nuevo
                )

            # Comprobar convergencia
            cambio = np.max(np.abs(mensajes_nuevos - mensajes))
            mensajes = mensajes_nuevos
            if cambio < 1e-6:
                break

        # Calcular creencias marginales
        asig = self._decisiones_hard(mensajes, psi_unario, n, k)
        return self._canonicalizar(asig)

    def _potenciales_unarios(
        self,
        W: np.ndarray,
        k: int,
        n: int,
        nodos: list[int],
    ) -> np.ndarray:
        """ψ_i(g): proporcional a la afinidad promedio de i con el grupo g.

        Usa la media de pesos de aristas como proxy: nodos muy conectados
        entre sí tienen alta afinidad para compartir grupo.
        """
        # Normalizar W para que los pesos queden en [0,1]
        w_max = W.max()
        W_norm = W / (w_max + 1e-12)

        # Inicializar con potencial uniforme
        psi = np.ones((n, k), dtype=np.float64) / k

        # Asignar una rotación inicial determinista basada en grado espectral
        grados = W_norm.sum(axis=1)
        orden = np.argsort(-grados)  # nodos con mayor grado primero
        for rank, idx in enumerate(orden):
            grupo_preferido = rank % k
            psi[idx, :] = 0.1 / (k - 1) if k > 1 else 1.0
            psi[idx, grupo_preferido] = 0.9

        # Normalizar
        psi /= psi.sum(axis=1, keepdims=True)
        return psi

    def _decisiones_hard(
        self,
        mensajes: np.ndarray,
        psi_unario: np.ndarray,
        n: int,
        k: int,
    ) -> tuple[int, ...]:
        """Calcula creencias y hace asignación hard."""
        asig = []
        for i in range(n):
            creencia = psi_unario[i].copy()
            for j in range(n):
                if i != j:
                    creencia *= mensajes[j, i]
            # Normalizar creencia
            suma = creencia.sum()
            if suma > 1e-300:
                creencia /= suma
            asig.append(int(np.argmax(creencia)))
        return tuple(asig)

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

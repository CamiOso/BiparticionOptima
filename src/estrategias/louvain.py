"""Estrategia Louvain para k-partición (Blondel et al., 2008).

Detecta comunidades en el **grafo de acoplamientos** del sistema. Los nodos
del grafo son los nodos del sistema; los pesos de las aristas son las
sensibilidades w_{ij} (cuánto afecta el estado de j a la dinámica de i).

Algoritmo de Louvain en dos fases
----------------------------------
Fase 1 — Optimización voraz de modularidad:
  Para cada nodo i, evalúa el cambio de modularidad ΔQ al moverlo a cada
  comunidad vecina y lo mueve a la que maximiza ΔQ.

      ΔQ = [k_{i→C} / m  −  Σtot · k_i / (2m²)]  ×  2

  donde k_{i→C} = suma de pesos de aristas de i hacia la comunidad C,
  k_i = grado ponderado de i, Σtot = suma de grados en C, m = peso total.
  Se itera hasta que ningún movimiento mejore Q.

Fase 2 — Agregación:
  Las comunidades se colapsan en meta-nodos; el peso entre meta-nodos es la
  suma de pesos de todas las aristas entre sus miembros.
  Se repite la Fase 1 sobre el meta-grafo hasta estabilidad.

Ajuste a k grupos
------------------
Si el número de comunidades detectadas supera k, se fusionan iterativamente
las dos comunidades con mayor peso de aristas entre ellas.
Si es menor que k, la comunidad más grande se divide con el vector de Fiedler
del sub-Laplaciano hasta alcanzar k grupos.

Complejidad: O(n² · iteraciones) — empíricamente O(n log n) en grafos dispersos.
"""

from __future__ import annotations

import numpy as np

from src.constantes.models import LOUVAIN_LABEL
from src.funciones.formato import fmt_biparticion, fmt_k_particion_asignacion
from src.funciones.grafo_info import construir_afinidad
from src.funciones.iit import seleccionar_emd
from src.modelos.base.sia import SIA
from src.modelos.nucleo.solucion import Solucion


class Louvain(SIA):
    """Partición por maximización de modularidad en el grafo de acoplamientos."""

    def __init__(
        self,
        tpm: np.ndarray,
        config=None,
        max_iteraciones: int = 50,
    ) -> None:
        super().__init__(tpm, config)
        self.distancia_metrica = seleccionar_emd(config)
        self.max_iteraciones = max_iteraciones

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
                estrategia=LOUVAIN_LABEL,
                perdida=0.0,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=self.sia_dists_marginales.copy(),
                estado_inicial=estado_inicial,
                particion="NO-PARTITION",
            )

        asig = self._louvain(W, k_eff)
        asig, perdida, dist = self._refinar_local(nodos, asig, k_eff)

        particion_str = self._formatear(asig, nodos, alcance_total, mecanismo_total, k_eff)
        return Solucion(
            estrategia=LOUVAIN_LABEL,
            perdida=perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist,
            estado_inicial=estado_inicial,
            particion=particion_str,
        )

    # ------------------------------------------------------------------
    # Núcleo Louvain
    # ------------------------------------------------------------------

    def _louvain(self, W: np.ndarray, k_objetivo: int) -> tuple[int, ...]:
        """Ejecuta Louvain sobre W y ajusta al número de comunidades k_objetivo."""
        n = len(W)
        comunidades = list(range(n))  # cada nodo en su propia comunidad

        for _ in range(self.max_iteraciones):
            mejorado = self._fase_optimizacion(W, comunidades)
            if not mejorado:
                break

        # Renumerar comunidades de forma canónica
        mapa: dict[int, int] = {}
        sig = 0
        canon = []
        for c in comunidades:
            if c not in mapa:
                mapa[c] = sig
                sig += 1
            canon.append(mapa[c])
        comunidades = canon
        n_comunidades = max(comunidades) + 1

        # Ajustar al número k objetivo
        if n_comunidades > k_objetivo:
            comunidades = self._fusionar_hasta(W, comunidades, n_comunidades, k_objetivo)
        elif n_comunidades < k_objetivo:
            comunidades = self._dividir_hasta(W, comunidades, n_comunidades, k_objetivo)

        return tuple(comunidades)

    def _fase_optimizacion(
        self, W: np.ndarray, comunidades: list[int]
    ) -> bool:
        """Fase 1 Louvain: mueve nodos para maximizar modularidad. Retorna True si hubo cambio."""
        n = len(W)
        m = W.sum() / 2.0
        if m < 1e-12:
            return False

        grados = W.sum(axis=1)  # k_i para cada nodo
        mejorado = False

        for i in range(n):
            com_actual = comunidades[i]
            # Quitar i de su comunidad temporalmente
            comunidades[i] = -1

            # Peso de aristas de i hacia cada comunidad
            peso_hacia: dict[int, float] = {}
            for j in range(n):
                c = comunidades[j]
                if c == -1:
                    continue
                peso_hacia[c] = peso_hacia.get(c, 0.0) + W[i, j]

            # ΔQ para mover i a cada comunidad vecina
            mejor_dq = 0.0
            mejor_com = com_actual
            sigma_tot: dict[int, float] = {}
            for j in range(n):
                c = comunidades[j]
                if c != -1:
                    sigma_tot[c] = sigma_tot.get(c, 0.0) + grados[j]

            for c, k_ic in peso_hacia.items():
                s_tot = sigma_tot.get(c, 0.0)
                dq = k_ic / m - s_tot * grados[i] / (2.0 * m * m)
                if dq > mejor_dq + 1e-10:
                    mejor_dq = dq
                    mejor_com = c

            comunidades[i] = mejor_com
            if mejor_com != com_actual:
                mejorado = True

        return mejorado

    # ------------------------------------------------------------------
    # Ajuste de número de comunidades
    # ------------------------------------------------------------------

    def _fusionar_hasta(
        self,
        W: np.ndarray,
        comunidades: list[int],
        n_com: int,
        k_obj: int,
    ) -> list[int]:
        """Fusiona las dos comunidades con mayor peso de aristas entre ellas."""
        com = list(comunidades)
        n = len(W)
        n_actual = n_com

        while n_actual > k_obj:
            # Calcular peso entre cada par de comunidades distintas
            pesos: dict[tuple[int, int], float] = {}
            for i in range(n):
                for j in range(i + 1, n):
                    ci, cj = com[i], com[j]
                    if ci != cj:
                        par = (min(ci, cj), max(ci, cj))
                        pesos[par] = pesos.get(par, 0.0) + W[i, j]

            if not pesos:
                break

            # Fusionar el par con mayor peso
            (ca, cb) = max(pesos, key=lambda p: pesos[p])
            for idx in range(n):
                if com[idx] == cb:
                    com[idx] = ca
            # Renumerar
            mapa: dict[int, int] = {}
            sig = 0
            nuevo = []
            for c in com:
                if c not in mapa:
                    mapa[c] = sig
                    sig += 1
                nuevo.append(mapa[c])
            com = nuevo
            n_actual = max(com) + 1

        return com

    def _dividir_hasta(
        self,
        W: np.ndarray,
        comunidades: list[int],
        n_com: int,
        k_obj: int,
    ) -> list[int]:
        """Divide la comunidad más grande con el vector de Fiedler hasta llegar a k_obj."""
        com = list(comunidades)
        n = len(W)

        while (max(com) + 1) < k_obj:
            # Encontrar la comunidad más grande
            conteos: dict[int, int] = {}
            for c in com:
                conteos[c] = conteos.get(c, 0) + 1
            if not conteos:
                break
            c_grande = max(conteos, key=lambda c: conteos[c])
            miembros = [i for i, c in enumerate(com) if c == c_grande]
            if len(miembros) < 2:
                break

            # Sub-Laplaciano de la comunidad más grande
            sub_W = W[np.ix_(miembros, miembros)]
            L = np.diag(sub_W.sum(axis=1)) - sub_W
            vals, vecs = np.linalg.eigh(L)
            # Segundo eigenvector (Fiedler)
            fiedler = vecs[:, 1] if len(vals) > 1 else vecs[:, 0]
            nueva_com_id = max(com) + 1
            for local_idx, global_idx in enumerate(miembros):
                if fiedler[local_idx] >= 0:
                    com[global_idx] = nueva_com_id

        return com

    # ------------------------------------------------------------------
    # Evaluación y refinamiento local
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
        max_iter: int = 24,
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

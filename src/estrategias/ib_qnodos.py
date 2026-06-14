"""Estrategia hibrida IB + QNodos.

Information Bottleneck (Tishby et al., 1999) genera una semilla de alta
calidad agrupando nodos por similitud de perfiles causales. QNodos refina
esa semilla con simulated annealing usando la metrica phi directamente,
combinando la velocidad y alineacion teorica de IB con la precision
empirica de QNodos.

Flujo
-----
k = 2:
    IB -> semilla (subalcance, submecanismo) -> SA de QNodos
k > 2:
    IB -> semilla (asignacion) -> BuscadorKRecocido.buscar_con_semilla

Complejidad: O(n^2 * k * iter_IB + n * k * iter_SA)
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.constantes.models import IB_QNODES_LABEL
from src.estrategias.informacion_bottleneck import InformacionBottleneck, _kl_suavizada
from src.estrategias.q_nodos import (
    QNodos,
    _BuscadorKQNodos,
    _alinear_distribucion,
    _grupos_desde_asignacion,
)
from src.funciones.formato import fmt_biparticion, fmt_k_particion_asignacion
from src.funciones.iit import seleccionar_emd
from src.modelos.base.aplicacion import aplicacion
from src.modelos.base.sia import SIA
from src.modelos.nucleo.solucion import Solucion


class IBQNodos(SIA):
    """Hibrido: Information Bottleneck como semilla + QNodos como refinador.

    Parametros
    ----------
    beta : float
        Factor de compresion para IB (por defecto 4.0).
    max_iter_ib : int
        Iteraciones maximas de la minimizacion alternada IB.
    n_restarts_ib : int
        Reinicios aleatorios de IB para elegir la mejor semilla.
    temp_inicial : float
        Temperatura inicial del SA (por defecto 1.0).
    temp_final : float
        Temperatura final del SA (por defecto 0.001).
    factor_enfriamiento : float
        Factor de enfriamiento del SA (por defecto 0.92).
    pasos_por_temp : int
        Pasos por nivel de temperatura en SA (por defecto 30).
    """

    def __init__(
        self,
        tpm: np.ndarray,
        config=None,
        beta: float = 4.0,
        max_iter_ib: int = 60,
        n_restarts_ib: int = 8,
        temp_inicial: float = 1.0,
        temp_final: float = 0.001,
        factor_enfriamiento: float = 0.92,
        pasos_por_temp: int = 30,
    ) -> None:
        super().__init__(tpm, config)
        self.distancia_metrica = seleccionar_emd(config)
        self.beta = beta
        self.max_iter_ib = max_iter_ib
        self.n_restarts_ib = n_restarts_ib
        self.temp_inicial = temp_inicial
        self.temp_final = temp_final
        self.factor_enfriamiento = factor_enfriamiento
        self.pasos_por_temp = pasos_por_temp

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
        if k < 2:
            raise ValueError(f"k debe ser >= 2, se recibio {k}.")

        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)
        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None

        alcance_total = tuple(int(v) for v in self.sia_subsistema.indices_ncubos.tolist())
        mecanismo_total = tuple(int(v) for v in self.sia_subsistema.dims_ncubos.tolist())

        nodos = sorted(
            set(int(v) for v in self.sia_subsistema.indices_ncubos.tolist())
            | set(int(v) for v in self.sia_subsistema.dims_ncubos.tolist())
        )
        n = len(nodos)
        k_eff = min(k, n)

        if n <= 1 or k_eff < 2:
            return Solucion(
                estrategia=IB_QNODES_LABEL,
                perdida=0.0,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=self.sia_dists_marginales.copy(),
                estado_inicial=estado_inicial,
                particion="NO-PARTITION",
            )

        # Fase 1: IB genera la semilla
        semilla_ib = self._generar_semilla_ib(nodos, k_eff)

        if k_eff == 2:
            return self._refinar_k2(
                estado_inicial, alcance_total, mecanismo_total, semilla_ib
            )
        return self._refinar_kn(
            estado_inicial, nodos, alcance_total, mecanismo_total, k_eff, semilla_ib
        )

    # ------------------------------------------------------------------
    # Fase 1: semilla via IB
    # ------------------------------------------------------------------

    def _generar_semilla_ib(
        self, nodos: list[int], k: int
    ) -> tuple[int, ...]:
        """Ejecuta IB con multi-start paralelo y retorna la mejor asignacion."""
        perfiles = self._extraer_perfiles(nodos)
        rng = np.random.default_rng(73)

        asignaciones = [
            self._ib_alternating(perfiles, k, np.random.default_rng(73 + r), r)
            for r in range(self.n_restarts_ib)
        ]

        def evaluar(asig):
            perdida, _ = self._evaluar_asig(nodos, asig)
            return perdida, asig

        n_workers = min(os.cpu_count() or 1, len(asignaciones))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            resultados = list(ex.map(evaluar, asignaciones))

        _, mejor_asig = min(resultados, key=lambda x: x[0])
        return mejor_asig

    # ------------------------------------------------------------------
    # Fase 2a: refinamiento SA para k=2
    # ------------------------------------------------------------------

    def _refinar_k2(
        self,
        estado_inicial: str,
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
        semilla_ib: tuple[int, ...],
    ) -> Solucion:
        """Convierte semilla IB a formato QNodos y refina con SA bipartito."""
        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None

        nodos_ordenados = sorted(
            set(int(v) for v in self.sia_subsistema.indices_ncubos.tolist())
            | set(int(v) for v in self.sia_subsistema.dims_ncubos.tolist())
        )

        # IB asigna grupo 0 o 1 a cada nodo. Extraer nodos del grupo 0.
        grupo0_nodos = {nodos_ordenados[i] for i, g in enumerate(semilla_ib) if g == 0}

        # Mapear a vertices QNodos: (0, idx)=mecanismo, (1, idx)=alcance
        vertices = list(
            [(0, int(i)) for i in self.sia_subsistema.dims_ncubos.tolist()]
            + [(1, int(i)) for i in self.sia_subsistema.indices_ncubos.tolist()]
        )
        full_set = set(vertices)
        grupo_a_inicial = {v for v in vertices if v[1] in grupo0_nodos}

        if not grupo_a_inicial or grupo_a_inicial == full_set:
            grupo_a_inicial = {vertices[0]}

        # SA de QNodos
        clave_sa, perdida_sa, dist_sa = self._sa_biparticion(
            vertices, grupo_a_inicial
        )

        subalc = tuple(v[1] for v in clave_sa if v[0] == 1)
        submec = tuple(v[1] for v in clave_sa if v[0] == 0)

        return Solucion(
            estrategia=IB_QNODES_LABEL,
            perdida=float(perdida_sa),
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist_sa,
            estado_inicial=estado_inicial,
            particion=fmt_biparticion(subalc, submec, alcance_total, mecanismo_total),
        )

    # ------------------------------------------------------------------
    # Fase 2b: refinamiento SA para k>2
    # ------------------------------------------------------------------

    def _refinar_kn(
        self,
        estado_inicial: str,
        nodos: list[int],
        alcance_total: tuple[int, ...],
        mecanismo_total: tuple[int, ...],
        k: int,
        semilla_ib: tuple[int, ...],
    ) -> Solucion:
        """Convierte semilla IB a asignacion sobre vertices y refina con BuscadorKRecocido."""
        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None

        # Construir vertices QNodos
        vertices = list(
            [(0, int(i)) for i in self.sia_subsistema.dims_ncubos.tolist()]
            + [(1, int(i)) for i in self.sia_subsistema.indices_ncubos.tolist()]
        )

        # Mapear semilla_ib (sobre nodos) a semilla_vertices
        nodo_a_grupo = {nodos[i]: g for i, g in enumerate(semilla_ib)}
        semilla_vertices = tuple(nodo_a_grupo.get(v[1], 0) for v in vertices)

        if len(set(semilla_vertices)) < 2:
            semilla_vertices = tuple(i % k for i in range(len(vertices)))

        cache: dict[tuple[int, ...], tuple[float, np.ndarray]] = {}
        buscador = _BuscadorKQNodos(
            vertices=vertices,
            sistema=self.sia_subsistema,
            dists=self.sia_dists_marginales,
            distancia_metrica=self.distancia_metrica,
            cache=cache,
        )
        buscador.temp_inicial = self.temp_inicial
        buscador.temp_final = self.temp_final
        buscador.factor_enfriamiento = self.factor_enfriamiento
        buscador.pasos_por_temp = self.pasos_por_temp

        resultado_k = buscador.buscar_con_semilla(
            k, semilla_vertices,
            semilla=aplicacion.semilla_numpy + len(vertices),
        )

        # Mapear resultado (sobre vertices) a asignacion sobre nodos
        asig_final: list[int] = []
        for nodo in nodos:
            g = 0
            for i, v in enumerate(vertices):
                if v[1] == nodo:
                    g = resultado_k.asignacion[i]
                    break
            asig_final.append(g)

        return Solucion(
            estrategia=IB_QNODES_LABEL,
            perdida=float(resultado_k.perdida),
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=resultado_k.distribucion,
            estado_inicial=estado_inicial,
            particion=self._formatear(
                tuple(asig_final), nodos, alcance_total, mecanismo_total, k
            ),
        )

    # ------------------------------------------------------------------
    # SA bipartito (replica de QNodos._sa_biparticion)
    # ------------------------------------------------------------------

    def _sa_biparticion(
        self,
        vertices: list[tuple[int, int]],
        grupo_a_inicial: set[tuple[int, int]],
    ) -> tuple[tuple[tuple[int, int], ...], float, np.ndarray]:
        """SA sobre biparticiones usando bipartir (mismo evaluador que QNodos)."""
        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None

        full_set = set(vertices)

        def evaluar(grupo_a: set) -> tuple[float, np.ndarray]:
            mec = [v for t, v in grupo_a if t == 0]
            alc = [v for t, v in grupo_a if t == 1]
            sp = self.sia_subsistema.bipartir(
                np.array(alc, dtype=np.int8),
                np.array(mec, dtype=np.int8),
            )
            dp = _alinear_distribucion(
                sp.distribucion_marginal(), self.sia_dists_marginales
            )
            return float(self.distancia_metrica(dp, self.sia_dists_marginales)), dp

        actual_a = set(grupo_a_inicial)
        actual_perdida, actual_dist = evaluar(actual_a)
        mejor_a = frozenset(actual_a)
        mejor_perdida = actual_perdida
        mejor_dist = actual_dist

        if mejor_perdida <= 1e-12:
            clave_sa = tuple(sorted(mejor_a, key=lambda v: (v[0], v[1])))
            return clave_sa, mejor_perdida, mejor_dist

        rng = np.random.default_rng(aplicacion.semilla_numpy + len(vertices))
        temp = self.temp_inicial
        niveles_sin_mejora = 0
        mejor_previa = mejor_perdida

        # Para mec grande (≥22 nodos) cada bipartir es O(2^n_mec) ≈ 3-7s/eval.
        # Reducir pasos_por_temp de 30 a 3 da 10× speedup sin perder calidad
        # cuando la semilla IB ya es buena.
        n_mec_vertices = sum(1 for t, _ in vertices if t == 0)
        pasos = self.pasos_por_temp if n_mec_vertices < 22 else max(3, self.pasos_por_temp // 10)

        while temp > self.temp_final:
            for _ in range(pasos):
                lista_a = list(actual_a)
                lista_b = list(full_set - actual_a)
                if not lista_a or not lista_b:
                    continue
                if lista_a and lista_b and rng.random() < 0.5:
                    v = lista_a[int(rng.integers(len(lista_a)))]
                    nueva_a = actual_a - {v}
                else:
                    v = lista_b[int(rng.integers(len(lista_b)))]
                    nueva_a = actual_a | {v}
                if not nueva_a or nueva_a == full_set:
                    continue
                nueva_perdida, nueva_dist = evaluar(nueva_a)
                delta = nueva_perdida - actual_perdida
                if delta < 0 or rng.random() < float(np.exp(-delta / temp)):
                    actual_a = nueva_a
                    actual_perdida = nueva_perdida
                    actual_dist = nueva_dist
                    if actual_perdida < mejor_perdida:
                        mejor_perdida = actual_perdida
                        mejor_a = frozenset(actual_a)
                        mejor_dist = actual_dist
            temp *= self.factor_enfriamiento
            if mejor_perdida <= 1e-12:
                break
            if mejor_perdida < mejor_previa:
                mejor_previa = mejor_perdida
                niveles_sin_mejora = 0
            else:
                niveles_sin_mejora += 1
                if niveles_sin_mejora >= 5 and temp < self.temp_inicial * 0.3:
                    break

        clave_sa = tuple(sorted(mejor_a, key=lambda v: (v[0], v[1])))
        return clave_sa, mejor_perdida, mejor_dist

    # ------------------------------------------------------------------
    # Metodos delegados de IB (perfiles + IB alternado + evaluacion)
    # ------------------------------------------------------------------

    def _extraer_perfiles(self, nodos: list[int]) -> np.ndarray:
        """Devuelve matriz (n_nodos, max_len) con el perfil causal de cada nodo."""
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

    def _ib_alternating(
        self,
        perfiles: np.ndarray,
        k: int,
        rng: np.random.Generator,
        restart: int,
    ) -> tuple[int, ...]:
        """Minimizacion alternada IB: retorna asignacion hard."""
        n = len(perfiles)
        pt_x = rng.dirichlet(np.ones(k), size=n)
        px = np.ones(n, dtype=np.float64) / n

        for _ in range(self.max_iter_ib):
            pt = pt_x.T @ px
            pt = np.clip(pt, 1e-12, None)
            pt /= pt.sum()

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

            log_nuevo = np.zeros((n, k), dtype=np.float64)
            for i in range(n):
                for t in range(k):
                    log_nuevo[i, t] = np.log(pt[t]) - self.beta * _kl_suavizada(
                        perfiles[i], centroides[t]
                    )
            log_nuevo -= log_nuevo.max(axis=1, keepdims=True)
            pt_x_new = np.exp(log_nuevo)
            pt_x_new /= pt_x_new.sum(axis=1, keepdims=True)

            if np.max(np.abs(pt_x_new - pt_x)) < 1e-8:
                break
            pt_x = pt_x_new

        return tuple(int(np.argmax(pt_x[i])) for i in range(n))

    def _evaluar_asig(
        self,
        nodos: list[int],
        asig: tuple[int, ...],
    ) -> tuple[float, np.ndarray]:
        """Evalua una asignacion usando bipartir (k=2) o k_bipartir (k>2)."""
        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None
        if len(set(asig)) < 2:
            asig = tuple(i % 2 for i in range(len(nodos)))

        k = max(asig) + 1
        if k == 2:
            alc_total = set(int(v) for v in self.sia_subsistema.indices_ncubos.tolist())
            mec_total = set(int(v) for v in self.sia_subsistema.dims_ncubos.tolist())
            alc = np.array([n for n, g in zip(nodos, asig) if g == 0 and n in alc_total], dtype=np.int8)
            mec = np.array([n for n, g in zip(nodos, asig) if g == 0 and n in mec_total], dtype=np.int8)
            sistema_partido = self.sia_subsistema.bipartir(alc, mec)
        else:
            sistema_partido = self.sia_subsistema.k_bipartir(nodos, asig)

        dist = sistema_partido.distribucion_marginal()
        if dist.size != self.sia_dists_marginales.size:
            alineada = np.zeros_like(self.sia_dists_marginales)
            alineada[: dist.size] = dist
            dist = alineada
        return float(self.distancia_metrica(self.sia_dists_marginales, dist)), dist

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
            sub_alc = tuple(
                nodos[i]
                for i, g in enumerate(asig)
                if g == 0 and nodos[i] in alcance_total
            )
            sub_mec = tuple(
                nodos[i]
                for i, g in enumerate(asig)
                if g == 0 and nodos[i] in mecanismo_total
            )
            return fmt_biparticion(sub_alc, sub_mec, alcance_total, mecanismo_total)
        return fmt_k_particion_asignacion(
            nodos, asig, alcance_total, mecanismo_total
        )

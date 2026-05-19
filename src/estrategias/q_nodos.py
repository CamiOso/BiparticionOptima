from dataclasses import dataclass

import numpy as np

from src.funciones.formato import fmt_biparticion_q, fmt_k_particion_q
from src.funciones.iit import seleccionar_emd
from src.funciones.k_particion_buscador import BuscadorKRecocido, ResultadoKParticion
from src.modelos.base.aplicacion import aplicacion
from src.modelos.base.sia import SIA
from src.modelos.nucleo.sistema import Sistema
from src.modelos.nucleo.solucion import Solucion


def _alinear_distribucion(distribucion: np.ndarray, referencia: np.ndarray) -> np.ndarray:
    if distribucion.size == referencia.size:
        return distribucion
    distribucion_alineada = np.zeros_like(referencia)
    distribucion_alineada[: distribucion.size] = distribucion
    return distribucion_alineada


def _grupos_desde_asignacion(
    asignacion: tuple[int, ...],
    vertices: list[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    if not asignacion:
        return []
    total_grupos = max(asignacion) + 1
    grupos: list[list[tuple[int, int]]] = [[] for _ in range(total_grupos)]
    for indice, grupo in enumerate(asignacion):
        grupos[grupo].append(vertices[indice])
    return grupos


class _BuscadorKQNodos(BuscadorKRecocido):
    """Buscador de k-particion para la estrategia QNodos.

    Evalua cada asignacion usando k_bipartir_temporal, que trabaja con pares
    (tiempo, indice) para separar el mecanismo (t=0) del alcance (t=1).

    Hereda de BuscadorKRecocido para escapar minimos locales con SA.
    """

    def __init__(
        self,
        vertices: list[tuple[int, int]],
        sistema: Sistema,
        dists: np.ndarray,
        distancia_metrica,
        cache: dict[tuple[int, ...], tuple[float, np.ndarray]],
    ) -> None:
        super().__init__(
            temp_inicial=1.0,
            temp_final=0.001,
            factor_enfriamiento=0.92,
            pasos_por_temp=30,
        )
        self.umbral_exacto = 8
        self._vertices = vertices
        self._sistema = sistema
        self._dists = dists
        self._distancia_metrica = distancia_metrica
        self._cache = cache

    def total_elementos(self) -> int:
        return len(self._vertices)

    def evaluar_asignacion(self, asignacion: tuple[int, ...]) -> tuple[float, np.ndarray]:
        en_cache = self._cache.get(asignacion)
        if en_cache is not None:
            return en_cache

        grupos = _grupos_desde_asignacion(asignacion, self._vertices)
        grupos_mecanismo = [
            tuple(indice for tiempo, indice in grupo if tiempo == 0)
            for grupo in grupos
        ]
        grupos_alcance = [
            tuple(indice for tiempo, indice in grupo if tiempo == 1)
            for grupo in grupos
        ]

        sistema_partido = self._sistema.k_bipartir_temporal(grupos_mecanismo, grupos_alcance)
        dist = _alinear_distribucion(sistema_partido.distribucion_marginal(), self._dists)
        perdida = float(self._distancia_metrica(dist, self._dists))
        self._cache[asignacion] = (perdida, dist)
        return perdida, dist


class QNodos(SIA):
    """Implementacion submodular de QNodos basada en deltas y omegas."""

    def __init__(self, tpm: np.ndarray, config=None) -> None:
        super().__init__(tpm, config)
        self.distancia_metrica = seleccionar_emd(config)
        self.memoria_delta: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[float, np.ndarray]] = {}
        self.memoria_grupo_candidato: dict[tuple[tuple[int, int], ...], tuple[float, np.ndarray | None]] = {}
        self.clave_submodular: list[list[int]] = [[], []]
        self.vertices: set[tuple[int, int]] = set()
        self._cache_k_particiones: dict[tuple[int, ...], tuple[float, np.ndarray]] = {}

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

        assert self.sia_dists_marginales is not None
        distribucion_subsistema = self.sia_dists_marginales

        assert self.sia_subsistema is not None
        self.memoria_delta.clear()
        self.memoria_grupo_candidato.clear()

        futuro = [(1, int(indice)) for indice in self.sia_subsistema.indices_ncubos.tolist()]
        presente = [(0, int(indice)) for indice in self.sia_subsistema.dims_ncubos.tolist()]
        vertices = list(presente + futuro)
        self.vertices = set(vertices)

        if k > 2:
            self._cache_k_particiones.clear()
            # Warm-start 1: arbol de contracciones — para despues de n-k pasos de Queyranne.
            semilla_asig = self._k_particion_arbol_contracciones(vertices, k)
            # Warm-start 2 (fallback): particion recursiva Q con memoizacion DP.
            if semilla_asig is None:
                memo_q: dict[tuple, tuple[int, ...]] = {}
                semilla_asig = self._particionar_recursivo_q(vertices, k, memo_q)

            buscador = _BuscadorKQNodos(
                vertices=vertices,
                sistema=self.sia_subsistema,
                dists=self.sia_dists_marginales,
                distancia_metrica=self.distancia_metrica,
                cache=self._cache_k_particiones,
            )
            if semilla_asig is not None:
                resultado_k = buscador.buscar_con_semilla(
                    k, semilla_asig, semilla=aplicacion.semilla_numpy + len(vertices)
                )
            else:
                resultado_k = buscador.buscar(k, semilla=aplicacion.semilla_numpy + len(vertices))
            grupos = _grupos_desde_asignacion(resultado_k.asignacion, vertices)
            return Solucion(
                estrategia="QNodos",
                perdida=float(resultado_k.perdida),
                distribucion_subsistema=distribucion_subsistema,
                distribucion_particion=resultado_k.distribucion,
                estado_inicial=estado_inicial,
                particion=fmt_k_particion_q(grupos),
            )

        # MAO da el corte inicial; SA refina usando el mismo evaluador (bipartir)
        # para escapar minimos locales por no-submodularidad de f.
        clave_mip, perdida_mao, dist_mao = self._mao_multi_start(vertices)
        assert dist_mao is not None

        grupo_a_inicial = set(clave_mip)
        clave_sa, perdida_sa, dist_sa = self._sa_biparticion(vertices, grupo_a_inicial)

        if perdida_sa < perdida_mao - 1e-9:
            clave_mip = clave_sa
            perdida_mip = perdida_sa
            distribucion_particion = dist_sa
        else:
            perdida_mip = perdida_mao
            distribucion_particion = dist_mao

        biparticion = fmt_biparticion_q(
            list(clave_mip),
            self.nodos_complemento(list(clave_mip)),
        )
        return Solucion(
            estrategia="QNodos",
            perdida=float(perdida_mip),
            distribucion_subsistema=distribucion_subsistema,
            distribucion_particion=distribucion_particion,
            estado_inicial=estado_inicial,
            particion=biparticion,
        )

    def _sa_biparticion(
        self,
        vertices: list[tuple[int, int]],
        grupo_a_inicial: set[tuple[int, int]],
        temp_inicial: float = 1.0,
        temp_final: float = 0.001,
        factor: float = 0.92,
        pasos_por_temp: int = 30,
    ) -> tuple[tuple[tuple[int, int], ...], float, np.ndarray]:
        """SA sobre biparticiones usando el mismo evaluador (bipartir) que MAO.

        Realiza movimientos de flip (mover un vertice de A a B o viceversa).
        Al usar bipartir directamente no hay inconsistencia con el resultado MAO.
        """
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
            dp = _alinear_distribucion(sp.distribucion_marginal(), self.sia_dists_marginales)
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
        temp = temp_inicial

        while temp > temp_final:
            for _ in range(pasos_por_temp):
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
            temp *= factor
            if mejor_perdida <= 1e-12:
                break

        clave_sa = tuple(sorted(mejor_a, key=lambda v: (v[0], v[1])))
        return clave_sa, mejor_perdida, mejor_dist

    def _mao_multi_start(
        self,
        vertices: list[tuple[int, int]],
        n_starts: int = 8,
    ) -> tuple[tuple[tuple[int, int], ...], float, np.ndarray | None]:
        """MAO con multiples ordenes de inicio para funciones no submodulares.

        Rota el vertice de arranque hasta n_starts veces. Cada rotacion puede
        encontrar un corte minimo distinto cuando f viola submodularidad. Usa el
        mismo evaluador (bipartir) que algoritmo_q para garantizar consistencia.
        """
        n = len(vertices)
        n_starts = min(n_starts, n)

        mejor_perdida = float("inf")
        mejor_clave: tuple[tuple[int, int], ...] | None = None
        mejor_dist: np.ndarray | None = None

        mem_candidato_global: dict[tuple, tuple[float, np.ndarray | None]] = {}

        for i in range(n_starts):
            k = (i * max(1, n // n_starts)) % n
            orden = vertices[k:] + vertices[:k]

            mem_prev = dict(self.memoria_grupo_candidato)
            self.memoria_grupo_candidato.clear()
            try:
                clave = self.algoritmo_q(orden)
                perdida, dist = self.memoria_grupo_candidato.get(clave, (float("inf"), None))
                if perdida < mejor_perdida and dist is not None:
                    mejor_perdida = perdida
                    mejor_clave = clave
                    mejor_dist = dist
                mem_candidato_global.update(self.memoria_grupo_candidato)
            except Exception:
                pass
            self.memoria_grupo_candidato = mem_prev
            if mejor_perdida <= 1e-12:
                break

        self.memoria_grupo_candidato.update(mem_candidato_global)
        if mejor_clave is None:
            mejor_clave = self.algoritmo_q(vertices)
            mejor_perdida, mejor_dist = self.memoria_grupo_candidato[mejor_clave]

        return mejor_clave, mejor_perdida, mejor_dist

    def algoritmo_q(
        self,
        vertices: list[tuple[int, int] | list[tuple[int, int]]],
    ) -> tuple[tuple[int, int], ...]:
        for _ in range(len(vertices) - 1):
            omegas_ciclo = [vertices[0]]
            deltas_ciclo = vertices[1:]

            emd_particion_candidata = np.inf
            dist_particion_candidata: np.ndarray | None = None

            for _ in range(max(0, len(deltas_ciclo) - 1)):
                emd_local = np.inf
                indice_mip = 0

                for indice_delta, delta in enumerate(deltas_ciclo):
                    emd_union, emd_delta, dist_marginal_delta = self.funcion_submodular(
                        delta,
                        omegas_ciclo,
                    )

                    emd_iteracion = emd_union - emd_delta
                    if emd_iteracion < emd_local:
                        if emd_delta == 0.0:
                            clave = self._normalizar_grupo(delta)
                            self.memoria_grupo_candidato[clave] = (
                                emd_delta,
                                dist_marginal_delta,
                            )
                            return clave

                        emd_local = emd_iteracion
                        indice_mip = indice_delta
                        emd_particion_candidata = emd_delta
                        dist_particion_candidata = dist_marginal_delta

                omegas_ciclo.append(deltas_ciclo[indice_mip])
                deltas_ciclo.pop(indice_mip)

            if deltas_ciclo:
                clave_final = self._normalizar_grupo(deltas_ciclo[-1])
                self.memoria_grupo_candidato[clave_final] = (
                    float(emd_particion_candidata),
                    dist_particion_candidata,
                )

                ultimo_omega = omegas_ciclo.pop()
                nuevo_grupo = self._desplegar_nodos(ultimo_omega) + self._desplegar_nodos(
                    deltas_ciclo[-1]
                )
                omegas_ciclo.append(nuevo_grupo)
                vertices = omegas_ciclo

        return min(
            self.memoria_grupo_candidato,
            key=lambda clave: self.memoria_grupo_candidato[clave][0],
        )

    def funcion_submodular(
        self,
        delta: tuple[int, int] | list[tuple[int, int]],
        omegas: list[tuple[int, int] | list[tuple[int, int]]],
    ) -> tuple[float, float, np.ndarray]:
        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None

        self.clave_submodular = [[], []]
        mecanismo_delta, alcance_delta = self.definir_clave(delta)
        clave_delta = (tuple(mecanismo_delta), tuple(alcance_delta))

        if clave_delta not in self.memoria_delta:
            particion_delta = self.sia_subsistema.bipartir(
                np.array(alcance_delta, dtype=np.int8),
                np.array(mecanismo_delta, dtype=np.int8),
            )
            vector_delta = _alinear_distribucion(
                particion_delta.distribucion_marginal(),
                self.sia_dists_marginales,
            )
            emd_delta = float(self.distancia_metrica(vector_delta, self.sia_dists_marginales))
            self.memoria_delta[clave_delta] = (emd_delta, vector_delta)
        else:
            emd_delta, vector_delta = self.memoria_delta[clave_delta]

        for omega in omegas:
            self.definir_clave(omega)

        mecanismos_union, alcances_union = self.clave_submodular[0], self.clave_submodular[1]
        particion_union = self.sia_subsistema.bipartir(
            np.array(alcances_union, dtype=np.int8),
            np.array(mecanismos_union, dtype=np.int8),
        )
        vector_union = _alinear_distribucion(
            particion_union.distribucion_marginal(),
            self.sia_dists_marginales,
        )
        emd_union = float(self.distancia_metrica(vector_union, self.sia_dists_marginales))

        return emd_union, emd_delta, vector_delta

    def definir_clave(
        self,
        conjunto: tuple[int, int] | list[tuple[int, int]],
    ) -> tuple[list[int], list[int]]:
        for tiempo, indice in self._desplegar_nodos(conjunto):
            self.clave_submodular[tiempo].append(indice)
        self.clave_submodular[0] = sorted(set(self.clave_submodular[0]))
        self.clave_submodular[1] = sorted(set(self.clave_submodular[1]))
        return self.clave_submodular[0], self.clave_submodular[1]

    def nodos_complemento(self, nodos: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return sorted(list(self.vertices - set(nodos)), key=lambda v: (v[0], v[1]))

    def _normalizar_grupo(
        self,
        conjunto: tuple[int, int] | list[tuple[int, int]],
    ) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(self._desplegar_nodos(conjunto), key=lambda v: (v[0], v[1])))

    def _desplegar_nodos(
        self,
        conjunto: tuple[int, int] | list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        if isinstance(conjunto, tuple) and len(conjunto) == 2 and all(
            isinstance(v, int) for v in conjunto
        ):
            return [conjunto]
        return [(int(t), int(i)) for t, i in conjunto]

    def _k_particion_arbol_contracciones(
        self,
        vertices: list[tuple[int, int]],
        k: int,
    ) -> tuple[int, ...] | None:
        """K-particion desde el arbol de contracciones de Queyranne.

        Ejecuta exactamente n-k pasos del algoritmo de Queyranne (MAO +
        contraccion), que corresponden a las n-k fusiones mas "costosas"
        (fuerte acoplamiento). Los k grupos resultantes son los componentes
        activos al terminar esos n-k pasos.

        No hace early-stopping para no saltarse contracciones intermedias.
        Guarda y restaura el estado de la instancia para no interferir con
        otras llamadas.
        """
        n = len(vertices)
        if n <= 1 or k >= n:
            return None
        k_eff = min(k, n)
        n_pasos = n - k_eff  # contracciones a realizar

        # Guardar estado
        vertices_prev = self.vertices
        mem_prev = dict(self.memoria_grupo_candidato)
        self.vertices = set(vertices)
        self.memoria_grupo_candidato.clear()

        parent: dict[tuple[int, int], tuple[int, int]] = {v: v for v in vertices}

        def find(x: tuple[int, int]) -> tuple[int, int]:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: tuple[int, int], b: tuple[int, int]) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        try:
            vertices_act: list = list(vertices)
            for _ in range(n_pasos):
                if len(vertices_act) < 2:
                    break

                omegas: list = [vertices_act[0]]
                deltas: list = list(vertices_act[1:])
                emd_candidata = float("inf")
                dist_candidata: np.ndarray | None = None

                for _ in range(max(0, len(deltas) - 1)):
                    emd_local = float("inf")
                    mejor_idx = 0
                    for idx, delta in enumerate(deltas):
                        emd_u, emd_d, dist_d = self.funcion_submodular(delta, omegas)
                        ei = emd_u - emd_d
                        if ei < emd_local:
                            emd_local = ei
                            mejor_idx = idx
                            emd_candidata = emd_d
                            dist_candidata = dist_d
                    omegas.append(deltas[mejor_idx])
                    deltas.pop(mejor_idx)

                if deltas:
                    pendant = deltas[-1]
                    penultimate = omegas[-1]
                    clave = self._normalizar_grupo(pendant)
                    self.memoria_grupo_candidato[clave] = (float(emd_candidata), dist_candidata)

                    orig_p = self._desplegar_nodos(pendant)
                    orig_pp = self._desplegar_nodos(penultimate)
                    todos = orig_pp + orig_p
                    for v in todos[1:]:
                        union(todos[0], v)

                    nuevo = orig_pp + orig_p
                    omegas.pop()
                    omegas.append(nuevo)
                    vertices_act = omegas

        except Exception:
            self.vertices = vertices_prev
            self.memoria_grupo_candidato = mem_prev
            return None

        self.vertices = vertices_prev
        self.memoria_grupo_candidato = mem_prev

        raices: dict[tuple[int, int], int] = {}
        sig = 0
        asig: list[int] = []
        for v in vertices:
            r = find(v)
            if r not in raices:
                raices[r] = sig
                sig += 1
            asig.append(raices[r])

        result = tuple(asig)
        return result if len(set(result)) >= 2 else None

    def _particionar_recursivo_q(
        self,
        vertices: list[tuple[int, int]],
        k: int,
        memo: dict[tuple, tuple[int, ...] | None],
    ) -> tuple[int, ...] | None:
        """Particion jerarquica con memoizacion DP usando el algoritmo Q.

        Aplica algoritmo_q recursivamente (divide y venceras) para obtener
        una asignacion inicial de k grupos sin evaluaciones redundantes.
        Memoiza por (subconjunto_ordenado, k) para evitar recomputo.
        Retorna una asignacion sobre `vertices` en orden de entrada, o None.
        """
        clave = (tuple(sorted(vertices, key=lambda v: (v[0], v[1]))), k)
        if clave in memo:
            return memo[clave]

        n = len(vertices)
        if k >= n or n <= 1:
            memo[clave] = None
            return None

        # Solo guardar/restaurar vertices y memoria_grupo_candidato.
        # memoria_delta se acumula entre llamadas (es independiente del subconjunto).
        vertices_prev = self.vertices
        mem_cand_prev = self.memoria_grupo_candidato.copy()

        self.vertices = set(vertices)
        self.memoria_grupo_candidato.clear()
        try:
            clave_mip = self.algoritmo_q(list(vertices))
        except Exception:
            self.vertices = vertices_prev
            self.memoria_grupo_candidato = mem_cand_prev
            memo[clave] = None
            return None

        grupo_a = list(clave_mip)
        grupo_b = [v for v in vertices if v not in set(clave_mip)]

        self.vertices = vertices_prev
        self.memoria_grupo_candidato = mem_cand_prev

        if k == 2:
            set_a = set(grupo_a)
            asig = tuple(0 if v in set_a else 1 for v in vertices)
            memo[clave] = asig
            return asig

        # Para k>2: partir el grupo mayor en k-1 subgrupos.
        mayor, menor = (grupo_a, grupo_b) if len(grupo_a) >= len(grupo_b) else (grupo_b, grupo_a)
        sub_asig = self._particionar_recursivo_q(mayor, k - 1, memo)

        if sub_asig is None:
            memo[clave] = None
            return None

        # Indice de cada vertice dentro de `mayor` para mapear sub_asig.
        pos_en_mayor = {v: i for i, v in enumerate(mayor)}
        menor_set = set(menor)

        asig_global = [0] * n
        for i, v in enumerate(vertices):
            if v in menor_set:
                asig_global[i] = k - 1
            else:
                asig_global[i] = sub_asig[pos_en_mayor[v]]

        # Canonicalizar grupos (0, 1, ...).
        mapa: dict[int, int] = {}
        sig = 0
        canon = []
        for g in asig_global:
            if g not in mapa:
                mapa[g] = sig
                sig += 1
            canon.append(mapa[g])
        asig_canon = tuple(canon)
        memo[clave] = asig_canon
        return asig_canon


# Alias retrocompatible.
QNodes = QNodos

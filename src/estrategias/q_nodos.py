import numpy as np

from src.funciones.formato import fmt_biparticion_q
from src.funciones.iit import seleccionar_emd
from src.modelos.base.sia import SIA
from src.modelos.nucleo.solucion import Solucion


class QNodos(SIA):
    """Implementacion submodular de QNodos basada en deltas y omegas."""

    def __init__(self, tpm: np.ndarray) -> None:
        super().__init__(tpm)
        self.distancia_metrica = seleccionar_emd()
        self.memoria_delta: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[float, np.ndarray]] = {}
        self.memoria_grupo_candidato: dict[tuple[tuple[int, int], ...], tuple[float, np.ndarray]] = {}
        self.clave_submodular: list[list[int]] = [[], []]
        self.vertices: set[tuple[int, int]] = set()

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ) -> Solucion:
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

        clave_mip = self.algoritmo_q(vertices)
        perdida_mip, distribucion_particion = self.memoria_grupo_candidato[clave_mip]
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
                if dist_particion_candidata is None:
                    assert self.sia_dists_marginales is not None
                    dist_particion_candidata = self.sia_dists_marginales.copy()
                    emd_particion_candidata = 0.0
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
            vector_delta = self._alinear_distribucion(
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
        vector_union = self._alinear_distribucion(
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

    def _alinear_distribucion(
        self,
        distribucion: np.ndarray,
        referencia: np.ndarray,
    ) -> np.ndarray:
        if distribucion.size == referencia.size:
            return distribucion
        distribucion_alineada = np.zeros_like(referencia)
        distribucion_alineada[: distribucion.size] = distribucion
        return distribucion_alineada


# Alias retrocompatible.
QNodes = QNodos

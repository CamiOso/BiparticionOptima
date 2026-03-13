import numpy as np

from src.funciones.formato import fmt_biparticion
from src.funciones.iit import seleccionar_emd
from src.modelos.base.sia import SIA
from src.modelos.nucleo.solucion import Solucion


class QNodos(SIA):
    """Heuristica QNodos con busqueda reducida de biparticiones candidatas."""

    def __init__(self, tpm: np.ndarray) -> None:
        super().__init__(tpm)
        self.distancia_metrica = seleccionar_emd()

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
        alcance_indices = self.sia_subsistema.indices_ncubos
        mecanismo_indices = self.sia_subsistema.dims_ncubos

        candidatos_alcance = [
            (int(idx),) for idx in alcance_indices.tolist()
        ] + [tuple(int(v) for v in alcance_indices.tolist())]
        candidatos_mecanismo = [
            (int(idx),) for idx in mecanismo_indices.tolist()
        ] + [tuple(int(v) for v in mecanismo_indices.tolist())]

        mejor_perdida = np.inf
        mejor_distribucion_particion = distribucion_subsistema.copy()
        mejor_subalcance: tuple[int, ...] = tuple(int(v) for v in alcance_indices.tolist())
        mejor_submecanismo: tuple[int, ...] = tuple(
            int(v) for v in mecanismo_indices.tolist()
        )

        for subalcance in candidatos_alcance:
            for submecanismo in candidatos_mecanismo:
                sistema_partido = self.sia_subsistema.bipartir(
                    np.array(subalcance, dtype=np.int8),
                    np.array(submecanismo, dtype=np.int8),
                )
                distribucion_particion = sistema_partido.distribucion_marginal()

                if distribucion_particion.size != distribucion_subsistema.size:
                    distribucion_alineada = np.zeros_like(distribucion_subsistema)
                    distribucion_alineada[: distribucion_particion.size] = (
                        distribucion_particion
                    )
                    distribucion_particion = distribucion_alineada

                perdida = self.distancia_metrica(
                    distribucion_subsistema,
                    distribucion_particion,
                )
                if perdida < mejor_perdida:
                    mejor_perdida = perdida
                    mejor_distribucion_particion = distribucion_particion
                    mejor_subalcance = subalcance
                    mejor_submecanismo = submecanismo

        biparticion = fmt_biparticion(
            mejor_subalcance,
            mejor_submecanismo,
            tuple(int(v) for v in alcance_indices.tolist()),
            tuple(int(v) for v in mecanismo_indices.tolist()),
        )

        return Solucion(
            estrategia="QNodos",
            perdida=float(mejor_perdida),
            distribucion_subsistema=distribucion_subsistema,
            distribucion_particion=mejor_distribucion_particion,
            estado_inicial=estado_inicial,
            particion=biparticion,
        )


# Alias retrocompatible.
QNodes = QNodos

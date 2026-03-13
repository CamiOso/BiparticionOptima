import numpy as np

from src.funciones.formato import fmt_biparticion
from src.funciones.iit import seleccionar_emd
from src.modelos.base.sia import SIA
from src.modelos.nucleo.solucion import Solucion


class Phi(SIA):
    """Estrategia Phi: usa PyPhi si esta disponible y heuristica si no."""

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

        solucion_pyphi = self._resolver_con_pyphi(
            estado_inicial,
            condicion,
            alcance,
            mecanismo,
        )
        if solucion_pyphi is not None:
            return solucion_pyphi

        return self._resolver_heuristico(estado_inicial)

    def _resolver_con_pyphi(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ) -> Solucion | None:
        try:
            from pyphi import Network, Subsystem
        except Exception:
            return None

        estado = tuple(int(bit) for bit in estado_inicial)
        total_nodos = len(estado)
        nodos_candidatos = tuple(i for i, bit in enumerate(condicion) if bit == "1")
        if not nodos_candidatos:
            return None

        mecanismo_pyphi = tuple(
            i
            for i, (bit_m, bit_c) in enumerate(zip(mecanismo, condicion))
            if bit_m == "1" and bit_c == "1"
        )
        alcance_pyphi = tuple(
            i
            for i, (bit_a, bit_c) in enumerate(zip(alcance, condicion))
            if bit_a == "1" and bit_c == "1"
        )
        if not mecanismo_pyphi or not alcance_pyphi:
            return None

        red = Network(self.tpm, node_labels=tuple(range(total_nodos)))
        subsistema = Subsystem(red, state=estado, nodes=nodos_candidatos)
        mip = subsistema.effect_mip(mecanismo_pyphi, alcance_pyphi)
        if mip is None:
            return None

        distribucion_subsistema = np.array(mip.repertoire.flatten(), dtype=np.float32)
        distribucion_particion = np.array(
            mip.partitioned_repertoire.flatten(), dtype=np.float32
        )

        particion = "NO-PARTITION"
        if mip.partition is not None:
            parte_a = mip.partition.parts[True]
            submecanismo = tuple(int(v) for v in parte_a.mechanism)
            subalcance = tuple(int(v) for v in parte_a.purview)
            particion = fmt_biparticion(
                subalcance,
                submecanismo,
                tuple(int(v) for v in alcance_pyphi),
                tuple(int(v) for v in mecanismo_pyphi),
            )

        return Solucion(
            estrategia="PyPhi",
            perdida=float(mip.phi),
            distribucion_subsistema=distribucion_subsistema,
            distribucion_particion=distribucion_particion,
            estado_inicial=estado_inicial,
            particion=particion,
        )

    def _resolver_heuristico(self, estado_inicial: str) -> Solucion:
        assert self.sia_dists_marginales is not None
        assert self.sia_subsistema is not None

        distribucion_subsistema = self.sia_dists_marginales
        alcance_indices = self.sia_subsistema.indices_ncubos
        mecanismo_indices = self.sia_subsistema.dims_ncubos

        candidatos_alcance = [
            (int(idx),) for idx in alcance_indices.tolist()
        ] + [tuple(int(v) for v in alcance_indices.tolist())]
        candidatos_mecanismo = [
            (int(idx),) for idx in mecanismo_indices.tolist()
        ] + [tuple(int(v) for v in mecanismo_indices.tolist())]

        mejor_perdida = np.inf
        mejor_distribucion = distribucion_subsistema.copy()
        mejor_subalcance = tuple(int(v) for v in alcance_indices.tolist())
        mejor_submecanismo = tuple(int(v) for v in mecanismo_indices.tolist())

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
                    mejor_distribucion = distribucion_particion
                    mejor_subalcance = subalcance
                    mejor_submecanismo = submecanismo

        particion = fmt_biparticion(
            mejor_subalcance,
            mejor_submecanismo,
            tuple(int(v) for v in alcance_indices.tolist()),
            tuple(int(v) for v in mecanismo_indices.tolist()),
        )

        return Solucion(
            estrategia="PhiHeuristica",
            perdida=float(mejor_perdida),
            distribucion_subsistema=distribucion_subsistema,
            distribucion_particion=mejor_distribucion,
            estado_inicial=estado_inicial,
            particion=particion,
        )

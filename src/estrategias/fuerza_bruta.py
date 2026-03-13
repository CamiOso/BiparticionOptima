import numpy as np

from src.funciones.particiones import biparticiones
from src.funciones.formato import fmt_biparticion
from src.funciones.iit import seleccionar_emd
from src.intermedios.perfil import gestor_perfilado, profile
from src.intermedios.registro import SafeLogger
from src.modelos.base.sia import SIA
from src.modelos.nucleo.solucion import Solucion


class FuerzaBruta(SIA):
    """Implementacion inicial de fuerza bruta (version didactica minima)."""

    def __init__(self, tpm: np.ndarray) -> None:
        super().__init__(tpm)
        self.distancia_metrica = seleccionar_emd()
        self.logger = SafeLogger("bruteforce_strategy")
        gestor_perfilado.start_session("FuerzaBruta")

    @profile(name="BruteForce_aplicar_estrategia")
    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ) -> Solucion:
        self.logger.info("Iniciando estrategia FuerzaBruta.")
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        assert self.sia_dists_marginales is not None
        dist_subsistema = self.sia_dists_marginales

        assert self.sia_subsistema is not None
        alcance_indices = self.sia_subsistema.indices_ncubos
        mecanismo_indices = self.sia_subsistema.dims_ncubos

        mejor_perdida = np.inf
        mejor_dist_particion = dist_subsistema.copy()
        mejor_subalcance: tuple[int, ...] = ()
        mejor_submecanismo: tuple[int, ...] = ()

        for subalcance, submecanismo in biparticiones(alcance_indices, mecanismo_indices):
            sistema_partido = self.sia_subsistema.bipartir(
                np.array(subalcance, dtype=np.int8),
                np.array(submecanismo, dtype=np.int8),
            )
            dist_particion = sistema_partido.distribucion_marginal()

            if dist_particion.size != dist_subsistema.size:
                # Alineacion didactica: completamos con ceros para comparar en igual espacio.
                aligned = np.zeros_like(dist_subsistema)
                aligned[: dist_particion.size] = dist_particion
                dist_particion = aligned

            perdida = self.distancia_metrica(dist_subsistema, dist_particion)
            if perdida < mejor_perdida:
                mejor_perdida = perdida
                mejor_dist_particion = dist_particion
                mejor_subalcance = subalcance
                mejor_submecanismo = submecanismo

        perdida = mejor_perdida
        dist_particion = mejor_dist_particion
        biparticion_fmt = fmt_biparticion(
            mejor_subalcance,
            mejor_submecanismo,
            tuple(int(v) for v in alcance_indices.tolist()),
            tuple(int(v) for v in mecanismo_indices.tolist()),
        )
        self.logger.debug(f"Perdida calculada: {perdida:.4f}")

        result = Solucion(
            estrategia="FuerzaBruta",
            perdida=float(perdida),
            distribucion_subsistema=dist_subsistema,
            distribucion_particion=dist_particion,
            estado_inicial=estado_inicial,
            particion=biparticion_fmt,
        )
        self.logger.info("Estrategia FuerzaBruta finalizada.")
        return result


# Alias retrocompatible.
BruteForce = FuerzaBruta

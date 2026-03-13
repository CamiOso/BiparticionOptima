import numpy as np

from src.modelos.base.sia import SIA
from src.modelos.nucleo.solucion import Solution


class QNodes(SIA):
    """Estrategia Q-Nodes minima para fase didactica inicial."""

    def __init__(self, tpm: np.ndarray) -> None:
        super().__init__(tpm)

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ) -> Solution:
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        assert self.sia_dists_marginales is not None
        dist_subsistema = self.sia_dists_marginales

        # Placeholder didactico: Q-Nodes iniciara con una aproximacion neutra
        # mientras implementamos la version submodular completa.
        return Solution(
            estrategia="Q-Nodes",
            perdida=0.0,
            distribucion_subsistema=dist_subsistema,
            distribucion_particion=dist_subsistema.copy(),
            estado_inicial=estado_inicial,
            particion="(M=(0,), A=(0,)) | (M*=(), A*=())",
        )

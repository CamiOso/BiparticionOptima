import numpy as np

from src.funcs.iit import seleccionar_emd
from src.models.base.sia import SIA
from src.models.core.solution import Solution


class BruteForce(SIA):
    """Implementacion inicial de fuerza bruta (version didactica minima)."""

    def __init__(self, tpm: np.ndarray) -> None:
        super().__init__(tpm)
        self.distancia_metrica = seleccionar_emd()

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

        # Placeholder temporal: hasta implementar biparticiones reales,
        # usamos una permutacion simple para simular una distribucion particionada.
        dist_particion = np.roll(dist_subsistema, 1)
        perdida = self.distancia_metrica(dist_subsistema, dist_particion)

        return Solution(
            estrategia="BruteForce",
            perdida=float(perdida),
            distribucion_subsistema=dist_subsistema,
            distribucion_particion=dist_particion,
            estado_inicial=estado_inicial,
        )

import numpy as np

from src.funcs.iit import seleccionar_emd
from src.middlewares.profile import gestor_perfilado, profile
from src.middlewares.slogger import SafeLogger
from src.models.base.sia import SIA
from src.models.core.solution import Solution


class BruteForce(SIA):
    """Implementacion inicial de fuerza bruta (version didactica minima)."""

    def __init__(self, tpm: np.ndarray) -> None:
        super().__init__(tpm)
        self.distancia_metrica = seleccionar_emd()
        self.logger = SafeLogger("bruteforce_strategy")
        gestor_perfilado.start_session("BruteForce")

    @profile(name="BruteForce_aplicar_estrategia")
    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ) -> Solution:
        self.logger.info("Iniciando estrategia BruteForce.")
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        assert self.sia_dists_marginales is not None
        dist_subsistema = self.sia_dists_marginales

        # Placeholder temporal: hasta implementar biparticiones reales,
        # usamos una permutacion simple para simular una distribucion particionada.
        dist_particion = np.roll(dist_subsistema, 1)
        perdida = self.distancia_metrica(dist_subsistema, dist_particion)
        self.logger.debug(f"Perdida calculada: {perdida:.4f}")

        result = Solution(
            estrategia="BruteForce",
            perdida=float(perdida),
            distribucion_subsistema=dist_subsistema,
            distribucion_particion=dist_particion,
            estado_inicial=estado_inicial,
        )
        self.logger.info("Estrategia BruteForce finalizada.")
        return result

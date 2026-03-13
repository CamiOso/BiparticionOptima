import numpy as np

from src.models.base.sia import SIA


class BruteForce(SIA):
    """Implementacion inicial de fuerza bruta (version didactica minima)."""

    def __init__(self, tpm: np.ndarray) -> None:
        super().__init__(tpm)

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ) -> dict:
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        assert self.sia_dists_marginales is not None
        return {
            "estrategia": "BruteForce",
            "dist_marginal": self.sia_dists_marginales.tolist(),
            "estado_inicial": estado_inicial,
        }

from dataclasses import dataclass

import numpy as np


@dataclass
class Solution:
    """Resultado estandar de una estrategia de analisis."""

    estrategia: str
    perdida: float
    distribucion_subsistema: np.ndarray
    distribucion_particion: np.ndarray
    estado_inicial: str

    def __str__(self) -> str:
        return (
            f"Estrategia: {self.estrategia}\n"
            f"Estado inicial: {self.estado_inicial}\n"
            f"Perdida: {self.perdida:.4f}\n"
            f"Dist. subsistema: {self.distribucion_subsistema.tolist()}\n"
            f"Dist. particion: {self.distribucion_particion.tolist()}"
        )

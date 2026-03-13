from dataclasses import dataclass

import numpy as np

from src.funciones.formato import fmt_solution_block


@dataclass
class Solucion:
    """Resultado estandar de una estrategia de analisis."""

    estrategia: str
    perdida: float
    distribucion_subsistema: np.ndarray
    distribucion_particion: np.ndarray
    estado_inicial: str
    particion: str = "NO-PARTITION"

    def __str__(self) -> str:
        return fmt_solution_block(
            estrategia=self.estrategia,
            estado_inicial=self.estado_inicial,
            perdida=self.perdida,
            distribucion_subsistema=self.distribucion_subsistema,
            distribucion_particion=self.distribucion_particion,
            particion=self.particion,
        )


# Alias retrocompatible.
Solution = Solucion

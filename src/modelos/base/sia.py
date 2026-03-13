from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from src.constantes.error import ERROR_INVALID_BITSTRING
from src.modelos.nucleo.sistema import System


class SIA(ABC):
    """Base abstracta para estrategias de Analisis de Irreducibilidad Sistemica."""

    def __init__(self, tpm: np.ndarray) -> None:
        self.tpm = tpm
        self.sia_subsistema: System | None = None
        self.sia_dists_marginales: NDArray[np.float32] | None = None

    @abstractmethod
    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ):
        """Cada estrategia implementa su forma de resolver el problema."""

    def chequear_parametros(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ) -> None:
        expected = self.tpm.shape[1]
        for value in (estado_inicial, condicion, alcance, mecanismo):
            if len(value) != expected:
                raise ValueError(
                    f"Longitud invalida: se esperaba {expected} y llego {len(value)}."
                )
            if any(char not in {"0", "1"} for char in value):
                raise ValueError(ERROR_INVALID_BITSTRING)

    def sia_preparar_subsistema(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ) -> None:
        """Prepara un subsistema base. En este paso inicial no hay particionado."""
        self.chequear_parametros(estado_inicial, condicion, alcance, mecanismo)

        estado_vec = np.array([int(bit) for bit in estado_inicial], dtype=np.int8)
        system = System(self.tpm, estado_vec)

        self.sia_subsistema = system
        self.sia_dists_marginales = system.distribucion_marginal()

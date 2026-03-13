from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from src.constantes.error import ERROR_INVALID_BITSTRING
from src.modelos.nucleo.sistema import Sistema


class SIA(ABC):
    """Base abstracta para estrategias de Analisis de Irreducibilidad Sistemica."""

    def __init__(self, tpm: np.ndarray) -> None:
        self.tpm = tpm
        self.sia_subsistema: Sistema | None = None
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
        esperado = self.tpm.shape[1]
        for valor in (estado_inicial, condicion, alcance, mecanismo):
            if len(valor) != esperado:
                raise ValueError(
                    f"Longitud invalida: se esperaba {esperado} y llego {len(valor)}."
                )
            if any(char not in {"0", "1"} for char in valor):
                raise ValueError(ERROR_INVALID_BITSTRING)

    def sia_preparar_subsistema(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ) -> None:
        """Prepara subsistema aplicando condicionamiento y sustracciones de entrada."""
        self.chequear_parametros(estado_inicial, condicion, alcance, mecanismo)

        estado_vec = np.array([int(bit) for bit in estado_inicial], dtype=np.int8)
        sistema_completo = Sistema(self.tpm, estado_vec)

        indices_condicionados = np.array(
            [indice for indice, bit in enumerate(condicion) if bit == "0"],
            dtype=np.int8,
        )
        indices_alcance_sustraidos = np.array(
            [indice for indice, bit in enumerate(alcance) if bit == "0"],
            dtype=np.int8,
        )
        indices_mecanismo_sustraidos = np.array(
            [indice for indice, bit in enumerate(mecanismo) if bit == "0"],
            dtype=np.int8,
        )

        sistema_candidato = sistema_completo.condicionar(indices_condicionados)
        sistema = sistema_candidato.substraer(
            indices_alcance_sustraidos,
            indices_mecanismo_sustraidos,
        )

        self.sia_subsistema = sistema
        self.sia_dists_marginales = sistema.distribucion_marginal()

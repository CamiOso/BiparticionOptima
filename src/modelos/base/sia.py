from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.constantes.error import ERROR_INVALID_BITSTRING
from src.modelos.nucleo.ncubo import NCube
from src.modelos.nucleo.sistema import Sistema

if TYPE_CHECKING:
    from src.aplicacion.configuracion import AppConfig


class SIA(ABC):
    """Base abstracta para estrategias de Analisis de Irreducibilidad Sistemica.

    Acepta un ``AppConfig`` opcional para inyección de dependencias. Si no se
    proporciona, las subclases recurren al singleton global (retrocompatible).
    """

    def __init__(self, tpm: np.ndarray, config: AppConfig | None = None) -> None:
        self.tpm = tpm
        self.config = config
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

        n_total = self.tpm.shape[1]
        estado_vec = np.array([int(bit) for bit in estado_inicial], dtype=np.int8)

        indices_condicionados = np.array(
            [i for i, bit in enumerate(condicion) if bit == "0"], dtype=np.int8
        )
        alc_nodos = np.array(
            [i for i, bit in enumerate(alcance) if bit == "1"], dtype=np.int8
        )
        mec_nodos = np.array(
            [i for i, bit in enumerate(mecanismo) if bit == "1"], dtype=np.int8
        )

        # Para sistemas grandes sin condicionamiento, construir el subsistema
        # directamente desde la TPM sin crear los n_total NCubes intermedios.
        if indices_condicionados.size == 0 and n_total >= 15:
            sistema = _crear_subsistema_premarginal(
                self.tpm, estado_vec, alc_nodos, mec_nodos, n_total
            )
            self.sia_subsistema = sistema
            self.sia_dists_marginales = sistema.distribucion_marginal()
            return

        indices_alcance_sustraidos = np.array(
            [i for i, bit in enumerate(alcance) if bit == "0"], dtype=np.int8
        )
        indices_mecanismo_sustraidos = np.array(
            [i for i, bit in enumerate(mecanismo) if bit == "0"], dtype=np.int8
        )

        sistema_completo = Sistema(self.tpm, estado_vec)
        sistema_candidato = sistema_completo.condicionar(indices_condicionados)
        sistema = sistema_candidato.substraer(
            indices_alcance_sustraidos,
            indices_mecanismo_sustraidos,
        )

        self.sia_subsistema = sistema
        self.sia_dists_marginales = sistema.distribucion_marginal()


def _crear_subsistema_premarginal(
    tpm: np.ndarray,
    estado_vec: "NDArray[np.int8]",
    alc_nodos: "NDArray[np.int8]",
    mec_nodos: "NDArray[np.int8]",
    n_total: int,
) -> Sistema:
    """Crea el subsistema directo desde la TPM sin NCubes intermedios de n_total nodos.

    Para cada nodo en alcance, lee su columna de la TPM, hace reshape a
    (2,)*n_total y promedia sobre los ejes de los nodos que NO son mecanismo.
    Produce NCubes de forma (2,)*n_mec directamente, evitando crear y luego
    descartar los n_total NCubes grandes del sistema completo.
    """
    mec_set = {int(v) for v in mec_nodos}
    # Eje del nodo k en el arreglo (2,)*n_total: n_total-1-k
    non_mec_axes = tuple(n_total - 1 - k for k in range(n_total) if k not in mec_set)

    cubos = []
    for alc_node in alc_nodos:
        col = tpm[:, int(alc_node)]
        data_full = np.asarray(col).reshape((2,) * n_total)
        data_marg = data_full.mean(axis=non_mec_axes)
        del data_full, col
        cubos.append(NCube(
            indice=int(alc_node),
            dims=np.array(mec_nodos, dtype=np.int8),
            data=data_marg,
        ))

    return Sistema._from_cubes(estado_vec, tuple(cubos))

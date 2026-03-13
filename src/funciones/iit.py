from typing import Callable

import numpy as np
from numpy.typing import NDArray

from src.modelos.base.aplicacion import aplicacion
from src.modelos.enumeraciones.emd_temporal import TimeEMD


ABECEDARIO = tuple(chr(ord("A") + indice) for indice in range(26))


def literales(indices_restantes: NDArray[np.int8], minuscula: bool = False) -> str:
    """Convierte indices de nodos a una etiqueta literal amigable."""
    if indices_restantes.size == 0:
        return "vacio"
    letras = []
    for indice in indices_restantes.tolist():
        letra = ABECEDARIO[int(indice)] if int(indice) < len(ABECEDARIO) else f"N{indice}"
        letras.append(letra.lower() if minuscula else letra)
    return "".join(letras)


def emd_efecto(u: NDArray[np.float32], v: NDArray[np.float32]) -> float:
    """Distancia EMD simplificada para el modo efecto."""
    return float(np.sum(np.abs(u - v)))


def seleccionar_emd() -> Callable[[NDArray[np.float32], NDArray[np.float32]], float]:
    """Selecciona la funcion de distancia EMD segun la configuracion global."""
    emd_metricas = {
        TimeEMD.EMD_EFECTO.value: emd_efecto,
    }

    if aplicacion.tiempo_emd not in emd_metricas:
        raise ValueError(
            f"Tiempo EMD no soportado en esta etapa: {aplicacion.tiempo_emd}"
        )

    return emd_metricas[aplicacion.tiempo_emd]

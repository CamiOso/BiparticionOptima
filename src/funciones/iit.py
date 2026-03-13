from itertools import product
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from src.modelos.base.aplicacion import aplicacion
from src.modelos.enumeraciones.distancia import MetricDistance
from src.modelos.enumeraciones.notacion import Notation
from src.modelos.enumeraciones.emd_temporal import TimeEMD


def _etiqueta_excel(numero: int) -> str:
    if numero <= 0:
        return ""
    return _etiqueta_excel((numero - 1) // 26) + chr((numero - 1) % 26 + ord("A"))


ABECEDARIO = tuple(_etiqueta_excel(indice) for indice in range(1, 41))


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


def emd_causal(u: NDArray[np.float32], v: NDArray[np.float32]) -> float:
    """Aproximacion causal base mientras no se use una distancia mas rica."""
    return float(np.sum(np.abs(u - v)))


def contar_bits(numero: int) -> int:
    return bin(numero).count("1")


def distancia_hamming(a: int, b: int) -> int:
    return contar_bits(a ^ b)


def seleccionar_distancia() -> Callable[[int, int], float]:
    distancias = {
        MetricDistance.HAMMING.value: distancia_hamming,
        MetricDistance.MANHATTAN.value: lambda a, b: abs(a - b),
        MetricDistance.EUCLIDIANA.value: lambda a, b: float(abs(a - b)),
    }
    if aplicacion.distancia_metrica not in distancias:
        raise ValueError(
            f"Distancia no soportada en esta etapa: {aplicacion.distancia_metrica}"
        )
    return distancias[aplicacion.distancia_metrica]


def big_endian(n: int) -> NDArray[np.uint32]:
    return np.array(range(1 << n), dtype=np.uint32)


def lil_endian(n: int) -> NDArray[np.uint32]:
    if n <= 0:
        return np.array([0], dtype=np.uint32)
    indices = np.arange(1 << n, dtype=np.uint32)
    salida = np.zeros_like(indices)
    for bit in range(n):
        salida |= ((indices >> bit) & 1) << (n - bit - 1)
    return salida


def reindexar(n: int) -> NDArray[np.uint32]:
    notaciones = {
        Notation.BIG_ENDIAN.value: big_endian(n),
        Notation.LIL_ENDIAN.value: lil_endian(n),
    }
    if aplicacion.notacion_indexado not in notaciones:
        raise ValueError(
            f"Notacion no soportada en esta etapa: {aplicacion.notacion_indexado}"
        )
    return notaciones[aplicacion.notacion_indexado]


def seleccionar_estado(subestado: NDArray[np.int8]) -> NDArray[np.int8]:
    notaciones = {
        Notation.BIG_ENDIAN.value: subestado,
        Notation.LIL_ENDIAN.value: subestado[::-1],
    }
    if aplicacion.notacion_indexado not in notaciones:
        raise ValueError(
            f"Notacion no soportada en esta etapa: {aplicacion.notacion_indexado}"
        )
    return notaciones[aplicacion.notacion_indexado]


def dec2bin(decimal: int, ancho: int) -> str:
    return format(decimal, f"0{ancho}b")


def estados_binarios(n: int) -> list[str]:
    return [dec2bin(indice, n) for indice in range(1 << n)][1:]


def combinaciones_restringidas(binario: str) -> tuple[list[str], list[str]]:
    cantidad_unos = binario.count("1")
    posiciones = [indice for indice, bit in enumerate(binario) if bit == "1"]
    base = list(product(["0", "1"], repeat=cantidad_unos))
    combinaciones = []
    for combinacion in base:
        bits = ["0"] * len(binario)
        for posicion, bit in zip(posiciones, combinacion):
            bits[posicion] = bit
        combinaciones.append("".join(bits))
    return combinaciones, combinaciones.copy()


def generar_combinaciones(a: str) -> list[tuple[str, str, str]]:
    b, c = combinaciones_restringidas(a)
    return list(product([a], b, c))[1:]


def seleccionar_emd() -> Callable[[NDArray[np.float32], NDArray[np.float32]], float]:
    """Selecciona la funcion de distancia EMD segun la configuracion global."""
    emd_metricas = {
        TimeEMD.EMD_EFECTO.value: emd_efecto,
        TimeEMD.EMD_CAUSA.value: emd_causal,
        TimeEMD.EMD_INTEGRADA.value: emd_efecto,
    }

    if aplicacion.tiempo_emd not in emd_metricas:
        raise ValueError(
            f"Tiempo EMD no soportado en esta etapa: {aplicacion.tiempo_emd}"
        )

    return emd_metricas[aplicacion.tiempo_emd]

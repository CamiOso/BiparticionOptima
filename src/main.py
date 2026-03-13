from src.constants.base import PROJECT_NAME, PROJECT_VERSION
from src.constants.error import ERROR_EMPTY_INPUT, ERROR_INVALID_BITSTRING
from src.constants.models import BRUTEFORCE_LABEL
from src.models.enums.distance import MetricDistance
from src.models.enums.notation import Notation
from src.models.enums.temporal_emd import TimeEMD


def validar_bitstring(value: str) -> None:
    """Valida una cadena binaria para entradas iniciales del sistema."""
    if not value:
        raise ValueError(ERROR_EMPTY_INPUT)
    if any(char not in {"0", "1"} for char in value):
        raise ValueError(ERROR_INVALID_BITSTRING)


def iniciar() -> None:
    """Orquestador inicial del proyecto."""
    validar_bitstring("1000")
    print(
        f"{PROJECT_NAME} v{PROJECT_VERSION}: proyecto iniciado correctamente con estrategia base {BRUTEFORCE_LABEL}."
    )
    print(
        "Configuracion base -> "
        f"distancia: {MetricDistance.HAMMING.value}, "
        f"notacion: {Notation.LIL_ENDIAN.value}, "
        f"tiempo EMD: {TimeEMD.EMD_EFECTO.value}."
    )

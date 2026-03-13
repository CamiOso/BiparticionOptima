from src.constants.base import PROJECT_NAME, PROJECT_VERSION
from src.constants.error import ERROR_EMPTY_INPUT, ERROR_INVALID_BITSTRING


def validar_bitstring(value: str) -> None:
    """Valida una cadena binaria para entradas iniciales del sistema."""
    if not value:
        raise ValueError(ERROR_EMPTY_INPUT)
    if any(char not in {"0", "1"} for char in value):
        raise ValueError(ERROR_INVALID_BITSTRING)


def iniciar() -> None:
    """Orquestador inicial del proyecto."""
    validar_bitstring("1000")
    print(f"{PROJECT_NAME} v{PROJECT_VERSION}: proyecto iniciado correctamente.")

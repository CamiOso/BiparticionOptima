"""Puerto de salida: contrato para el sistema de registro (logging).

Los casos de uso emiten mensajes a través de este puerto sin acoplarse
a ninguna librería concreta (SafeLogger, logging estándar, null logger, etc.).
"""

from typing import Protocol


class IRegistro(Protocol):
    """Contrato mínimo de cualquier adaptador de logging."""

    def debug(self, mensaje: str) -> None: ...
    def info(self, mensaje: str) -> None: ...
    def warn(self, mensaje: str) -> None: ...
    def error(self, mensaje: str) -> None: ...

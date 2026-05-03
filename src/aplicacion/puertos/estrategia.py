"""Puerto de salida: contrato que debe cumplir toda estrategia de partición.

Cualquier clase que implemente ``aplicar_estrategia`` con la firma correcta
satisface este puerto automáticamente (duck typing estructural vía Protocol).
No es necesario heredar explícitamente de ``IEstrategia``.
"""

from typing import Protocol, runtime_checkable

from src.modelos.nucleo.solucion import Solucion


@runtime_checkable
class IEstrategia(Protocol):
    """Contrato mínimo de cualquier algoritmo de búsqueda de partición óptima."""

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
        **kwargs,
    ) -> Solucion:
        """Ejecuta la estrategia y devuelve la mejor partición encontrada."""
        ...

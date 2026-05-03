"""Adaptadores de repositorio — implementaciones concretas de IRepositorioTPM.

Cada clase expuesta aquí satisface el puerto ``IRepositorioTPM`` y puede ser
inyectada en el caso de uso ``EstimarTPM``.

Uso::

    from src.infraestructura.repositorios import Gestor

Repositorios disponibles
------------------------
- Gestor : carga TPMs desde CSV predefinidas o estima desde muestras temporales
"""

from src.controladores.gestor import Gestor

__all__ = ["Gestor"]

"""Entidades del dominio.

Re-exporta las entidades centrales desde su ubicación canónica para que
el resto del código pueda importar desde esta capa sin conocer la ruta interna.

Uso::

    from src.dominio.entidades import NCube, Sistema, Solucion
"""

from src.modelos.nucleo.ncubo import NCube
from src.modelos.nucleo.sistema import Sistema
from src.modelos.nucleo.solucion import Solucion

__all__ = ["NCube", "Sistema", "Solucion"]

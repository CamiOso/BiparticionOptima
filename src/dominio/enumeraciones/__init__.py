"""Enumeraciones del dominio.

Re-exporta todos los tipos enumerados para que las capas superiores los
consuman desde la frontera del dominio.

Uso::

    from src.dominio.enumeraciones import MetricDistance, TimeEMD, GeometricMode, Notation
"""

from src.modelos.enumeraciones.distancia import MetricDistance
from src.modelos.enumeraciones.emd_temporal import TimeEMD
from src.modelos.enumeraciones.geometric_mode import GeometricMode
from src.modelos.enumeraciones.notacion import Notation

__all__ = ["MetricDistance", "TimeEMD", "GeometricMode", "Notation"]

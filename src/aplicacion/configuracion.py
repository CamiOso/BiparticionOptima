"""Configuración inmutable inyectable de la aplicación.

Reemplaza al singleton mutable ``Application`` para permitir inyección de
dependencias y testing determinista: en lugar de mutar un objeto global,
cada componente recibe su configuración explícitamente en el constructor.

Ejemplo de uso::

    from src.aplicacion.configuracion import AppConfig
    from src.dominio.enumeraciones import TimeEMD, GeometricMode

    config = AppConfig(
        tiempo_emd=TimeEMD.JENSEN_SHANNON.value,
        modo_geometrico=GeometricMode.REFINED.value,
    )
    solver = FuerzaBruta(tpm, config=config)
"""

from dataclasses import dataclass, field

from src.modelos.enumeraciones.distancia import MetricDistance
from src.modelos.enumeraciones.emd_temporal import TimeEMD
from src.modelos.enumeraciones.geometric_mode import GeometricMode
from src.modelos.enumeraciones.notacion import Notation


@dataclass(frozen=True)
class AppConfig:
    """Parámetros de ejecución de la aplicación.

    Al ser ``frozen=True``, las instancias son inmutables: para cambiar un
    parámetro se crea una nueva instancia mediante ``dataclasses.replace()``.
    """

    semilla_numpy: int = 73
    pagina_red_muestra: str = "A"
    distancia_metrica: str = field(default_factory=lambda: MetricDistance.HAMMING.value)
    notacion_indexado: str = field(default_factory=lambda: Notation.LIL_ENDIAN.value)
    tiempo_emd: str = field(default_factory=lambda: TimeEMD.EMD_EFECTO.value)
    modo_geometrico: str = field(default_factory=lambda: GeometricMode.REFINED.value)

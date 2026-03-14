"""Configuracion global de la aplicacion."""

from src.modelos.enumeraciones.distancia import MetricDistance
from src.modelos.enumeraciones.geometric_mode import GeometricMode
from src.modelos.enumeraciones.notacion import Notation
from src.modelos.enumeraciones.emd_temporal import TimeEMD


class Application:
    """Singleton simple para parametros globales de ejecucion."""

    def __init__(self) -> None:
        self.semilla_numpy: int = 73
        self.pagina_red_muestra: str = "A"
        self.distancia_metrica: str = MetricDistance.HAMMING.value
        self.notacion_indexado: str = Notation.LIL_ENDIAN.value
        self.tiempo_emd: str = TimeEMD.EMD_EFECTO.value
        self.modo_geometrico: str = GeometricMode.REFINED.value

    def set_pagina_red_muestra(self, pagina: str) -> None:
        self.pagina_red_muestra = pagina

    def set_distancia(self, distancia: MetricDistance) -> None:
        self.distancia_metrica = distancia.value

    def set_notacion(self, notacion: Notation) -> None:
        self.notacion_indexado = notacion.value

    def set_tiempo_emd(self, tiempo: TimeEMD) -> None:
        self.tiempo_emd = tiempo.value

    def set_modo_geometrico(self, modo: GeometricMode) -> None:
        self.modo_geometrico = modo.value


aplicacion = Application()

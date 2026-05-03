"""Caso de uso: BuscarParticionOptima.

Encuentra la partición del sistema que minimiza la pérdida de información
integrada (φ). No conoce ninguna estrategia concreta; delega en el puerto
``IEstrategia`` recibido por inyección de dependencias.
"""

from dataclasses import dataclass

from src.aplicacion.puertos.estrategia import IEstrategia
from src.aplicacion.puertos.registro import IRegistro
from src.modelos.nucleo.solucion import Solucion


@dataclass
class EntradaBusqueda:
    """DTO de entrada para el caso de uso.

    Encapsula todos los parámetros necesarios para iniciar la búsqueda,
    aislando al caso de uso de la interfaz de presentación.
    """

    estado_inicial: str
    condicion: str
    alcance: str
    mecanismo: str
    k: int = 2


@dataclass
class BuscarParticionOptima:
    """Orquesta la búsqueda de la partición óptima.

    Recibe sus dependencias por constructor (inyección de dependencias):
    no importa directamente ninguna estrategia concreta.

    Atributos
    ---------
    estrategia : IEstrategia
        Adaptador que implementa el algoritmo de búsqueda.
    registro   : IRegistro
        Adaptador de logging para observabilidad.
    """

    estrategia: IEstrategia
    registro: IRegistro

    def ejecutar(self, entrada: EntradaBusqueda) -> Solucion:
        """Ejecuta la búsqueda y devuelve la solución encontrada."""
        self.registro.info(
            f"Iniciando búsqueda con {type(self.estrategia).__name__} "
            f"| estado='{entrada.estado_inicial}' k={entrada.k}"
        )

        resultado = self.estrategia.aplicar_estrategia(
            estado_inicial=entrada.estado_inicial,
            condicion=entrada.condicion,
            alcance=entrada.alcance,
            mecanismo=entrada.mecanismo,
            k=entrada.k,
        )

        self.registro.info(
            f"Partición encontrada: perdida={resultado.perdida:.4f} "
            f"particion={resultado.particion}"
        )
        return resultado

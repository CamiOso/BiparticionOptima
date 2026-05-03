"""Composition Root — único lugar donde los puertos se conectan a los adaptadores.

En arquitectura hexagonal el "composition root" (raíz de composición) es el
punto donde se instancian todas las dependencias concretas y se inyectan en
los casos de uso. Ninguna otra capa sabe qué implementación concreta se usa.

Regla de oro: sólo este módulo importa de ``src.infraestructura``.

Ejemplo de uso::

    from src.contenedor import Contenedor
    from src.aplicacion.configuracion import AppConfig

    contenedor = Contenedor(AppConfig())
    caso = contenedor.caso_uso_buscar_particion("fuerza_bruta", tpm)
    resultado = caso.ejecutar(entrada)
"""

import numpy as np

from src.aplicacion.casos_de_uso.buscar_particion import BuscarParticionOptima
from src.aplicacion.casos_de_uso.estimar_tpm import EstimarTPM
from src.aplicacion.configuracion import AppConfig
from src.aplicacion.puertos.estrategia import IEstrategia
from src.infraestructura.observabilidad import SafeLogger
from src.infraestructura.repositorios import Gestor


class Contenedor:
    """Contenedor de dependencias (IoC container ligero).

    Crea y ensambla adaptadores concretos según la configuración recibida.
    Expone métodos de fábrica para cada adaptador y caso de uso.

    Parámetros
    ----------
    config : AppConfig
        Configuración inmutable. Si se omite se usan los valores por defecto.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or AppConfig()

    # ── Fábricas de adaptadores ──────────────────────────────────────────────

    def registro(self, nombre: str = "app") -> SafeLogger:
        """Devuelve un adaptador de logging para el nombre dado."""
        return SafeLogger(nombre)

    def repositorio_tpm(self, estado_inicial: str) -> Gestor:
        """Devuelve el repositorio de TPMs configurado con la página activa."""
        return Gestor(estado_inicial=estado_inicial)

    def estrategia(self, nombre: str, tpm: np.ndarray) -> IEstrategia:
        """Instancia y devuelve la estrategia concreta pedida.

        La config actual se pasa a la estrategia para evitar que consuma el
        singleton global.

        Parámetros
        ----------
        nombre : str
            Nombre de la estrategia ("fuerza_bruta", "phi", "qnodos", "geometric", "circuito").
        tpm    : np.ndarray
            Matriz de transición de probabilidades [2^n × n].
        """
        nombre_lower = nombre.lower()

        if nombre_lower in {"fuerza_bruta", "bruteforce", "fuerzabruta"}:
            from src.infraestructura.estrategias import FuerzaBruta
            return FuerzaBruta(tpm, config=self._config)

        if nombre_lower == "phi":
            from src.infraestructura.estrategias import Phi
            return Phi(tpm, config=self._config)

        if nombre_lower in {"qnodos", "q_nodes", "qnodes"}:
            from src.infraestructura.estrategias import QNodos
            return QNodos(tpm, config=self._config)

        if nombre_lower == "geometric":
            from src.infraestructura.estrategias import Geometric
            return Geometric(tpm, config=self._config)

        if nombre_lower == "circuito":
            from src.infraestructura.estrategias import Circuito
            return Circuito(tpm, config=self._config)

        if nombre_lower in {"ib", "information_bottleneck", "informacion_bottleneck"}:
            from src.infraestructura.estrategias import InformacionBottleneck
            return InformacionBottleneck(tpm, config=self._config)

        if nombre_lower == "louvain":
            from src.infraestructura.estrategias import Louvain
            return Louvain(tpm, config=self._config)

        if nombre_lower in {"genetico", "ga", "algoritmo_genetico"}:
            from src.infraestructura.estrategias import AlgoritmoGenetico
            return AlgoritmoGenetico(tpm, config=self._config)

        if nombre_lower in {"ilp", "particion_ilp"}:
            from src.infraestructura.estrategias import ParticionILP
            return ParticionILP(tpm, config=self._config)

        if nombre_lower in {"bp", "belief_propagation"}:
            from src.infraestructura.estrategias import BeliefPropagation
            return BeliefPropagation(tpm, config=self._config)

        raise ValueError(
            f"Estrategia desconocida: '{nombre}'. "
            "Opciones: fuerza_bruta, phi, qnodos, geometric, circuito, "
            "ib, louvain, genetico, ilp, bp."
        )

    # ── Fábricas de casos de uso ─────────────────────────────────────────────

    def caso_uso_buscar_particion(
        self,
        nombre_estrategia: str,
        tpm: np.ndarray,
    ) -> BuscarParticionOptima:
        """Construye el caso de uso con la estrategia y el logger inyectados."""
        return BuscarParticionOptima(
            estrategia=self.estrategia(nombre_estrategia, tpm),
            registro=self.registro("busqueda"),
        )

    def caso_uso_estimar_tpm(self, estado_inicial: str) -> EstimarTPM:
        """Construye el caso de uso con el repositorio y el logger inyectados."""
        return EstimarTPM(
            repositorio=self.repositorio_tpm(estado_inicial),
            registro=self.registro("tpm"),
        )

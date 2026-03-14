from src.constantes.base import PROJECT_NAME, PROJECT_VERSION
from src.constantes.error import ERROR_EMPTY_INPUT, ERROR_INVALID_BITSTRING
from src.constantes.models import BRUTEFORCE_LABEL
from src.controladores.gestor import Gestor
from src.modelos.base.aplicacion import aplicacion
from src.modelos.nucleo.ncubo import NCube
from src.modelos.nucleo.sistema import Sistema
from src.modelos.enumeraciones.distancia import MetricDistance
from src.modelos.enumeraciones.notacion import Notation
from src.modelos.enumeraciones.emd_temporal import TimeEMD
from src.intermedios.registro import SafeLogger
from src.estrategias.fuerza_bruta import FuerzaBruta
from src.estrategias.geometrica import Geometric
from src.estrategias.phi import Phi
from src.estrategias.q_nodos import QNodos
import numpy as np


logger = SafeLogger("main")


def validar_bitstring(value: str) -> None:
    """Valida una cadena binaria para entradas iniciales del sistema."""
    if not value:
        raise ValueError(ERROR_EMPTY_INPUT)
    if any(char not in {"0", "1"} for char in value):
        raise ValueError(ERROR_INVALID_BITSTRING)


def iniciar() -> None:
    """Orquestador inicial del proyecto."""
    logger.info("Inicio de ejecucion en main.iniciar")
    validar_bitstring("1000")
    aplicacion.set_pagina_red_muestra("A")
    print(
        f"{PROJECT_NAME} v{PROJECT_VERSION}: proyecto iniciado correctamente con estrategia base {BRUTEFORCE_LABEL}."
    )
    print(
        "Configuracion base -> "
        f"distancia: {MetricDistance.HAMMING.value}, "
        f"notacion: {Notation.LIL_ENDIAN.value}, "
        f"tiempo EMD: {TimeEMD.EMD_EFECTO.value}."
    )
    print(
        "Application singleton -> "
        f"pagina: {aplicacion.pagina_red_muestra}, "
        f"distancia: {aplicacion.distancia_metrica}."
    )

    estado_inicial = "1000"
    gestor = Gestor(estado_inicial=estado_inicial)
    tpm = gestor.cargar_red()
    logger.debug(f"TPM cargada con forma {tpm.shape}")
    print(f"TPM cargada desde {gestor.archivo_tpm} con forma {tpm.shape}.")

    estado_vector = np.array([int(bit) for bit in estado_inicial], dtype=np.int8)
    sistema = Sistema(tpm, estado_vector)
    dist_marginal = sistema.distribucion_marginal()
    print(f"Sistema demo -> distribucion marginal: {dist_marginal.tolist()}")

    estrategia = FuerzaBruta(tpm)
    resultado = estrategia.aplicar_estrategia(
        estado_inicial=estado_inicial,
        condicion="1111",
        alcance="1111",
        mecanismo="1111",
    )
    print(f"SIA demo ->\n{resultado}")
    print(
        "Perdida demo -> "
        f"{resultado.perdida:.4f} | "
        f"subsistema={resultado.distribucion_subsistema.tolist()} vs "
        f"particion={resultado.distribucion_particion.tolist()}"
    )

    estrategia_phi = Phi(tpm)
    resultado_phi = estrategia_phi.aplicar_estrategia(
        estado_inicial=estado_inicial,
        condicion="1111",
        alcance="1111",
        mecanismo="1111",
    )
    print(f"Phi demo ->\n{resultado_phi}")

    estrategia_q = QNodos(tpm)
    resultado_q = estrategia_q.aplicar_estrategia(
        estado_inicial=estado_inicial,
        condicion="1111",
        alcance="1111",
        mecanismo="1111",
    )
    print(f"Q-Nodes demo ->\n{resultado_q}")

    estrategia_geometrica = Geometric(tpm)
    resultado_geometrica = estrategia_geometrica.aplicar_estrategia(
        estado_inicial=estado_inicial,
        condicion="1111",
        alcance="1111",
        mecanismo="1111",
    )
    print(f"Geometric demo ->\n{resultado_geometrica}")

    cubo_demo = NCube(
        indice=0,
        dims=np.array([0, 1], dtype=np.int8),
        data=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    )
    cubo_marginal = cubo_demo.marginalizar(np.array([1], dtype=np.int8))
    print(
        "NCube demo -> "
        f"dims originales: {cubo_demo.dims.tolist()}, "
        f"dims marginalizadas: {cubo_marginal.dims.tolist()}, "
        f"data: {cubo_marginal.data.tolist()}"
    )
    logger.info("Fin de ejecucion en main.iniciar")

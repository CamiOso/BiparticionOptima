from src.constants.base import PROJECT_NAME, PROJECT_VERSION
from src.constants.error import ERROR_EMPTY_INPUT, ERROR_INVALID_BITSTRING
from src.constants.models import BRUTEFORCE_LABEL
from src.controllers.manager import Manager
from src.models.base.application import aplicacion
from src.models.core.ncube import NCube
from src.models.core.system import System
from src.models.enums.distance import MetricDistance
from src.models.enums.notation import Notation
from src.models.enums.temporal_emd import TimeEMD
from src.strategies.force import BruteForce
import numpy as np


def validar_bitstring(value: str) -> None:
    """Valida una cadena binaria para entradas iniciales del sistema."""
    if not value:
        raise ValueError(ERROR_EMPTY_INPUT)
    if any(char not in {"0", "1"} for char in value):
        raise ValueError(ERROR_INVALID_BITSTRING)


def iniciar() -> None:
    """Orquestador inicial del proyecto."""
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
    gestor = Manager(estado_inicial=estado_inicial)
    tpm = gestor.cargar_red()
    print(f"TPM cargada desde {gestor.tpm_filename} con forma {tpm.shape}.")

    estado_vector = np.array([int(bit) for bit in estado_inicial], dtype=np.int8)
    system = System(tpm, estado_vector)
    dist_marginal = system.distribucion_marginal()
    print(f"System demo -> distribucion marginal: {dist_marginal.tolist()}")

    estrategia = BruteForce(tpm)
    resultado = estrategia.aplicar_estrategia(
        estado_inicial=estado_inicial,
        condicion="1111",
        alcance="1111",
        mecanismo="1111",
    )
    print(f"SIA demo -> {resultado}")
    print(
        "Perdida demo -> "
        f"{resultado['perdida']:.4f} | "
        f"subsistema={resultado['dist_marginal_subsistema']} vs "
        f"particion={resultado['dist_marginal_particion']}"
    )

    demo_cube = NCube(
        indice=0,
        dims=np.array([0, 1], dtype=np.int8),
        data=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    )
    cube_marginal = demo_cube.marginalizar(np.array([1], dtype=np.int8))
    print(
        "NCube demo -> "
        f"dims originales: {demo_cube.dims.tolist()}, "
        f"dims marginalizadas: {cube_marginal.dims.tolist()}, "
        f"data: {cube_marginal.data.tolist()}"
    )

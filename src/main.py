import json
from pathlib import Path
import time

from src.constantes.base import PROJECT_NAME, PROJECT_VERSION
from src.constantes.error import ERROR_EMPTY_INPUT, ERROR_INVALID_BITSTRING
from src.constantes.models import BRUTEFORCE_LABEL
from src.controladores.gestor import Gestor
from src.modelos.base.aplicacion import aplicacion
from src.modelos.nucleo.ncubo import NCube
from src.modelos.nucleo.sistema import Sistema
from src.modelos.enumeraciones.distancia import MetricDistance
from src.modelos.enumeraciones.geometric_mode import GeometricMode
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


def _solucion_a_dict(resultado, elapsed_seconds: float) -> dict[str, object]:
    return {
        "estrategia": resultado.estrategia,
        "estado_inicial": resultado.estado_inicial,
        "perdida": float(resultado.perdida),
        "particion": resultado.particion,
        "distribucion_subsistema": resultado.distribucion_subsistema.tolist(),
        "distribucion_particion": resultado.distribucion_particion.tolist(),
        "elapsed_seconds": elapsed_seconds,
    }


def _ejecutar_estrategia(
    estrategia: str,
    tpm: np.ndarray,
    estado_inicial: str,
    mascara: str,
):
    estrategia = estrategia.lower()

    if estrategia in {"fuerza_bruta", "bruteforce", "fuerzabruta"}:
        solver = FuerzaBruta(tpm)
        inicio = time.perf_counter()
        resultado = solver.aplicar_estrategia(
            estado_inicial=estado_inicial,
            condicion=mascara,
            alcance=mascara,
            mecanismo=mascara,
        )
        elapsed = time.perf_counter() - inicio
        print(f"FuerzaBruta ->\n{resultado}")
        print(
            "Perdida -> "
            f"{resultado.perdida:.4f} | "
            f"subsistema={resultado.distribucion_subsistema.tolist()} vs "
            f"particion={resultado.distribucion_particion.tolist()}"
        )
        return "fuerza_bruta", resultado, elapsed

    if estrategia == "phi":
        solver = Phi(tpm)
        inicio = time.perf_counter()
        resultado = solver.aplicar_estrategia(
            estado_inicial=estado_inicial,
            condicion=mascara,
            alcance=mascara,
            mecanismo=mascara,
        )
        elapsed = time.perf_counter() - inicio
        print(f"Phi ->\n{resultado}")
        return "phi", resultado, elapsed

    if estrategia in {"qnodos", "q_nodes", "qnodes"}:
        solver = QNodos(tpm)
        inicio = time.perf_counter()
        resultado = solver.aplicar_estrategia(
            estado_inicial=estado_inicial,
            condicion=mascara,
            alcance=mascara,
            mecanismo=mascara,
        )
        elapsed = time.perf_counter() - inicio
        print(f"Q-Nodos ->\n{resultado}")
        return "qnodos", resultado, elapsed

    if estrategia == "geometric":
        solver = Geometric(tpm)
        inicio = time.perf_counter()
        resultado = solver.aplicar_estrategia(
            estado_inicial=estado_inicial,
            condicion=mascara,
            alcance=mascara,
            mecanismo=mascara,
        )
        elapsed = time.perf_counter() - inicio
        print(f"Geometric ({aplicacion.modo_geometrico}) ->\n{resultado}")
        return f"geometric_{aplicacion.modo_geometrico}", resultado, elapsed

    raise ValueError(f"Estrategia no soportada: {estrategia}")


def iniciar(
    estrategia: str = "todas",
    modo_geometric: str | None = None,
    estado_inicial: str = "1000",
    output_json: str | None = None,
    csv_muestras: str | None = None,
) -> dict[str, object]:
    """Orquestador inicial del proyecto."""
    logger.info("Inicio de ejecucion en main.iniciar")
    validar_bitstring(estado_inicial)
    aplicacion.set_pagina_red_muestra("A")
    if modo_geometric is not None:
        if modo_geometric not in {GeometricMode.STRICT.value, GeometricMode.REFINED.value}:
            raise ValueError(f"Modo geometrico no soportado: {modo_geometric}")
        aplicacion.modo_geometrico = modo_geometric

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
        f"distancia: {aplicacion.distancia_metrica}, "
        f"modo geometrico: {aplicacion.modo_geometrico}."
    )

    mascara = "1" * len(estado_inicial)
    gestor = Gestor(estado_inicial=estado_inicial)
    if csv_muestras:
        ruta_muestras = Path(csv_muestras)
        tpm = gestor.construir_tpm_desde_csv_muestras(ruta_muestras)
        fuente_tpm = str(ruta_muestras)
        print(
            "TPM estimada desde muestras temporales "
            f"{ruta_muestras} con forma {tpm.shape}."
        )
    else:
        tpm = gestor.cargar_red()
        fuente_tpm = str(gestor.archivo_tpm)
        print(f"TPM cargada desde {gestor.archivo_tpm} con forma {tpm.shape}.")
    logger.debug(f"TPM cargada con forma {tpm.shape}")

    estado_vector = np.array([int(bit) for bit in estado_inicial], dtype=np.int8)
    sistema = Sistema(tpm, estado_vector)
    dist_marginal = sistema.distribucion_marginal()
    print(f"Sistema demo -> distribucion marginal: {dist_marginal.tolist()}")

    resultados: dict[str, object] = {}

    if estrategia == "todas":
        clave, resultado, elapsed = _ejecutar_estrategia("fuerza_bruta", tpm, estado_inicial, mascara)
        resultados[clave] = _solucion_a_dict(resultado, elapsed)

        clave, resultado, elapsed = _ejecutar_estrategia("phi", tpm, estado_inicial, mascara)
        resultados[clave] = _solucion_a_dict(resultado, elapsed)

        clave, resultado, elapsed = _ejecutar_estrategia("qnodos", tpm, estado_inicial, mascara)
        resultados[clave] = _solucion_a_dict(resultado, elapsed)

        aplicacion.set_modo_geometrico(GeometricMode.STRICT)
        clave, resultado, elapsed = _ejecutar_estrategia("geometric", tpm, estado_inicial, mascara)
        resultados[clave] = _solucion_a_dict(resultado, elapsed)

        aplicacion.set_modo_geometrico(GeometricMode.REFINED)
        clave, resultado, elapsed = _ejecutar_estrategia("geometric", tpm, estado_inicial, mascara)
        resultados[clave] = _solucion_a_dict(resultado, elapsed)
    else:
        clave, resultado, elapsed = _ejecutar_estrategia(estrategia, tpm, estado_inicial, mascara)
        resultados[clave] = _solucion_a_dict(resultado, elapsed)

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

    payload = {
        "estrategia_solicitada": estrategia,
        "modo_geometric": aplicacion.modo_geometrico,
        "estado_inicial": estado_inicial,
        "archivo_tpm": fuente_tpm,
        "resultados": resultados,
    }

    if output_json:
        ruta = Path(output_json)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Salida JSON guardada en: {ruta}")

    logger.info("Fin de ejecucion en main.iniciar")
    return payload

"""Orquestador de la capa de Presentación.

Punto de entrada alternativo a ``src/main.py`` que usa inyección de dependencias
en lugar de instanciar estrategias directamente. Delega toda la lógica a los
casos de uso del ``Contenedor``.

Ejemplo de uso::

    from src.presentacion.orquestador import ejecutar
    from src.aplicacion.configuracion import AppConfig
    from src.dominio.enumeraciones import TimeEMD

    resultado = ejecutar(
        estrategia="fuerza_bruta",
        estado_inicial="1000",
        config=AppConfig(tiempo_emd=TimeEMD.JENSEN_SHANNON.value),
    )
"""

import json
import time
from dataclasses import replace
from pathlib import Path

from src.aplicacion.casos_de_uso.buscar_particion import EntradaBusqueda
from src.aplicacion.configuracion import AppConfig
from src.constantes.base import PROJECT_NAME, PROJECT_VERSION
from src.contenedor import Contenedor
from src.modelos.enumeraciones.geometric_mode import GeometricMode


def _solucion_a_dict(resultado, elapsed: float) -> dict:
    return {
        "estrategia": resultado.estrategia,
        "estado_inicial": resultado.estado_inicial,
        "perdida": float(resultado.perdida),
        "particion": resultado.particion,
        "distribucion_subsistema": resultado.distribucion_subsistema.tolist(),
        "distribucion_particion": resultado.distribucion_particion.tolist(),
        "elapsed_seconds": elapsed,
    }


def ejecutar(
    estrategia: str = "fuerza_bruta",
    estado_inicial: str = "1000",
    output_json: str | None = None,
    csv_muestras: str | None = None,
    k_particiones: int = 2,
    config: AppConfig | None = None,
) -> dict:
    """Ejecuta una o todas las estrategias usando el contenedor de DI.

    Parámetros
    ----------
    estrategia      : nombre de la estrategia o "todas"
    estado_inicial  : cadena binaria del estado inicial del sistema
    output_json     : ruta opcional para guardar resultados en JSON
    csv_muestras    : ruta opcional a CSV de muestras para estimar TPM
    k_particiones   : número de grupos para k-partición
    config          : configuración inyectada (usa AppConfig por defecto)
    """
    config = config or AppConfig()
    contenedor = Contenedor(config)
    registro = contenedor.registro("orquestador")

    registro.info(
        f"Presentacion.ejecutar iniciado | estrategia={estrategia} "
        f"estado={estado_inicial} k={k_particiones}"
    )
    print(f"{PROJECT_NAME} v{PROJECT_VERSION} [arquitectura hexagonal]")

    # ── Caso de uso: cargar/estimar TPM ─────────────────────────────────────
    caso_tpm = contenedor.caso_uso_estimar_tpm(estado_inicial)
    if csv_muestras:
        tpm = caso_tpm.estimar_desde_csv(Path(csv_muestras))
    else:
        tpm = caso_tpm.cargar_desde_muestra_predefinida()
    print(f"TPM cargada con forma {tpm.shape}")

    # ── Determinar estrategias a ejecutar ────────────────────────────────────
    if estrategia == "todas":
        nombres = [
            "fuerza_bruta",
            "phi",
            "qnodos",
            f"geometric_{GeometricMode.STRICT.value}",
            f"geometric_{GeometricMode.REFINED.value}",
        ]
    else:
        nombres = [estrategia]

    resultados: dict = {}
    mascara = "1" * len(estado_inicial)

    for nombre in nombres:
        cfg_actual = config
        nombre_base = nombre

        # Estrategia geométrica requiere ajustar el modo en la config
        if nombre.startswith("geometric_"):
            modo = nombre.split("_", 1)[1]
            nombre_base = "geometric"
            cfg_actual = replace(config, modo_geometrico=modo)
            contenedor_actual = Contenedor(cfg_actual)
        else:
            contenedor_actual = contenedor

        caso_busqueda = contenedor_actual.caso_uso_buscar_particion(nombre_base, tpm)
        entrada = EntradaBusqueda(
            estado_inicial=estado_inicial,
            condicion=mascara,
            alcance=mascara,
            mecanismo=mascara,
            k=k_particiones,
        )

        inicio = time.perf_counter()
        resultado = caso_busqueda.ejecutar(entrada)
        elapsed = time.perf_counter() - inicio

        print(f"{nombre} → perdida={resultado.perdida:.4f} ({elapsed:.3f}s)")
        resultados[nombre] = _solucion_a_dict(resultado, elapsed)

    payload = {
        "estrategia_solicitada": estrategia,
        "estado_inicial": estado_inicial,
        "k_particiones": k_particiones,
        "modo_geometrico": config.modo_geometrico,
        "resultados": resultados,
    }

    if output_json:
        ruta = Path(output_json)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Salida JSON guardada en: {ruta}")

    registro.info("Presentacion.ejecutar finalizado")
    return payload

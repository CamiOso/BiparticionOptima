import os
import time

import numpy as np
import pytest

from src.controladores.gestor import Gestor
from src.estrategias.fuerza_bruta import FuerzaBruta
from src.modelos.base.aplicacion import aplicacion
from src.modelos.enumeraciones.geometric_mode import GeometricMode
from src.modelos.nucleo.solucion import Solucion
from src.strategies.geometric import Geometric


def _sample_tpm_4nodes() -> np.ndarray:
    return np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=np.float32,
    )


def _random_tpm(num_nodes: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((1 << num_nodes, num_nodes), dtype=np.float32)


def test_geometric_returns_solution() -> None:
    aplicacion.set_pagina_red_muestra("A")
    aplicacion.set_modo_geometrico(GeometricMode.REFINED)
    estrategia = Geometric(_sample_tpm_4nodes())

    resultado = estrategia.aplicar_estrategia(
        estado_inicial="1000",
        condicion="1111",
        alcance="1111",
        mecanismo="1111",
    )

    assert isinstance(resultado, Solucion)
    assert resultado.estrategia == "Geometric"
    assert resultado.perdida >= 0.0
    assert resultado.distribucion_subsistema.shape == (4,)


def test_geometric_matches_bruteforce_in_small_case() -> None:
    aplicacion.set_pagina_red_muestra("A")
    tpm = Gestor("1000").cargar_red()

    fuerza_bruta = FuerzaBruta(tpm)
    geometrica = Geometric(tpm)

    resultado_fb = fuerza_bruta.aplicar_estrategia(
        estado_inicial="1000",
        condicion="1110",
        alcance="1110",
        mecanismo="1110",
    )
    resultado_geo = geometrica.aplicar_estrategia(
        estado_inicial="1000",
        condicion="1110",
        alcance="1110",
        mecanismo="1110",
    )

    assert np.isclose(resultado_geo.perdida, resultado_fb.perdida)
    assert np.allclose(
        resultado_geo.distribucion_subsistema,
        resultado_fb.distribucion_subsistema,
    )
    assert np.allclose(
        resultado_geo.distribucion_particion,
        resultado_fb.distribucion_particion,
    )


def test_geometric_strict_returns_solution() -> None:
    estrategia = Geometric(_sample_tpm_4nodes(), mode=GeometricMode.STRICT)

    resultado = estrategia.aplicar_estrategia(
        estado_inicial="1000",
        condicion="1111",
        alcance="1111",
        mecanismo="1111",
    )

    assert isinstance(resultado, Solucion)
    assert resultado.estrategia == "Geometric"
    assert resultado.perdida >= 0.0


def test_geometric_uses_application_mode_by_default() -> None:
    aplicacion.set_modo_geometrico(GeometricMode.STRICT)
    estrategia = Geometric(_sample_tpm_4nodes())
    assert estrategia.mode == GeometricMode.STRICT.value

    aplicacion.set_modo_geometrico(GeometricMode.REFINED)
    estrategia = Geometric(_sample_tpm_4nodes())
    assert estrategia.mode == GeometricMode.REFINED.value


def test_geometric_matches_bruteforce_for_5_nodes() -> None:
    num_nodos = 5
    tpm = _random_tpm(num_nodos, seed=23)
    estado = "0" * num_nodos
    mascara = "1" * num_nodos

    fuerza_bruta = FuerzaBruta(tpm)
    geometrica = Geometric(tpm, mode=GeometricMode.REFINED)

    resultado_fb = fuerza_bruta.aplicar_estrategia(
        estado_inicial=estado,
        condicion=mascara,
        alcance=mascara,
        mecanismo=mascara,
    )
    resultado_geo = geometrica.aplicar_estrategia(
        estado_inicial=estado,
        condicion=mascara,
        alcance=mascara,
        mecanismo=mascara,
    )

    assert np.isclose(resultado_geo.perdida, resultado_fb.perdida)
    assert np.allclose(resultado_geo.distribucion_particion, resultado_fb.distribucion_particion)


@pytest.mark.skipif(
    os.getenv("RUN_BENCHMARKS") != "1",
    reason="Benchmark desactivado por defecto (export RUN_BENCHMARKS=1 para habilitarlo).",
)
def test_geometric_is_faster_than_bruteforce_for_8_nodes() -> None:
    num_nodos = 8
    tpm = _random_tpm(num_nodos)
    estado = "0" * num_nodos
    mascara = "1" * num_nodos

    fuerza_bruta = FuerzaBruta(tpm)
    geometrica = Geometric(tpm)

    inicio_fb = time.perf_counter()
    _ = fuerza_bruta.aplicar_estrategia(
        estado_inicial=estado,
        condicion=mascara,
        alcance=mascara,
        mecanismo=mascara,
    )
    tiempo_fb = time.perf_counter() - inicio_fb

    inicio_geo = time.perf_counter()
    _ = geometrica.aplicar_estrategia(
        estado_inicial=estado,
        condicion=mascara,
        alcance=mascara,
        mecanismo=mascara,
    )
    tiempo_geo = time.perf_counter() - inicio_geo

    assert tiempo_geo < tiempo_fb

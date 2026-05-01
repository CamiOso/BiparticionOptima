import numpy as np

from src.controladores.gestor import Gestor
from src.funciones.particiones import k_particiones_asignacion
from src.modelos.base.aplicacion import aplicacion
from src.modelos.nucleo.solucion import Solucion
from src.estrategias.q_nodos import QNodos, _BuscadorKQNodos


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


def test_qnodes_returns_solution() -> None:
    strategy = QNodos(_sample_tpm_4nodes())

    result = strategy.aplicar_estrategia(
        estado_inicial="1000",
        condicion="1111",
        alcance="1111",
        mecanismo="1111",
    )

    assert isinstance(result, Solucion)
    assert result.perdida >= 0.0
    assert result.distribucion_subsistema.shape == (4,)
    assert result.distribucion_particion.shape == (4,)
    assert "G1(" in result.particion
    assert "G2(" in result.particion


def test_qnodes_matches_sample_a_reference_case() -> None:
    aplicacion.set_pagina_red_muestra("A")
    estrategia = QNodos(Gestor("1000").cargar_red())

    resultado = estrategia.aplicar_estrategia(
        estado_inicial="1000",
        condicion="1110",
        alcance="1110",
        mecanismo="1110",
    )

    assert resultado.distribucion_subsistema.tolist() == [0.0, 0.0, 1.0]
    assert resultado.distribucion_particion.tolist() == [0.0, 0.0, 0.5]
    assert resultado.perdida == 0.5


def test_qnodes_accepts_k_particiones() -> None:
    estrategia = QNodos(_sample_tpm_4nodes())

    resultado = estrategia.aplicar_estrategia(
        estado_inicial="1000",
        condicion="1111",
        alcance="1111",
        mecanismo="1111",
        k=3,
    )

    assert isinstance(resultado, Solucion)
    assert resultado.perdida >= 0.0
    assert "G0(" in resultado.particion


def test_qnodes_k_particiones_exacto_mejora_o_iguala_k2_temporal() -> None:
    estrategia = QNodos(_sample_tpm_4nodes())
    estrategia.sia_preparar_subsistema("1000", "1111", "1111", "1111")
    assert estrategia.sia_subsistema is not None

    futuro = [(1, int(indice)) for indice in estrategia.sia_subsistema.indices_ncubos.tolist()]
    presente = [(0, int(indice)) for indice in estrategia.sia_subsistema.dims_ncubos.tolist()]
    vertices = list(presente + futuro)

    resultado_k3 = estrategia.aplicar_estrategia(
        estado_inicial="1000",
        condicion="1111",
        alcance="1111",
        mecanismo="1111",
        k=3,
    )

    estrategia.sia_preparar_subsistema("1000", "1111", "1111", "1111")
    assert estrategia.sia_dists_marginales is not None
    cache_k2: dict = {}
    buscador_k2 = _BuscadorKQNodos(
        vertices=vertices,
        sistema=estrategia.sia_subsistema,
        dists=estrategia.sia_dists_marginales,
        distancia_metrica=estrategia.distancia_metrica,
        cache=cache_k2,
    )
    mejor_k2 = min(
        buscador_k2.evaluar_asignacion(asignacion)[0]
        for asignacion in k_particiones_asignacion(len(vertices), 2)
    )

    assert resultado_k3.perdida <= mejor_k2 + 1e-12

import numpy as np

from src.controladores.gestor import Gestor
from src.modelos.base.aplicacion import aplicacion
from src.modelos.nucleo.solucion import Solucion
from src.estrategias.q_nodos import QNodos


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

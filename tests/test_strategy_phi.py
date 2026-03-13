import numpy as np

from src.modelos.nucleo.solucion import Solucion
from src.estrategias.phi import Phi


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


def test_phi_returns_solution() -> None:
    strategy = Phi(_sample_tpm_4nodes())

    result = strategy.aplicar_estrategia(
        estado_inicial="1000",
        condicion="1111",
        alcance="1111",
        mecanismo="1111",
    )

    assert isinstance(result, Solucion)
    assert result.perdida >= 0.0
    assert result.distribucion_subsistema.size > 0
    assert result.distribucion_particion.size > 0


def test_phi_aplica_filtros_de_subsistema() -> None:
    strategy = Phi(_sample_tpm_4nodes())

    result = strategy.aplicar_estrategia(
        estado_inicial="1000",
        condicion="1110",
        alcance="1110",
        mecanismo="1110",
    )

    assert isinstance(result, Solucion)
    assert result.distribucion_subsistema.size > 0

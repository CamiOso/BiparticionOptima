import numpy as np

from src.modelos.nucleo.sistema import Sistema


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


def test_system_distribucion_marginal_shape() -> None:
    tpm = _sample_tpm_4nodes()
    estado = np.array([1, 0, 0, 0], dtype=np.int8)
    system = Sistema(tpm, estado)

    dist = system.distribucion_marginal()

    assert dist.shape == (4,)


def test_system_bipartir_keeps_indices_subset() -> None:
    tpm = _sample_tpm_4nodes()
    estado = np.array([1, 0, 0, 0], dtype=np.int8)
    system = Sistema(tpm, estado)

    partido = system.bipartir(
        alcance_preservado=np.array([0, 1], dtype=np.int8),
        mecanismo_preservado=np.array([0, 1], dtype=np.int8),
    )

    assert set(partido.indices_ncubos.tolist()).issubset({0, 1})

import numpy as np

from src.funciones.iit import emd_efecto, estados_binarios, lil_endian, literales


def test_emd_efecto_zero_when_equal() -> None:
    u = np.array([0.2, 0.8], dtype=np.float32)
    v = np.array([0.2, 0.8], dtype=np.float32)
    assert emd_efecto(u, v) == 0.0


def test_emd_efecto_sum_abs_diff() -> None:
    u = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    assert emd_efecto(u, v) == 1.0


def test_literales_generates_expected_label() -> None:
    indices = np.array([0, 1, 3], dtype=np.int8)
    assert literales(indices) == "ABD"


def test_lil_endian_reorders_states() -> None:
    assert lil_endian(2).tolist() == [0, 2, 1, 3]


def test_estados_binarios_excludes_zero_state() -> None:
    assert estados_binarios(2) == ["01", "10", "11"]

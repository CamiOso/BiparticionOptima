import numpy as np

from src.funcs.iit import emd_efecto


def test_emd_efecto_zero_when_equal() -> None:
    u = np.array([0.2, 0.8], dtype=np.float32)
    v = np.array([0.2, 0.8], dtype=np.float32)
    assert emd_efecto(u, v) == 0.0


def test_emd_efecto_sum_abs_diff() -> None:
    u = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    assert emd_efecto(u, v) == 1.0

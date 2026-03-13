import numpy as np

from src.funciones.particiones import biparticiones


def test_biparticiones_not_empty_and_omits_empty_empty() -> None:
    alcance = np.array([0, 1], dtype=np.int8)
    mecanismo = np.array([0], dtype=np.int8)

    parts = list(biparticiones(alcance, mecanismo))

    assert len(parts) > 0
    assert ((), ()) not in parts


def test_biparticiones_contains_full_partition() -> None:
    alcance = np.array([0, 1], dtype=np.int8)
    mecanismo = np.array([0], dtype=np.int8)

    parts = list(biparticiones(alcance, mecanismo))

    assert ((0, 1), (0,)) in parts

from itertools import combinations
from typing import Iterator

import numpy as np


def subconjuntos(arr: np.ndarray) -> Iterator[tuple[int, ...]]:
    """Genera todos los subconjuntos posibles de un arreglo de indices."""
    for r in range(len(arr) + 1):
        yield from combinations(arr.tolist(), r)


def biparticiones(
    alcance_indices: np.ndarray,
    mecanismo_indices: np.ndarray,
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Genera pares (subalcance, submecanismo), omitiendo el caso vacio-vacio."""
    for subalcance in subconjuntos(alcance_indices):
        for submecanismo in subconjuntos(mecanismo_indices):
            if len(subalcance) == 0 and len(submecanismo) == 0:
                continue
            yield subalcance, submecanismo

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
    """Genera biparticiones, omitiendo los casos triviales vacio-vacio y total-total."""
    alcance_total = tuple(int(v) for v in alcance_indices.tolist())
    mecanismo_total = tuple(int(v) for v in mecanismo_indices.tolist())
    for subalcance in subconjuntos(alcance_indices):
        for submecanismo in subconjuntos(mecanismo_indices):
            if len(subalcance) == 0 and len(submecanismo) == 0:
                continue
            if subalcance == alcance_total and submecanismo == mecanismo_total:
                continue
            yield subalcance, submecanismo


def generar_candidatos(cantidad_nodos: int) -> Iterator[tuple[int, ...]]:
    """Genera subconjuntos de indices a condicionar, excluyendo condicionar todo."""
    indices = np.arange(cantidad_nodos, dtype=np.int8)
    for cantidad in range(cantidad_nodos):
        yield from combinations(indices.tolist(), cantidad)


def generar_subsistemas(
    dimensiones_candidato: np.ndarray,
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Genera pares de indices a sustraer para alcance y mecanismo."""
    for alcance_removido in subconjuntos(dimensiones_candidato):
        for mecanismo_removido in subconjuntos(dimensiones_candidato):
            yield alcance_removido, mecanismo_removido


def etiqueta_subconjunto(
    subconjunto: tuple[int, ...],
    total: tuple[int, ...],
) -> str:
    """Convierte un subconjunto a mascara binaria segun el total ordenado."""
    return "".join("1" if indice in subconjunto else "0" for indice in total)

from itertools import combinations, product as iproduct
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


def k_particiones_asignacion(n_nodos: int, k: int) -> Iterator[tuple[int, ...]]:
    """Genera asignaciones canonicas de n nodos con entre 2 y k grupos.

    Canonico: los grupos aparecen en orden de primera ocurrencia, asi 0 siempre
    aparece antes que 1, 1 antes que 2, etc. Esto elimina duplicados por
    permutacion de etiquetas de grupo.

    Buscar hasta k grupos (no exactamente k) permite que la MIP sea siempre
    igual o mejor que la biparticion: mas grados de libertad, menor o igual perdida.
    """
    if k < 2 or n_nodos < 2:
        return
    k_eff = min(k, n_nodos)
    for asignacion in iproduct(range(k_eff), repeat=n_nodos):
        siguiente = 0
        es_canon = True
        grupos_vistos: set[int] = set()
        for g in asignacion:
            if g not in grupos_vistos:
                if g != siguiente:
                    es_canon = False
                    break
                grupos_vistos.add(g)
                siguiente += 1
        if es_canon and siguiente >= 2:
            yield asignacion


def etiqueta_subconjunto(
    subconjunto: tuple[int, ...],
    total: tuple[int, ...],
) -> str:
    """Convierte un subconjunto a mascara binaria segun el total ordenado."""
    return "".join("1" if indice in subconjunto else "0" for indice in total)

from itertools import combinations, product as iproduct
from typing import Iterator

import numpy as np


def subconjuntos(arr: np.ndarray) -> Iterator[tuple[int, ...]]:
    """Genera todos los subconjuntos posibles de un arreglo de índices.

    Itera de tamaño 0 (conjunto vacío) hasta tamaño len(arr) (conjunto
    completo), incluyendo ambos extremos.

    Parámetros
    ----------
    arr : np.ndarray
        Arreglo de índices enteros del que se generan los subconjuntos.

    Yields
    ------
    tuple[int, ...]
        Cada subconjunto como tupla ordenada de enteros.
    """
    for r in range(len(arr) + 1):
        yield from combinations(arr.tolist(), r)


def biparticiones(
    alcance_indices: np.ndarray,
    mecanismo_indices: np.ndarray,
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Genera todas las biparticiones válidas del sistema (alcance × mecanismo).

    Una bipartición válida es un par (subalcance, submecanismo) que no sea
    ninguno de los dos casos triviales:
      - ``(vacío, vacío)``  → no parte nada.
      - ``(alcance_total, mecanismo_total)`` → partición que coincide con el
        sistema completo (equivale a no particionar).

    Parámetros
    ----------
    alcance_indices : np.ndarray
        Índices de los nodos del alcance (t+1) del subsistema.
    mecanismo_indices : np.ndarray
        Índices de los nodos del mecanismo (t) del subsistema.

    Yields
    ------
    tuple[tuple[int, ...], tuple[int, ...]]
        Par ``(subalcance, submecanismo)`` donde cada elemento es un
        subconjunto de los índices correspondientes.
    """
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
    """Genera subconjuntos de índices a condicionar, excluyendo condicionar todo.

    Produce todos los subconjuntos de tamaño 0 a cantidad_nodos - 1 de los
    índices [0, 1, ..., cantidad_nodos-1]. Se omite el subconjunto completo
    porque condicionar todos los nodos vaciaría el sistema.

    Parámetros
    ----------
    cantidad_nodos : int
        Número total de nodos del sistema.

    Yields
    ------
    tuple[int, ...]
        Subconjunto de índices a condicionar (como enteros).
    """
    indices = np.arange(cantidad_nodos, dtype=np.int8)
    for cantidad in range(cantidad_nodos):
        yield from combinations(indices.tolist(), cantidad)


def generar_subsistemas(
    dimensiones_candidato: np.ndarray,
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Genera pares de índices a sustraer para alcance y mecanismo.

    Combina todos los subconjuntos de ``dimensiones_candidato`` para el
    alcance con todos los subconjuntos para el mecanismo. El resultado
    representa las posibles sustracciones de entrada al construir subsistemas.

    Parámetros
    ----------
    dimensiones_candidato : np.ndarray
        Arreglo de índices de dimensiones candidatas a sustraer.

    Yields
    ------
    tuple[tuple[int, ...], tuple[int, ...]]
        Par ``(alcance_removido, mecanismo_removido)`` con los índices a
        eliminar de cada componente del subsistema.
    """
    for alcance_removido in subconjuntos(dimensiones_candidato):
        for mecanismo_removido in subconjuntos(dimensiones_candidato):
            yield alcance_removido, mecanismo_removido


def k_particiones_asignacion(n_nodos: int, k: int) -> Iterator[tuple[int, ...]]:
    """Genera asignaciones canónicas de n nodos con entre 2 y k grupos.

    Una asignación canónica es aquella donde los grupos aparecen en orden de
    primera ocurrencia: el grupo 0 siempre aparece antes que el 1, el 1 antes
    que el 2, etc. Esto elimina duplicados por permutación de etiquetas de
    grupo (e.g. (0,1,0) y (1,0,1) representan la misma partición estructural;
    solo se genera la forma (0,1,0)).

    Buscar hasta k grupos (no exactamente k) garantiza que la MIP k-partición
    sea siempre igual o mejor que la bipartición: más grados de libertad
    implican menor o igual pérdida.

    Parámetros
    ----------
    n_nodos : int
        Número de elementos a particionar. Debe ser >= 2.
    k : int
        Número máximo de grupos permitidos. Debe ser >= 2.

    Yields
    ------
    tuple[int, ...]
        Asignación canónica con exactamente entre 2 y min(k, n_nodos) grupos.

    Precondición
    ------------
    Si k < 2 o n_nodos < 2, no se genera ningún elemento.
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
    """Convierte un subconjunto a máscara binaria según el total ordenado.

    Cada posición de la cadena de salida corresponde al elemento en la misma
    posición de ``total``: '1' si el elemento está en ``subconjunto``,
    '0' si no.

    Parámetros
    ----------
    subconjunto : tuple[int, ...]
        Subconjunto cuyos elementos se marcan con '1'.
    total : tuple[int, ...]
        Conjunto de referencia ordenado que define el orden de los bits.

    Retorna
    -------
    str
        Cadena binaria de longitud ``len(total)``.

    Ejemplo
    -------
    >>> etiqueta_subconjunto((0, 2), (0, 1, 2))
    '101'
    """
    return "".join("1" if indice in subconjunto else "0" for indice in total)

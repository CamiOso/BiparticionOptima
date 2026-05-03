"""Helper compartido: construcción del grafo de acoplamientos del sistema.

La matriz de afinidad W[i][j] mide la sensibilidad promedio del nodo i a
cambios en el nodo j, derivada de los n-cubos del subsistema. Es simétrica y
no negativa. Se usa como entrada en Louvain, ILP y Belief Propagation.
"""

from __future__ import annotations

import numpy as np

from src.modelos.nucleo.sistema import Sistema


def construir_afinidad(subsistema: Sistema) -> tuple[list[int], np.ndarray]:
    """Construye la matriz de afinidad W y la lista de nodos del subsistema.

    Parámetros
    ----------
    subsistema : Sistema
        Sistema ya condicionado y sustraído (resultado de sia_preparar_subsistema).

    Retorna
    -------
    nodos : list[int]
        Índices de los nodos en orden ascendente.
    W : np.ndarray  shape (n, n)
        Matriz simétrica de acoplamientos. W[i][j] = sensibilidad media del
        nodo nodos[i] a cambios en el nodo nodos[j].
    """
    nodos = sorted(
        set(int(v) for v in subsistema.indices_ncubos.tolist())
        | set(int(v) for v in subsistema.dims_ncubos.tolist())
    )
    n = len(nodos)
    idx_de = {nodo: i for i, nodo in enumerate(nodos)}
    W = np.zeros((n, n), dtype=np.float64)

    for cubo in subsistema.ncubos:
        fi = idx_de.get(int(cubo.indice))
        if fi is None:
            continue
        for dim in cubo.dims.tolist():
            fj = idx_de.get(int(dim))
            if fj is None:
                continue
            s = _sensibilidad(cubo, int(dim))
            W[fi, fj] += s
            W[fj, fi] += s

    return nodos, W


def _sensibilidad(cubo, dim_nodo: int) -> float:
    """Diferencia finita promediada: |P(i=1|X_j=0,…) - P(i=1|X_j=1,…)|."""
    dims = cubo.dims.tolist()
    if dim_nodo not in dims:
        return 0.0

    pos = dims.index(dim_nodo)
    data = cubo.data
    n_dims = len(dims)

    if n_dims == 1:
        return abs(float(data[0]) - float(data[1]))

    otras_dims = [d for d in range(n_dims) if d != pos]
    otras_sizes = [data.shape[d] for d in otras_dims]
    n_otras = 1
    for s in otras_sizes:
        n_otras *= s

    total = 0.0
    for estado_idx in range(n_otras):
        idx_otras: list[int] = []
        temp = estado_idx
        for s in reversed(otras_sizes):
            idx_otras.append(temp % s)
            temp //= s
        idx_otras = list(reversed(idx_otras))

        idx_0 = [0] * n_dims
        idx_1 = [0] * n_dims
        idx_0[pos] = 0
        idx_1[pos] = 1
        for k_ot, d_ot in enumerate(otras_dims):
            idx_0[d_ot] = idx_otras[k_ot]
            idx_1[d_ot] = idx_otras[k_ot]

        total += abs(float(data[tuple(idx_0)]) - float(data[tuple(idx_1)]))

    return total / n_otras

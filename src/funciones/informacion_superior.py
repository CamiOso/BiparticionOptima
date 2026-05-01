"""Informacion mutua de orden superior: O-information y correlacion total.

La informacion mutua clasica mide dependencia entre dos variables. Para sistemas
de n variables, existen generalizaciones que capturan efectos colectivos:

    Correlacion total:  TC(X) = Σᵢ H(Xᵢ) - H(X₁,...,Xₙ)           >= 0
    O-information:      Ω(X)  = (n-2)H(X) - ΣH(Xᵢ) + Σᵢ<ⱼ H(Xᵢ,Xⱼ)

La O-information (Rosas et al. 2019) mide si el sistema es dominantemente
redundante (Ω > 0) o sinergico (Ω < 0):

    Redundancia: varios nodos codifican la misma informacion => Ω > 0
    Sinergia:    la informacion conjunta supera la suma de las partes => Ω < 0

En el contexto de IIT, la sinergia esta relacionada con la integracion del
sistema: un sistema altamente sinergico es mas resistente a ser partido porque
pierde informacion colectiva que ninguna parte contiene individualmente.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from src.funciones.entropia import shannon


def _distribucion_conjunta(tpm: NDArray, estado_inicial: NDArray) -> NDArray:
    """P(X_{t+1} = s | X_t = x0) bajo independencia condicional entre nodos.

    Bajo este supuesto (comun en modelos IIT de campo medio), la distribucion
    conjunta futura se factoriza como el producto de las marginales:

        P(X_{t+1} = s | X_t = x₀) = Πᵢ P(Xᵢ_{t+1} = sᵢ | X_t = x₀)

    Retorna un vector de longitud 2^n con las probabilidades de cada estado futuro.
    """
    n = tpm.shape[1]
    potencias = 1 << np.arange(n - 1, -1, -1, dtype=np.int64)
    idx = int(np.clip(np.sum(estado_inicial.astype(np.int64) * potencias), 0, tpm.shape[0] - 1))
    probs = tpm[idx, :].astype(np.float64)

    estados = np.arange(2 ** n, dtype=np.int64)
    joint = np.ones(2 ** n, dtype=np.float64)
    for i in range(n):
        bit_i = (estados >> (n - 1 - i)) & 1
        joint *= np.where(bit_i == 1, probs[i], 1.0 - probs[i])
    return joint


def _marginalizar(joint: NDArray, nodos: list[int], n_total: int) -> NDArray:
    """Marginaliza la distribucion conjunta sobre los nodos indicados."""
    n_sub = len(nodos)
    marginal = np.zeros(2 ** n_sub, dtype=np.float64)
    for estado in range(2 ** n_total):
        idx_sub = 0
        for pos, nodo in enumerate(nodos):
            bit = (estado >> (n_total - 1 - nodo)) & 1
            idx_sub = (idx_sub << 1) | bit
        marginal[idx_sub] += joint[estado]
    return marginal


def o_information(tpm: NDArray, estado_inicial: NDArray) -> dict[str, object]:
    """Calcula la O-information y la correlacion total del sistema.

    Interpretacion de los resultados:
        o_information > 0  => redundancia dominante (nodos comparten informacion)
        o_information < 0  => sinergia dominante (el todo supera a las partes)
        correlacion_total  => informacion compartida total (siempre >= 0)

    La relacion con phi de IIT es directa: sistemas con alta sinergia tienden
    a tener mayor phi porque la particion destruye informacion que solo existe
    en el sistema completo.
    """
    n = tpm.shape[1]
    joint = _distribucion_conjunta(tpm, estado_inicial)

    h_conjunta = shannon(joint)

    h_marginales = [
        shannon(_marginalizar(joint, [i], n))
        for i in range(n)
    ]

    h_pares = [
        shannon(_marginalizar(joint, [i, j], n))
        for i, j in combinations(range(n), 2)
    ]

    # Ω = (n-2)·H(X) - Σᵢ H(Xᵢ) + Σᵢ<ⱼ H(Xᵢ,Xⱼ)
    omega = (
        (n - 2) * h_conjunta
        - sum(h_marginales)
        + sum(h_pares)
    )

    # TC = Σᵢ H(Xᵢ) - H(X₁,...,Xₙ)
    tc = sum(h_marginales) - h_conjunta

    if omega > 1e-9:
        tipo = "redundancia"
    elif omega < -1e-9:
        tipo = "sinergia"
    else:
        tipo = "neutral"

    return {
        "o_information": float(omega),
        "correlacion_total": float(tc),
        "entropia_conjunta": float(h_conjunta),
        "entropias_marginales": [float(h) for h in h_marginales],
        "entropias_pares": [float(h) for h in h_pares],
        "tipo": tipo,
        "n_nodos": n,
    }


def informacion_mutua_condicional(
    tpm: NDArray,
    estado_inicial: NDArray,
    nodo_x: int,
    nodo_y: int,
    nodos_condicion: list[int],
) -> float:
    """I(X;Y|Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z).

    Mide si X y Y siguen siendo dependientes una vez conocido Z.
    En un sistema irreducible, muchos pares (X,Y) mantienen dependencia
    condicional significativa: la particion no puede eliminarla completamente.
    """
    n = tpm.shape[1]
    joint = _distribucion_conjunta(tpm, estado_inicial)

    nodos_xyz = sorted(set([nodo_x, nodo_y] + nodos_condicion))
    nodos_xz = sorted(set([nodo_x] + nodos_condicion))
    nodos_yz = sorted(set([nodo_y] + nodos_condicion))
    nodos_z = sorted(nodos_condicion)

    h_xyz = shannon(_marginalizar(joint, nodos_xyz, n))
    h_xz = shannon(_marginalizar(joint, nodos_xz, n))
    h_yz = shannon(_marginalizar(joint, nodos_yz, n))
    h_z = shannon(_marginalizar(joint, nodos_z, n)) if nodos_z else 0.0

    return float(h_xz + h_yz - h_xyz - h_z)


def matriz_dependencia(tpm: NDArray, estado_inicial: NDArray) -> NDArray:
    """Matriz n x n de informacion mutua I(Xᵢ; Xⱼ) entre todos los pares de nodos.

    Permite visualizar rapidamente que pares de nodos comparten mas informacion
    y cuales son mas independientes. La diagonal es H(Xᵢ) (entropia propia).
    """
    n = tpm.shape[1]
    joint = _distribucion_conjunta(tpm, estado_inicial)
    matriz = np.zeros((n, n), dtype=np.float64)

    h_marginales = [shannon(_marginalizar(joint, [i], n)) for i in range(n)]
    for i in range(n):
        matriz[i, i] = h_marginales[i]

    for i, j in combinations(range(n), 2):
        h_ij = shannon(_marginalizar(joint, [i, j], n))
        im = h_marginales[i] + h_marginales[j] - h_ij
        matriz[i, j] = im
        matriz[j, i] = im

    return matriz.astype(np.float32)

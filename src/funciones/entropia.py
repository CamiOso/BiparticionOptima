"""Medidas de entropia de orden superior para distribuciones de probabilidad.

Implementa la familia parametrica de entropias que generaliza a Shannon:

    - Shannon:  H(X) = -Σ pᵢ log₂ pᵢ
    - Renyi:    H_α(X) = 1/(1-α) · log₂(Σ pᵢᵅ)          [α ≠ 1]
    - Tsallis:  Sq(X) = (1 - Σ pᵢq) / (q - 1)             [q ≠ 1]

Para α=1 o q=1 ambas convergen a Shannon (por L'Hopital). Para α=2 la entropia
de Renyi corresponde a la informacion de colision; para α→∞ converge al
logaritmo del maximo de la distribucion (entropia min).

La entropia de Tsallis no es extensiva (Sq(AB) ≠ Sq(A) + Sq(B)), lo que la hace
adecuada para sistemas con correlaciones de largo alcance, como los sistemas
biologicos que modela IIT.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _preparar(p: NDArray) -> NDArray:
    """Normaliza y recorta para evitar log(0)."""
    p64 = np.clip(p.astype(np.float64), 0.0, None)
    total = p64.sum()
    if total <= 0:
        return np.ones(len(p64), dtype=np.float64) / len(p64)
    return p64 / total


def shannon(p: NDArray) -> float:
    """Entropia de Shannon H(X) = -Σ pᵢ log₂ pᵢ."""
    p64 = _preparar(p)
    mascara = p64 > 0
    return float(-np.sum(p64[mascara] * np.log2(p64[mascara])))


def renyi(p: NDArray, alpha: float = 2.0) -> float:
    """Entropia de Renyi de orden alpha.

    Casos especiales:
        alpha = 0  -> log₂(|soporte|)   (entropia de Hartley)
        alpha = 1  -> entropia de Shannon (limite por L'Hopital)
        alpha = 2  -> informacion de colision (-log₂ pureza)
        alpha → ∞  -> -log₂ max(p)      (entropia min)
    """
    if abs(alpha - 1.0) < 1e-9:
        return shannon(p)

    p64 = _preparar(p)

    if alpha == 0.0:
        return float(np.log2(np.sum(p64 > 1e-15)))

    suma_potencia = float(np.sum(p64 ** alpha))
    if suma_potencia <= 0:
        return 0.0
    return float(np.log2(suma_potencia) / (1.0 - alpha))


def tsallis(p: NDArray, q: float = 2.0) -> float:
    """Entropia de Tsallis de orden q.

    A diferencia de Renyi, Tsallis es no extensiva: para sistemas independientes
    A y B se cumple Sq(AB) = Sq(A) + Sq(B) + (1-q)·Sq(A)·Sq(B).

    La no extensividad captura correlaciones estadisticas en sistemas complejos:
    si q < 1 el sistema exhibe multifractalidad; si q > 1, las probabilidades
    pequeñas contribuyen menos que en Shannon (mas robusto ante eventos raros).
    """
    if abs(q - 1.0) < 1e-9:
        return shannon(p)

    p64 = _preparar(p)
    return float((1.0 - np.sum(p64 ** q)) / (q - 1.0))


def perfil_entropia(p: NDArray, alphas: list[float] | None = None) -> dict[float, float]:
    """Calcula la entropia de Renyi para un rango de ordenes alpha.

    El perfil describe la estructura de la distribucion: una curva plana
    indica una distribucion uniforme; una curva con pendiente pronunciada
    indica alta concentracion en pocos estados.
    """
    if alphas is None:
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, float("inf")]
    resultado = {}
    p64 = _preparar(p)
    for alpha in alphas:
        if alpha == float("inf"):
            max_p = float(p64.max())
            resultado[alpha] = float(-np.log2(max_p)) if max_p > 0 else 0.0
        else:
            resultado[alpha] = renyi(p64, alpha)
    return resultado


def divergencia_renyi(p: NDArray, q: NDArray, alpha: float = 2.0) -> float:
    """Divergencia de Renyi D_alpha(p || q).

    D_α(p||q) = 1/(α-1) · log₂ Σᵢ pᵢᵅ · qᵢ^(1-α)

    Generaliza la divergencia KL (limite α→1). Para α=0.5, la divergencia
    de Renyi es monotonamente relacionada con la distancia de Bhattacharyya.
    """
    if abs(alpha - 1.0) < 1e-9:
        p64 = _preparar(p)
        q64 = _preparar(q)
        q64 = np.clip(q64, 1e-300, None)
        return float(np.sum(p64 * np.log2(p64 / q64 + 1e-300)))

    p64 = _preparar(p)
    q64 = _preparar(q)
    q64 = np.clip(q64, 1e-300, None)

    suma = float(np.sum(p64 ** alpha * q64 ** (1.0 - alpha)))
    if suma <= 0:
        return float("inf")
    return float(np.log2(suma) / (alpha - 1.0))

"""Analisis espectral de la Matriz de Probabilidad de Transicion (TPM).

La TPM define un proceso de Markov de tiempo discreto sobre {0,1}^n. Su
eigendescomposicion revela la dinamica de largo plazo del sistema:

    - Eigenvalor dominante λ₁ = 1 (Perron-Frobenius): el sistema converge.
    - Eigenvector asociado: la distribucion estacionaria π.
    - Brecha espectral gap = 1 - |λ₂|: velocidad de convergencia a π.
    - Tiempo de mezcla t_mix: pasos para estar ε-cerca de π desde cualquier estado.

Relacion con IIT: sistemas con brecha espectral pequeña (|λ₂| ≈ 1) tienen
memoria larga — tardan mucho en olvidar el estado inicial. Esto se correlaciona
con alta irreducibilidad: un sistema con dependencias temporales fuertes es
dificil de particionar sin perder informacion dinamica.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class ResultadoEspectral:
    """Resultado del analisis espectral de una TPM."""

    eigenvalores: NDArray
    distribucion_estacionaria: NDArray
    brecha_espectral: float
    tiempo_mezcla_cota: float
    radio_espectral: float
    es_ergodica: bool
    entropia_estacionaria: float
    n_nodos: int
    n_estados: int

    def imprimir(self) -> None:
        ancho = 60
        print("=" * ancho)
        print(f"{'ANALISIS ESPECTRAL':^{ancho}}")
        print("=" * ancho)
        print(f"  Nodos          : {self.n_nodos}")
        print(f"  Estados        : {self.n_estados}")
        print(f"  Ergodica       : {'si' if self.es_ergodica else 'no'}")
        print(f"  Brecha espect. : {self.brecha_espectral:.6f}")
        print(f"  Radio espect.  : {self.radio_espectral:.6f}")
        print(f"  Tiempo mezcla  : {self.tiempo_mezcla_cota:.2f} pasos")
        print(f"  H(estacionaria): {self.entropia_estacionaria:.4f} bits")
        print(f"  Eigenvalores   : {[f'{v:.4f}' for v in self.eigenvalores[:6]]}")
        print("=" * ancho)


def _construir_matriz_transicion(tpm: NDArray) -> NDArray:
    """Convierte TPM (2^n x n) en matriz de transicion P (2^n x 2^n).

    P[estado_actual, estado_futuro] = P(X_{t+1} = estado_futuro | X_t = estado_actual)

    Bajo independencia condicional entre nodos (supuesto estandar en IIT de campo
    medio), la distribucion conjunta futura se factoriza como el producto de
    las marginales de cada nodo.
    """
    num_estados, n = tpm.shape[0], tpm.shape[1]
    P = np.zeros((num_estados, num_estados), dtype=np.float64)

    estados_futuros = np.arange(num_estados, dtype=np.int64)

    for estado_actual in range(num_estados):
        probs = tpm[estado_actual, :].astype(np.float64)
        fila = np.ones(num_estados, dtype=np.float64)
        for i in range(n):
            bit_i = (estados_futuros >> (n - 1 - i)) & 1
            fila *= np.where(bit_i == 1, probs[i], 1.0 - probs[i])
        P[estado_actual, :] = fila

    return P


def analizar_tpm(tpm: NDArray, epsilon: float = 0.01) -> ResultadoEspectral:
    """Realiza el analisis espectral completo de la TPM.

    Args:
        tpm:     Matriz de probabilidad de transicion (2^n x n).
        epsilon: Precision para la cota del tiempo de mezcla (default 0.01 = 1%).

    Returns:
        ResultadoEspectral con eigenvalores, distribucion estacionaria,
        brecha espectral, tiempo de mezcla y entropia de la distribucion limit.
    """
    n = tpm.shape[1]
    num_estados = tpm.shape[0]

    P = _construir_matriz_transicion(tpm)

    # La distribucion estacionaria satisface π P = π (eigenvector izquierdo de P,
    # que es eigenvector derecho de P^T con eigenvalor 1).
    eigenvalores, eigenvectores = np.linalg.eig(P.T)

    # Ordenar por magnitud descendente: el dominante (≈1) queda primero.
    idx_ord = np.argsort(-np.abs(eigenvalores.real))
    eigenvalores_ord = eigenvalores[idx_ord].real
    eigenvectores_ord = eigenvectores[:, idx_ord].real

    # Distribucion estacionaria: eigenvector del eigenvalor dominante normalizado.
    vec_dom = eigenvectores_ord[:, 0]
    dist_estacionaria = np.abs(vec_dom)
    total = dist_estacionaria.sum()
    if total > 1e-12:
        dist_estacionaria /= total
    else:
        dist_estacionaria = np.ones(num_estados) / num_estados

    # Brecha espectral: gap = 1 - |λ₂|. Cuanto mayor, mas rapida la convergencia.
    lambda_2 = float(np.abs(eigenvalores_ord[1])) if len(eigenvalores_ord) > 1 else 0.0
    brecha = max(0.0, 1.0 - lambda_2)

    # Cota superior del tiempo de mezcla: t_mix(ε) <= log(1/ε) / brecha_espectral
    if brecha > 1e-10:
        t_mix = float(np.log(1.0 / max(epsilon, 1e-12)) / brecha)
    else:
        t_mix = float("inf")

    # Entropia de Shannon de la distribucion estacionaria.
    pi_safe = np.clip(dist_estacionaria, 1e-300, None)
    h_estacionaria = float(-np.sum(dist_estacionaria * np.log2(pi_safe)))

    es_ergodica = bool(dist_estacionaria.min() > 1e-9)

    return ResultadoEspectral(
        eigenvalores=eigenvalores_ord[:min(num_estados, 8)],
        distribucion_estacionaria=dist_estacionaria.astype(np.float32),
        brecha_espectral=float(brecha),
        tiempo_mezcla_cota=float(t_mix),
        radio_espectral=float(lambda_2),
        es_ergodica=es_ergodica,
        entropia_estacionaria=float(h_estacionaria),
        n_nodos=n,
        n_estados=num_estados,
    )


def comparar_espectros(tpm_a: NDArray, tpm_b: NDArray) -> dict[str, float]:
    """Compara dos TPMs por sus caracteristicas espectrales.

    Util para comparar el sistema original con el sistema particionado:
    una particion que preserva el espectro es menos destructiva.
    """
    res_a = analizar_tpm(tpm_a)
    res_b = analizar_tpm(tpm_b)
    return {
        "diff_brecha_espectral": abs(res_a.brecha_espectral - res_b.brecha_espectral),
        "diff_tiempo_mezcla": abs(res_a.tiempo_mezcla_cota - res_b.tiempo_mezcla_cota),
        "diff_entropia_estacionaria": abs(res_a.entropia_estacionaria - res_b.entropia_estacionaria),
        "diff_radio_espectral": abs(res_a.radio_espectral - res_b.radio_espectral),
    }


def potencia_iterada(
    P: NDArray,
    estado_inicial: int,
    pasos: int,
) -> NDArray:
    """Simula la evolucion de una distribucion inicial durante 'pasos' pasos.

    Permite verificar empiricamente la convergencia a la distribucion estacionaria
    y contrastar con la cota teorica del tiempo de mezcla.
    """
    num_estados = P.shape[0]
    if estado_inicial < 0 or estado_inicial >= num_estados:
        raise ValueError(f"Estado inicial fuera de rango: {estado_inicial}")

    dist = np.zeros(num_estados, dtype=np.float64)
    dist[estado_inicial] = 1.0

    for _ in range(pasos):
        dist = dist @ P

    return dist.astype(np.float32)


def matriz_transicion_desde_tpm(tpm: NDArray) -> NDArray:
    """Expone la matriz de transicion completa para uso externo."""
    return _construir_matriz_transicion(tpm).astype(np.float32)

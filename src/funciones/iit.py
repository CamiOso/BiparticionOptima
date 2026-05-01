from itertools import product
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from src.modelos.base.aplicacion import aplicacion
from src.modelos.enumeraciones.distancia import MetricDistance
from src.modelos.enumeraciones.notacion import Notation
from src.modelos.enumeraciones.emd_temporal import TimeEMD


def _etiqueta_excel(numero: int) -> str:
    if numero <= 0:
        return ""
    return _etiqueta_excel((numero - 1) // 26) + chr((numero - 1) % 26 + ord("A"))


ABECEDARIO = tuple(_etiqueta_excel(indice) for indice in range(1, 41))


def literales(indices_restantes: NDArray[np.int8], minuscula: bool = False) -> str:
    """Convierte indices de nodos a una etiqueta literal amigable."""
    if indices_restantes.size == 0:
        return "vacio"
    letras = []
    for indice in indices_restantes.tolist():
        letra = ABECEDARIO[int(indice)] if int(indice) < len(ABECEDARIO) else f"N{indice}"
        letras.append(letra.lower() if minuscula else letra)
    return "".join(letras)


def emd_efecto(u: NDArray[np.float32], v: NDArray[np.float32]) -> float:
    """Distancia EMD simplificada para el modo efecto."""
    return float(np.sum(np.abs(u - v)))


def emd_causal(u: NDArray[np.float32], v: NDArray[np.float32]) -> float:
    """Aproximacion causal base mientras no se use una distancia mas rica."""
    return float(np.sum(np.abs(u - v)))


def jensen_shannon(u: NDArray[np.float32], v: NDArray[np.float32]) -> float:
    """Distancia Jensen-Shannon entre dos distribuciones de probabilidad.

    Raiz cuadrada de la divergencia JS, lo que la hace una metrica valida.
    Penaliza diferencias de forma no lineal: es mas sensible a divergencias
    extremas que el EMD, util para detectar particiones que destruyen
    estructuras de dependencia asimetrica.
    """
    u_safe = np.clip(u.astype(np.float64), 1e-12, None)
    v_safe = np.clip(v.astype(np.float64), 1e-12, None)
    m = 0.5 * (u_safe + v_safe)
    divergencia = 0.5 * (
        np.sum(u_safe * np.log(u_safe / m)) + np.sum(v_safe * np.log(v_safe / m))
    )
    return float(np.sqrt(max(0.0, divergencia)))


def wasserstein_sinkhorn(
    u: NDArray[np.float32],
    v: NDArray[np.float32],
    reg: float = 0.05,
    max_iter: int = 200,
    tol: float = 1e-9,
) -> float:
    """Distancia de Wasserstein W1 mediante el algoritmo de Sinkhorn-Knopp.

    Resuelve el problema de transporte optimo regularizado:
        W_reg(u,v) = min_{T >= 0, T1=u, T^T 1=v} <C, T> + reg * KL(T || u⊗v)

    La matriz de costo C[i,j] = |i-j|/n captura el costo de "mover" activacion
    del nodo i al nodo j. Con reg pequeño, la solucion se acerca a la Wasserstein
    exacta; con reg grande, el transporte se suaviza hacia el producto exterior.

    A diferencia del EMD (que es puntual, L1), esta metrica considera la estructura
    geometrica del espacio de nodos: mover activacion entre nodos adyacentes es mas
    barato que entre nodos lejanos.
    """
    n = len(u)
    if n == 0:
        return 0.0

    u64 = np.clip(u.astype(np.float64), 1e-12, None)
    v64 = np.clip(v.astype(np.float64), 1e-12, None)
    u64 /= u64.sum()
    v64 /= v64.sum()

    # Costo normalizado por numero de nodos para que sea invariante a la escala.
    indices = np.arange(n, dtype=np.float64)
    C = np.abs(indices[:, None] - indices[None, :]) / max(1, n - 1)

    K = np.exp(-C / reg)
    b = np.ones(n, dtype=np.float64)
    a = np.ones(n, dtype=np.float64)

    for _ in range(max_iter):
        a_prev = a.copy()
        a = u64 / (K @ b + 1e-300)
        b = v64 / (K.T @ a + 1e-300)
        if np.max(np.abs(a - a_prev)) < tol:
            break

    T = np.diag(a) @ K @ np.diag(b)
    return float(np.sum(T * C))


def fisher_rao(u: NDArray[np.float32], v: NDArray[np.float32]) -> float:
    """Distancia geodesica de Fisher-Rao sobre la variedad estadistica.

    El espacio de distribuciones de probabilidad es una variedad Riemanniana
    con metrica de Fisher: g_ij = E[d/dθ_i log p · d/dθ_j log p]. La distancia
    geodesica entre dos distribuciones es el angulo de Bhattacharyya:

        d_FR(u, v) = 2 · arccos(Σᵢ √(uᵢ · vᵢ))

    Propiedades: es una metrica, varia en [0, π], es 0 solo cuando u = v, y
    es π cuando u y v tienen soportes disjuntos. Es mas sensible a diferencias
    en las colas que el EMD y es intrinseca a la geometria del simplex.
    """
    u64 = np.clip(u.astype(np.float64), 0.0, None)
    v64 = np.clip(v.astype(np.float64), 0.0, None)

    su = u64.sum()
    sv = v64.sum()
    if su <= 0 or sv <= 0:
        return 0.0

    u64 /= su
    v64 /= sv

    coeficiente_bhattacharyya = float(np.sum(np.sqrt(u64 * v64)))
    coeficiente_bhattacharyya = np.clip(coeficiente_bhattacharyya, 0.0, 1.0)
    return float(2.0 * np.arccos(coeficiente_bhattacharyya))


def kl_divergencia(u: NDArray[np.float32], v: NDArray[np.float32]) -> float:
    """Divergencia KL simetrica (u||v + v||u) / 2.

    Mide cuanta informacion se pierde al aproximar u con v y viceversa.
    A diferencia del EMD, es infinita si v tiene ceros donde u no los tiene,
    lo que la hace mas estricta para particiones que colapsan distribuciones.
    """
    u_safe = np.clip(u.astype(np.float64), 1e-12, None)
    v_safe = np.clip(v.astype(np.float64), 1e-12, None)
    kl_uv = float(np.sum(u_safe * np.log(u_safe / v_safe)))
    kl_vu = float(np.sum(v_safe * np.log(v_safe / u_safe)))
    return (kl_uv + kl_vu) / 2.0


def contar_bits(numero: int) -> int:
    return bin(numero).count("1")


def distancia_hamming(a: int, b: int) -> int:
    return contar_bits(a ^ b)


def seleccionar_distancia() -> Callable[[int, int], float]:
    distancias = {
        MetricDistance.HAMMING.value: distancia_hamming,
        MetricDistance.MANHATTAN.value: lambda a, b: abs(a - b),
        MetricDistance.EUCLIDIANA.value: lambda a, b: float(abs(a - b)),
    }
    if aplicacion.distancia_metrica not in distancias:
        raise ValueError(
            f"Distancia no soportada en esta etapa: {aplicacion.distancia_metrica}"
        )
    return distancias[aplicacion.distancia_metrica]


def big_endian(n: int) -> NDArray[np.uint32]:
    return np.array(range(1 << n), dtype=np.uint32)


def lil_endian(n: int) -> NDArray[np.uint32]:
    if n <= 0:
        return np.array([0], dtype=np.uint32)
    indices = np.arange(1 << n, dtype=np.uint32)
    salida = np.zeros_like(indices)
    for bit in range(n):
        salida |= ((indices >> bit) & 1) << (n - bit - 1)
    return salida


def reindexar(n: int) -> NDArray[np.uint32]:
    notaciones = {
        Notation.BIG_ENDIAN.value: big_endian(n),
        Notation.LIL_ENDIAN.value: lil_endian(n),
    }
    if aplicacion.notacion_indexado not in notaciones:
        raise ValueError(
            f"Notacion no soportada en esta etapa: {aplicacion.notacion_indexado}"
        )
    return notaciones[aplicacion.notacion_indexado]


def seleccionar_estado(subestado: NDArray[np.int8]) -> NDArray[np.int8]:
    notaciones = {
        Notation.BIG_ENDIAN.value: subestado,
        Notation.LIL_ENDIAN.value: subestado[::-1],
    }
    if aplicacion.notacion_indexado not in notaciones:
        raise ValueError(
            f"Notacion no soportada en esta etapa: {aplicacion.notacion_indexado}"
        )
    return notaciones[aplicacion.notacion_indexado]


def dec2bin(decimal: int, ancho: int) -> str:
    return format(decimal, f"0{ancho}b")


def estados_binarios(n: int) -> list[str]:
    return [dec2bin(indice, n) for indice in range(1 << n)][1:]


def combinaciones_restringidas(binario: str) -> tuple[list[str], list[str]]:
    cantidad_unos = binario.count("1")
    posiciones = [indice for indice, bit in enumerate(binario) if bit == "1"]
    base = list(product(["0", "1"], repeat=cantidad_unos))
    combinaciones = []
    for combinacion in base:
        bits = ["0"] * len(binario)
        for posicion, bit in zip(posiciones, combinacion):
            bits[posicion] = bit
        combinaciones.append("".join(bits))
    return combinaciones, combinaciones.copy()


def generar_combinaciones(a: str) -> list[tuple[str, str, str]]:
    b, c = combinaciones_restringidas(a)
    return list(product([a], b, c))[1:]


def seleccionar_emd() -> Callable[[NDArray[np.float32], NDArray[np.float32]], float]:
    """Selecciona la funcion de distancia segun la configuracion global."""
    emd_metricas = {
        TimeEMD.EMD_EFECTO.value: emd_efecto,
        TimeEMD.EMD_CAUSA.value: emd_causal,
        TimeEMD.EMD_INTEGRADA.value: emd_efecto,
        TimeEMD.JENSEN_SHANNON.value: jensen_shannon,
        TimeEMD.KL_DIVERGENCIA.value: kl_divergencia,
        TimeEMD.WASSERSTEIN.value: wasserstein_sinkhorn,
        TimeEMD.FISHER_RAO.value: fisher_rao,
    }

    if aplicacion.tiempo_emd not in emd_metricas:
        raise ValueError(
            f"Tiempo EMD no soportado en esta etapa: {aplicacion.tiempo_emd}"
        )

    return emd_metricas[aplicacion.tiempo_emd]

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
    """Cuenta la cantidad de bits en '1' de la representación binaria de número.

    Parámetros
    ----------
    numero : int
        Entero no negativo.

    Retorna
    -------
    int
        Número de bits activos (peso de Hamming).
    """
    return bin(numero).count("1")


def distancia_hamming(a: int, b: int) -> int:
    """Distancia de Hamming entre dos enteros (número de bits que difieren).

    Parámetros
    ----------
    a, b : int
        Enteros no negativos que representan estados binarios.

    Retorna
    -------
    int
        Número de posiciones de bit donde ``a`` y ``b`` difieren.
    """
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
    """Genera la permutación identidad de estados para notación big-endian.

    En big-endian el bit de mayor peso es el más a la izquierda, por lo que
    el índice numérico del estado ya coincide con su posición en la TPM.

    Parámetros
    ----------
    n : int
        Número de nodos (la TPM tiene 2^n filas).

    Retorna
    -------
    NDArray[np.uint32]
        Arreglo [0, 1, 2, ..., 2^n - 1] que mapea cada estado a sí mismo.
    """
    return np.array(range(1 << n), dtype=np.uint32)


def lil_endian(n: int) -> NDArray[np.uint32]:
    """Genera la permutación de estados para notación little-endian.

    En little-endian el bit de menor peso es el más a la izquierda. Esto
    invierte el orden de los bits de cada índice: el estado binario "01"
    (índice 1 en big-endian) pasa a ser el índice 2, y así sucesivamente.

    Parámetros
    ----------
    n : int
        Número de nodos.

    Retorna
    -------
    NDArray[np.uint32]
        Arreglo de tamaño 2^n donde la posición i contiene el índice
        little-endian correspondiente al estado i en big-endian.

    Ejemplo
    -------
    >>> lil_endian(2).tolist()
    [0, 2, 1, 3]
    """
    if n <= 0:
        return np.array([0], dtype=np.uint32)
    indices = np.arange(1 << n, dtype=np.uint32)
    salida = np.zeros_like(indices)
    for bit in range(n):
        salida |= ((indices >> bit) & 1) << (n - bit - 1)
    return salida


def reindexar(n: int) -> NDArray[np.uint32]:
    """Devuelve la permutación de reindexado configurada en la aplicación.

    Selecciona entre big-endian (identidad) y little-endian (bits invertidos)
    según ``aplicacion.notacion_indexado``. Se usa para reordenar las filas
    de la TPM antes de calcular distribuciones marginales.

    Parámetros
    ----------
    n : int
        Número de nodos del subsistema.

    Retorna
    -------
    NDArray[np.uint32]
        Permutación de tamaño 2^n para reindexar las distribuciones.

    Raises
    ------
    ValueError
        Si la notación configurada no está entre las soportadas.
    """
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
    """Ordena un subestado binario según la notación de indexado configurada.

    En big-endian devuelve el subestado sin cambios. En little-endian lo
    invierte, ya que el primer elemento pasa a ser el bit de menor peso.

    Parámetros
    ----------
    subestado : NDArray[np.int8]
        Vector binario con los valores de los nodos del subestado.

    Retorna
    -------
    NDArray[np.int8]
        Subestado reordenado según la notación activa.

    Raises
    ------
    ValueError
        Si la notación configurada no está entre las soportadas.
    """
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
    """Convierte un entero a su representación binaria con ancho fijo.

    Parámetros
    ----------
    decimal : int
        Valor entero no negativo a convertir.
    ancho : int
        Número mínimo de dígitos binarios; se rellena con ceros a la izquierda.

    Retorna
    -------
    str
        Cadena binaria de longitud ``ancho``.

    Ejemplo
    -------
    >>> dec2bin(3, 4)
    '0011'
    """
    return format(decimal, f"0{ancho}b")


def estados_binarios(n: int) -> list[str]:
    """Genera todas las cadenas binarias de longitud n excluyendo el estado cero.

    El estado todo-ceros se excluye porque en el contexto IIT representa el
    estado de referencia (fondo), no un estado de transición válido.

    Parámetros
    ----------
    n : int
        Número de bits (nodos).

    Retorna
    -------
    list[str]
        Lista de 2^n - 1 cadenas binarias de longitud n, desde '0...01'
        hasta '1...1'.
    """
    return [dec2bin(indice, n) for indice in range(1 << n)][1:]


def combinaciones_restringidas(binario: str) -> tuple[list[str], list[str]]:
    """Genera combinaciones de activación restringidas a las posiciones de '1'.

    Dado un patrón binario, produce todos los posibles valores que pueden
    tomar sus posiciones activas (bits '1'), manteniendo las posiciones '0'
    fijas. Útil para generar los estados compatibles con un subconjunto de
    nodos activos.

    Parámetros
    ----------
    binario : str
        Cadena binaria que define qué posiciones son variables (las que
        contienen '1').

    Retorna
    -------
    tuple[list[str], list[str]]
        Par de listas idénticas con todas las cadenas resultantes de asignar
        '0' o '1' a cada posición activa. Se devuelven dos copias para
        permitir su uso como componentes de producto cartesiano independientes.
    """
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
    """Genera tripletas (patrón, estado_efecto, estado_causa) para cálculos IIT.

    Fija el patrón ``a`` y genera todas las combinaciones de estados de
    efecto y causa restringidos a las posiciones activas de ``a``, excluyendo
    la primera tripleta que corresponde al estado base (estado nulo, excluido
    para evitar singularidades en el cálculo de phi).

    Parámetros
    ----------
    a : str
        Cadena binaria que define el patrón de nodos activos.

    Retorna
    -------
    list[tuple[str, str, str]]
        Lista de tripletas ``(a, estado_efecto, estado_causa)`` válidas.
    """
    b, c = combinaciones_restringidas(a)
    return list(product([a], b, c))[1:]


def seleccionar_emd(config=None) -> Callable[[NDArray[np.float32], NDArray[np.float32]], float]:
    """Selecciona la funcion de distancia EMD.

    Parámetros
    ----------
    config : AppConfig | None
        Configuración inyectada. Si es ``None`` usa el singleton global
        (comportamiento retrocompatible).
    """
    emd_metricas = {
        TimeEMD.EMD_EFECTO.value: emd_efecto,
        TimeEMD.EMD_CAUSA.value: emd_causal,
        TimeEMD.EMD_INTEGRADA.value: emd_efecto,
        TimeEMD.JENSEN_SHANNON.value: jensen_shannon,
        TimeEMD.KL_DIVERGENCIA.value: kl_divergencia,
        TimeEMD.WASSERSTEIN.value: wasserstein_sinkhorn,
        TimeEMD.FISHER_RAO.value: fisher_rao,
    }

    tiempo = config.tiempo_emd if config is not None else aplicacion.tiempo_emd

    if tiempo not in emd_metricas:
        raise ValueError(f"Tiempo EMD no soportado en esta etapa: {tiempo}")

    return emd_metricas[tiempo]

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from src.funciones.particiones import k_particiones_asignacion


@dataclass(frozen=True)
class ResultadoKParticion:
    """Resultado de una búsqueda de k-partición.

    Atributos
    ---------
    perdida : float
        Valor de la función objetivo (distancia entre distribución del sistema
        completo y la del sistema particionado). Siempre >= 0; 0 significa
        partición sin pérdida de información.
    distribucion : np.ndarray
        Distribución de probabilidad del sistema después de aplicar la
        partición representada por ``asignacion``.
    asignacion : tuple[int, ...]
        Etiqueta de grupo para cada elemento (nodo o vértice temporal).
        Siempre en forma canónica: el grupo 0 aparece antes que el 1,
        el 1 antes que el 2, etc.
    """

    perdida: float
    distribucion: np.ndarray
    asignacion: tuple[int, ...]


class BuscadorKParticion(ABC):
    """Algoritmo de búsqueda de k-partición óptima (patrón Template Method).

    Define la estructura del algoritmo: búsqueda exacta para sistemas
    pequeños (n <= umbral_exacto) y búsqueda local con reinicios para
    sistemas grandes. Las subclases concretas implementan
    ``evaluar_asignacion()`` según cómo construyen la partición del sistema.

    Parámetros
    ----------
    umbral_exacto : int
        Número máximo de elementos para los que se realiza enumeración
        completa (garantía de óptimo global).
    max_iter_refinamiento : int
        Iteraciones máximas de la búsqueda local voraz (hill-climbing).
    max_restarts : int
        Número de reinicios aleatorios en ``_buscar_local``.
    """

    def __init__(
        self,
        umbral_exacto: int = 8,
        max_iter_refinamiento: int = 24,
        max_restarts: int = 16,
    ) -> None:
        self.umbral_exacto = umbral_exacto
        self.max_iter_refinamiento = max_iter_refinamiento
        self.max_restarts = max_restarts

    @abstractmethod
    def evaluar_asignacion(self, asignacion: tuple[int, ...]) -> tuple[float, np.ndarray]:
        """Calcula la pérdida y la distribución para una asignación de grupos.

        Parámetros
        ----------
        asignacion : tuple[int, ...]
            Etiqueta de grupo para cada elemento, en forma canónica.
            Longitud igual a ``total_elementos()``.

        Retorna
        -------
        perdida : float
            Distancia entre la distribución del sistema completo y la del
            sistema con la partición aplicada.
        distribucion : np.ndarray
            Distribución de probabilidad del sistema particionado.

        Precondición
        ------------
        ``asignacion`` debe contener al menos 2 grupos distintos.
        """

    @abstractmethod
    def total_elementos(self) -> int:
        """Número de elementos (nodos o vértices) que se van a particionar.

        Retorna
        -------
        int
            Entero positivo igual al largo esperado de cada ``asignacion``.
        """

    def canonicalizar(self, asignacion: tuple[int, ...]) -> tuple[int, ...]:
        """Normaliza etiquetas de grupo al orden de primera aparición.

        El grupo que aparece primero en ``asignacion`` recibe la etiqueta 0,
        el siguiente grupo nuevo recibe la etiqueta 1, etc. Esto elimina
        duplicados semánticos: (0,1,0) y (1,0,1) representan la misma
        partición estructural y ambas se mapean a (0,1,0).

        Parámetros
        ----------
        asignacion : tuple[int, ...]
            Asignación de grupos con etiquetas arbitrarias no-negativas.

        Retorna
        -------
        tuple[int, ...]
            Asignación equivalente en forma canónica.
        """
        mapa: dict[int, int] = {}
        siguiente = 0
        canon = []
        for grupo in asignacion:
            if grupo not in mapa:
                mapa[grupo] = siguiente
                siguiente += 1
            canon.append(mapa[grupo])
        return tuple(canon)

    def vecinos(self, asignacion: tuple[int, ...], k: int) -> list[tuple[int, ...]]:
        """Genera vecinos Hamming-1 canónicos válidos de una asignación.

        Un vecino se obtiene moviendo un único elemento a un grupo diferente.
        Se descartan los vecinos que colapsan a un solo grupo (partición
        trivial) y los duplicados canónicos.

        Parámetros
        ----------
        asignacion : tuple[int, ...]
            Asignación actual en forma canónica.
        k : int
            Número máximo de grupos permitidos (etiquetas 0..k-1).

        Retorna
        -------
        list[tuple[int, ...]]
            Lista de asignaciones canónicas distintas con al menos 2 grupos.

        Precondición
        ------------
        ``asignacion`` debe estar en forma canónica y tener al menos 2 grupos.
        """
        resultado: list[tuple[int, ...]] = []
        vistos: set[tuple[int, ...]] = set()
        for idx in range(len(asignacion)):
            for nuevo_grupo in range(k):
                if nuevo_grupo == asignacion[idx]:
                    continue
                nueva = list(asignacion)
                nueva[idx] = nuevo_grupo
                canon = self.canonicalizar(tuple(nueva))
                if len(set(canon)) < 2 or canon in vistos:
                    continue
                vistos.add(canon)
                resultado.append(canon)
        return resultado

    def refinar_local(self, inicio: ResultadoKParticion, k: int) -> ResultadoKParticion:
        """Búsqueda local voraz (hill-climbing) desde una solución inicial.

        En cada iteración evalúa todos los vecinos Hamming-1 de la solución
        actual y se mueve al mejor si mejora estrictamente la pérdida. Termina
        cuando ningún vecino mejora o se alcanza ``max_iter_refinamiento``.

        Parámetros
        ----------
        inicio : ResultadoKParticion
            Punto de partida para la búsqueda local.
        k : int
            Número máximo de grupos permitidos.

        Retorna
        -------
        ResultadoKParticion
            Mejor solución encontrada (óptimo local).
        """
        actual = inicio
        mejor = inicio
        if mejor.perdida <= 1e-12:
            return mejor
        for _ in range(self.max_iter_refinamiento):
            candidatos = self.vecinos(actual.asignacion, k)
            if not candidatos:
                break
            mejor_vecino = actual
            for asig in candidatos:
                perdida, dist = self.evaluar_asignacion(asig)
                if perdida < mejor_vecino.perdida:
                    mejor_vecino = ResultadoKParticion(
                        perdida=perdida,
                        distribucion=dist,
                        asignacion=asig,
                    )
            if mejor_vecino.perdida + 1e-12 >= actual.perdida:
                break
            actual = mejor_vecino
            if actual.perdida < mejor.perdida:
                mejor = actual
        return mejor

    def buscar(self, k: int, semilla: int = 42) -> ResultadoKParticion:
        """Punto de entrada principal: busca la k-partición de mínima pérdida.

        Elige automáticamente entre búsqueda exacta (n <= umbral_exacto)
        y búsqueda local con reinicios (n > umbral_exacto).

        Parámetros
        ----------
        k : int
            Número máximo de grupos. Se recorta a min(k, n) si k > n.
        semilla : int
            Semilla para el generador aleatorio (reproducibilidad).

        Retorna
        -------
        ResultadoKParticion
            Mejor partición encontrada. Óptimo global garantizado solo
            cuando n <= umbral_exacto.

        Precondición
        ------------
        ``total_elementos()`` y ``evaluar_asignacion()`` deben estar
        implementados antes de llamar a este método.
        """
        n = self.total_elementos()
        k_eff = min(k, n)
        if n <= self.umbral_exacto:
            return self._buscar_exacto(k_eff)
        return self._buscar_local(k_eff, semilla)

    def _buscar_exacto(self, k: int) -> ResultadoKParticion:
        """Enumeración exhaustiva de todas las asignaciones canónicas válidas.

        Itera sobre todas las k-particiones canónicas con entre 2 y k grupos
        generadas por ``k_particiones_asignacion``. Garantiza el óptimo global.

        Parámetros
        ----------
        k : int
            Número máximo de grupos (ya ajustado a min(k, n)).

        Retorna
        -------
        ResultadoKParticion
            Partición con la menor pérdida entre todas las asignaciones
            canónicas evaluadas.
        """
        n = self.total_elementos()
        asig_base = tuple([0] * (n - 1) + [1])
        perdida_base, dist_base = self.evaluar_asignacion(asig_base)
        mejor = ResultadoKParticion(
            perdida=perdida_base,
            distribucion=dist_base,
            asignacion=asig_base,
        )
        for asig in k_particiones_asignacion(n, k):
            perdida, dist = self.evaluar_asignacion(asig)
            if perdida < mejor.perdida:
                mejor = ResultadoKParticion(perdida=perdida, distribucion=dist, asignacion=asig)
        return mejor

    def _buscar_local(self, k: int, semilla: int) -> ResultadoKParticion:
        """Búsqueda local con múltiples reinicios aleatorios.

        Genera un punto de inicio aleatorio que garantiza la presencia de al
        menos un elemento de cada grupo (list(range(k)) como prefijo).
        Refina cada inicio con ``refinar_local`` y conserva el mejor resultado.

        Parámetros
        ----------
        k : int
            Número máximo de grupos.
        semilla : int
            Semilla base del RNG; cada reinicio desplaza la secuencia.

        Retorna
        -------
        ResultadoKParticion
            Mejor solución local encontrada tras ``max_restarts + 1`` búsquedas.
        """
        n = self.total_elementos()
        rng = np.random.default_rng(semilla)

        # Primer inicio: prefijo [0..k-1] garantiza al menos k grupos distintos
        asig_inicial = self.canonicalizar(
            tuple(list(range(k)) + [int(rng.integers(0, k)) for _ in range(max(0, n - k))])
        )
        perdida_ini, dist_ini = self.evaluar_asignacion(asig_inicial)
        mejor = self.refinar_local(
            ResultadoKParticion(perdida=perdida_ini, distribucion=dist_ini, asignacion=asig_inicial),
            k,
        )

        for _ in range(self.max_restarts):
            perm = list(range(k)) + [int(rng.integers(0, k)) for _ in range(max(0, n - k))]
            rng.shuffle(perm)
            asig = self.canonicalizar(tuple(int(v) for v in perm))
            perdida, dist = self.evaluar_asignacion(asig)
            semilla_local = ResultadoKParticion(perdida=perdida, distribucion=dist, asignacion=asig)
            refinado = self.refinar_local(semilla_local, k)
            if refinado.perdida < mejor.perdida:
                mejor = refinado

        return mejor


class BuscadorKRecocido(BuscadorKParticion):
    """Búsqueda de k-partición usando recocido simulado (Simulated Annealing).

    A diferencia de la búsqueda local codiciosa, acepta soluciones peores con
    probabilidad exp(-delta/T), donde T disminuye geométricamente. Esto permite
    escapar de mínimos locales que atrapan a los buscadores voraces.

    Parámetros
    ----------
    temp_inicial : float
        Temperatura de arranque; controla la exploración inicial.
        Valores más altos → mayor aceptación de movimientos malos.
    temp_final : float
        Criterio de parada por temperatura. La cadena termina cuando
        ``temp < temp_final``.
    factor_enfriamiento : float
        Multiplicador por ciclo, en (0, 1). Valores cercanos a 1 enfrían
        lentamente (mejor calidad, más tiempo).
    pasos_por_temp : int
        Número de propuestas de movimiento evaluadas por nivel de temperatura.
    n_cadenas : int
        Número de corridas SA independientes en ``_multi_recocido``.
    """

    def __init__(
        self,
        temp_inicial: float = 1.0,
        temp_final: float = 0.001,
        factor_enfriamiento: float = 0.92,
        pasos_por_temp: int = 30,
        n_cadenas: int = 3,
    ) -> None:
        super().__init__(umbral_exacto=6, max_iter_refinamiento=0, max_restarts=0)
        self.temp_inicial = temp_inicial
        self.temp_final = temp_final
        self.factor_enfriamiento = factor_enfriamiento
        self.pasos_por_temp = pasos_por_temp
        self.n_cadenas = n_cadenas

    def buscar(self, k: int, semilla: int = 42) -> ResultadoKParticion:
        """Punto de entrada: enumeración exacta para n pequeño, SA para n grande.

        Parámetros
        ----------
        k : int
            Número máximo de grupos.
        semilla : int
            Semilla para reproducibilidad.

        Retorna
        -------
        ResultadoKParticion
            Mejor solución encontrada.
        """
        n = self.total_elementos()
        k_eff = min(k, n)
        if n <= self.umbral_exacto:
            return self._buscar_exacto(k_eff)
        return self._multi_recocido(k_eff, semilla)

    def _recocido(self, k: int, semilla: int) -> ResultadoKParticion:
        """Ejecuta una sola cadena de recocido simulado desde un inicio aleatorio.

        En cada paso propone con probabilidad 0.5 un swap de dos elementos
        (permite alcanzar vecinos a distancia 2 en un solo movimiento) o el
        cambio de grupo de un único elemento. Acepta el movimiento si mejora
        la pérdida o con probabilidad de Boltzmann exp(-Δ/T).

        Parámetros
        ----------
        k : int
            Número máximo de grupos.
        semilla : int
            Semilla del RNG para esta cadena.

        Retorna
        -------
        ResultadoKParticion
            Mejor resultado observado a lo largo de toda la cadena SA.
        """
        n = self.total_elementos()
        rng = np.random.default_rng(semilla)

        asig_actual = self.canonicalizar(
            tuple(list(range(k)) + [int(rng.integers(0, k)) for _ in range(max(0, n - k))])
        )
        perm = list(asig_actual)
        rng.shuffle(perm)
        asig_actual = self.canonicalizar(tuple(perm))

        perdida_actual, dist_actual = self.evaluar_asignacion(asig_actual)
        mejor = ResultadoKParticion(
            perdida=perdida_actual,
            distribucion=dist_actual,
            asignacion=asig_actual,
        )

        if mejor.perdida <= 1e-12:
            return mejor

        temp = self.temp_inicial
        niveles_sin_mejora = 0
        mejor_previa = mejor.perdida
        while temp > self.temp_final:
            for _ in range(self.pasos_por_temp):
                nueva = list(asig_actual)
                if n >= 2 and rng.random() < 0.5:
                    # Swap: intercambiar grupos de dos nodos distintos.
                    # Permite llegar en un paso a vecinos que el movimiento
                    # individual necesita dos pasos aceptados para alcanzar.
                    i = int(rng.integers(0, n))
                    j = int(rng.integers(0, n - 1))
                    if j >= i:
                        j += 1
                    nueva[i], nueva[j] = nueva[j], nueva[i]
                else:
                    idx = int(rng.integers(0, n))
                    nuevo_grupo = int(rng.integers(0, k))
                    nueva[idx] = nuevo_grupo

                asig_vecina = self.canonicalizar(tuple(nueva))

                # Descartar movimientos que colapsan a un solo grupo
                if len(set(asig_vecina)) < 2:
                    continue

                perdida_vecina, dist_vecina = self.evaluar_asignacion(asig_vecina)
                delta = perdida_vecina - perdida_actual

                # Criterio de Metropolis: aceptar mejoras siempre;
                # aceptar empeoramientos con probabilidad exp(-delta/T)
                if delta < 0 or rng.random() < math.exp(-delta / temp):
                    asig_actual = asig_vecina
                    perdida_actual = perdida_vecina
                    dist_actual = dist_vecina

                    if perdida_actual < mejor.perdida:
                        mejor = ResultadoKParticion(
                            perdida=perdida_actual,
                            distribucion=dist_actual,
                            asignacion=asig_actual,
                        )

            temp *= self.factor_enfriamiento
            if mejor.perdida <= 1e-12:
                break
            if mejor.perdida < mejor_previa:
                mejor_previa = mejor.perdida
                niveles_sin_mejora = 0
            else:
                niveles_sin_mejora += 1
                if niveles_sin_mejora >= 5 and temp < self.temp_inicial * 0.3:
                    break

        return mejor

    def buscar_con_semilla(
        self, k: int, semilla_asig: tuple[int, ...], semilla: int = 42
    ) -> ResultadoKParticion:
        """Combina refinamiento local de la semilla con SA para no quedar atrapado.

        Evalúa la semilla, la refina con búsqueda local codiciosa y compara
        contra una corrida SA independiente. Retorna el mejor de ambos.

        Parámetros
        ----------
        k : int
            Número máximo de grupos.
        semilla_asig : tuple[int, ...]
            Asignación inicial (warm-start) ya en forma canónica.
        semilla : int
            Semilla aleatoria para la corrida SA independiente.

        Retorna
        -------
        ResultadoKParticion
            El mejor resultado entre el refinamiento de la semilla y el SA.
        """
        perdida_s, dist_s = self.evaluar_asignacion(semilla_asig)
        inicio = ResultadoKParticion(perdida=perdida_s, distribucion=dist_s, asignacion=semilla_asig)
        refinado = self.refinar_local(inicio, k)
        resultado_sa = self._multi_recocido(k, semilla)
        return refinado if refinado.perdida <= resultado_sa.perdida else resultado_sa

    def _multi_recocido(self, k: int, semilla: int) -> ResultadoKParticion:
        """Corre ``n_cadenas`` corridas SA secuenciales y retorna la mejor.

        Cada cadena arranca desde un punto aleatorio distinto (semillas
        separadas por 1009 para evitar correlaciones).

        Parámetros
        ----------
        k : int
            Número máximo de grupos.
        semilla : int
            Semilla base; la cadena i usa ``semilla + i * 1009``.

        Retorna
        -------
        ResultadoKParticion
            Mejor resultado entre todas las cadenas SA.
        """
        semillas = [semilla + i * 1009 for i in range(self.n_cadenas)]
        resultados = [self._recocido(k, s) for s in semillas]
        mejor = min(resultados, key=lambda r: r.perdida)
        return mejor


class BuscadorKDP(BuscadorKRecocido):
    """DP de subconjuntos para inicialización + recocido simulado para refinamiento.

    Fase 1 — Programación Dinámica (n <= umbral_dp):
        Precalcula el costo de bipartición de cada subconjunto de elementos y
        aplica DP de subconjuntos O(3^n · k) para encontrar la asignación
        inicial de mínimo costo estimado.

    Fase 2 — Refinamiento SA:
        Refina la semilla DP con recocido simulado (heredado de
        ``BuscadorKRecocido``).

    Si se pasan ``costos_subconjuntos`` precalculados (e.g. costos locales del
    hipercubo geométrico), la Fase 1 no requiere evaluaciones adicionales.

    Para n > umbral_dp cae directamente al recocido puro del padre.

    Parámetros
    ----------
    costos_subconjuntos : np.ndarray | None
        Arreglo de tamaño 2^n con el costo de bipartición de cada máscara de
        bits. Si es ``None`` o tiene tamaño incorrecto se calculan en línea.
    umbral_dp : int
        Número máximo de elementos para aplicar la fase DP (2^n debe ser
        manejable en memoria).
    temp_inicial, temp_final, factor_enfriamiento, pasos_por_temp :
        Parámetros SA heredados de ``BuscadorKRecocido``.
    """

    def __init__(
        self,
        costos_subconjuntos: np.ndarray | None = None,
        umbral_dp: int = 12,
        temp_inicial: float = 1.0,
        temp_final: float = 0.001,
        factor_enfriamiento: float = 0.92,
        pasos_por_temp: int = 30,
    ) -> None:
        super().__init__(
            temp_inicial=temp_inicial,
            temp_final=temp_final,
            factor_enfriamiento=factor_enfriamiento,
            pasos_por_temp=pasos_por_temp,
        )
        self._costos_subconjuntos = costos_subconjuntos
        self._umbral_dp = umbral_dp

    def buscar(self, k: int, semilla: int = 42) -> ResultadoKParticion:
        """Punto de entrada: exacto → DP+SA → SA puro según tamaño del sistema.

        Parámetros
        ----------
        k : int
            Número máximo de grupos.
        semilla : int
            Semilla para el RNG de SA.

        Retorna
        -------
        ResultadoKParticion
            Mejor partición encontrada con la estrategia apropiada al tamaño.
        """
        n = self.total_elementos()
        k_eff = min(k, n)
        if n <= self.umbral_exacto:
            return self._buscar_exacto(k_eff)
        if n <= self._umbral_dp:
            return self._buscar_dp_sa(k_eff, semilla)
        return self._multi_recocido(k_eff, semilla)

    def _obtener_costos_sub(self, n: int) -> np.ndarray:
        """Construye el arreglo de costos de bipartición para cada subconjunto.

        Retorna ``costos_subconjuntos`` si ya están precalculados con el tamaño
        correcto (2^n). De lo contrario evalúa cada máscara de bits como la
        bipartición {elementos en la máscara} | {resto}, almacenando el resultado.

        Parámetros
        ----------
        n : int
            Número de elementos del sistema (igual a ``total_elementos()``).

        Retorna
        -------
        np.ndarray
            Arreglo float64 de tamaño 2^n donde ``costos[mask]`` es la pérdida
            de separar los bits activos de ``mask`` del resto.

        Precondición
        ------------
        n <= umbral_dp (la tabla tiene 2^n entradas, se asume que cabe en RAM).
        """
        total = 1 << n
        if self._costos_subconjuntos is not None and len(self._costos_subconjuntos) == total:
            return self._costos_subconjuntos.astype(np.float64)
        full_mask = total - 1
        costos = np.full(total, np.inf, dtype=np.float64)
        costos[0] = 0.0
        costos[full_mask] = 0.0
        for mask in range(1, full_mask):
            # Cada máscara define la bipartición: grupo 0 = bits activos, grupo 1 = resto
            asig = tuple(0 if (mask >> i) & 1 else 1 for i in range(n))
            perdida, _ = self.evaluar_asignacion(asig)
            costos[mask] = perdida
        return costos

    def _buscar_dp_sa(self, k: int, semilla: int) -> ResultadoKParticion:
        """Inicialización DP de subconjuntos seguida de refinamiento SA.

        Fase DP:
            Construye ``dp_costo[mask][j]`` = mínimo costo estimado de
            j-particionar los elementos representados por ``mask``, usando
            recurrencia de subconjuntos O(3^n · k). Reconstruye la asignación
            de mínimo costo para k grupos.

        Fase SA:
            Refina la semilla DP con una cadena de recocido sobre el espacio
            completo de asignaciones. Luego corre ``n_cadenas - 1`` SA
            adicionales desde puntos aleatorios y retorna el mejor global.

        Parámetros
        ----------
        k : int
            Número máximo de grupos.
        semilla : int
            Semilla para el RNG del SA de refinamiento.

        Retorna
        -------
        ResultadoKParticion
            Mejor partición encontrada entre la semilla DP refinada y los SA
            adicionales.

        Precondición
        ------------
        n <= umbral_dp (la tabla DP de tamaño 2^n × (k+1) debe caber en RAM).
        """
        n = self.total_elementos()
        total = 1 << n
        full_mask = total - 1

        costos_sub = self._obtener_costos_sub(n)
        INF = float("inf")

        # dp_costo[mask][j] = min costo estimado de j-partición de elementos en mask
        dp_costo = np.full((total, k + 1), INF, dtype=np.float64)
        dp_division = np.full((total, k + 1), -1, dtype=np.int32)

        for mask in range(total):
            dp_costo[mask, 0] = 0.0
            if mask > 0:
                dp_costo[mask, 1] = float(costos_sub[mask])
                dp_division[mask, 1] = mask

        # Recorrido por submáscaras: para cada máscara, probar todas las
        # particiones {submask} | {mask ^ submask} con j-1 grupos en el resto
        for mask in range(1, total):
            submask = (mask - 1) & mask
            while submask > 0:
                costo_s = float(costos_sub[submask])
                if np.isfinite(costo_s):
                    resto = mask ^ submask
                    for j in range(2, k + 1):
                        nuevo = dp_costo[resto, j - 1] + costo_s
                        if nuevo < dp_costo[mask, j]:
                            dp_costo[mask, j] = nuevo
                            dp_division[mask, j] = submask
                submask = (submask - 1) & mask

        asig_dp = self._reconstruir_dp(n, k, full_mask, dp_division)
        asig_dp = self.canonicalizar(asig_dp)
        perdida_dp, dist_dp = self.evaluar_asignacion(asig_dp)
        mejor = ResultadoKParticion(perdida=perdida_dp, distribucion=dist_dp, asignacion=asig_dp)

        # Refinar la semilla DP con SA
        rng = np.random.default_rng(semilla)
        asig_actual = asig_dp
        perdida_actual = perdida_dp
        temp = self.temp_inicial
        while temp > self.temp_final:
            for _ in range(self.pasos_por_temp):
                nueva = list(asig_actual)
                if n >= 2 and rng.random() < 0.5:
                    i = int(rng.integers(0, n))
                    j = int(rng.integers(0, n - 1))
                    if j >= i:
                        j += 1
                    nueva[i], nueva[j] = nueva[j], nueva[i]
                else:
                    idx = int(rng.integers(0, n))
                    nuevo_grupo = int(rng.integers(0, k))
                    nueva[idx] = nuevo_grupo
                asig_v = self.canonicalizar(tuple(nueva))
                if len(set(asig_v)) < 2:
                    continue
                perdida_v, dist_v = self.evaluar_asignacion(asig_v)
                delta = perdida_v - perdida_actual
                if delta < 0 or rng.random() < math.exp(-delta / temp):
                    asig_actual = asig_v
                    perdida_actual = perdida_v
                    if perdida_actual < mejor.perdida:
                        mejor = ResultadoKParticion(
                            perdida=perdida_actual,
                            distribucion=dist_v,
                            asignacion=asig_actual,
                        )
            temp *= self.factor_enfriamiento

        for s in [semilla + i * 1009 for i in range(1, self.n_cadenas)]:
            candidato = self._recocido(k, s)
            if candidato.perdida < mejor.perdida:
                mejor = candidato
        return mejor

    def _reconstruir_dp(
        self,
        n: int,
        k: int,
        full_mask: int,
        dp_division: np.ndarray,
    ) -> tuple[int, ...]:
        """Reconstruye la asignación óptima a partir de los punteros de DP.

        Sigue los punteros ``dp_division`` hacia atrás desde la máscara
        completa, asignando a cada grupo los elementos de la submáscara
        elegida en cada nivel de la recurrencia.

        Parámetros
        ----------
        n : int
            Número de elementos.
        k : int
            Número de grupos objetivo.
        full_mask : int
            Máscara con todos los bits activos (2^n - 1).
        dp_division : np.ndarray
            Arreglo int32 de forma (2^n, k+1) con la submáscara óptima
            para cada (mask, j) calculada en ``_buscar_dp_sa``.

        Retorna
        -------
        tuple[int, ...]
            Asignación de grupo para cada elemento (índice 0..n-1).
            No necesariamente en forma canónica; se debe llamar a
            ``canonicalizar`` antes de usar.
        """
        asig = [0] * n
        mask = full_mask
        for grupo in range(k):
            j = k - grupo
            if mask <= 0 or j <= 0:
                break
            submask = int(dp_division[mask, j])
            if submask < 0:
                for i in range(n):
                    if (mask >> i) & 1:
                        asig[i] = grupo
                break
            for i in range(n):
                if (submask >> i) & 1:
                    asig[i] = grupo
            mask ^= submask
        return tuple(asig)

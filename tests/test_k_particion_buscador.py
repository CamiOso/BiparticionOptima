"""Tests unitarios para k_particion_buscador y funciones de partición.

Valida:
1. Propiedades de ``k_particiones_asignacion`` (forma canónica, cardinalidad).
2. Métodos auxiliares de ``BuscadorKParticion``: ``canonicalizar`` y ``vecinos``.
3. Evaluación correcta de k-particiones mediante un buscador mock.
4. Consistencia entre ``buscar(k=2)`` y la enumeración explícita de
   biparticiones (para k=2 el resultado debe coincidir con el mínimo exacto).
5. Consistencia para estrategias reales (QNodos, BranchBound) en sistemas
   pequeños con solución conocida.
"""

import math

import numpy as np
import pytest

from src.funciones.k_particion_buscador import BuscadorKParticion, ResultadoKParticion
from src.funciones.particiones import k_particiones_asignacion


# ---------------------------------------------------------------------------
# Mock concreto de BuscadorKParticion
# ---------------------------------------------------------------------------

class _MockBuscador(BuscadorKParticion):
    """Buscador cuya función de costo es una tabla predefinida.

    La pérdida de una asignación canónica se lee directamente de
    ``tabla_costos``. Si la asignación no está en la tabla, se usa un
    costo por defecto alto para que el buscador la evite.

    Esto permite verificar que el algoritmo de búsqueda encuentra el mínimo
    correcto sin depender de la infraestructura SIA/TPM.
    """

    def __init__(self, n: int, tabla_costos: dict[tuple[int, ...], float]) -> None:
        super().__init__(umbral_exacto=8, max_iter_refinamiento=24, max_restarts=16)
        self._n = n
        self._tabla = tabla_costos

    def total_elementos(self) -> int:
        return self._n

    def evaluar_asignacion(self, asignacion: tuple[int, ...]) -> tuple[float, np.ndarray]:
        perdida = self._tabla.get(asignacion, 999.0)
        dist = np.array([perdida], dtype=np.float32)
        return perdida, dist


# ---------------------------------------------------------------------------
# 1. Propiedades de k_particiones_asignacion
# ---------------------------------------------------------------------------

class TestKParticionesAsignacion:
    """Valida el generador de asignaciones canónicas."""

    def test_genera_asignaciones_con_n2_k2(self) -> None:
        """Para n=2 y k=2 la única bipartición canónica es (0, 1)."""
        resultado = list(k_particiones_asignacion(2, 2))
        assert resultado == [(0, 1)]

    def test_genera_todas_biparticiones_n3_k2(self) -> None:
        """Para n=3 y k=2 hay exactamente 3 biparticiones canónicas."""
        resultado = list(k_particiones_asignacion(3, 2))
        # Todas deben empezar en 0 y tener exactamente 2 grupos distintos
        assert len(resultado) == 3
        for asig in resultado:
            assert asig[0] == 0, "La forma canónica debe empezar en 0"
            assert len(set(asig)) == 2, "Deben existir exactamente 2 grupos"

    def test_no_genera_nada_si_k_menor_2(self) -> None:
        """k < 2 no tiene sentido como partición; no debe generar nada."""
        assert list(k_particiones_asignacion(4, 1)) == []

    def test_no_genera_nada_si_n_menor_2(self) -> None:
        """Un sistema de un solo nodo no puede particionarse."""
        assert list(k_particiones_asignacion(1, 2)) == []

    def test_formas_canonicas_no_tienen_duplicados(self) -> None:
        """Ningún par de asignaciones generadas debe ser semánticamente igual."""
        asignaciones = list(k_particiones_asignacion(4, 3))
        assert len(asignaciones) == len(set(asignaciones))

    def test_primera_ocurrencia_de_cada_grupo_es_creciente(self) -> None:
        """En toda asignación canónica el grupo 0 aparece antes que el 1, etc."""
        for asig in k_particiones_asignacion(5, 3):
            siguiente_esperado = 0
            grupos_vistos: set[int] = set()
            for g in asig:
                if g not in grupos_vistos:
                    assert g == siguiente_esperado, (
                        f"Primera ocurrencia de grupo {g} no es {siguiente_esperado} "
                        f"en {asig}"
                    )
                    siguiente_esperado += 1
                    grupos_vistos.add(g)

    def test_k_eff_recortado_a_n(self) -> None:
        """Si k > n, los grupos se recortan a n (nadie puede tener más grupos que nodos)."""
        # Para n=3, k=10 debe generar lo mismo que k=3
        asig_k10 = set(k_particiones_asignacion(3, 10))
        asig_k3 = set(k_particiones_asignacion(3, 3))
        assert asig_k10 == asig_k3


# ---------------------------------------------------------------------------
# 2. canonicalizar y vecinos
# ---------------------------------------------------------------------------

class TestCanonicalizar:
    """Valida la normalización de etiquetas de grupo."""

    def setup_method(self) -> None:
        self.buscador = _MockBuscador(4, {})

    def test_identidad_para_forma_ya_canonica(self) -> None:
        asig = (0, 1, 0, 1)
        assert self.buscador.canonicalizar(asig) == asig

    def test_renombra_etiquetas_arbitrarias(self) -> None:
        # (2, 5, 2, 5) → (0, 1, 0, 1) porque 2 aparece primero → 0, luego 5 → 1
        assert self.buscador.canonicalizar((2, 5, 2, 5)) == (0, 1, 0, 1)

    def test_tres_grupos_desordenados(self) -> None:
        # (3, 1, 3, 2) → (0, 1, 0, 2)
        assert self.buscador.canonicalizar((3, 1, 3, 2)) == (0, 1, 0, 2)

    def test_un_solo_grupo(self) -> None:
        # Un solo grupo siempre se mapea a todo-ceros
        assert self.buscador.canonicalizar((7, 7, 7)) == (0, 0, 0)

    def test_idempotente(self) -> None:
        """Aplicar canonicalizar dos veces da el mismo resultado."""
        asig = (3, 1, 3, 2, 1)
        canon = self.buscador.canonicalizar(asig)
        assert self.buscador.canonicalizar(canon) == canon


class TestVecinos:
    """Valida la generación de vecinos Hamming-1."""

    def setup_method(self) -> None:
        self.buscador = _MockBuscador(4, {})

    def test_vecinos_no_vacios(self) -> None:
        """Cualquier asignación válida tiene al menos un vecino."""
        vecinos = self.buscador.vecinos((0, 1, 0, 1), k=2)
        assert len(vecinos) > 0

    def test_vecinos_tienen_al_menos_dos_grupos(self) -> None:
        """Ningún vecino debe colapsar a un único grupo."""
        for asig in [(0, 1, 0, 1), (0, 0, 1, 1), (0, 1, 1, 0)]:
            for vecino in self.buscador.vecinos(asig, k=2):
                assert len(set(vecino)) >= 2, f"Vecino inválido: {vecino}"

    def test_vecinos_estan_en_forma_canonica(self) -> None:
        """Todos los vecinos deben estar en forma canónica."""
        for vecino in self.buscador.vecinos((0, 0, 1, 1), k=3):
            assert vecino == self.buscador.canonicalizar(vecino), (
                f"Vecino {vecino} no está en forma canónica"
            )

    def test_sin_duplicados(self) -> None:
        """La lista de vecinos no debe tener duplicados."""
        vecinos = self.buscador.vecinos((0, 1, 0, 1), k=3)
        assert len(vecinos) == len(set(vecinos))

    def test_k2_limita_a_dos_grupos(self) -> None:
        """Con k=2 todos los vecinos deben usar solo grupos 0 y 1."""
        for vecino in self.buscador.vecinos((0, 0, 1, 1), k=2):
            assert all(g in {0, 1} for g in vecino)


# ---------------------------------------------------------------------------
# 3. Evaluación correcta de k-particiones con tabla conocida
# ---------------------------------------------------------------------------

class TestEvaluacionKParticion:
    """Verifica que el buscador encuentra el mínimo de la tabla de costos."""

    def test_buscar_exacto_encuentra_minimo_conocido(self) -> None:
        """Para n=4, k=2 el buscador exacto debe encontrar la asignación de menor costo."""
        # Tabla con un mínimo en (0,1,1,1) → costo 0.1
        tabla = {
            (0, 0, 0, 1): 0.8,
            (0, 0, 1, 0): 0.7,
            (0, 0, 1, 1): 0.5,
            (0, 1, 0, 0): 0.9,
            (0, 1, 0, 1): 0.6,
            (0, 1, 1, 0): 0.4,
            (0, 1, 1, 1): 0.1,  # ← mínimo global para k=2
        }
        buscador = _MockBuscador(4, tabla)
        resultado = buscador.buscar(k=2)

        assert resultado.asignacion == (0, 1, 1, 1)
        assert math.isclose(resultado.perdida, 0.1, abs_tol=1e-9)

    def test_buscar_exacto_k3_encuentra_mejor_que_k2(self) -> None:
        """Para k=3, el buscador puede encontrar un costo menor que para k=2."""
        tabla_k2 = {
            (0, 0, 0, 1): 0.5,
            (0, 0, 1, 0): 0.6,
            (0, 0, 1, 1): 0.4,
            (0, 1, 0, 0): 0.7,
            (0, 1, 0, 1): 0.8,
            (0, 1, 1, 0): 0.3,
            (0, 1, 1, 1): 0.35,
        }
        tabla_k3 = {
            (0, 1, 2, 0): 0.15,  # ← mejor que cualquier bipartición
            (0, 1, 0, 2): 0.20,
            (0, 1, 2, 1): 0.25,
        }
        tabla = {**tabla_k2, **tabla_k3}
        buscador = _MockBuscador(4, tabla)

        resultado_k2 = buscador.buscar(k=2)
        resultado_k3 = buscador.buscar(k=3)

        assert resultado_k2.perdida == pytest.approx(0.3, abs=1e-9)
        assert resultado_k3.perdida <= resultado_k2.perdida + 1e-9, (
            "k=3 debe ser igual o mejor que k=2"
        )

    def test_perdida_no_negativa(self) -> None:
        """La pérdida siempre debe ser >= 0."""
        tabla = {
            (0, 1, 0, 1): 0.0,
            (0, 0, 1, 1): 0.5,
            (0, 1, 1, 0): 0.3,
        }
        buscador = _MockBuscador(4, tabla)
        resultado = buscador.buscar(k=2)
        assert resultado.perdida >= 0.0


# ---------------------------------------------------------------------------
# 4. Consistencia entre buscar(k=2) y enumeración explícita de biparticiones
# ---------------------------------------------------------------------------

class TestConsistenciaK2Biparticion:
    """Garantiza que buscar(k=2) equivale al mínimo exacto de biparticiones."""

    def _tabla_aleatoria_n4(self, semilla: int = 0) -> dict[tuple[int, ...], float]:
        """Genera una tabla de costos aleatoria para n=4, k=2."""
        rng = np.random.default_rng(semilla)
        tabla = {}
        for asig in k_particiones_asignacion(4, 2):
            tabla[asig] = float(rng.random())
        return tabla

    def test_buscar_k2_coincide_con_minimo_exacto_semilla_0(self) -> None:
        tabla = self._tabla_aleatoria_n4(semilla=0)
        buscador = _MockBuscador(4, tabla)

        resultado = buscador.buscar(k=2)

        minimo_exacto = min(tabla.values())
        assert math.isclose(resultado.perdida, minimo_exacto, abs_tol=1e-9), (
            f"buscar(k=2)={resultado.perdida} != mínimo exacto={minimo_exacto}"
        )

    def test_buscar_k2_coincide_con_minimo_exacto_semilla_42(self) -> None:
        tabla = self._tabla_aleatoria_n4(semilla=42)
        buscador = _MockBuscador(4, tabla)

        resultado = buscador.buscar(k=2)

        minimo_exacto = min(tabla.values())
        assert math.isclose(resultado.perdida, minimo_exacto, abs_tol=1e-9)

    def test_buscar_k2_asignacion_en_tabla_de_biparticiones(self) -> None:
        """La asignación devuelta por buscar(k=2) debe ser una bipartición válida."""
        tabla = self._tabla_aleatoria_n4(semilla=7)
        buscador = _MockBuscador(4, tabla)

        resultado = buscador.buscar(k=2)

        biparticiones_validas = set(k_particiones_asignacion(4, 2))
        assert resultado.asignacion in biparticiones_validas, (
            f"La asignación {resultado.asignacion} no es una bipartición canónica válida"
        )

    @pytest.mark.parametrize("semilla", [1, 5, 13, 21, 99])
    def test_buscar_k2_minimo_parametrico(self, semilla: int) -> None:
        """Verifica la consistencia en múltiples semillas aleatorias."""
        tabla = self._tabla_aleatoria_n4(semilla=semilla)
        buscador = _MockBuscador(4, tabla)

        resultado = buscador.buscar(k=2)
        minimo_exacto = min(tabla.values())

        assert math.isclose(resultado.perdida, minimo_exacto, abs_tol=1e-9)

    def test_kmas_siempre_es_igual_o_mejor_que_k2(self) -> None:
        """buscar(k=3) debe retornar perdida <= buscar(k=2) para la misma tabla."""
        rng = np.random.default_rng(17)
        tabla: dict[tuple[int, ...], float] = {}
        for asig in k_particiones_asignacion(4, 3):
            tabla[asig] = float(rng.random())

        buscador = _MockBuscador(4, tabla)
        resultado_k2 = buscador.buscar(k=2)
        resultado_k3 = buscador.buscar(k=3)

        assert resultado_k3.perdida <= resultado_k2.perdida + 1e-9, (
            f"k=3 ({resultado_k3.perdida}) debe ser <= k=2 ({resultado_k2.perdida})"
        )


# ---------------------------------------------------------------------------
# 5. Consistencia con estrategias reales (QNodos, BranchBound)
# ---------------------------------------------------------------------------

class TestConsistenciaEstrategiasReales:
    """Valida k=2 vs fuerza bruta en sistemas pequeños con solución conocida."""

    @staticmethod
    def _tpm_4nodos() -> np.ndarray:
        """TPM de identidad para 4 nodos: cada estado apunta a sí mismo."""
        return np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 1, 1],
                [0, 1, 0, 0],
                [0, 1, 0, 1],
                [0, 1, 1, 0],
                [0, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 1],
                [1, 0, 1, 0],
                [1, 0, 1, 1],
                [1, 1, 0, 0],
                [1, 1, 0, 1],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
            ],
            dtype=np.float32,
        )

    def test_qnodos_k2_coincide_con_fuerza_bruta(self) -> None:
        """QNodos con k=2 debe dar el mismo phi que FuerzaBruta."""
        from src.controladores.gestor import Gestor
        from src.modelos.base.aplicacion import aplicacion
        from src.estrategias.fuerza_bruta import FuerzaBruta
        from src.estrategias.q_nodos import QNodos

        aplicacion.set_pagina_red_muestra("A")
        tpm = Gestor("1000").cargar_red()

        fb = FuerzaBruta(tpm)
        qn = QNodos(tpm)

        res_fb = fb.aplicar_estrategia("1000", "1110", "1110", "1110")
        res_qn = qn.aplicar_estrategia("1000", "1110", "1110", "1110", k=2)

        assert np.isclose(res_qn.perdida, res_fb.perdida, atol=1e-6), (
            f"QNodos k=2: {res_qn.perdida} != FuerzaBruta: {res_fb.perdida}"
        )
        assert np.allclose(res_qn.distribucion_subsistema, res_fb.distribucion_subsistema)

    def test_branch_bound_k2_coincide_con_fuerza_bruta(self) -> None:
        """BranchBound con k=2 debe dar el mismo phi que FuerzaBruta."""
        from src.controladores.gestor import Gestor
        from src.modelos.base.aplicacion import aplicacion
        from src.estrategias.fuerza_bruta import FuerzaBruta
        from src.estrategias.branch_bound import BranchBound

        aplicacion.set_pagina_red_muestra("A")
        tpm = Gestor("1000").cargar_red()

        fb = FuerzaBruta(tpm)
        bb = BranchBound(tpm)

        res_fb = fb.aplicar_estrategia("1000", "1110", "1110", "1110")
        res_bb = bb.aplicar_estrategia("1000", "1110", "1110", "1110", k=2)

        assert np.isclose(res_bb.perdida, res_fb.perdida, atol=1e-6), (
            f"BranchBound k=2: {res_bb.perdida} != FuerzaBruta: {res_fb.perdida}"
        )
        assert np.allclose(res_bb.distribucion_subsistema, res_fb.distribucion_subsistema)

    def test_qnodos_k3_es_igual_o_mejor_que_k2(self) -> None:
        """QNodos con k=3 debe retornar perdida <= que con k=2."""
        from src.modelos.base.aplicacion import aplicacion
        from src.estrategias.q_nodos import QNodos

        aplicacion.set_pagina_red_muestra("A")
        tpm = self._tpm_4nodos()
        qn = QNodos(tpm)

        res_k2 = qn.aplicar_estrategia("1000", "1111", "1111", "1111", k=2)
        res_k3 = qn.aplicar_estrategia("1000", "1111", "1111", "1111", k=3)

        assert res_k3.perdida <= res_k2.perdida + 1e-9, (
            f"QNodos k=3 ({res_k3.perdida}) > k=2 ({res_k2.perdida})"
        )

    def test_branch_bound_k3_es_igual_o_mejor_que_k2(self) -> None:
        """BranchBound con k=3 debe retornar perdida <= que con k=2."""
        from src.modelos.base.aplicacion import aplicacion
        from src.estrategias.branch_bound import BranchBound

        aplicacion.set_pagina_red_muestra("A")
        tpm = self._tpm_4nodos()
        bb = BranchBound(tpm)

        res_k2 = bb.aplicar_estrategia("1000", "1111", "1111", "1111", k=2)
        res_k3 = bb.aplicar_estrategia("1000", "1111", "1111", "1111", k=3)

        assert res_k3.perdida <= res_k2.perdida + 1e-9, (
            f"BranchBound k=3 ({res_k3.perdida}) > k=2 ({res_k2.perdida})"
        )

    def test_distribucion_particion_tiene_forma_correcta(self) -> None:
        """La distribución retornada por k-partición debe tener la misma forma que el subsistema."""
        from src.modelos.base.aplicacion import aplicacion
        from src.estrategias.branch_bound import BranchBound

        aplicacion.set_pagina_red_muestra("A")
        tpm = self._tpm_4nodos()
        bb = BranchBound(tpm)

        for k in (2, 3):
            res = bb.aplicar_estrategia("1000", "1111", "1111", "1111", k=k)
            assert res.distribucion_particion.shape == res.distribucion_subsistema.shape, (
                f"Formas inconsistentes para k={k}: "
                f"particion={res.distribucion_particion.shape} vs "
                f"subsistema={res.distribucion_subsistema.shape}"
            )

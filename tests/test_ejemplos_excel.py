"""Tests derivados de Ejemplos.xlsx.

Cada test usa datos calculados manualmente en el Excel y verifica que el código
produce el mismo resultado. Las hojas fuente se indican en cada sección.
"""

import numpy as np
import pytest

from src.estrategias.fuerza_bruta import FuerzaBruta
from src.funciones.iit import emd_efecto
from src.modelos.base.aplicacion import aplicacion
from src.modelos.enumeraciones.notacion import Notation
from src.modelos.nucleo.sistema import Sistema
from src.modelos.nucleo.solucion import Solucion


# ---------------------------------------------------------------------------
# TPM del sistema ABC (3 nodos) — hoja EjParticion / Ej1RepEfectSinEV
#
# Estado actual (A, B, C) → estado futuro determinístico:
#   (0,0,0) → (0,0,0)   (1,0,0) → (0,0,1)
#   (0,1,0) → (1,0,1)   (1,1,0) → (1,0,0)
#   (0,0,1) → (1,0,0)   (1,0,1) → (1,1,1)
#   (0,1,1) → (1,0,1)   (1,1,1) → (1,1,0)
#
# Formato código: filas = estados (A varía más rápido), columnas = [P(A=1), P(B=1), P(C=1)]
# ---------------------------------------------------------------------------

TPM_ABC = np.array(
    [
        [0, 0, 0],  # A=0,B=0,C=0 → (0,0,0)
        [0, 0, 1],  # A=1,B=0,C=0 → (0,0,1)
        [1, 0, 1],  # A=0,B=1,C=0 → (1,0,1)
        [1, 0, 0],  # A=1,B=1,C=0 → (1,0,0)
        [1, 0, 0],  # A=0,B=0,C=1 → (1,0,0)
        [1, 1, 1],  # A=1,B=0,C=1 → (1,1,1)
        [1, 0, 1],  # A=0,B=1,C=1 → (1,0,1)
        [1, 1, 0],  # A=1,B=1,C=1 → (1,1,0)
    ],
    dtype=np.float32,
)


@pytest.fixture(autouse=True)
def usar_lil_endian():
    """Garantiza la notación del Excel (A varía más rápido = little-endian)."""
    aplicacion.set_notacion(Notation.LIL_ENDIAN)
    yield
    aplicacion.set_notacion(Notation.LIL_ENDIAN)


# ---------------------------------------------------------------------------
# Bloque 1 — Estructura de la TPM
# ---------------------------------------------------------------------------


def test_tpm_abc_forma() -> None:
    assert TPM_ABC.shape == (8, 3)


def test_tpm_abc_cada_fila_es_probabilidad() -> None:
    assert np.all((TPM_ABC >= 0) & (TPM_ABC <= 1))


def test_tpm_abc_sistema_se_crea_sin_error() -> None:
    estado = np.array([0, 0, 0], dtype=np.int8)
    sistema = Sistema(TPM_ABC, estado)
    assert len(sistema.ncubos) == 3


# ---------------------------------------------------------------------------
# Bloque 2 — distribucion_marginal por estado inicial
# Hoja EjParticion: cada estado determina la distribución futura.
# ---------------------------------------------------------------------------


def _marginal(estado_str: str) -> list[float]:
    estado = np.array([int(b) for b in estado_str], dtype=np.int8)
    sistema = Sistema(TPM_ABC, estado)
    return sistema.distribucion_marginal().tolist()


def test_distribucion_marginal_estado_000() -> None:
    # (0,0,0) → (0,0,0): todos los nodos van a 0
    assert _marginal("000") == [0.0, 0.0, 0.0]


def test_distribucion_marginal_estado_100() -> None:
    # (A=1,B=0,C=0) → (0,0,1): sólo C sube a 1
    assert _marginal("100") == [0.0, 0.0, 1.0]


def test_distribucion_marginal_estado_010() -> None:
    # (A=0,B=1,C=0) → (1,0,1): A y C suben, B baja
    assert _marginal("010") == [1.0, 0.0, 1.0]


def test_distribucion_marginal_estado_101() -> None:
    # (A=1,B=0,C=1) → (1,1,1): todos suben
    assert _marginal("101") == [1.0, 1.0, 1.0]


def test_distribucion_marginal_estado_111() -> None:
    # (A=1,B=1,C=1) → (1,1,0): A y B suben, C baja
    assert _marginal("111") == [1.0, 1.0, 0.0]


def test_distribucion_marginal_estado_011() -> None:
    # (A=0,B=1,C=1) → (1,0,1): A y C suben
    assert _marginal("011") == [1.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# Bloque 3 — EMD (hoja EMD)
#
# Ejemplo del Excel:
#   p = [0.5, 0, 0.5, 0]  sobre estados (B=0,C=0),(B=1,C=0),(B=0,C=1),(B=1,C=1)
#   q = [0.25, 0.25, 0.25, 0.25]  (uniforme)
#
# Marginales de p:  p_B = [1.0, 0.0],  p_C = [0.5, 0.5]
# Marginales de q:  q_B = [0.5, 0.5],  q_C = [0.5, 0.5]
#
# Propiedad (nodos independientes):
#   EMD(p, q) = EMD(p_B, q_B) + EMD(p_C, q_C)
# ---------------------------------------------------------------------------

P_JOINT = np.array([0.5, 0.0, 0.5, 0.0], dtype=np.float32)
Q_JOINT = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

P_B = np.array([1.0, 0.0], dtype=np.float32)   # marginal de B en p
Q_B = np.array([0.5, 0.5], dtype=np.float32)   # marginal de B en q

P_C = np.array([0.5, 0.5], dtype=np.float32)   # marginal de C en p
Q_C = np.array([0.5, 0.5], dtype=np.float32)   # marginal de C en q


def test_emd_joint_hoja_emd() -> None:
    # EMD(p, q) = 0.25*4 = 1.0
    assert emd_efecto(P_JOINT, Q_JOINT) == pytest.approx(1.0)


def test_emd_marginal_b_hoja_emd() -> None:
    # EMD(p_B, q_B) = |1.0-0.5| + |0.0-0.5| = 1.0
    assert emd_efecto(P_B, Q_B) == pytest.approx(1.0)


def test_emd_marginal_c_hoja_emd() -> None:
    # p_C == q_C → EMD = 0
    assert emd_efecto(P_C, Q_C) == pytest.approx(0.0)


def test_emd_independencia_suma_marginales() -> None:
    # Propiedad del Excel: EMD_joint = EMD_B + EMD_C
    emd_joint = emd_efecto(P_JOINT, Q_JOINT)
    emd_marginal = emd_efecto(P_B, Q_B) + emd_efecto(P_C, Q_C)
    assert emd_joint == pytest.approx(emd_marginal)


# ---------------------------------------------------------------------------
# Bloque 4 — FuerzaBruta sobre el sistema ABC (hoja EjParticion)
# ---------------------------------------------------------------------------


def test_fuerza_bruta_abc_retorna_solucion() -> None:
    estrategia = FuerzaBruta(TPM_ABC)
    resultado = estrategia.aplicar_estrategia(
        estado_inicial="100",
        condicion="111",
        alcance="111",
        mecanismo="111",
    )
    assert isinstance(resultado, Solucion)


def test_fuerza_bruta_abc_perdida_no_negativa() -> None:
    estrategia = FuerzaBruta(TPM_ABC)
    resultado = estrategia.aplicar_estrategia(
        estado_inicial="100",
        condicion="111",
        alcance="111",
        mecanismo="111",
    )
    assert resultado.perdida >= 0.0


def test_fuerza_bruta_abc_distribucion_tiene_3_elementos() -> None:
    estrategia = FuerzaBruta(TPM_ABC)
    resultado = estrategia.aplicar_estrategia(
        estado_inicial="100",
        condicion="111",
        alcance="111",
        mecanismo="111",
    )
    assert resultado.distribucion_subsistema.shape == (3,)


def test_fuerza_bruta_abc_estado_000_dist_subsistema() -> None:
    # Estado (0,0,0) → futuro (0,0,0): P(cada nodo=ON) = [0,0,0]
    estrategia = FuerzaBruta(TPM_ABC)
    resultado = estrategia.aplicar_estrategia(
        estado_inicial="000",
        condicion="111",
        alcance="111",
        mecanismo="111",
    )
    assert resultado.distribucion_subsistema.tolist() == [0.0, 0.0, 0.0]


def test_fuerza_bruta_abc_estado_101_dist_subsistema() -> None:
    # Estado (1,0,1) → futuro (1,1,1): P(cada nodo=ON) = [1,1,1]
    estrategia = FuerzaBruta(TPM_ABC)
    resultado = estrategia.aplicar_estrategia(
        estado_inicial="101",
        condicion="111",
        alcance="111",
        mecanismo="111",
    )
    assert resultado.distribucion_subsistema.tolist() == [1.0, 1.0, 1.0]


def test_fuerza_bruta_abc_particion_existe() -> None:
    estrategia = FuerzaBruta(TPM_ABC)
    resultado = estrategia.aplicar_estrategia(
        estado_inicial="111",
        condicion="111",
        alcance="111",
        mecanismo="111",
    )
    assert resultado.particion != ""

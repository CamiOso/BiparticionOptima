from pathlib import Path

import numpy as np
import pytest

from src.controladores.gestor import Gestor


def test_construir_tpm_desde_muestras_basica() -> None:
    gestor = Gestor(estado_inicial="00")
    muestras = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0],
            [0, 0],
        ],
        dtype=np.int8,
    )

    tpm = gestor.construir_tpm_desde_muestras(muestras)

    esperado = np.array(
        [
            [0.0, 1.0],  # 00 -> 01
            [1.0, 1.0],  # 01 -> 11
            [0.0, 0.0],  # 10 -> 00
            [1.0, 0.0],  # 11 -> 10
        ],
        dtype=np.float32,
    )

    assert tpm.shape == (4, 2)
    assert np.allclose(tpm, esperado)


def test_construir_tpm_rellena_estados_no_observados() -> None:
    gestor = Gestor(estado_inicial="00")
    muestras = np.array(
        [
            [0, 0],
            [0, 0],
            [0, 0],
        ],
        dtype=np.int8,
    )

    tpm = gestor.construir_tpm_desde_muestras(muestras, valor_no_observado=0.5)

    # Solo se observa estado 00. Los demas deben quedar con relleno 0.5.
    assert np.allclose(tpm[0], np.array([0.0, 0.0], dtype=np.float32))
    assert np.allclose(tpm[1:], 0.5)


def test_cargar_muestras_temporales_desde_csv(tmp_path: Path) -> None:
    archivo = tmp_path / "muestras.csv"
    archivo.write_text("0,0\n0,1\n1,1\n", encoding="utf-8")

    gestor = Gestor(estado_inicial="00")
    muestras = gestor.cargar_muestras_temporales(archivo)

    assert muestras.shape == (3, 2)
    assert muestras.dtype == np.int8


def test_cargar_muestras_temporales_rechaza_no_binarias(tmp_path: Path) -> None:
    archivo = tmp_path / "muestras_invalidas.csv"
    archivo.write_text("0,0\n0,2\n", encoding="utf-8")

    gestor = Gestor(estado_inicial="00")
    with pytest.raises(ValueError, match="binarias"):
        _ = gestor.cargar_muestras_temporales(archivo)


def test_construir_tpm_desde_csv_muestras(tmp_path: Path) -> None:
    archivo = tmp_path / "muestras.csv"
    archivo.write_text("0,0\n0,1\n1,1\n1,0\n0,0\n", encoding="utf-8")

    gestor = Gestor(estado_inicial="00")
    tpm = gestor.construir_tpm_desde_csv_muestras(archivo)

    assert tpm.shape == (4, 2)
    assert np.isclose(tpm[0, 1], 1.0)

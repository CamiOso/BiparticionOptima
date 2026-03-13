from pathlib import Path

import numpy as np
import pandas as pd

from src.controladores.gestor import Gestor
from src.modelos.base.aplicacion import aplicacion
from src.modelos.nucleo.solucion import Solucion
from src.estrategias.fuerza_bruta import FuerzaBruta


def _sample_tpm_4nodes() -> np.ndarray:
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


def test_bruteforce_returns_solution() -> None:
    aplicacion.set_pagina_red_muestra("A")
    strategy = FuerzaBruta(_sample_tpm_4nodes())

    result = strategy.aplicar_estrategia(
        estado_inicial="1000",
        condicion="1111",
        alcance="1111",
        mecanismo="1111",
    )

    assert isinstance(result, Solucion)
    assert result.perdida >= 0.0
    assert result.distribucion_subsistema.shape == (4,)


def test_bruteforce_generates_analysis_report(tmp_path: Path) -> None:
    aplicacion.set_pagina_red_muestra("A")
    estrategia = FuerzaBruta(_sample_tpm_4nodes())

    directorio = estrategia.analizar_red_completa(
        estado_inicial="1000",
        directorio_salida=tmp_path / "salida",
    )

    archivos = sorted(directorio.glob("*.xlsx"))

    assert directorio.exists()
    assert archivos

    libro = pd.ExcelFile(archivos[0])
    assert libro.sheet_names


def test_bruteforce_matches_reference_case_on_sample_a() -> None:
    aplicacion.set_pagina_red_muestra("A")
    estrategia = FuerzaBruta(Gestor("1000").cargar_red())

    resultado = estrategia.aplicar_estrategia(
        estado_inicial="1000",
        condicion="1110",
        alcance="1110",
        mecanismo="1110",
    )

    assert resultado.distribucion_subsistema.tolist() == [0.0, 0.0, 1.0]
    assert resultado.distribucion_particion.tolist() == [0.0, 0.25, 1.0]
    assert resultado.perdida == 0.25

import json
from pathlib import Path

import numpy as np
import pytest

from src import main as main_module
from src.modelos.nucleo.solucion import Solucion


def _solucion_dummy(estrategia: str = "Dummy", estado_inicial: str = "0") -> Solucion:
    return Solucion(
        estrategia=estrategia,
        perdida=0.25,
        distribucion_subsistema=np.array([1.0], dtype=np.float32),
        distribucion_particion=np.array([0.5], dtype=np.float32),
        estado_inicial=estado_inicial,
        particion="A|B",
    )


def test_validar_bitstring_ok_and_errors() -> None:
    main_module.validar_bitstring("1010")

    with pytest.raises(ValueError):
        main_module.validar_bitstring("")

    with pytest.raises(ValueError):
        main_module.validar_bitstring("10a0")


def test_solucion_a_dict_includes_elapsed_seconds() -> None:
    resultado = _solucion_dummy(estrategia="Geometric", estado_inicial="1000")

    payload = main_module._solucion_a_dict(resultado, elapsed_seconds=1.5)

    assert payload["estrategia"] == "Geometric"
    assert payload["estado_inicial"] == "1000"
    assert payload["particion"] == "A|B"
    assert payload["elapsed_seconds"] == 1.5


@pytest.mark.parametrize(
    ("estrategia", "attr", "expected_key", "expected_nombre"),
    [
        ("fuerza_bruta", "FuerzaBruta", "fuerza_bruta", "FB"),
        ("phi", "Phi", "phi", "Phi"),
        ("qnodos", "QNodos", "qnodos", "Q"),
        ("geometric", "Geometric", "geometric_refinado", "Geo"),
    ],
)
def test_ejecutar_estrategia_por_rama(
    monkeypatch: pytest.MonkeyPatch,
    estrategia: str,
    attr: str,
    expected_key: str,
    expected_nombre: str,
) -> None:
    class DummySolver:
        def __init__(self, _tpm):
            pass

        def aplicar_estrategia(self, **_kwargs):
            return _solucion_dummy(estrategia=expected_nombre, estado_inicial="0")

    monkeypatch.setattr(main_module, attr, DummySolver)
    main_module.aplicacion.modo_geometrico = "refinado"

    clave, resultado, elapsed = main_module._ejecutar_estrategia(
        estrategia=estrategia,
        tpm=np.array([[0.0]], dtype=np.float32),
        estado_inicial="0",
        mascara="1",
    )

    assert clave == expected_key
    assert isinstance(resultado, Solucion)
    assert resultado.estrategia == expected_nombre
    assert elapsed >= 0.0


def test_ejecutar_estrategia_rechaza_estrategia_invalida() -> None:
    with pytest.raises(ValueError):
        main_module._ejecutar_estrategia(
            estrategia="invalida",
            tpm=np.array([[0.0]], dtype=np.float32),
            estado_inicial="0",
            mascara="1",
        )


def test_iniciar_single_strategy_con_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyGestor:
        def __init__(self, estado_inicial: str):
            self.estado_inicial = estado_inicial
            self.archivo_tpm = Path("src/.samples/N1A.csv")

        def cargar_red(self):
            return np.array([[0.0]], dtype=np.float32)

    class DummySistema:
        def __init__(self, _tpm, _estado):
            pass

        def distribucion_marginal(self):
            return np.array([1.0], dtype=np.float32)

    class DummyCube:
        def __init__(self, indice, dims, data):
            self.indice = indice
            self.dims = dims
            self.data = data

        def marginalizar(self, _dims):
            return DummyCube(
                indice=self.indice,
                dims=np.array([1], dtype=np.int8),
                data=np.array([1.0], dtype=np.float32),
            )

    def fake_ejecutar(estrategia, tpm, estado_inicial, mascara):
        assert estrategia == "geometric"
        assert estado_inicial == "0"
        assert mascara == "1"
        return "geometric_refinado", _solucion_dummy("Geometric", estado_inicial), 0.01

    monkeypatch.setattr(main_module, "Gestor", DummyGestor)
    monkeypatch.setattr(main_module, "Sistema", DummySistema)
    monkeypatch.setattr(main_module, "NCube", DummyCube)
    monkeypatch.setattr(main_module, "_ejecutar_estrategia", fake_ejecutar)

    output_path = tmp_path / "salida" / "resultado.json"
    payload = main_module.iniciar(
        estrategia="geometric",
        modo_geometric="refinado",
        estado_inicial="0",
        output_json=str(output_path),
    )

    assert payload["estado_inicial"] == "0"
    assert "geometric_refinado" in payload["resultados"]
    assert output_path.exists()

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["resultados"]["geometric_refinado"]["elapsed_seconds"] == 0.01


def test_iniciar_todas_ejecuta_flujo_completo(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyGestor:
        def __init__(self, estado_inicial: str):
            self.archivo_tpm = Path("src/.samples/N1A.csv")
            self.estado_inicial = estado_inicial

        def cargar_red(self):
            return np.array([[0.0]], dtype=np.float32)

    class DummySistema:
        def __init__(self, _tpm, _estado):
            pass

        def distribucion_marginal(self):
            return np.array([1.0], dtype=np.float32)

    class DummyCube:
        def __init__(self, indice, dims, data):
            self.indice = indice
            self.dims = dims
            self.data = data

        def marginalizar(self, _dims):
            return DummyCube(
                indice=self.indice,
                dims=np.array([1], dtype=np.int8),
                data=np.array([1.0], dtype=np.float32),
            )

    llamadas: list[str] = []

    def fake_ejecutar(estrategia, tpm, estado_inicial, mascara):
        llamadas.append(estrategia)
        if estrategia == "geometric":
            clave = f"geometric_{main_module.aplicacion.modo_geometrico}"
            return clave, _solucion_dummy("Geometric", estado_inicial), 0.02
        return estrategia, _solucion_dummy(estrategia, estado_inicial), 0.02

    monkeypatch.setattr(main_module, "Gestor", DummyGestor)
    monkeypatch.setattr(main_module, "Sistema", DummySistema)
    monkeypatch.setattr(main_module, "NCube", DummyCube)
    monkeypatch.setattr(main_module, "_ejecutar_estrategia", fake_ejecutar)

    payload = main_module.iniciar(estrategia="todas", estado_inicial="0")

    assert llamadas == ["fuerza_bruta", "phi", "qnodos", "geometric", "geometric"]
    assert "fuerza_bruta" in payload["resultados"]
    assert "phi" in payload["resultados"]
    assert "qnodos" in payload["resultados"]
    assert "geometric_estricto" in payload["resultados"]
    assert "geometric_refinado" in payload["resultados"]


def test_iniciar_rechaza_modo_geometrico_invalido() -> None:
    with pytest.raises(ValueError):
        main_module.iniciar(
            estrategia="geometric",
            modo_geometric="modo-incorrecto",
            estado_inicial="0",
        )
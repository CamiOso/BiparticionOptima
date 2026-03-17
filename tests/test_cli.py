import json
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "exec.py", *args]
    return subprocess.run(
        cmd,
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_geometric_refinado_runs_ok() -> None:
    result = _run_cli("--estrategia", "geometric", "--modo-geometric", "refinado")

    assert result.returncode == 0
    assert "Geometric (refinado)" in result.stdout


def test_cli_geometric_estricto_runs_ok() -> None:
    result = _run_cli("--estrategia", "geometric", "--modo-geometric", "estricto")

    assert result.returncode == 0
    assert "Geometric (estricto)" in result.stdout


def test_cli_fuerza_bruta_runs_ok() -> None:
    result = _run_cli("--estrategia", "fuerza_bruta")

    assert result.returncode == 0
    assert "FuerzaBruta" in result.stdout


def test_cli_rejects_invalid_strategy() -> None:
    result = _run_cli("--estrategia", "invalida")

    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_cli_rejects_invalid_estado_inicial() -> None:
    result = _run_cli("--estrategia", "geometric", "--estado-inicial", "10a0")

    assert result.returncode != 0
    assert "cadena binaria" in result.stderr


def test_cli_output_json_creates_file(tmp_path: Path) -> None:
    output_path = tmp_path / "salida" / "resultado.json"
    result = _run_cli(
        "--estrategia",
        "geometric",
        "--modo-geometric",
        "refinado",
        "--estado-inicial",
        "1000",
        "--output-json",
        str(output_path),
    )

    assert result.returncode == 0
    assert output_path.exists()

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["estado_inicial"] == "1000"
    assert "resultados" in data
    assert "geometric_refinado" in data["resultados"]

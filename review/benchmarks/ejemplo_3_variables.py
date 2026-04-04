from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.modelos.enumeraciones.geometric_mode import GeometricMode
from src.strategies.geometric import Geometric


def _indice_a_estado(indice: int, n: int) -> str:
    return format(indice, f"0{n}b")


def _distancia_hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def _tabla_costos_estados(n: int) -> list[list[float]]:
    estados = [_indice_a_estado(i, n) for i in range(1 << n)]
    tabla = []
    for estado_i in estados:
        fila = []
        for estado_j in estados:
            d = _distancia_hamming(estado_i, estado_j)
            fila.append(2.0 ** (-d))
        tabla.append(fila)
    return tabla


def _guardar_tabla_costos(tabla: list[list[float]], n: int, destino: Path) -> None:
    estados = [_indice_a_estado(i, n) for i in range(1 << n)]
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["estado"] + estados)
        for estado, fila in zip(estados, tabla):
            writer.writerow([estado] + [f"{valor:.6f}" for valor in fila])


def _tpm_ejemplo_3_variables() -> np.ndarray:
    # TPM sintetica 2^3 x 3 para demostrar flujo de analisis en 3 nodos.
    return np.array(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.8, 0.4],
            [0.7, 0.3, 0.4],
            [0.8, 0.9, 0.6],
            [0.4, 0.4, 0.2],
            [0.6, 0.7, 0.5],
            [0.5, 0.2, 0.8],
            [0.9, 0.8, 0.9],
        ],
        dtype=np.float32,
    )


def main() -> None:
    n = 3
    estado_inicial = "000"
    mascara = "111"

    tpm = _tpm_ejemplo_3_variables()

    estrategia = Geometric(tpm, mode=GeometricMode.STRICT)
    resultado = estrategia.aplicar_estrategia(
        estado_inicial=estado_inicial,
        condicion=mascara,
        alcance=mascara,
        mecanismo=mascara,
    )

    tabla_costos = _tabla_costos_estados(n)
    salida_tabla = Path("review") / "salidas" / "tabla_costos_3_variables.csv"
    _guardar_tabla_costos(tabla_costos, n, salida_tabla)

    origen = "000"
    destino = "011"
    d = _distancia_hamming(origen, destino)
    gamma = 2.0 ** (-d)

    print("=== Ejemplo practico de 3 variables ===")
    print(f"TPM forma: {tpm.shape}")
    print(f"Transicion {origen} -> {destino}: distancia={d}, gamma=2^(-d)={gamma:.4f}")
    print(f"Biparticion optima (Geometric estricto): {resultado.particion}")
    print(f"Perdida minima estimada: {resultado.perdida:.6f}")
    print(f"Tabla de costos guardada en: {salida_tabla}")


if __name__ == "__main__":
    main()

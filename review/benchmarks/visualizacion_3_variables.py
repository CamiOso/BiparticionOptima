from __future__ import annotations

import csv
from pathlib import Path


def _estados_binarios(n: int) -> list[str]:
    return [format(indice, f"0{n}b") for indice in range(1 << n)]


def _hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def _coordenadas_3d(estado: str) -> tuple[int, int, int]:
    # A=bit 0, B=bit 1, C=bit 2
    return int(estado[0]), int(estado[1]), int(estado[2])


def _proyeccion_isometrica(x: int, y: int, z: int, escala: float = 90.0) -> tuple[float, float]:
    u = (x - y) * escala
    v = ((x + y) * 0.5 - z) * escala
    return u, v


def _edges_hipercubo(estados: list[str]) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for i, estado_i in enumerate(estados):
        for estado_j in estados[i + 1 :]:
            if _hamming(estado_i, estado_j) == 1:
                edges.append((estado_i, estado_j))
    return edges


def _guardar_svg_hipercubo(destino: Path) -> None:
    estados = _estados_binarios(3)
    edges = _edges_hipercubo(estados)

    puntos: dict[str, tuple[float, float]] = {}
    for estado in estados:
        x, y, z = _coordenadas_3d(estado)
        u, v = _proyeccion_isometrica(x, y, z)
        puntos[estado] = (u + 180.0, v + 170.0)

    lineas = []
    for a, b in edges:
        x1, y1 = puntos[a]
        x2, y2 = puntos[b]
        lineas.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            'stroke="#1f2937" stroke-width="2" />'
        )

    nodos = []
    for estado in estados:
        x, y = puntos[estado]
        nodos.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="#0ea5e9" stroke="#082f49" stroke-width="2" />'
        )
        nodos.append(
            f'<text x="{x + 14:.1f}" y="{y + 4:.1f}" font-size="12" fill="#111827" font-family="monospace">{estado}</text>'
        )

    contenido = "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="420" height="340" viewBox="0 0 420 340">',
            '<rect x="0" y="0" width="420" height="340" fill="#f8fafc" />',
            '<text x="16" y="24" font-size="16" fill="#0f172a" font-family="sans-serif">Hipercubo de 3 variables (proyeccion isometrica)</text>',
            *lineas,
            *nodos,
            '</svg>',
        ]
    )

    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(contenido, encoding="utf-8")


def _guardar_proyecciones_csv(destino: Path) -> None:
    estados = _estados_binarios(3)
    proyecciones = {
        "AB": lambda s: s[:2],
        "AC": lambda s: s[0] + s[2],
        "BC": lambda s: s[1:],
    }

    filas: list[tuple[str, str, str]] = []
    for nombre, fn in proyecciones.items():
        grupos: dict[str, list[str]] = {}
        for estado in estados:
            clave = fn(estado)
            grupos.setdefault(clave, []).append(estado)
        for clave, miembros in sorted(grupos.items()):
            filas.append((nombre, clave, " ".join(miembros)))

    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["proyeccion", "grupo", "estados"])
        writer.writerows(filas)


def _guardar_adyacencia_csv(destino: Path) -> None:
    estados = _estados_binarios(3)
    edges = _edges_hipercubo(estados)

    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["origen", "destino", "distancia_hamming"])
        for origen, destino_estado in edges:
            writer.writerow([origen, destino_estado, 1])


def main() -> None:
    salida_svg = Path("review") / "salidas" / "hipercubo_3_variables.svg"
    salida_proy = Path("review") / "salidas" / "proyecciones_3_variables.csv"
    salida_edges = Path("review") / "salidas" / "adyacencia_hipercubo_3_variables.csv"

    _guardar_svg_hipercubo(salida_svg)
    _guardar_proyecciones_csv(salida_proy)
    _guardar_adyacencia_csv(salida_edges)

    print(f"SVG generado en: {salida_svg}")
    print(f"Proyecciones generadas en: {salida_proy}")
    print(f"Adyacencia generada en: {salida_edges}")


if __name__ == "__main__":
    main()

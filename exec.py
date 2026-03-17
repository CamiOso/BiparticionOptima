import argparse
import sys

from src.main import iniciar


def _crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ejecutar estrategias SIA por consola.")
    parser.add_argument(
        "--estrategia",
        default="todas",
        choices=["todas", "fuerza_bruta", "phi", "qnodos", "geometric"],
        help="Estrategia a ejecutar.",
    )
    parser.add_argument(
        "--modo-geometric",
        default="refinado",
        choices=["estricto", "refinado"],
        help="Modo de Geometric (solo aplica cuando la estrategia es geometric).",
    )
    parser.add_argument(
        "--estado-inicial",
        default="1000",
        help="Estado inicial binario del sistema (ej: 1000).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Ruta opcional para guardar resultados en JSON.",
    )
    return parser


def main() -> None:
    """Punto de entrada del proyecto."""
    parser = _crear_parser()
    args = parser.parse_args()
    try:
        iniciar(
            estrategia=args.estrategia,
            modo_geometric=args.modo_geometric,
            estado_inicial=args.estado_inicial,
            output_json=args.output_json,
        )
    except (ValueError, FileNotFoundError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()

import argparse

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
    return parser


def main() -> None:
    """Punto de entrada del proyecto."""
    parser = _crear_parser()
    args = parser.parse_args()

    iniciar(
        estrategia=args.estrategia,
        modo_geometric=args.modo_geometric,
    )


if __name__ == "__main__":
    main()

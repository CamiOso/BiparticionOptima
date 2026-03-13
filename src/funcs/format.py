from __future__ import annotations

import numpy as np


def fmt_vector(values: np.ndarray) -> str:
    """Formatea un vector numerico con precision fija para mostrar en consola."""
    return "[" + ", ".join(f"{float(v):.4f}" for v in values.tolist()) + "]"


def fmt_solution_block(
    estrategia: str,
    estado_inicial: str,
    perdida: float,
    distribucion_subsistema: np.ndarray,
    distribucion_particion: np.ndarray,
) -> str:
    """Renderiza un bloque de salida consistente para una solucion de estrategia."""
    return (
        f"Estrategia: {estrategia}\n"
        f"Estado inicial: {estado_inicial}\n"
        f"Perdida: {perdida:.4f}\n"
        f"Dist. subsistema: {fmt_vector(distribucion_subsistema)}\n"
        f"Dist. particion: {fmt_vector(distribucion_particion)}"
    )

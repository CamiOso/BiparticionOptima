from __future__ import annotations

import numpy as np


def fmt_vector(values: np.ndarray) -> str:
    """Formatea un vector numerico con precision fija para mostrar en consola."""
    return "[" + ", ".join(f"{float(v):.4f}" for v in values.tolist()) + "]"


def fmt_biparticion(
    subalcance: tuple[int, ...],
    submecanismo: tuple[int, ...],
    alcance_total: tuple[int, ...],
    mecanismo_total: tuple[int, ...],
) -> str:
    """Renderiza una biparticion didactica con grupos preservados y su complemento."""
    alcance_comp = tuple(v for v in alcance_total if v not in subalcance)
    mecanismo_comp = tuple(v for v in mecanismo_total if v not in submecanismo)
    return (
        f"(M={submecanismo}, A={subalcance}) | "
        f"(M*={mecanismo_comp}, A*={alcance_comp})"
    )


def fmt_solution_block(
    estrategia: str,
    estado_inicial: str,
    perdida: float,
    distribucion_subsistema: np.ndarray,
    distribucion_particion: np.ndarray,
    particion: str,
) -> str:
    """Renderiza un bloque de salida consistente para una solucion de estrategia."""
    return (
        f"Estrategia: {estrategia}\n"
        f"Estado inicial: {estado_inicial}\n"
        f"Perdida: {perdida:.4f}\n"
        f"Biparticion: {particion}\n"
        f"Dist. subsistema: {fmt_vector(distribucion_subsistema)}\n"
        f"Dist. particion: {fmt_vector(distribucion_particion)}"
    )


def fmt_biparticion_q(
    grupo_uno: list[tuple[int, int]],
    grupo_dos: list[tuple[int, int]],
) -> str:
    """Formatea la biparticion de QNodos usando nodos (tiempo, indice)."""

    def separar(grupo: list[tuple[int, int]]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        mecanismos = tuple(idx for tiempo, idx in sorted(grupo, key=lambda v: v[1]) if tiempo == 0)
        alcances = tuple(idx for tiempo, idx in sorted(grupo, key=lambda v: v[1]) if tiempo == 1)
        return mecanismos, alcances

    mecanismos_uno, alcances_uno = separar(grupo_uno)
    mecanismos_dos, alcances_dos = separar(grupo_dos)
    return (
        f"G1(M={mecanismos_uno}, A={alcances_uno}) | "
        f"G2(M={mecanismos_dos}, A={alcances_dos})"
    )

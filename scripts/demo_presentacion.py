"""Demo en vivo para la presentación final — k-particiones IIT.

Ejecutar:
    python demo_presentacion.py            # demo completo
    python demo_presentacion.py --paso 1   # solo bipartición k=2 (QNodos vs Geometric)
    python demo_presentacion.py --paso 2   # k=3 vs k=2 (QNodos vs Geometric)
    python demo_presentacion.py --paso 3   # visualización hipercubo
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Sistema de prueba: red de 4 nodos, estado 1000, subsistema ABC (1110)
# ──────────────────────────────────────────────────────────────────────────────
ESTADO    = "1000"
CONDICION = "1110"
ALCANCE   = "1110"
MECANISMO = "1110"
MUESTRA   = "A"          # hoja del Excel de referencia

SEP = "─" * 60


def _cargar_tpm() -> np.ndarray:
    from src.controladores.gestor import Gestor
    from src.modelos.base.aplicacion import aplicacion
    aplicacion.set_pagina_red_muestra(MUESTRA)
    return Gestor(ESTADO).cargar_red()


def _print_resultado(titulo: str, res, tiempo: float) -> None:
    print(f"\n  {titulo}")
    print(f"  {'Pérdida φ':.<30} {res.perdida:.6f}")
    print(f"  {'Partición':.<30} {res.particion}")
    print(f"  {'Tiempo':.<30} {tiempo:.3f} s")


# ──────────────────────────────────────────────────────────────────────────────
# Paso 1 — Bipartición clásica (k=2)
# ──────────────────────────────────────────────────────────────────────────────

def paso1_biparticion() -> None:
    print(f"\n{SEP}")
    print("  PASO 1 — Bipartición (k=2)  [caso base IIT clásica]")
    print(SEP)
    print(f"  Sistema   : {MUESTRA}  |  Estado inicial : {ESTADO}")
    print(f"  Subsistema: alcance={ALCANCE}, mecanismo={MECANISMO}")

    from src.estrategias.q_nodos import QNodos
    from src.strategies.geometric import Geometric

    tpm = _cargar_tpm()

    print("\n  [Q-Nodos — k=2]")
    t0 = time.perf_counter()
    res_qn = QNodos(tpm).aplicar_estrategia(ESTADO, CONDICION, ALCANCE, MECANISMO, k=2)
    _print_resultado("Q-Nodos", res_qn, time.perf_counter() - t0)

    print("\n  [Geometric — k=2]")
    t0 = time.perf_counter()
    res_geo = Geometric(tpm).aplicar_estrategia(ESTADO, CONDICION, ALCANCE, MECANISMO, k=2)
    _print_resultado("Geometric", res_geo, time.perf_counter() - t0)

    ok = np.isclose(res_qn.perdida, res_geo.perdida, atol=1e-4)
    print(f"\n  ✓ Q-Nodos y Geometric coinciden en k=2: {ok}")


# ──────────────────────────────────────────────────────────────────────────────
# Paso 2 — Extensión a k=3: ¿mejora φ?
# ──────────────────────────────────────────────────────────────────────────────

def paso2_k3_vs_k2() -> None:
    print(f"\n{SEP}")
    print("  PASO 2 — Extensión a k=3  [más grupos → menor o igual pérdida]")
    print(SEP)

    from src.estrategias.q_nodos import QNodos
    from src.strategies.geometric import Geometric

    tpm = _cargar_tpm()

    print("\n  [Q-Nodos]")
    res_qn = {}
    for k in (2, 3):
        t0 = time.perf_counter()
        res_qn[k] = QNodos(tpm).aplicar_estrategia(ESTADO, CONDICION, ALCANCE, MECANISMO, k=k)
        _print_resultado(f"k={k}", res_qn[k], time.perf_counter() - t0)
    delta_qn = res_qn[3].perdida - res_qn[2].perdida
    print(f"\n  Δφ Q-Nodos (k=3 − k=2) = {delta_qn:+.6f}")
    print(f"  {'⬇ k=3 encontró partición mejor' if delta_qn < -1e-6 else '= bipartición ya era óptima'}")

    print("\n  [Geometric]")
    res_geo = {}
    for k in (2, 3):
        t0 = time.perf_counter()
        res_geo[k] = Geometric(tpm).aplicar_estrategia(ESTADO, CONDICION, ALCANCE, MECANISMO, k=k)
        _print_resultado(f"k={k}", res_geo[k], time.perf_counter() - t0)
    delta_geo = res_geo[3].perdida - res_geo[2].perdida
    print(f"\n  Δφ Geometric (k=3 − k=2) = {delta_geo:+.6f}")
    print(f"  {'⬇ k=3 encontró partición mejor' if delta_geo < -1e-6 else '= bipartición ya era óptima'}")


# ──────────────────────────────────────────────────────────────────────────────
# Paso 3 — Visualización del hipercubo con k-particiones
# ──────────────────────────────────────────────────────────────────────────────

def paso3_hipercubo() -> None:
    print(f"\n{SEP}")
    print("  PASO 3 — Visualización hipercubo con k-particiones")
    print(SEP)

    from src.estrategias.q_nodos import QNodos
    from src.strategies.geometric import Geometric

    tpm = _cargar_tpm()

    print("\n  [Q-Nodos]")
    for k in (2, 3):
        res = QNodos(tpm).aplicar_estrategia(ESTADO, CONDICION, ALCANCE, MECANISMO, k=k)
        print(f"  k={k}: φ={res.perdida:.6f}  partición={res.particion}")

    print("\n  [Geometric]")
    for k in (2, 3):
        res = Geometric(tpm).aplicar_estrategia(ESTADO, CONDICION, ALCANCE, MECANISMO, k=k)
        print(f"  k={k}: φ={res.perdida:.6f}  partición={res.particion}")

    # Generar visualizaciones
    salida_dir = Path("presentacion_assets")
    salida_dir.mkdir(exist_ok=True)

    _visualizar_hipercubo_kparticion(
        n=3,
        k=2,
        guardar_en=salida_dir / "hipercubo_k2.png",
    )
    _visualizar_hipercubo_kparticion(
        n=3,
        k=3,
        guardar_en=salida_dir / "hipercubo_k3.png",
    )
    print(f"\n  Imágenes guardadas en {salida_dir.resolve()}/")


def _visualizar_hipercubo_kparticion(
    n: int,
    k: int,
    guardar_en: Path | None = None,
) -> None:
    """Dibuja el hipercubo n-dimensional con los estados coloreados por grupo.

    Para n=3 usa proyección isométrica 3D. Los estados se asignan a grupos
    según sus coordenadas: el hiperplano de corte se visualiza como frontera
    entre colores, ilustrando los k-1 planos de separación.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Paleta
    COLORES = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
               "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]

    estados = [format(i, f"0{n}b") for i in range(1 << n)]
    coords = {s: tuple(int(b) for b in s) for s in estados}

    # Asignación de grupos: dividimos los estados en k partes según
    # su índice para ilustrar cómo k-1 hiperplanos cortan el cubo.
    total = len(estados)
    grupo_estado: dict[str, int] = {}
    for idx, estado in enumerate(estados):
        # Corte uniforme: grupo = floor(idx * k / total)
        grupo_estado[estado] = min(int(idx * k / total), k - 1)

    # Aristas del hipercubo (Hamming-1)
    aristas = [
        (estados[i], estados[j])
        for i in range(total)
        for j in range(i + 1, total)
        if sum(a != b for a, b in zip(estados[i], estados[j])) == 1
    ]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Dibujar aristas
    for a, b in aristas:
        xa, ya, za = coords[a]
        xb, yb, zb = coords[b]
        ga, gb = grupo_estado[a], grupo_estado[b]
        color = "#cccccc" if ga != gb else "#555555"
        lw = 0.8 if ga != gb else 1.5
        ls = "--" if ga != gb else "-"
        ax.plot([xa, xb], [ya, yb], [za, zb], color=color, linewidth=lw, linestyle=ls)

    # Dibujar vértices coloreados por grupo
    for estado in estados:
        x, y, z = coords[estado]
        g = grupo_estado[estado]
        ax.scatter(x, y, z, color=COLORES[g], s=200, zorder=5,
                   edgecolors="white", linewidths=1.5)
        ax.text(x + 0.05, y + 0.05, z + 0.05, estado, fontsize=8,
                color="#222222", fontfamily="monospace")

    # Dibujar hiperplano(s) de corte como planos semitransparentes
    if n == 3 and k == 2:
        # Un hiperplano: x = 0.5 (separa los estados con x=0 de x=1)
        xx, yy = np.meshgrid([0, 1], [0, 1])
        zz = np.ones_like(xx) * 0.5
        ax.plot_surface(np.full_like(xx, 0.5), xx, yy,
                        alpha=0.15, color="#DD8452", zorder=0)
        ax.text(0.5, 0.5, 1.05, "Hiperplano\nde corte",
                fontsize=8, color="#995522", ha="center")
    elif n == 3 and k == 3:
        for x_corte, color_p in [(0.33, "#4C72B0"), (0.67, "#55A868")]:
            xx, yy = np.meshgrid([0, 1], [0, 1])
            ax.plot_surface(np.full_like(xx, x_corte), xx, yy,
                            alpha=0.12, color=color_p, zorder=0)

    # Leyenda
    parches = [
        mpatches.Patch(color=COLORES[g], label=f"Grupo {g}")
        for g in range(k)
    ]
    ax.legend(handles=parches, loc="upper left", fontsize=9)

    ax.set_xlabel("Nodo A")
    ax.set_ylabel("Nodo B")
    ax.set_zlabel("Nodo C")
    ax.set_title(
        f"Hipercubo Q₃ — k={k} particiones\n"
        f"({k-1} hiperplano{'s' if k > 2 else ''} de corte)",
        fontsize=12,
    )
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_zlim(-0.1, 1.1)
    ax.view_init(elev=20, azim=35)

    plt.tight_layout()
    if guardar_en:
        plt.savefig(guardar_en, dpi=150, bbox_inches="tight")
        print(f"  → {guardar_en}")
    else:
        plt.show()
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Entrada principal
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Demo de k-particiones para presentación.")
    parser.add_argument(
        "--paso",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Ejecutar solo un paso (1=bipartición, 2=k=3 vs k=2, 3=hipercubo).",
    )
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  DEMO — Búsqueda de k-Particiones (IIT / MIP)")
    print("  ProyectoAnalisis2026")
    print("═" * 60)

    pasos = {1: paso1_biparticion, 2: paso2_k3_vs_k2, 3: paso3_hipercubo}

    if args.paso:
        pasos[args.paso]()
    else:
        for fn in pasos.values():
            fn()

    print(f"\n{SEP}\n  Demo finalizado.\n{SEP}\n")


if __name__ == "__main__":
    main()

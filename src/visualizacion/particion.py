"""Visualizacion de particiones sobre grafos de nodos."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

# Paleta de colores para hasta 8 grupos.
_COLORES_GRUPOS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]


def _color_grupo(indice: int) -> str:
    return _COLORES_GRUPOS[indice % len(_COLORES_GRUPOS)]


def dibujar_biparticion(
    subalcance: tuple[int, ...],
    submecanismo: tuple[int, ...],
    alcance_total: tuple[int, ...],
    mecanismo_total: tuple[int, ...],
    perdida: float,
    titulo: str = "Biparticion MIP",
    guardar_en: str | Path | None = None,
) -> None:
    """Dibuja la biparticion como grafo bipartito mecanismo -> alcance.

    Los nodos del Grupo A (preservados) aparecen en azul y los del Grupo B
    (complemento) en naranja. Las aristas muestran las conexiones que la
    particion corta (linea discontinua) frente a las que conserva (continua).
    """
    G = nx.DiGraph()

    nodos_mec = [f"m{i}" for i in mecanismo_total]
    nodos_alc = [f"a{i}" for i in alcance_total]
    G.add_nodes_from(nodos_mec, capa=0)
    G.add_nodes_from(nodos_alc, capa=1)

    aristas_conservadas = []
    aristas_cortadas = []
    for i in mecanismo_total:
        for j in alcance_total:
            if i in submecanismo and j in subalcance:
                aristas_conservadas.append((f"m{i}", f"a{j}"))
            else:
                aristas_cortadas.append((f"m{i}", f"a{j}"))

    G.add_edges_from(aristas_conservadas + aristas_cortadas)

    pos = nx.bipartite_layout(G, nodes=nodos_mec)

    colores_nodos = []
    for nodo in G.nodes():
        idx = int(nodo[1:])
        tiempo = 0 if nodo.startswith("m") else 1
        en_grupo_a = (tiempo == 0 and idx in submecanismo) or (tiempo == 1 and idx in subalcance)
        colores_nodos.append(_COLORES_GRUPOS[0] if en_grupo_a else _COLORES_GRUPOS[1])

    fig, ax = plt.subplots(figsize=(8, 5))
    nx.draw_networkx_nodes(G, pos, node_color=colores_nodos, node_size=700, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax, font_color="white", font_weight="bold")
    nx.draw_networkx_edges(
        G, pos, edgelist=aristas_conservadas,
        edge_color="#2d2d2d", width=2.0, arrows=True, ax=ax,
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=aristas_cortadas,
        edge_color="#cccccc", width=1.0, style="dashed", arrows=True, ax=ax,
    )

    parche_a = mpatches.Patch(color=_COLORES_GRUPOS[0], label="Grupo A (preservado)")
    parche_b = mpatches.Patch(color=_COLORES_GRUPOS[1], label="Grupo B (complemento)")
    ax.legend(handles=[parche_a, parche_b], loc="upper right")
    ax.set_title(f"{titulo}\nPerdida φ = {perdida:.4f}")
    ax.axis("off")

    plt.tight_layout()
    if guardar_en:
        plt.savefig(guardar_en, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def dibujar_k_particion(
    nodos: list[int],
    asignacion: tuple[int, ...],
    alcance_total: tuple[int, ...],
    mecanismo_total: tuple[int, ...],
    perdida: float,
    titulo: str = "K-Particion",
    guardar_en: str | Path | None = None,
) -> None:
    """Dibuja la k-particion como grafo con nodos coloreados por grupo.

    Cada grupo recibe un color distinto. Las aristas intragrupales se dibujan
    solidas y las intergrupales punteadas, ilustrando lo que la particion corta.
    """
    if not nodos or not asignacion:
        return

    nodo_a_grupo = {nodos[i]: asignacion[i] for i in range(len(nodos))}
    k = max(asignacion) + 1

    G = nx.DiGraph()
    for nodo in nodos:
        G.add_node(nodo)

    aristas_intra = []
    aristas_inter = []
    for src in mecanismo_total:
        for dst in alcance_total:
            if nodo_a_grupo.get(src) == nodo_a_grupo.get(dst):
                aristas_intra.append((src, dst))
            else:
                aristas_inter.append((src, dst))

    G.add_edges_from(aristas_intra + aristas_inter)

    try:
        pos = nx.spring_layout(G, seed=42)
    except Exception:
        pos = {n: (i, 0) for i, n in enumerate(nodos)}

    colores_nodos = [_color_grupo(nodo_a_grupo.get(n, 0)) for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(7, 5))
    nx.draw_networkx_nodes(G, pos, node_color=colores_nodos, node_size=700, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax, font_color="white", font_weight="bold")
    nx.draw_networkx_edges(
        G, pos, edgelist=aristas_intra,
        edge_color="#333333", width=2.0, arrows=True, ax=ax,
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=aristas_inter,
        edge_color="#bbbbbb", width=1.0, style="dashed", arrows=True, ax=ax,
    )

    parches = [
        mpatches.Patch(color=_color_grupo(g), label=f"Grupo {g}")
        for g in range(k)
    ]
    ax.legend(handles=parches, loc="upper right")
    ax.set_title(f"{titulo}\nPerdida φ = {perdida:.4f}")
    ax.axis("off")

    plt.tight_layout()
    if guardar_en:
        plt.savefig(guardar_en, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def dibujar_comparacion_perdidas(
    resultados: dict[str, float],
    titulo: str = "Comparacion de perdidas por estrategia",
    guardar_en: str | Path | None = None,
) -> None:
    """Grafica de barras comparando la perdida φ de cada estrategia."""
    estrategias = list(resultados.keys())
    perdidas = [resultados[e] for e in estrategias]
    colores = [_color_grupo(i) for i in range(len(estrategias))]

    fig, ax = plt.subplots(figsize=(max(6, len(estrategias) * 1.4), 4))
    barras = ax.bar(estrategias, perdidas, color=colores, edgecolor="white", linewidth=0.8)

    for barra, valor in zip(barras, perdidas):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + 0.002,
            f"{valor:.4f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylabel("Perdida φ")
    ax.set_title(titulo)
    ax.set_ylim(0, max(perdidas) * 1.2 if perdidas else 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    if guardar_en:
        plt.savefig(guardar_en, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

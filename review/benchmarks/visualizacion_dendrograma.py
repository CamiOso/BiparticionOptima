"""Visualizacion del dendrograma divisivo de la estrategia GeometricK.

Construye y dibuja el arbol de cortes que GeometricK usa para inicializar
la busqueda k-particion. El dendrograma muestra como el sistema se divide
recursivamente segun el corte de menor EMD en cada componente.

Salida:
  - Terminal: arbol ASCII con costo de cada corte
  - Archivo:  review/benchmarks/dendrograma_geometric.png

Para reproducir:
    source .venv/bin/activate
    PYTHONPATH=. python review/benchmarks/visualizacion_dendrograma.py
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sin ventana de GUI
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.modelos.base.sia import SIA
from src.strategies.geometric import Geometric


# ---------------------------------------------------------------------------
# Subclase que registra el arbol del dendrograma completo
# ---------------------------------------------------------------------------

@dataclass
class NodoDendrograma:
    id: int
    nodos: frozenset[int]
    costo: float          # costo del corte que produjo este nodo (0 para la raiz)
    padre_id: int | None
    hijo_izq_id: int | None = None
    hijo_der_id: int | None = None

    def es_hoja(self) -> bool:
        return self.hijo_izq_id is None and self.hijo_der_id is None


class _GeometricConArbol(Geometric):
    """Geometric que registra el arbol completo al calcular el dendrograma."""

    def construir_arbol_completo(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ) -> dict[int, NodoDendrograma]:
        """Prepara el subsistema y construye el dendrograma hasta hojas singleton."""
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)
        assert self.sia_subsistema is not None
        assert self.sia_dists_marginales is not None

        self._cache_particiones.clear()
        self._cache_k_particiones.clear()
        _ = self._tpm_a_tensores_elementales()

        alcance_total = tuple(int(v) for v in self.sia_subsistema.indices_ncubos.tolist())
        mecanismo_total = tuple(int(v) for v in self.sia_subsistema.dims_ncubos.tolist())
        nodos = sorted(set(alcance_total) | set(mecanismo_total))

        arbol: dict[int, NodoDendrograma] = {}
        id_cnt = 0

        comp_raiz = frozenset(nodos)
        nodo_raiz = NodoDendrograma(id=id_cnt, nodos=comp_raiz, costo=0.0, padre_id=None)
        arbol[id_cnt] = nodo_raiz
        id_cnt += 1

        split = self._bipartir_componente(list(comp_raiz), alcance_total, mecanismo_total)
        if split is None:
            return arbol

        splits_info: dict[int, tuple[frozenset, frozenset, float, int]] = {}
        heap: list[tuple[float, int]] = []
        comp_a_id: dict[frozenset, int] = {comp_raiz: 0}

        split_id = id_cnt
        splits_info[split_id] = (*split, 0)  # (izq, der, costo, padre_id)
        heapq.heappush(heap, (split[2], split_id))
        id_cnt += 1

        hojas: set[frozenset] = {comp_raiz}

        while heap:
            _, eid = heapq.heappop(heap)
            if eid not in splits_info:
                continue
            izq, der, costo, padre_nodo_id = splits_info.pop(eid)

            padre_comp = izq | der
            if padre_comp not in hojas:
                continue

            hojas.discard(padre_comp)

            nodo_izq = NodoDendrograma(
                id=id_cnt, nodos=izq, costo=costo, padre_id=padre_nodo_id
            )
            arbol[id_cnt] = nodo_izq
            comp_a_id[izq] = id_cnt
            arbol[padre_nodo_id].hijo_izq_id = id_cnt
            id_cnt += 1

            nodo_der = NodoDendrograma(
                id=id_cnt, nodos=der, costo=costo, padre_id=padre_nodo_id
            )
            arbol[id_cnt] = nodo_der
            comp_a_id[der] = id_cnt
            arbol[padre_nodo_id].hijo_der_id = id_cnt
            id_cnt += 1

            hojas.add(izq)
            hojas.add(der)

            for hijo_nodo, hijo_comp in [(nodo_izq, izq), (nodo_der, der)]:
                if len(hijo_comp) > 1:
                    s = self._bipartir_componente(list(hijo_comp), alcance_total, mecanismo_total)
                    if s is not None:
                        sid = id_cnt
                        splits_info[sid] = (*s, hijo_nodo.id)
                        heapq.heappush(heap, (s[2], sid))
                        id_cnt += 1

        return arbol


# ---------------------------------------------------------------------------
# Impresion ASCII del arbol
# ---------------------------------------------------------------------------

def _imprimir_arbol_ascii(arbol: dict[int, NodoDendrograma], nodo_id: int = 0, prefijo: str = "", es_ultimo: bool = True) -> None:
    nodo = arbol[nodo_id]
    conector = "└── " if es_ultimo else "├── "
    nodos_str = "{" + ",".join(str(n) for n in sorted(nodo.nodos)) + "}"
    costo_str = f"  φ={nodo.costo:.4f}" if nodo.padre_id is not None else ""
    print(f"{prefijo}{conector}{nodos_str}{costo_str}")

    hijos = []
    if nodo.hijo_izq_id is not None:
        hijos.append(nodo.hijo_izq_id)
    if nodo.hijo_der_id is not None:
        hijos.append(nodo.hijo_der_id)

    nuevo_prefijo = prefijo + ("    " if es_ultimo else "│   ")
    for i, hijo_id in enumerate(hijos):
        _imprimir_arbol_ascii(arbol, hijo_id, nuevo_prefijo, i == len(hijos) - 1)


# ---------------------------------------------------------------------------
# Layout y dibujo matplotlib
# ---------------------------------------------------------------------------

def _calcular_posiciones(
    arbol: dict[int, NodoDendrograma],
    nodo_id: int,
) -> dict[int, tuple[float, float]]:
    """Asigna posicion (x, y) a cada nodo. Las hojas en y=0, la raiz en y=max_depth."""
    posiciones: dict[int, tuple[float, float]] = {}
    hoja_x: list[float] = []

    def _asignar_hojas(nid: int) -> list[int]:
        nodo = arbol[nid]
        if nodo.es_hoja():
            x = float(len(hoja_x))
            hoja_x.append(x)
            posiciones[nid] = (x, 0.0)
            return [nid]
        hijos_izq = _asignar_hojas(nodo.hijo_izq_id) if nodo.hijo_izq_id is not None else []
        hijos_der = _asignar_hojas(nodo.hijo_der_id) if nodo.hijo_der_id is not None else []
        return hijos_izq + hijos_der

    _asignar_hojas(nodo_id)

    def _asignar_internos(nid: int, profundidad: int) -> None:
        nodo = arbol[nid]
        if nodo.es_hoja():
            return
        xs_hijos = []
        for hijo_id in [nodo.hijo_izq_id, nodo.hijo_der_id]:
            if hijo_id is not None:
                _asignar_internos(hijo_id, profundidad + 1)
                xs_hijos.append(posiciones[hijo_id][0])
        if xs_hijos:
            posiciones[nid] = (sum(xs_hijos) / len(xs_hijos), float(profundidad))

    # Calcular profundidad maxima
    def _max_depth(nid: int, d: int) -> int:
        nodo = arbol[nid]
        if nodo.es_hoja():
            return d
        hijos_d = []
        for hijo_id in [nodo.hijo_izq_id, nodo.hijo_der_id]:
            if hijo_id is not None:
                hijos_d.append(_max_depth(hijo_id, d + 1))
        return max(hijos_d) if hijos_d else d

    max_d = _max_depth(nodo_id, 0)

    # Invertir la raiz al top (y=max_depth, hojas en y=0)
    _asignar_internos(nodo_id, 0)
    for nid in posiciones:
        x, y = posiciones[nid]
        posiciones[nid] = (x, max_d - y)

    return posiciones


def _dibujar_dendrograma(
    arbol: dict[int, NodoDendrograma],
    titulo: str,
    ruta_salida: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    posiciones = _calcular_posiciones(arbol, 0)

    # Dibujar aristas
    def _dibujar_aristas(nid: int) -> None:
        nodo = arbol[nid]
        x_padre, y_padre = posiciones[nid]
        for hijo_id in [nodo.hijo_izq_id, nodo.hijo_der_id]:
            if hijo_id is not None:
                x_hijo, y_hijo = posiciones[hijo_id]
                ax.plot([x_padre, x_hijo], [y_padre, y_hijo],
                        color="#888888", linewidth=1.5, zorder=1)
                _dibujar_aristas(hijo_id)

    _dibujar_aristas(0)

    # Dibujar nodos
    for nid, (x, y) in posiciones.items():
        nodo = arbol[nid]
        es_hoja = nodo.es_hoja()
        color = "#2196F3" if not es_hoja else "#4CAF50"
        size = 400 if not es_hoja else 250
        ax.scatter([x], [y], s=size, c=color, zorder=3)

        nodos_str = "{" + ",".join(str(n) for n in sorted(nodo.nodos)) + "}"
        offset_y = 0.12 if not es_hoja else -0.15
        ax.text(x, y + offset_y, nodos_str, ha="center", va="center",
                fontsize=8, fontweight="bold")

        if nodo.costo > 0:
            xp, yp = posiciones[nodo.padre_id]
            mx, my = (x + xp) / 2, (y + yp) / 2
            ax.text(mx + 0.1, my, f"φ={nodo.costo:.3f}",
                    ha="left", va="center", fontsize=7, color="#D32F2F")

    ax.set_title(titulo, fontsize=12, pad=15)
    ax.set_ylabel("Profundidad del corte")
    ax.set_xticks([])
    max_y = max(y for _, y in posiciones.values())
    ax.set_ylim(-0.5, max_y + 0.5)

    parche_interno = mpatches.Patch(color="#2196F3", label="Nodo interno (componente)")
    parche_hoja = mpatches.Patch(color="#4CAF50", label="Hoja (grupo final)")
    ax.legend(handles=[parche_interno, parche_hoja], loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nDendrograma guardado en: {ruta_salida}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    n = 4
    semilla = 37  # Semilla con sistema interesante
    tpm = np.random.default_rng(semilla).random((1 << n, n), dtype=np.float32)

    geo = _GeometricConArbol(tpm)
    estado = "0" * n
    mascara = "1" * n

    arbol = geo.construir_arbol_completo(estado, mascara, mascara, mascara)

    print("=" * 60)
    print(f"Dendrograma divisivo — GeometricK (n={n}, semilla={semilla})")
    print("=" * 60)
    print(f"Total de nodos en el arbol: {len(arbol)}")
    print(f"Hojas: {sum(1 for n in arbol.values() if n.es_hoja())}")
    print()
    print("Arbol ASCII:")
    _imprimir_arbol_ascii(arbol)

    ruta = Path("review/benchmarks/dendrograma_geometric.png")
    titulo = f"Dendrograma GeometricK — n={n}, semilla={semilla}"
    _dibujar_dendrograma(arbol, titulo, ruta)

    # Mostrar para diferentes k la particion resultante
    print("\nParticiones para distintos k:")
    from src.strategies.geometric import Geometric
    for k in range(2, n + 1):
        geo2 = Geometric(tpm)
        res = geo2.aplicar_estrategia(estado, mascara, mascara, mascara, k=k)
        print(f"  k={k}: phi={res.perdida:.6f}  particion={res.particion}")


if __name__ == "__main__":
    main()

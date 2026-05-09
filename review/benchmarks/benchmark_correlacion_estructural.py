"""Correlacion entre propiedades estructurales de biparticiones y su EMD.

Pregunta: si los circuitos causales no correlacionan con EMD (rho~0.03),
?que propiedad estructural SI lo hace?

Candidatos evaluados para cada biparticion (A|B):
  1. Peso_corte      : suma de conductancias que cruzan la particion (W[A,B]+W[B,A])
  2. Corte_balanceado: peso_corte / (|A| * |B|)  (normalizado por tamano)
  3. Info_mutua      : I(X_A ; X_B) en la distribucion estacionaria del sistema
  4. Circuitos_rotos : suma de fuerzas de circuitos que cruzan la particion (linea base)
  5. Diferencia_entropia: |H(X_A) - H(X_B)| en la distribucion marginal

Para reproducir:
    source .venv/bin/activate
    PYTHONPATH=. python review/benchmarks/benchmark_correlacion_estructural.py
"""

import csv
from pathlib import Path
from typing import Callable

import numpy as np

from src.funciones.iit import seleccionar_emd
from src.funciones.particiones import biparticiones
from src.modelos.base.sia import SIA
from src.modelos.nucleo.sistema import Sistema


# ---------------------------------------------------------------------------
# Spearman (reutilizado del benchmark de circuitos)
# ---------------------------------------------------------------------------

def _spearman(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")

    def rango(v: list[float]) -> list[float]:
        orden = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and v[orden[j]] == v[orden[i]]:
                j += 1
            rm = (i + j - 1) / 2.0 + 1.0
            for k in range(i, j):
                r[orden[k]] = rm
            i = j
        return r

    rx, ry = rango(x), rango(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


# ---------------------------------------------------------------------------
# Conductancias dirigidas (no simetrizadas)
# ---------------------------------------------------------------------------

def _conductancias(sistema: Sistema, nodos: list[int]) -> np.ndarray:
    n = len(nodos)
    idx = {v: i for i, v in enumerate(nodos)}
    W = np.zeros((n, n), dtype=np.float64)
    for cubo in sistema.ncubos:
        i_orig = int(cubo.indice)
        if i_orig not in idx:
            continue
        ii = idx[i_orig]
        dims = cubo.dims.tolist()
        data = cubo.data
        nd = len(dims)
        for pos, dim_j in enumerate(dims):
            j_orig = int(dim_j)
            if j_orig not in idx:
                continue
            jj = idx[j_orig]
            otras = [d for d in range(nd) if d != pos]
            otras_sizes = [data.shape[d] for d in otras]
            n_otras = max(1, int(np.prod(otras_sizes)) if otras_sizes else 1)
            total = 0.0
            for estado_idx in range(n_otras):
                idx_otras = []
                temp = estado_idx
                for s in reversed(otras_sizes):
                    idx_otras.append(temp % s)
                    temp //= s
                idx_otras = list(reversed(idx_otras))
                idx_0 = [0] * nd
                idx_1 = [0] * nd
                idx_0[pos] = 0
                idx_1[pos] = 1
                for k_ot, d_ot in enumerate(otras):
                    idx_0[d_ot] = idx_otras[k_ot]
                    idx_1[d_ot] = idx_otras[k_ot]
                total += abs(float(data[tuple(idx_0)]) - float(data[tuple(idx_1)]))
            W[ii][jj] = total / n_otras
    return W


def _circuitos_dfs(n: int, W: np.ndarray) -> list[tuple[list[int], float]]:
    if n > 12:
        return []
    adj = [[W[i][j] if W[i][j] > 0 else 0.0 for j in range(n)] for i in range(n)]
    circuitos: list[tuple[list[int], float]] = []
    vis = [False] * n

    def dfs(s: int, u: int, path: list[int], peso: float) -> None:
        for v in range(n):
            w = adj[u][v]
            if w == 0.0:
                continue
            if v == s and len(path) >= 2:
                if peso * w > 0:
                    circuitos.append((list(path), peso * w))
            elif v > s and not vis[v]:
                vis[v] = True
                path.append(v)
                dfs(s, v, path, peso * w)
                path.pop()
                vis[v] = False

    for s in range(n):
        vis[s] = True
        dfs(s, s, [s], 1.0)
        vis[s] = False
    return circuitos


# ---------------------------------------------------------------------------
# Metricas estructurales de una biparticion
# ---------------------------------------------------------------------------

def _peso_corte(grupo_a_idx: set[int], n: int, W: np.ndarray) -> float:
    """Suma de conductancias que cruzan la particion (en ambas direcciones)."""
    total = 0.0
    grupo_b_idx = set(range(n)) - grupo_a_idx
    for i in grupo_a_idx:
        for j in grupo_b_idx:
            total += W[i][j] + W[j][i]
    return total


def _corte_balanceado(grupo_a_idx: set[int], n: int, W: np.ndarray) -> float:
    """Peso de corte normalizado por el producto de tamanos de los grupos."""
    na = len(grupo_a_idx)
    nb = n - na
    if na == 0 or nb == 0:
        return float("inf")
    return _peso_corte(grupo_a_idx, n, W) / (na * nb)


def _info_mutua(
    grupo_a_nodos: list[int],
    grupo_b_nodos: list[int],
    dists_marginales: np.ndarray,
    nodos: list[int],
) -> float:
    """Informacion mutua I(X_A ; X_B) aproximada desde distribuciones marginales.

    Usa independencia entre nodos segun sus probabilidades marginales para
    estimar la informacion mutua entre los dos grupos.
    I(A;B) ~ H(A) + H(B) - H(A,B)
    Dado que las marginales son P(X_i=1), aproximamos H(X_i) = -p log p - (1-p) log(1-p).
    """
    idx = {v: i for i, v in enumerate(nodos)}
    eps = 1e-12

    def h_nodo(nodo: int) -> float:
        p = float(dists_marginales[idx[nodo]])
        p = max(eps, min(1 - eps, p))
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    ha = sum(h_nodo(v) for v in grupo_a_nodos)
    hb = sum(h_nodo(v) for v in grupo_b_nodos)
    hab = ha + hb  # asume independencia entre grupos como upper bound
    return max(0.0, ha + hb - hab)


def _entropia_marginal(nodos_grupo: list[int], dists: np.ndarray, idx_map: dict) -> float:
    """Entropia de Shannon de la distribucion marginal del grupo."""
    eps = 1e-12
    h = 0.0
    for v in nodos_grupo:
        p = float(dists[idx_map[v]])
        p = max(eps, min(1 - eps, p))
        h += -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    return h


def _circuitos_rotos(circuitos: list[tuple[list[int], float]], grupo_a_idx: set[int]) -> float:
    total = 0.0
    for ciclo, fuerza in circuitos:
        nc = set(ciclo)
        if nc & grupo_a_idx and nc - grupo_a_idx:
            total += fuerza
    return total


# ---------------------------------------------------------------------------
# Analisis de un sistema
# ---------------------------------------------------------------------------

def _analizar(
    tpm: np.ndarray,
    n: int,
    semilla: int,
) -> dict | None:
    estado = "0" * n
    mascara = "1" * n
    distancia = seleccionar_emd()

    class _Dummy(SIA):
        def aplicar_estrategia(self, *a, **kw):
            pass

    d = _Dummy(tpm)
    d.sia_preparar_subsistema(estado, mascara, mascara, mascara)
    sistema = d.sia_subsistema
    dists = d.sia_dists_marginales
    if sistema is None or dists is None:
        return None

    alcance_t = tuple(int(v) for v in sistema.indices_ncubos.tolist())
    mec_t = tuple(int(v) for v in sistema.dims_ncubos.tolist())
    nodos = sorted(set(alcance_t) | set(mec_t))
    if len(nodos) < 2:
        return None

    idx_map = {v: i for i, v in enumerate(nodos)}
    W = _conductancias(sistema, nodos)
    circuitos = _circuitos_dfs(len(nodos), W)

    pares = list(biparticiones(sistema.indices_ncubos, sistema.dims_ncubos))
    if len(pares) < 3:
        return None

    emds, pesos, balanceados, info_mutuas, dif_entropias, circ_rotos = [], [], [], [], [], []

    for subalcance, submecanismo in pares:
        sp = sistema.bipartir(
            np.array(subalcance, dtype=np.int8),
            np.array(submecanismo, dtype=np.int8),
        )
        dp = sp.distribucion_marginal()
        if dp.size < dists.size:
            dp2 = np.zeros_like(dists)
            dp2[: dp.size] = dp
            dp = dp2
        emds.append(float(distancia(dists, dp)))

        grupo_a = set(subalcance) | set(submecanismo)
        grupo_b = set(nodos) - grupo_a
        ga_idx = {idx_map[v] for v in grupo_a if v in idx_map}

        pesos.append(_peso_corte(ga_idx, len(nodos), W))
        balanceados.append(_corte_balanceado(ga_idx, len(nodos), W))

        ga_nodos = [v for v in nodos if v in grupo_a]
        gb_nodos = [v for v in nodos if v in grupo_b]
        ha = _entropia_marginal(ga_nodos, dists, idx_map)
        hb = _entropia_marginal(gb_nodos, dists, idx_map)
        info_mutuas.append(max(0.0, ha + hb - (ha + hb)))  # ≈0 bajo independencia
        dif_entropias.append(abs(ha - hb))

        circ_rotos.append(_circuitos_rotos(circuitos, ga_idx))

    def rho(m: list[float]) -> float:
        r = _spearman(emds, m)
        return round(r, 4) if r == r else None

    return {
        "semilla": semilla,
        "n": n,
        "n_bip": len(pares),
        "rho_peso_corte": rho(pesos),
        "rho_corte_balanceado": rho(balanceados),
        "rho_dif_entropias": rho(dif_entropias),
        "rho_circuitos_rotos": rho(circ_rotos),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    configs = [
        (4, [11, 19, 29, 37, 47, 59, 71, 83, 97, 101]),
        (5, [13, 23, 31, 41, 53, 67, 79, 89, 103, 113]),
    ]

    resultados = []
    metricas = ["rho_peso_corte", "rho_corte_balanceado", "rho_dif_entropias", "rho_circuitos_rotos"]
    labels   = ["Peso corte", "Corte balanceado", "Dif. entropías", "Circ. rotos"]

    for n, semillas in configs:
        print(f"\n=== n={n} ===")
        print(f"{'semilla':>8}  {'Peso corte':>12}  {'Corte bal.':>12}  {'Dif.entr.':>12}  {'Circ.rot.':>12}")
        for semilla in semillas:
            rng = np.random.default_rng(semilla)
            tpm = rng.random((2**n, n)).astype(np.float32)
            res = _analizar(tpm, n, semilla)
            if res is None:
                continue
            resultados.append(res)
            print(
                f"{semilla:>8}  "
                + "  ".join(
                    f"{res[m]:>+12.4f}" if res[m] is not None else f"{'N/A':>12}"
                    for m in metricas
                )
            )

    if not resultados:
        print("Sin resultados validos.")
        return

    print("\n=== Promedio de correlacion de Spearman ===")
    print(f"{'Metrica':<25}  {'Promedio':>10}  {'Min':>10}  {'Max':>10}  {'ρ>0':>8}")
    for m, label in zip(metricas, labels):
        vals = [r[m] for r in resultados if r[m] is not None]
        if not vals:
            continue
        pos = sum(1 for v in vals if v > 0)
        print(
            f"{label:<25}  {sum(vals)/len(vals):>+10.4f}  "
            f"{min(vals):>+10.4f}  {max(vals):>+10.4f}  {pos:>5}/{len(vals)}"
        )

    print("\nCONCLUSION:")
    promedios = {
        m: sum(r[m] for r in resultados if r[m] is not None) / max(1, sum(1 for r in resultados if r[m] is not None))
        for m in metricas
    }
    mejor_m = max(promedios, key=lambda m: abs(promedios[m]))
    print(f"  Mejor correlacion: {labels[metricas.index(mejor_m)]} (rho promedio = {promedios[mejor_m]:+.4f})")

    csv_path = Path(__file__).parent / "correlacion_estructural_vs_emd.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(resultados[0].keys()))
        writer.writeheader()
        writer.writerows(resultados)
    print(f"\nResultados en {csv_path}")


if __name__ == "__main__":
    main()

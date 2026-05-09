"""Validacion de la metrica de circuitos rotos como proxy del EMD.

Hipotesis: la suma de fuerzas de los circuitos causales rotos por una
biparticion correlaciona con el EMD de esa misma biparticion. Si la
correlacion de Spearman es alta, la metrica de circuitos puede servir
para guiar la busqueda de la MIP sin calcular EMD completo.

Metodologia:
  1. Generar sistemas aleatorios de n=4 y n=5 nodos.
  2. Enumerar TODAS las biparticiones con sus EMDs (referencia exacta).
  3. Para cada sistema, construir el grafo causal y encontrar todos los
     circuitos con un DFS recursivo (equivalente simplificado de Johnson).
  4. Para cada biparticion, calcular la metrica: suma de fuerzas de los
     circuitos cuyos nodos quedan en grupos distintos.
  5. Calcular la correlacion de Spearman entre ranking EMD y ranking circuitos.

Para reproducir:
    source .venv/bin/activate
    PYTHONPATH=. python review/benchmarks/benchmark_metrica_circuitos.py
"""

import csv
from pathlib import Path

import numpy as np

from src.funciones.iit import seleccionar_emd
from src.funciones.particiones import biparticiones
from src.modelos.base.sia import SIA
from src.modelos.nucleo.sistema import Sistema


# ---------------------------------------------------------------------------
# Encontrar circuitos en un grafo dirigido ponderado
# ---------------------------------------------------------------------------

def _encontrar_circuitos_dfs(
    nodos: list[int],
    W: np.ndarray,
    umbral: float = 0.0,
) -> list[tuple[list[int], float]]:
    """Todos los circuitos simples del grafo dirigido W > umbral.

    Retorna lista de (ciclo_como_lista_de_indices, fuerza). La fuerza es el
    producto de los pesos de las aristas a lo largo del ciclo. Usa DFS con
    backtracking; limitado a n<=8 para que sea tractable.
    """
    n = len(nodos)
    adj = [[W[i][j] if W[i][j] > umbral else 0.0 for j in range(n)] for i in range(n)]

    circuitos: list[tuple[list[int], float]] = []
    visitado = [False] * n

    def dfs(inicio: int, actual: int, camino: list[int], peso_acum: float) -> None:
        for vecino in range(n):
            w = adj[actual][vecino]
            if w == 0.0:
                continue
            if vecino == inicio and len(camino) >= 2:
                fuerza = peso_acum * w
                if fuerza > 0.0:
                    circuitos.append((list(camino), fuerza))
            elif vecino > inicio and not visitado[vecino]:
                visitado[vecino] = True
                camino.append(vecino)
                dfs(inicio, vecino, camino, peso_acum * w)
                camino.pop()
                visitado[vecino] = False

    for s in range(n):
        visitado[s] = True
        dfs(s, s, [s], 1.0)
        visitado[s] = False

    return circuitos


# ---------------------------------------------------------------------------
# Metrica de circuitos rotos para una biparticion
# ---------------------------------------------------------------------------

def _circuitos_rotos(
    circuitos: list[tuple[list[int], float]],
    grupo_a: set[int],
) -> float:
    """Suma de fuerzas de circuitos que tienen nodos en ambos grupos."""
    total = 0.0
    for ciclo, fuerza in circuitos:
        nodos_ciclo = set(ciclo)
        if nodos_ciclo & grupo_a and nodos_ciclo - grupo_a:
            total += fuerza
    return total


# ---------------------------------------------------------------------------
# Correlacion de Spearman
# ---------------------------------------------------------------------------

def _spearman(x: list[float], y: list[float]) -> float:
    """Coeficiente de correlacion de Spearman entre dos listas."""
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
            rango_medio = (i + j - 1) / 2.0 + 1.0
            for k in range(i, j):
                r[orden[k]] = rango_medio
            i = j
        return r

    rx = rango(x)
    ry = rango(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den_x = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    den_y = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


# ---------------------------------------------------------------------------
# Construir conductancias dirigidas (no simetrizadas) para el grafo causal
# ---------------------------------------------------------------------------

def _conductancias_dirigidas(sistema: Sistema, nodos: list[int]) -> np.ndarray:
    """W[i][j] = sensibilidad del nodo i al estado del nodo j (dirigida, no simetrica)."""
    n = len(nodos)
    idx = {nodo: i for i, nodo in enumerate(nodos)}
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
            n_otras = 1
            for s in otras_sizes:
                n_otras *= s

            total = 0.0
            for estado_idx in range(max(1, n_otras)):
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

            W[ii][jj] = total / max(1, n_otras)

    return W


# ---------------------------------------------------------------------------
# Una corrida para un sistema dado
# ---------------------------------------------------------------------------

def _analizar_sistema(
    tpm: np.ndarray,
    estado_inicial: str,
    condicion: str,
    alcance: str,
    mecanismo: str,
    semilla_id: int,
) -> dict | None:
    distancia = seleccionar_emd()

    class _Dummy(SIA):
        def aplicar_estrategia(self, *a, **kw):
            pass

    dummy = _Dummy(tpm)
    dummy.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)
    sistema = dummy.sia_subsistema
    dists = dummy.sia_dists_marginales

    if sistema is None or dists is None:
        return None

    alcance_t = tuple(int(v) for v in sistema.indices_ncubos.tolist())
    mec_t = tuple(int(v) for v in sistema.dims_ncubos.tolist())
    nodos = sorted(set(alcance_t) | set(mec_t))

    if len(nodos) < 2:
        return None

    # Todas las biparticiones con su EMD
    pares: list[tuple[tuple, tuple]] = list(biparticiones(sistema.indices_ncubos, sistema.dims_ncubos))
    if len(pares) < 3:
        return None

    emds: list[float] = []
    grupo_a_list: list[set[int]] = []

    for subalcance, submecanismo in pares:
        sistema_partido = sistema.bipartir(
            np.array(subalcance, dtype=np.int8),
            np.array(submecanismo, dtype=np.int8),
        )
        dist_part = sistema_partido.distribucion_marginal()
        if dist_part.size < dists.size:
            dist_part2 = np.zeros_like(dists)
            dist_part2[: dist_part.size] = dist_part
            dist_part = dist_part2
        emd = float(distancia(dists, dist_part))
        emds.append(emd)
        grupo_a_list.append(set(subalcance) | set(submecanismo))

    # Grafo causal y circuitos
    idx_nodos = {n: i for i, n in enumerate(nodos)}
    W_dir = _conductancias_dirigidas(sistema, nodos)
    circuitos = _encontrar_circuitos_dfs(nodos, W_dir, umbral=0.0)

    if not circuitos:
        return None

    # Metrica de circuitos rotos para cada biparticion
    metricas: list[float] = []
    for grupo_a in grupo_a_list:
        grupo_a_idx = {idx_nodos[n] for n in grupo_a if n in idx_nodos}
        m = _circuitos_rotos(circuitos, grupo_a_idx)
        metricas.append(m)

    rho = _spearman(emds, metricas)

    return {
        "semilla": semilla_id,
        "n_nodos": len(nodos),
        "n_biparticiones": len(pares),
        "n_circuitos": len(circuitos),
        "spearman_rho": round(rho, 4) if not (rho != rho) else None,
        "emd_min": round(min(emds), 4),
        "emd_max": round(max(emds), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    configs = [
        {"n": 4, "semillas": [11, 19, 29, 37, 47, 59, 71, 83, 97, 101]},
        {"n": 5, "semillas": [13, 23, 31, 41, 53, 67, 79, 89, 103, 113]},
    ]

    resultados: list[dict] = []

    for cfg in configs:
        n = cfg["n"]
        estado = "0" * n
        mascara = "1" * n

        for semilla in cfg["semillas"]:
            rng = np.random.default_rng(semilla)
            tpm = rng.random((2**n, n)).astype(np.float32)

            res = _analizar_sistema(
                tpm=tpm,
                estado_inicial=estado,
                condicion=mascara,
                alcance=mascara,
                mecanismo=mascara,
                semilla_id=semilla,
            )
            if res is not None:
                resultados.append(res)
                rho = res["spearman_rho"]
                rho_str = f"{rho:+.4f}" if rho is not None else "  N/A "
                print(
                    f"n={res['n_nodos']}  semilla={semilla:4d}  "
                    f"circuitos={res['n_circuitos']:3d}  "
                    f"biparticiones={res['n_biparticiones']:3d}  "
                    f"rho={rho_str}"
                )
            else:
                print(f"n={n}  semilla={semilla:4d}  [sistema sin circuitos o biparticiones]")

    if not resultados:
        print("\nSin resultados validos.")
        return

    rhos = [r["spearman_rho"] for r in resultados if r["spearman_rho"] is not None]
    print(f"\n--- Resumen ---")
    print(f"Sistemas analizados: {len(resultados)}")
    print(f"Con correlacion valida: {len(rhos)}")
    if rhos:
        print(f"Spearman promedio:  {sum(rhos)/len(rhos):+.4f}")
        print(f"Spearman minimo:    {min(rhos):+.4f}")
        print(f"Spearman maximo:    {max(rhos):+.4f}")

        positivos = sum(1 for r in rhos if r > 0)
        print(f"Correlacion positiva (rho > 0): {positivos}/{len(rhos)}")
        print()
        if sum(rhos) / len(rhos) > 0.4:
            print("CONCLUSION: la metrica de circuitos rotos correlaciona con EMD.")
            print("Puede usarse como proxy para guiar la busqueda de la MIP.")
        else:
            print("CONCLUSION: correlacion debil. La metrica de circuitos no es")
            print("un buen proxy para EMD en estos sistemas aleatorios.")

    csv_path = Path(__file__).parent / "metrica_circuitos_vs_emd.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(resultados[0].keys()))
        writer.writeheader()
        writer.writerows(resultados)
    print(f"\nResultados guardados en {csv_path}")


if __name__ == "__main__":
    main()

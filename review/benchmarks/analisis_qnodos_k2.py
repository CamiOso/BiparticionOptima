"""Analisis de la falla de QNodos para k=2: submodularidad del EMD.

Hipotesis: la brecha de QNodos (155% en n=4, k=2) se debe a que la funcion
f(S) = EMD(dist_completa, dist_biparticion_S) NO es submodular en general.
El algoritmo de Queyranne solo garantiza el minimo para funciones simetrico-
submodulares; si f viola submodularidad, el MAO puede converger a un minimo
local que no es el global.

Metodologia:
  1. Para cada semilla con n=4: enumerar todas las biparticiones.
  2. Medir f(S) para cada subconjunto S de vertices.
  3. Contar violaciones de submodularidad: f(A) + f(B) < f(A∪B) + f(A∩B).
  4. Mostrar la biparticion encontrada por QNodos vs la optima (FuerzaBruta).

Para reproducir:
    source .venv/bin/activate
    PYTHONPATH=. python review/benchmarks/analisis_qnodos_k2.py
"""

from __future__ import annotations

import itertools

import numpy as np

from src.funciones.iit import seleccionar_emd
from src.funciones.particiones import biparticiones
from src.modelos.base.sia import SIA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_tpm(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random((1 << n, n), dtype=np.float32)


def _alinear(dist: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if dist.size == ref.size:
        return dist
    out = np.zeros_like(ref)
    out[: dist.size] = dist
    return out


class _Dummy(SIA):
    def aplicar_estrategia(self, *a, **kw):
        pass


# ---------------------------------------------------------------------------
# Funcion de particion sobre vertices (tiempo, indice)
# ---------------------------------------------------------------------------

def _construir_funcion_f(tpm: np.ndarray, n: int) -> tuple[
    object,           # sistema
    np.ndarray,       # dists
    list,             # vertices
    callable,         # f(subconjunto_de_vertices)
]:
    """Construye f(S) = EMD(dist, biparticion_S) sobre todos los subconjuntos."""
    distancia = seleccionar_emd()
    dummy = _Dummy(tpm)
    dummy.sia_preparar_subsistema("0" * n, "1" * n, "1" * n, "1" * n)
    sistema = dummy.sia_subsistema
    dists = dummy.sia_dists_marginales

    alcance = tuple(int(v) for v in sistema.indices_ncubos.tolist())
    mecanismo = tuple(int(v) for v in sistema.dims_ncubos.tolist())
    vertices = list([(0, int(v)) for v in mecanismo] + [(1, int(v)) for v in alcance])

    cache: dict[frozenset, float] = {}

    def f(subconjunto: frozenset) -> float:
        if subconjunto in cache:
            return cache[subconjunto]
        mec = [v for t, v in subconjunto if t == 0]
        alc = [v for t, v in subconjunto if t == 1]
        sp = sistema.bipartir(
            np.array(alc, dtype=np.int8),
            np.array(mec, dtype=np.int8),
        )
        dp = _alinear(sp.distribucion_marginal(), dists)
        val = float(distancia(dists, dp))
        cache[subconjunto] = val
        return val

    return sistema, dists, vertices, f


# ---------------------------------------------------------------------------
# Contar violaciones de submodularidad
# ---------------------------------------------------------------------------

def _contar_violaciones(vertices: list, f: callable) -> dict:
    """Cuenta pares (A, B) donde f(A) + f(B) < f(A union B) + f(A inter B)."""
    n = len(vertices)
    total = 0
    violaciones = 0
    max_viola = 0.0
    ejemplos_viola: list[tuple] = []

    verts = list(vertices)
    subconjuntos = []
    for r in range(1, n):
        for combo in itertools.combinations(range(n), r):
            subconjuntos.append(frozenset(verts[i] for i in combo))

    # Limitar la busqueda exhaustiva para n grandes
    if len(subconjuntos) > 200:
        rng = np.random.default_rng(42)
        idxs = rng.choice(len(subconjuntos), size=200, replace=False).tolist()
        subconjuntos_test = [subconjuntos[i] for i in sorted(idxs)]
    else:
        subconjuntos_test = subconjuntos

    full = frozenset(verts)
    for i, A in enumerate(subconjuntos_test):
        for B in subconjuntos_test[i + 1:]:
            union_ab = A | B
            inter_ab = A & B
            if union_ab == full or union_ab == frozenset():
                continue
            if inter_ab == full or inter_ab == frozenset():
                continue
            total += 1
            lhs = f(A) + f(B)
            rhs = f(union_ab) + f(inter_ab)
            delta = lhs - rhs  # debe ser >= 0 para submodularidad
            if delta < -1e-9:
                violaciones += 1
                if abs(delta) > max_viola:
                    max_viola = abs(delta)
                if len(ejemplos_viola) < 2:
                    ejemplos_viola.append((A, B, lhs, rhs, delta))

    return {
        "total_pares": total,
        "violaciones": violaciones,
        "porcentaje": 100.0 * violaciones / max(1, total),
        "max_violacion": max_viola,
        "ejemplos": ejemplos_viola,
    }


# ---------------------------------------------------------------------------
# Comparacion QNodos vs FuerzaBruta
# ---------------------------------------------------------------------------

def _comparar_estrategias(tpm: np.ndarray, n: int, semilla: int) -> dict:
    from src.estrategias.q_nodos import QNodos
    from src.estrategias.fuerza_bruta import FuerzaBruta

    estado = "0" * n
    mascara = "1" * n

    fb = FuerzaBruta(tpm)
    res_fb = fb.aplicar_estrategia(estado, mascara, mascara, mascara, k=2)

    qn = QNodos(tpm)
    res_qn = qn.aplicar_estrategia(estado, mascara, mascara, mascara, k=2)

    phi_opt = float(res_fb.perdida)
    phi_qn = float(res_qn.perdida)
    brecha = (phi_qn - phi_opt) / phi_opt * 100 if phi_opt > 1e-12 else 0.0

    return {
        "semilla": semilla,
        "phi_opt": phi_opt,
        "phi_qnodos": phi_qn,
        "brecha_pct": brecha,
        "particion_opt": res_fb.particion,
        "particion_qnodos": res_qn.particion,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    n = 4
    semillas = [11, 23, 37, 53, 71]

    print("=" * 70)
    print(f"Analisis de submodularidad de f(S) = EMD(biparticion_S) — n={n}")
    print("=" * 70)

    total_viola = 0
    total_pares = 0

    for semilla in semillas:
        tpm = _random_tpm(n, semilla)
        _, _, vertices, f = _construir_funcion_f(tpm, n)

        comp = _comparar_estrategias(tpm, n, semilla)
        viola = _contar_violaciones(vertices, f)

        total_viola += viola["violaciones"]
        total_pares += viola["total_pares"]

        brecha = comp["brecha_pct"]
        pct_v = viola["porcentaje"]
        marca = "***" if brecha > 5 else "   "

        print(f"\nSemilla {semilla}  {marca}")
        print(f"  FuerzaBruta: phi = {comp['phi_opt']:.6f}  particion: {comp['particion_opt']}")
        print(f"  QNodos:      phi = {comp['phi_qnodos']:.6f}  particion: {comp['particion_qnodos']}")
        print(f"  Brecha: {brecha:+.1f}%")
        print(f"  Submodularidad: {viola['violaciones']}/{viola['total_pares']} pares violan "
              f"f(A)+f(B) >= f(A∪B)+f(A∩B)  ({pct_v:.1f}%)")
        if viola["max_violacion"] > 0:
            print(f"  Mayor violacion: {viola['max_violacion']:.6f}")

    print("\n" + "=" * 70)
    print(f"Resumen global (n={n}, {len(semillas)} semillas):")
    pct_total = 100.0 * total_viola / max(1, total_pares)
    print(f"  Pares submodularidad violados: {total_viola}/{total_pares} ({pct_total:.1f}%)")

    print("\nCONCLUSION:")
    if pct_total > 5:
        print(f"  f(S) NO es submodular: {pct_total:.1f}% de los pares testeados la violan.")
        print("  Queyranne/MAO solo garantiza el minimo para funciones simetrico-")
        print("  submodulares. Si f viola submodularidad, el MAO puede converger")
        print("  a un minimo local no global, explicando la brecha de ~155%.")
    else:
        print(f"  f(S) parece aproximadamente submodular ({pct_total:.1f}% de violaciones).")
        print("  La falla de QNodos se debe a otro factor.")


if __name__ == "__main__":
    main()

"""
Análisis comparativo QNodos vs Geometric para hoja 25A-Elementos.
Genera tabla de métricas y gráficas de comparación.
Correr desde el directorio ProyectoAnalisis2026/.
"""
import openpyxl
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

EXCEL  = "DatosPruebas2026_1.xlsx"
SHEET  = "25A-Elementos "
OUTDIR = Path("resultados_25A")
OUTDIR.mkdir(exist_ok=True)

FILAS_META = {
    6:  (25,25), 7:  (25,24), 8:  (25,24), 9:  (25,24),
    10: (25,17), 11: (25,13), 12: (25,12), 13: (24,25),
    14: (24,24), 15: (24,24), 16: (24,24), 17: (24,17),
    18: (24,13), 19: (24,12), 20: (24,25), 21: (24,24),
    22: (24,24), 23: (24,24), 24: (24,17), 25: (24,13),
    26: (24,12), 27: (24,25), 28: (24,24), 29: (24,24),
    30: (24,24), 31: (24,17), 32: (24,13), 33: (24,12),
    34: (17,25), 35: (17,24), 36: (17,24), 37: (17,24),
    38: (17,17), 39: (17,13), 40: (17,12), 41: (13,25),
    42: (13,24), 43: (13,24), 44: (13,24), 45: (13,17),
    46: (13,13), 47: (13,12), 48: (12,25), 49: (12,24),
    50: (12,24), 51: (12,24), 52: (12,17), 53: (12,13),
    54: (12,12), 55: (21,21),
}

def fval(v):
    if v is None: return None
    if isinstance(v, (int, float)): return float(v)
    return None

wb = openpyxl.load_workbook(EXCEL, data_only=True)
ws = wb[SHEET]

data = {}
for row_idx, row in enumerate(ws.iter_rows(min_row=6, max_row=70, values_only=True), start=6):
    entry = {
        "q2": fval(row[4]),  "q3": fval(row[10]), "q4": fval(row[16]), "q5": fval(row[21]),
        "g2": fval(row[7]),  "g3": fval(row[13]), "g4": fval(row[19]), "g5": fval(row[24]),
        "q2t": fval(row[5]), "q3t": fval(row[11]),
        "g2t": fval(row[8]), "g3t": fval(row[14]),
    }
    if any(v is not None for v in entry.values()):
        alc_n, mec_n = FILAS_META.get(row_idx, (0, 0))
        entry["alc_n"] = alc_n
        entry["mec_n"] = mec_n
        data[row_idx] = entry
wb.close()

# ── 1. k=2 siempre es MIP? ──────────────────────────────────────────────────
print("\n=== 1. Verificación k=2 = MIP ===")
mip_check_q, mip_check_g = [], []
for fila, d in sorted(data.items()):
    qs = {k: d[f"q{k}"] for k in [2,3,4,5] if d[f"q{k}"] is not None}
    gs = {k: d[f"g{k}"] for k in [2,3,4,5] if d[f"g{k}"] is not None}
    if len(qs) >= 2:
        is_mip = min(qs, key=qs.get) == 2
        mip_check_q.append(is_mip)
        print(f"  fila {fila:>2} QNodos: k2={qs[2]:.6f}  k3={qs.get(3,'?')}  MIP=k{min(qs,key=qs.get)}  → {'✓' if is_mip else '✗'}")
    if len(gs) >= 2:
        is_mip = min(gs, key=gs.get) == 2
        mip_check_g.append(is_mip)
        print(f"  fila {fila:>2} Geo:    k2={gs[2]:.6f}  k3={gs.get(3,'?')}  MIP=k{min(gs,key=gs.get)}  → {'✓' if is_mip else '✗'}")

pct_q = 100*sum(mip_check_q)/len(mip_check_q) if mip_check_q else 0
pct_g = 100*sum(mip_check_g)/len(mip_check_g) if mip_check_g else 0
print(f"\n  QNodos k=2 es MIP: {sum(mip_check_q)}/{len(mip_check_q)} = {pct_q:.1f}%")
print(f"  Geo    k=2 es MIP: {sum(mip_check_g)}/{len(mip_check_g)} = {pct_g:.1f}%")

# ── 2. Comparación perdida Q vs G en k=2 ─────────────────────────────────────
# Identificar filas cuyo QNodos vino de cache (tiempo=0) vs computación directa
print("\n=== 2. Acuerdo QNodos vs Geometric (k=2) ===")
filas_ambos = [(f, d) for f, d in sorted(data.items()) if d["q2"] is not None and d["g2"] is not None]

# Separar: directo (q2t > 0) vs cache (q2t == 0 o None)
filas_directas  = [(f, d) for f, d in filas_ambos if d["q2t"] and d["q2t"] > 1.0]
filas_cache     = [(f, d) for f, d in filas_ambos if not d["q2t"] or d["q2t"] <= 1.0]

def stats(pares, label):
    if len(pares) < 2:
        print(f"  [{label}] Solo {len(pares)} fila(s) — insuficiente para correlación")
        for f, d in pares:
            q, g = d["q2"], d["g2"]
            re = abs(q-g)/max(q, 1e-12)
            print(f"    fila {f:>2} mec={d['mec_n']:>2}: Q={q:.8f}  G={g:.8f}  err_rel={re*100:.4f}%")
        return
    q_arr = np.array([d["q2"] for _, d in pares])
    g_arr = np.array([d["g2"] for _, d in pares])
    r = np.corrcoef(q_arr, g_arr)[0, 1]
    diffs = np.abs(q_arr - g_arr)
    rel   = diffs / np.maximum(q_arr, 1e-12)
    exact = np.sum(diffs < 1e-6)
    print(f"  [{label}] n={len(pares)}  Pearson r={r:.6f}  err_rel_medio={np.mean(rel)*100:.4f}%  acuerdo_exacto={exact}/{len(pares)}")
    for (f, d), q, g, re in zip(pares, q_arr, g_arr, rel):
        flag = "  ⚠ DISCREPANCIA" if re > 0.01 else ""
        print(f"    fila {f:>2} mec={d['mec_n']:>2}: Q={q:.8f}  G={g:.8f}  err_rel={re*100:.4f}%{flag}")

stats(filas_directas, "QNodos computado directamente")
print()
stats(filas_cache,    "QNodos desde cache (posible error de suposición)")

if filas_cache:
    print()
    print("  NOTA: filas con QNodos desde cache asumen perdida idéntica para mismo mec con")
    print("  distinto alc. Si Geometric discrepa, la suposición de cache puede ser incorrecta.")
    print("  Recomendación: recomputar esas filas directamente para validar.")

# Para gráficas usar solo directas
q_vals = np.array([d["q2"] for _, d in filas_directas]) if filas_directas else np.array([])
g_vals = np.array([d["g2"] for _, d in filas_directas]) if filas_directas else np.array([])

# ── 3. Tiempos Q vs G ─────────────────────────────────────────────────────────
print("\n=== 3. Tiempos de cómputo k=2 ===")
print(f"  {'Fila':>4} {'mec_n':>5} | {'Q_k2_t (s)':>12} {'G_k2_t (s)':>12} {'G/Q':>8}")
for fila, d in sorted(data.items()):
    qt = d["q2t"]
    gt = d["g2t"]
    if qt or gt:
        ratio = f"{gt/qt:.2f}x" if qt and gt and qt > 0 else "-"
        print(f"  {fila:>4} {d['mec_n']:>5} | {qt or 0:>12.2f} {gt or 0:>12.2f} {ratio:>8}")

# ── 4. Gráficas ───────────────────────────────────────────────────────────────
if len(q_vals) >= 2:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter: Q perdida vs G perdida
    ax = axes[0]
    ax.scatter(q_vals, g_vals, s=80, color="steelblue", zorder=3)
    lim = max(q_vals.max(), g_vals.max()) * 1.1
    ax.plot([0, lim], [0, lim], "r--", alpha=0.5, label="Q=G")
    for (f, _), q, g in zip(filas_ambos, q_vals, g_vals):
        ax.annotate(str(f), (q, g), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("QNodos φ (k=2)")
    ax.set_ylabel("Geometric φ (k=2)")
    r_direct = np.corrcoef(q_vals, g_vals)[0, 1] if len(q_vals) >= 2 else float('nan')
    ax.set_title(f"Pérdida mínima: QNodos vs Geometric (directo)\nPearson r = {r_direct:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bar: k=2 pérdida por fila comparando Q y G (solo directas)
    ax = axes[1]
    filas_idx = [f for f, _ in filas_directas]
    x = np.arange(len(filas_idx))
    w = 0.35
    ax.bar(x - w/2, q_vals, w, label="QNodos", color="steelblue")
    ax.bar(x + w/2, g_vals, w, label="Geometric", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([str(f) for f in filas_idx])
    ax.set_xlabel("Fila")
    ax.set_ylabel("Pérdida φ (k=2)")
    ax.set_title("Pérdida k=2 por fila")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = OUTDIR / "comparacion_Q_vs_G_k2.png"
    plt.savefig(out, dpi=150)
    print(f"\nGráfica guardada: {out}")

# ── 5. MIP confirmation plot ──────────────────────────────────────────────────
filas_multik = [(f, d) for f, d in sorted(data.items())
                if sum(1 for k in [2,3,4,5] if d[f"q{k}"] is not None) >= 3]
if filas_multik:
    fig, ax = plt.subplots(figsize=(10, 5))
    ks = [2, 3, 4, 5]
    colors = ["steelblue", "orange", "green", "red"]
    for i, (fila, d) in enumerate(filas_multik):
        ys = [d[f"q{k}"] for k in ks if d[f"q{k}"] is not None]
        xs = [k for k in ks if d[f"q{k}"] is not None]
        ax.plot(xs, ys, "o-", color=f"C{i}", label=f"fila {fila} (mec={d['mec_n']})", alpha=0.7)
    ax.set_xlabel("k (número de partes)")
    ax.set_ylabel("Pérdida φ")
    ax.set_title("QNodos: φ mínima siempre en k=2 (MIP = bipartición)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([2, 3, 4, 5])
    out2 = OUTDIR / "mip_confirmacion_k2.png"
    plt.savefig(out2, dpi=150)
    print(f"Gráfica MIP guardada: {out2}")

print("\n=== Análisis completo ===")
print(f"Resultados en: {OUTDIR}/")

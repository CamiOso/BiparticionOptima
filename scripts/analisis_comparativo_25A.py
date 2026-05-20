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
print("\n=== 2. Acuerdo QNodos vs Geometric (k=2) ===")
filas_ambos = [(f, d) for f, d in sorted(data.items()) if d["q2"] is not None and d["g2"] is not None]
q_vals = np.array([d["q2"] for _, d in filas_ambos])
g_vals = np.array([d["g2"] for _, d in filas_ambos])

if len(q_vals) >= 2:
    r = np.corrcoef(q_vals, g_vals)[0, 1]
    abs_diffs = np.abs(q_vals - g_vals)
    rel_errs  = abs_diffs / np.maximum(q_vals, 1e-12)
    exact_match = np.sum(abs_diffs < 1e-6)
    print(f"  Filas con ambos resultados k=2: {len(q_vals)}")
    print(f"  Pearson r(Q,G): {r:.6f}")
    print(f"  Error relativo medio: {np.mean(rel_errs)*100:.4f}%")
    print(f"  Error relativo max:   {np.max(rel_errs)*100:.4f}%")
    print(f"  Acuerdo exacto (|Q-G|<1e-6): {exact_match}/{len(q_vals)} = {100*exact_match/len(q_vals):.1f}%")
    for (f, d), q, g, re in zip(filas_ambos, q_vals, g_vals, rel_errs):
        print(f"    fila {f:>2} mec={d['mec_n']:>2}: Q={q:.8f}  G={g:.8f}  err_rel={re*100:.4f}%")
else:
    print("  Insuficientes datos para correlación (necesita ≥2 filas con ambos resultados)")

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
    ax.set_title(f"Pérdida mínima: QNodos vs Geometric\nPearson r = {r:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bar: k=2 pérdida por fila comparando Q y G
    ax = axes[1]
    filas_idx = [f for f, _ in filas_ambos]
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

"""
Genera las gráficas de los Requerimientos (preguntas 1a, 1b, 1c)
en un único PDF: review/graficas_requerimientos.pdf
"""
import openpyxl
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ── Paleta ────────────────────────────────────────────────────────────────────
C_Q  = "#1f77b4"   # azul  → QNodes
C_G  = "#d62728"   # rojo  → Geometric
REDS   = ["#fee0d2", "#fc9272", "#de2d26", "#a50f15"]
BLUES  = ["#deebf7", "#9ecae1", "#3182bd", "#08519c"]

# ── Carga de datos ─────────────────────────────────────────────────────────────
def cargar_sheet(wb, sheet_name):
    ws = wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))
    data = []
    for r in rows[5:]:
        if r[1] is None or r[4] is None:
            continue
        mec = str(r[2]) if r[2] else ""
        entry = {
            "mec": mec,
            "mec_size": len(mec),
            "q2_loss": r[4],  "q2_time": r[5],
            "g2_loss": r[7],  "g2_time": r[8],
            "q3_loss": r[10] if len(r) > 10 else None, "q3_time": r[11] if len(r) > 11 else None,
            "g3_loss": r[13] if len(r) > 13 else None, "g3_time": r[14] if len(r) > 14 else None,
            "q4_loss": r[16] if len(r) > 16 else None, "q4_time": r[17] if len(r) > 17 else None,
            "g4_loss": r[19] if len(r) > 19 else None, "g4_time": r[20] if len(r) > 20 else None,
            "q5_loss": r[22] if len(r) > 22 else None, "q5_time": r[23] if len(r) > 23 else None,
            "g5_loss": r[25] if len(r) > 25 else None, "g5_time": r[26] if len(r) > 26 else None,
        }
        data.append(entry)
    return data

wb = openpyxl.load_workbook("DatosPruebas2026_1.xlsx", read_only=True, data_only=True)
d10 = cargar_sheet(wb, "10A-Elementos")
d15 = cargar_sheet(wb, "15B-Elementos")
d20 = cargar_sheet(wb, "20A-Elementos")

sheets = [("10A  (N=10)", d10), ("15B  (N=15)", d15), ("20A  (N=20)", d20)]
ks = [2, 3, 4, 5]

def avg(lst): return sum(lst) / len(lst) if lst else float("nan")

def tiempos_por_size(data, k):
    """Devuelve (sizes, q_avgs, g_avgs) para un k dado."""
    sizes = sorted(set(d["mec_size"] for d in data))
    q_avgs, g_avgs = [], []
    for sz in sizes:
        sub = [d for d in data if d["mec_size"] == sz
               and d[f"q{k}_time"] and d[f"g{k}_time"]]
        q_avgs.append(avg([d[f"q{k}_time"] for d in sub]))
        g_avgs.append(avg([d[f"g{k}_time"] for d in sub]))
    return sizes, q_avgs, g_avgs

def perdidas_por_size(data, k):
    sizes = sorted(set(d["mec_size"] for d in data))
    q_avgs, g_avgs = [], []
    for sz in sizes:
        sub = [d for d in data if d["mec_size"] == sz
               and d[f"q{k}_loss"] is not None and d[f"g{k}_loss"] is not None]
        q_avgs.append(avg([d[f"q{k}_loss"] for d in sub]))
        g_avgs.append(avg([d[f"g{k}_loss"] for d in sub]))
    return sizes, q_avgs, g_avgs

# ── PDF ────────────────────────────────────────────────────────────────────────
pdf_path = "review/graficas_requerimientos.pdf"
with PdfPages(pdf_path) as pdf:

    # ══════════════════════════════════════════════════════════════════════════
    # PÁGINA 1 — Portada
    # ══════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#f5f5f5")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.5, 0.72, "Análisis y Diseño de Algoritmos — 2026",
            ha="center", va="center", fontsize=18, color="#444")
    ax.text(0.5, 0.60, "Comparación de Estrategias IIT\nQNodes vs Geometric",
            ha="center", va="center", fontsize=26, fontweight="bold", color="#1a1a2e",
            linespacing=1.4)
    ax.text(0.5, 0.43,
            "Redes: 10A (N=10) · 15B (N=15) · 20A (N=20)\n"
            "Particiones: k = 2, 3, 4, 5\n\n"
            "Requerimiento 1a — Tiempos de ejecución por tamaño\n"
            "Requerimiento 1b — Pérdidas φ por k-partición\n"
            "Requerimiento 1c — Variación relativa EMD (Geo vs QNodes)",
            ha="center", va="center", fontsize=13, color="#333",
            linespacing=1.8)
    ax.text(0.5, 0.08, "Universidad de Caldas · Proyecto 2026-1",
            ha="center", va="center", fontsize=11, color="#888")
    patch_q = mpatches.Patch(color=C_Q, label="QNodes")
    patch_g = mpatches.Patch(color=C_G, label="Geometric")
    fig.legend(handles=[patch_q, patch_g], loc="lower right",
               fontsize=12, framealpha=0.8, bbox_to_anchor=(0.92, 0.12))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PÁGINAS 2–5 — 1a: Tiempo por tamaño, un panel por red, un subplot por k
    # ══════════════════════════════════════════════════════════════════════════
    for label, data in sheets:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle(f"1a — Tiempo de ejecución promedio por tamaño de mecanismo\nRed {label}",
                     fontsize=14, fontweight="bold", y=1.01)
        for idx, k in enumerate(ks):
            ax = axes[idx // 2][idx % 2]
            sizes, q_avgs, g_avgs = tiempos_por_size(data, k)
            valid = [(s, q, g) for s, q, g in zip(sizes, q_avgs, g_avgs)
                     if not (np.isnan(q) or np.isnan(g))]
            if not valid:
                ax.set_visible(False)
                continue
            sv, qv, gv = zip(*valid)
            x = np.arange(len(sv))
            w = 0.35
            bars_q = ax.bar(x - w/2, qv, w, color=C_Q, alpha=0.85, label="QNodes", zorder=3)
            bars_g = ax.bar(x + w/2, gv, w, color=C_G, alpha=0.85, label="Geometric", zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels([f"mec={s}" for s in sv], fontsize=9)
            ax.set_ylabel("Tiempo promedio (s)", fontsize=9)
            ax.set_title(f"k = {k}", fontsize=11, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.4, zorder=0)
            ax.set_yscale("log" if max(max(qv), max(gv)) / (min(min(qv), min(gv)) + 1e-9) > 20 else "linear")
            # ratio label encima de la barra mayor
            for xi, (q, g) in enumerate(zip(qv, gv)):
                if q > 0 and g > 0:
                    ratio = g / q
                    ypos = max(q, g) * 1.05
                    ax.text(xi, ypos, f"×{ratio:.1f}", ha="center", va="bottom", fontsize=7, color="#555")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PÁGINA 6 — 1a resumen: líneas de tiempo por tamaño, todas las redes
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("1a — Escalamiento del tiempo: QNodes vs Geometric (k=2)",
                 fontsize=13, fontweight="bold")
    for ax, (label, data) in zip(axes, [("10A (N=10)", d10), ("15B (N=15)", d15)]):
        sizes, qv, gv = tiempos_por_size(data, 2)
        ax.plot(sizes, qv, "o-", color=C_Q, lw=2, label="QNodes")
        ax.plot(sizes, gv, "s-", color=C_G, lw=2, label="Geometric")
        ax.set_xlabel("Tamaño mecanismo (nodos)", fontsize=10)
        ax.set_ylabel("Tiempo promedio (s)", fontsize=10)
        ax.set_title(f"Red {label}", fontsize=11)
        ax.legend(); ax.grid(alpha=0.4)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PÁGINAS 7–9 — 1b: Pérdidas φ promedio por k, una página por red
    # ══════════════════════════════════════════════════════════════════════════
    for label, data in sheets:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle(f"1b — Pérdida φ promedio por tamaño de mecanismo\nRed {label}",
                     fontsize=14, fontweight="bold", y=1.01)
        for idx, k in enumerate(ks):
            ax = axes[idx // 2][idx % 2]
            sizes, qv, gv = perdidas_por_size(data, k)
            valid = [(s, q, g) for s, q, g in zip(sizes, qv, gv)
                     if not (np.isnan(q) or np.isnan(g))]
            if not valid:
                ax.set_visible(False)
                continue
            sv, qv, gv = zip(*valid)
            x = np.arange(len(sv))
            w = 0.35
            ax.bar(x - w/2, qv, w, color=C_Q, alpha=0.85, label="QNodes", zorder=3)
            ax.bar(x + w/2, gv, w, color=C_G, alpha=0.85, label="Geometric", zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels([f"mec={s}" for s in sv], fontsize=9)
            ax.set_ylabel("φ promedio (EMD)", fontsize=9)
            ax.set_title(f"k = {k}", fontsize=11, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.4, zorder=0)
            if max(max(qv), max(gv)) / (min(min(qv), min(gv)) + 1e-12) > 30:
                ax.set_yscale("log")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PÁGINA 10 — 1b: QNodes gana vs Geo gana por k y red (heatmap conteo)
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("1b — ¿Quién halla la menor pérdida φ? (conteo por k-partición)",
                 fontsize=13, fontweight="bold")
    for ax, (label, data) in zip(axes, sheets):
        complete = [d for d in data if all(
            d[f"q{k}_loss"] is not None and d[f"g{k}_loss"] is not None for k in ks)]
        q_wins = [sum(1 for d in complete if d[f"q{k}_loss"] < d[f"g{k}_loss"] - 1e-8) for k in ks]
        g_wins = [sum(1 for d in complete if d[f"g{k}_loss"] < d[f"q{k}_loss"] - 1e-8) for k in ks]
        ties   = [sum(1 for d in complete if abs(d[f"q{k}_loss"] - d[f"g{k}_loss"]) < 1e-8) for k in ks]
        n = len(complete)
        x = np.arange(4)
        w = 0.28
        ax.bar(x - w, q_wins, w, color=C_Q, alpha=0.85, label="QNodes gana", zorder=3)
        ax.bar(x,     ties,   w, color="#888888", alpha=0.7, label="Empate exacto", zorder=3)
        ax.bar(x + w, g_wins, w, color=C_G, alpha=0.85, label="Geo gana", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"k={k}" for k in ks])
        ax.set_ylabel("Número de subsistemas")
        ax.set_title(f"Red {label}\n(n={n} completos)")
        ax.axhline(n, color="k", lw=0.8, ls="--", alpha=0.4)
        ax.text(3.55, n + 0.3, f"n={n}", fontsize=8, color="#555")
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.4, zorder=0)
        ax.set_ylim(0, n * 1.15)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PÁGINA 11 — 1b: Monotonicity violations
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("1b — Violaciones de monotonicidad φ_k > φ_{k-1}\n(ambas estrategias deberían tener φ_k ≤ φ_{k-1})",
                 fontsize=12, fontweight="bold")
    for ax, (label, data) in zip(axes, sheets):
        complete = [d for d in data if all(
            d[f"q{k}_loss"] is not None and d[f"g{k}_loss"] is not None for k in ks)]
        n = len(complete)
        ks_check = [3, 4, 5]
        q_viols = [sum(1 for d in complete
                       if d[f"q{k}_loss"] > d[f"q{k-1}_loss"] + 1e-8) for k in ks_check]
        g_viols = [sum(1 for d in complete
                       if d[f"g{k}_loss"] > d[f"g{k-1}_loss"] + 1e-8) for k in ks_check]
        x = np.arange(3)
        w = 0.35
        ax.bar(x - w/2, [100*v/n for v in q_viols], w, color=C_Q, alpha=0.85, label="QNodes")
        ax.bar(x + w/2, [100*v/n for v in g_viols], w, color=C_G, alpha=0.85, label="Geometric")
        ax.set_xticks(x)
        ax.set_xticklabels([f"k={k} vs k={k-1}" for k in ks_check], fontsize=9)
        ax.set_ylabel("% subsistemas con violación")
        ax.set_title(f"Red {label}\n(n={n} completos)")
        ax.set_ylim(0, 110)
        ax.axhline(50, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.4)
        for xi, (qv, gv) in enumerate(zip(q_viols, g_viols)):
            ax.text(xi - w/2, 100*qv/n + 2, f"{qv}/{n}", ha="center", fontsize=7)
            ax.text(xi + w/2, 100*gv/n + 2, f"{gv}/{n}", ha="center", fontsize=7)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PÁGINAS 12–14 — 1c: Variación relativa Δ% por subsistema (scatter)
    # ══════════════════════════════════════════════════════════════════════════
    for label, data in sheets:
        complete = [d for d in data if all(
            d[f"q{k}_loss"] is not None and d[f"g{k}_loss"] is not None for k in ks)]
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle(f"1c — Variación relativa de pérdida EMD: Δ% = (φ_Geo − φ_Q) / φ_Q × 100\n"
                     f"Red {label}   (QNodes = referencia; Δ>0 → Geo peor; Δ<0 → Geo mejor)",
                     fontsize=12, fontweight="bold", y=1.02)
        for idx, k in enumerate(ks):
            ax = axes[idx // 2][idx % 2]
            qk = f"q{k}_loss"; gk = f"g{k}_loss"
            valid = [d for d in complete if d[qk] and d[qk] > 1e-10]
            if not valid:
                ax.set_visible(False)
                continue
            diffs = [(d[gk] - d[qk]) / d[qk] * 100 for d in valid]
            sizes = [d["mec_size"] for d in valid]
            colors = [C_R if d > 0 else C_Q for d, C_R in
                      [(dv, C_G) for dv in diffs]]
            colors = [C_G if dv > 0.01 else (C_Q if dv < -0.01 else "#888") for dv in diffs]
            sc = ax.scatter(sizes, diffs, c=colors, alpha=0.75, edgecolors="white", lw=0.4, s=60, zorder=3)
            ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.7)
            ax.set_xlabel("Tamaño mecanismo (nodos)", fontsize=9)
            ax.set_ylabel("Δ% (Geo vs QNodes)", fontsize=9)
            ax.set_title(f"k = {k}", fontsize=11, fontweight="bold")
            ax.grid(alpha=0.35, zorder=0)
            avg_d = sum(diffs) / len(diffs)
            ax.axhline(avg_d, color="#e07b00", lw=1.5, ls=":", alpha=0.9, label=f"Promedio {avg_d:+.0f}%")
            ax.legend(fontsize=8)
            # Zona coloreada
            ymin, ymax = ax.get_ylim()
            ax.fill_between([min(sizes)-0.5, max(sizes)+0.5], 0, max(ymax, 1),
                            alpha=0.05, color=C_G, label="_Geo peor")
            ax.fill_between([min(sizes)-0.5, max(sizes)+0.5], min(ymin, -1), 0,
                            alpha=0.05, color=C_Q, label="_Geo mejor")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PÁGINA 15 — 1c resumen: Δ% promedio por k y red (heatmap numérico)
    # ══════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("1c — Resumen: Diferencia promedio de pérdida EMD\n"
                 "Δ% = (φ_Geo − φ_QNodes) / φ_QNodes × 100   (negativo = Geo mejor)",
                 fontsize=12, fontweight="bold")

    sheet_labels = ["10A (N=10)", "15B (N=15)", "20A (N=20)"]
    matrix = []
    for _, data in sheets:
        complete = [d for d in data if all(
            d[f"q{k}_loss"] is not None and d[f"g{k}_loss"] is not None for k in ks)]
        row = []
        for k in ks:
            qk = f"q{k}_loss"; gk = f"g{k}_loss"
            valid = [d for d in complete if d[qk] and d[qk] > 1e-10]
            diffs = [(d[gk] - d[qk]) / d[qk] * 100 for d in valid]
            row.append(avg(diffs) if diffs else float("nan"))
        matrix.append(row)

    matrix = np.array(matrix)
    # Clip for color mapping — values can be extreme
    vmax = 500; vmin = -100
    mat_clipped = np.clip(matrix, vmin, vmax)

    im = ax.imshow(mat_clipped, cmap="RdBu_r", aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Δ% (saturado en ±500%)")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"k={k}" for k in ks], fontsize=12)
    ax.set_yticks(range(3)); ax.set_yticklabels(sheet_labels, fontsize=12)
    for i in range(3):
        for j in range(4):
            val = matrix[i, j]
            txt = f"{val:+.0f}%" if not np.isnan(val) else "—"
            color = "white" if abs(mat_clipped[i, j]) > 200 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=13,
                    fontweight="bold", color=color)
    ax.set_xlabel("k-partición", fontsize=11)
    ax.set_ylabel("Red analizada", fontsize=11)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PÁGINA 16 — Conclusiones
    # ══════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#f9f9f9")
    ax = fig.add_axes([0.08, 0.05, 0.84, 0.90])
    ax.axis("off")
    ax.text(0.5, 0.97, "Conclusiones — Requerimientos 1a, 1b, 1c",
            ha="center", va="top", fontsize=15, fontweight="bold", color="#1a1a2e")

    conclusiones = [
        ("1a  Tiempos", [
            "QNodes es más rápido que Geometric en todos los tamaños para k=2.",
            "La brecha crece con N: 4x en N=10 → 11x en N=15 → hasta 150x en N=20.",
            "Para k>2, QNodes reutiliza la semilla k=2 → tiempos casi constantes.",
            "Geometric mantiene el mismo presupuesto SA para k=3,4,5 → tiempo estable pero alto.",
        ]),
        ("1b  Pérdidas / misma partición", [
            "k=2: ambas estrategias son EXACTAMENTE IGUALES en N=10 y N=15 (100% coincidencia).",
            "     En N=20 coinciden en 81% — Geometric a veces queda en mínimo local.",
            "k≥3: coincidencia colapsa a 8–18% — particiones completamente distintas.",
            "N=10/15: QNodes halla menor φ en 80-86% de casos para k>2.",
            "N=20: Geometric invierte la ventaja — gana en k=4 (81%) y k=5 (100%).",
            "Violación de monotonicidad φ_k ≤ φ_{k-1}: 57–85% de los casos en ambas estrategias.",
        ]),
        ("1c  Variación EMD", [
            "k=2: Δ = 0% en N=10/15 (idénticas). Δ = +77% en N=20 (QNodes exacto gana).",
            "k=3: Geo hasta 7900% peor en N=10; mixto en N=20 (promedio +529%).",
            "k=4: Punto de inversión — Geo −16% mejor en N=20 pero 3200% peor en N=15.",
            "k=5: Geo domina en N=20 (−73%, gana 21/21). Geo peor en N=10/15.",
            "Conclusión: la ventaja de cada estrategia depende fuertemente del tamaño N.",
            "No existe una estrategia universalmente superior para k>2 en todos los tamaños.",
        ]),
    ]

    y = 0.88
    for titulo, puntos in conclusiones:
        ax.text(0.0, y, titulo, fontsize=12, fontweight="bold", color="#1f77b4", va="top")
        y -= 0.045
        for p in puntos:
            ax.text(0.02, y, f"• {p}", fontsize=9.5, va="top", color="#222",
                    wrap=True)
            y -= 0.042
        y -= 0.02

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print(f"PDF generado: {pdf_path}")

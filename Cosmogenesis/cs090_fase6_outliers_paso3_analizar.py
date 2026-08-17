"""
cs090_fase6_outliers_paso3_analizar.py -- FASE VI, cierre numerico del Paso 3 (solo lectura).

Cruza las 11 corridas nuevas de Phantom sobre reglas de pendiente muy negativa
(cs090_fase6_outliers_phantom_metricas.csv) con:
  - la distribucion de fraccion de masa de las 80 corridas ya existentes de Fase V-B
    (cs090_fase5b_TOTAL_40pares.csv), para ubicar cada una en percentil, y
  - la pendiente CORREGIDA (diametro medido sobre la componente gigante) calculada en
    cs090_fase6_outliers_diam_gigante.csv,
y responde las dos preguntas del encargo: (a) ¿la fraccion de masa alta de los 3 outliers originales se
repite en las reglas nuevas de pendiente muy negativa, o se dispersan?, y (b) ¿la pendiente corregida
predice esa fraccion de masa, tanto dentro del grupo nuevo como sobre el conjunto total de reglas
distintas medidas?

Salidas: cs090_fase6_outliers_paso3_resumen.csv, cs090_fase6_outliers_paso3.png.
No corre Phantom. No modifica nada. No declara veredicto.
"""
from __future__ import annotations
import csv
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"


def main():
    # --- las 80 corridas previas de Fase V-B (referencia de distribucion) ---
    prev = list(csv.DictReader(open(f"{HERE}/cs090_fase5b_TOTAL_40pares.csv")))
    fm80 = np.array([float(r["fraccion_masa_en_sumideros"]) for r in prev])
    kv80 = np.array([float(r["kappa_v_agregado"]) for r in prev])

    # --- pendiente corregida (diagnostico del Paso 1) ---
    corr = {}
    for r in csv.DictReader(open(f"{HERE}/cs090_fase6_outliers_diam_gigante.csv")):
        corr[int(r["seed"])] = dict(pend_corr=float(r["pend_corr"]), pend_orig=float(r["pend_orig"]),
                                    fraccion_masa=r["fraccion_masa"], corrio=r["corrio_phantom"] == "True",
                                    clase=r["clase"], giant=float(r["giant_b1"]))

    # --- las 11 corridas nuevas ---
    nuevas = list(csv.DictReader(open(f"{HERE}/cs090_fase6_outliers_phantom_metricas.csv")))
    print(f"[paso3] {len(nuevas)} corridas nuevas de Phantom sobre reglas de pendiente muy negativa\n")
    print(f"[referencia] las 80 de Fase V-B: media={fm80.mean():.4f} sd={fm80.std(ddof=1):.4f} "
          f"min={fm80.min():.4f} mediana={np.median(fm80):.4f} p75={np.percentile(fm80,75):.4f} "
          f"max={fm80.max():.4f}\n")

    filas = []
    print("   regla                     pend_orig  pend_corr   fracc_masa   percentil_vs_80   kappa_V")
    for r in sorted(nuevas, key=lambda x: float(x["pendiente"])):
        s = int(r["seed"])
        fm = float(r["fraccion_masa_en_sumideros"])
        pc = corr[s]["pend_corr"]
        pct = 100.0 * float((fm80 < fm).mean())
        kv = float(r["kappa_v_agregado"])
        print(f"   {r['rule_id']:<24} {float(r['pendiente']):+8.4f}  {pc:+8.4f}   {fm:.4f}       "
              f"{pct:5.1f}          {kv:.3f}")
        filas.append(dict(rule_id=r["rule_id"], seed=s, clase_csv=r["clase_csv"],
                          pendiente_original=float(r["pendiente"]), pendiente_corregida=pc,
                          fraccion_masa_en_sumideros=fm, percentil_vs_80=round(pct, 1),
                          kappa_v_agregado=kv, n_sumideros=int(r["n_sumideros"]),
                          t_primer_sumidero=r["t_primer_sumidero"],
                          giant_grafo_final=float(r["giant_grafo_final"]),
                          grado_medio_grafo_final=float(r["grado_medio_grafo_final"]),
                          K=r["K"], J=r["J"], noise=r["noise"], meandeg=r["meandeg"], kcap=r["kcap"]))

    fmn = np.array([f["fraccion_masa_en_sumideros"] for f in filas])
    pcn = np.array([f["pendiente_corregida"] for f in filas])
    pon = np.array([f["pendiente_original"] for f in filas])
    print(f"\n[las 11 nuevas] media={fmn.mean():.4f} sd={fmn.std(ddof=1):.4f} min={fmn.min():.4f} "
          f"mediana={np.median(fmn):.4f} max={fmn.max():.4f}")
    print(f"   cuantas por encima de la MEDIANA de las 80 ({np.median(fm80):.4f}): "
          f"{int((fmn > np.median(fm80)).sum())}/11")
    print(f"   cuantas por encima del p75 de las 80 ({np.percentile(fm80,75):.4f}): "
          f"{int((fmn > np.percentile(fm80,75)).sum())}/11")
    print(f"   cuantas por encima del MAXIMO de las 80 ({fm80.max():.4f}): "
          f"{int((fmn > fm80.max()).sum())}/11")
    u, pu = stats.mannwhitneyu(fmn, fm80, alternative="two-sided")
    print(f"   Mann-Whitney U (11 nuevas vs 80 previas, dos colas): U={u:.1f} p={pu:.3e}")

    print(f"\n[dentro de las 11 nuevas] pendiente CORREGIDA vs fraccion de masa: "
          f"Spearman rho={stats.spearmanr(pcn, fmn)[0]:+.4f} (p={stats.spearmanr(pcn, fmn)[1]:.4f}), "
          f"R2 lineal={np.corrcoef(pcn, fmn)[0,1]**2:.4f}")
    print(f"[dentro de las 11 nuevas] pendiente ORIGINAL  vs fraccion de masa: "
          f"Spearman rho={stats.spearmanr(pon, fmn)[0]:+.4f} (p={stats.spearmanr(pon, fmn)[1]:.4f}), "
          f"R2 lineal={np.corrcoef(pon, fmn)[0,1]**2:.4f}")

    # --- conjunto total: 76 reglas distintas de Fase V-B + 11 nuevas = 87 ---
    todo_x_corr, todo_x_orig, todo_y, todo_grupo = [], [], [], []
    for s, d in corr.items():
        if d["corrio"]:
            todo_x_corr.append(d["pend_corr"]); todo_x_orig.append(d["pend_orig"])
            todo_y.append(float(d["fraccion_masa"])); todo_grupo.append("faseVB")
    for f in filas:
        todo_x_corr.append(f["pendiente_corregida"]); todo_x_orig.append(f["pendiente_original"])
        todo_y.append(f["fraccion_masa_en_sumideros"]); todo_grupo.append("nueva_neg")
    X, Xo, Y = np.array(todo_x_corr), np.array(todo_x_orig), np.array(todo_y)
    print(f"\n[conjunto total: {len(Y)} reglas distintas con Phantom (76 de Fase V-B + 11 nuevas)]")
    for etq, x in (("pendiente ORIGINAL ", Xo), ("pendiente CORREGIDA", X)):
        rho, pv = stats.spearmanr(x, Y)
        print(f"   {etq}: Spearman rho={rho:+.4f} (p={pv:.2e})  R2 lineal={np.corrcoef(x,Y)[0,1]**2:.4f}")

    campos = list(filas[0].keys())
    with open(f"{HERE}/cs090_fase6_outliers_paso3_resumen.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos); w.writeheader(); w.writerows(filas)
    print(f"\n[csv] cs090_fase6_outliers_paso3_resumen.csv")

    # --- grafico ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    ax.hist(fm80, bins=22, color="#4477aa", alpha=0.75, label="80 corridas de Fase V-B", edgecolor="white")
    for v in fmn:
        ax.axvline(v, color="#cc3311", lw=1.6, alpha=0.9)
    ax.axvline(fmn[0], color="#cc3311", lw=1.6, label="11 nuevas (pendiente muy negativa)")
    ax.set_xlabel("fraccion de masa en sumideros"); ax.set_ylabel("nº de corridas")
    ax.set_title("¿Donde caen las 11 nuevas?"); ax.legend(fontsize=8)

    ax = axes[1]
    m = np.array(todo_grupo) == "faseVB"
    ax.scatter(Xo[m], Y[m], s=26, alpha=0.75, color="#4477aa", label="76 de Fase V-B")
    ax.scatter(Xo[~m], Y[~m], s=34, alpha=0.9, color="#cc3311", label="11 nuevas (pend. muy neg.)")
    rho, _ = stats.spearmanr(Xo, Y)
    ax.set_xlabel("pendiente ORIGINAL"); ax.set_ylabel("fraccion de masa")
    ax.set_title(f"predictor ORIGINAL (Spearman {rho:+.3f}, n={len(Y)})")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[2]
    ax.scatter(X[m], Y[m], s=26, alpha=0.75, color="#4477aa", label="76 de Fase V-B")
    ax.scatter(X[~m], Y[~m], s=34, alpha=0.9, color="#cc3311", label="11 nuevas (pend. muy neg.)")
    rho, _ = stats.spearmanr(X, Y)
    ax.set_xlabel("pendiente CORREGIDA (diam de la componente gigante)"); ax.set_ylabel("fraccion de masa")
    ax.set_title(f"predictor CORREGIDO (Spearman {rho:+.3f}, n={len(Y)})")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.suptitle("Paso 3: Phantom sobre 11 reglas nuevas de pendiente muy negativa", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{HERE}/cs090_fase6_outliers_paso3.png", dpi=130)
    print("[png] cs090_fase6_outliers_paso3.png")


if __name__ == "__main__":
    main()

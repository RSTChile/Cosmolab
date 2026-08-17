"""
cs090_fase8_f803_analizar.py — FASE VIII F8-03: ¿queda algo de masa cuando el PICO LOCAL está igualado?
========================================================================================================

ENTRADAS
--------
  - `cs090_fase8_f803_estructura_shard*.csv`  (estructura + pico de cada variante)
  - `cs090_fase8_f803_pares_elegidos.csv`     (el par de cada grafo, elegido A CIEGAS DE LA MASA)
  - las carpetas de `/Users/alexis/phantom_cs073/bateria_fase8_f803_mismo_pico/`

Reusa `cs090_fase5b_analizar.analizar_carpeta` TAL CUAL (sólo import): la misma extracción de la
fracción de masa en sumideros, κ_V y tiempo del primer sumidero que toda la línea desde Fase V-B.

LAS TRES COMPARACIONES, DECLARADAS ANTES DE MIRAR
--------------------------------------------------
1. **ANCLA (sin controlar el pico)** — promedio de las R realizaciones de `solap` contra el promedio de
   las R de `disp`, dentro de cada grafo. Es el contraste de F7-03/F8-01 rehecho en esta misma batería;
   debería reproducir ~+28 partículas. Si no lo reproduce, no hay nada que interpretar del resto.
2. **CONTROL (pico igualado)** — el par elegido por mínimo |Δpico|. Es el número de la tarea.
3. **MEDIACIÓN sobre las 6R corridas** — regresión de la masa contra (pico local, brazo) con efecto fijo
   de grafo (centrando cada variable en la media de su propio grafo). El coeficiente del brazo con el
   pico ya adentro es "cuánto queda ADEMÁS del pico"; el del pico es "cuánto va A TRAVÉS".

Observable principal declarado de antemano: **fracción de masa en sumideros** (criterio de densidad de
Phantom, `rho_crit_cgs=1000`). Grano: 1 partícula = 0.0005. Piso práctico de un pareado (F8-01): ~5
partículas. No se declara cierre ni veredicto.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)
from cs090_fase5b_analizar import analizar_carpeta          # sólo import, script congelado

BASE = Path("/Users/alexis/phantom_cs073/bateria_fase8_f803_mismo_pico")
GRANO = 0.0005          # 1 partícula de N=2000 en fracción de masa
PICO = "pico_p90_med"


# =============================================================================================
# 1) Verificación cruzada contra meta_regla.json — antes de cualquier estadística
# =============================================================================================
def verificar(carpetas):
    avisos = []
    metas = {}
    for c in carpetas:
        m = json.loads((c / "meta_regla.json").read_text())
        metas[c.name] = m
        if m.get("tarea") != "FASE8_F803_mismo_pico_distinta_topologia":
            avisos.append(f"{c.name}: tarea declarada = {m.get('tarea')!r}")
        esperado = f"{m['rule_id']}_s{m['seed']}_f803_{m['variante']}"
        if c.name != esperado:
            avisos.append(f"{c.name}: el nombre de carpeta no coincide con meta ({esperado})")
        if os.path.basename(str(m.get("carpeta", ""))) != c.name:
            avisos.append(f"{c.name}: carpeta declarada != carpeta real")
        if not m.get("grados_identicos_al_original"):
            avisos.append(f"{c.name}: grados_identicos_al_original = False")
        if m.get("seed_layout") != 12345:
            avisos.append(f"{c.name}: seed_layout = {m.get('seed_layout')}")
        # F7-03 admitía diferencia máxima de 1 triángulo entre brazos (el rebobinado cae al swap más
        # cercano, y un swap puede cerrar más de un triángulo). Se avisa sólo si pasa de ahí, y el
        # valor real se reporta en el informe grafo por grafo.
        if m.get("dif_max_triangulos", 99) > 1:
            avisos.append(f"{c.name}: dif_max_triangulos = {m.get('dif_max_triangulos')}")
    # dentro de cada grafo base: mismas aristas, mismo T, misma masa total, mismo layout
    por_clave = {}
    for n, m in metas.items():
        por_clave.setdefault(f"{m['rule_id']}_s{m['seed']}", []).append(m)
    for clave, ms in por_clave.items():
        for campo in ("n_aristas_grafo_final", "n_triangulos", "seed_layout", "masa_total_ic",
                      "T_objetivo"):
            vals = {m.get(campo) for m in ms}
            if len(vals) != 1:
                avisos.append(f"{clave}: {campo} NO idéntico entre variantes -> {sorted(map(str, vals))}")
    return avisos


# =============================================================================================
# 2) Lectura de Phantom + unión con estructura
# =============================================================================================
def cargar_todo():
    carpetas = sorted(c for c in BASE.glob("*_f803_*")
                      if c.is_dir() and (c / "meta_regla.json").exists())
    avisos = verificar(carpetas)
    print(f"carpetas: {len(carpetas)}   AVISOS de verificación cruzada: "
          f"{len(avisos)}" + ("" if not avisos else "\n  " + "\n  ".join(avisos)))

    filas = []
    for c in carpetas:
        m = json.loads((c / "meta_regla.json").read_text())
        f = analizar_carpeta(c)
        if f.get("fraccion_masa_en_sumideros") is None:
            print(f"  [sin dumps] {c.name}")
            continue
        filas.append(dict(
            carpeta=c.name, rule_id=m["rule_id"], seed=int(m["seed"]), variante=m["variante"],
            brazo=m["brazo"], realizacion=int(m["realizacion"]), lote=m["lote"],
            T_objetivo=m["T_objetivo"], n_aristas=m["n_aristas_grafo_final"],
            n_triangulos=m["n_triangulos"], pico_p90_med=m["pico_p90_med"], pico_cv=m["pico_cv"],
            pico_max_med=m["pico_max_med"],
            frac_aristas_en_triangulo=m["frac_aristas_en_triangulo"],
            tri_por_arista_media=m["tri_por_arista_media"],
            frac_aristas_multi_tri=m["frac_aristas_multi_tri"], gini_tri_nodo=m["gini_tri_nodo"],
            pendiente_corregida=m["pendiente_corregida"],
            frac_masa=f["fraccion_masa_en_sumideros"], n_sumideros=f.get("n_sumideros"),
            kappa_v=f.get("kappa_v_agregado"), t_primer_sumidero=f.get("t_primer_sumidero"),
            n_gas_inicial=f.get("n_gas_inicial"), n_dump_final=f.get("n_dump_final"),
        ))
    D = pd.DataFrame(filas)
    D["clave"] = D["rule_id"] + "_s" + D["seed"].astype(str)
    D.to_csv(f"{HERE}/cs090_fase8_f803_phantom_crudo.csv", index=False)
    print(f"corridas con masa: {len(D)}  -> cs090_fase8_f803_phantom_crudo.csv")
    if (D["n_gas_inicial"] != 2000).any():
        print("  AVISO: alguna corrida no arrancó con 2000 partículas de gas")
    return D


# =============================================================================================
# 3) Estadística pareada (misma forma que F7-03/F8-01)
# =============================================================================================
def pareado(d, etiqueta):
    d = np.asarray(d, dtype=float)
    n = len(d)
    media = float(d.mean())
    ee = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    pos = int((d > 0).sum())
    try:
        w = stats.wilcoxon(d)
        pw = float(w.pvalue)
    except Exception:
        pw = float("nan")
    t = stats.ttest_1samp(d, 0.0) if n > 1 else None
    binom = float(stats.binomtest(pos, n, 0.5).pvalue) if n else float("nan")
    return dict(contraste=etiqueta, n=n, delta_medio=media, delta_particulas=media / GRANO,
                ee_particulas=ee / GRANO, ic95_lo=(media - 1.96 * ee) / GRANO,
                ic95_hi=(media + 1.96 * ee) / GRANO, signos_pos=pos,
                p_wilcoxon=pw, p_t=float(t.pvalue) if t is not None else float("nan"),
                p_binomial=binom)


def main():
    D = cargar_todo()
    S = pd.read_csv(f"{HERE}/cs090_fase8_f803_pares_elegidos.csv")
    sigma_pico = float(S["sigma_pico"].iloc[0])

    # ---------------- 1) ANCLA: promedio de brazo, sin controlar el pico ----------------
    filas_g = []
    for clave, sub in D.groupby("clave"):
        s = sub[sub.brazo == "solap"]
        p = sub[sub.brazo == "disp"]
        if len(s) == 0 or len(p) == 0:
            continue
        filas_g.append(dict(clave=clave, n_solap=len(s), n_disp=len(p),
                            T=sub["T_objetivo"].iloc[0],
                            masa_solap=s["frac_masa"].mean(), masa_disp=p["frac_masa"].mean(),
                            d_masa_ancla=s["frac_masa"].mean() - p["frac_masa"].mean(),
                            d_pico_ancla=s[PICO].mean() - p[PICO].mean(),
                            d_soporte_ancla=(s["frac_aristas_en_triangulo"].mean()
                                             - p["frac_aristas_en_triangulo"].mean()),
                            kappa_solap=s["kappa_v"].mean(), kappa_disp=p["kappa_v"].mean(),
                            t1_solap=s["t_primer_sumidero"].mean(),
                            t1_disp=p["t_primer_sumidero"].mean(),
                            nsum_solap=s["n_sumideros"].mean(), nsum_disp=p["n_sumideros"].mean()))
    G = pd.DataFrame(filas_g)

    # ---------------- 2) CONTROL: el par elegido a ciegas ----------------
    idx = D.set_index(["clave", "variante"])
    filas_c = []
    for _, r in S.iterrows():
        try:
            a = idx.loc[(r["clave"], r["var_solap"])]
            b = idx.loc[(r["clave"], r["var_disp"])]
        except KeyError:
            continue
        filas_c.append(dict(
            clave=r["clave"], T=r["T_objetivo"], var_solap=r["var_solap"], var_disp=r["var_disp"],
            igualado=bool(r["igualado"]), abs_d_pico=r["abs_d_pico"], d_pico=r["d_pico"],
            d_pico_naive=r["abs_d_pico_naive_mismo_indice"],
            pico_solap=a[PICO], pico_disp=b[PICO],
            soporte_solap=a["frac_aristas_en_triangulo"], soporte_disp=b["frac_aristas_en_triangulo"],
            d_soporte=a["frac_aristas_en_triangulo"] - b["frac_aristas_en_triangulo"],
            A_solap=a["tri_por_arista_media"], A_disp=b["tri_por_arista_media"],
            gini_solap=a["gini_tri_nodo"], gini_disp=b["gini_tri_nodo"],
            masa_solap=a["frac_masa"], masa_disp=b["frac_masa"],
            d_masa=a["frac_masa"] - b["frac_masa"],
            d_masa_part=(a["frac_masa"] - b["frac_masa"]) / GRANO,
            kappa_solap=a["kappa_v"], kappa_disp=b["kappa_v"],
            d_kappa=a["kappa_v"] - b["kappa_v"],
            t1_solap=a["t_primer_sumidero"], t1_disp=b["t_primer_sumidero"],
            nsum_solap=a["n_sumideros"], nsum_disp=b["n_sumideros"]))
    C = pd.DataFrame(filas_c)

    G.to_csv(f"{HERE}/cs090_fase8_f803_por_grafo.csv", index=False)
    C.to_csv(f"{HERE}/cs090_fase8_f803_pares_con_masa.csv", index=False)

    # ---------------- 3) pruebas ----------------
    pruebas = [pareado(G["d_masa_ancla"], "ANCLA solap-disp (promedio de R, sin controlar el pico)")]
    if len(C):
        pruebas.append(pareado(C["d_masa"], "CONTROL par elegido (pico igualado por selección)"))
        ci = C[C.igualado]
        cn = C[~C.igualado]
        if len(ci) > 1:
            pruebas.append(pareado(ci["d_masa"], f"CONTROL sólo IGUALADOS (|d_pico|<={sigma_pico:.2f})"))
        if len(cn) > 1:
            pruebas.append(pareado(cn["d_masa"], "CONTROL sólo NO igualados"))
    P = pd.DataFrame(pruebas)
    P.to_csv(f"{HERE}/cs090_fase8_f803_estadistica.csv", index=False)

    print("\n=== ESTADÍSTICA PAREADA (observable: fracción de masa en sumideros) ===")
    print(P.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---------------- 4) mediación sobre las 6R corridas, con efecto fijo de grafo ----------------
    Z = D.copy()
    Z["es_solap"] = (Z.brazo == "solap").astype(float)
    for c in (PICO, "frac_masa", "es_solap", "frac_aristas_en_triangulo"):
        Z[c + "_c"] = Z[c] - Z.groupby("clave")[c].transform("mean")
    X1 = np.column_stack([Z["es_solap_c"], np.ones(len(Z))])
    b1, *_ = np.linalg.lstsq(X1, Z["frac_masa_c"].values, rcond=None)
    X2 = np.column_stack([Z["es_solap_c"], Z[PICO + "_c"], np.ones(len(Z))])
    b2, res2, *_ = np.linalg.lstsq(X2, Z["frac_masa_c"].values, rcond=None)
    # errores estándar del modelo 2
    resid = Z["frac_masa_c"].values - X2 @ b2
    gl = len(Z) - X2.shape[1] - Z["clave"].nunique()
    s2 = float(resid @ resid / gl)
    cov = s2 * np.linalg.inv(X2.T @ X2)
    ee_b = np.sqrt(np.diag(cov))
    rho_pico = stats.spearmanr(Z[PICO + "_c"], Z["frac_masa_c"])
    rho_sop = stats.spearmanr(Z["frac_aristas_en_triangulo_c"], Z["frac_masa_c"])
    med = dict(
        n_corridas=len(Z),
        b_brazo_solo=b1[0], b_brazo_solo_part=b1[0] / GRANO,
        b_brazo_con_pico=b2[0], b_brazo_con_pico_part=b2[0] / GRANO,
        ee_b_brazo_con_pico_part=ee_b[0] / GRANO,
        b_pico=b2[1], frac_efecto_que_sobrevive=(b2[0] / b1[0]) if b1[0] else float("nan"),
        rho_pico_masa_intragrafo=rho_pico.statistic, p_rho_pico=rho_pico.pvalue,
        rho_soporte_masa_intragrafo=rho_sop.statistic, p_rho_soporte=rho_sop.pvalue,
        sigma_pico=sigma_pico)
    pd.DataFrame([med]).to_csv(f"{HERE}/cs090_fase8_f803_mediacion.csv", index=False)
    print("\n=== MEDIACIÓN (todas las corridas, centrado dentro de cada grafo) ===")
    for k, v in med.items():
        print(f"  {k:34s} {v:.5g}" if isinstance(v, float) else f"  {k:34s} {v}")

    # ---------------- 5) figura ----------------
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))

    a = ax[0, 0]
    a.axhline(0, color="k", lw=0.8)
    xs = np.arange(len(G))
    a.bar(xs - 0.2, G["d_masa_ancla"] / GRANO, width=0.4, color="#c0392b",
          label="ancla: promedio solap − promedio disp")
    if len(C):
        Cg = C.set_index("clave").reindex(G["clave"])
        a.bar(xs + 0.2, Cg["d_masa_part"].values, width=0.4, color="#2980b9",
              label="control: par con el pico igualado")
    a.set_xticks(xs)
    a.set_xticklabels([c.split("_s")[0].replace("A2-B0-C2-", "") for c in G["clave"]],
                      rotation=45, ha="right", fontsize=8)
    a.set_ylabel("Δ masa en sumideros [partículas]")
    a.set_title("Por grafo: sin controlar vs con el pico local igualado")
    a.legend(fontsize=8)

    a = ax[0, 1]
    if len(C):
        a.scatter(C["d_pico"], C["d_masa_part"], c=np.where(C["igualado"], "#2980b9", "#95a5a6"),
                  s=60, edgecolor="k", linewidth=0.4, zorder=3, label="par elegido")
        a.scatter(G["d_pico_ancla"], G["d_masa_ancla"] / GRANO, c="#c0392b", marker="^", s=55,
                  alpha=0.85, label="ancla (promedios)")
        if len(C) > 2:
            m, b = np.polyfit(C["d_pico"], C["d_masa_part"], 1)
            xx = np.linspace(min(C["d_pico"].min(), 0), max(C["d_pico"].max(), 0), 20)
            a.plot(xx, m * xx + b, "k--", lw=1,
                   label=f"ajuste: {b:+.1f} part. en Δpico=0")
    a.axhline(0, color="k", lw=0.8)
    a.axvline(0, color="k", lw=0.8)
    a.axvspan(-sigma_pico, sigma_pico, color="#2980b9", alpha=0.10)
    a.set_xlabel("Δ pico local (p90/mediana), solap − disp")
    a.set_ylabel("Δ masa [partículas]")
    a.set_title("La masa contra el pico local igualado\n(banda = ±σ del pico entre realizaciones)")
    a.legend(fontsize=8)

    a = ax[1, 0]
    for br, col in (("solap", "#c0392b"), ("disp", "#2980b9")):
        sub = D[D.brazo == br]
        a.scatter(sub[PICO], sub["frac_masa"] / GRANO, s=32, color=col, alpha=0.75, label=br)
    a.set_xlabel("pico local de la IC (p90/mediana)")
    a.set_ylabel("masa en sumideros [partículas]")
    a.set_title(f"Las {len(D)} corridas: masa contra pico local")
    a.legend(fontsize=8)

    a = ax[1, 1]
    if len(C):
        a.scatter(C["d_soporte"], C["d_masa_part"], c=np.where(C["igualado"], "#2980b9", "#95a5a6"),
                  s=60, edgecolor="k", linewidth=0.4)
    a.axhline(0, color="k", lw=0.8)
    a.set_xlabel("Δ tamaño del soporte (frac. de aristas con triángulo), solap − disp")
    a.set_ylabel("Δ masa [partículas]")
    a.set_title("Con el pico igualado, ¿la masa todavía sigue al soporte?")

    fig.suptitle("F8-03 · mismo pico local, distinta topología — 1 partícula = 0.0005", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{HERE}/cs090_fase8_f803_mismo_pico.png", dpi=130)
    print("\nPNG -> cs090_fase8_f803_mismo_pico.png")

    if len(C):
        print("\n=== TABLA DE PARES (con masa) ===")
        cols = ["clave", "T", "var_solap", "var_disp", "abs_d_pico", "igualado", "d_soporte",
                "masa_solap", "masa_disp", "d_masa_part"]
        print(C[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()

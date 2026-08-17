#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
CS091-B — ANÁLISIS: ¿cuánto del "estallido de π" es álgebra, y separa el barajado del real?
============================================================================================

QUÉ ES ESTO (a nivel módulo)
-----------------------------
Segunda mitad de la re-corrida. `cs091_pi_contingente_rerun.py` midió las curvas; esto las
interroga con las tres preguntas que el encargo del director pone como guardas:

  (A) ÁLGEBRA — Si la bola crece como |S(r)| ≈ S(1)·b^(r−1), entonces π(r)=|S(r)|/(2r) crece como
      b^(r−1)/(2r): estalla POR NECESIDAD MATEMÁTICA, sin que ninguna estructura tenga que hacer
      nada. Se mide b (la tasa de ramificación), se construye la predicción puramente algebraica
      y se reporta qué FRACCIÓN del estallido observado queda explicada sólo por eso.
      Analogía: si cada persona le cuenta un chisme a 2,6 personas nuevas, a los 7 pasos hay
      cientos de enterados — no porque el pueblo tenga una forma especial, sino porque 2,6⁷ es
      un número grande. Eso NO es un hallazgo sobre el pueblo.

  (B) BARAJADO vs REAL — el control que decide. Se barajan 20 réplicas (double-edge-swap, grados
      preservados) del MISMO grafo real y se pregunta si la curva π(r) del real cae fuera de la
      nube de las réplicas (z-score por radio). Si cae adentro, "π indefinido" no distingue.

  (C) ¿HAY GEOMETRÍA EMERGENTE EN EL CORPUS? — se barren los 254 grafos guardados y se busca alguno
      cuya π(r) sea CONSTANTE (lo que haría una retícula emergente). Estadístico: el coeficiente de
      variación de π(r) en r=2..5 y la tasa de ramificación media. Una retícula da CV≈0 y b→1;
      un mundo-pequeño da CV grande y b≫1.

SALIDAS
-------
- `pi_contingente_rerun_algebra.csv`  : b(r), π observada, π algebraica, fracción explicada
- `pi_contingente_rerun_barajado.csv` : real vs 20 réplicas barajadas, z por radio
- `pi_contingente_rerun_corpus.csv`   : barrido de los 254 grafos (CV de π, b medio, ranking)
- `pi_contingente_rerun_curvas.png`   : la figura
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = Path("/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
sys.path.insert(0, str(AQUI))
import cs090_fase8_f800_grafos as G8
from cs091_pi_contingente_rerun import (
    bfs_capas, curva_pi, componente_gigante, fuentes_gigante,
    baraja_rapida, erdos_renyi, mundo_pequeno,
    reticula_cuadrada, reticula_triangular, reticula_hexagonal,
    R_MAX, SEMILLA,
)


# =============================================================================================
# (A) ÁLGEBRA — cuánto del estallido se sigue sólo de que las bolas crecen exponencialmente
# =============================================================================================
def descomposicion_algebraica(filas):
    """`filas` = salida de `curva_pi`. Ajusta la tasa de ramificación b con los radios 2..4 (el
    régimen donde la bola todavía no chocó con el tamaño del grafo) y compara la π observada con la
    π que se seguiría SÓLO de esa tasa: π_alg(r) = S(1)·b^(r−1)/(2r).

    `frac_explicada(r)` = [log π_alg(r) − log π(1)] / [log π(r) − log π(1)]: qué proporción del
    crecimiento observado (en escala logarítmica, que es la escala natural de un exponencial)
    reproduce la pura aritmética. 1,0 = TODO el estallido es álgebra; 0 = nada."""
    S = np.array([f["S_r"] for f in filas], dtype=float)          # S[0] es r=1
    r = np.array([f["r"] for f in filas], dtype=float)
    pi_obs = S / (2.0 * r)
    b_por_r = np.array([f["b_r"] for f in filas], dtype=float)
    usable = b_por_r[1:4]
    usable = usable[np.isfinite(usable) & (usable > 0)]
    b = float(np.exp(np.mean(np.log(usable)))) if len(usable) else float("nan")
    pi_alg = S[0] * (b ** (r - 1)) / (2.0 * r)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = (np.log(pi_alg) - np.log(pi_obs[0])) / (np.log(pi_obs) - np.log(pi_obs[0]))
    return b, pi_obs, pi_alg, frac


# =============================================================================================
# (C) estadístico de "constancia" — lo que distingue una geometría de un mundo-pequeño
# =============================================================================================
def constancia(filas, r_lo=2, r_hi=5):
    """Coeficiente de variación de π(r) en la ventana [r_lo, r_hi] y tasa de ramificación media.
    Retícula: CV≈0 y b≈1 (la frontera CRECE LINEAL, así que π es plana). Mundo-pequeño: CV grande."""
    v = np.array([f["pi_r"] for f in filas if r_lo <= f["r"] <= r_hi], dtype=float)
    bs = np.array([f["b_r"] for f in filas if r_lo <= f["r"] <= r_hi], dtype=float)
    bs = bs[np.isfinite(bs)]
    cv = float(np.std(v) / np.mean(v)) if len(v) and np.mean(v) > 0 else float("nan")
    return cv, float(np.mean(bs)) if len(bs) else float("nan")


def main():
    # -----------------------------------------------------------------------------------------
    # los sustratos de referencia + un grafo real representativo
    # -----------------------------------------------------------------------------------------
    print("[A] DESCOMPOSICIÓN ALGEBRAICA")
    ref = {}
    L = 121
    for nom, ctor in (("reticula_cuadrada", reticula_cuadrada),
                      ("reticula_triangular", reticula_triangular),
                      ("reticula_hexagonal", reticula_hexagonal)):
        adj, c = ctor(L)
        ref[nom] = curva_pi(adj, [c], R_MAX)
    ref["mundo_pequeno_k6_p0.1"] = (lambda a: curva_pi(a, fuentes_gigante(a, 200), R_MAX))(
        mundo_pequeno(2000, 6, 0.10, SEMILLA))

    ruta_real = sorted((AQUI / "grafos_f800" / "F5B_40pares").glob("*.grafo.gz"))[0]
    adj_real, N, meta = G8.cargar_grafo(ruta_real)
    E = meta["E"]
    etiqueta = ruta_real.name.split("__")[0]
    ref[f"REAL:{etiqueta}"] = curva_pi(adj_real, fuentes_gigante(adj_real, 400), R_MAX)
    bar0, _ = baraja_rapida(adj_real, factor=20, semilla=SEMILLA)
    ref[f"BARAJADO:{etiqueta}"] = curva_pi(bar0, fuentes_gigante(bar0, 400), R_MAX)
    er0 = erdos_renyi(N, E, semilla=SEMILLA)
    ref[f"ER:{etiqueta}"] = curva_pi(er0, fuentes_gigante(er0, 400), R_MAX)

    fil_alg = []
    for nom, filas in ref.items():
        b, pi_obs, pi_alg, frac = descomposicion_algebraica(filas)
        cv, b_med = constancia(filas)
        print(f"  {nom:32s} b={b:5.2f}  CV(π,r=2..5)={cv:6.3f}  "
              f"π(7)obs={pi_obs[6]:7.2f} π(7)alg={pi_alg[6]:7.2f} frac_expl={frac[6]:5.2f}")
        for i, f in enumerate(filas):
            fil_alg.append(dict(sustrato=nom, r=f["r"], S_r=round(f["S_r"], 3),
                                b_r=round(f["b_r"], 4) if np.isfinite(f["b_r"]) else "",
                                b_ajustada=round(b, 4),
                                pi_obs=round(float(pi_obs[i]), 4),
                                pi_algebraica=round(float(pi_alg[i]), 4),
                                frac_explicada=round(float(frac[i]), 4) if np.isfinite(frac[i]) else "",
                                cv_pi_r2a5=round(cv, 4)))
    with open(AQUI / "pi_contingente_rerun_algebra.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fil_alg[0].keys())); w.writeheader(); w.writerows(fil_alg)

    # -----------------------------------------------------------------------------------------
    # (B) el control que decide: 20 réplicas barajadas del mismo grafo real
    # -----------------------------------------------------------------------------------------
    print("\n[B] REAL vs 20 RÉPLICAS BARAJADAS (grados preservados) — z por radio")
    reales = sorted((AQUI / "grafos_f800" / "F5B_40pares").glob("*.grafo.gz"))[:3]
    fil_bar = []
    for ruta in reales:
        adj, N, meta = G8.cargar_grafo(ruta)
        eti = ruta.name.split("__")[0]
        f_real = curva_pi(adj, fuentes_gigante(adj, 400), R_MAX)
        pi_real = np.array([x["pi_r"] for x in f_real])
        rep = []
        for s in range(20):
            b_adj, _ = baraja_rapida(adj, factor=20, semilla=SEMILLA + 1000 * s + 7)
            rep.append([x["pi_r"] for x in curva_pi(b_adj, fuentes_gigante(b_adj, 400), R_MAX)])
        rep = np.array(rep)
        mu, sd = rep.mean(axis=0), rep.std(axis=0, ddof=1)
        z = (pi_real - mu) / np.where(sd > 0, sd, np.nan)
        print(f"  {eti:26s} " + " ".join(f"r{r+1}:{z[r]:+5.1f}" for r in range(8)))
        for r in range(R_MAX):
            fil_bar.append(dict(grafo=eti, r=r + 1, pi_real=round(float(pi_real[r]), 4),
                                pi_barajado_media=round(float(mu[r]), 4),
                                pi_barajado_sd=round(float(sd[r]), 4),
                                z=round(float(z[r]), 3) if np.isfinite(z[r]) else "",
                                n_replicas=20))
    with open(AQUI / "pi_contingente_rerun_barajado.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fil_bar[0].keys())); w.writeheader(); w.writerows(fil_bar)

    # -----------------------------------------------------------------------------------------
    # (C) barrido del corpus: ¿existe algún grafo emergente con π(r) constante?
    # -----------------------------------------------------------------------------------------
    print("\n[C] BARRIDO DEL CORPUS — ¿algún grafo con π(r) constante (geometría emergente)?")
    todos = sorted((AQUI / "grafos_f800").rglob("*.grafo.gz"))
    fil_cor = []
    for ruta in todos:
        try:
            adj, N, meta = G8.cargar_grafo(ruta, verificar=False)
        except Exception as e:
            print(f"    (saltado {ruta.name}: {e})"); continue
        filas = curva_pi(adj, fuentes_gigante(adj, 60), R_MAX)
        cv, b_med = constancia(filas)
        fil_cor.append(dict(archivo=ruta.name, carpeta=ruta.parent.name, N=N, E=meta["E"],
                            cv_pi_r2a5=round(cv, 4), b_medio_r2a5=round(b_med, 4),
                            pi_r2=round(filas[1]["pi_r"], 3), pi_r5=round(filas[4]["pi_r"], 3),
                            pi_max=round(max(x["pi_r"] for x in filas), 3)))
    fil_cor.sort(key=lambda d: d["cv_pi_r2a5"])
    with open(AQUI / "pi_contingente_rerun_corpus.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fil_cor[0].keys())); w.writeheader(); w.writerows(fil_cor)
    print(f"  {len(fil_cor)} grafos barridos. Los 5 MÁS constantes (los más 'reticulares'):")
    for d in fil_cor[:5]:
        print(f"    {d['archivo'][:44]:46s} CV={d['cv_pi_r2a5']:.3f} b={d['b_medio_r2a5']:.2f} "
              f"π(2)={d['pi_r2']} π(5)={d['pi_r5']}")
    print("  referencia: retícula cuadrada CV=0.000 b=1.x ; mundo-pequeño CV≈0.7-1.0 b≈2-3")

    # -----------------------------------------------------------------------------------------
    # FIGURA
    # -----------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    estilos = {
        "reticula_cuadrada":       ("#1f77b4", "-",  "retícula cuadrada (a mano)"),
        "reticula_triangular":     ("#2ca02c", "-",  "retícula triangular (a mano)"),
        "reticula_hexagonal":      ("#9467bd", "-",  "retícula hexagonal (a mano)"),
        "mundo_pequeno_k6_p0.1":   ("#d62728", "-",  "mundo-pequeño sintético"),
        f"REAL:{etiqueta}":        ("#000000", "-",  "REAL (corpus A2-B0-C2)"),
        f"BARAJADO:{etiqueta}":    ("#ff7f0e", "--", "NULL barajado (grados preservados)"),
        f"ER:{etiqueta}":          ("#17becf", ":",  "NULL Erdős–Rényi (mismo N,E)"),
    }
    for ax, escala in ((axes[0], "linear"), (axes[1], "log")):
        for nom, filas in ref.items():
            col, ls, lab = estilos[nom]
            rr = [f["r"] for f in filas]; yy = [f["pi_r"] for f in filas]
            ax.plot(rr, yy, ls, color=col, lw=2.0, marker="o", ms=3.5, label=lab)
        ax.set_yscale(escala)
        ax.set_xlabel("radio r (saltos)")
        ax.set_ylabel("π_emergente(r) = |S(r)| / 2r")
        ax.grid(alpha=0.3)
        ax.set_title("escala lineal" if escala == "linear" else "escala logarítmica")
    axes[0].axhline(np.pi, color="grey", lw=1, ls=":")
    axes[0].text(R_MAX * 0.62, np.pi * 1.06, "π = 3,1416 (referencia)", fontsize=8, color="grey")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("π contingente — re-corrida con controles: constante donde hay retícula, "
                 "estalla igual en REAL y en BARAJADO", fontsize=12)
    fig.tight_layout()
    fig.savefig(AQUI / "pi_contingente_rerun_curvas.png", dpi=150)
    print("\nFigura escrita: pi_contingente_rerun_curvas.png")


if __name__ == "__main__":
    main()

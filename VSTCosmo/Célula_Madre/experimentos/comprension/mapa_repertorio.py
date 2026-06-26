#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAPA DEL REPERTORIO VOCAL (read-only) — VER el espacio de gestos y su deriva.
Proyecta los gestos (g_freq/g_intensidad/g_pausa/g_repeticion) a 2D (las dos primeras dimensiones) y los
dibuja: A vs B (colores), con el TIEMPO codificado en transparencia (claro=antiguo, oscuro=reciente).
Permite VER con los ojos si A y B convergen (se solapan) o se diferencian, y cómo deriva el repertorio.
Guarda un PNG en Docker_Historia/. ENV: DOWNSAMPLE(40), OUT.
"""
import os, sys, csv, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HIST = os.path.join(os.path.dirname(RAIZ), "Docker_Historia")
DOWN = int(os.environ.get("DOWNSAMPLE", "40"))
OUT = os.environ.get("OUT", os.path.join(HIST, "MAPA_REPERTORIO.png"))
G = ["g_freq", "g_intensidad", "g_pausa", "g_repeticion"]

def leer(org):
    fs = sorted(glob.glob(os.path.join(HIST, f"organismo_{org}", "fisiologia", "fisiologia_*.csv")))
    pts = []; i = 0
    for fp in fs:
        with open(fp, encoding="utf-8", errors="replace") as fh:
            r = csv.reader(fh); cab = None
            for row in r:
                if not row or row[0].startswith("#"): continue
                if cab is None:
                    cab = row; idx = {c: cab.index(c) for c in (G + ["expr_vocalizando"]) if c in cab}; continue
                i += 1
                if i % DOWN: continue
                if "expr_vocalizando" in idx:
                    try:
                        if float(row[idx["expr_vocalizando"]] or 0) < 0.5: continue   # solo cuando VOCALIZA
                    except Exception: pass
                try:
                    pts.append([float(row[idx[c]] or 0) for c in G])
                except Exception: pass
    return np.array(pts) if pts else np.empty((0, 4))

def main():
    A = leer("ANIMA_A"); B = leer("ANIMA_B")
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (nom, P, col) in zip(axs, [("freq × intensidad", [0, 1], None), ("pausa × repetición", [2, 3], None)]):
        d0, d1 = P
        for org, M, c in [("A", A, "#1f77b4"), ("B", B, "#d62728")]:
            if len(M) == 0: continue
            t = np.linspace(0.12, 0.9, len(M))   # transparencia = tiempo (antiguo→reciente)
            ax.scatter(M[:, d0] + 0.02*np.random.randn(len(M)), M[:, d1] + 0.02*np.random.randn(len(M)),
                       s=8, c=c, alpha=0.0, label=f"{org} (n={len(M)})")
            ax.scatter(M[:, d0] + 0.02*np.random.randn(len(M)), M[:, d1] + 0.02*np.random.randn(len(M)),
                       s=8, color=c, alpha=t * 0.5)
        ax.set_title(nom); ax.set_xlabel(G[d0]); ax.set_ylabel(G[d1])
        ax.axhline(0, color="#ccc", lw=.5); ax.axvline(0, color="#ccc", lw=.5); ax.legend(markerscale=2)
    fig.suptitle("Repertorio vocal — espacio de gestos (azul=A, rojo=B; opacidad=tiempo)", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT, dpi=110)
    # solapamiento numérico (¿convergen?): distancia entre centroides vs dispersión
    if len(A) and len(B):
        cA, cB = A.mean(0), B.mean(0); sep = float(np.linalg.norm(cA - cB))
        disp = float((A.std(0).mean() + B.std(0).mean()) / 2)
        print(f"  separación de centroides A↔B={sep:.3f} · dispersión media={disp:.3f} · ratio={sep/(disp+1e-6):.2f}")
        print("  ratio bajo (<1) = repertorios SOLAPADOS (convergencia); alto = SEPARADOS (diferenciación).")
    print(f"  PNG guardado: {OUT}  ({len(A)} gestos A · {len(B)} gestos B)")

if __name__ == "__main__":
    main()

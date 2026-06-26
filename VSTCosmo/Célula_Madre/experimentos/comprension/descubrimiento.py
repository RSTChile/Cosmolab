#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCUBRIMIENTO NO SUPERVISADO (read-only) — encuentra estructura que NO predefinimos.
  · Agrupa el ESTADO del organismo en regímenes (k-means en numpy, sin sklearn) — "modos de ser" que
    el organismo visita, sin que nosotros definamos las categorías.
  · Busca SECUENCIAS de gestos recurrentes (n-gramas de g_bucket) — "frases" vocales que reaparecen.
  · Da una huella vocal por organismo y la compara (¿A y B convergen o se diferencian?).
NO toca a los organismos. Lee Docker_Historia. ENV: DOWNSAMPLE(60), K(6), N_GRAMA(3).
"""
import os, sys, csv, glob, math, random
from collections import Counter, defaultdict
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HIST = os.path.join(os.path.dirname(RAIZ), "Docker_Historia")
DOWN = int(os.environ.get("DOWNSAMPLE", "60"))
K = int(os.environ.get("K", "6"))
NG = int(os.environ.get("N_GRAMA", "3"))
# estado SIN privilegiar: lo que define "cómo está" el organismo (mismo espíritu que el OrganoExpresion)
EST = ["OI", "necesidad", "H_homeostasis", "met_energia", "voz_arousal", "voz_valence",
       "act_orientacion_deg", "act_atencion_L", "act_atencion_R", "RC_total", "expr_p_voz"]

def leer(org, cols):
    fs = sorted(glob.glob(os.path.join(HIST, f"organismo_{org}", "fisiologia", "fisiologia_*.csv")))
    out = defaultdict(list); i = 0
    for fp in fs:
        with open(fp, encoding="utf-8", errors="replace") as fh:
            r = csv.reader(fh); cab = None
            for row in r:
                if not row or row[0].startswith("#"): continue
                if cab is None:
                    cab = row; idx = {c: cab.index(c) for c in cols if c in cab}; continue
                i += 1
                if i % DOWN: continue
                for c, j in idx.items():
                    out[c].append(row[j])
    return out

def kmeans(X, k, it=30, seed=0):
    rng = random.Random(seed); n = len(X)
    if n < k: return None, None
    C = np.array([X[rng.randrange(n)] for _ in range(k)], dtype=float)
    asg = np.zeros(n, dtype=int)
    for _ in range(it):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)
        asg = d.argmin(1)
        for j in range(k):
            m = X[asg == j]
            if len(m): C[j] = m.mean(0)
    return C, asg

def analiza(org):
    print(f"\n{'='*78}\n  {org}\n{'='*78}")
    d = leer(org, EST + ["g_bucket", "expr_vocalizando"])
    n = min(len(v) for v in d.values()) if d else 0
    if n < 100:
        print("  (datos insuficientes)"); return None
    # 1) REGÍMENES de estado (k-means)
    X = np.array([[float(d[c][i] or 0) if d[c][i] not in ("", None) else 0.0 for c in EST] for i in range(n)])
    mu = X.mean(0); sd = X.std(0) + 1e-6; Xn = (X - mu) / sd
    C, asg = kmeans(Xn, K)
    print(f"\n  REGÍMENES DE ESTADO ({K} modos de ser que el organismo visita, % del tiempo):")
    cnt = Counter(asg.tolist())
    for j, c in cnt.most_common():
        # las 3 variables más DISTINTIVAS de este régimen (mayor desviación del promedio global)
        dist = sorted(zip(EST, C[j]), key=lambda kv: -abs(kv[1]))[:3]
        desc = ", ".join(f"{k}{'↑' if v>0 else '↓'}" for k, v in dist)
        # ¿cuánto vocaliza en este régimen?
        voc = np.array([float(d["expr_vocalizando"][i] or 0) for i in range(n)])[asg == j]
        pv = 100 * (voc >= 0.5).mean() if len(voc) else 0
        print(f"    régimen {j}: {100*c/n:4.1f}% del tiempo · {desc:38s} · habla {pv:3.0f}%")
    # 2) SECUENCIAS de gestos recurrentes (n-gramas de g_bucket, solo cuando vocaliza)
    gb = [d["g_bucket"][i] for i in range(n) if float(d["expr_vocalizando"][i] or 0) >= 0.5 and d["g_bucket"][i] not in ("", "fisio", "·", None)]
    grams = Counter(tuple(gb[i:i+NG]) for i in range(len(gb)-NG))
    print(f"\n  SECUENCIAS VOCALES recurrentes ({NG}-gramas más repetidos — 'frases' que reaparecen):")
    for seq, c in grams.most_common(6):
        print(f"    ×{c:4d}  {' → '.join(seq)}")
    # huella vocal = distribución de gestos (para comparar A vs B)
    huella = Counter(gb)
    return huella, set(grams)

def main():
    print("="*78); print("DESCUBRIMIENTO NO SUPERVISADO — estructura que el organismo formó sin que la definiéramos")
    rA = analiza("ANIMA_A"); rB = analiza("ANIMA_B")
    if rA and rB:
        hA, gA = rA; hB, gB = rB
        # similitud de repertorios (coseno sobre la distribución de gestos)
        todos = set(hA) | set(hB)
        va = np.array([hA.get(g, 0) for g in todos], float); vb = np.array([hB.get(g, 0) for g in todos], float)
        cos = float(va @ vb / (np.linalg.norm(va)*np.linalg.norm(vb) + 1e-9))
        comp = len(gA & gB) / max(1, len(gA | gB))
        print(f"\n{'='*78}\n  A ↔ B (¿convergen o se diferencian?)\n{'='*78}")
        print(f"    similitud de repertorio (coseno): {cos:.3f}   (1=idénticos, 0=distintos)")
        print(f"    secuencias compartidas / totales: {comp:.3f}")
        print("    → alto = convergencia (imitación/cultura común); bajo = estilos individuales (diferenciación).")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOMENTOS INTERESANTES (read-only) — encuentra DÓNDE mirar en una biografía larga.
No hay que ver miles de pasos: este instrumento marca los instantes donde algo SE SALE de la rutina:
  · arousal/valencia vocal por encima del percentil 97 (picos expresivos),
  · GESTO nuevo (un g_bucket que no se había visto en una ventana larga),
  · CONTINGENCIA social alta (alt_contingencia_social — el otro respondió de verdad),
  · SINCRONÍA súbita con el par (salto de correlación corta),
  · transición de SILENCIO↔VOZ marcada (cambio de conducta).
Imprime los top-N momentos con su timestamp y por qué destacan, para luego usar inspector_momento.py.
Lee Docker_Historia. ENV: ORG(ANIMA_A), TOPN(25), DOWNSAMPLE(1).
"""
import os, csv, glob
from collections import deque
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HIST = os.path.join(os.path.dirname(RAIZ), "Docker_Historia")
ORG = os.environ.get("ORG", "ANIMA_A")
TOPN = int(os.environ.get("TOPN", "25"))
DOWN = int(os.environ.get("DOWNSAMPLE", "1"))
COLS = ["ts_real", "voz_arousal", "voz_valence", "expr_vocalizando", "g_bucket",
        "alt_contingencia_social", "OI", "necesidad", "met_energia"]

def leer():
    fs = sorted(glob.glob(os.path.join(HIST, f"organismo_{ORG}", "fisiologia", "fisiologia_*.csv")))
    rows = []; i = 0
    for fp in fs:
        with open(fp, encoding="utf-8", errors="replace") as fh:
            r = csv.reader(fh); cab = None
            for row in r:
                if not row or row[0].startswith("#"): continue
                if cab is None:
                    cab = row; idx = {c: cab.index(c) for c in COLS if c in cab}; continue
                i += 1
                if i % DOWN: continue
                d = {}
                for c, j in idx.items():
                    v = row[j] if j < len(row) else ""
                    d[c] = v
                rows.append(d)
    return rows

def f(x):
    try: return float(x)
    except Exception: return 0.0

def main():
    rows = leer()
    if len(rows) < 50:
        print("  (datos insuficientes)"); return
    aro = np.array([f(r.get("voz_arousal")) for r in rows])
    val = np.array([f(r.get("voz_valence")) for r in rows])
    cont = np.array([f(r.get("alt_contingencia_social")) for r in rows])
    voc = np.array([f(r.get("expr_vocalizando")) for r in rows])
    p_aro = np.percentile(aro, 97); p_cont = np.percentile(cont[cont > 0], 90) if (cont > 0).any() else 1e9
    visto = set(); ventana = deque(maxlen=400)
    eventos = []
    for i, r in enumerate(rows):
        razones = []; score = 0.0
        if aro[i] >= p_aro and p_aro > 0:
            razones.append(f"pico de arousal ({aro[i]:.2f})"); score += aro[i]
        if cont[i] >= p_cont and cont[i] > 0:
            razones.append(f"CONTINGENCIA social ({cont[i]:.2f}) — el otro respondió"); score += 2 + cont[i]
        gb = r.get("g_bucket", "")
        if gb and gb not in ("", "fisio", "·") and gb not in visto and len(ventana) > 200 and voc[i] >= 0.5:
            razones.append(f"GESTO NUEVO ({gb})"); score += 1.5
        if gb: visto.add(gb)
        if i > 0 and abs(voc[i] - voc[i-1]) > 0.5:
            razones.append("transición " + ("SILENCIO→VOZ" if voc[i] > voc[i-1] else "VOZ→SILENCIO")); score += 0.5
        ventana.append(i)
        if razones:
            eventos.append((score, i, r.get("ts_real", "?"), razones))
    eventos.sort(key=lambda e: -e[0])
    print(f"\n{'='*78}\n  MOMENTOS INTERESANTES — {ORG}  ({len(rows)} pasos, top {TOPN})\n{'='*78}")
    print("  (usa el timestamp con: ORG=%s TS='<ts>' python inspector_momento.py)\n" % ORG)
    for score, i, ts, razones in eventos[:TOPN]:
        print(f"  [{ts}]  paso ~{i}  (score {score:.1f})")
        for r in razones: print(f"       · {r}")
    if not eventos:
        print("  (ningún momento se salió de la rutina — biografía homogénea)")

if __name__ == "__main__":
    main()

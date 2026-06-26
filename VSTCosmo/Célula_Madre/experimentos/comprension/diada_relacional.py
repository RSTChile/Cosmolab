#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIADA RELACIONAL (read-only) — mide la RELACIÓN entre A y B, no a cada uno por separado.
Alinea las dos biografías por timestamp (ts_real) y calcula:
  · correlación CRUZADA con desfase (lag) de arousal y de vocalización A↔B — ¿quién lidera a quién?,
    con qué retardo (turnos), y si la sincronía supera a la de un control desplazado (shuffle temporal),
  · CONVERGENCIA del gesto en el tiempo (distancia media entre el gesto de A y el de B por tramos),
  · TURNOS: con qué frecuencia uno habla justo después de que el otro calla (alternancia vs solapamiento).
Es el instrumento del NOSOTROS. Lee Docker_Historia. ENV: DOWNSAMPLE(2), MAXLAG(8), TRAMOS(6).
"""
import os, csv, glob
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HIST = os.path.join(os.path.dirname(RAIZ), "Docker_Historia")
DOWN = int(os.environ.get("DOWNSAMPLE", "2"))
MAXLAG = int(os.environ.get("MAXLAG", "8"))
TRAMOS = int(os.environ.get("TRAMOS", "6"))
G = ["g_freq", "g_intensidad", "g_pausa", "g_repeticion"]
COLS = ["ts_real", "voz_arousal", "expr_vocalizando"] + G

def leer(org):
    fs = sorted(glob.glob(os.path.join(HIST, f"organismo_{org}", "fisiologia", "fisiologia_*.csv")))
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
                rows.append({c: (row[j] if j < len(row) else "") for c, j in idx.items()})
    return rows

def f(x):
    try: return float(x)
    except Exception: return 0.0

def alinear(A, B):
    """Empareja por ts_real (string ordenable); si no hay, empareja por índice."""
    mb = {}
    for r in B:
        ts = r.get("ts_real", "")
        if ts: mb.setdefault(ts, r)
    if mb and any(r.get("ts_real") in mb for r in A):
        pares = [(a, mb[a["ts_real"]]) for a in A if a.get("ts_real") in mb]
    else:
        n = min(len(A), len(B)); pares = list(zip(A[:n], B[:n]))
    return pares

def xcorr(a, b, maxlag):
    a = (a - a.mean()) / (a.std() + 1e-9); b = (b - b.mean()) / (b.std() + 1e-9)
    n = len(a); out = []
    for lag in range(-maxlag, maxlag + 1):
        if lag < 0: x, y = a[:lag], b[-lag:]
        elif lag > 0: x, y = a[lag:], b[:-lag]
        else: x, y = a, b
        m = min(len(x), len(y))
        out.append((lag, float((x[:m] * y[:m]).mean()) if m > 5 else 0.0))
    return out

def main():
    A = leer("ANIMA_A"); B = leer("ANIMA_B")
    pares = alinear(A, B)
    print(f"\n{'='*78}\n  DIADA RELACIONAL — el NOSOTROS A↔B  ({len(pares)} pasos alineados)\n{'='*78}")
    if len(pares) < 50:
        print("  (datos alineados insuficientes — ¿corrieron A y B a la vez?)"); return
    aA = np.array([f(a.get("voz_arousal")) for a, b in pares]); aB = np.array([f(b.get("voz_arousal")) for a, b in pares])
    vA = np.array([f(a.get("expr_vocalizando")) for a, b in pares]); vB = np.array([f(b.get("expr_vocalizando")) for a, b in pares])
    for nom, xa, xb in [("AROUSAL vocal", aA, aB), ("VOCALIZACIÓN (habla/calla)", vA, vB)]:
        xc = xcorr(xa, xb, MAXLAG)
        lag, peak = max(xc, key=lambda lv: abs(lv[1]))
        ctrl = xcorr(xa, np.roll(xb, len(xb)//2), MAXLAG)  # control: B desplazado medio registro
        cpk = max(abs(v) for _, v in ctrl)
        quien = "simultáneo" if lag == 0 else (f"A lidera por {lag}" if lag > 0 else f"B lidera por {-lag}")
        print(f"\n  {nom}: correlación máx {peak:+.3f} en lag {lag:+d} ({quien})")
        print(f"     control (B desplazado): {cpk:.3f}  →  {'REAL > control (sincronía genuina)' if abs(peak) > cpk + 0.05 else 'NO supera al control (sin sincronía clara)'}")
    # convergencia del gesto por tramos
    gA = np.array([[f(a.get(c)) for c in G] for a, b in pares]); gB = np.array([[f(b.get(c)) for c in G] for a, b in pares])
    print("\n  CONVERGENCIA del gesto (distancia media A↔B por tramo; ↓ en el tiempo = se parecen más):")
    L = len(pares) // TRAMOS
    for t in range(TRAMOS):
        s = slice(t*L, (t+1)*L)
        d = float(np.linalg.norm(gA[s] - gB[s], axis=1).mean()) if L > 0 else 0
        print(f"     tramo {t+1}/{TRAMOS}: distancia {d:.3f}  " + "█"*int(d*12))
    # turnos: B habla justo después de que A calla
    al = sum(1 for i in range(1, len(pares)) if vA[i-1] >= .5 and vA[i] < .5 and vB[i] >= .5)
    sol = sum(1 for i in range(len(pares)) if vA[i] >= .5 and vB[i] >= .5)
    print(f"\n  TURNOS: B arranca al callar A {al} veces · solapamiento (ambos hablan) {sol} pasos "
          f"({100*sol/max(1,len(pares)):.0f}% del tiempo)")

if __name__ == "__main__":
    main()

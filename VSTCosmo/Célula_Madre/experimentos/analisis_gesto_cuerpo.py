#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANÁLISIS GESTO ↔ CUERPO (read-only) — ¿el balbuceo es exploración vocal suelta, o parte de un ACTO
corporal integrado? Lee la biografía de la noche y discrimina:
  A) gesto independiente de la orientación  → exploración vocal pura.
  B) gestos asociados a orientaciones fijas  → el sonido es parte de una postura.
  C) gesto + orientación + atención cambian JUNTOS → 189 CONDUCTAS, no 189 sonidos.
Y va más allá: gesto ↔ {orientación, atención, confianza, estado interno (arousal/valencia/OI/necesidad)}.
  ¿el mismo gesto aparece en distintos estados internos?  ¿un mismo estado produce FAMILIAS de gestos?

Mide:
  · η² (eta²): fracción de la varianza de cada variable corporal/interna EXPLICADA por la identidad del gesto.
      η²≈0 → gesto no informa esa variable (escenario A).   η² alto → gesto predice esa variable (B).
  · COINCIDENCIA de cambios: cuando el gesto CAMBIA, ¿cambian también orientación/atención más que al azar?
      ratio≫1 → paquete motor integrado (escenario C).
DOWNSAMPLE (def 50), MIN_N (def 30 muestras por gesto para η²).
"""
import os, sys, csv, glob, math
from collections import defaultdict

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST = os.path.join(os.path.dirname(RAIZ), "Docker_Historia")
DOWN = int(os.environ.get("DOWNSAMPLE", "50"))
MIN_N = int(os.environ.get("MIN_N", "30"))
ORG = os.environ.get("ORG", "ANIMA_A")

CUERPO = ["act_orientacion_deg", "act_objetivo_deg", "act_confianza", "act_atencion_L", "act_atencion_R", "act_fatiga"]
INTERNO = ["voz_arousal", "voz_valence", "OI", "necesidad", "met_energia", "H_homeostasis"]
NUM = CUERPO + INTERNO

def leer(org):
    fs = sorted(glob.glob(os.path.join(HIST, f"organismo_{org}", "fisiologia", "fisiologia_*.csv")))
    g = []; vals = defaultdict(list); i = 0
    for fp in fs:
        with open(fp, encoding="utf-8", errors="replace") as fh:
            r = csv.reader(fh); cab = None
            for row in r:
                if not row or row[0].startswith("#"): continue
                if cab is None:
                    cab = row; idx = {c: cab.index(c) for c in (NUM + ["g_bucket"]) if c in cab}; continue
                i += 1
                if i % DOWN: continue
                gb = row[idx["g_bucket"]] if "g_bucket" in idx else None
                if gb in (None, "", "fisio", "·"): continue
                g.append(gb)
                for c in NUM:
                    try: vals[c].append(float(row[idx[c]]))
                    except Exception: vals[c].append(float("nan"))
    return g, vals

def eta2(g, x):
    # fracción de varianza de x explicada por la identidad del gesto (one-way ANOVA η²)
    by = defaultdict(list)
    for gi, xi in zip(g, x):
        if xi == xi: by[gi].append(xi)
    by = {k: v for k, v in by.items() if len(v) >= MIN_N}
    todos = [xi for v in by.values() for xi in v]
    if len(todos) < 50 or len(by) < 3: return None, len(by)
    gm = sum(todos) / len(todos)
    ss_tot = sum((xi - gm) ** 2 for xi in todos)
    ss_bet = sum(len(v) * (sum(v) / len(v) - gm) ** 2 for v in by.values())
    return (ss_bet / ss_tot if ss_tot > 0 else 0.0), len(by)

def coincidencia(g, vals):
    # cuando el gesto CAMBIA vs no cambia, ¿cuánto cambia cada variable corporal? (escenario C)
    out = {}
    for c in ["act_orientacion_deg", "act_atencion_L", "act_atencion_R", "act_confianza"]:
        x = vals[c]; dch = []; dno = []
        for k in range(1, len(g)):
            if x[k] != x[k] or x[k-1] != x[k-1]: continue
            d = abs(x[k] - x[k-1])
            (dch if g[k] != g[k-1] else dno).append(d)
        mch = sum(dch)/len(dch) if dch else float("nan")
        mno = sum(dno)/len(dno) if dno else float("nan")
        out[c] = (mch, mno, (mch/mno if mno else float("nan")))
    return out

def main():
    g, vals = leer(ORG)
    print("=" * 84)
    print(f"GESTO ↔ CUERPO · {ORG} · {len(g)} muestras (downsample {DOWN}) · gestos distintos: {len(set(g))}")
    print("=" * 84)
    print("\n1) η²  (cuánto la IDENTIDAD del gesto explica cada variable; 0=nada, 1=todo):")
    print("   CUERPO:")
    res = {}
    for c in CUERPO:
        e, ng = eta2(g, vals[c]); res[c] = e
        print(f"     {c:22s} η²={'  n/a' if e is None else f'{e:.3f}'}   (gestos≥{MIN_N}: {ng})")
    print("   ESTADO INTERNO:")
    for c in INTERNO:
        e, ng = eta2(g, vals[c]); res[c] = e
        print(f"     {c:22s} η²={'  n/a' if e is None else f'{e:.3f}'}   (gestos≥{MIN_N}: {ng})")

    print("\n2) COINCIDENCIA de cambios (|Δ| cuando el gesto CAMBIA / cuando NO cambia):")
    co = coincidencia(g, vals)
    for c, (mch, mno, r) in co.items():
        print(f"     {c:22s} cambio={mch:.4f}  no-cambio={mno:.4f}  ratio={'n/a' if r!=r else f'{r:.2f}'}")

    # veredicto escenario
    eta_orient = res.get("act_orientacion_deg") or 0
    eta_aten = max(res.get("act_atencion_L") or 0, res.get("act_atencion_R") or 0)
    ratio_orient = co["act_orientacion_deg"][2]
    print("\n" + "-" * 84)
    print("  VEREDICTO:")
    if eta_orient < 0.05 and (ratio_orient != ratio_orient or ratio_orient < 1.3):
        print("  → ESCENARIO A: el gesto NO explica la orientación ni coincide con sus cambios →")
        print("    el balbuceo es EXPLORACIÓN VOCAL suelta, no acoplada al cuerpo.")
    elif eta_orient >= 0.15:
        print("  → ESCENARIO B: gestos asociados a orientaciones distintas → el sonido es parte de una POSTURA.")
    else:
        print("  → mixto/parcial: revisa η² y coincidencia abajo.")
    if ratio_orient == ratio_orient and ratio_orient >= 1.3:
        print(f"  → señal de ESCENARIO C: al cambiar el gesto, la orientación cambia ×{ratio_orient:.1f} más que al azar")
        print("    (gesto y cuerpo se mueven JUNTOS → conductas, no sólo sonidos).")
    # estado interno: ¿el gesto está más ligado al ESTADO que a la postura?
    eta_int = max((res.get(c) or 0) for c in INTERNO)
    print(f"\n  Nota: η² máximo en ESTADO INTERNO = {eta_int:.3f} (¿el gesto refleja más el estado que la postura?)")
    print("  (η² alto en arousal/valencia/necesidad → 'un mismo estado interno produce familias de gestos'.)")

if __name__ == "__main__":
    main()

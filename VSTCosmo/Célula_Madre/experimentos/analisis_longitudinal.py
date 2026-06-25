#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANÁLISIS LONGITUDINAL (read-only) — leer la biografía acumulada y responder:
  ¿CUÁL variable instrumental (expectativa, agencia, intención, valor ecológico) abandona PRIMERO el
  nivel basal, y cuándo? = el orden REAL de emergencia que predice la genealogía cosmosemiótica.

Lee Docker_Historia/organismo_ANIMA_{A,B}/fisiologia/*.csv (DOWNSAMPLEA por eficiencia), establece el
basal con la 1ª hora (media + 3σ), y detecta la PRIMERA salida SOSTENIDA del basal por variable.
NO modifica nada. Correr al final del periodo:  DOWNSAMPLE=200  python analisis_longitudinal.py
"""
import os, sys, csv, glob, math
from collections import defaultdict

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST = os.path.join(os.path.dirname(RAIZ), "Docker_Historia")
DOWN = int(os.environ.get("DOWNSAMPLE", "200"))    # 1 de cada N filas (~10/s → cada 20s con N=200)
BASE_FRAC = float(os.environ.get("BASE_FRAC", "0.1"))   # fracción inicial que define el basal (1ª parte)
SOSTEN = int(os.environ.get("SOSTEN", "20"))       # muestras consecutivas sobre el basal = salida "sostenida"

# Variables ABSOLUTAS (no el ratio agencia_otro, inestable): la genealogía a vigilar.
VARS = ["expectativa", "alt_contingencia_social", "alt_intencion_comunicativa",
        "voz_otro_valor_ecologico", "voz_otro_confianza_ecologica", "expectativa_confianza"]

def leer(org):
    """Lee (downsampleado) t y las VARS de todos los CSV de fisiología de un organismo, en orden temporal."""
    fs = sorted(glob.glob(os.path.join(HIST, f"organismo_{org}", "fisiologia", "fisiologia_*.csv")))
    serie = defaultdict(list); ts = []
    i = 0
    for fp in fs:
        with open(fp, encoding="utf-8", errors="replace") as fh:
            r = csv.reader(fh); cab = None
            for row in r:
                if not row or row[0].startswith("#"):
                    continue
                if cab is None:
                    cab = row; idx = {c: cab.index(c) for c in VARS if c in cab}
                    it = cab.index("t") if "t" in cab else 0
                    continue
                i += 1
                if i % DOWN:
                    continue
                try:
                    ts.append(float(row[it]))
                except Exception:
                    ts.append(len(ts))
                for v, j in idx.items():
                    try:
                        serie[v].append(float(row[j]))
                    except Exception:
                        serie[v].append(float("nan"))
    return ts, serie

def basal_y_salida(vals):
    vals = [x for x in vals if x == x]
    if len(vals) < 50:
        return None
    nb = max(10, int(len(vals) * BASE_FRAC))
    base = vals[:nb]
    mu = sum(base) / len(base)
    sd = (sum((x - mu) ** 2 for x in base) / len(base)) ** 0.5
    umbral = mu + 3 * sd + 0.02   # + piso para no disparar con ruido casi-cero
    cont = 0
    for i, x in enumerate(vals):
        if x > umbral:
            cont += 1
            if cont >= SOSTEN:
                return {"mu": mu, "sd": sd, "umbral": umbral, "idx_salida": i - SOSTEN + 1,
                        "frac_salida": (i - SOSTEN + 1) / len(vals), "pico": max(vals)}
        else:
            cont = 0
    return {"mu": mu, "sd": sd, "umbral": umbral, "idx_salida": None, "frac_salida": None, "pico": max(vals)}

def main():
    print("=" * 80)
    print("ANÁLISIS LONGITUDINAL — orden de emergencia de las variables instrumentales")
    print("=" * 80)
    salidas = []
    for org in ("ANIMA_A", "ANIMA_B"):
        ts, serie = leer(org)
        if not ts:
            print(f"  {org}: sin biografía aún."); continue
        dur_h = (max(ts) - min(ts)) / 3600.0 if ts else 0
        print(f"\n  {org} · {len(ts)} muestras (downsample {DOWN}) · ~{dur_h:.1f} h de vida")
        for v in VARS:
            r = basal_y_salida(serie.get(v, []))
            if r is None:
                print(f"    {v:30s} (datos insuficientes)"); continue
            if r["idx_salida"] is not None:
                t_h = r["frac_salida"] * dur_h
                print(f"    {v:30s} basal={r['mu']:.4f}  pico={r['pico']:.4f}  → SALIÓ a ~{t_h:.2f} h")
                salidas.append((t_h, f"{org}:{v}", r["pico"]))
            else:
                print(f"    {v:30s} basal={r['mu']:.4f}  pico={r['pico']:.4f}  → en basal (no salió)")
    print("\n  " + "-" * 76)
    print("  ORDEN DE EMERGENCIA (quién abandonó primero el basal):")
    if salidas:
        for t_h, k, pico in sorted(salidas):
            print(f"    {t_h:6.2f} h   {k}  (pico {pico:.4f})")
    else:
        print("    NINGUNA variable salió del basal en el periodo observado.")
        print("    → la voz del otro sigue sin volverse expectativa/agencia/intención. Resultado VÁLIDO:")
        print("      la ausencia persistente es evidencia tan fuerte como la aparición.")

if __name__ == "__main__":
    main()

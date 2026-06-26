#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INSPECTOR DE MOMENTO (read-only) — abre UN instante y muestra el estado COMPLETO de la díada.
Dado un timestamp (o el más reciente), vuelca lado a lado A y B: qué percibían, su estado interno
(OI, necesidad, energía, homeostasis), qué expresaban (voz/silencio, gesto, arousal/valencia),
qué recordaban/imitaban (OAO), agencia/contingencia y valor ecológico — TODO en un solo cuadro.
Es la lupa: momentos_interesantes.py te dice DÓNDE mirar, esto te deja VER ese instante entero.
Lee Docker_Historia. ENV: TS('' = el último), VENTANA(0 = solo ese paso; N = ±N pasos de contexto).
"""
import os, csv, glob

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HIST = os.path.join(os.path.dirname(RAIZ), "Docker_Historia")
TS = os.environ.get("TS", "").strip()
VEN = int(os.environ.get("VENTANA", "0"))
GRUPOS = [
    ("percepción", ["ent_energia", "ent_L", "ent_R", "act_atencion_L", "act_atencion_R", "act_orientacion_deg"]),
    ("estado interno", ["OI", "necesidad", "met_energia", "H_homeostasis", "RC_total", "act_perm"]),
    ("expresión", ["expr_vocalizando", "expr_silencio", "expr_p_voz", "voz_arousal", "voz_valence",
                   "g_bucket", "voz_emitida", "expr_long_conducta"]),
    ("aprendizaje (OAO)", ["oao_oido", "oao_aprendio", "oao_echoica_n", "oao_imitacion_mag"]),
    ("alteridad / valor", ["alt_contingencia_social", "alt_agencia_otro", "voz_otro_valor", "expectativa_exploracion"]),
]
COLS = ["ts_real", "modo_vida"] + [c for _, g in GRUPOS for c in g]

def fila_en(org, ts):
    fs = sorted(glob.glob(os.path.join(HIST, f"organismo_{org}", "fisiologia", "fisiologia_*.csv")))
    if not ts: fs = fs[-1:]          # sin TS: el último archivo basta (rápido)
    ult = None; cabG = None
    for fp in fs:
        with open(fp, encoding="utf-8", errors="replace") as fh:
            r = csv.reader(fh); cab = None
            for row in r:
                if not row or row[0].startswith("#"): continue
                if cab is None: cab = row; cabG = cab; continue
                d = dict(zip(cab, row))
                if ts and d.get("ts_real", "") == ts: return d, cab
                ult = d
    return ult, cabG

def main():
    dA, _ = fila_en("ANIMA_A", TS); dB, _ = fila_en("ANIMA_B", TS)
    if not dA and not dB:
        print("  (no se encontró ese instante)"); return
    ref = dA or dB
    print(f"\n{'='*78}\n  INSPECTOR DE MOMENTO — ts={ref.get('ts_real','?')}  modo={ref.get('modo_vida','?')}\n{'='*78}")
    print(f"  {'':26s} {'ANIMA_A':>22s}   {'ANIMA_B':>22s}")
    for titulo, campos in GRUPOS:
        print(f"\n  ── {titulo} ──")
        for c in campos:
            va = (dA or {}).get(c, "·"); vb = (dB or {}).get(c, "·")
            def fmt(v):
                try: return f"{float(v):.3f}"
                except Exception: return str(v)[:22]
            print(f"  {c:26s} {fmt(va):>22s}   {fmt(vb):>22s}")

if __name__ == "__main__":
    main()

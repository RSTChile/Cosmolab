#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
bateria_actuador_rc.py — BATERÍA del actuador/RC (E018): ¿qué orienta la cabeza?
================================================================================
QUIÉN SOY: batería headless/reproducible para el cuello de botella de orientación
diagnosticado (la decisión = razon_R − razon_L; la comprensión es un producto de 3
factores <1 que colapsa; y la zona muerta (~2.58°) > el objetivo (~0.3°)).

Barre, sin tocar el código de forma permanente (parámetros reversibles, default=baseline):
  · CONDICIONES DE ESTÍMULO: lateral_L / lateral_R (blanco claro) · simetrico (control,
    = caso díada) · paneo (móvil).  → ¿hay asimetría de comprensión que orientar?
  · RC_COMP_GAIN  ∈ {1,2,4,8}  (ganancia de comprensión; 1=baseline)
  · ACT_ZONA_ESCALA ∈ {1,0.3,0.1}  (escala de zona muerta; 1=baseline)

Mide TODA la cadena por corrida: atención→comprensión→razón→decisión_organísmica→
permiso→bloqueo→decision_RC→objetivo→delta→orientación final, y si la cabeza ORIENTÓ.

Pregunta que responde: ¿con un blanco lateral claro la cabeza orienta (y con simétrico
NO)? ¿qué (comp_gain, zona_escala) hace falta? → distingue 'estímulo ambiguo' de
'RC sub-ponderado'. Corre: venv/bin/python3 experimentos/bateria_actuador_rc.py
================================================================================
"""
from __future__ import annotations
import os, sys, json
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
RES = os.path.join(AQUI, "resultados")
sys.path.insert(0, RAIZ)
[sys.path.insert(0, os.path.join(RAIZ, _d)) for _d in ("genoma","campo","organelos","diada","web","audio") if os.path.isdir(os.path.join(RAIZ, _d))]  # Célula Madre en subcarpetas
import VST_CelulaMadre_WebLive_A as A      # actuador + _fila + cmf
import VST_RC_A                            # RC (RC_COMP_GAIN)
cmf = A.cmf
SR, DT = cmf.SR, cmf.DT

CONDS = ["lateral_L", "lateral_R", "simetrico", "paneo"]
GAINS = [1.0, 2.0, 4.0, 8.0]              # ganancia de comprensión (1=baseline)
ZONAS = [1.0, 0.3, 0.1]                   # escala de zona muerta (1=baseline)


def estimulo(cond, seg=8.0, f=300.0):
    """(L, R) sintético. Tono 300 Hz; lateral = un canal con señal, el otro en silencio."""
    n = int(seg * SR); t = np.arange(n) / SR
    tono = 0.3 * np.sin(2 * np.pi * f * t); z = np.zeros(n)
    if cond == "lateral_L":  return tono, z.copy()
    if cond == "lateral_R":  return z.copy(), tono
    if cond == "simetrico":  return tono.copy(), tono.copy()
    if cond == "paneo":
        r = np.linspace(0.0, 1.0, n); return tono * (1.0 - r), tono * r
    raise ValueError(cond)


def correr(cond, comp_gain, zona_escala, sim=8.0):
    VST_RC_A.RC_COMP_GAIN = float(comp_gain)      # parámetros barridos (reversibles)
    A.ACT_ZONA_ESCALA = float(zona_escala)
    A.ORGANO_RC = A.OrganoRC()                    # reset del estado de RC (EMA) por corrida
    L, R = estimulo(cond, seg=sim)
    cel = cmf.celula_madre_funcional((L, R), binaural=True)
    act = A.ActuadorEsferaV122()                  # actuador fresco (theta/fatiga)
    rows = []
    for _ in range(int(sim / DT)):
        cel.vivir_un_paso(DT)
        rows.append(A._fila(cel, act))
    return rows


def resumen(rows, cond, cg, ze):
    a = lambda c: np.array([float(r.get(c, 0.0)) for r in rows])
    n = len(rows); fin = lambda c: a(c)[-max(1, n // 5):].mean()
    orient = fin("act_orientacion_deg")
    return {
        "cond": cond, "comp_gain": cg, "zona_escala": ze,
        "comp_dif_absmax": round(float(np.abs(a("act_comprension_R") - a("act_comprension_L")).max()), 4),
        "razon_L_max": round(float(a("act_razon_L").max()), 4),
        "razon_R_max": round(float(a("act_razon_R").max()), 4),
        "decision_org_absmax": round(float(np.abs(a("act_decision_organismica")).max()), 4),
        "permiso_max": round(float(a("act_permiso_decisional").max()), 4),
        "bloqueo_med": round(float(a("act_bloqueo_IRDE").mean()), 4),
        "decision_rc_absmax": round(float(np.abs(a("act_decision_RC")).max()), 4),
        "objetivo_absmax": round(float(np.abs(a("act_objetivo_deg")).max()), 3),
        "zona_med": round(float(a("act_zona_muerta").mean()), 3),
        "delta_absmax": round(float(np.abs(a("act_delta_deg")).max()), 4),
        "orientacion_fin": round(float(orient), 3),
        "oriento": "SI" if abs(orient) > 1.0 else "no",
    }


def main():
    os.makedirs(RES, exist_ok=True)
    out = []
    print(f"{'cond':10} {'cg':>3} {'ze':>4} | {'comp_dif':>8} {'dec_org':>7} {'dec_rc':>7} "
          f"{'objet°':>7} {'zona°':>6} {'orient°':>8} {'¿oriento?':>9}")
    for cond in CONDS:
        for cg in GAINS:
            for ze in ZONAS:
                r = resumen(correr(cond, cg, ze), cond, cg, ze); out.append(r)
                print(f"{cond:10} {cg:>3.0f} {ze:>4.1f} | {r['comp_dif_absmax']:>8.4f} "
                      f"{r['decision_org_absmax']:>7.4f} {r['decision_rc_absmax']:>7.4f} "
                      f"{r['objetivo_absmax']:>7.2f} {r['zona_med']:>6.2f} "
                      f"{r['orientacion_fin']:>8.2f} {r['oriento']:>9}")
    with open(os.path.join(RES, "bateria_actuador_rc.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    cols = list(out[0].keys())
    with open(os.path.join(RES, "bateria_actuador_rc.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n" + "\n".join(",".join(str(r[c]) for c in cols) for r in out))
    # restaurar baseline al terminar (no dejar el módulo con parámetros barridos)
    VST_RC_A.RC_COMP_GAIN = 1.0; A.ACT_ZONA_ESCALA = 1.0
    orientan = [r for r in out if r["oriento"] == "SI"]
    print(f"\nResumen: {len(orientan)}/{len(out)} configuraciones ORIENTARON la cabeza.")
    print("  → en", os.path.join(RES, "bateria_actuador_rc.csv"))


if __name__ == "__main__":
    main()

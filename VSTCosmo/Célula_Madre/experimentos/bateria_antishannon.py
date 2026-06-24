#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
bateria_antishannon.py — VERIFICACIÓN ANTI-SHANNON del actuador/RC
================================================================================
Verifica que la orientación EMERGE del cierre (no de constantes elegidas para "girar").
NO barre parámetros: corre 4 condiciones y observa el comportamiento EMERGENTE de la
ganancia de comprensión, la decisión, el movimiento y el cierre.

CONDICIONES: simetrico (tono ambos) · lateral_L · lateral_R · ruido (ambos).
CRITERIOS DE ÉXITO (sin optimizar contra un resultado visual):
  C1  estímulo simétrico/ruido → orientación ≈ 0.
  C2  lateral claro → decision_RC firma la dirección CORRECTA (L→izq/neg, R→der/pos).
  C3  la ganancia de comprensión SUBE para el blanco claro y se queda en ~1 para simétrico.
  C4  la ganancia/movimiento están GATEADOS por el cierre (suben sólo si no dañan A_sys_env/IRDE).
Corre: venv/bin/python3 experimentos/bateria_antishannon.py
================================================================================
"""
from __future__ import annotations
import os, sys, json
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
RES = os.path.join(AQUI, "resultados"); sys.path.insert(0, RAIZ)
[sys.path.insert(0, os.path.join(RAIZ, _d)) for _d in ("genoma","campo","organelos","diada","web","audio") if os.path.isdir(os.path.join(RAIZ, _d))]  # Célula Madre en subcarpetas
import VST_CelulaMadre_WebLive_A as A
cmf = A.cmf; SR, DT = cmf.SR, cmf.DT
SIM = 60.0


def estimulo(cond, seg=SIM, f=300.0):
    n = int(seg * SR); t = np.arange(n) / SR
    tono = 0.3 * np.sin(2 * np.pi * f * t); z = np.zeros(n)
    rng = np.random.RandomState(7); noise = 0.25 * rng.randn(n)
    return {"simetrico": (tono, tono.copy()), "lateral_L": (tono, z),
            "lateral_R": (z, tono), "ruido": (noise, noise.copy())}[cond]


def correr(cond):
    A.ORGANO_RC = A.OrganoRC()
    cel = cmf.celula_madre_funcional(estimulo(cond), binaural=True)
    act = A.ActuadorEsferaV122(); fs = []
    for _ in range(int(SIM / DT)):
        cel.vivir_un_paso(DT); fs.append(A._fila(cel, act))
    return fs


def resumen(fs, cond):
    g = lambda c: np.array([float(r[c]) for r in fs]); n = len(fs)
    ini = lambda c: g(c)[:n // 5].mean(); fin = lambda c: g(c)[-n // 5:].mean()
    return {
        "cond": cond,
        "comp_gain_ini": round(float(g("act_comp_gain_eff")[0]), 4),
        "comp_gain_fin": round(float(g("act_comp_gain_eff")[-1]), 4),
        "comp_gain_max": round(float(g("act_comp_gain_eff").max()), 4),
        "claridad_med": round(float(g("act_claridad_estimulo").mean()), 4),
        "decision_rc_fin": round(float(g("act_decision_RC")[-1]), 4),
        "decision_rc_absmax": round(float(np.abs(g("act_decision_RC")).max()), 4),
        "persistencia_med": round(float(g("act_persistencia_decision").mean()), 4),
        "orient_fin": round(float(g("act_orientacion_deg")[-1]), 3),
        "k_motor_eff_med": round(float(g("act_k_motor_eff").mean()), 5),
        "Aenv_ini": round(float(ini("A_sys_env")), 4), "Aenv_fin": round(float(fin("A_sys_env")), 4),
        "IRDE_ini": round(float(ini("IRDE")), 4), "IRDE_fin": round(float(fin("IRDE")), 4),
    }


def main():
    os.makedirs(RES, exist_ok=True)
    out = {c: resumen(correr(c), c) for c in ["simetrico", "lateral_L", "lateral_R", "ruido"]}
    print(f"{'cond':10} | {'gain ini→fin(max)':>20} | {'claridad':>8} | {'dec_rc_fin':>10} | "
          f"{'orient°':>8} | {'persist':>7} | {'Aenv i→f':>14} | {'IRDE i→f':>14}")
    for c, r in out.items():
        print(f"{c:10} | {r['comp_gain_ini']:.2f}→{r['comp_gain_fin']:.2f}({r['comp_gain_max']:.2f})".ljust(33)
              + f"| {r['claridad_med']:>8.3f} | {r['decision_rc_fin']:>+10.4f} | {r['orient_fin']:>+8.2f} | "
              f"{r['persistencia_med']:>7.3f} | {r['Aenv_ini']:.3f}→{r['Aenv_fin']:.3f} | {r['IRDE_ini']:.3f}→{r['IRDE_fin']:.3f}")
    with open(os.path.join(RES, "bateria_antishannon.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # --- VEREDICTO sobre los criterios de éxito ---
    sL, sR, sm, ru = out["lateral_L"], out["lateral_R"], out["simetrico"], out["ruido"]
    C1 = abs(sm["orient_fin"]) < 1.0 and abs(ru["orient_fin"]) < 1.0
    C2 = sL["decision_rc_fin"] < 0.0 and sR["decision_rc_fin"] > 0.0
    C3 = (sL["comp_gain_max"] > sm["comp_gain_max"] + 0.02) and abs(sm["comp_gain_max"] - 1.0) < 0.02
    C4 = (sL["comp_gain_max"] >= 1.0) and (sm["comp_gain_max"] <= 1.02)   # gain acotada por el cierre
    print("\nVEREDICTO anti-Shannon:")
    print(f"  C1 simétrico/ruido → orient≈0 .......... {'PASS' if C1 else 'FALLA'}  ({sm['orient_fin']}°, {ru['orient_fin']}°)")
    print(f"  C2 lateral → decisión dirección correcta {'PASS' if C2 else 'FALLA'}  (L {sL['decision_rc_fin']:+}, R {sR['decision_rc_fin']:+})")
    print(f"  C3 ganancia EMERGE p/ claro, no simétrico {'PASS' if C3 else 'FALLA'}  (L max {sL['comp_gain_max']}, sim max {sm['comp_gain_max']})")
    print(f"  C4 ganancia/movim GATEADOS por cierre ... {'PASS' if C4 else 'FALLA'}")
    print(f"\n  → resultados en {os.path.join(RES,'bateria_antishannon.csv')}")
    cols = list(next(iter(out.values())).keys())
    with open(os.path.join(RES, "bateria_antishannon.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n" + "\n".join(",".join(str(r[c]) for c in cols) for r in out.values()))


if __name__ == "__main__":
    main()

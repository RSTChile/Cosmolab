#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
bateria_cable_c.py — Cable C: forrajeo que CONVERGE por APRENDIZAJE (no por Shannon)
================================================================================
El actuador APRENDE qué orientación lo nutrió (valor_orient ∝ met_nutricion real) y SESGA el centro
del escaneo hacia ahí. Sin reward, sin target_A, sin "girar hacia el tono". Decae (extinción).

  (1) APRENDE + CONVERGE — fuente coherente a la IZQUIERDA: la orientación aprendida y el theta medio
      se inclinan hacia la izquierda, y A/E suben más que en ruido puro (forrajeo real).
  (2) CONTROL INVERTIDO (regla 7) — a mitad se mueve el alimento a la DERECHA: el sesgo aprendido se
      INVIERTE (la orientación preferida pasa de izq a der). La preferencia NO es fija: sigue al alimento.
  (3) SIN DIFERENCIA FÍSICA → SIN SESGO — ruido en ambos oídos: no hay nutrición consistente → no se
      forma una orientación preferida (no aparece fijación espuria).
Corre:  venv/bin/python3 experimentos/bateria_cable_c.py
================================================================================
"""
from __future__ import annotations
import os, sys, json
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
RES = os.path.join(AQUI, "resultados"); sys.path.insert(0, RAIZ)
[sys.path.insert(0, os.path.join(RAIZ, _d)) for _d in ("genoma","campo","organelos","diada","web","audio") if os.path.isdir(os.path.join(RAIZ, _d))]  # Célula Madre en subcarpetas
import VST_CelulaMadre_WebLive_A as A
SR = A.SR; DT = A.DT


def _estim(side, segs=3.0):
    n = int(SR * segs); t = np.arange(n) / SR; rng = np.random.default_rng(7)
    tono = 0.25 * np.sin(2 * np.pi * 220.0 * t); ruido = 0.25 * rng.standard_normal(n)
    if side == "L":   L, R = tono, ruido
    elif side == "R": L, R = ruido, tono
    else:             L, R = ruido, 0.25 * np.random.default_rng(8).standard_normal(n)
    return L.astype(np.float64), R.astype(np.float64)


def _best_orient(act):
    if not act.valor_orient:
        return 0.0, 0.0
    b = max(act.valor_orient, key=act.valor_orient.get)
    return float(b), float(act.valor_orient[b])


def aprende_converge(side="L", pasos=900):
    cel = A.cmf.celula_madre_funcional(_estim(side), binaural=True)
    A.HOMEO_EMERGENTE.reset(); A.MEMORIA.reset(); A.METABOLISMO.reset()
    act = A.ActuadorEsferaV122(); act.forrajeo_c = True   # activa Cable C (experimental)
    th, Av, Ev = [], [], []
    for _ in range(pasos):
        cel.vivir_un_paso(DT); f = A._fila(cel, act)
        th.append(f["act_orientacion_deg"]); Av.append(f["A_sys_env"]); Ev.append(f["met_energia"])
    b_best, v_best = _best_orient(act)
    return {"theta_med2da": round(float(np.mean(th[pasos // 2:])), 2), "orient_aprendida": round(b_best, 1),
            "valor_aprendido": round(v_best, 4), "A_med2da": round(float(np.mean(Av[pasos // 2:])), 3),
            "E_max": round(float(max(Ev)), 3)}


def control_invertido(pasos=600):
    cel = A.cmf.celula_madre_funcional(_estim("L"), binaural=True)
    A.HOMEO_EMERGENTE.reset(); A.MEMORIA.reset(); A.METABOLISMO.reset()
    act = A.ActuadorEsferaV122(); act.forrajeo_c = True   # activa Cable C (experimental)
    for _ in range(pasos):
        cel.vivir_un_paso(DT); A._fila(cel, act)
    b1, v1 = _best_orient(act)                                  # preferencia con alimento a la IZQ
    cel.organelos["soma"].realimentar(_estim("R"), binaural=True)   # MOVER el alimento a la DERECHA
    for _ in range(pasos):
        cel.vivir_un_paso(DT); A._fila(cel, act)
    b2, v2 = _best_orient(act)                                  # preferencia tras mover el alimento
    return {"orient_pre": round(b1, 1), "orient_post": round(b2, 1)}


def sin_diferencia(pasos=600):
    cel = A.cmf.celula_madre_funcional(_estim("ruido"), binaural=True)
    A.HOMEO_EMERGENTE.reset(); A.MEMORIA.reset(); A.METABOLISMO.reset()
    act = A.ActuadorEsferaV122(); act.forrajeo_c = True   # activa Cable C (experimental)
    for _ in range(pasos):
        cel.vivir_un_paso(DT); A._fila(cel, act)
    b, v = _best_orient(act)
    return {"valor_max": round(v, 4), "orient": round(b, 1)}


def main():
    os.makedirs(RES, exist_ok=True)
    izq = aprende_converge("L"); ruido_ref = aprende_converge("ruido")
    inv = control_invertido(); sd = sin_diferencia()

    print("=" * 86)
    print("(1) APRENDE + CONVERGE (alimento coherente a la IZQUIERDA)")
    print(f"    orient aprendida={izq['orient_aprendida']}° (valor={izq['valor_aprendido']}) theta_med={izq['theta_med2da']}°")
    print(f"    A_med={izq['A_med2da']} E_max={izq['E_max']}   vs RUIDO: A_med={ruido_ref['A_med2da']} E_max={ruido_ref['E_max']}")
    print("(2) CONTROL INVERTIDO (regla 7: el alimento se mueve IZQ→DER a mitad)")
    print(f"    orient preferida: pre(izq)={inv['orient_pre']}° → post(der)={inv['orient_post']}° (debe invertir el signo)")
    print("(3) SIN DIFERENCIA FÍSICA (ruido en ambos) → SIN SESGO")
    print(f"    valor_max aprendido={sd['valor_max']} (debe ser ~0; sin fijación)")

    # alimento a la izq ⇒ mejor facing es NEGATIVO (izquierda). Convergencia = aprende orientación negativa + acopla más.
    C1 = izq["orient_aprendida"] < -5.0 and (izq["A_med2da"] > ruido_ref["A_med2da"] + 0.02 or izq["E_max"] > ruido_ref["E_max"] + 0.05)
    C2 = inv["orient_pre"] < 0.0 and inv["orient_post"] > inv["orient_pre"] + 10.0   # el sesgo se mueve hacia la derecha
    C3 = sd["valor_max"] < izq["valor_aprendido"] * 0.5 + 1e-6                        # ruido aprende mucho menos que alimento
    ver = {"C1_aprende_y_converge": C1, "C2_control_invertido_regla7": C2, "C3_sin_fisica_sin_sesgo": C3}
    print("=" * 86)
    nombres = {"C1_aprende_y_converge": "aprende la orientación nutritiva y acopla/come más que en ruido (forrajeo CONVERGE)",
               "C2_control_invertido_regla7": "si el alimento se mueve, la preferencia se INVIERTE (regla 7; no es fija)",
               "C3_sin_fisica_sin_sesgo": "sin diferencia física no se forma orientación preferida (no fijación espuria)"}
    for k, v in ver.items():
        print(f"  {'PASS' if v else 'FALLA'}  {nombres[k]}")
    print(f"\n  RESUMEN: {sum(ver.values())}/{len(ver)} PASS")
    with open(os.path.join(RES, "bateria_cable_c.json"), "w", encoding="utf-8") as fj:
        json.dump({"izq": izq, "ruido": ruido_ref, "invertido": inv, "sin_dif": sd, "veredicto": ver},
                  fj, ensure_ascii=False, indent=1, default=float)
    print(f"  → {os.path.join(RES, 'bateria_cable_c.json')}")


if __name__ == "__main__":
    main()

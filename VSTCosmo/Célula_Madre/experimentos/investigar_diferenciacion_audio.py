#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
investigar_diferenciacion_audio.py — ¿QUÉ propiedades del audio mueven al organismo de forma DISTINTA?

Hallazgo previo: a todo volumen, cualquier sonido COLAPSA el cierre (Δ_struct→0, LF→0, e_R→28) de forma
idéntica → no diferencia. Pero a intensidad VIABLE (~0.01) el organismo alcanza su mejor cierre
(A_sys-env≈0.94, LF≈0.37). Pregunta fundamental (Alexis): buscar hasta encontrar audios a los que reacciona
DISTINTO, para entender qué audios determinan qué respuesta fisiológica.

Método: barre UNA dimensión del audio a la vez (intensidad, frecuencia, estructura temporal, lateralidad),
a intensidad viable, y mide el ESTADO DE CIERRE canónico (Δ_struct, LF, e_R, A_sys-env, Λ_Cos) + el RÉGIMEN
canónico. La dimensión que CAMBIA el régimen es un determinante real de la respuesta fisiológica.

Régimen canónico (cierre, no beneficio/tensión): dos ejes —
  VIABILIDAD = A_sys-env alto Y e_R acotado     ·   ACTIVACIÓN = LF≥κ_LF Y Δ_struct>umbral
    viable + activo  → JARDIN_FERTIL   ·   viable + quieto → CERRADO
    no-viable+ activo→ SELVA_HOSTIL    ·   no-viable+quieto→ COLAPSO
(C-N2.8.8 invariantes de cierre; Λ_Cos=(Δ_struct·LF)/|e_R|·A_sys-env).
"""
from __future__ import annotations
import os, sys
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
[sys.path.insert(0, os.path.join(RAIZ, _d)) for _d in ("genoma", "campo", "organelos", "diada", "web", "audio")
 if os.path.isdir(os.path.join(RAIZ, _d))]
import VST_CelulaMadre_WebLive_A as A
SR = A.SR; DT = A.DT

# umbrales canónicos calibrados a los rangos reales de ANIMA (silencio A≈0.62 e_R≈3.6; viable A≈0.94 e_R≈0.4; colapso A≈0.17 e_R≈28)
U_VIAB_A = 0.45      # A_sys-env por encima → acoplado (viable)
U_VIAB_eR = 6.0      # e_R por debajo → error acotado (κ_O)
U_LF = 0.05          # κ_LF: libertad mínima
U_DELTA = 0.012      # Δ_struct por encima → hay diferenciación


def clasificar_cierre(d):
    """Régimen canónico desde el estado de cierre ABSOLUTO (no el Δ)."""
    viable = (d["A_sys_env"] > U_VIAB_A) and (d["e_R"] < U_VIAB_eR)
    activo = (d["LF"] >= U_LF) and (d["delta_struct"] > U_DELTA)
    if viable and activo:   return "JARDIN_FERTIL"
    if viable and not activo: return "CERRADO"
    if (not viable) and activo: return "SELVA_HOSTIL"
    return "COLAPSO"


def medir(audio_LR, settle=80, prom=30):
    """Estado de cierre promediado en los últimos `prom` pasos tras `settle` (evita el transitorio)."""
    cel = A.cmf.celula_madre_funcional(audio_LR, binaural=True)
    acc = {k: [] for k in ("delta_struct", "LF", "e_R", "A_sys_env", "Lambda_Cos")}
    for i in range(settle):
        cel.vivir_un_paso(DT)
        if i >= settle - prom:
            for k in acc:
                acc[k].append(float(cel.milieu.leer(k, 0.0)))
    d = {k: round(float(np.mean(v)), 4) for k, v in acc.items()}
    d["regimen"] = clasificar_cierre(d)
    return d


# ---------------------------------------------------------------- generadores de estímulo
def tono(f, amp, seg=2.0):
    t = np.arange(int(SR * seg)) / SR; x = amp * np.sin(2 * np.pi * f * t); return (x, x)

def ruido(amp, seg=2.0, seed=3):
    x = amp * np.random.RandomState(seed).standard_normal(int(SR * seg)); return (x, x)

def pulsos(f, amp, periodo_s, seg=2.0):
    t = np.arange(int(SR * seg)) / SR
    env = ((t % periodo_s) < periodo_s / 2).astype(float)
    x = amp * np.sin(2 * np.pi * f * t) * env; return (x, x)

def lateral(f, ampL, ampR, seg=2.0):
    t = np.arange(int(SR * seg)) / SR
    return (ampL * np.sin(2 * np.pi * f * t), ampR * np.sin(2 * np.pi * f * t))


def barrido(nombre, casos):
    print(f"\n=== {nombre} ===")
    print("  estímulo              Δ_struct   LF      e_R     A_sys   Λ_Cos    RÉGIMEN")
    regs = []
    for etiq, audio in casos:
        d = medir(audio); regs.append(d["regimen"])
        print(f"  {etiq:20s} {d['delta_struct']:7}  {d['LF']:6}  {d['e_R']:7}  {d['A_sys_env']:6}  {d['Lambda_Cos']:7}  {d['regimen']}")
    distintos = len(set(regs))
    print(f"  → {distintos} régimen(es) distinto(s) en esta dimensión "
          + ("✅ DIFERENCIA" if distintos > 1 else "✗ no diferencia"))
    return distintos


def main():
    AV = 0.012   # intensidad viable base
    dims = {}
    dims["INTENSIDAD (tono 220Hz)"] = barrido("INTENSIDAD (tono 220Hz)",
        [(f"amp={a}", tono(220, a)) for a in [0.004, 0.008, 0.012, 0.02, 0.04, 0.08]])
    dims["FRECUENCIA (amp viable)"] = barrido("FRECUENCIA (amp viable 0.012)",
        [(f"{f}Hz", tono(f, AV)) for f in [60, 120, 220, 440, 880, 1760]])
    dims["ESTRUCTURA TEMPORAL"] = barrido("ESTRUCTURA TEMPORAL (amp viable)",
        [("sostenido", tono(220, AV)), ("pulso lento 0.5s", pulsos(220, AV, 0.5)),
         ("pulso rápido 0.1s", pulsos(220, AV, 0.1)), ("ruido", ruido(AV))])
    dims["LATERALIDAD"] = barrido("LATERALIDAD (amp viable)",
        [("L=R", lateral(220, AV, AV)), ("solo L", lateral(220, AV, 0.0)),
         ("solo R", lateral(220, 0.0, AV)), ("L>>R", lateral(220, AV, AV * 0.2))])

    print("\n" + "=" * 70)
    print("RESUMEN — qué dimensiones del audio DETERMINAN la respuesta fisiológica:")
    for nombre, n in dims.items():
        print(f"  {'✅' if n > 1 else '✗ '} {nombre}: {n} régimen(es)")
    difs = [k for k, n in dims.items() if n > 1]
    print(f"\n  Diferencian: {', '.join(difs) if difs else 'NINGUNA (el organismo no distingue a esta intensidad)'}")


if __name__ == "__main__":
    main()

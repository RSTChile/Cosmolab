#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
bateria_propiocepcion.py — Cable A AISLADO: ¿orientar la cabeza cambia el acople (A) por FÍSICA?
================================================================================
Verifica el Cable A (propiocepción del soma) SIN hambre ni motor: se fija orient_ext a mano y se
mide A_sys-env. Estímulo binaural ASIMÉTRICO (un oído coherente, otro ruidoso) → enfrentar el oído
de MENOR desajuste debe SUBIR A; el de MAYOR, bajarlo.

CRITERIOS (anti-Shannon):
  C1  ON: A VARÍA con la orientación (orientar IMPORTA físicamente).
  C2  OFF: A es PLANA con la orientación (control: sin el cable, orientar no hace nada).
  C3  INVARIANTE: en el centro (orient=0) A(ON) == A(OFF) == comportamiento histórico EXACTO.
  C4  CONSISTENCIA: la orientación que da MAYOR A enfrenta el oído de MENOR desajuste (física, no premio).
Corre:  venv/bin/python3 experimentos/bateria_propiocepcion.py
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


def _estimulo_asimetrico(segs=3.0):
    n = int(SR * segs); t = np.arange(n) / SR
    L = 0.25 * np.sin(2 * np.pi * 220.0 * t)                      # oído IZQ: tono coherente
    rng = np.random.default_rng(7)
    R = 0.25 * rng.standard_normal(n)                             # oído DER: ruido blanco
    return L.astype(np.float64), R.astype(np.float64)


def _medir(orient, propio, pasos=250):
    cel = A.cmf.celula_madre_funcional(_estimulo_asimetrico(), binaural=True)
    soma = cel.organelos["soma"]
    soma.propiocepcion = propio
    As, gLs, gRs = [], [], []
    for _ in range(pasos):
        soma.orient_ext = float(orient)              # fijamos la cabeza a mano (sin actuador ni hambre)
        cel.vivir_un_paso(DT)
        As.append(cel.milieu.leer("A_sys_env", 0.0))
        gLs.append(abs(soma.omega_A_L)); gRs.append(abs(soma.omega_A_R))
    return (float(np.mean(As[-60:])), float(np.mean(gLs[-60:])), float(np.mean(gRs[-60:])))


def main():
    os.makedirs(RES, exist_ok=True)
    angulos = [-90.0, -45.0, 0.0, 45.0, 90.0]
    on = {a: _medir(a, True) for a in angulos}
    off = {a: _medir(a, False) for a in angulos}

    print("=" * 84)
    print("Cable A AISLADO — estímulo: IZQ=tono coherente, DER=ruido (sin hambre ni motor)")
    print("-" * 84)
    print("  orient | A(ON)   A(OFF)  | desajuste_izq(gL)  desajuste_der(gR)")
    for a in angulos:
        Aon, gL, gR = on[a]; Aoff, _, _ = off[a]
        print("  %+5.0f° | %.4f  %.4f | gL=%.4f          gR=%.4f%s" %
              (a, Aon, Aoff, gL, gR, "   ← centro (histórico)" if a == 0 else ""))

    A_on = [on[a][0] for a in angulos]; A_off = [off[a][0] for a in angulos]
    rango_on = max(A_on) - min(A_on); rango_off = max(A_off) - min(A_off)
    # ¿qué orientación da mayor A, y enfrenta el oído de menor desajuste?
    best = max(angulos, key=lambda a: on[a][0]); gL_b, gR_b = on[best][1], on[best][2]
    f = best / 90.0
    # facing>0 (der) ayuda si gR<gL ; facing<0 (izq) ayuda si gL<gR ; centro no aplica
    consistente = (f == 0) or (f > 0 and gR_b <= gL_b) or (f < 0 and gL_b <= gR_b)

    C1 = rango_on > 0.02
    C2 = rango_off < 1e-6
    C3 = abs(on[0.0][0] - off[0.0][0]) < 1e-9
    C4 = consistente

    print("=" * 84)
    print(f"  rango A(ON)={rango_on:.4f}   rango A(OFF)={rango_off:.2e}")
    print(f"  mejor orientación={best:+.0f}° (A={on[best][0]:.4f}); enfrenta oído de menor desajuste: {consistente}")
    ver = {"C1_orientar_mueve_A": C1, "C2_off_plano": C2, "C3_centro_invariante": C3, "C4_consistencia_fisica": C4}
    nombres = {"C1_orientar_mueve_A": "ON: orientar cambia A (importa físicamente)",
               "C2_off_plano": "OFF: A plana (control — sin cable, orientar no hace nada)",
               "C3_centro_invariante": "centro (0°) = comportamiento histórico EXACTO",
               "C4_consistencia_fisica": "mayor A = enfrentar el oído de menor desajuste (física, no premio)"}
    for k, v in ver.items():
        print(f"  {'PASS' if v else 'FALLA'}  {nombres[k]}")
    print(f"\n  RESUMEN: {sum(ver.values())}/{len(ver)} PASS")
    with open(os.path.join(RES, "bateria_propiocepcion.json"), "w", encoding="utf-8") as fj:
        json.dump({"on": {str(k): v for k, v in on.items()}, "off": {str(k): v for k, v in off.items()},
                   "veredicto": ver}, fj, ensure_ascii=False, indent=1, default=float)
    print(f"  → {os.path.join(RES, 'bateria_propiocepcion.json')}")


if __name__ == "__main__":
    main()

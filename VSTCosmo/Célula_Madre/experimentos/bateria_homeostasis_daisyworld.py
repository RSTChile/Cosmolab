#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
bateria_homeostasis_daisyworld.py — VERIFICACIÓN ANTI-SHANNON de la homeostasis
================================================================================
Verifica que la homeostasis EMERGE del acoplamiento (Daisyworld de Lovelock), sin
setpoint ni ganancia impuestos. NO ajusta constantes para estabilizar.

  (1) BARRIDO LOVELOCK: estrés constante creciente → H robusta en una BANDA (regula),
      colapsa SOLO en extremos (una margarita muere). El equilibrio de x NO está fijado.
  (2) NO-DERIVA: en la célula real, H fluctúa con el estrés pero NO cae monótona a 0
      (el controlador-fantasma anterior derivaba 0.6→0.35).
Corre: venv/bin/python3 experimentos/bateria_homeostasis_daisyworld.py
================================================================================
"""
from __future__ import annotations
import os, sys, json
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
RES = os.path.join(AQUI, "resultados"); sys.path.insert(0, RAIZ)
[sys.path.insert(0, os.path.join(RAIZ, _d)) for _d in ("genoma","campo","organelos","diada","web","audio") if os.path.isdir(os.path.join(RAIZ, _d))]  # Célula Madre en subcarpetas
from VST_Homeostasis import OrganeloHomeostasis, transcribir_celula_homeostatica
from VST_Genoma import Milieu


def lovelock_sweep():
    out = []
    for eR in [0.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0]:
        h = OrganeloHomeostasis(); m = Milieu(); m.secretar("A_sys_env", 1.0); m.secretar("e_R", eR)
        for _ in range(800):
            h.percibir(m); h.metabolizar(0.1, 1.0); h.secretar(m)
        out.append({"e_R": eR, "estres": round(h._estres, 4), "H": round(h.H, 4),
                    "x": round(h.x, 4), "orden_b": round(h.b, 4), "entropia_w": round(h.w, 4)})
    return out


def no_deriva(pasos=2000):
    cel = transcribir_celula_homeostatica(); Hs = []
    for _ in range(pasos):
        cel.vivir_un_paso(0.1); Hs.append(cel.organelos["homeostasis::x_interna"].H)
    Hs = np.array(Hs)
    mitad = Hs[len(Hs) // 2:]
    return {"H_ini": round(float(Hs[:len(Hs)//5].mean()), 4), "H_fin": round(float(Hs[-len(Hs)//5:].mean()), 4),
            "H_min_2a_mitad": round(float(mitad.min()), 4), "H_media_2a_mitad": round(float(mitad.mean()), 4),
            "monotona_decreciente": bool(np.all(np.diff(Hs[::200]) <= 1e-6))}


def escenarios_A():
    """O-N2.1: H ponderada por A_sys-env. Conduzco A con trayectorias controladas (e_R constante)."""
    def corre(A_traj, eR=8.0, pasos=400):
        h = OrganeloHomeostasis(); m = Milieu()
        for k in range(pasos):
            m.secretar("A_sys_env", A_traj(k / pasos)); m.secretar("e_R", eR)
            h.percibir(m); h.metabolizar(0.1, 1.0); h.secretar(m)
        return round(h.H, 4)
    return {"A_estable": corre(lambda f: 0.6),
            "A_sube": corre(lambda f: 0.3 + 0.5 * f),
            "A_cae_autoencierro": corre(lambda f: 0.8 - 0.5 * f),
            "sin_estres": corre(lambda f: 0.6, eR=0.0)}


def main():
    os.makedirs(RES, exist_ok=True)
    sw = lovelock_sweep(); nd = no_deriva(); ea = escenarios_A()
    print("=== (0) H PONDERADA POR A_sys-env (O-N2.1) ===")
    print(f"  A estable={ea['A_estable']}  A sube={ea['A_sube']}  "
          f"A CAE(autoencierro)={ea['A_cae_autoencierro']}  sin estrés={ea['sin_estres']}")
    print("=== (1) BARRIDO LOVELOCK (sin setpoint) ===")
    print(" e_R   estrés |   H     x(desord) orden_b entropia_w")
    for r in sw:
        print("%5.0f  %.3f  | %.3f   %.3f    %.3f    %.3f" % (r["e_R"], r["estres"], r["H"], r["x"], r["orden_b"], r["entropia_w"]))
    print("\n=== (2) NO-DERIVA en célula real ===")
    print(f"  H ini→fin = {nd['H_ini']}→{nd['H_fin']} | H min(2ª mitad)={nd['H_min_2a_mitad']} "
          f"media={nd['H_media_2a_mitad']} | ¿monótona decreciente?={nd['monotona_decreciente']}")

    # --- VEREDICTO ---
    H_band = [r["H"] for r in sw if r["estres"] <= 0.2]           # banda de regulación
    H_extremo = [r["H"] for r in sw if r["estres"] >= 0.8]        # extremos
    x_vals = [r["x"] for r in sw[:5]]                             # ¿x fijo? (setpoint) o emergente
    C1 = min(H_band) >= 0.5 and max(H_band) >= 0.95               # robusta en la banda
    C2 = max(H_extremo) <= 0.05                                   # colapsa en extremos (falla de regulación)
    C3 = (max(x_vals) - min(x_vals)) > 0.02                       # x NO está fijado (no hay setpoint)
    C4 = (not nd["monotona_decreciente"]) and nd["H_media_2a_mitad"] > 0.3   # no deriva monótona a 0
    C5 = ea["A_cae_autoencierro"] < 0.3 and ea["A_estable"] > 0.7 and ea["A_sube"] >= ea["A_estable"]  # H ponderada por A (O-N2.1)
    print("\nVEREDICTO anti-Shannon (Daisyworld):")
    print(f"  C1 H robusta en banda de regulación ..... {'PASS' if C1 else 'FALLA'}  (min {min(H_band)}, max {max(H_band)})")
    print(f"  C2 colapsa solo en extremos (falla real) . {'PASS' if C2 else 'FALLA'}  (H extremo max {max(H_extremo)})")
    print(f"  C3 equilibrio EMERGE (x no fijado) ....... {'PASS' if C3 else 'FALLA'}  (x rango {round(max(x_vals)-min(x_vals),3)})")
    print(f"  C4 NO deriva (vs controlador anterior) ... {'PASS' if C4 else 'FALLA'}")
    print(f"  C5 H ponderada por A_sys-env (O-N2.1) .... {'PASS' if C5 else 'FALLA'}  (autoencierro→{ea['A_cae_autoencierro']}, sano→{ea['A_estable']})")
    with open(os.path.join(RES, "bateria_homeostasis_daisyworld.json"), "w", encoding="utf-8") as f:
        json.dump({"lovelock": sw, "no_deriva": nd}, f, ensure_ascii=False, indent=1)
    print(f"\n  → {os.path.join(RES, 'bateria_homeostasis_daisyworld.json')}")


if __name__ == "__main__":
    main()

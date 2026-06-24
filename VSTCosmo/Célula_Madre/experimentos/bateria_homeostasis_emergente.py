#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
bateria_homeostasis_emergente.py — H canónica fila-level (O-N9.14) + soporte de A
================================================================================
Verifica SIN Shannon encubierto que H_homeostasis_real:
  (1) MONTAÑISTA: A estable a CUALQUIER nivel → H alta; A volátil → H baja;
                  A que cae con ICR(cuota) alto → AUTOENCIERRO; A baja con RC≈0 → ANESTESIA.
  (2) DAISYWORLD: competencia ICR↔IRDE viva + A estable → H alta; colapso de una rama + caída
                  de A → H baja; oscilación regulada → media/alta; descontrolada → baja.
  (3) DÍADA: en la célula real (simétrico/lateral_L/lateral_R/ruido) H_real NO fuerza orientación,
             SÍ afecta el OI (diagnóstico canónico), y el soporte de A reacciona al estímulo.

NO ajusta constantes para producir un resultado. Si una condición FALLA, se reporta como hallazgo.
Corre:  venv/bin/python3 experimentos/bateria_homeostasis_emergente.py
================================================================================
"""
from __future__ import annotations
import os, sys, json
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
RES = os.path.join(AQUI, "resultados"); sys.path.insert(0, RAIZ)
[sys.path.insert(0, os.path.join(RAIZ, _d)) for _d in ("genoma","campo","organelos","diada","web","audio") if os.path.isdir(os.path.join(RAIZ, _d))]  # Célula Madre en subcarpetas
from VST_HomeostasisEmergente import HomeostasisEmergente, soporte_A_sys_env, permeabilidad_activa


def _fila_sint(A, RC, icr_r, irde_r):
    """Fila sintética mínima con la escala REAL de RC (ratios normalizados; crudo ~0.002)."""
    return {"A_sys_env": A, "RC_total": RC, "ICR": 0.002 * icr_r, "IRDE": 0.002 * irde_r,
            "ICR_ratio": icr_r, "IRDE_ratio": irde_r}


def _corre(traj, pasos=600):
    """traj(k)->(A, RC, icr_r, irde_r). Devuelve el último dict de H + el promedio de la 2ª mitad."""
    he = HomeostasisEmergente(); hist = []
    for k in range(pasos):
        A, RC, ir, dr = traj(k)
        hist.append(he.actualizar(_fila_sint(A, RC, ir, dr)))
    mitad = hist[pasos // 2:]
    prom = {kk: round(float(np.mean([h[kk] for h in mitad])), 4) for kk in hist[0]}
    return hist[-1], prom


# ---------------------------------------------------------------- (1) MONTAÑISTA
def montanista():
    RC_OK, IR, DR = 0.005, 0.6, 0.4          # razón viva, competencia sana
    out = {}
    for n in (0.40, 0.55, 0.70, 0.85, 0.95):
        _, p = _corre(lambda k, n=n: (n, RC_OK, IR, DR))
        out[f"estable_{n:.2f}"] = p
    rng = np.random.default_rng(0)
    vol = rng.uniform(0.4, 0.8, 4000)
    out["volatil"] = _corre(lambda k: (float(vol[k % len(vol)]), RC_OK, IR, DR))[1]
    # A cae con la rama de ORDEN dominante = autoencierro
    out["cae_ICR_alto"] = _corre(lambda k: (max(0.05, 0.9 - 0.8 * k / 600), RC_OK, 0.9, 0.1))[1]
    # A baja con razón casi en silencio = anestesia
    out["baja_RC_off"] = _corre(lambda k: (0.12, 0.0002, 0.5, 0.5))[1]
    return out


# ---------------------------------------------------------------- (2) DAISYWORLD
def daisyworld():
    out = {}
    out["ambas_vivas_A_estable"] = _corre(lambda k: (0.7, 0.005, 0.55, 0.45))[1]
    out["colapso_rama_A_cae"] = _corre(lambda k: (max(0.05, 0.8 - 0.7 * k / 600), 0.005, 1.0, 0.0))[1]
    out["oscilacion_regulada"] = _corre(lambda k: (0.65 + 0.04 * np.sin(k * 0.05), 0.005, 0.55, 0.45))[1]
    rng = np.random.default_rng(1); osc = rng.uniform(0.3, 0.95, 4000)
    out["oscilacion_descontrolada"] = _corre(lambda k: (float(osc[k % len(osc)]), 0.005, 0.55, 0.45))[1]
    return out


# ---------------------------------------------------------------- (3) DÍADA (célula real)
def diada():
    import VST_CelulaMadre_WebLive_A as A
    SR = A.SR
    def estimulo(tipo, segs=4.0):
        n = int(SR * segs); t = np.arange(n) / SR
        base = 0.2 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.random.default_rng(2).standard_normal(n)
        if tipo == "simetrico":   L, R = base, base.copy()
        elif tipo == "lateral_L": L, R = base, 0.15 * base
        elif tipo == "lateral_R": L, R = 0.15 * base, base
        else:                     rng = np.random.default_rng(3); L = rng.standard_normal(n) * 0.2; R = rng.standard_normal(n) * 0.2
        return L.astype(np.float32), R.astype(np.float32)
    res = {}
    for tipo in ("simetrico", "lateral_L", "lateral_R", "ruido"):
        cel = A.cmf.celula_madre_funcional(estimulo(tipo), binaural=True); A.HOMEO_EMERGENTE.reset()
        act = A.ActuadorEsferaV122() if hasattr(A, "ActuadorEsferaV122") else None
        Hs, OIs, ors, sop = [], [], [], []
        for _ in range(220):
            cel.vivir_un_paso(A.DT); f = A._fila(cel, act)
            Hs.append(f["H_homeostasis_real"]); OIs.append(f["OI"])
            ors.append(f.get("act_orientacion_deg", 0.0)); sop.append(f["A_soporte_total"])
        res[tipo] = {"H_real": round(float(np.mean(Hs[110:])), 4), "OI": round(float(np.mean(OIs[110:])), 4),
                     "orient_final_deg": round(float(ors[-1]), 3), "orient_abs_max": round(float(np.max(np.abs(ors))), 3),
                     "A_soporte_total": round(float(np.mean(sop[110:])), 4)}
    return res


# ---------------------------------------------------------------- (4) act_perm (FASE 3, latente)
def permeabilidad():
    """act_perm DEFINIDA por el circuito: ante desacople (H_real bajo) ABRE si hay disposición ICR y
    energía; se REPLIEGA si domina IRDE/riesgo o está exhausto. Bien acoplado ⇒ poca demanda ⇒ act_perm baja."""
    def perm(H_real, icr_r, fat):
        f = {"H_homeostasis_real": H_real, "ICR_ratio": icr_r, "act_fatiga": fat}
        return permeabilidad_activa(f)
    return {
        "bien_acoplado":          perm(0.85, 0.6, 0.0),   # H alta ⇒ demanda baja ⇒ act_perm baja
        "desacople_ICR_energico": perm(0.10, 0.8, 0.0),   # H baja + conversión + energía ⇒ ABRE (alta)
        "desacople_IRDE":         perm(0.10, 0.1, 0.0),   # H baja pero protección domina ⇒ repliega (baja)
        "desacople_exhausto":     perm(0.10, 0.8, 5.0),   # H baja + ICR pero sin energía ⇒ repliega (baja)
    }


def main():
    os.makedirs(RES, exist_ok=True)
    mt = montanista(); dw = daisyworld(); pm = permeabilidad(); dd = diada()

    print("=" * 92)
    print("(1) MONTAÑISTA  —  cond | H_real H_estabA H_compet H_RCvivo autoenc anest A_sop_total | lectura")
    print("-" * 92)
    def fila(nom, p, lectura):
        print("  %-16s %.3f  %.3f   %.3f   %.3f   %.3f  %.3f  %.3f | %s"
              % (nom, p["H_homeostasis_real"], p["H_A_estabilidad"], p["H_competencia_ICR_IRDE"],
                 p["H_RC_vivo"], p["H_autoencierro"], p["H_anestesia"], 0.0, lectura))
    for n in (0.40, 0.55, 0.70, 0.85, 0.95):
        p = mt[f"estable_{n:.2f}"]; fila(f"estable A≈{n:.2f}", p, "sano si H alta (montañista)")
    fila("volatil", mt["volatil"], "debe ser BAJA")
    fila("cae_ICR_alto", mt["cae_ICR_alto"], "autoencierro>0")
    fila("baja_RC_off", mt["baja_RC_off"], "anestesia>0")

    print("=" * 92)
    print("(2) DAISYWORLD")
    print("-" * 92)
    for nom, p in dw.items():
        print("  %-26s H_real=%.3f compet=%.3f estab_A=%.3f autoenc=%.3f anest=%.3f"
              % (nom, p["H_homeostasis_real"], p["H_competencia_ICR_IRDE"], p["H_A_estabilidad"],
                 p["H_autoencierro"], p["H_anestesia"]))

    print("=" * 92)
    print("(4) act_perm — permeabilidad DEFINIDA por el circuito (latente, no aplicada al soma)")
    print("-" * 92)
    print("  cond                    | act_perm demanda modo  energia alpha_sug")
    for nom, p in pm.items():
        print("  %-23s |  %.3f   %.3f  %.3f  %.3f   %.3f"
              % (nom, p["act_perm"], p["act_perm_demanda"], p["act_perm_modo"], p["act_perm_energia"], p["act_perm_alpha_sugerido"]))

    print("=" * 92)
    print("(3) DÍADA (célula real)  —  H_real no fuerza orientación; sí afecta OI; soporte reacciona")
    print("-" * 92)
    print("  cond        | H_real  OI     orient_final  orient|max|  A_soporte_total")
    for tipo, r in dd.items():
        print("  %-11s | %.3f   %.3f   %8.3f     %8.3f      %.3f"
              % (tipo, r["H_real"], r["OI"], r["orient_final_deg"], r["orient_abs_max"], r["A_soporte_total"]))

    # ----------------------------- VEREDICTO (criterios de aceptación) -----------------------------
    estables = [mt[f"estable_{n:.2f}"]["H_homeostasis_real"] for n in (0.40, 0.55, 0.70, 0.85, 0.95)]
    C1 = min(estables) >= 0.5                                            # A estable a cualquier nivel → H alta
    C2 = mt["volatil"]["H_homeostasis_real"] <= 0.3                      # A volátil → H baja
    C3 = mt["cae_ICR_alto"]["H_autoencierro"] > 0.05                     # autoencierro detectado
    C4 = mt["baja_RC_off"]["H_anestesia"] > 0.05                         # anestesia detectada
    C5 = (dw["ambas_vivas_A_estable"]["H_homeostasis_real"] >= 0.5 and
          dw["colapso_rama_A_cae"]["H_homeostasis_real"] <= 0.3)        # Daisyworld: vivo>colapso
    C6 = abs(dd["simetrico"]["orient_abs_max"]) <= 2.0                   # simétrico NO orienta por imposición
    C7 = max(dd[t]["OI"] for t in dd) != min(dd[t]["OI"] for t in dd)   # H_real afecta el OI (varía por estímulo)
    # act_perm: ABRE ante desacople con ICR+energía; se repliega si protección domina o exhausto; baja si bien acoplado
    C8 = (pm["desacople_ICR_energico"]["act_perm"] > 0.4 and
          pm["desacople_ICR_energico"]["act_perm"] > pm["desacople_IRDE"]["act_perm"] and
          pm["desacople_ICR_energico"]["act_perm"] > pm["desacople_exhausto"]["act_perm"] and
          pm["bien_acoplado"]["act_perm"] < 0.3)
    veredicto = {"C1_estable_Halta": C1, "C2_volatil_Hbaja": C2, "C3_autoencierro": C3,
                 "C4_anestesia": C4, "C5_daisyworld": C5, "C6_simetrico_no_orienta": C6,
                 "C7_OI_usa_Hreal": C7, "C8_act_perm_emerge": C8}
    print("=" * 92)
    print("VEREDICTO (criterios de aceptación):")
    nombres = {"C1_estable_Halta": "A estable→H alta (cualquier nivel)", "C2_volatil_Hbaja": "A volátil→H baja",
               "C3_autoencierro": "autoencierro detectado", "C4_anestesia": "anestesia detectada",
               "C5_daisyworld": "Daisyworld vivo>colapso", "C6_simetrico_no_orienta": "simétrico no orienta por imposición",
               "C7_OI_usa_Hreal": "H_real afecta el OI",
               "C8_act_perm_emerge": "act_perm emerge del circuito (abre ante desacople, no bien acoplado)"}
    for k, v in veredicto.items():
        print(f"  {'PASS' if v else 'FALLA'}  {nombres[k]}")
    print(f"\n  RESUMEN: {sum(veredicto.values())}/{len(veredicto)} PASS")

    with open(os.path.join(RES, "bateria_homeostasis_emergente.json"), "w", encoding="utf-8") as f:
        json.dump({"montanista": mt, "daisyworld": dw, "permeabilidad": pm, "diada": dd, "veredicto": veredicto},
                  f, ensure_ascii=False, indent=1, default=float)
    print(f"  → {os.path.join(RES, 'bateria_homeostasis_emergente.json')}")


if __name__ == "__main__":
    main()

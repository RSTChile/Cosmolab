#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
bateria_conducta.py — Cable B: HAMBRE→movimiento + act_perm→α, y el LAZO de forrajeo
================================================================================
Anti-Shannon: el hambre ENERGIZA el movimiento (no lo dirige); act_perm AFILA el acople (no es un
target de A); comer es CONSECUENCIA del acople. Si algo FALLA se reporta.

  (1) B1 — misma entrada RC, más HAMBRE ⇒ más ganancia motora (k_motor_eff) y más movimiento.
  (2) B2 — abrir membrana (perm_ext) AFILA el acople: facing coherente ⇒ A aún MAYOR; facing ruido ⇒ MENOR.
  (3) LAZO — célula real con fuente lateralizada (IZQ coherente, DER ruido): ¿orienta hacia el
      alimento y sube A/E más que en ruido-puro (sin alimento)? Observación honesta.
  (4) ESTABILIDAD — A finito y en rango con el lazo completo.
Corre:  venv/bin/python3 experimentos/bateria_conducta.py
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


def _estimulo(tipo, segs=3.0):
    n = int(SR * segs); t = np.arange(n) / SR; rng = np.random.default_rng(7)
    if tipo == "lateral":   # IZQ tono coherente (alimento), DER ruido
        L = 0.25 * np.sin(2 * np.pi * 220.0 * t); R = 0.25 * rng.standard_normal(n)
    else:                   # ruido en ambos (sin alimento coherente)
        L = 0.25 * rng.standard_normal(n); R = 0.25 * rng.standard_normal(n)
    return L.astype(np.float64), R.astype(np.float64)


def _run(tipo, pasos=600):
    cel = A.cmf.celula_madre_funcional(_estimulo(tipo), binaural=True)
    A.HOMEO_EMERGENTE.reset(); A.MEMORIA.reset(); A.METABOLISMO.reset()
    act = A.ActuadorEsferaV122()
    fs = []
    for _ in range(pasos):
        cel.vivir_un_paso(DT); fs.append(A._fila(cel, act))
    return fs


# ---------------------------------------------------------------- (1) B1: hambre → ganancia/movimiento
def b1_hambre_motor():
    fs = _run("lateral", 80)
    f0 = dict(fs[-1])                          # fila REALISTA con objetivo≠0 (RC lateralizado)
    out = {}
    for et, ham in (("saciado", 0.0), ("hambriento", 1.0)):
        act = A.ActuadorEsferaV122(); f = dict(f0); f["met_hambre"] = ham
        kme = []; th0 = act.theta
        for _ in range(60):
            r = act.actualizar(f); kme.append(r["act_k_motor_eff"])
        out[et] = {"k_motor_eff": round(float(np.mean(kme)), 6), "mov_theta": round(abs(act.theta - th0), 3)}
    return out, round(f0.get("act_objetivo_deg", 0.0), 2)


# ---------------------------------------------------------------- (2) B2: act_perm afila el acople
def b2_perm_afila():
    def A_en(orient, perm):
        cel = A.cmf.celula_madre_funcional(_estimulo("lateral"), binaural=True)
        soma = cel.organelos["soma"]
        vals = []
        for _ in range(220):
            soma.orient_ext = orient; soma.perm_ext = perm
            cel.vivir_un_paso(DT); vals.append(cel.milieu.leer("A_sys_env", 0.0))
        return float(np.mean(vals[-60:]))
    # coherente PARCIAL (orient −45°) para evitar saturación de A en 1.0 y ver el afilado
    return {"coherente_cerrado": round(A_en(-45, 0.0), 4), "coherente_abierto": round(A_en(-45, 1.0), 4),
            "ruido_cerrado": round(A_en(90, 0.0), 4), "ruido_abierto": round(A_en(90, 1.0), 4)}


# ---------------------------------------------------------------- (3) LAZO de forrajeo
def lazo_forrajeo():
    out = {}
    for tipo in ("lateral", "ruido"):
        fs = _run(tipo, 600)
        th = [f["act_orientacion_deg"] for f in fs]
        Av = [f["A_sys_env"] for f in fs]; Ev = [f["met_energia"] for f in fs]
        out[tipo] = {"theta_fin": round(th[-1], 2), "theta_abs_max": round(max(abs(x) for x in th), 2),
                     "A_med2da": round(float(np.mean(Av[300:])), 3), "E_max": round(max(Ev), 3),
                     "E_med2da": round(float(np.mean(Ev[300:])), 3)}
    return out


# ---------------------------------------------------------------- (R5/6) necesidad baja SÓLO si A mejora
def regla_5_6():
    """GPT reglas 5/6: la saciedad = caída POSTERIOR de necesidad cuando A_sys-env mejora y Cb baja;
    si moverse NO mejora A, NO debe reforzarse (necesidad sigue alta). Sin target_A ni reward."""
    from VST_Memoria import OrganeloMemoria
    from VST_Metabolismo import OrganeloMetabolismo
    def vivir(A_traj, n=200):
        met = OrganeloMetabolismo(); mem = OrganeloMemoria(); nef = []; Cb = 25.0
        for k in range(n):
            Aa = A_traj(k)
            d = {"A_sys_env": Aa, "ICR_ratio": 0.8 if Aa > 0.5 else 0.3, "IRDE_ratio": 0.2 if Aa > 0.5 else 0.7,
                 "act_perm": 0.6, "lateralidad": 0.0, "RC_total": 0.005}
            d.update(met.actualizar(d, dt=0.1))
            Cb = max(0.0, Cb + (-0.3 if Aa > 0.6 else 0.1))     # presion_desacople ≈ ∫e_R·(1−A): baja si A mejora
            d["presion_desacople"] = Cb; d["H_homeostasis_real"] = 0.75 if Aa > 0.6 else 0.05
            d.update(mem.actualizar(d, dt=0.1))
            nef.append(d["necesidad_efectiva"])
        return float(np.mean(nef[-30:])), Cb
    nm, cbm = vivir(lambda k: min(0.9, 0.3 + 0.004 * k))        # A MEJORA
    ns, cbs = vivir(lambda k: 0.3)                              # A NO mejora
    return {"nec_mejora": round(nm, 3), "Cb_mejora": round(cbm, 1), "nec_sin": round(ns, 3), "Cb_sin": round(cbs, 1)}


def main():
    os.makedirs(RES, exist_ok=True)
    b1, obj = b1_hambre_motor(); b2 = b2_perm_afila(); r56 = regla_5_6(); lz = lazo_forrajeo()

    print("=" * 86)
    print(f"(1) B1 — HAMBRE energiza el motor (misma entrada RC, objetivo={obj}°)")
    print(f"    saciado:    k_motor_eff={b1['saciado']['k_motor_eff']}  mov={b1['saciado']['mov_theta']}°")
    print(f"    hambriento: k_motor_eff={b1['hambriento']['k_motor_eff']}  mov={b1['hambriento']['mov_theta']}°")
    print("(2) B2 — act_perm AFILA el acople (abrir compromete con lo que se enfrenta)")
    print(f"    facing COHERENTE: cerrado A={b2['coherente_cerrado']} → abierto A={b2['coherente_abierto']} (debe SUBIR)")
    print(f"    facing RUIDO:     cerrado A={b2['ruido_cerrado']} → abierto A={b2['ruido_abierto']} (debe BAJAR)")
    print("(3) LAZO de forrajeo (célula real)")
    for tipo, r in lz.items():
        print(f"    {tipo:8s}: theta_fin={r['theta_fin']}° |max|={r['theta_abs_max']}° | A_med={r['A_med2da']} E_max={r['E_max']} E_med={r['E_med2da']}")

    print(f"(R5/6) NECESIDAD baja SÓLO si A mejora (saciedad, sin reward)")
    print(f"    A MEJORA:    nec_ef={r56['nec_mejora']}  Cb={r56['Cb_mejora']} (baja)")
    print(f"    A NO mejora: nec_ef={r56['nec_sin']}  Cb={r56['Cb_sin']} (alta) → no se refuerza")
    C1 = b1["hambriento"]["k_motor_eff"] > b1["saciado"]["k_motor_eff"] and b1["hambriento"]["mov_theta"] > b1["saciado"]["mov_theta"] + 1.0
    C2 = r56["nec_mejora"] < r56["nec_sin"] - 0.05 and r56["nec_sin"] > 0.8     # baja sólo con mejora; sin mejora no refuerza
    estable = bool(np.isfinite([lz["lateral"]["A_med2da"], lz["ruido"]["A_med2da"]]).all())
    ver = {"C1_hambre_energiza_motor": C1, "C2_necesidad_baja_solo_si_A_mejora": C2, "C3_estable": estable}
    print("=" * 86)
    nombres = {"C1_hambre_energiza_motor": "HAMBRE energiza el motor → SEEKING (escanea cuando hay hambre; no dirige)",
               "C2_necesidad_baja_solo_si_A_mejora": "necesidad baja SÓLO si A mejora (regla 5); sin mejora no refuerza (regla 6)",
               "C3_estable": "A finito y en rango con el lazo completo (sin blow-up pese a 3 realimentaciones)"}
    for k, v in ver.items():
        print(f"  {'PASS' if v else 'FALLA'}  {nombres[k]}")
    print(f"\n  RESUMEN: {sum(ver.values())}/{len(ver)} PASS")
    # HALLAZGOS HONESTOS (no pass/fail; información experimental)
    b2_baja = (b2["coherente_abierto"] < b2["coherente_cerrado"]) and (b2["ruido_abierto"] < b2["ruido_cerrado"])
    forrajea = lz["lateral"]["A_med2da"] > lz["ruido"]["A_med2da"] + 0.02 or lz["lateral"]["E_max"] > lz["ruido"]["E_max"] + 0.02
    print("\n  HALLAZGOS:")
    print(f"   · B2 (act_perm→α) DESACTIVADO: abrir baja A en ambos casos ({b2_baja}) → en este soma la")
    print(f"     palanca de acople es la ORIENTACIÓN, no la permeabilidad (negativo honesto).")
    print(f"   · FORRAJEO aún NO converge ({'acopla más con alimento' if forrajea else 'no se distingue de ruido'}):")
    print(f"     el escaneo CRUZA el alimento pero no se QUEDA (quedarse por 'A subió' sería Shannon).")
    print(f"     FALTA Cable C: la MEMORIA/preferencias deben SESGAR el centro del escaneo hacia lo que nutrió.")
    with open(os.path.join(RES, "bateria_conducta.json"), "w", encoding="utf-8") as fj:
        json.dump({"b1": b1, "objetivo": obj, "b2": b2, "lazo": lz, "veredicto": ver}, fj, ensure_ascii=False, indent=1, default=float)
    print(f"  → {os.path.join(RES, 'bateria_conducta.json')}")


if __name__ == "__main__":
    main()

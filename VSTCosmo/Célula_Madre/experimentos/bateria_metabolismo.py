#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
bateria_metabolismo.py — verifica el OrganeloMetabolismo (consumo/costo/degradación/reposición)
================================================================================
SIN Shannon (no hay setpoint de energía; E emerge de ingesta−gasto). Si algo FALLA se reporta.

  (1) NUTRITIVO repone / TÓXICO agota / NEUTRO-silencio drena por basal (no hay almuerzo gratis).
  (2) SACIEDAD DIFERENCIAL (v069): dieta VARIADA sostiene más E que MONO-dieta del mismo alimento.
  (3) PREFERENCIA (v035): tras dieta mixta, la preferencia es mayor para el tipo más nutritivo.
  (4) COMER SACIA LA NECESIDAD: con metabolismo+memoria, comer bien (alta nutrición) BAJA
      necesidad_efectiva; un organismo que no puede comer (desacoplado) la mantiene saturada.
  (5) OBSERVACIÓN célula real: trayectoria de energía (hallazgo, no pass/fail).
Corre:  venv/bin/python3 experimentos/bateria_metabolismo.py
================================================================================
"""
from __future__ import annotations
import os, sys, json
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
RES = os.path.join(AQUI, "resultados"); sys.path.insert(0, RAIZ)
[sys.path.insert(0, os.path.join(RAIZ, _d)) for _d in ("genoma","campo","organelos","diada","web","audio") if os.path.isdir(os.path.join(RAIZ, _d))]  # Célula Madre en subcarpetas
from VST_Metabolismo import OrganeloMetabolismo
from VST_Memoria import OrganeloMemoria


def _f(A, icr, irde, act_perm=0.3, lat=0.0, es=0.26):
    # es = ENERGÍA SEMIÓTICA presente (RC_total). Un alimento (nutritivo/tóxico) la tiene; el silencio no.
    return {"A_sys_env": A, "ICR_ratio": icr, "IRDE_ratio": irde, "act_perm": act_perm, "lateralidad": lat,
            "RC_total": es}


def _corre(met, seq, n=60):
    r = None
    for _ in range(n):
        for f in seq:
            r = met.actualizar(f)
    return r


# ---------------------------------------------------------------- (1) reposición / agotamiento
def reposicion():
    NUT = _f(0.8, 0.8, 0.2); TOX = _f(0.3, 0.2, 0.8, act_perm=0.6); NEU = _f(0.5, 0.5, 0.5, act_perm=0.2, es=0.0)
    e_nut = _corre(OrganeloMetabolismo(), [NUT])["met_energia"]
    e_tox = _corre(OrganeloMetabolismo(), [TOX])["met_energia"]
    e_neu = _corre(OrganeloMetabolismo(), [NEU])["met_energia"]
    return {"nutritivo": e_nut, "toxico": e_tox, "neutro_silencio": e_neu}


# ---------------------------------------------------------------- (2) saciedad diferencial
def saciedad_diferencial():
    mono = _corre(OrganeloMetabolismo(), [_f(0.8, 0.8, 0.2, lat=0.0)])["met_energia"]
    variada = _corre(OrganeloMetabolismo(),
                     [_f(0.8, 0.8, 0.2, lat=0.0), _f(0.8, 0.75, 0.25, lat=-0.3), _f(0.8, 0.7, 0.3, lat=0.3)])["met_energia"]
    return {"mono_dieta": mono, "dieta_variada": variada}


# ---------------------------------------------------------------- (3) preferencia aprendida
def preferencia():
    met = OrganeloMetabolismo()
    # alterna un alimento MUY nutritivo (izq) con uno pobre (der)
    rico = _f(0.8, 0.85, 0.15, lat=-0.3); pobre = _f(0.6, 0.45, 0.55, lat=0.3)
    for _ in range(80):
        met.actualizar(rico); met.actualizar(pobre)
    pref_rico = met.preferencia.get(met._clave_alimento(-0.3, 0.7), 0.0)
    pref_pobre = met.preferencia.get(met._clave_alimento(0.3, -0.1), 0.0)
    return {"pref_rico": round(pref_rico, 3), "pref_pobre": round(pref_pobre, 3)}


# ---------------------------------------------------------------- (4) comer sacia la necesidad
def comer_sacia():
    def vivir(A, icr, irde, Cb=20.0, n=120):
        met = OrganeloMetabolismo(); mem = OrganeloMemoria()
        nef = []
        for _ in range(n):
            d = _f(A, icr, irde, act_perm=0.6); d["RC_total"] = 0.005
            d.update(met.actualizar(d, dt=0.1))
            d["presion_desacople"] = Cb; d["H_homeostasis_real"] = (0.7 if A > 0.6 else 0.05)
            d.update(mem.actualizar(d, dt=0.1, milieu=None, soma=None))
            nef.append(d["necesidad_efectiva"])
        return nef
    bien = vivir(0.85, 0.85, 0.15)     # acoplado y bien alimentado → debe SACIAR (nec_ef baja)
    mal = vivir(0.30, 0.20, 0.80)      # desacoplado, no puede comer → necesidad sigue alta
    return {"nec_ef_bien_alimentado": round(float(np.mean(bien[-20:])), 3),
            "nec_ef_hambriento": round(float(np.mean(mal[-20:])), 3)}


# ---------------------------------------------------------------- (5) célula real (observación)
def celula_real():
    import VST_CelulaMadre_WebLive_A as A
    out = {}
    for spec in ("demo:tono", "demo:clicks"):
        nom, audio = A.cmf.cargar_audio(spec, binaural=True)
        cel = A.cmf.celula_madre_funcional(audio, binaural=True)
        A.HOMEO_EMERGENTE.reset(); A.MEMORIA.reset(); A.METABOLISMO.reset()
        E, IM, nec = [], [], []
        for _ in range(600):
            cel.vivir_un_paso(A.DT); f = A._fila(cel)
            E.append(f["met_energia"]); IM.append(f["met_IM"]); nec.append(f["necesidad_efectiva"])
        out[spec] = {"E_50": round(E[50], 3), "E_300": round(E[300], 3), "E_fin": round(E[-1], 3),
                     "IM_med": round(float(np.mean(IM[250:])), 3), "nec_ef_med": round(float(np.mean(nec[250:])), 3)}
    return out


def niche_sostiene():
    """REBALANCE (a): enfrentar comida debe SOSTENER E (emergente, sin setpoint) → el hambre baja →
    la búsqueda se aquieta por SACIEDAD (no por A). Ruido/nicho pobre → se agota. Orient FIJO (sin actuador)."""
    import VST_CelulaMadre_WebLive_A as A
    SR = A.SR
    n = int(SR * 60.0); t = np.arange(n) / SR; rng = np.random.default_rng(7)   # bloque largo (la corrida son ~50s)
    tono = (0.25 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float64); ru = (0.25 * rng.standard_normal(n)).astype(np.float64)
    def corre(audio, orient):
        cel = A.cmf.celula_madre_funcional(audio, binaural=True)
        A.HOMEO_EMERGENTE.reset(); A.MEMORIA.reset(); A.METABOLISMO.reset()
        s = cel.organelos["soma"]; E = []
        for _ in range(500):
            s.orient_ext = orient; cel.vivir_un_paso(A.DT); f = A._fila(cel); E.append(f["met_energia"])
        return round(float(np.mean(E[250:])), 3)
    sil = np.zeros(n)
    return {"comida_E": corre((tono, sil), 0), "ruido_E": corre((ru, ru), 0)}   # nicho CON energía semiótica (tono) vs ruido


def main():
    os.makedirs(RES, exist_ok=True)
    rep = reposicion(); sac = saciedad_diferencial(); pref = preferencia(); cs = comer_sacia()
    nich = niche_sostiene(); cr = celula_real()

    print("=" * 88)
    print("(1) REPOSICIÓN / AGOTAMIENTO")
    print(f"    nutritivo E={rep['nutritivo']:.3f}  ·  tóxico E={rep['toxico']:.3f}  ·  neutro/silencio E={rep['neutro_silencio']:.3f}")
    print("(2) SACIEDAD DIFERENCIAL (variar > repetir, v069)")
    print(f"    mono-dieta E={sac['mono_dieta']:.3f}  ·  dieta variada E={sac['dieta_variada']:.3f}")
    print("(3) PREFERENCIA APRENDIDA (v035)")
    print(f"    pref(rico)={pref['pref_rico']}  ·  pref(pobre)={pref['pref_pobre']}")
    print("(4) COMER SACIA LA NECESIDAD (lazo necesidad→comer→saciedad)")
    print(f"    nec_ef bien alimentado={cs['nec_ef_bien_alimentado']}  ·  hambriento/desacoplado={cs['nec_ef_hambriento']}")
    print("(5) NICHE SOSTIENE (rebalance a): enfrentar comida SOSTIENE E; ruido se agota (emergente, sin setpoint)")
    print(f"    facing comida E_med={nich['comida_E']}  ·  ruido E_med={nich['ruido_E']}")
    print("(6) CÉLULA REAL móvil (observación)")
    for spec, r in cr.items():
        print(f"    {spec}: E 50→300→fin = {r['E_50']}→{r['E_300']}→{r['E_fin']} | IM={r['IM_med']} nec_ef={r['nec_ef_med']}")

    C1 = rep["nutritivo"] > 0.6 and rep["toxico"] < 0.2 and rep["neutro_silencio"] < rep["nutritivo"]
    C2 = sac["dieta_variada"] > sac["mono_dieta"]
    C3 = pref["pref_rico"] > pref["pref_pobre"]
    C4 = cs["nec_ef_bien_alimentado"] < cs["nec_ef_hambriento"]
    C5 = nich["comida_E"] > 0.2 and nich["comida_E"] > nich["ruido_E"] + 0.05   # tono sostiene MÁS (aunque el ruido aún nutre algo)
    ver = {"C1_reposicion_agotamiento": C1, "C2_saciedad_diferencial": C2,
           "C3_preferencia": C3, "C4_comer_sacia_necesidad": C4, "C5_niche_sostiene_E": C5}
    print("=" * 88)
    nombres = {"C1_reposicion_agotamiento": "nutritivo repone / tóxico agota / silencio drena (no hay almuerzo gratis)",
               "C2_saciedad_diferencial": "dieta variada sostiene más que mono-dieta (v069)",
               "C3_preferencia": "preferencia aprendida hacia el alimento más nutritivo (v035)",
               "C4_comer_sacia_necesidad": "comer bien BAJA la necesidad; no poder comer la mantiene",
               "C5_niche_sostiene_E": "enfrentar comida SOSTIENE E (emergente); ruido se agota → la saciedad ya puede aquietar la búsqueda"}
    for k, val in ver.items():
        print(f"  {'PASS' if val else 'FALLA'}  {nombres[k]}")
    print(f"\n  RESUMEN: {sum(ver.values())}/{len(ver)} PASS")
    print("\n  NOTA (6): facing FIJO la comida sostiene E (C5); en la célula MÓVIL aún oscila porque la cabeza")
    print("  no se QUEDA en la comida (eso es Cable C, pendiente: el forrajeo aún no converge).")
    with open(os.path.join(RES, "bateria_metabolismo.json"), "w", encoding="utf-8") as fj:
        json.dump({"reposicion": rep, "saciedad": sac, "preferencia": pref, "comer_sacia": cs,
                   "niche": nich, "celula_real": cr, "veredicto": ver}, fj, ensure_ascii=False, indent=1, default=float)
    print(f"  → {os.path.join(RES, 'bateria_metabolismo.json')}")


if __name__ == "__main__":
    main()

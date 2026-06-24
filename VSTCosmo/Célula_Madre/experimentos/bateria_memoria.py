#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
bateria_memoria.py — verifica el OrganeloMemoria (6 capas + necesidad Cb→act_perm)
================================================================================
SIN Shannon. NO ajusta para producir un resultado; si una condición FALLA se reporta.

  (1) EPISÓDICA + RECALL EXPLÍCITO — resuelve el 0/50 de v180: un evento saliente se GRABA y
      luego se RECUPERA al revisitar su clave; NO se recupera en una clave distinta.
  (2) NECESIDAD ACUMULATIVA + REFRACTARIEDAD — necesidad sube con Cb (presión acumulada);
      tras re-acople (saciedad) necesidad_efectiva CAE por debajo de necesidad.
  (3) VALOR DOBLE-DECAY — tras dejar de visitar un estado, el valor de CORTO decae más rápido
      que el de LARGO (hábito vs identidad).
  (4) PERSISTENCIA — en ausencia la confianza decae exp(−t/τ); τ es MAYOR con más vida vivida.
  (5) CAPA 5 (célula real) — familiaridad/novedad/carga_estructural salen del soma (implícita→consultable).
Corre:  venv/bin/python3 experimentos/bateria_memoria.py
================================================================================
"""
from __future__ import annotations
import os, sys, json
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
RES = os.path.join(AQUI, "resultados"); sys.path.insert(0, RAIZ)
[sys.path.insert(0, os.path.join(RAIZ, _d)) for _d in ("genoma","campo","organelos","diada","web","audio") if os.path.isdir(os.path.join(RAIZ, _d))]  # Célula Madre en subcarpetas
from VST_Memoria import OrganeloMemoria, TIPO


def _fila(A, lat, e_R, icr_r, irde_r, act_perm, H_real, Cb=0.0, fat=0.0):
    return {"A_sys_env": A, "lateralidad": lat, "e_R": e_R, "ICR_ratio": icr_r, "IRDE_ratio": irde_r,
            "act_perm": act_perm, "H_homeostasis_real": H_real, "presion_desacople": Cb, "act_fatiga": fat}


# ---------------------------------------------------------------- (1) episódica + recall
def episodica_recall():
    mem = OrganeloMemoria()
    # Evento SALIENTE de amenaza en clave (lat=-0.3→izq): caída AGUDA de A (>escala_dA/paso) + riesgo
    # alto + energía. Una amenaza es un desacople ABRUPTO; una caída lenta NO es amenaza (correcto).
    A = 0.9
    for _ in range(40):
        A = max(0.2, A - 0.06)
        mem.actualizar(_fila(A, -0.3, 8.0, 0.1, 0.9, 0.7, 0.1, Cb=3.0, fat=0.0), dt=0.1)
    grabados = len(mem.episodios)
    # Revisitar la MISMA clave en calma → debe RECUPERAR
    r_mismo = mem.actualizar(_fila(0.5, -0.3, 1.0, 0.6, 0.4, 0.2, 0.5, Cb=1.0, fat=0.0), dt=0.1)
    # Visitar una clave DISTINTA (der) → NO debe recuperar
    r_otro = mem.actualizar(_fila(0.5, 0.3, 1.0, 0.6, 0.4, 0.2, 0.5, Cb=1.0, fat=0.0), dt=0.1)
    return {"episodios_grabados": grabados, "recall_mismo": r_mismo["mem_recall"],
            "tipo_mismo": r_mismo["mem_recall_tipo"], "costo_mismo": r_mismo["mem_recall_costo"],
            "recall_otro": r_otro["mem_recall"]}


# ---------------------------------------------------------------- (2) necesidad + refractariedad
def necesidad_refractariedad():
    mem = OrganeloMemoria()
    # presión acumulada creciente (Cb sube), desacople sostenido → necesidad debe subir
    nec = []
    for k in range(60):
        Cb = 0.2 * k
        r = mem.actualizar(_fila(0.4, -0.2, 6.0, 0.5, 0.5, 0.6, 0.15, Cb=Cb, fat=0.0), dt=0.1)
        nec.append(r["necesidad"])
    nec_pico = max(nec)
    # RE-ACOPLE: A sube fuerte con H alta (satisfacción) → saciedad sube → necesidad_efectiva cae
    sac, nef, n = [], [], []
    A = 0.4
    for _ in range(40):
        A = min(0.95, A + 0.04)
        r = mem.actualizar(_fila(A, -0.2, 1.0, 0.6, 0.4, 0.6, 0.85, Cb=2.0, fat=0.0), dt=0.1)
        sac.append(r["mem_saciedad"]); nef.append(r["necesidad_efectiva"]); n.append(r["necesidad"])
    return {"necesidad_inicial": round(nec[0], 3), "necesidad_pico": round(nec_pico, 3),
            "saciedad_post_reacople": round(max(sac), 3),
            "necesidad_post": round(n[-1], 3), "necesidad_efectiva_post": round(nef[-1], 3)}


# ---------------------------------------------------------------- (3) valor doble-decay
def valor_doble_decay():
    mem = OrganeloMemoria()
    # Visita repetida de una clave BUENA (H alta) → consolida corto y largo
    for _ in range(80):
        mem.actualizar(_fila(0.8, 0.0, 1.0, 0.7, 0.3, 0.2, 0.9, Cb=0.5, fat=0.0), dt=0.1)
    clave = mem._clave(0.0, 0.8)
    v0 = dict(mem.valencia[clave])
    # Dejar de visitar (visitar OTRA clave) durante 60s → corto cae más que largo
    for _ in range(600):
        mem.actualizar(_fila(0.3, 0.5, 1.0, 0.4, 0.6, 0.3, 0.2, Cb=0.5, fat=0.0), dt=0.1)
    v1 = dict(mem.valencia[clave])
    ret_corto = v1["corto"] / max(1e-6, v0["corto"])
    ret_largo = v1["largo"] / max(1e-6, v0["largo"])
    return {"corto_0": round(v0["corto"], 3), "largo_0": round(v0["largo"], 3),
            "corto_ret": round(ret_corto, 3), "largo_ret": round(ret_largo, 3)}


# ---------------------------------------------------------------- (4) persistencia y vida
def persistencia():
    def decae(vida):
        mem = OrganeloMemoria()
        mem.actualizar(_fila(0.6, -0.3, 8.0, 0.6, 0.4, 0.3, 0.5), dt=0.1)   # estímulo presente
        # ausencia: sin estímulo (e_R≈0, lat≈0). milieu=None → vida via self.t; forzamos vida con t
        conf = []
        for _ in range(50):
            # inyectamos 'vida' alta vía un milieu falso
            class _M:  # milieu mínimo que entrega 'historia'=vida y presion/fatiga 0
                def leer(self, k, d=0.0): return vida if k == "historia" else 0.0
                def secretar(self, *a, **k): pass
            r = mem.actualizar(_fila(0.6, 0.0, 0.0, 0.5, 0.5, 0.2, 0.5), dt=0.1, milieu=_M())
            conf.append(r["mem_persistencia"])
        return conf
    joven = decae(1.0); viejo = decae(500.0)
    return {"conf_joven_t5s": round(joven[-1], 3), "conf_viejo_t5s": round(viejo[-1], 3)}


# ---------------------------------------------------------------- (5) capa 5 célula real
def capa5_real():
    import VST_CelulaMadre_WebLive_A as A
    out = {}
    for spec in ("demo:tono", "demo:rosa"):
        nom, audio = A.cmf.cargar_audio(spec, binaural=True)
        cel = A.cmf.celula_madre_funcional(audio, binaural=True); A.HOMEO_EMERGENTE.reset(); A.MEMORIA.reset()
        fam, nov, carga, nece, epis = [], [], [], [], 0
        for _ in range(500):
            cel.vivir_un_paso(A.DT); f = A._fila(cel)
            fam.append(f["mem_familiaridad"]); nov.append(f["mem_novedad"])
            carga.append(f["mem_carga_estructural"]); nece.append(f["necesidad"])
        out[spec] = {"familiaridad_med": round(float(np.mean(fam[250:])), 3),
                     "novedad_med": round(float(np.mean(nov[250:])), 3),
                     "carga_estructural_fin": round(float(carga[-1]), 5),
                     "necesidad_med": round(float(np.mean(nece[250:])), 3),
                     "episodios": int(f["mem_episodios_n"])}
    return out


def main():
    os.makedirs(RES, exist_ok=True)
    e = episodica_recall(); n = necesidad_refractariedad(); v = valor_doble_decay()
    p = persistencia(); c5 = capa5_real()

    print("=" * 90)
    print("(1) EPISÓDICA + RECALL EXPLÍCITO (resuelve el 0/50 de v180)")
    print(f"    episodios grabados ante saliencia: {e['episodios_grabados']}")
    print(f"    recall en MISMA clave: {e['recall_mismo']} (tipo={e['tipo_mismo']}=amenaza?{TIPO['amenaza']}, costo={e['costo_mismo']})")
    print(f"    recall en clave DISTINTA: {e['recall_otro']} (debe ser 0)")
    print("(2) NECESIDAD ACUMULATIVA + REFRACTARIEDAD")
    print(f"    necesidad {n['necesidad_inicial']}→pico {n['necesidad_pico']} (sube con Cb)")
    print(f"    tras re-acople: saciedad={n['saciedad_post_reacople']} → necesidad={n['necesidad_post']} "
          f"vs necesidad_efectiva={n['necesidad_efectiva_post']} (efectiva debe ser < necesidad)")
    print("(3) VALOR DOBLE-DECAY (hábito vs identidad)")
    print(f"    retención tras 60s sin visitar: corto={v['corto_ret']}  largo={v['largo_ret']} (largo > corto)")
    print("(4) PERSISTENCIA (permanencia; τ crece con la vida)")
    print(f"    confianza a 5s de ausencia: joven={p['conf_joven_t5s']}  viejo(+vida)={p['conf_viejo_t5s']} (viejo > joven)")
    print("(5) CAPA 5 — memoria estructural del soma (implícita→consultable), célula real")
    for spec, r in c5.items():
        print(f"    {spec}: familiaridad={r['familiaridad_med']} novedad={r['novedad_med']} "
              f"carga_W={r['carga_estructural_fin']} necesidad={r['necesidad_med']} episodios={r['episodios']}")

    # --------------- VEREDICTO ---------------
    C1 = e["episodios_grabados"] > 0 and e["recall_mismo"] == 1.0 and e["recall_otro"] == 0.0 and e["tipo_mismo"] == TIPO["amenaza"]
    C2 = n["necesidad_pico"] > n["necesidad_inicial"] and n["necesidad_efectiva_post"] < n["necesidad_post"]
    C3 = v["largo_ret"] > v["corto_ret"]
    C4 = p["conf_viejo_t5s"] > p["conf_joven_t5s"]
    C5 = all(0.0 <= c5[s]["familiaridad_med"] <= 1.0 for s in c5) and any(c5[s]["carga_estructural_fin"] > 0 for s in c5)
    ver = {"C1_recall_explicito": C1, "C2_necesidad_refractaria": C2, "C3_doble_decay": C3,
           "C4_persistencia_vida": C4, "C5_capa5_soma": C5}
    print("=" * 90)
    nombres = {"C1_recall_explicito": "recall explícito (graba + recupera misma clave, no otra) — resuelve 0/50",
               "C2_necesidad_refractaria": "necesidad acumulativa (Cb) + refractariedad tras saciarse",
               "C3_doble_decay": "valor: largo persiste más que corto (identidad vs hábito)",
               "C4_persistencia_vida": "persistencia: τ mayor con más vida vivida",
               "C5_capa5_soma": "capa 5: memoria estructural del soma consultable"}
    for k, val in ver.items():
        print(f"  {'PASS' if val else 'FALLA'}  {nombres[k]}")
    print(f"\n  RESUMEN: {sum(ver.values())}/{len(ver)} PASS")
    with open(os.path.join(RES, "bateria_memoria.json"), "w", encoding="utf-8") as fjson:
        json.dump({"episodica": e, "necesidad": n, "valor": v, "persistencia": p, "capa5": c5, "veredicto": ver},
                  fjson, ensure_ascii=False, indent=1, default=float)
    print(f"  → {os.path.join(RES, 'bateria_memoria.json')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validar_cuadrantes_fonador.py — ¿podemos CREAR a propósito palabras JARDIN y SELVA con el ARP 2600?

Validación: sabiendo qué le gusta al organismo (coherente, que lo nutre) y qué no (intenso/áspero, que lo
abruma), sintetizamos vocalizaciones con el fonador y medimos en qué cuadrante caen tras aprenderlas.
  JARDIN_FERTIL = ACTIVA (Δ_struct alto) + le GUSTA (viable: A alto, e_R bajo, W alto)
  SELVA_HOSTIL  = ACTIVA + le DISGUSTA (no viable: dispara aversión/reflejo, W bajo)
Iterable: cada llamada prueba un set de candidatos y reporta dónde cayó cada uno.
"""
from __future__ import annotations
import sys, os
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
[sys.path.insert(0, os.path.join(RAIZ, _d)) for _d in ("genoma", "campo", "organelos", "diada", "web", "audio", "experimentos")
 if os.path.isdir(os.path.join(RAIZ, _d))]
import VST_CelulaMadre_WebLive_A as A
from VST_OrganoFonador import OrganoFonador
import importlib.util
_spec = importlib.util.spec_from_file_location("inv", os.path.join(AQUI, "investigar_diferenciacion_audio.py"))
inv = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(inv)
SR = A.SR; DT = A.DT
FON = OrganoFonador(SR)


def _pico(a, p):
    a = np.asarray(a, float); a = a - a.mean(); pk = float(np.max(np.abs(a))) or 1.0
    return a / pk * p


def _tile(audio, pasos):
    """Repite el sonido para que SUENE durante toda la exposición (realimentar resetea t=0 cada vez, y cada
    exposición corre `pasos`·DT segundos; si el sonido es más corto, la ventana queda en silencio)."""
    audio = np.asarray(audio, float); need = int((pasos + 3) * DT * SR)
    if len(audio) < need and len(audio) > 0:
        audio = np.tile(audio, int(np.ceil(need / len(audio))))[:need]
    return audio


def _medir(audio, exposiciones=6, pasos=35):
    audio = _tile(audio, pasos)
    cel = A.cmf.celula_madre_funcional((audio, audio), binaural=True); soma = cel.organelos.get("soma")
    for _ in range(exposiciones):
        if soma is not None:
            soma.realimentar((audio, audio), True)
        for _ in range(pasos):
            cel.vivir_un_paso(DT)
        A._fila(cel, None)                                # ensambla + corre propiocepción (secreta prop_bienestar al milieu)
    M = cel.milieu                                        # LEER DEL MILIEU (la fila no expone delta_struct)
    d = {k: float(M.leer(k, 0.0)) for k in ("delta_struct", "LF", "e_R", "A_sys_env")}
    return {"reg": inv.clasificar_cierre(d), "W": float(M.leer("prop_bienestar", 0.5)),
            "ds": d["delta_struct"], "LF": d["LF"], "e_R": d["e_R"], "A": d["A_sys_env"]}


def probar(candidatos):
    print(f"{'intento':28s} {'pico':>5s} | cuadrante       Δs     LF    e_R    A     W")
    for nombre, params, pico in candidatos:
        try:
            audio = _pico(np.asarray(FON.vocalizar(**params), float), pico)
            r = _medir(audio)
            print(f"  {nombre:26s} {pico:5.3f} | {r['reg']:14s} {r['ds']:.3f}  {r['LF']:.2f}  {r['e_R']:.1f}  {r['A']:.2f}  {r['W']:.2f}")
        except Exception as e:
            print(f"  {nombre:26s} {pico:5.3f} | ERROR {e}")


if __name__ == "__main__":
    import sys as _s
    EXP = int(_s.argv[1]) if len(_s.argv) > 1 else 9     # más exposiciones = más aprendido (sube W del placentero)
    # JARDIN: MUY armónico/suave, contorno que sube, sin tensión — a BAJA intensidad (LF>0 + W alto)
    jardin = [
        ("J:tono puro grave", dict(duracion=0.8, f_ini=220, f_fin=300, fm_ratio=1.0, fm_index=(0.1, 0.12),
                                   vibrato=(0.1, 0.1), tension=0.0, resonancia=0.6, res_centro=350), p)
        for p in (0.004, 0.006, 0.008, 0.011)
    ]
    # SELVA: disonante/áspero pero a intensidad BAJA-MEDIA (para que LF>0 = activo, no colapse)
    selva = [
        ("S:disonante", dict(duracion=0.7, f_ini=620, f_fin=340, fm_ratio=3.3, fm_index=(0.5, 7.0),
                             vibrato=(20.0, 26.0), tension=0.7, resonancia=0.1, res_centro=2000), p)
        for p in (0.012, 0.020, 0.030, 0.045)
    ]
    def _probar(titulo, cands):
        print(f"\n=== {titulo} ({EXP} exposiciones) ===")
        print(f"{'intento':22s} {'pico':>5s} | cuadrante       Δs     LF    e_R    A     W")
        for nombre, params, pico in cands:
            audio = _pico(np.asarray(FON.vocalizar(**params), float), pico)
            r = _medir(audio, exposiciones=EXP)
            print(f"  {nombre:20s} {pico:5.3f} | {r['reg']:14s} {r['ds']:.3f}  {r['LF']:.2f}  {r['e_R']:.1f}  {r['A']:.2f}  {r['W']:.2f}")
    _probar("JARDIN (coherente+suave, baja intensidad)", jardin)
    _probar("SELVA (disonante, baja-media intensidad)", selva)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_notas_resonancia.py — ¿con qué sonido RESUENA A? (búsqueda de Jardín Fértil, idea de Alexis).

El vocabulario R2D2 dio 0 Jardín Fértil (todo Colapso/Selva). Aquí barremos el espacio acústico
—sinusoides PURAS en un rango de frecuencias + las notas reales (Do/Fa/La)— por el MISMO
clasificador sintiente fresco (clasificar_cierre, sin tocar umbrales), buscando qué frecuencia/nota
deja a A en JARDÍN FÉRTIL (viable + activo = florece). Lo que resuene = palabra 'buena' candidata.
Pista previa: A comió el tono agudo Do_alto (IM +0.275) → los agudos son sospechosos.
"""
from __future__ import annotations
import os, sys, json, time, importlib.util
import numpy as np
AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
sp = importlib.util.spec_from_file_location("rec", os.path.join(AQUI, "reclasificar_vocabulario_sintiente.py"))
rec = importlib.util.module_from_spec(sp); sp.loader.exec_module(rec)
SR = rec.A.SR
EXP, PASOS, PICO = 3, 16, 0.012
AUD = "/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/audio_binaural"

def sine(f, dur=1.2):
    t = np.arange(int(dur * SR)) / SR
    return np.sin(2 * np.pi * f * t).astype(np.float64)

# barrido de sinusoides puras (log-espaciado, rango musical) + notas reales
FREqs = [98, 131, 165, 208, 262, 330, 415, 523, 659, 831, 1047, 1319, 1661, 2093, 2637]
NOTAS = ["Do.wav", "Do_alto.wav", "Fa.wav", "La.wav"]

def main():
    tonos = {}
    for f in FREqs:
        tonos[f"sine_{f}Hz"] = ("tono", sine(f), f)
    for nm in NOTAS:
        p = os.path.join(AUD, nm)
        if os.path.exists(p):
            tonos[nm.replace(".wav", "")] = ("nota", rec._load(p), None)
    print(f"{len(tonos)} sonidos (sinusoides + notas), clasificador fresco (exp={EXP})...", flush=True)
    res = {}; t0 = time.time()
    for k, (tipo, audio, freq) in tonos.items():
        try:
            r = rec._vivir_con(rec._viable(np.asarray(audio, float), PICO), EXP, pasos=PASOS)
            res[k] = {"tipo": tipo, "freq": freq, "regimen": r["regimen"], "W": r["W"], "gusto": r["gusto"]}
        except Exception as e:
            res[k] = {"tipo": tipo, "freq": freq, "regimen": "ERROR", "W": 0.0, "gusto": 0.0, "err": str(e)}
        print(f"  {k:16s} → {res[k]['regimen']:14s} W={res[k]['W']:.3f} gusto={res[k]['gusto']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    out = os.path.join(AQUI, "resultado_notas_resonancia.json")
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    jf = [k for k, r in res.items() if r["regimen"] == "JARDIN_FERTIL"]
    top = sorted(res.items(), key=lambda kv: -kv[1]["W"])
    print(f"\n=== RESONANCIA ({time.time()-t0:.0f}s) ===", flush=True)
    print(f"JARDÍN FÉRTIL: {jf if jf else '(ninguno)'}", flush=True)
    print("Top por bienestar W (con qué resuena más):", flush=True)
    for k, r in top[:8]:
        print(f"  {k:16s} {r['regimen']:14s} W={r['W']:.3f}", flush=True)
    print(f"→ {out}", flush=True)

if __name__ == "__main__":
    main()

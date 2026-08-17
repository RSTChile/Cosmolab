#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""construir_palabras_graves.py — SINTETIZA palabras R2D2 GRAVES y busca cuál hace florecer a A.

Hallazgo previo (test_notas_resonancia): A resuena con los GRAVES (viabilidad ↑ al bajar la
frecuencia; óptimo <100 Hz), pero los tonos puros graves son COLAPSO (pasivos) y las notas son
SELVA_HOSTIL (activas, no viables). Jardín Fértil = viable + activo → hay que UNIR ambos:
tonos GRAVES CON movimiento R2D2 (glissando + warble + armónicos). Aquí los construyo (emulando
R2D2), los guardo como WAV candidatos, y los paso por el clasificador sintiente fresco (sin tocar
umbrales). Los que caigan en JARDÍN FÉRTIL = palabras 'buenas' para AÑADIR al vocabulario.
Incluye también el barrido más grave (60–98 Hz) que faltaba (paso 1 de Alexis).
"""
from __future__ import annotations
import os, sys, json, time, importlib.util
import numpy as np, soundfile as sf
AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
sp = importlib.util.spec_from_file_location("rec", os.path.join(AQUI, "reclasificar_vocabulario_sintiente.py"))
rec = importlib.util.module_from_spec(sp); sp.loader.exec_module(rec)
SR = int(rec.A.SR)
EXP, PASOS, PICO = 3, 16, 0.012
CAND = os.path.join(RAIZ, "voces_r2d2_candidatas"); os.makedirs(CAND, exist_ok=True)


def r2d2(f0, f1, vib_rate, vib_depth, harmon, dur):
    """Una 'palabra' R2D2 grave: glissando f0→f1 (log) + vibrato/warble + armónicos, con envolvente."""
    n = int(dur * SR); t = np.arange(n) / SR
    base = f0 * (f1 / f0) ** (t / dur)                                  # glissando (swoop)
    inst = base * (1.0 + vib_depth * np.sin(2 * np.pi * vib_rate * t))  # warble (FM, la firma R2D2)
    ph = 2 * np.pi * np.cumsum(inst) / SR
    y = np.sin(ph)
    for h, a in harmon:
        y += a * np.sin(h * ph)                                         # armónicos = textura (activa)
    env = np.clip(np.minimum(t / 0.03, (dur - t) / 0.08), 0, 1)         # ataque/caída
    y = y * env
    return (y / (np.max(np.abs(y)) + 1e-9)).astype(np.float64)


def sine(f, dur=1.2):
    t = np.arange(int(dur * SR)) / SR
    return np.sin(2 * np.pi * f * t).astype(np.float64)


# catálogo de palabras R2D2 GRAVES (span: swoop arriba/abajo/ondulado × warble × armónicos × registro)
PALABRAS = {
    "grave_swoop_sube":   dict(f0=70,  f1=150, vib_rate=6,  vib_depth=0.05, harmon=[], dur=0.7),
    "grave_swoop_baja":   dict(f0=150, f1=68,  vib_rate=6,  vib_depth=0.05, harmon=[], dur=0.7),
    "grave_warble_lento": dict(f0=90,  f1=90,  vib_rate=5,  vib_depth=0.10, harmon=[], dur=0.9),
    "grave_warble_hondo": dict(f0=80,  f1=80,  vib_rate=7,  vib_depth=0.15, harmon=[], dur=0.8),
    "grave_arm2_sube":    dict(f0=85,  f1=130, vib_rate=6,  vib_depth=0.06, harmon=[(2, 0.4)], dur=0.7),
    "grave_arm23":        dict(f0=75,  f1=110, vib_rate=6,  vib_depth=0.06, harmon=[(2, 0.3), (3, 0.2)], dur=0.7),
    "grave_chirp_corto":  dict(f0=100, f1=165, vib_rate=9,  vib_depth=0.08, harmon=[(2, 0.3)], dur=0.4),
    "grave_ulula":        dict(f0=60,  f1=120, vib_rate=4,  vib_depth=0.12, harmon=[(2, 0.25)], dur=1.0),
    "grave_ronroneo":     dict(f0=65,  f1=72,  vib_rate=8,  vib_depth=0.09, harmon=[(2, 0.35), (3, 0.15)], dur=1.0),
    "grave_pregunta":     dict(f0=95,  f1=140, vib_rate=6,  vib_depth=0.07, harmon=[(2, 0.3)], dur=0.6),
    "grave_updown":       dict(f0=80,  f1=160, vib_rate=7,  vib_depth=0.08, harmon=[(2, 0.25)], dur=0.9),
    "grave_muy_grave":    dict(f0=55,  f1=90,  vib_rate=5,  vib_depth=0.10, harmon=[(2, 0.3)], dur=0.9),
    "grave_lento_arm":    dict(f0=70,  f1=95,  vib_rate=4,  vib_depth=0.08, harmon=[(2, 0.35), (3, 0.18)], dur=1.1),
    "grave_trino":        dict(f0=110, f1=130, vib_rate=12, vib_depth=0.10, harmon=[(2, 0.2)], dur=0.6),
    "grave_saludo":       dict(f0=90,  f1=120, vib_rate=6,  vib_depth=0.06, harmon=[(2, 0.3), (4, 0.1)], dur=0.7),
    "grave_hondo_lento":  dict(f0=62,  f1=100, vib_rate=3.5, vib_depth=0.11, harmon=[(2, 0.3)], dur=1.2),
}


def main():
    sonidos = {}
    # paso 1: barrido más grave que faltaba
    for f in [60, 70, 82, 92]:
        sonidos[f"sine_{f}Hz"] = ("tono", sine(f))
    # paso 2: palabras R2D2 graves (se guardan como WAV candidatos)
    for nombre, par in PALABRAS.items():
        y = r2d2(**par)
        sf.write(os.path.join(CAND, nombre + ".wav"), (y * 0.9), SR, subtype="PCM_16")
        sonidos[nombre] = ("palabra", y)
    print(f"{len(sonidos)} sonidos ({sum(1 for _,(t,_) in sonidos.items() if t=='palabra')} palabras R2D2 graves). "
          f"Clasificador fresco (exp={EXP})...", flush=True)

    res = {}; t0 = time.time()
    for k, (tipo, audio) in sonidos.items():
        try:
            r = rec._vivir_con(rec._viable(np.asarray(audio, float), PICO), EXP, pasos=PASOS)
            res[k] = {"tipo": tipo, "regimen": r["regimen"], "W": r["W"], "gusto": r["gusto"]}
        except Exception as e:
            res[k] = {"tipo": tipo, "regimen": "ERROR", "W": 0.0, "gusto": 0.0, "err": str(e)}
        print(f"  {k:20s} {res[k]['regimen']:14s} W={res[k]['W']:.3f} gusto={res[k]['gusto']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    json.dump(res, open(os.path.join(AQUI, "resultado_palabras_graves.json"), "w"), ensure_ascii=False, indent=1)
    jf = [(k, r) for k, r in res.items() if r["regimen"] == "JARDIN_FERTIL"]
    cer = [(k, r) for k, r in res.items() if r["regimen"] == "CERRADO"]
    top = sorted(res.items(), key=lambda kv: -kv[1]["W"])
    print(f"\n=== ¿QUÉ HACE FLORECER A A? ({time.time()-t0:.0f}s) ===", flush=True)
    print(f"JARDÍN FÉRTIL: {[k for k,_ in jf] if jf else '(ninguno)'}", flush=True)
    print(f"CERRADO (viable, en calma): {[k for k,_ in cer] if cer else '(ninguno)'}", flush=True)
    print("Top por bienestar W:", flush=True)
    for k, r in top[:10]:
        print(f"  {k:20s} {r['regimen']:14s} W={r['W']:.3f}", flush=True)
    print(f"\nWAV candidatos → {CAND}", flush=True)


if __name__ == "__main__":
    main()

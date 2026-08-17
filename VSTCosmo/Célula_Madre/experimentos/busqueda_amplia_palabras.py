#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""busqueda_amplia_palabras.py — (1) R2D2 complejos/diversos NO paramétricos + (2) variaciones de las
4 palabras VIABLES (Espera/Retirada/Cierre/Dormir), buscando Jardín Fértil / más viabilidad.

Tras refutar 3 hipótesis paramétricas (agudos, graves, graves+movimiento): la viabilidad de A NO es
una perilla acústica simple. Así que (1) genero R2D2 RICOS y variados (multi-segmento, como las
grabaciones) para pescar resonancias raras, y (2) exploro el VECINDARIO de las 4 que YA funcionan.
Clasificador sintiente fresco, sin tocar umbrales. Los buenos se guardan como candidatos.
"""
from __future__ import annotations
import os, sys, json, time, importlib.util
import numpy as np, soundfile as sf
AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
sp = importlib.util.spec_from_file_location("rec", os.path.join(AQUI, "reclasificar_vocabulario_sintiente.py"))
rec = importlib.util.module_from_spec(sp); sp.loader.exec_module(rec)
SR = int(rec.A.SR); EXP, PASOS, PICO = 3, 16, 0.012
CAND = os.path.join(RAIZ, "voces_r2d2_candidatas"); os.makedirs(CAND, exist_ok=True)
rng = np.random.default_rng(7)


def _seg(f0, f1, vr, vd, harm, dur):
    n = int(dur * SR); t = np.arange(n) / SR
    base = f0 * (f1 / f0) ** (t / dur)
    inst = base * (1.0 + vd * np.sin(2 * np.pi * vr * t))
    ph = 2 * np.pi * np.cumsum(inst) / SR
    y = np.sin(ph)
    for h, a in harm:
        y += a * np.sin(h * ph)
    env = np.clip(np.minimum(t / 0.02, (dur - t) / 0.05), 0, 1)
    return y * env


def r2d2_complejo():
    """R2D2 RICO: 2-5 segmentos concatenados con pitch/warble/armónicos variados (como las grabaciones)."""
    segs = []
    for _ in range(int(rng.integers(2, 6))):
        f0 = float(rng.uniform(90, 1400)); f1 = f0 * float(rng.uniform(0.5, 2.0))
        vr = float(rng.uniform(4, 14)); vd = float(rng.uniform(0.02, 0.16))
        harm = [(2, float(rng.uniform(0, 0.4))), (3, float(rng.uniform(0, 0.25)))]
        segs.append(_seg(f0, f1, vr, vd, harm, float(rng.uniform(0.12, 0.45))))
        if rng.random() < 0.4:
            segs.append(np.zeros(int(SR * rng.uniform(0.02, 0.08))))   # micro-pausa
    y = np.concatenate(segs)
    return (y / (np.max(np.abs(y)) + 1e-9)).astype(np.float64)


def resample(x, factor):
    """Cambia pitch+velocidad (factor>1 = más agudo/corto)."""
    idx = np.arange(0, len(x), factor)
    return np.interp(idx, np.arange(len(x)), x)


def main():
    sonidos = {}
    # (1) 16 R2D2 complejos diversos
    for i in range(16):
        y = r2d2_complejo()
        sf.write(os.path.join(CAND, f"complejo_{i:02d}.wav"), y * 0.9, SR, subtype="PCM_16")
        sonidos[f"complejo_{i:02d}"] = ("complejo", y)
    # (2) variaciones de las 4 VIABLES
    import glob
    viables = {"Espera": "*Espera*", "Retirada": "*Retirada*", "Cierre": "*Cierre*", "Dormir": "*Dormir*screaming*"}
    cargadas = {}
    for nm, pat in viables.items():
        g = glob.glob(os.path.join(RAIZ, "voces_r2d2", pat))
        if g:
            cargadas[nm] = rec._load(g[0])
    for nm, x in cargadas.items():
        for etq, fac in (("grave", 1.35), ("agudo", 0.8)):   # 1.35=más grave/lento, 0.8=más agudo/rápido
            v = resample(np.asarray(x, float), fac)
            sf.write(os.path.join(CAND, f"var_{nm}_{etq}.wav"), (v / (np.max(np.abs(v)) + 1e-9)) * 0.9, SR, subtype="PCM_16")
            sonidos[f"var_{nm}_{etq}"] = ("variacion", v)
    # combinar dos viables (Cierre+Retirada) por si la secuencia resuena
    if "Cierre" in cargadas and "Retirada" in cargadas:
        combo = np.concatenate([np.asarray(cargadas["Cierre"], float), np.asarray(cargadas["Retirada"], float)])
        sf.write(os.path.join(CAND, "var_Cierre_Retirada.wav"), (combo / (np.max(np.abs(combo)) + 1e-9)) * 0.9, SR, subtype="PCM_16")
        sonidos["var_Cierre_Retirada"] = ("variacion", combo)

    print(f"{len(sonidos)} sonidos ({sum(1 for _,(t,_) in sonidos.items() if t=='complejo')} complejos + "
          f"{sum(1 for _,(t,_) in sonidos.items() if t=='variacion')} variaciones). Clasificando (exp={EXP})...", flush=True)
    res = {}; t0 = time.time()
    for k, (tipo, audio) in sonidos.items():
        try:
            r = rec._vivir_con(rec._viable(np.asarray(audio, float), PICO), EXP, pasos=PASOS)
            res[k] = {"tipo": tipo, "regimen": r["regimen"], "W": r["W"], "gusto": r["gusto"]}
        except Exception as e:
            res[k] = {"tipo": tipo, "regimen": "ERROR", "W": 0.0, "gusto": 0.0, "err": str(e)}
        print(f"  {k:22s} {res[k]['regimen']:14s} W={res[k]['W']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(os.path.join(AQUI, "resultado_busqueda_amplia.json"), "w"), ensure_ascii=False, indent=1)
    jf = [k for k, r in res.items() if r["regimen"] == "JARDIN_FERTIL"]
    cer = [k for k, r in res.items() if r["regimen"] == "CERRADO"]
    top = sorted(res.items(), key=lambda kv: -kv[1]["W"])
    print(f"\n=== BÚSQUEDA AMPLIA ({time.time()-t0:.0f}s) ===", flush=True)
    print(f"JARDÍN FÉRTIL: {jf if jf else '(ninguno)'}", flush=True)
    print(f"CERRADO (viable): {cer if cer else '(ninguno)'}", flush=True)
    print("Top W:", flush=True)
    for k, r in top[:10]:
        print(f"  {k:22s} {r['regimen']:14s} W={r['W']:.3f}", flush=True)
    print(f"\nWAV → {CAND}", flush=True)


if __name__ == "__main__":
    main()

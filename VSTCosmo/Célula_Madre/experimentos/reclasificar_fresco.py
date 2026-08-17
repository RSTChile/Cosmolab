#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""reclasificar_fresco.py — clasificación LIMPIA: organismo FRESCO por palabra (reset del campo),
aprende el tono en sus exposiciones, y se lee el régimen LF que emerge (clasificar_cierre, sin tocar).
Es el método que SÍ discrimina (el persistente colapsaba todo a Selva Hostil por churn).
Escribe el json en el formato que categorizar_y_renombrar.py consume."""
from __future__ import annotations
import os, sys, json, time, importlib.util
AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
sp = importlib.util.spec_from_file_location("rec", os.path.join(AQUI, "reclasificar_vocabulario_sintiente.py"))
rec = importlib.util.module_from_spec(sp); sp.loader.exec_module(rec)

EXP, PASOS, PICO = 3, 16, 0.012
base = os.path.join(RAIZ, "voces_r2d2")
items = sorted((os.path.splitext(f)[0].split("__")[-1], os.path.join(base, f))
               for f in os.listdir(base) if f.lower().endswith(".wav"))
print(f"{len(items)} palabras, FRESCO por palabra (exp={EXP}, pasos={PASOS})...", flush=True)
res = {}; t0 = time.time()
from collections import Counter
for k, (vid, p) in enumerate(items):
    try:
        r = rec._vivir_con(rec._viable(rec._load(p), PICO), EXP, pasos=PASOS)
        reg = r["regimen"]
        res[vid] = {"final": reg, "secuencia": [reg], "cambios": 0, "estable": True,
                    "detalle": [{"regimen": reg, "W": r["W"], "gusto": r["gusto"]}]}
    except Exception as e:
        res[vid] = {"final": "ERROR", "secuencia": ["ERROR"], "cambios": 0, "estable": False,
                    "detalle": [{"regimen": "ERROR", "W": 0.0, "gusto": 0.0, "err": str(e)}]}
    if (k + 1) % 10 == 0:
        d = Counter(v["final"] for v in res.values())
        print(f"  {k+1}/{len(items)} ({time.time()-t0:.0f}s) · {dict(d)}", flush=True)
out = os.path.join(AQUI, "resultado_reclasificar_aprendiendo.json")
json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
print(f"\n=== FINAL ({time.time()-t0:.0f}s) === {dict(Counter(v['final'] for v in res.values()))}\n→ {out}", flush=True)

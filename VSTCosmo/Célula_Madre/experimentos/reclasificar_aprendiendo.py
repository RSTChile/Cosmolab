#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""reclasificar_aprendiendo.py — pasa las 70 palabras por UN organismo que PERSISTE y APRENDE, N pasadas.

No impongo métricas: reuso EXACTAMENTE la medición del organismo sintiente (clasificar_cierre =
régimen LF canónico + prop_bienestar W + gusto soma._valor), sin tocar umbrales. La única diferencia
con reclasificar_vocabulario_sintiente.py es que aquí el organismo NO es fresco por palabra: es UNO
solo que oye las 70, pasada tras pasada, y su memoria/gusto ACUMULAN. Así vemos si la categoría de
cada palabra CAMBIA porque el organismo APRENDE, y si converge (legitimidad).

Uso:  venv/bin/python3 experimentos/reclasificar_aprendiendo.py [--pasadas 10] [--exp 5] [--pico 0.012] [--limite 0]
"""
from __future__ import annotations
import os, sys, json, argparse, time
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
import importlib.util
_spec = importlib.util.spec_from_file_location("rec", os.path.join(AQUI, "reclasificar_vocabulario_sintiente.py"))
rec = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(rec)   # reusa _load/_viable/_tile/A/inv SIN tocar

A, inv, SR, DT = rec.A, rec.inv, rec.SR, rec.DT


def medir(cel, soma, audio, exposiciones, pasos):
    """MISMA medición que el script sintiente, pero sobre un organismo YA EXISTENTE (persistente)."""
    audio = rec._tile(audio, pasos)
    for _ in range(exposiciones):
        if soma is not None and hasattr(soma, "realimentar"):
            soma.realimentar((audio, audio), True)
        for _ in range(pasos):
            cel.vivir_un_paso(DT)
        A._fila(cel, None)
    M = cel.milieu
    d = {k: float(M.leer(k, 0.0)) for k in ("delta_struct", "LF", "e_R", "A_sys_env")}
    return {"regimen": inv.clasificar_cierre(d), "W": float(M.leer("prop_bienestar", 0.5)),
            "gusto": float(getattr(soma, "_valor", 0.0)) if soma is not None else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasadas", type=int, default=10)
    ap.add_argument("--exp", type=int, default=5)
    ap.add_argument("--pico", type=float, default=0.012)
    ap.add_argument("--pasos", type=int, default=28)
    ap.add_argument("--limite", type=int, default=0, help="0=todas; N=solo N palabras (prueba de tiempo)")
    args = ap.parse_args()

    base = os.path.join(RAIZ, "voces_r2d2")
    ids = {}
    for fn in sorted(os.listdir(base)):
        if fn.lower().endswith(".wav"):
            ids[os.path.splitext(fn)[0].split("__")[-1]] = os.path.join(base, fn)
    items = sorted(ids.items())
    if args.limite > 0:
        items = items[:args.limite]
    audios = {vid: rec._viable(rec._load(p), args.pico) for vid, p in items}

    # UN organismo persistente (nace en silencio; aprende oyendo las palabras pasada tras pasada)
    sil = np.zeros(int(4 * DT * SR))
    cel = A.cmf.celula_madre_funcional((sil, sil), binaural=True)
    soma = cel.organelos.get("soma")

    print(f"{len(items)} palabras × {args.pasadas} pasadas, UN organismo que aprende (exp {args.exp})...", flush=True)
    matriz = {vid: [] for vid, _ in items}     # vid -> [regimen por pasada]
    detalle = {vid: [] for vid, _ in items}    # vid -> [(regimen,W,gusto) por pasada]
    t0 = time.time()
    for pasada in range(args.pasadas):
        for vid, _ in items:
            r = medir(cel, soma, audios[vid], args.exp, args.pasos)
            matriz[vid].append(r["regimen"]); detalle[vid].append(r)
        dt = time.time() - t0
        from collections import Counter
        dist = Counter(matriz[vid][-1] for vid, _ in items)
        print(f"  pasada {pasada+1}/{args.pasadas} ({dt:.0f}s) · dist={dict(dist)}", flush=True)

    # convergencia: ¿cambió la categoría al aprender? ¿se estabilizó al final?
    def conv(seq):
        cambios = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i-1])
        ult = seq[-3:] if len(seq) >= 3 else seq
        estable = len(set(ult)) == 1
        final = max(set(ult), key=ult.count)
        return cambios, estable, final

    out = os.path.join(RAIZ, "experimentos", "resultado_reclasificar_aprendiendo.json")
    res = {vid: {"secuencia": matriz[vid], "cambios": conv(matriz[vid])[0],
                 "estable": conv(matriz[vid])[1], "final": conv(matriz[vid])[2],
                 "detalle": detalle[vid]} for vid, _ in items}
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    from collections import Counter
    distf = Counter(res[vid]["final"] for vid, _ in items)
    n_cambiaron = sum(1 for vid, _ in items if res[vid]["cambios"] > 0)
    n_estables = sum(1 for vid, _ in items if res[vid]["estable"])
    print(f"\n=== FINAL ({time.time()-t0:.0f}s) ===")
    print("Distribución final:", dict(distf))
    print(f"Palabras que CAMBIARON de categoría al aprender: {n_cambiaron}/{len(items)}")
    print(f"Palabras ESTABLES al final (últimas 3 pasadas iguales): {n_estables}/{len(items)}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()

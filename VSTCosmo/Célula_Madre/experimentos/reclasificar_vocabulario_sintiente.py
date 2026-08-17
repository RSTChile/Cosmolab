#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reclasificar_vocabulario_sintiente.py — el organismo CLASIFICA su repertorio por cómo lo hace SENTIR.

Cierre del arco (29-jun-2026). Ya no clasificamos por el espectro (Shannon, prohibido) ni por reglas
impuestas. Cada palabra se mete al organismo a INTENSIDAD VIABLE y se la deja OÍR REPETIDAS veces
(aprenderla); luego se mide el estado ESTABLE en que lo deja:
  - RÉGIMEN canónico de cierre (Jardín Fértil / Cerrado / Colapso / Selva Hostil),
  - BIENESTAR propioceptivo W (la suma de sus estados: ¿se siente bien o mal al oírla?),
  - GUSTO (valor aprendido: ¿le gusta?).
La cadena: tímpano (Von Békésy) → memoria (familiaridad) → propiocepción (W) → valor → cierre → cuadrante.

Mapa de cuadrantes (emergente, no impuesto):
  JARDIN_FERTIL = le gusta Y lo activa (rico)   ·   CERRADO = le gusta pero en reposo
  SELVA_HOSTIL  = NO le gusta Y lo activa (tenso) ·  COLAPSO = NO le gusta y lo agota

Uso:  venv/bin/python3 experimentos/reclasificar_vocabulario_sintiente.py [--exposiciones 6] [--pico 0.012]
Escribe ~/Downloads/RECLASIFICACION_SINTIENTE_<ts>/ con resumen.md + csv + json.
"""
from __future__ import annotations
import os, sys, json, argparse, time, wave as wavemod
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
[sys.path.insert(0, os.path.join(RAIZ, _d)) for _d in ("genoma", "campo", "organelos", "diada", "web", "audio", "experimentos")
 if os.path.isdir(os.path.join(RAIZ, _d))]
import VST_CelulaMadre_WebLive_A as A
import importlib.util
_spec = importlib.util.spec_from_file_location("inv", os.path.join(AQUI, "investigar_diferenciacion_audio.py"))
inv = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(inv)
SR = A.SR; DT = A.DT


def _load(path):
    w = wavemod.open(path, "rb")
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float) / 32768.0
    if w.getnchannels() == 2:
        a = a.reshape(-1, 2).mean(axis=1)
    return a

def _viable(a, pico):
    a = a - a.mean(); pk = float(np.max(np.abs(a))) or 1.0
    return a / pk * pico


def _tile(audio, pasos):
    """Repite el sonido para que SUENE durante toda la exposición (si no, la ventana queda en silencio y
    medimos silencio — el bug que la validación destapó)."""
    audio = np.asarray(audio, float); need = int((pasos + 3) * DT * SR)
    if 0 < len(audio) < need:
        audio = np.tile(audio, int(np.ceil(need / len(audio))))[:need]
    return audio


def _vivir_con(audio, exposiciones, pasos=32):
    """Un organismo FRESCO oye el sonido `exposiciones` veces (lo aprende). Devuelve el estado ESTABLE.
    Lee del MILIEU (la fila no expone delta_struct) y tilea el audio (que suene de verdad)."""
    audio = _tile(audio, pasos)
    cel = A.cmf.celula_madre_funcional((audio, audio), binaural=True)
    soma = cel.organelos.get("soma")
    for _ in range(exposiciones):
        if soma is not None and hasattr(soma, "realimentar"):
            soma.realimentar((audio, audio), True)
        for _ in range(pasos):
            cel.vivir_un_paso(DT)
        A._fila(cel, None)                                # ensambla + corre propiocepción (secreta prop_bienestar)
    M = cel.milieu
    d = {k: float(M.leer(k, 0.0)) for k in ("delta_struct", "LF", "e_R", "A_sys_env")}
    return {"regimen": inv.clasificar_cierre(d), "W": float(M.leer("prop_bienestar", 0.5)),
            "gusto": float(getattr(soma, "_valor", 0.0)) if soma is not None else 0.0,
            "e_R": d["e_R"], "A": d["A_sys_env"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposiciones", type=int, default=6)
    ap.add_argument("--pico", type=float, default=0.012)
    args = ap.parse_args()

    base = os.path.join(RAIZ, "voces_r2d2")
    ids = {}
    for fn in sorted(os.listdir(base)):
        if fn.lower().endswith(".wav"):
            ids[os.path.splitext(fn)[0].split("__")[-1]] = os.path.join(base, fn)
    print(f"Reclasificando {len(ids)} palabras (organismo sintiente, {args.exposiciones} exposiciones, pico {args.pico})...")

    res = {}
    for k, (vid, p) in enumerate(sorted(ids.items())):
        try:
            r = _vivir_con(_viable(_load(p), args.pico), args.exposiciones)
        except Exception as e:
            r = {"regimen": "ERROR", "W": 0.0, "gusto": 0.0, "e_R": 0.0, "A": 0.0, "err": str(e)}
        res[vid] = r
        if k % 12 == 0:
            print(f"  …{k}/{len(ids)}")

    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(os.stat(__file__).st_mtime + 1))
    out = os.path.join(os.path.expanduser("~/Downloads"), f"RECLASIFICACION_SINTIENTE_{ts}")
    os.makedirs(out, exist_ok=True)
    from collections import Counter
    dist = Counter(r["regimen"] for r in res.values())
    # csv
    with open(os.path.join(out, "reclasificacion.csv"), "w", encoding="utf-8") as f:
        f.write("voz_id,regimen,bienestar_W,gusto,e_R,A_sys_env\n")
        for vid, r in sorted(res.items()):
            f.write(f"{vid},{r['regimen']},{r['W']:.3f},{r['gusto']:.3f},{r['e_R']:.2f},{r['A']:.3f}\n")
    with open(os.path.join(out, "reclasificacion.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    # resumen
    gustadas = sorted([v for v in res if res[v]["W"] >= 0.55], key=lambda v: -res[v]["W"])
    disgustadas = sorted([v for v in res if res[v]["W"] < 0.45], key=lambda v: res[v]["W"])
    with open(os.path.join(out, "resumen.md"), "w", encoding="utf-8") as f:
        f.write("# Reclasificación del vocabulario — por cómo hace SENTIR al organismo\n\n")
        f.write(f"{len(ids)} palabras, organismo SINTIENTE (tímpano + memoria + propiocepción + gusto). "
                f"{args.exposiciones} exposiciones (aprendidas), intensidad viable {args.pico}.\n\n")
        f.write("> No se clasifica por el espectro (Shannon) sino por el RÉGIMEN de cierre en que la palabra "
                "deja al organismo y por su BIENESTAR (W = suma de estados): le gusta o no.\n\n")
        f.write("## Distribución por cuadrante\n\n")
        for r in ("JARDIN_FERTIL", "CERRADO", "COLAPSO", "SELVA_HOSTIL"):
            f.write(f"- **{r}**: {dist.get(r,0)}\n")
        f.write(f"\n## Le GUSTAN (bienestar alto) — {len(gustadas)}\n\n")
        f.write(", ".join(f"{v}(W={res[v]['W']:.2f})" for v in gustadas[:30]) + "\n")
        f.write(f"\n## NO le gustan (bienestar bajo) — {len(disgustadas)}\n\n")
        f.write(", ".join(f"{v}(W={res[v]['W']:.2f})" for v in disgustadas[:30]) + "\n")
        f.write("\n## Tabla completa\n\n| palabra | cuadrante | W (siente) | gusto |\n|---|---|---|---|\n")
        for vid, r in sorted(res.items(), key=lambda kv: -kv[1]["W"]):
            f.write(f"| {vid} | {r['regimen']} | {r['W']:.2f} | {r['gusto']:.2f} |\n")
    print(f"\nDISTRIBUCIÓN: " + " · ".join(f"{r}={dist.get(r,0)}" for r in ("JARDIN_FERTIL","CERRADO","COLAPSO","SELVA_HOSTIL")))
    print(f"Le gustan: {len(gustadas)} · No le gustan: {len(disgustadas)}")
    print(f"→ {out}/resumen.md")


if __name__ == "__main__":
    main()

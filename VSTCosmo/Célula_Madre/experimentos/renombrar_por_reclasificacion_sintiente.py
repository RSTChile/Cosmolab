#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
renombrar_por_reclasificacion_sintiente.py — pone el CUADRANTE REAL (sentido) en el nombre del archivo.

Renombra cada WAV del banco según la reclasificación del organismo SINTIENTE (membrana + memoria +
propiocepción): <REGIMEN>__<nombre_observacional>__<id>.wav  (p.ej. SELVA_HOSTIL__Alerta__alerta.wav).
El id (último '__') se PRESERVA = identidad estable (el loader strippea el prefijo). Reversible.

Uso:  venv/bin/python3 experimentos/renombrar_por_reclasificacion_sintiente.py [--dry-run] [--revertir]
Lee la reclasificación más reciente de ~/Downloads/RECLASIFICACION_SINTIENTE_*/reclasificacion.json
"""
from __future__ import annotations
import os, sys, json, glob, argparse, unicodedata, re

AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, os.path.join(RAIZ, "organelos"))
from VST_CalibradorLexicoExperiencial import REGIMENES, _nombre
VOCES = os.path.join(RAIZ, "voces_r2d2")


def _ascii(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")

def _id(stem):
    return stem.split("__")[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revertir", action="store_true", help="vuelve a <id>.wav (quita prefijo de cuadrante)")
    args = ap.parse_args()

    reclas = {}
    if not args.revertir:
        dirs = sorted(glob.glob(os.path.expanduser("~/Downloads/RECLASIFICACION_SINTIENTE_*")))
        if not dirs:
            print("No hay reclasificación sintiente. Corre reclasificar_vocabulario_sintiente.py primero."); return
        jpath = os.path.join(dirs[-1], "reclasificacion.json")
        reclas = json.load(open(jpath, encoding="utf-8"))
        print(f"Reclasificación: {jpath}")

    cambios, sin = [], []
    for nombre in sorted(os.listdir(VOCES)):
        if not nombre.lower().endswith(".wav"):
            continue
        vid = _id(os.path.splitext(nombre)[0])
        if args.revertir:
            nuevo = f"{vid}.wav"
        else:
            r = reclas.get(vid)
            if not r or r.get("regimen") not in REGIMENES:
                sin.append(vid); continue
            reg = r["regimen"]; nom = _nombre(reg, vid)
            nuevo = f"{reg}__{_ascii(nom)}__{vid}.wav"
        if nuevo != nombre:
            cambios.append((nombre, nuevo))

    print(f"{'REVERTIR' if args.revertir else 'RENOMBRAR'} — {len(cambios)} cambios"
          + (f" · {len(sin)} sin clasificación (se omiten): {sin}" if sin else ""))
    for v, n in cambios:
        print(f"  {v:40s} → {n}")
    if args.dry_run:
        print("\n(dry-run: nada se tocó)"); return
    if len(set(n for _, n in cambios)) != len(cambios):
        print("\n⚠ COLISIÓN de nombres — abortado."); return
    hechos = 0
    for v, n in cambios:
        try:
            os.rename(os.path.join(VOCES, v), os.path.join(VOCES, n)); hechos += 1
        except Exception as e:
            print(f"  ✗ {v}: {e}")
    print(f"\n{hechos}/{len(cambios)} renombrados.")


if __name__ == "__main__":
    main()

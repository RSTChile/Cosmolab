#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
renombrar_repertorio_por_convencion.py — pone el CUADRANTE experiencial en el nombre del archivo.

Lee lexico_comun/convencion_lexica.json y renombra cada WAV del banco a:

    <REGIMEN>__<nombre_observacional>__<id_original>.wav     (p.ej. COLAPSO__Perdida__dolor.wav)

El id_original (último segmento tras '__') se PRESERVA: es la identidad estable que el organismo usa
(VST_OrganoComunicacion._cargar_voces strippea el prefijo). Por eso renombrar NO cambia quién es la
palabra para el organismo — sólo la hace legible para el observador. Idempotente y REVERSIBLE: re-deriva
el id del nombre actual, así una recalibración colectiva re-escribe el prefijo sin perder el id.

Uso:  venv/bin/python3 experimentos/renombrar_repertorio_por_convencion.py [--dry-run] [--revertir]
"""
from __future__ import annotations
import os, sys, json, argparse, unicodedata, re

AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
VOCES = os.path.join(RAIZ, "voces_r2d2")
CONV = os.path.join(os.environ.get("ANIMA_LEXICO_DIR") or os.path.join(RAIZ, "lexico_comun"),
                    "convencion_lexica.json")


def _ascii(s: str) -> str:
    """Sin acentos ni espacios, seguro para nombre de archivo (Pérdida→Perdida)."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def _id_original(stem: str) -> str:
    return stem.split("__")[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="muestra los cambios sin renombrar")
    ap.add_argument("--revertir", action="store_true", help="quita el prefijo de cuadrante (vuelve a <id>.wav)")
    args = ap.parse_args()

    if not os.path.isdir(VOCES):
        print(f"No existe el banco: {VOCES}"); return
    conv = {}
    if not args.revertir:
        if not os.path.isfile(CONV):
            print(f"No hay convención: {CONV}. Corre primero calibrar_repertorio_experiencial.py"); return
        conv = json.load(open(CONV, encoding="utf-8")).get("convencion", {})

    cambios, sin_conv = [], []
    for nombre in sorted(os.listdir(VOCES)):
        if not nombre.lower().endswith(".wav"):
            continue
        stem = os.path.splitext(nombre)[0]; vid = _id_original(stem)
        if args.revertir:
            nuevo = f"{vid}.wav"
        else:
            est = conv.get(vid)
            if not est:
                sin_conv.append(vid); continue
            nuevo = f"{est['regimen_experiencial']}__{_ascii(est['nombre_observacional'])}__{vid}.wav"
        if nuevo != nombre:
            cambios.append((nombre, nuevo))

    print(f"Banco: {VOCES}")
    print(f"{'REVERTIR' if args.revertir else 'RENOMBRAR'} — {len(cambios)} cambios"
          + (f", {len(sin_conv)} sin convención (se omiten): {sin_conv}" if sin_conv else ""))
    for viejo, nuevo in cambios:
        print(f"  {viejo:42s} → {nuevo}")
    if args.dry_run:
        print("\n(dry-run: no se tocó nada)"); return
    # colisiones
    destinos = [n for _, n in cambios]
    if len(set(destinos)) != len(destinos):
        print("\n⚠ COLISIÓN de nombres destino — abortado, nada se renombró."); return
    hechos = 0
    for viejo, nuevo in cambios:
        try:
            os.rename(os.path.join(VOCES, viejo), os.path.join(VOCES, nuevo)); hechos += 1
        except Exception as e:
            print(f"  ✗ {viejo}: {e}")
    print(f"\n{hechos}/{len(cambios)} renombrados.")


if __name__ == "__main__":
    main()

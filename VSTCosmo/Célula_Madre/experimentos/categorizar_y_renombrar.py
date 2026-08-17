#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""categorizar_y_renombrar.py — lee el resultado del aprendizaje, categoriza, compara con la
clasificación ACTUAL (¿coincide?), y renombra CORTO y reversible (git).

No decide nada: usa la categoría FINAL que emergió del organismo (reclasificar_aprendiendo.py).
Nombre corto: <CODE>__<id>.wav con CODE ∈ {JF, SH, CE, CO}. El id (identidad estable) se preserva.
Reversible: git + mapping.csv. --apply para ejecutar; sin flag = dry-run.
"""
from __future__ import annotations
import os, sys, json, argparse

AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
VOCES = os.path.join(RAIZ, "voces_r2d2")
JSON = os.path.join(AQUI, "resultado_reclasificar_aprendiendo.json")
CODE = {"JARDIN_FERTIL": "JF", "SELVA_HOSTIL": "SH", "CERRADO": "CE", "COLAPSO": "CO"}
NOMBRE_LARGO = {v: k for k, v in CODE.items()}


def _id(stem):
    return stem.split("__")[-1]

def _cuad_actual(stem):
    segs = stem.split("__")
    return segs[0] if len(segs) >= 2 and segs[0] in CODE else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    res = json.load(open(JSON, encoding="utf-8"))

    # mapear id → archivo actual
    porid = {}
    for fn in sorted(os.listdir(VOCES)):
        if fn.lower().endswith(".wav"):
            porid[_id(os.path.splitext(fn)[0])] = fn

    filas, cambios_nombre, coincide, discrepa, aprendieron = [], [], 0, 0, 0
    for vid, r in sorted(res.items()):
        fn = porid.get(vid)
        if not fn:
            continue
        final = r["final"]; actual = _cuad_actual(os.path.splitext(fn)[0])
        code = CODE.get(final, "?")
        coinc = (actual == final)
        coincide += coinc; discrepa += (not coinc)
        aprendieron += (r["cambios"] > 0)
        nuevo = f"{code}__{vid}.wav"
        if nuevo != fn:
            cambios_nombre.append((fn, nuevo))
        filas.append((vid, actual, final, r["cambios"], r["estable"], "".join(s[0] for s in r["secuencia"])))

    # informe
    out = os.path.join(AQUI, "INFORME_reclasificacion_aprendida.md")
    from collections import Counter
    distf = Counter(r["final"] for r in res.values())
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Reclasificación del vocabulario POR APRENDIZAJE del organismo\n\n")
        f.write(f"{len(res)} palabras, {len(res[next(iter(res))]['secuencia'])} pasadas con un organismo que "
                "PERSISTE y APRENDE. Sin métricas impuestas: sólo el régimen LF que emerge (clasificar_cierre).\n\n")
        f.write("## Distribución final por cuadrante\n\n")
        for k in ("JARDIN_FERTIL", "CERRADO", "SELVA_HOSTIL", "COLAPSO"):
            f.write(f"- **{k}** ({CODE[k]}): {distf.get(k,0)}\n")
        f.write(f"\n## ¿Coincide con la clasificación actual?\n\n")
        f.write(f"- Coinciden: **{coincide}/{len(filas)}** · Discrepan: **{discrepa}/{len(filas)}**\n")
        f.write(f"- Palabras que CAMBIARON de categoría mientras aprendían: **{aprendieron}/{len(filas)}**\n\n")
        f.write("## Tabla (secuencia = inicial de cada pasada: J/C/S/O)\n\n")
        f.write("| palabra | actual | final | ¿coincide? | cambios | estable | trayectoria |\n|---|---|---|---|---|---|---|\n")
        for vid, actual, final, camb, est, traj in sorted(filas, key=lambda x: x[2]):
            f.write(f"| {vid} | {actual} | {CODE.get(final,'?')} | {'sí' if actual==final else 'NO'} | {camb} | {'sí' if est else 'no'} | {traj} |\n")
    print(f"Distribución final: {dict(distf)}")
    print(f"Coinciden con lo actual: {coincide}/{len(filas)} · aprendieron(cambiaron): {aprendieron}/{len(filas)}")
    print(f"Informe → {out}")

    # rename
    print(f"\n{'APLICANDO' if args.apply else 'DRY-RUN'} — {len(cambios_nombre)} renombres:")
    for v, n in cambios_nombre[:80]:
        print(f"  {v:42s} → {n}")
    if len(set(n for _, n in cambios_nombre)) != len(cambios_nombre):
        print("⚠ COLISIÓN de nombres — abortado (dos palabras al mismo cuadrante+id)."); return
    map_path = os.path.join(AQUI, "mapping_rename.csv")
    with open(map_path, "w") as f:
        f.write("antes,despues\n")
        for v, n in cambios_nombre: f.write(f"{v},{n}\n")
    if args.apply:
        hechos = 0
        for v, n in cambios_nombre:
            try:
                os.rename(os.path.join(VOCES, v), os.path.join(VOCES, n)); hechos += 1
            except Exception as e:
                print(f"  ✗ {v}: {e}")
        print(f"\n{hechos}/{len(cambios_nombre)} renombrados. Mapping → {map_path} (revertir: git checkout voces_r2d2/)")
    else:
        print(f"\n(dry-run) mapping → {map_path}")


if __name__ == "__main__":
    main()

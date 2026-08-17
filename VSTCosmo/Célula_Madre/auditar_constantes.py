#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GUARDIÁN DE CONSTANTES — que el número no vuelva a crecer en silencio
================================================================================
Por qué existe (4-ago-2026): una auditoría encontró 355 constantes numéricas
usadas como umbral o escala en los organelos, y sólo 10 podían comprobarse
contra datos: las otras 345 comparan magnitudes que nunca salen al CSV. Ese es
el problema de fondo — no que haya constantes, sino que son INVISIBLES, y por
eso los defectos aparecieron de uno en uno, por accidente, cuando algo se moría.

Siete se corrigieron ese día, todos la misma falta: una magnitud comparada
contra algo que no es su escala. `estructura` recortada a 1; `e_R` en escala ~20
metida en una media de cantidades en [0,1]; `Λ_Cos` en 0,002 con umbral 0,005;
la valencia incapaz de ser negativa; `es_ref` calibrado para un mundo diez veces
más ruidoso; una multa por no convertir que era el 27% del gasto; y `C_b`, que
cuenta 5 de 5 representables y por tanto vale 5 siempre.

QUÉ HACE. Congela el inventario actual como LÍNEA BASE y falla si aparece una
constante NUEVA. No corrige las que ya están —eso se hace de a una, con datos
antes y después— pero impide que el número crezca mientras tanto.

La norma que aplica (Alexis, 3-ago-2026): «el organismo decide con sus propios
estados internos: comparación, equilibrio, autorregulación. Cada parámetro
puesto a mano o es homeostático, cibernético o autorregulado, o no sirve.»
Las constantes físicas y los parámetros estructurales son legítimos: nadie se
enoja porque vivamos a 1 atmósfera. Lo que no vale es un número que decide QUÉ
LE PASA al organismo comparado contra una escala que nadie midió.

Uso:
    python auditar_constantes.py                 # comprobar contra la línea base
    python auditar_constantes.py --sellar        # aceptar el estado actual
    python auditar_constantes.py --informe       # listar lo que no discrimina
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import pathlib
import re
import statistics as st
import sys

RAIZ = pathlib.Path(__file__).resolve().parent
# La linea base vive junto al codigo si no hay carpeta docs (el arbol de build no la tiene).
_D = RAIZ.parent / "docs"
LINEA_BASE = (_D / "LINEA_BASE_CONSTANTES.json") if _D.is_dir() else (RAIZ / "LINEA_BASE_CONSTANTES.json")
HISTORIA = r"C:\Users\adale\.anima\history"

# Un umbral: VARIABLE <op> 0.NNN — el número decide algo sobre esa variable.
_COMP = re.compile(
    r'([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)?|(?:fila|r|d|self)\.get\(\s*["\'](\w+)["\'])'
    r'\s*(<=|>=|<|>)\s*(\d+\.\d+)')
# Una escala: constante con nombre de escala, tasa o umbral.
_ASIG = re.compile(
    r'^\s*(?:self\.)?([A-Z_]{2,}[A-Z0-9_]*|k_\w+|tau_\w+|ema_\w+|es_ref|umbral_\w+)'
    r'\s*[:=]\s*(?:float\s*=\s*)?(\d+\.\d+)')


def inventariar(raiz: pathlib.Path) -> list[dict]:
    """Todas las constantes numéricas que deciden o escalan, con su ubicación."""
    fuera = []
    for p in sorted(raiz.rglob("*.py")):
        # `analisis/` NO se audita, y la frontera es exacta: ahí va lo que LEE los registros
        # del organismo (replays, perfiles de columnas, descomposiciones) y nunca corre dentro
        # de él. Un umbral en un script que mira un CSV no decide nada sobre ningún organismo;
        # contarlo sólo enseña a ignorar al guardián, que es peor que no tenerlo. La regla que
        # hace legítima la excepción: **nada del organismo puede importar `analisis/`** — si
        # algún día lo hace, este código vuelve a auditarse.
        if ("__pycache__" in str(p) or p.name == pathlib.Path(__file__).name
                or "analisis/" in str(p).replace("\\", "/")):
            continue
        try:
            lineas = p.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        rel = str(p.relative_to(raiz)).replace("\\", "/")
        for i, l in enumerate(lineas):
            if l.strip().startswith("#"):
                continue
            for m in _COMP.finditer(l):
                bruto, dentro, op, k = m.groups()
                var = dentro or bruto.split(".")[-1]
                if k in ("0.0", "1.0"):     # comparar con 0 o 1 no es calibrar
                    continue
                fuera.append({"clave": f"{rel}:{var}{op}{k}", "tipo": "umbral",
                              "archivo": rel, "variable": var, "op": op, "constante": float(k)})
            for m in _ASIG.finditer(l):
                nom, k = m.groups()
                fuera.append({"clave": f"{rel}:{nom}={k}", "tipo": "escala",
                              "archivo": rel, "variable": nom, "op": "=", "constante": float(k)})
    return fuera


def _series() -> dict:
    """Distribución real de cada columna de la fisiología, para saber si un umbral decide algo."""
    filas = []
    for f in sorted(glob.glob(os.path.join(HISTORIA, "*", "fisiologia", "*.csv")),
                    key=os.path.getmtime)[-6:]:
        try:
            filas += list(csv.DictReader(open(f, newline="", encoding="utf-8", errors="replace")))
        except Exception:
            pass
    if not filas:
        return {}
    fuera = {}
    for k in filas[0]:
        v = []
        for x in filas:
            try:
                v.append(float(x[k]))
            except Exception:
                pass
        if len(v) >= 30:
            fuera[k] = v
    return fuera


def degenerados(inv: list[dict], series: dict) -> list[dict]:
    """Umbrales que caen SIEMPRE del mismo lado: no miden, deciden lo mismo cada vez."""
    fuera = []
    for it in inv:
        if it["tipo"] != "umbral" or it["variable"] not in series:
            continue
        v = series[it["variable"]]
        k = it["constante"]
        lado = sum(1 for x in v if (x < k if it["op"] in ("<", "<=") else x > k))
        frac = lado / len(v)
        if max(frac, 1 - frac) >= 0.90:
            fuera.append(dict(it, fraccion=round(frac, 4), mediana=round(st.median(v), 6)))
    return fuera


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sellar", action="store_true", help="aceptar el estado actual como línea base")
    ap.add_argument("--informe", action="store_true", help="listar los umbrales que no discriminan")
    args = ap.parse_args()

    inv = inventariar(RAIZ)
    claves = sorted({i["clave"] for i in inv})

    if args.sellar:
        LINEA_BASE.parent.mkdir(exist_ok=True)
        LINEA_BASE.write_text(json.dumps({"constantes": claves}, ensure_ascii=False, indent=1),
                              encoding="utf-8")
        print(f"  línea base sellada: {len(claves)} constantes")
        return 0

    series = _series()
    if args.informe:
        deg = degenerados(inv, series)
        print(f"  {len(inv)} constantes · {len(series)} variables medibles")
        print(f"  umbrales que NO discriminan: {len(deg)}\n")
        for d in sorted(deg, key=lambda x: -x["fraccion"]):
            print(f"    {d['variable']} {d['op']} {d['constante']}  se cumple el "
                  f"{100*d['fraccion']:.0f}%  (mediana real {d['mediana']})")
            print(f"      {d['archivo']}")
        return 0

    if not LINEA_BASE.exists():
        print("  no hay línea base: ejecuta --sellar una vez")
        return 0
    base = set(json.loads(LINEA_BASE.read_text(encoding="utf-8"))["constantes"])
    nuevas = sorted(set(claves) - base)
    idas = len(base - set(claves))

    if idas:
        print(f"  {idas} constantes eliminadas desde la línea base (bien)")
    if nuevas:
        print(f"\n  {len(nuevas)} CONSTANTE(S) NUEVA(S) sin justificar:\n")
        for n in nuevas:
            print(f"    {n}")
        print("\n  Cada número que decide qué le pasa al organismo debe medirse contra")
        print("  SU PROPIA historia, no contra una escala que nadie verificó.")
        print("  Si es una constante física o estructural, séllala con --sellar y")
        print("  deja escrito de dónde sale.")
        return 1

    print(f"  sin constantes nuevas ({len(claves)} en la línea base)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

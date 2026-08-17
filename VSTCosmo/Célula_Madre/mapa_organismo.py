#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
MAPA DEL ORGANISMO — quién lee qué, quién produce qué, y quién queda a ciegas
================================================================================
Por qué existe (4-ago-2026). Alexis: «creo que tienes que hacer un mapa de los
organismos, sus organelos, datos, integración… porque estás deduciendo a ciegas;
los organismos ya son complejos y eso te hace más difícil entender.»

Tenía razón, y se puede fechar el daño: arreglar el oído mató el metabolismo
(3-ago) porque no vi que `estructura` la consumen cuatro organelos; corregir la
membrana dejó al organismo sordo del oído derecho porque no vi que ese oído era
el Rødecaster del laboratorio; y relativizar el recurso lo volvió inmortal
porque no vi que el campo nunca entrega cero. Tres cascadas, tres veces el mismo
error de método: leer un archivo a la vez.

QUÉ HACE. No dibuja el mapa: lo EXTRAE del código. Los organelos declaran lo que
leen y secretan (`lee=[...]`, `secreta=[...]`) y además lo hacen con llamadas
explícitas (`milieu.leer("x")`, `secretar("y")`, `fila.get("z")`). De ahí sale
un grafo real, no uno recordado. Después lo cruza con las columnas que de verdad
llegan al CSV, y así distingue tres cosas que hoy se confunden:

  · lo que un organelo produce y OTRO consume  → cascada: tocarlo se propaga
  · lo que produce y NADIE consume             → salida muerta
  · lo que consume y NADIE produce             → lee un valor por defecto, en
                                                 silencio, y decide con él

Esa última categoría es la peligrosa: no falla, no avisa, y el organismo toma
decisiones con un cero que nadie puso.

Uso:
    python mapa_organismo.py              # resumen en pantalla
    python mapa_organismo.py --md         # documento completo en docs/
    python mapa_organismo.py --var e_R    # la cascada de UNA magnitud
"""
from __future__ import annotations
import argparse
import collections
import csv
import glob
import json
import os
import pathlib
import re
import sys

RAIZ = pathlib.Path(__file__).resolve().parent
HISTORIA = r"C:\Users\adale\.anima\history"

# Lo que un organelo DECLARA en su contrato.
_DECL = re.compile(r'(lee|secreta)\s*=\s*\[([^\]]*)\]', re.S)
# Lo que hace de verdad.
_LEE = re.compile(r'(?:milieu\.leer|\.leer)\(\s*["\']([\w.]+)["\']')
_GET = re.compile(r'(?:fila|d|r|estado)\.get\(\s*["\']([\w.]+)["\']')
_SEC = re.compile(r'(?:milieu\.secretar|\.secretar|m)\(\s*["\']([\w.]+)["\']')
_COL = re.compile(r'^\s*(COLS_\w+)\s*=\s*\[', re.M)


def _nombres(texto: str) -> list[str]:
    return [s.strip().strip('"\'') for s in texto.split(",") if s.strip().strip('"\'')]


def escanear(raiz: pathlib.Path) -> dict:
    """Grafo {archivo: {lee, secreta, declara_lee, declara_secreta}} extraído del código."""
    fuera = {}
    for p in sorted(raiz.rglob("*.py")):
        if "__pycache__" in str(p) or p.name in (pathlib.Path(__file__).name,
                                                 "auditar_constantes.py", "glosario.py", "escala.py"):
            continue
        try:
            t = p.read_text(encoding="utf-8")
        except Exception:
            continue
        rel = str(p.relative_to(raiz)).replace("\\", "/")
        d = {"lee": set(), "secreta": set(), "declara_lee": set(), "declara_secreta": set()}
        for tipo, cuerpo in _DECL.findall(t):
            d["declara_lee" if tipo == "lee" else "declara_secreta"].update(_nombres(cuerpo))
        d["lee"].update(_LEE.findall(t))
        d["lee"].update(_GET.findall(t))
        d["secreta"].update(_SEC.findall(t))
        if any(d.values()):
            fuera[rel] = d
    return fuera


def columnas_publicadas() -> set:
    """Las que de verdad llegan al CSV: lo que se puede auditar."""
    fs = sorted(glob.glob(os.path.join(HISTORIA, "*", "fisiologia", "*.csv")),
                key=os.path.getmtime)
    for f in reversed(fs):
        try:
            with open(f, newline="", encoding="utf-8", errors="replace") as fh:
                cab = next(csv.reader(fh))
            if cab:
                return set(cab)
        except Exception:
            continue
    return set()


def analizar(grafo: dict, publicadas: set) -> dict:
    productores = collections.defaultdict(set)
    consumidores = collections.defaultdict(set)
    for arch, d in grafo.items():
        for v in d["secreta"] | d["declara_secreta"]:
            productores[v].add(arch)
        for v in d["lee"] | d["declara_lee"]:
            consumidores[v].add(arch)

    todas = set(productores) | set(consumidores)
    huerfanas = sorted(v for v in todas if v in consumidores and v not in productores)
    mudas = sorted(v for v in todas if v in productores and v not in consumidores)
    compartidas = sorted(v for v in todas
                         if len(consumidores.get(v, ())) >= 2 and v in productores)
    ciegas = sorted(v for v in todas if v not in publicadas and consumidores.get(v))
    return {"productores": productores, "consumidores": consumidores, "todas": todas,
            "huerfanas": huerfanas, "mudas": mudas, "compartidas": compartidas, "ciegas": ciegas}


def cascada(v: str, a: dict) -> str:
    p = sorted(a["productores"].get(v, ()))
    c = sorted(a["consumidores"].get(v, ()))
    lin = [f"  {v}"]
    lin.append("    la produce : " + (", ".join(p) if p else "NADIE (se lee un valor por defecto)"))
    lin.append("    la consumen: " + (", ".join(c) if c else "nadie"))
    return "\n".join(lin)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="escribir el documento completo")
    ap.add_argument("--var", help="ver la cascada de una magnitud")
    args = ap.parse_args()

    grafo = escanear(RAIZ)
    pub = columnas_publicadas()
    a = analizar(grafo, pub)

    if args.var:
        print(cascada(args.var, a))
        return 0

    print(f"  {len(grafo)} archivos con tráfico de datos · {len(a['todas'])} magnitudes distintas")
    print(f"  {len(pub)} columnas llegan al CSV\n")
    print(f"  compartidas (tocarlas se propaga): {len(a['compartidas'])}")
    print(f"  HUÉRFANAS (alguien las lee y nadie las produce): {len(a['huerfanas'])}")
    print(f"  mudas (se producen y nadie las usa): {len(a['mudas'])}")
    print(f"  ciegas (se consumen y no salen al CSV): {len(a['ciegas'])}")

    if args.md:
        try:
            sys.path.insert(0, str(RAIZ))
            from glosario import nombre as _nom
        except Exception:
            def _nom(x):
                return x
        d = RAIZ.parent / "docs"
        d.mkdir(exist_ok=True)
        L = ["# Mapa del organismo — quién lee qué y quién produce qué", "",
             "Extraído del código, no dibujado de memoria. Lo genera "
             "`celula_madre/mapa_organismo.py`.", "",
             f"- **{len(grafo)}** archivos con tráfico de datos",
             f"- **{len(a['todas'])}** magnitudes distintas circulando",
             f"- **{len(pub)}** columnas llegan al CSV de fisiología", "",
             "## Huérfanas — alguien las lee y NADIE las produce", "",
             "Se leen con un valor por defecto, en silencio. El organismo decide con un cero "
             "que nadie puso. Es la categoría más peligrosa porque no falla ni avisa.", ""]
        for v in a["huerfanas"]:
            L.append(f"- `{v}` ({_nom(v)}) — la leen: " +
                     ", ".join(sorted(a["consumidores"][v])[:4]))
        L += ["", "## Compartidas — tocarlas se propaga", "",
              "Cada una la consumen dos o más organelos. Aquí es donde una corrección "
              "local se convierte en una cascada.", ""]
        for v in a["compartidas"]:
            L.append(f"- `{v}` ({_nom(v)}) — produce: " +
                     ", ".join(sorted(a["productores"][v])[:2]) + " → consumen: " +
                     ", ".join(sorted(a["consumidores"][v])[:5]))
        L += ["", "## Ciegas — se consumen y no salen al CSV", "",
              "No se pueden auditar: cualquier umbral que las compare es indemostrable.", ""]
        L += [f"- `{v}` ({_nom(v)})" for v in a["ciegas"]]
        L += ["", "## Mudas — se producen y nadie las usa", ""]
        L += [f"- `{v}`" for v in a["mudas"]]
        (d / "MAPA_ORGANISMO.md").write_text("\n".join(L), encoding="utf-8")
        (d / "MAPA_ORGANISMO.json").write_text(json.dumps(
            {k: {kk: sorted(vv) for kk, vv in v.items()} for k, v in grafo.items()},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print("\n  -> docs/MAPA_ORGANISMO.md y .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

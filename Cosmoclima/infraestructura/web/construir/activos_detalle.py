"""
ACTIVOS UNO POR UNO · para poder decir «el paso bajo nivel ZZ se inunda»
=========================================================================

INSTRUCCIÓN (Alexis, 24-ago-2026): «si seleccionas una comuna y una categoría,
los elementos deberían visualizarse individualmente, sólo los afectados, que es
el filtro: esto sirve para mostrar que si llueve X el paso bajo nivel zz va a
inundarse sí o sí.»

`activos.py` deja sólo CONTEOS por comuna × ítem, que sirven para el mapa
agregado pero no permiten nombrar nada. Sin nombre y coordenada no se puede
señalar un activo concreto, que es justamente lo que le sirve a un alcalde.

★ UN ARCHIVO POR COMUNA, Y POR QUÉ NO UNO SOLO
------------------------------------------------
Son 92.481 activos. En un único JSON serían unos 7 MB que el navegador tendría
que bajar y parsear antes de mostrar nada — y para ver una comuna, que son ~270
activos. Partido por comuna, cada archivo pesa unos 20 KB y se baja **sólo cuando
alguien elige esa comuna**.

Es la misma lógica que ya usa el proyecto con las celdas: no bajar la serie
completa cuando la pregunta es sobre un punto.

★ NOMBRES DE CAMPO DE UNA LETRA
---------------------------------
`n` (ítem), `a` (activo), `y` (lat), `x` (lon). Con 92.481 filas, escribir
`"latitud"` en vez de `"y"` cuesta ~600 KB repartidos en los archivos. No es
elegancia mal entendida: es la diferencia entre que una comuna grande cargue en
un parpadeo o en dos segundos.

★★ CADA ACTIVO LLEVA SUS ANTECEDENTES
---------------------------------------
Un punto que sólo dice su nombre no sirve para decidir nada. Si ese paso bajo
nivel ya está catalogado como punto crítico por SENAPRED, o si esa ruta se cortó
en el temporal de julio, **eso es lo que convierte una estimación en un aviso**:
no es que «podría» inundarse, es que ya se inundó y está registrado.

Se cruzan tres fuentes por cercanía (250 m, que es el largo de una manzana
grande — más lejos ya es otro lugar):

    · los 15.799 puntos críticos de SENAPRED (causa, nivel de riesgo, quién
      debe resolverlo)
    · los 1.289 tramos de vía cortados en el temporal de julio 2026, con fecha
      y gravedad
    · los 612 eventos de CIGIDEN con fecha, proceso y la lluvia que los provocó

⚠️ Cercanía NO es identidad. Que un punto crítico esté a 200 m de una escuela no
prueba que la escuela se inunde: prueba que el sector tiene antecedentes. Por eso
en pantalla se muestran como «antecedentes en el sector» y con su distancia, no
como historial del activo.

★ EL 27 % DE LOS ACTIVOS NO TIENE NOMBRE
------------------------------------------
24.896 de 92.481 traen el literal `None` en la fuente —16.668 son torres de
telecomunicaciones—. Se guardan con el nombre vacío y la aplicación muestra el
tipo de elemento en su lugar. Escribir «None» en el mapa era peor que no escribir
nada: parece un error de programa cuando en realidad es un hueco del catastro.

USO
---
    ../../.venv-esa/bin/python construir/activos_detalle.py
"""

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from math import cos, radians
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
RAIZ = AQUI.parent.parent
DATOS = AQUI.parent / "publico" / "datos"
DESTINO = DATOS / "activos"
FUENTE = RAIZ / "datos" / "fuente_climatica_por_activo.csv"

import activos as A  # noqa: E402  — se reusa su punto-en-polígono


# 250 m: el largo de una manzana grande. Más lejos ya es otro lugar y el
# antecedente deja de decir algo sobre ESTE activo.
RADIO_M = 250.0
CELDA_G = 0.005  # ~500 m: la rejilla del índice, para no comparar todo con todo


def _rej(la, lo):
    return (round(la / CELDA_G), round(lo / CELDA_G))


def indice_antecedentes():
    """Rejilla {celda: [antecedente]} con las tres fuentes."""
    rej = defaultdict(list)
    crudo = RAIZ / "datos" / "crudo" / "puntos_criticos"
    if not crudo.exists():
        return rej
    ult = sorted(crudo.glob("*"))[-1]

    f = ult / "pc_2026.json"
    if f.exists():
        for x in json.loads(f.read_text(encoding="utf-8"))["features"]:
            g = (x.get("geometry") or {}).get("coordinates")
            if not g:
                continue
            p = x["properties"]
            rej[_rej(g[1], g[0])].append({
                "t": "pc", "y": g[1], "x": g[0],
                "c": (p.get("Causa_del_") or "")[:60],
                "r": p.get("Nivel_de_R") or "",
                "s": (p.get("Sector") or "")[:50],
                "q": (p.get("Si_la_resp") or "")[:50],
            })

    f = ult / "vias_julio2026.json"
    if f.exists():
        for x in json.loads(f.read_text(encoding="utf-8"))["features"]:
            g = x.get("geometry") or {}
            a = x["attributes"]
            if not g.get("x") or not a.get("fecha"):
                continue
            fecha = datetime.fromtimestamp(a["fecha"] / 1000, timezone.utc).date()
            rej[_rej(g["y"], g["x"])].append({
                "t": "via", "y": g["y"], "x": g["x"],
                "f": fecha.isoformat(), "g": a.get("gravedad") or "",
                "o": a.get("operatividad") or "", "e": (a.get("emergencia") or "")[:70],
            })

    f = ult / "cigiden_crm.json"
    if f.exists():
        for x in json.loads(f.read_text(encoding="utf-8"))["features"]:
            g = (x.get("geometry") or {}).get("coordinates")
            if not g:
                continue
            p = x["properties"]
            rej[_rej(g[1], g[0])].append({
                "t": "ev", "y": g[1], "x": g[0],
                "p": (p.get("Proceso_P") or "")[:40],
                "a": str(p.get("Año") or ""), "m": str(p.get("Mes") or ""),
                "pp": (str(p.get("PP_mm") or "")[:24]),
            })
    return rej


def buscar_antecedentes(la, lo, rej):
    """Los antecedentes a menos de RADIO_M, con su distancia en metros."""
    i, j = _rej(la, lo)
    out = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for c in rej.get((i + di, j + dj), ()):
                dy = (c["y"] - la) * 111_320
                dx = (c["x"] - lo) * 111_320 * cos(radians(la))
                d = (dy * dy + dx * dx) ** 0.5
                if d <= RADIO_M:
                    e = {k: v for k, v in c.items() if k not in ("y", "x")}
                    e["d"] = round(d)
                    out.append(e)
    out.sort(key=lambda e: e["d"])
    return out[:6]


def construir():
    if not FUENTE.exists():
        print(f"  falta {FUENTE.name}")
        return None
    DESTINO.mkdir(parents=True, exist_ok=True)
    for viejo in DESTINO.glob("*.json"):
        viejo.unlink()

    print("  cargando geometría…", flush=True)
    indice = A.cargar_geometria()

    # ★ Resolver 92.481 puntos contra 345 polígonos es caro, así que se agrupa
    #   por celda de 0,01° (~1 km): los activos de un mismo barrio caen en la
    #   misma comuna y se resuelve una vez para todos.
    cache = {}
    por_comuna = defaultdict(list)
    leidos = sin_coord = sin_comuna = 0

    with FUENTE.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                la, lo = float(r["lat"]), float(r["lon"])
            except (TypeError, ValueError):
                sin_coord += 1
                continue
            leidos += 1
            k = (round(la, 2), round(lo, 2))
            if k not in cache:
                cache[k] = A.resolver(lo, la, indice)
            cut = cache[k]
            if not cut:
                sin_comuna += 1
                continue
            nombre = (r.get("activo") or "").strip()
            # ★ La fuente escribe el literal «None», que no es un nombre.
            if nombre.lower() in ("none", "null", "s/i", "-"):
                nombre = ""
            por_comuna[cut].append({
                "n": r.get("item") or "",
                "a": nombre[:70],
                "y": round(la, 5),
                "x": round(lo, 5),
            })
            if leidos % 20000 == 0:
                print(f"    {leidos:,} activos…", flush=True)

    # ── antecedentes: qué le ha pasado antes a este lugar ───────────────────
    print("\n  cruzando antecedentes (SENAPRED, vías cortadas, CIGIDEN)…", flush=True)
    ant = indice_antecedentes()
    con_ant = 0
    for lista in por_comuna.values():
        for a in lista:
            h = buscar_antecedentes(a["y"], a["x"], ant)
            if h:
                a["h"] = h
                con_ant += 1
    print(f"  activos con antecedentes a menos de {RADIO_M:.0f} m: {con_ant:,}")

    total = 0
    for cut, lista in por_comuna.items():
        # ordenados por ítem para que la aplicación agrupe sin volver a ordenar
        lista.sort(key=lambda d: (d["n"], d["a"]))
        (DESTINO / f"{cut}.json").write_text(
            json.dumps(lista, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8")
        total += len(lista)

    pesos = [p.stat().st_size for p in DESTINO.glob("*.json")]
    print(f"\n  activos con coordenada : {leidos:,}")
    print(f"  ubicados en comuna     : {total:,}")
    print(f"  fuera de toda comuna   : {sin_comuna:,}  (insulares y de borde)")
    print(f"  archivos               : {len(pesos)} · "
          f"mediano {sorted(pesos)[len(pesos)//2]/1e3:.0f} KB · "
          f"mayor {max(pesos)/1e3:.0f} KB · total {sum(pesos)/1e6:.1f} MB")

    (DATOS / "activos_indice.json").write_text(json.dumps({
        "por_comuna": {c: len(v) for c, v in por_comuna.items()},
        "total": total,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"  escrito: activos/ y activos_indice.json")
    return {"activos_detalle": total, "archivos": len(pesos)}


if __name__ == "__main__":
    print("=" * 70)
    print("ACTIVOS UNO POR UNO · por comuna, para carga bajo demanda")
    print("=" * 70)
    sys.exit(0 if construir() else 1)

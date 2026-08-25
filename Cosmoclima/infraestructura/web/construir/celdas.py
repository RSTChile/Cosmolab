"""
CELDAS POR COMUNA · el puente entre el territorio y el clima
==============================================================

★ LA PIEZA QUE FALTABA
------------------------
El pronóstico viene por CELDA de 0,1° y la aplicación se navega por COMUNA.
Nadie había construido el puente entre ambas, así que la app no podía contestar
lo único que un alcalde necesita: **cuánta lluvia le viene a MI comuna**.

★ CÓMO SE ASIGNA, Y POR QUÉ NO BASTA EL CENTROIDE
---------------------------------------------------
Se toman **todas las celdas cuyo centro cae dentro de la comuna**. Recién si no
hay ninguna —comunas chicas, más pequeñas que los ~9 km de la malla— se usa la
celda más cercana al centroide.

Usar sólo el centroide sería más simple y estaría mal en los casos que importan:
Antofagasta mide 30.718 km² y San Joaquín 9,7 km². En la primera, un único punto
representa mal un territorio donde la lluvia cambia de un extremo al otro; en la
segunda, el centroide es lo único que hay.

Cuando una comuna tiene varias celdas, la aplicación se queda con **la más
lluviosa del día**: para decidir si hay que preocuparse, lo que manda es el peor
punto del territorio, no el promedio. Un promedio diluye justamente el sector que
se va a cortar.

⚠️ Las celdas sin pronóstico se excluyen. La aplicación distingue «esta comuna no
tiene cobertura» de «a esta comuna no le va a llover», que son cosas muy
distintas y que confundirlas sería grave.

USO
---
    ../../.venv-esa/bin/python construir/celdas.py
"""

import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
DATOS = AQUI.parent / "publico" / "datos"
SALIDA = DATOS / "celdas_por_comuna.json"
MALLA = 0.10

import activos as A  # noqa: E402  — se reusa su punto-en-polígono, ya probado


def centro(clave):
    i, j = (int(x) for x in clave.split("_"))
    return round(i * MALLA, 4), round(j * MALLA, 4)


def clave_de(la, lo):
    """Coordenada → clave de celda. La inversa de `centro()`."""
    return f"{round(la / MALLA)}_{round(lo / MALLA)}"


def centroide(caja):
    """Centro de la caja de la geometría. Basta: sólo se usa como último recurso
    para comunas sin ninguna celda dentro."""
    lo1, la1, lo2, la2 = caja
    return (lo1 + lo2) / 2, (la1 + la2) / 2


def construir():
    DATOS.mkdir(parents=True, exist_ok=True)
    pron = DATOS / "pronostico.json"
    if not pron.exists():
        print("  falta pronostico.json — corre construir/pronostico.py")
        return None
    claves = list(json.loads(pron.read_text(encoding="utf-8"))["celdas"])
    print(f"  celdas con pronóstico : {len(claves)}")

    # ★ `cargar_geometria` devuelve una LISTA de (cut, caja, geometría), no un
    #   diccionario: hay comunas con varias piezas y agruparlas por clave las
    #   perdería.
    indice = A.cargar_geometria()
    cajas = {}
    for cut, caja, _g in indice:
        if cut in cajas:
            x0, y0, x1, y1 = cajas[cut]
            a0, b0, a1, b1 = caja
            cajas[cut] = (min(x0, a0), min(y0, b0), max(x1, a1), max(y1, b1))
        else:
            cajas[cut] = caja
    print(f"  comunas con geometría : {len(cajas)} ({len(indice)} piezas)")

    # celda → (lon, lat) de su centro
    puntos = {k: centro(k) for k in claves}

    dentro = {}
    for k, (la, lo) in puntos.items():
        cut = A.resolver(lo, la, indice)
        if cut:
            dentro.setdefault(cut, []).append(k)

    # ★★ SEGUNDO INTENTO: LA CELDA DONDE ESTÁN SUS ACTIVOS
    #   El criterio del centro-dentro falla en las islas por geometría pura:
    #   ninguna celda de 0,1° (~9 km de lado) puede tener su centro dentro de
    #   Robinson Crusoe, que mide 48 km². Y el respaldo por centroide era peor
    #   todavía: la comuna de Juan Fernández abarca TAMBIÉN las Desventuradas,
    #   800 km al norte, así que el centro de su caja cae en mitad del Pacífico
    #   y no se parece a ningún lugar donde viva alguien. Dónde está la
    #   infraestructura es mejor referencia que el centro geométrico de un
    #   polígono con dos archipiélagos.
    detalle = DATOS / "activos"
    por_activos = 0
    por_cercania = 0
    salida = {}
    for cut, caja in cajas.items():
        if cut in dentro:
            salida[cut] = {"celdas": sorted(dentro[cut]), "modo": "dentro"}
            continue
        f = detalle / f"{cut}.json"
        if f.exists():
            usados = sorted({clave_de(x["y"], x["x"])
                             for x in json.loads(f.read_text(encoding="utf-8"))}
                            & set(claves))
            if usados:
                salida[cut] = {"celdas": usados, "modo": "por_activos"}
                por_activos += 1
                continue
        clo, cla = centroide(caja)
        mejor = min(claves, key=lambda k: (puntos[k][0] - cla) ** 2
                    + (puntos[k][1] - clo) ** 2)
        d = ((puntos[mejor][0] - cla) ** 2 + (puntos[mejor][1] - clo) ** 2) ** 0.5
        # ⚠️ Si la celda más cercana está a más de ~0,5° (≈55 km), no se asigna:
        #    es preferible declarar «sin cobertura» que inventar una referencia.
        if d <= 0.5:
            salida[cut] = {"celdas": [mejor], "modo": "cercana",
                           "grados": round(d, 3)}
            por_cercania += 1
        else:
            salida[cut] = {"celdas": [], "modo": "sin_cobertura"}

    con = sum(1 for v in salida.values() if v["celdas"])
    sin = len(salida) - con
    print(f"\n  comunas con celda dentro   : "
          f"{sum(1 for v in salida.values() if v['modo']=='dentro')}")
    print(f"  comunas por sus activos    : {por_activos}")
    print(f"  comunas por cercanía       : {por_cercania}")
    print(f"  comunas SIN cobertura      : {sin}")
    print(f"  ★ cobertura: {con} de {len(salida)} "
          f"({100*con/max(len(salida),1):.1f} %)")

    SALIDA.write_text(json.dumps({
        "malla_grados": MALLA,
        "criterio": ("todas las celdas cuyo centro cae dentro de la comuna; "
                     "si ninguna, las celdas donde están sus activos; y en último caso la más cercana al centroide hasta 0,5°"),
        "agregacion": "maximo",
        "por_comuna": salida,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"\n  escrito: {SALIDA.name} ({SALIDA.stat().st_size/1e3:.0f} KB)")
    return {"comunas_con_celda": con, "comunas_sin_cobertura": sin}


if __name__ == "__main__":
    print("=" * 70)
    print("CELDAS POR COMUNA · el puente territorio ↔ clima")
    print("=" * 70)
    sys.exit(0 if construir() else 1)

"""
CRUCE: AMENAZA DE INUNDACIÓN DEL MOP × RED VIAL Y PUENTES
==========================================================

QUÉ HACE
--------
Responde la pregunta que el proyecto existe para responder, sin modelar nada:
**qué infraestructura vial está dentro de una zona de amenaza de inundación
declarada por el propio organismo técnico.**

Cruza tres capas que ya existen y que nadie había cruzado, todas del SIT-MOP:

  · 6.066 polígonos de «Amenaza de Inundación»  (EMERGENCIA/Emergencia_Mayo/3)
  · 14.039 tramos de la Red Vial de Chile        (VIALIDAD/Red_Vial_Chile/3)
  ·  6.742 puentes                                (VIALIDAD/Puentes/0)

POR QUÉ ESTO NO ES UN MODELO
-----------------------------
No estimamos peligro ni calibramos coeficientes: la amenaza ya está zonificada
por el MOP y el inventario ya está levantado por el MOP. Lo único que falta es
la intersección. Por eso este cruce no hereda ninguna de las incertidumbres del
instrumento climático — es aritmética sobre dos capas oficiales.

CÓMO SE RESUELVE LA GEOMETRÍA (y su límite declarado)
------------------------------------------------------
El proyecto no tiene GDAL ni geopandas (gdal falló dos veces), así que la
geometría se resuelve a mano:

  · **Punto en polígono** por lanzamiento de rayo con regla par-impar sobre
    TODOS los anillos del polígono a la vez. Eso maneja los huecos sin código
    extra: un punto dentro de un hueco cruza un número par de bordes y queda
    fuera, que es lo correcto.
  · **Tramo (línea) en polígono** por MUESTREO: cada tramo se recorre tomando un
    punto cada PASO_M metros y se pregunta por cada punto. La fracción de puntos
    dentro por el largo del tramo da los kilómetros expuestos.

    ★ LÍMITE DECLARADO: el muestreo es una aproximación. Con paso de 100 m, un
    cruce de amenaza más angosto que 100 m puede pasarse por alto, y el largo
    expuesto tiene una incertidumbre del orden del paso por cada entrada y
    salida de la zona. Para tramos largos el error relativo es despreciable;
    para cruces puntuales de quebrada, no. Se reporta también el conteo de
    tramos TOCADOS (con al menos un punto dentro), que es robusto al paso.

  · Índice espacial de rejilla de 0,05° (~5 km) para no comparar cada punto
    contra los 6.066 polígonos.

SALIDAS
-------
  datos/vial_en_amenaza_inundacion.csv     un registro por tramo expuesto
  datos/puentes_en_amenaza_inundacion.csv  un registro por puente expuesto
"""

import csv, gzip, json, math, sys
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
CRUDO = AQUI / "datos" / "crudo" / "mop"
AMENAZA = CRUDO / "2026-08-19" / "amenaza_inundacion.esrijson"
TRAMOS = CRUDO / "2026-08-17" / "tramos.geojson.gz"
PUENTES = CRUDO / "2026-08-17" / "puentes.geojson"

PASO_M = 100.0          # cada cuántos metros se muestrea un tramo
CELDA = 0.05            # grados; tamaño de la celda del índice espacial


# --- geometría a mano --------------------------------------------------------

def metros_por_grado(lat):
    """Cuánto mide un grado en metros a esa latitud. Chile va de -17 a -56, así
    que un factor fijo introduciría varios por ciento de error en el largo."""
    return 111_320.0, 111_320.0 * math.cos(math.radians(lat))


def dentro(x, y, anillos):
    """Regla par-impar sobre todos los anillos juntos: maneja huecos gratis."""
    adentro = False
    for anillo in anillos:
        n = len(anillo)
        j = n - 1
        for i in range(n):
            xi, yi = anillo[i][0], anillo[i][1]
            xj, yj = anillo[j][0], anillo[j][1]
            if (yi > y) != (yj > y):
                if x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                    adentro = not adentro
            j = i
    return adentro


def caja(anillos):
    xs = [p[0] for a in anillos for p in a]
    ys = [p[1] for a in anillos for p in a]
    return min(xs), min(ys), max(xs), max(ys)


# --- carga -------------------------------------------------------------------

def cargar_amenaza():
    d = json.loads(AMENAZA.read_text(encoding="utf8"))
    poligonos = []
    for f in d["features"]:
        anillos = f.get("geometry", {}).get("rings")
        if not anillos:
            continue
        a = f.get("attributes", {})
        poligonos.append(dict(
            anillos=anillos, caja=caja(anillos),
            region=(a.get("REGION") or "").strip(),
            clase=(a.get("INUNDACION") or "").strip(),
            fuente=(a.get("FUENTE") or "").strip(),
            oid=a.get("OBJECTID")))
    return poligonos


def indexar(poligonos):
    """Rejilla celda → índices de polígonos cuya caja toca la celda."""
    rejilla = defaultdict(list)
    for k, p in enumerate(poligonos):
        x0, y0, x1, y1 = p["caja"]
        for cx in range(int(math.floor(x0 / CELDA)), int(math.floor(x1 / CELDA)) + 1):
            for cy in range(int(math.floor(y0 / CELDA)), int(math.floor(y1 / CELDA)) + 1):
                rejilla[(cx, cy)].append(k)
    return rejilla


def buscar(x, y, poligonos, rejilla):
    """Devuelve el primer polígono que contiene el punto, o None."""
    for k in rejilla.get((int(math.floor(x / CELDA)), int(math.floor(y / CELDA))), ()):
        p = poligonos[k]
        x0, y0, x1, y1 = p["caja"]
        if x0 <= x <= x1 and y0 <= y <= y1 and dentro(x, y, p["anillos"]):
            return p
    return None


def muestrear(linea):
    """Puntos cada PASO_M metros a lo largo de la línea, con su largo total."""
    puntos, largo, resto = [], 0.0, 0.0
    for i in range(len(linea) - 1):
        (x1, y1), (x2, y2) = linea[i][:2], linea[i + 1][:2]
        mlat, mlon = metros_por_grado((y1 + y2) / 2)
        dx, dy = (x2 - x1) * mlon, (y2 - y1) * mlat
        d = math.hypot(dx, dy)
        largo += d
        if d == 0:
            continue
        t = resto
        while t < d:
            f = t / d
            puntos.append((x1 + (x2 - x1) * f, y1 + (y2 - y1) * f))
            t += PASO_M
        resto = t - d
    if not puntos and linea:
        puntos.append((linea[0][0], linea[0][1]))
    return puntos, largo / 1000.0


# --- proceso -----------------------------------------------------------------

def main():
    print("=" * 74)
    print("AMENAZA DE INUNDACIÓN DEL MOP × RED VIAL")
    print("=" * 74)

    poligonos = cargar_amenaza()
    rejilla = indexar(poligonos)
    print(f"\n  polígonos de amenaza      : {len(poligonos):,}")
    print(f"  celdas del índice          : {len(rejilla):,}")
    print(f"  clases de amenaza          : "
          f"{sorted({p['clase'] for p in poligonos if p['clase']})}")

    # --- puentes ---
    pts = json.loads(PUENTES.read_text(encoding="utf8"))["features"]
    exp_puentes = []
    for f in pts:
        g = f.get("geometry") or {}
        if g.get("type") != "Point":
            continue
        x, y = g["coordinates"][:2]
        p = buscar(x, y, poligonos, rejilla)
        if p:
            a = f["properties"]
            exp_puentes.append(dict(
                codigo=a.get("CODIGO_PUENTE") or a.get("codigo_puente") or "",
                nombre=a.get("NOMBRE_PUENTE") or a.get("nombre_puente") or "",
                rol=a.get("ROL") or a.get("rol") or "",
                cauce=a.get("CAUCE_QUEB") or a.get("cauce_queb") or "",
                region_puente=a.get("REGION") or a.get("region") or "",
                provincia=a.get("PROVINCIA") or a.get("provincia") or "",
                lat=y, lon=x,
                amenaza_clase=p["clase"], amenaza_region=p["region"],
                amenaza_fuente=p["fuente"]))
    print(f"\n  puentes evaluados          : {len(pts):,}")
    print(f"  ★ puentes DENTRO de amenaza: {len(exp_puentes):,}  "
          f"({100*len(exp_puentes)/len(pts):.1f} %)")

    # --- tramos ---
    with gzip.open(TRAMOS, "rt", encoding="utf8") as fh:
        tr = json.load(fh)["features"]
    exp_tramos = []
    km_total = km_exp = 0.0
    for n, f in enumerate(tr, 1):
        g = f.get("geometry") or {}
        if g.get("type") == "LineString":
            partes = [g["coordinates"]]
        elif g.get("type") == "MultiLineString":
            partes = g["coordinates"]
        else:
            continue
        a = f["properties"]
        dentro_n = total_n = 0
        largo = 0.0
        clases = defaultdict(int)
        for parte in partes:
            puntos, lg = muestrear(parte)
            largo += lg
            for x, y in puntos:
                total_n += 1
                p = buscar(x, y, poligonos, rejilla)
                if p:
                    dentro_n += 1
                    clases[p["clase"]] += 1
        km_total += largo
        if dentro_n:
            frac = dentro_n / max(total_n, 1)
            kme = largo * frac
            km_exp += kme
            exp_tramos.append(dict(
                rol=a.get("ROL_LABEL") or a.get("ROL") or "",
                nombre=a.get("NOMBRE_CAMINO") or "",
                clasificacion=a.get("CLASIFICACION") or "",
                carpeta=a.get("CARPETA") or "",
                region=a.get("REGION") or "",
                concesionado=a.get("CONCESIONADO") or "",
                km_tramo=round(largo, 3),
                km_en_amenaza=round(kme, 3),
                fraccion=round(frac, 4),
                puntos_dentro=dentro_n, puntos_total=total_n,
                clase_dominante=max(clases, key=clases.get) if clases else ""))
        if n % 2000 == 0:
            print(f"     … {n:,}/{len(tr):,} tramos", flush=True)

    print(f"\n  tramos evaluados           : {len(tr):,}")
    print(f"  ★ tramos que TOCAN amenaza : {len(exp_tramos):,}  "
          f"({100*len(exp_tramos)/len(tr):.1f} %)")
    print(f"  kilómetros de red          : {km_total:,.0f} km")
    print(f"  ★ kilómetros EN amenaza    : {km_exp:,.0f} km  "
          f"({100*km_exp/km_total:.1f} %)")

    # --- salidas ---
    d1 = AQUI / "datos" / "vial_en_amenaza_inundacion.csv"
    with open(d1, "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(exp_tramos[0].keys()))
        w.writeheader(); w.writerows(exp_tramos)
    d2 = AQUI / "datos" / "puentes_en_amenaza_inundacion.csv"
    with open(d2, "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(exp_puentes[0].keys()))
        w.writeheader(); w.writerows(exp_puentes)
    print(f"\n  escrito: {d1.name} ({len(exp_tramos):,}) · {d2.name} ({len(exp_puentes):,})")

    # --- lecturas ---
    print("\n" + "=" * 74); print("DÓNDE ESTÁ EL RIESGO"); print("=" * 74)

    porreg = defaultdict(lambda: [0.0, 0.0, 0])
    for t in exp_tramos:
        r = porreg[t["region"] or "(sin región)"]
        r[0] += t["km_en_amenaza"]; r[2] += 1
    print("\n  kilómetros expuestos por región:")
    for r, v in sorted(porreg.items(), key=lambda x: -x[1][0])[:12]:
        print(f"     {str(r)[:30]:32s} {v[0]:8,.0f} km   ({v[2]:4d} tramos)")

    porcla = defaultdict(float)
    for t in exp_tramos:
        porcla[t["clasificacion"] or "(sin clasificar)"] += t["km_en_amenaza"]
    print("\n  por clasificación del camino:")
    for c, v in sorted(porcla.items(), key=lambda x: -x[1])[:10]:
        print(f"     {str(c)[:34]:36s} {v:8,.0f} km")

    print("\n  los 12 tramos con más kilómetros dentro de la amenaza:")
    for t in sorted(exp_tramos, key=lambda x: -x["km_en_amenaza"])[:12]:
        print(f"     {t['km_en_amenaza']:7.1f} km de {t['km_tramo']:7.1f}  "
              f"({t['fraccion']*100:4.0f}%)  {t['rol'][:9]:10s} "
              f"{t['nombre'][:38]:40s} {str(t['region'])[:14]}")

    cauces = defaultdict(int)
    for p in exp_puentes:
        c = (p["cauce"] or "").strip().upper()
        if c and c not in ("S/I", "SIN NOMBRE", "S/N", "CAUCE SIN IDENTIFICACION",
                           "ESTERO SIN NOMBRE", "DEFINIR REGION"):
            cauces[(c, p["region_puente"])] += 1
    print("\n  cauces con más puentes DENTRO de zona de amenaza:")
    for (c, r), n in sorted(cauces.items(), key=lambda x: -x[1])[:12]:
        print(f"     {n:3d} puentes   {c[:38]:40s} región {r}")


if __name__ == "__main__":
    main()

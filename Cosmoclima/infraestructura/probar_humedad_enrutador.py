#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probar_humedad_enrutador.py — PRUEBA FALSABLE del «enrutador de amenazas».

================================================================================
QUÉ AFIRMA LA HIPÓTESIS (Alexis, 16-ago-2026)
================================================================================
El terreno no modula el peligro: lo ENCAMINA. Con lluvia parecida,

    suelo SECO      → la energía se gasta en mover material  → REMOCIÓN EN MASA
    suelo SATURADO  → el suelo ya no infiltra, todo escurre   → DESBORDE FLUVIAL

Consecuencia técnica y comprobable: la humedad del suelo PREVIA al evento entra
con SIGNO OPUESTO en las dos familias. Debería ser BAJA antes de una remoción y
ALTA antes de una inundación.

================================================================================
CRITERIO — copiado tal cual de ESTUDIO_VECTORES_DE_AMENAZA.md §6, fijado ANTES
de calcular nada. No se toca aunque falle; si falla, se reporta el fallo.
================================================================================
  «la diferencia de humedad previa entre las dos familias debe ser significativa
   contra un nulo de fechas barajadas dentro del mismo punto (p < 0,01), y el
   signo debe ser el predicho — no basta con que difieran.»

Traducido a números, y también fijado antes de mirar ningún resultado:

  ESTADÍSTICO   D = media(z_humedad | remoción) − media(z_humedad | inundación)
  PREDICCIÓN    D < 0   (remoción con humedad MENOR)
  NULO          10.000 barajadas de la FECHA de cada evento dentro de su MISMO
                punto y su MISMO mes calendario (el mes se fija para que la
                estacionalidad compartida por ambas familias no infle el nulo)
  APRUEBA       p bilateral < 0,01  Y  D < 0.   Las dos cosas, o no pasa.

================================================================================
DECISIONES DE MÉTODO, TODAS DECLARADAS ANTES DE CORRER
================================================================================
1. HUMEDAD PREVIA = MÁXIMO de la humedad diaria en los 30 días ANTERIORES al
   evento (ventana [t−30, t−1], el día del evento EXCLUIDO para que la lluvia
   que causa el evento no contamine la medida). El máximo y no el promedio es
   una regla heredada del proyecto Cosmoclima: la serie ESA CCI tiene huecos y
   días enmascarados, y el promedio los arrastra.
2. Se exige un mínimo de 8 días válidos en la ventana; si no, el evento queda
   «sin dato» y NO se rellena ni se aproxima.
3. NORMALIZACIÓN POR PUNTO (z): el norte árido y el sur húmedo viven en rangos
   distintos de humedad absoluta. Comparar crudo mediría geografía, no
   encaminamiento. Cada evento se expresa como z = (V − μ_celda) / σ_celda con
   μ y σ de la propia celda sobre 2015-2024. Así el contraste norte↔sur queda
   descontado y lo que se compara es «¿venía esta celda más seca o más húmeda
   QUE SU PROPIA COSTUMBRE?».
4. FAMILIAS
   · remoción  = ReTeRM SERNAGEOMIN (posición en terreno, lat/lon reales),
                 sólo eventos con detonante meteorológico declarado.
   · inundación = SENAPRED, eventos cuya clase/tipo/sub-evento nombra inundación,
                 desborde, anegamiento, crecida o aumento de caudal, y que NO
                 nombran a la vez remoción (las mixtas se descartan: no
                 pertenecen a ninguna de las dos familias puras).
   · Prueba simétrica de control: remoción de SENAPRED contra inundación de
     SENAPRED — misma fuente, misma forma de ubicar (centroide comunal), mismo
     sesgo de registro. Sirve para saber si un resultado de la prueba principal
     viene del fenómeno o de mezclar dos catálogos distintos.
5. PRIVACIDAD: del Excel de SENAPRED se leen SÓLO las columnas 1..10 (fecha,
   región, provincia, comuna, origen, clase, tipo, sub-eventos). La columna
   «Antecedentes Observaciones» contiene RUT y descripciones de personas
   fallecidas y está PROHIBIDA: este script nunca la abre. Se trabaja con
   conteos agregados comuna+fecha.
6. Cada par (comuna, fecha) cuenta como UN evento. Un temporal que genera 30
   partes en la misma comuna el mismo día es un evento, no treinta.

================================================================================
DATO
================================================================================
Humedad: ESA CCI Soil Moisture COMBINED v09.2, malla 0,25°, diaria, por OPeNDAP
anónimo del CEDA. Se baja de una vez la caja completa de Chile por día (una
petición por día devuelve TODAS las celdas del país), lo que hace barata la
cobertura nacional. La serie termina el 31-dic-2024: los eventos ReTeRM de 2025
y 2026 quedan fuera por falta de dato satelital, y se declara.

Etapas:  python3 probar_humedad_enrutador.py --etapa bajar     (lento, reanuda)
         python3 probar_humedad_enrutador.py --etapa analizar
         python3 probar_humedad_enrutador.py                   (las dos)
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import numpy as np
import requests

# ─────────────────────────────────────────────────────────────────────────────
# Rutas y constantes
# ─────────────────────────────────────────────────────────────────────────────
RAIZ = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima"
INFRA = os.path.join(RAIZ, "infraestructura")
DATOS = os.path.join(INFRA, "datos")

RETERM_EVENTOS = os.path.join(DATOS, "reterm_eventos.csv")
RETERM_PUNTOS = os.path.join(DATOS, "reterm_puntos.csv")
SENAPRED_XLSX = os.path.join(
    DATOS, "crudo/senapred/2026-08-15/Eventos_Emergencia_2015_2024.xlsx")
COMUNAS_GEOJSON = os.path.join(DATOS, "capas/comunas.geojson")
CLIMA_RETERM = os.path.join(DATOS, "clima_diario_reterm_era5.csv")

SALIDA_CSV = os.path.join(DATOS, "humedad_eventos.csv")

# El cubo crudo de humedad es grande (~120 MB) y es material de trabajo, no
# entregable: vive en el scratchpad de la sesión. Cambiar CACHE si se quiere
# conservar en el proyecto.
CACHE = os.environ.get(
    "CACHE_HUMEDAD",
    "/private/tmp/claude-501/-Users-alexis-Desktop-RMD-Cosmolab-Cosmoclima/"
    "546b1aae-d9cd-4050-a668-92de57d23185/scratchpad/esacci_chile")

# ESA CCI: malla global 0,25°, lat[i] = 89.875 − 0.25·i, lon[j] = −179.875 + 0.25·j
BASE_CEDA = ("https://dap.ceda.ac.uk/thredds/dodsC/neodc/esacci/soil_moisture/"
             "data/daily_files/COMBINED/v09.2")
I0, I1 = 429, 585        # lat −17,375 … −56,375  (toda la franja chilena)
J0, J1 = 412, 460        # lon −76,875 … −64,875
NLAT, NLON = I1 - I0 + 1, J1 - J0 + 1

# Ventana temporal: 2015-2024 es donde coexisten SENAPRED (2015-2024) y ESA CCI
# (termina 2024-12-31). Se baja desde 2014-12-01 para tener los 30 días previos
# de los eventos de enero de 2015.
DIA_INI, DIA_FIN = date(2014, 12, 1), date(2024, 12, 31)
ANALISIS_INI, ANALISIS_FIN = date(2015, 1, 1), date(2024, 12, 31)

VENTANA_PREVIA = 30      # días anteriores al evento
MIN_DIAS_VALIDOS = 8     # mínimo de días con dato en la ventana
N_PERMUTACIONES = 10_000
SEMILLA = 20260816       # fecha de la prueba; fijada para que sea reproducible

HILOS = 12
REINTENTOS = 3
RELLENO = -9999.0

# Vocabulario de clasificación de eventos (fijado antes de calcular)
PAL_INUNDACION = ["inundacion", "inundaciones", "desborde", "anegamiento",
                  "crecida", "aumento caudal"]
PAL_REMOCION = ["remocion", "deslizamiento", "derrumbe", "aluvion",
                "flujo de detritos", "caida de rocas", "desprendimiento"]
# Detonantes de ReTeRM que cuentan como meteorológicos
PAL_DETONANTE_METEO = ["precipitac", "lluvia", "luvias", "deshielo", "nieve",
                       "sistema frontal", "baja segregada"]
PAL_DETONANTE_EXCLUIR = ["sismo", "antrop", "antropog", "excavacion", "matriz",
                         "purines", "pozo"]

# Variantes de escritura de comunas entre SENAPRED y la capa oficial de límites.
# NO es inventar dato: son la misma comuna escrita de dos maneras, verificado
# nombre por nombre contra comunas.geojson. Juan Fernández queda fuera a
# propósito: es archipiélago oceánico y la celda satelital sobre él es mar.
ALIAS_COMUNAS = {
    "coyhaique": "coihaique",
    "aysen": "aisen",
    "paihuano": "paiguano",
    "trehuaco": "treguaco",
    "la calera": "calera",
    "san francisco de mostazal": "mostazal",
}


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────
def nx(s):
    """Normaliza texto: sin tildes, minúsculas, espacios colapsados."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " ".join(s.lower().split())


def celda(lat, lon):
    """Índices (fila, columna) de la caja chilena para un punto. None si cae fuera."""
    i = int(round((89.875 - lat) / 0.25))
    j = int(round((lon + 179.875) / 0.25))
    if not (I0 <= i <= I1 and J0 <= j <= J1):
        return None
    return i - I0, j - J0


def dias_del_rango():
    d, out = DIA_INI, []
    while d <= DIA_FIN:
        out.append(d)
        d += timedelta(days=1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 1 · Bajar el cubo diario de humedad para toda la caja de Chile
# ─────────────────────────────────────────────────────────────────────────────
_re_bloque = re.compile(r"sm\.sm\[1\]\[\d+\]\[\d+\]\n(.*?)\n\n", re.S)


def _parsear_dia(texto):
    """Del ASCII de OPeNDAP saca la matriz (NLAT, NLON). NaN donde hay relleno."""
    ini = texto.index("sm.sm[1]")
    ini = texto.index("\n", ini) + 1
    fin = texto.index("sm.time", ini)
    vals = []
    for linea in texto[ini:fin].strip().splitlines():
        if "," not in linea:
            continue
        # cada línea es "[0][fila], v, v, v, ..."
        vals.extend(float(x) for x in linea.split(",")[1:])
    a = np.array(vals, dtype=np.float32)
    if a.size != NLAT * NLON:
        raise ValueError(f"esperaba {NLAT*NLON} valores, llegaron {a.size}")
    a = a.reshape(NLAT, NLON)
    a[a <= RELLENO + 1] = np.nan
    return a


def _url(d):
    f = f"ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED-{d:%Y%m%d}000000-fv09.2.nc"
    consulta = f"sm%5B0%5D%5B{I0}:{I1}%5D%5B{J0}:{J1}%5D"
    return f"{BASE_CEDA}/{d:%Y}/{f}.ascii?{consulta}"


def _pedir(args):
    k, d, ses = args
    for intento in range(REINTENTOS):
        try:
            r = ses.get(_url(d), timeout=90)
            if r.status_code == 404:
                return k, None, "sin_archivo"
            if r.status_code != 200:
                time.sleep(1 + 2 * intento)
                continue
            return k, _parsear_dia(r.text), "ok"
        except Exception:
            time.sleep(1 + 2 * intento)
    return k, None, "falla_red"


def bajar():
    os.makedirs(CACHE, exist_ok=True)
    dias = dias_del_rango()
    n = len(dias)
    ruta_cubo = os.path.join(CACHE, "sm_chile.npy")
    ruta_listo = os.path.join(CACHE, "dias_listos.npy")

    if os.path.exists(ruta_cubo):
        cubo = np.lib.format.open_memmap(ruta_cubo, mode="r+")
        listo = np.load(ruta_listo)
    else:
        cubo = np.lib.format.open_memmap(
            ruta_cubo, mode="w+", dtype=np.float32, shape=(n, NLAT, NLON))
        cubo[:] = np.nan
        listo = np.zeros(n, dtype=bool)
        np.save(ruta_listo, listo)

    faltan = [(k, d) for k, d in enumerate(dias) if not listo[k]]
    if not faltan:
        print(f"Humedad ya bajada: {n:,} días en {ruta_cubo}")
        return
    print(f"ESA CCI COMBINED v09.2 · caja Chile {NLAT}×{NLON} celdas de 0,25°")
    print(f"A pedir {len(faltan):,} días de {n:,} · {HILOS} en paralelo\n")

    ses = requests.Session()
    ses.headers.update({"User-Agent": "cosmoclima-infraestructura/1.0"})
    t0, hechos, fallos = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=HILOS) as ex:
        for k, mat, estado in ex.map(_pedir, ((k, d, ses) for k, d in faltan)):
            if mat is not None:
                cubo[k] = mat
                listo[k] = True
            else:
                fallos += 1
            hechos += 1
            if hechos % 250 == 0:
                cubo.flush()
                np.save(ruta_listo, listo)
                seg = time.time() - t0
                falta = (len(faltan) - hechos) / (hechos / seg) / 60
                print(f"  {hechos:,}/{len(faltan):,} "
                      f"({100*hechos/len(faltan):.0f}%) · faltan ~{falta:.0f} min "
                      f"· fallos {fallos}")
    cubo.flush()
    np.save(ruta_listo, listo)
    print(f"\nListo en {(time.time()-t0)/60:.1f} min · días sin dato: {fallos}")


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 2 · Construir las familias de eventos
# ─────────────────────────────────────────────────────────────────────────────
def centroides_comunales():
    """
    Centroide ponderado por área de cada comuna, desde comunas.geojson.

    LIMITACIÓN DECLARADA: SENAPRED registra por comuna, sin coordenada. El
    centroide de una comuna cordillerana enorme puede caer lejos del río que se
    desbordó. Es la mejor ubicación disponible con el dato público y se declara
    como tal; no se inventa una coordenada más fina.
    """
    with open(COMUNAS_GEOJSON, encoding="utf-8") as f:
        g = json.load(f)
    out = {}
    for feat in g["features"]:
        nombre = feat["properties"]["COMUNA"]
        geom = feat["geometry"]
        polis = (geom["coordinates"] if geom["type"] == "MultiPolygon"
                 else [geom["coordinates"]])
        sx = sy = sa = 0.0
        for poli in polis:
            anillo = np.asarray(poli[0], dtype=float)
            x, y = anillo[:, 0], anillo[:, 1]
            # fórmula del polígono (shoelace); área en grados² sirve de peso
            cruz = x[:-1] * y[1:] - x[1:] * y[:-1]
            a = cruz.sum() / 2.0
            if abs(a) < 1e-12:
                continue
            cx = ((x[:-1] + x[1:]) * cruz).sum() / (6 * a)
            cy = ((y[:-1] + y[1:]) * cruz).sum() / (6 * a)
            sx += cx * abs(a)
            sy += cy * abs(a)
            sa += abs(a)
        if sa > 0:
            out[nx(nombre)] = (sy / sa, sx / sa, feat["properties"]["REGION"],
                               feat["properties"].get("SUPERFICIE"))
    return out


def eventos_reterm():
    """Remoción en masa de SERNAGEOMIN, con lat/lon medidos en terreno."""
    out = []
    with open(RETERM_EVENTOS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            det = nx(r["detonante"])
            meteo = (any(p in det for p in PAL_DETONANTE_METEO)
                     and not any(p in det for p in PAL_DETONANTE_EXCLUIR))
            try:
                fecha = datetime.strptime(r["fecha"], "%Y-%m-%d").date()
            except Exception:
                continue
            out.append({
                "fuente": "ReTeRM/SERNAGEOMIN",
                "familia": "remocion",
                "fecha": fecha,
                "comuna": r["comuna"],
                "region": r["region"],
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "ubicacion": "punto_terreno",
                "detonante_meteorologico": meteo,
                "detalle": r["tipo"],
            })
    return out


def eventos_senapred(centroides):
    """
    Inundación y remoción de SENAPRED, agregados a un evento por comuna+fecha.

    Sólo se leen las columnas 1..10. La columna «Antecedentes Observaciones»
    (la 44) contiene datos personales y NO se toca.
    """
    import openpyxl
    wb = openpyxl.load_workbook(SENAPRED_XLSX, read_only=True)
    ws = wb["Eventos_de_Emergencia_2015_2024"]
    vistos = {}
    mixtas = 0
    for row in ws.iter_rows(min_row=2, min_col=1, max_col=10, values_only=True):
        fecha, _, region, _, comuna = row[0], row[1], row[2], row[3], row[4]
        if fecha is None or comuna is None:
            continue
        texto = " | ".join(nx(x) for x in (row[5], row[6], row[7], row[8], row[9]))
        hay_i = any(p in texto for p in PAL_INUNDACION)
        hay_r = any(p in texto for p in PAL_REMOCION)
        if hay_i and hay_r:
            mixtas += 1
            continue
        if not (hay_i or hay_r):
            continue
        familia = "inundacion" if hay_i else "remocion"
        f = fecha.date() if hasattr(fecha, "date") else fecha
        clave = (familia, nx(comuna), f)
        if clave in vistos:
            vistos[clave]["n_partes"] += 1
            continue
        clave_c = nx(comuna)
        clave_c = ALIAS_COMUNAS.get(clave_c, clave_c)
        c = centroides.get(clave_c)
        vistos[clave] = {
            "fuente": "SENAPRED",
            "familia": familia,
            "fecha": f,
            "comuna": comuna,
            "region": region,
            "lat": c[0] if c else None,
            "lon": c[1] if c else None,
            "ubicacion": "centroide_comunal" if c else "sin_dato",
            "detonante_meteorologico": True,   # SENAPRED no declara detonante
            "detalle": str(row[6] or "") + " / " + str(row[7] or ""),
            "n_partes": 1,
        }
    print(f"  SENAPRED: filas mixtas (inundación Y remoción) descartadas: {mixtas}")
    return list(vistos.values())


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 3 · Humedad previa por evento y prueba de permutación
# ─────────────────────────────────────────────────────────────────────────────
def cargar_cubo():
    ruta_cubo = os.path.join(CACHE, "sm_chile.npy")
    if not os.path.exists(ruta_cubo):
        sys.exit("No hay cubo de humedad: correr primero --etapa bajar")
    cubo = np.load(ruta_cubo, mmap_mode="r")
    listo = np.load(os.path.join(CACHE, "dias_listos.npy"))
    return cubo, listo


def maximo_previo(cubo):
    """
    V[t] = máximo de humedad en [t−30, t−1] por celda, y n.º de días válidos.

    Se recorre desplazamiento por desplazamiento con np.fmax/np.nansum, que
    ignora los NaN sin llamarlos cero (rellenar con cero sería inventar suelo
    seco donde el satélite simplemente no midió).
    """
    cubo = np.asarray(cubo)          # ~113 MB, se lee una sola vez del disco
    n, nl, nc = cubo.shape
    hay = ~np.isnan(cubo)
    V = np.full((n, nl, nc), np.nan, dtype=np.float32)
    C = np.zeros((n, nl, nc), dtype=np.int16)
    for k in range(1, VENTANA_PREVIA + 1):
        V[k:] = np.fmax(V[k:], cubo[:-k])
        C[k:] += hay[:-k]
    V[C < MIN_DIAS_VALIDOS] = np.nan
    return V, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa", default="todo",
                    choices=["bajar", "analizar", "todo"])
    args = ap.parse_args()

    if args.etapa in ("bajar", "todo"):
        bajar()
    if args.etapa == "bajar":
        return

    print("\n" + "=" * 78)
    print("PRUEBA FALSABLE DEL ENRUTADOR DE AMENAZAS")
    print("=" * 78)

    # ── eventos ──────────────────────────────────────────────────────────────
    print("\n[1] Construyendo las familias de eventos")
    centroides = centroides_comunales()
    print(f"  comunas con centroide: {len(centroides)}")
    ev = eventos_reterm() + eventos_senapred(centroides)
    print(f"  eventos brutos: {len(ev)}")

    # ── humedad ──────────────────────────────────────────────────────────────
    print("\n[2] Cargando humedad ESA CCI y calculando el máximo de 30 días previos")
    cubo, listo = cargar_cubo()
    dias = dias_del_rango()
    idx_dia = {d: k for k, d in enumerate(dias)}
    print(f"  cubo {cubo.shape} · días bajados {listo.sum():,}/{len(dias):,}")
    V, C = maximo_previo(cubo)

    # máscara de fechas utilizables para el análisis y para el nulo
    k_ini, k_fin = idx_dia[ANALISIS_INI], idx_dia[ANALISIS_FIN]
    mes_de = np.array([d.month for d in dias])

    # media y desviación por celda sobre 2015-2024 (la «costumbre» del lugar)
    Vw = V[k_ini:k_fin + 1]
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(Vw, axis=0)
        sd = np.nanstd(Vw, axis=0)
    sd[sd < 1e-6] = np.nan
    n_validos_celda = np.sum(~np.isnan(Vw), axis=0)
    # una celda sirve de referencia sólo si tiene al menos 2 años de días válidos
    ref_ok = n_validos_celda >= 730

    # ── humedad por evento ───────────────────────────────────────────────────
    print("\n[3] Humedad previa por evento")
    for e in ev:
        e["motivo_sin_dato"] = ""
        e["sm_max30_m3m3"] = None
        e["z_sm"] = None
        e["dias_validos"] = None
        e["celda_lat"] = e["celda_lon"] = None
        if e["lat"] is None:
            e["motivo_sin_dato"] = "comuna sin centroide"
            continue
        if not (ANALISIS_INI <= e["fecha"] <= ANALISIS_FIN):
            e["motivo_sin_dato"] = "fecha fuera de 2015-2024 (ESA CCI acaba 31-dic-2024)"
            continue
        cl = celda(e["lat"], e["lon"])
        if cl is None:
            e["motivo_sin_dato"] = "punto fuera de la caja descargada"
            continue
        i, j = cl
        e["celda_lat"] = round(89.875 - 0.25 * (i + I0), 3)
        e["celda_lon"] = round(-179.875 + 0.25 * (j + J0), 3)
        e["ij"] = (i, j)
        k = idx_dia[e["fecha"]]
        e["k"] = k
        v, c = float(V[k, i, j]), int(C[k, i, j])
        e["dias_validos"] = c
        if np.isnan(v):
            e["motivo_sin_dato"] = (
                f"menos de {MIN_DIAS_VALIDOS} días con dato satelital en la ventana")
            continue
        if not ref_ok[i, j] or np.isnan(mu[i, j]) or np.isnan(sd[i, j]):
            e["motivo_sin_dato"] = "celda sin climatología suficiente (<2 años de días válidos)"
            continue
        e["sm_max30_m3m3"] = v
        e["z_sm"] = float((v - mu[i, j]) / sd[i, j])

    # ── selección de las dos familias de la prueba principal ─────────────────
    # Deduplicación por (fuente, familia, celda satelital, fecha): dos partes
    # registradas el mismo día en la misma celda de 25 km comparten LITERALMENTE
    # el mismo valor de humedad. Contarlas dos veces no agrega información, sólo
    # infla la n y estrecha el nulo. Regla declarada antes de calcular.
    def usable(e):
        return e["usado_en_prueba"]

    vistos_celda = set()
    for e in ev:
        e["usado_en_prueba"] = False
        if e["z_sm"] is None:
            continue
        clave = (e["fuente"], e["familia"], e["ij"], e["fecha"])
        if clave in vistos_celda:
            e["motivo_sin_dato"] = "duplicado: misma celda y misma fecha que otro evento"
        else:
            vistos_celda.add(clave)
            e["usado_en_prueba"] = True
    print(f"  duplicados celda+fecha apartados: "
          f"{sum(1 for e in ev if e['motivo_sin_dato'].startswith('duplicado'))}")

    rem_p = [e for e in ev if e["fuente"].startswith("ReTeRM")
             and e["familia"] == "remocion" and e["detonante_meteorologico"]
             and usable(e)]
    inu_p = [e for e in ev if e["fuente"] == "SENAPRED"
             and e["familia"] == "inundacion" and usable(e)]
    rem_s = [e for e in ev if e["fuente"] == "SENAPRED"
             and e["familia"] == "remocion" and usable(e)]

    # ── el nulo ──────────────────────────────────────────────────────────────
    rng = np.random.default_rng(SEMILLA)

    def piscina(e, mismo_mes):
        """Fechas alternativas válidas en la MISMA celda (y mismo mes si toca)."""
        i, j = e["ij"]
        col = V[k_ini:k_fin + 1, i, j]
        ok = ~np.isnan(col)
        if mismo_mes:
            ok &= (mes_de[k_ini:k_fin + 1] == e["fecha"].month)
        return np.flatnonzero(ok) + k_ini

    def prueba(grupo_a, grupo_b, etiqueta, mismo_mes=True):
        """
        D = media(z | grupo_a) − media(z | grupo_b). grupo_a = remoción.
        Nulo: se rebaraja la FECHA de cada evento dentro de su misma celda.
        """
        za = np.array([e["z_sm"] for e in grupo_a])
        zb = np.array([e["z_sm"] for e in grupo_b])
        d_obs = za.mean() - zb.mean()

        # Piscina de fechas alternativas por evento, en una matriz rellenada para
        # poder barajar las 10.000 réplicas de una sola vez (en bucle puro de
        # Python esto tardaba minutos por prueba).
        cols = []
        for e in grupo_a + grupo_b:
            i, j = e["ij"]
            p = piscina(e, mismo_mes)
            if p.size == 0:              # celda sin días válidos ese mes
                p = piscina(e, False)    # se relaja a mes libre y se declara
            cols.append(V[p, i, j].astype(np.float64))
        largos = np.array([c.size for c in cols])
        M = np.zeros((len(cols), largos.max()))
        for r, c in enumerate(cols):
            M[r, :c.size] = c
        mus = np.array([mu[e["ij"]] for e in grupo_a + grupo_b])
        sds = np.array([sd[e["ij"]] for e in grupo_a + grupo_b])
        na = len(grupo_a)
        relajadas = sum(1 for e in grupo_a + grupo_b
                        if piscina(e, mismo_mes).size == 0)

        # una tirada = una fecha al azar por evento dentro de su propia piscina
        filas = np.arange(len(cols))[None, :]
        d_nulo = np.empty(N_PERMUTACIONES)
        paso = 1000
        for ini in range(0, N_PERMUTACIONES, paso):
            n_it = min(paso, N_PERMUTACIONES - ini)
            sorteo = (rng.random((n_it, len(cols))) * largos).astype(np.int64)
            v = M[filas, sorteo]
            z = (v - mus) / sds
            d_nulo[ini:ini + n_it] = z[:, :na].mean(axis=1) - z[:, na:].mean(axis=1)
        p_bil = float((np.abs(d_nulo) >= abs(d_obs)).mean())
        p_bil = max(p_bil, 1.0 / N_PERMUTACIONES)   # cota por resolución del nulo

        print(f"\n  ── {etiqueta} ──")
        print(f"     n remoción = {len(za)}   n inundación = {len(zb)}")
        print(f"     humedad cruda (máx 30 d previos, m³/m³): "
              f"remoción {np.mean([e['sm_max30_m3m3'] for e in grupo_a]):.4f} · "
              f"inundación {np.mean([e['sm_max30_m3m3'] for e in grupo_b]):.4f}")
        print(f"     z (anomalía dentro de la propia celda): "
              f"remoción {za.mean():+.4f} · inundación {zb.mean():+.4f}")
        print(f"     D = {d_obs:+.4f}   (la hipótesis predice D < 0)")
        print(f"     nulo: media {d_nulo.mean():+.4f} · sd {d_nulo.std():.4f} "
              f"· {N_PERMUTACIONES:,} barajadas"
              f"{' (mismo mes)' if mismo_mes else ' (mes libre)'}"
              f"{f' · {relajadas} eventos sin días válidos en su mes, relajados a mes libre' if relajadas else ''}")
        print(f"     p bilateral = {p_bil:.4f}")
        signo_ok = d_obs < 0
        pasa = (p_bil < 0.01) and signo_ok
        print(f"     signo predicho: {'SÍ' if signo_ok else 'NO'} · "
              f"p<0,01: {'SÍ' if p_bil < 0.01 else 'NO'}  →  "
              f"{'PASA' if pasa else 'NO PASA'}")
        return dict(etiqueta=etiqueta, n_a=len(za), n_b=len(zb), d=d_obs,
                    p=p_bil, pasa=pasa, z_a=za.mean(), z_b=zb.mean(),
                    sm_a=float(np.mean([e['sm_max30_m3m3'] for e in grupo_a])),
                    sm_b=float(np.mean([e['sm_max30_m3m3'] for e in grupo_b])),
                    nulo_sd=float(d_nulo.std()))

    print("\n[4] Prueba contra el nulo de fechas barajadas dentro del mismo punto")
    res = []
    res.append(prueba(rem_p, inu_p,
                      "PRINCIPAL · ReTeRM (remoción) vs SENAPRED (inundación)"))
    res.append(prueba(rem_s, inu_p,
                      "CONTROL SIMÉTRICO · SENAPRED remoción vs SENAPRED inundación"))
    res.append(prueba(rem_p, inu_p,
                      "SENSIBILIDAD · principal con nulo de mes libre",
                      mismo_mes=False))

    # ── control por lluvia ───────────────────────────────────────────────────
    print("\n[5] Control por lluvia (¿la humedad separa aunque la lluvia sea igual?)")
    control_lluvia(rem_p, inu_p, rem_s)

    # ── diagnósticos POSTERIORES (no cuentan para el veredicto) ──────────────
    print("\n[5b] DIAGNÓSTICOS POSTERIORES — se miran DESPUÉS del veredicto y NO")
    print("     lo modifican. Sirven para saber por qué salió lo que salió.")

    # (i) ¿qué tan ciego está el satélite en cada familia?
    print("\n  (i) Cobertura satelital en las celdas de cada familia")
    cubo_ram = np.asarray(cubo)
    frac = (~np.isnan(cubo_ram)).mean(axis=0)
    for etiqueta, grupo in [("remoción ReTeRM", rem_p),
                            ("remoción SENAPRED", rem_s),
                            ("inundación SENAPRED", inu_p)]:
        f = [frac[e["ij"]] for e in grupo]
        print(f"      {etiqueta:22s}: días con dato en su celda "
              f"{100*np.mean(f):.0f}% (mediana {100*np.median(f):.0f}%)")
    # y en las celdas que se PERDIERON por falta de dato
    perdidos = [e for e in ev if e["motivo_sin_dato"].startswith("menos de")
                and e.get("ij") is not None]
    if perdidos:
        f = [frac[e["ij"]] for e in perdidos]
        print(f"      {'eventos descartados':22s}: {100*np.mean(f):.0f}% "
              f"(n={len(perdidos)}) — la pérdida NO es al azar")

    # (ii) ¿y si la humedad se mide con más antelación, fuera del temporal?
    #      Ventana [−60, −31]: no puede contener la lluvia que detona el evento.
    print("\n  (ii) Humedad ANTECEDENTE con ventana [−60, −31] días")
    print("       (el máximo de 30 días previos puede estar recogiendo la propia")
    print("        borrasca que detona el evento; esto lo evita por construcción)")
    V2 = np.full(cubo_ram.shape, np.nan, dtype=np.float32)
    C2 = np.zeros(cubo_ram.shape, dtype=np.int16)
    hay = ~np.isnan(cubo_ram)
    for k in range(31, 61):
        V2[k:] = np.fmax(V2[k:], cubo_ram[:-k])
        C2[k:] += hay[:-k]
    V2[C2 < MIN_DIAS_VALIDOS] = np.nan
    V2w = V2[k_ini:k_fin + 1]
    with np.errstate(invalid="ignore"):
        mu2, sd2 = np.nanmean(V2w, axis=0), np.nanstd(V2w, axis=0)
    sd2[sd2 < 1e-6] = np.nan
    for etiqueta, ga, gb in [("ReTeRM vs SENAPRED", rem_p, inu_p),
                             ("SENAPRED vs SENAPRED", rem_s, inu_p)]:
        za = [float((V2[e["k"]][e["ij"]] - mu2[e["ij"]]) / sd2[e["ij"]]) for e in ga]
        zb = [float((V2[e["k"]][e["ij"]] - mu2[e["ij"]]) / sd2[e["ij"]]) for e in gb]
        za = [z for z in za if np.isfinite(z)]
        zb = [z for z in zb if np.isfinite(z)]
        d = np.mean(za) - np.mean(zb)
        print(f"       {etiqueta:22s}: z remoción {np.mean(za):+.3f} (n={len(za)}) · "
              f"z inundación {np.mean(zb):+.3f} (n={len(zb)}) · D={d:+.3f}")
    del V2, C2, V2w

    # ── salida ───────────────────────────────────────────────────────────────
    print("\n[6] Escribiendo", SALIDA_CSV)
    campos = ["fuente", "familia", "fecha", "comuna", "region", "lat", "lon",
              "ubicacion", "ubicacion_precision", "detonante_meteorologico",
              "detalle", "celda_lat", "celda_lon", "dias_validos",
              "sm_max30_m3m3", "z_sm", "usado_en_prueba", "motivo_sin_dato"]
    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        for e in sorted(ev, key=lambda x: (x["familia"], str(x["fecha"]))):
            e = dict(e)
            e["ubicacion_precision"] = (
                "punto medido en terreno" if e["ubicacion"] == "punto_terreno"
                else "centroide de la comuna" if e["ubicacion"] == "centroide_comunal"
                else "sin dato")
            e["sm_max30_m3m3"] = ("" if e["sm_max30_m3m3"] is None
                                  else f"{e['sm_max30_m3m3']:.5f}")
            e["z_sm"] = "" if e["z_sm"] is None else f"{e['z_sm']:.4f}"
            w.writerow(e)

    # cobertura declarada
    print("\n[7] Cobertura")
    cob = {}
    for e in ev:
        k = (e["fuente"], e["familia"])
        c = cob.setdefault(k, {"total": 0, "con_humedad": 0, "motivos": {}})
        c["total"] += 1
        if e["usado_en_prueba"]:
            c["con_humedad"] += 1
        else:
            c["motivos"][e["motivo_sin_dato"]] = c["motivos"].get(e["motivo_sin_dato"], 0) + 1
    for k, c in sorted(cob.items()):
        print(f"  {k[0]} · {k[1]}: {c['con_humedad']}/{c['total']} con humedad")
        for m, v in sorted(c["motivos"].items(), key=lambda x: -x[1]):
            print(f"       sin dato ({v}): {m}")

    celdas = {e["ij"] for e in ev if e["usado_en_prueba"]}
    lats = [89.875 - 0.25 * (i + I0) for i, _ in celdas]
    print(f"  celdas ESA CCI distintas usadas: {len(celdas)} "
          f"· latitud de {max(lats):.2f} a {min(lats):.2f}")

    print("\n" + "=" * 78)
    print("VEREDICTO (criterio fijado antes de calcular: p<0,01 Y D<0)")
    print("=" * 78)
    for r in res:
        print(f"  {'PASA    ' if r['pasa'] else 'NO PASA '} {r['etiqueta']}"
              f"   D={r['d']:+.4f}  p={r['p']:.4f}")
    print()


def control_lluvia(rem_p, inu_p, rem_s):
    """
    ¿La humedad separa las familias cuando la lluvia es parecida?

    Sólo hay lluvia diaria ERA5 en los 91 puntos de ReTeRM. Para que las dos
    familias tengan la MISMA vara, este control se restringe a los eventos que
    ocurren en comunas donde existe un punto ReTeRM, y usa ese punto como
    representante de la lluvia comunal. Es una aproximación y se declara: la
    comuna es grande y el punto es uno.
    """
    puntos = {}
    with open(RETERM_PUNTOS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            puntos[nx(r["comuna"])] = r["subestacion"]
    serie = {}
    with open(CLIMA_RETERM, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                serie[(r["subestacion"], r["fecha"])] = float(r["precip_mm"])
            except (TypeError, ValueError):
                pass

    def p48(e):
        k = nx(e["comuna"])
        s = puntos.get(ALIAS_COMUNAS.get(k, k))
        if s is None:
            return None
        acum = []
        for atras in range(0, 5):
            d = e["fecha"] - timedelta(days=atras)
            v = serie.get((s, d.isoformat()))
            if v is None:
                return None
            acum.append(v)
        # P48 = mayor acumulado de 2 días consecutivos en los 5 días que terminan
        # en el evento (el día del evento incluido: la lluvia que lo detona)
        return max(acum[k] + acum[k + 1] for k in range(4))

    for etiqueta, ga, gb in [
            ("principal (ReTeRM vs SENAPRED)", rem_p, inu_p),
            ("simétrico (SENAPRED vs SENAPRED)", rem_s, inu_p)]:
        A = [(p48(e), e["z_sm"]) for e in ga]
        B = [(p48(e), e["z_sm"]) for e in gb]
        A = [x for x in A if x[0] is not None]
        B = [x for x in B if x[0] is not None]
        if len(A) < 10 or len(B) < 10:
            print(f"  {etiqueta}: sin dato suficiente de lluvia "
                  f"(n={len(A)}/{len(B)}) — no se puede controlar")
            continue
        print(f"  {etiqueta}: n con lluvia = {len(A)} remoción / {len(B)} inundación")
        print(f"     P48 media: remoción {np.mean([a[0] for a in A]):.1f} mm · "
              f"inundación {np.mean([b[0] for b in B]):.1f} mm")
        # estratos de lluvia comunes a las dos familias
        todo = np.array([a[0] for a in A] + [b[0] for b in B])
        cortes = np.quantile(todo, [0, .25, .5, .75, 1.0])
        for s in range(4):
            lo, hi = cortes[s], cortes[s + 1]
            sa = [z for p, z in A if lo <= p <= hi]
            sb = [z for p, z in B if lo <= p <= hi]
            if len(sa) < 5 or len(sb) < 5:
                print(f"     P48 {lo:5.1f}-{hi:5.1f} mm: n insuficiente "
                      f"({len(sa)}/{len(sb)}) — sin dato")
                continue
            print(f"     P48 {lo:5.1f}-{hi:5.1f} mm: z remoción {np.mean(sa):+.3f} "
                  f"(n={len(sa)}) · z inundación {np.mean(sb):+.3f} (n={len(sb)}) "
                  f"· D={np.mean(sa)-np.mean(sb):+.3f}")


if __name__ == "__main__":
    main()

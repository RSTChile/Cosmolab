#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRUEBA DEL «ENRUTADOR DE AMENAZAS» CON HUMEDAD DE SUELO ERA5-Land (Open-Meteo)
=============================================================================

QUÉ PRUEBA
----------
La hipótesis del estudio `ESTUDIO_VECTORES_DE_AMENAZA.md` §3: el terreno no
modula el peligro, lo ENCAMINA. Consecuencia comprobable: la humedad del suelo
ANTECEDENTE debe venir MÁS BAJA antes de una remoción en masa que antes de una
inundación / desborde fluvial.

POR QUÉ SE REPITE LA PRUEBA
---------------------------
El intento anterior (`PRUEBA_HUMEDAD_ENRUTADOR.md`) usó humedad satelital
ESA CCI y NO PASÓ, pero con un instrumento ciego justo donde ocurre el
fenómeno: 235 de 398 eventos de remoción no tenían NI UN día de dato en los
30 previos, el píxel medía 25 km, y la serie termina el 31-dic-2024.

Aquí se cambia SÓLO la fuente de humedad: ERA5-Land vía la API de archivo de
Open-Meteo. ~9 km de píxel, reanálisis sin huecos, y llega hasta hoy (2026).
La clasificación de los eventos en familias se REUTILIZA tal cual del archivo
`datos/humedad_eventos.csv`, que ya venía verificada.

LAS CUATRO CONDICIONES DE VALIDEZ (fijadas por el director antes de calcular)
----------------------------------------------------------------------------
1. CRITERIO PRE-REGISTRADO — ver constantes `CRITERIO_*` más abajo y la
   sección «Criterio fijado antes» del informe. No se toca después.
2. LIKE-FOR-LIKE COMO PRUEBA PRINCIPAL — SENAPRED remoción vs SENAPRED
   inundación (mismo catálogo, misma forma de ubicar: centroide comunal).
   ReTeRM (coordenada de terreno) vs inundación queda como REPLICACIÓN
   secundaria, reportada aparte y nunca mezclada.
3. CASOS QUEMADOS EXCLUIDOS — el aluvión de Copiapó de marzo 2015 y el temporal
   de julio 2026 ya fueron mirados con este mismo instrumento; fueron
   diagnóstico, no prueba. Salen del conjunto de examen (ver `QUEMADOS`).
4. VENTANA ANTECEDENTE SIN CONTAMINAR — la humedad se promedia en los días
   −30 a −3 respecto de la fecha del evento. Si se incluyera el día del evento
   se estaría midiendo la lluvia que lo causó y la prueba sería circular.
   Variable principal: capa 7-28 cm (condición del terreno). Se reporta
   también 0-7 cm, como secundaria descriptiva.

ESTRUCTURA DEL PROGRAMA
-----------------------
`--etapa bajar`     : baja de Open-Meteo la serie diaria 2015-2026 completa de
                      cada punto único (0,1°) y la guarda CRUDA en
                      datos/crudo/era5land/<fecha-de-captura>/. Reanudable.
`--etapa analizar`  : arma la tabla por evento, corre las dos pruebas con su
                      nulo, los controles por lluvia y por región, y el poder
                      estadístico. Escribe datos/humedad_era5land_eventos.csv.

Lección heredada del proyecto: Open-Meteo cobra por PESO (días × variables),
no por llamadas. Por eso se piden SÓLO tres variables diarias y una sola
petición por punto en vez de una por evento.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# RUTAS Y CONSTANTES
# ---------------------------------------------------------------------------

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / "datos"
EVENTOS_CSV = DATOS / "humedad_eventos.csv"          # entrada, ya clasificada
SALIDA_CSV = DATOS / "humedad_era5land_eventos.csv"  # entregable
FECHA_CAPTURA = "2026-08-16"
CRUDO = DATOS / "crudo" / "era5land" / FECHA_CAPTURA

API = "https://archive-api.open-meteo.com/v1/archive"
V_EVENTO = ("precipitation_sum,soil_moisture_0_to_7cm_mean,"
            "soil_moisture_7_to_28cm_mean")
V_CLIMA = {"7_28": "soil_moisture_7_to_28cm_mean",
           "0_7": "soil_moisture_0_to_7cm_mean"}

RANGO_INI = pd.Timestamp("2015-01-01")
RANGO_FIN = pd.Timestamp("2026-08-10")   # ERA5-Land tiene ~5 días de latencia

# ★ CÓMO COBRA OPEN-METEO, medido a golpes en esta sesión.
#
# Hay DOS límites distintos y los dos muerden:
#   · «Too many concurrent requests» → es de SIMULTANEIDAD. Con 4-8 peticiones
#     en vuelo lo devuelve todo el rato y el cerrojo tarda ~2 min en soltarse.
#     Remedio: UNA petición a la vez.
#   · «Hourly API request limit exceeded» → es de PESO. Una petición no cuenta
#     como una: cuenta como  puntos × (días/14) × (variables/10).  Una sola
#     petición de 50 puntos × 4 años × 3 variables pesa ~1.566 «llamadas», o
#     sea un tercio del cupo de una hora.
#
# Por eso la captura NO baja la serie completa de cada punto. Baja exactamente
# lo que la prueba necesita:
#   A · VENTANAS  — para cada (año, mes) con eventos, los puntos que tuvieron
#       evento ese mes, desde 33 días antes del día 1 hasta fin de mes, con las
#       tres variables. Cubre la ventana antecedente de todos esos eventos.
#       Peso ≈ 1.844.
#   B · CLIMATOLOGÍA — para cada mes calendario, los puntos que alguna vez
#       tuvieron evento en ese mes, el mismo tramo de 63 días, año por año, con
#       UNA variable. Es la urna del nulo y el μ/σ del z.  Peso ≈ 446 por año
#       y por variable, así que se piden años de a uno y se para cuando el cupo
#       se acaba. Se declara en el informe cuántos años se lograron.
PUNTOS_POR_LOTE = 60   # sólo afecta al nº de peticiones, no al peso
DIAS_ANTES_DEL_MES = 33
#
# El plan entero pesa ≈ 7.200 «llamadas» de las 10.000 que da el cupo diario
# gratuito: ventanas 1.843 + climatología 2.676 por capa × 2 capas. Cabe en un
# día, pero no cabe DOS veces: si se vuelve a correr, el crudo ya está en disco
# y no se vuelve a pedir.
ANIOS_CLIMA = [2024, 2023, 2022, 2021, 2020, 2019]

SEMILLA = 20260816
N_BARAJADAS = 10000

# --- CRITERIO PRE-REGISTRADO (condición 1). NO SE TOCA DESPUÉS. -------------
CRITERIO_VENTANA = (-30, -3)   # días respecto del evento, ambos incluidos
CRITERIO_CAPA = "sm_7_28"      # variable principal: humedad 7-28 cm
CRITERIO_SIGNO = "D < 0"       # remoción MÁS SECA que inundación
CRITERIO_P = 0.01              # p bilateral contra el nulo
# D = media(z | remoción) − media(z | inundación)
# Aprueba SÓLO si D < 0 Y p < 0,01. Si falla, se reporta el fallo.
# ---------------------------------------------------------------------------

# --- CASOS QUEMADOS (condición 3): mirados antes con este instrumento -------
QUEMADOS = [
    # (nombre, condición sobre el DataFrame)
    ("aluvión de Copiapó, marzo 2015",
     lambda d: d["region"].str.contains("Atacama", na=False)
     & (d["fecha"] >= pd.Timestamp("2015-03-20"))
     & (d["fecha"] <= pd.Timestamp("2015-04-05"))),
    ("temporal de julio 2026",
     lambda d: (d["fecha"] >= pd.Timestamp("2026-07-01"))
     & (d["fecha"] <= pd.Timestamp("2026-07-31"))),
]
# ---------------------------------------------------------------------------

MIN_DIAS_VENTANA = 28   # la ventana tiene 28 días; ERA5-Land no tiene huecos,
                        # así que se exigen los 28. Si falta uno, «sin dato».
MIN_DIAS_CLIMA = 60     # mínimo de días de climatología para poder calcular
                        # μ y σ de un punto; si no, el evento queda sin z.


def clave_punto(lat: float, lon: float) -> tuple[float, float]:
    """La malla de ERA5-Land es de 0,1°. Dos eventos que caen en la misma
    celda comparten literalmente el mismo valor de humedad."""
    return (round(float(lat), 1), round(float(lon), 1))


MANIFIESTO = "manifiesto.json"


def armar_lotes(puntos: list) -> list:
    """Los puntos se agrupan de a tandas (van ordenados por latitud, así que
    quedan vecinos, que es lo que la API sirve más rápido)."""
    return [puntos[i:i + PUNTOS_POR_LOTE]
            for i in range(0, len(puntos), PUNTOS_POR_LOTE)]


def rango_del_mes(anio: int, mes: int) -> tuple[str, str]:
    """Del día 1 del mes menos 33 días, hasta el último día del mes. Cubre la
    ventana antecedente (−30 a −3) de cualquier día de ese mes."""
    ini = pd.Timestamp(anio, mes, 1) - pd.Timedelta(days=DIAS_ANTES_DEL_MES)
    fin = pd.Timestamp(anio, mes, 1) + pd.offsets.MonthEnd(0)
    ini = max(ini, RANGO_INI - pd.Timedelta(days=DIAS_ANTES_DEL_MES))
    fin = min(fin, RANGO_FIN)
    return ini.strftime("%Y-%m-%d"), fin.strftime("%Y-%m-%d")


def planear_captura() -> dict:
    """Arma el plan completo de peticiones a partir de los eventos.

    Devuelve un manifiesto: qué puntos van en cada petición y en qué orden se
    piden. El orden importa porque el cupo se puede acabar: primero lo
    imprescindible (las ventanas de los eventos), después la climatología año
    por año, empezando por el más reciente.
    """
    ev = pd.read_csv(EVENTOS_CSV, parse_dates=["fecha"])
    ev = ev[(ev.fecha >= RANGO_INI) & (ev.fecha <= RANGO_FIN)].copy()
    ev["pt"] = [clave_punto(a, b) for a, b in zip(ev.lat, ev.lon)]

    ventanas = {}
    for (y, m), g in ev.groupby([ev.fecha.dt.year, ev.fecha.dt.month]):
        ventanas[f"{y:04d}-{m:02d}"] = sorted({p for p in g.pt})
    meses = {}
    for m, g in ev.groupby(ev.fecha.dt.month):
        meses[f"{m:02d}"] = sorted({p for p in g.pt})

    pedidos = []
    for clave, pts in sorted(ventanas.items()):
        y, m = int(clave[:4]), int(clave[5:])
        a, b = rango_del_mes(y, m)
        for k, lote in enumerate(armar_lotes(pts)):
            pedidos.append(dict(tipo="ventana", archivo=f"vent_{clave}_{k}.json.gz",
                                puntos=lote, ini=a, fin=b, variables=V_EVENTO))
    for anio in ANIOS_CLIMA:
        for capa, var in V_CLIMA.items():
            for mes, pts in sorted(meses.items()):
                a, b = rango_del_mes(anio, int(mes))
                for k, lote in enumerate(armar_lotes(pts)):
                    pedidos.append(dict(
                        tipo=f"clima_{capa}", capa=capa, anio=anio,
                        archivo=f"clima_{capa}_{anio}-{mes}_{k}.json.gz",
                        puntos=lote, ini=a, fin=b, variables=var))
    return dict(pedidos=pedidos, ventanas=ventanas, meses=meses,
                anios_clima=ANIOS_CLIMA)


# ---------------------------------------------------------------------------
# ETAPA 1 · BAJAR EL CRUDO
# ---------------------------------------------------------------------------

def _pedir(sesion, pedido, destino, reintentos=6) -> str:
    """Baja un pedido y lo guarda CRUDO (gzip del JSON tal cual llegó; gzip no
    altera un byte del contenido).

    Devuelve 'ok', 'ya', 'cupo' (el cupo de la hora se agotó: hay que parar) o
    'falla:<motivo>'.
    """
    pts = [tuple(p) for p in pedido["puntos"]]
    if destino.exists():
        # No basta con que el archivo esté: tiene que traer tantos puntos como
        # pide el manifiesto, o las series se pegarían a coordenadas ajenas.
        try:
            with gzip.open(destino, "rb") as f:
                if len(json.load(f)) == len(pts):
                    return "ya"
        except Exception:                         # noqa: BLE001
            pass
        destino.unlink()
    params = dict(latitude=",".join(f"{p[0]:.2f}" for p in pts),
                  longitude=",".join(f"{p[1]:.2f}" for p in pts),
                  start_date=pedido["ini"], end_date=pedido["fin"],
                  daily=pedido["variables"], timezone="UTC")
    ultimo = ""
    for intento in range(reintentos):
        try:
            r = sesion.get(API, params=params, timeout=300)
            if r.status_code == 429:
                razon = (r.json().get("reason", "") if "json" in
                         r.headers.get("content-type", "") else r.text)[:80]
                if "Hourly" in razon or "Daily" in razon or "daily" in razon:
                    return f"cupo:{razon}"        # no insistir: se acabó
                ultimo = "429 simultáneas"        # el cerrojo tarda ~2 min
                time.sleep(75 * (intento + 1))
                continue
            r.raise_for_status()
            cuerpo = r.content
            j = json.loads(cuerpo)                # valida que el JSON esté entero
            if not isinstance(j, list) or len(j) != len(pts):
                raise ValueError(f"la respuesta trae {len(j)} puntos, "
                                 f"se pidieron {len(pts)}")
            tmp = destino.with_suffix(".parcial")
            with gzip.open(tmp, "wb") as f:
                f.write(cuerpo)
            tmp.rename(destino)
            return "ok"
        except Exception as e:                    # noqa: BLE001
            ultimo = f"{type(e).__name__}"
            time.sleep(8 * (intento + 1))
    return f"falla:{ultimo}"


def etapa_bajar(esperar_cupo: bool = True) -> None:
    """Baja el plan de `planear_captura()`, UNA petición a la vez y en orden de
    prioridad. Reanudable: lo que ya está en disco no se vuelve a pedir.

    Si el cupo de la hora se agota, espera a que la hora cambie y sigue. Si se
    agotó el cupo del día, para y deja escrito hasta dónde se llegó: la etapa
    de análisis trabaja con lo que haya y lo declara.
    """
    CRUDO.mkdir(parents=True, exist_ok=True)
    plan = planear_captura()
    (CRUDO / MANIFIESTO).write_text(json.dumps(plan, indent=1))
    pedidos = plan["pedidos"]
    n_vent = sum(1 for p in pedidos if p["tipo"] == "ventana")
    print(f"[bajar] plan: {len(pedidos)} peticiones "
          f"({n_vent} de ventanas de evento, el resto climatología)")
    print(f"[bajar] destino: {CRUDO}", flush=True)

    ses = requests.Session()
    cuenta = {"ok": 0, "ya": 0, "falla": 0, "sin_cupo": 0}
    t0 = time.time()
    esperas_cupo = 0
    i = 0
    while i < len(pedidos):
        p = pedidos[i]
        r = _pedir(ses, p, CRUDO / p["archivo"])
        if r.startswith("cupo"):
            if p["tipo"] == "ventana" and esperar_cupo and esperas_cupo < 6:
                # las ventanas son imprescindibles: se espera a la hora nueva
                esperas_cupo += 1
                print(f"  … cupo agotado en {p['archivo']} ({r}). "
                      f"Espera {esperas_cupo}/6 de 10 min.", flush=True)
                time.sleep(620)
                continue
            cuenta["sin_cupo"] += 1
            print(f"  ✗ SIN CUPO en {p['archivo']} → se corta la climatología "
                  f"aquí ({r})", flush=True)
            break
        cuenta["ok" if r == "ok" else ("ya" if r == "ya" else "falla")] += 1
        if r.startswith("falla"):
            print(f"  ! FALLA {p['archivo']} {r}", flush=True)
        if (i + 1) % 10 == 0 or i + 1 == len(pedidos):
            print(f"  {i+1}/{len(pedidos)}  {cuenta}  {time.time()-t0:.0f}s",
                  flush=True)
        i += 1
    print(f"[bajar] LISTO {cuenta} en {time.time()-t0:.0f}s "
          f"({i}/{len(pedidos)} pedidos atendidos)", flush=True)


# ---------------------------------------------------------------------------
# ETAPA 2 · SERIES EN MEMORIA
# ---------------------------------------------------------------------------

@dataclass
class SeriePunto:
    """La serie diaria de un punto, ya convertida a lo que la prueba necesita.

    `ant_*` son medias móviles de la ventana antecedente: el valor en el día t
    es el promedio de los días t−30 .. t−3. Así, «la humedad antecedente de un
    evento» es simplemente `ant_7_28[fecha del evento]`.
    """
    lat: float
    lon: float
    lat_malla: float
    lon_malla: float
    elevacion: float
    fechas: pd.DatetimeIndex
    ant_7_28: np.ndarray
    ant_0_7: np.ndarray
    p48: np.ndarray        # lluvia detonante: máx. suma de 2 días en [−1,+1]
    p_ant: np.ndarray      # lluvia acumulada en la MISMA ventana antecedente
    idx: dict              # fecha -> posición


def _media_ventana(v: np.ndarray, ini: int, fin: int) -> np.ndarray:
    """Media de v sobre el desfase [ini, fin] días (negativos = pasado).

    Devuelve NaN donde la ventana no cabe entera o donde falta algún día.
    """
    n = len(v)
    largo = fin - ini + 1
    acum = np.full(n, np.nan)
    # suma acumulada tolerante a NaN + conteo de válidos
    val = ~np.isnan(v)
    cs = np.concatenate([[0.0], np.nancumsum(np.where(val, v, 0.0))])
    cc = np.concatenate([[0], np.cumsum(val.astype(int))])
    for t in range(n):
        a, b = t + ini, t + fin
        if a < 0 or b >= n:
            continue
        k = cc[b + 1] - cc[a]
        if k < MIN_DIAS_VENTANA or k < largo:
            continue
        acum[t] = (cs[b + 1] - cs[a]) / k
    return acum


def _suma_ventana(v: np.ndarray, ini: int, fin: int) -> np.ndarray:
    """Suma sobre la misma ventana. NaN si falta algún día: no se rellena."""
    n = len(v)
    largo = fin - ini + 1
    out = np.full(n, np.nan)
    val = ~np.isnan(v)
    cs = np.concatenate([[0.0], np.nancumsum(np.where(val, v, 0.0))])
    cc = np.concatenate([[0], np.cumsum(val.astype(int))])
    for t in range(n):
        a, b = t + ini, t + fin
        if a < 0 or b >= n or (cc[b + 1] - cc[a]) < largo:
            continue
        out[t] = cs[b + 1] - cs[a]
    return out


def _p48(precip: np.ndarray) -> np.ndarray:
    """Lluvia detonante: el mayor acumulado de 48 h que toca al evento.

    Se mira la ventana [−1, +1] días alrededor del evento (aquí SÍ se incluye
    el día del evento: esto NO es la variable puesta a prueba, es el control
    por lluvia, y debe representar la borrasca que detonó el evento).
    """
    n = len(precip)
    dos = np.full(n, np.nan)
    dos[1:] = precip[:-1] + precip[1:]           # suma de t−1 y t
    out = np.full(n, np.nan)
    for t in range(n):
        tramo = dos[max(0, t): min(n, t + 2)]    # 48h que terminan en t o en t+1
        tramo = tramo[~np.isnan(tramo)]
        if len(tramo):
            out[t] = tramo.max()
    return out


CALENDARIO = pd.date_range(RANGO_INI - pd.Timedelta(days=DIAS_ANTES_DEL_MES),
                           RANGO_FIN, freq="D")
POS = {f: i for i, f in enumerate(CALENDARIO)}


def cargar_series() -> tuple[dict, dict]:
    """Pega todos los pedidos del crudo en un calendario diario COMPLETO por
    punto (los días que no se pidieron quedan en NaN, nunca rellenados).

    Trabajar sobre un calendario completo es lo que permite calcular la ventana
    antecedente con aritmética de posiciones: si faltara un día, la ventana
    saltaría 24 h sin avisar.
    """
    if not (CRUDO / MANIFIESTO).exists():
        # sin manifiesto no hay nada bajado: se deja el plan escrito igual, para
        # que la tabla del conjunto de examen se pueda producir de todos modos
        CRUDO.mkdir(parents=True, exist_ok=True)
        (CRUDO / MANIFIESTO).write_text(json.dumps(planear_captura(), indent=1))
    ini, fin = CRITERIO_VENTANA
    man = json.loads((CRUDO / MANIFIESTO).read_text())

    n = len(CALENDARIO)
    crudo: dict = {}          # punto -> {'p':arr,'s07':arr,'s728':arr, meta}
    logrado = {"ventana": 0, "clima_7_28": 0, "clima_0_7": 0}
    anios_logrados = {"7_28": set(), "0_7": set()}

    def hueco():
        return dict(p=np.full(n, np.nan), s07=np.full(n, np.nan),
                    s728=np.full(n, np.nan), meta=None)

    for ped in man["pedidos"]:
        arch = CRUDO / ped["archivo"]
        if not arch.exists():
            continue
        with gzip.open(arch, "rb") as f:
            resp = json.load(f)
        pts = [tuple(p) for p in ped["puntos"]]
        if len(resp) != len(pts):
            raise ValueError(f"{arch.name} trae {len(resp)} puntos y el "
                             f"manifiesto dice {len(pts)}")
        logrado[ped["tipo"]] = logrado.get(ped["tipo"], 0) + 1
        if ped["tipo"].startswith("clima_"):
            anios_logrados[ped["capa"]].add(int(ped["anio"]))
        for k, obj in enumerate(resp):
            c = crudo.setdefault(pts[k], hueco())
            c["meta"] = c["meta"] or obj
            d = obj["daily"]
            pos = np.array([POS[f] for f in pd.to_datetime(d["time"])])
            for col, nombre in (("p", "precipitation_sum"),
                                ("s07", "soil_moisture_0_to_7cm_mean"),
                                ("s728", "soil_moisture_7_to_28cm_mean")):
                if nombre in d:
                    c[col][pos] = np.array(d[nombre], dtype=float)

    series = {}
    for clave, c in crudo.items():
        meta = c["meta"]
        series[clave] = SeriePunto(
            lat=clave[0], lon=clave[1],
            lat_malla=meta["latitude"], lon_malla=meta["longitude"],
            elevacion=meta.get("elevation", np.nan),
            fechas=CALENDARIO,
            ant_7_28=_media_ventana(c["s728"], ini, fin),
            ant_0_7=_media_ventana(c["s07"], ini, fin),
            p48=_p48(c["p"]),
            p_ant=_suma_ventana(c["p"], ini, fin),
            idx=POS,
        )
    cobertura = dict(
        pedidos_en_disco=logrado,
        anios_clima_7_28=sorted(anios_logrados["7_28"]),
        anios_clima_0_7=sorted(anios_logrados["0_7"]),
        pedidos_del_plan=len(man["pedidos"]),
    )
    print(f"[analizar] {len(series)} puntos armados desde el crudo")
    print(f"[analizar] climatología lograda: 7-28 cm años "
          f"{cobertura['anios_clima_7_28']} · 0-7 cm años "
          f"{cobertura['anios_clima_0_7']}")
    return series, cobertura


# ---------------------------------------------------------------------------
# ETAPA 2 · TABLA POR EVENTO
# ---------------------------------------------------------------------------

def mascara_clima(cobertura: dict, capa: str) -> np.ndarray:
    """Días del calendario que pertenecen a los años de climatología logrados
    para esa capa. Son los únicos que se usan como referencia (μ, σ) y como
    urna del nulo: si se usara cualquier día bajado, la referencia sería una
    mezcla de años arbitraria y distinta punto por punto."""
    anios = set(cobertura[f"anios_clima_{capa}"])
    return np.array([f.year in anios for f in CALENDARIO])


def armar_tabla(series: dict, cobertura: dict) -> pd.DataFrame:
    ev = pd.read_csv(EVENTOS_CSV, parse_dates=["fecha"])
    ev = ev[["fuente", "familia", "fecha", "comuna", "region", "lat", "lon",
             "ubicacion_precision", "detonante_meteorologico", "detalle"]].copy()

    # -- condición 3: marcar los casos quemados ----------------------------
    ev["quemado"] = ""
    for nombre, cond in QUEMADOS:
        m = cond(ev)
        ev.loc[m & (ev["quemado"] == ""), "quemado"] = nombre

    filas = []
    for r in ev.itertuples():
        cl = clave_punto(r.lat, r.lon)
        s = series.get(cl)
        base = dict(
            fuente=r.fuente, familia=r.familia, fecha=r.fecha.date(),
            comuna=r.comuna, region=r.region, lat=r.lat, lon=r.lon,
            ubicacion_precision=r.ubicacion_precision,
            detonante_meteorologico=r.detonante_meteorologico,
            detalle=r.detalle, quemado=r.quemado,
            punto_lat=cl[0], punto_lon=cl[1],
            elevacion_m=np.nan,
            sm_7_28_ant=np.nan, sm_0_7_ant=np.nan,
            p48_mm=np.nan, p_ant_mm=np.nan,
            z_7_28=np.nan, z_0_7=np.nan,
            usado_en_prueba=False, prueba="", motivo_sin_dato="",
        )
        if s is None:
            base["motivo_sin_dato"] = "punto sin serie ERA5-Land descargada"
            filas.append(base); continue
        base["elevacion_m"] = s.elevacion
        i = s.idx.get(pd.Timestamp(r.fecha))
        if i is None:
            base["motivo_sin_dato"] = (
                f"fecha fuera del periodo {RANGO_INI.date()}..{RANGO_FIN.date()}")
            filas.append(base); continue
        if np.isnan(s.ant_7_28[i]):
            base["motivo_sin_dato"] = "ventana antecedente incompleta"
            filas.append(base); continue
        base["sm_7_28_ant"] = s.ant_7_28[i]
        base["sm_0_7_ant"] = s.ant_0_7[i]
        base["p48_mm"] = s.p48[i]
        base["p_ant_mm"] = s.p_ant[i]
        filas.append(base)

    t = pd.DataFrame(filas)

    # -- z por punto: ¿venía este punto más seco o más húmedo QUE SU PROPIA
    #    COSTUMBRE? Sin esto se compararía el norte árido con el sur húmedo,
    #    o sea geografía, no encaminamiento.
    msk = {c: mascara_clima(cobertura, c) for c in ("7_28", "0_7")}
    mu, sd = {}, {}
    for cl, s in series.items():
        for cual, arr in (("7_28", s.ant_7_28), ("0_7", s.ant_0_7)):
            v = arr[msk[cual] & ~np.isnan(arr)]
            mu[(cl, cual)] = v.mean() if len(v) >= MIN_DIAS_CLIMA else np.nan
            sd[(cl, cual)] = (v.std(ddof=1) if len(v) >= MIN_DIAS_CLIMA
                              else np.nan)
    for cual, col in (("7_28", "sm_7_28_ant"), ("0_7", "sm_0_7_ant")):
        zz = []
        for r in t.itertuples():
            k = ((r.punto_lat, r.punto_lon), cual)
            m, d = mu.get(k, np.nan), sd.get(k, np.nan)
            v = getattr(r, col)
            zz.append((v - m) / d if (not np.isnan(v) and d and d > 0
                                      and not np.isnan(m)) else np.nan)
        t[f"z_{cual}"] = zz

    t.loc[t["z_7_28"].isna() & (t["motivo_sin_dato"] == ""),
          "motivo_sin_dato"] = "punto sin climatología suficiente"

    # -- deduplicar: mismo punto + misma fecha + misma familia = un evento.
    #    Comparten el mismo valor de humedad; contarlos dos veces infla la n.
    t["_k"] = list(zip(t.punto_lat, t.punto_lon, t.fecha, t.familia, t.fuente))
    dup = t.duplicated("_k") & t["motivo_sin_dato"].eq("")
    t.loc[dup, "motivo_sin_dato"] = "duplicado de punto, fecha y familia"
    t = t.drop(columns=["_k"])
    return t, mu, sd


# ---------------------------------------------------------------------------
# ETAPA 2 · LA PRUEBA
# ---------------------------------------------------------------------------

def _urna_nula(series: dict, cl, mes: int, col: str,
               msk_clima: np.ndarray) -> np.ndarray:
    """Todas las fechas del MISMO punto y el MISMO mes calendario, dentro de
    los años de climatología, con ventana antecedente completa. Es el nulo
    NULL-1 del banco de pruebas: controla «este lugar, en esta época del año,
    es así siempre»."""
    s = series[cl]
    arr = s.ant_7_28 if col == "7_28" else s.ant_0_7
    m = np.asarray(CALENDARIO.month == mes) & msk_clima & ~np.isnan(arr)
    return arr[m]


def correr_prueba(nombre: str, rem: pd.DataFrame, inu: pd.DataFrame,
                  series: dict, mu: dict, sd: dict, msk_clima: np.ndarray,
                  col: str = "7_28",
                  n_baraj: int = N_BARAJADAS, verboso: bool = True) -> dict:
    """Calcula D, su nulo y el p bilateral. Nada de esto depende del resultado:
    el criterio ya está escrito arriba."""
    zc = f"z_{col}"
    zr = rem[zc].to_numpy(float)
    zi = inu[zc].to_numpy(float)
    D = zr.mean() - zi.mean()

    rng = np.random.default_rng(SEMILLA)
    # urnas precalculadas (en unidades z) por (punto, mes)
    cache = {}
    def urna(cl, mes):
        k = (cl, mes)
        if k not in cache:
            v = _urna_nula(series, cl, mes, col, msk_clima)
            m, d = mu.get((cl, col), np.nan), sd.get((cl, col), np.nan)
            cache[k] = (v - m) / d if (d and d > 0) else np.array([])
        return cache[k]

    vacias = {"n": 0}

    def urnas_de(df):
        out = []
        for r in df.itertuples():
            cl = (r.punto_lat, r.punto_lon)
            mes = pd.Timestamp(r.fecha).month
            u = urna(cl, mes)
            if not len(u):                # sin urna: el evento se representa a
                vacias["n"] += 1          # sí mismo (nulo conservador)
                u = np.array([getattr(r, zc)])
            out.append(u)
        return out

    # Cada evento aporta una muestra al azar de SU urna, en cada una de las
    # n_baraj barajadas. Se hace vectorizado por evento (no por barajada).
    def sortear(urnas):
        acc = np.zeros(n_baraj)
        for u in urnas:
            acc += u[rng.integers(0, len(u), size=n_baraj)]
        return acc / len(urnas)

    ur, ui = urnas_de(rem), urnas_de(inu)
    nulos = sortear(ur) - sortear(ui)

    # p bilateral de permutación (definido antes de mirar el resultado)
    p_izq = (np.sum(nulos <= D) + 1) / (n_baraj + 1)
    p_der = (np.sum(nulos >= D) + 1) / (n_baraj + 1)
    p = min(1.0, 2 * min(p_izq, p_der))

    res = dict(
        prueba=nombre, n_rem=len(rem), n_inu=len(inu),
        z_rem=zr.mean(), z_inu=zi.mean(), D=D,
        nulo_media=nulos.mean(), nulo_sd=nulos.std(ddof=1),
        p=p, sigmas=(D - nulos.mean()) / nulos.std(ddof=1),
        signo_ok=bool(D < 0), p_ok=bool(p < CRITERIO_P),
        cruda_rem=rem[f"sm_{col}_ant"].mean(),
        cruda_inu=inu[f"sm_{col}_ant"].mean(),
        # poder: qué apartamiento del nulo habría dado p<0,01 bilateral
        detectable_z=2.5758 * nulos.std(ddof=1),
        urnas_vacias=vacias["n"],
        urna_mediana=(int(np.median([len(u) for u in ur + ui]))
                      if (ur or ui) else 0),
    )
    res["veredicto"] = "PASA" if (res["signo_ok"] and res["p_ok"]) else "NO PASA"
    if verboso:
        print(f"\n=== {nombre} · capa {col} ===")
        print(f"  n remoción={res['n_rem']}  n inundación={res['n_inu']}")
        print(f"  z remoción={res['z_rem']:+.3f}  z inundación={res['z_inu']:+.3f}")
        print(f"  cruda (m3/m3) remoción={res['cruda_rem']:.4f}  "
              f"inundación={res['cruda_inu']:.4f}")
        print(f"  D={res['D']:+.4f}   nulo={res['nulo_media']:+.4f} "
              f"± {res['nulo_sd']:.4f}   ({res['sigmas']:+.2f} sd)")
        print(f"  p bilateral={res['p']:.4f}   signo predicho (D<0): "
              f"{'SÍ' if res['signo_ok'] else 'NO'}")
        print(f"  urna del nulo: mediana {res['urna_mediana']} fechas "
              f"candidatas por evento; {res['urnas_vacias']} sin urna")
        print(f"  ►► {res['veredicto']}")
    return res


def control_lluvia(nombre, rem, inu, n_estratos=4):
    """¿La diferencia es sólo «llovió más»? Se compara dentro de estratos de
    P48 comparable, definidos sobre la muestra combinada."""
    tot = pd.concat([rem.assign(_f="rem"), inu.assign(_f="inu")])
    tot = tot[tot.p48_mm.notna()]
    if len(tot) < 20:
        return []
    cortes = np.quantile(tot.p48_mm, np.linspace(0, 1, n_estratos + 1))
    cortes[0] -= 1e-9
    tot["_e"] = pd.cut(tot.p48_mm, bins=np.unique(cortes), labels=False,
                       include_lowest=True)
    out = []
    for e, g in tot.groupby("_e"):
        a, b = g[g._f == "rem"], g[g._f == "inu"]
        if len(a) < 3 or len(b) < 3:
            out.append(dict(estrato=int(e), n_rem=len(a), n_inu=len(b),
                            p48_lo=g.p48_mm.min(), p48_hi=g.p48_mm.max(),
                            D=np.nan))
            continue
        out.append(dict(estrato=int(e), n_rem=len(a), n_inu=len(b),
                        p48_lo=g.p48_mm.min(), p48_hi=g.p48_mm.max(),
                        D=a.z_7_28.mean() - b.z_7_28.mean()))
    print(f"\n--- control por lluvia · {nombre} ---")
    print(f"  P48 medio: remoción {rem.p48_mm.mean():.1f} mm  ·  "
          f"inundación {inu.p48_mm.mean():.1f} mm")
    for o in out:
        d = f"{o['D']:+.3f}" if not np.isnan(o["D"]) else "sin n"
        print(f"  estrato {o['estrato']} [{o['p48_lo']:.0f}–{o['p48_hi']:.0f} mm] "
              f"n={o['n_rem']}/{o['n_inu']}  D={d}")
    return out


def control_region(nombre, rem, inu, minimo=5):
    """¿El efecto es sólo «el sur es más húmedo»? Se compara DENTRO de cada
    región y se promedia ponderando por el n armónico de cada región."""
    regs = sorted(set(rem.region) & set(inu.region))
    filas, pesos, ds = [], [], []
    for rg in regs:
        a, b = rem[rem.region == rg], inu[inu.region == rg]
        if len(a) < minimo or len(b) < minimo:
            continue
        d = a.z_7_28.mean() - b.z_7_28.mean()
        w = 1.0 / (1.0 / len(a) + 1.0 / len(b))
        filas.append(dict(region=rg, n_rem=len(a), n_inu=len(b), D=d))
        pesos.append(w); ds.append(d)
    D_intra = float(np.average(ds, weights=pesos)) if ds else np.nan
    print(f"\n--- control por región · {nombre} ---")
    for f in filas:
        print(f"  {f['region']:<28} n={f['n_rem']:>3}/{f['n_inu']:<3} "
              f"D={f['D']:+.3f}")
    print(f"  D intra-región ponderado = {D_intra:+.4f}  "
          f"({len(filas)} regiones con n≥{minimo} en las dos familias)")
    return filas, D_intra


# ---------------------------------------------------------------------------

def etapa_analizar() -> None:
    series, cobertura = cargar_series()
    t, mu, sd = armar_tabla(series, cobertura)
    msk = {c: mascara_clima(cobertura, c) for c in ("7_28", "0_7")}

    print("\n" + "=" * 72)
    print("CRITERIO FIJADO ANTES (no se modifica según el resultado)")
    print("=" * 72)
    print(f"  variable principal : humedad de suelo {CRITERIO_CAPA} (7-28 cm)")
    print(f"  ventana            : días {CRITERIO_VENTANA[0]} a "
          f"{CRITERIO_VENTANA[1]} respecto del evento")
    print(f"  estadístico        : D = media(z|remoción) − media(z|inundación)")
    print(f"  predicción         : {CRITERIO_SIGNO}")
    print(f"  nulo               : {N_BARAJADAS} barajadas de la fecha dentro "
          f"del mismo punto y mes")
    print(f"  aprueba si         : D < 0  Y  p bilateral < {CRITERIO_P}")

    # -- conjunto de examen -------------------------------------------------
    con_dato = t["motivo_sin_dato"].eq("") & t["z_7_28"].notna()
    quemado = t["quemado"].ne("")
    t.loc[quemado & con_dato, "motivo_sin_dato"] = t.loc[
        quemado & con_dato, "quemado"].map(lambda q: f"caso quemado: {q}")
    con_dato = t["motivo_sin_dato"].eq("") & t["z_7_28"].notna()

    print("\n--- cobertura ---")
    print(t.groupby(["fuente", "familia"]).size().to_string())
    print("\nmotivos de exclusión:")
    print(t.loc[~con_dato, "motivo_sin_dato"].value_counts().to_string())
    print("\nquemados hallados (excluidos del examen):")
    if quemado.any():
        print(t[quemado].groupby(["quemado", "fuente", "familia"]).size().to_string())
    else:
        print("  ninguno")

    if not con_dato.any():
        # No hay humedad bajada (típicamente: el cupo diario de Open-Meteo se
        # agotó). Se escribe igual la tabla del conjunto de examen —qué evento
        # entra, cuál no y por qué— y se para sin inventar ningún número.
        t.loc[t["motivo_sin_dato"].eq(""), "motivo_sin_dato"] = \
            "sin humedad: captura ERA5-Land no realizada"
        t.to_csv(SALIDA_CSV, index=False)
        print(f"\n[analizar] NO HAY HUMEDAD BAJADA. Se escribió {SALIDA_CSV} "
              f"sólo con el conjunto de examen ({len(t)} filas) y NO se calculó "
              f"ninguna prueba. Corre --etapa bajar cuando haya cupo.")
        return

    base = t[con_dato]
    inu = base[(base.fuente == "SENAPRED") & (base.familia == "inundacion")]
    rem_sp = base[(base.fuente == "SENAPRED") & (base.familia == "remocion")]
    rem_rt = base[base.fuente.str.contains("ReTeRM") &
                  (base.familia == "remocion") &
                  (base.detonante_meteorologico == True)]  # noqa: E712

    # un punto-fecha que aparece en las DOS familias no pertenece a ninguna
    par_inu = set(zip(inu.punto_lat, inu.punto_lon, inu.fecha))
    mix_sp = [i for i, r in rem_sp.iterrows()
              if (r.punto_lat, r.punto_lon, r.fecha) in par_inu]
    mix_rt = [i for i, r in rem_rt.iterrows()
              if (r.punto_lat, r.punto_lon, r.fecha) in par_inu]
    t.loc[mix_sp + mix_rt, "motivo_sin_dato"] = \
        "mismo punto y fecha en las dos familias"
    rem_sp = rem_sp.drop(index=mix_sp)
    rem_rt = rem_rt.drop(index=mix_rt)
    print(f"\npunto-fecha en las dos familias, apartados: "
          f"SENAPRED {len(mix_sp)} · ReTeRM {len(mix_rt)}")

    resultados = []
    # ---- PRUEBA PRINCIPAL: like-for-like ---------------------------------
    r1 = correr_prueba("PRINCIPAL · SENAPRED remoción vs SENAPRED inundación",
                       rem_sp, inu, series, mu, sd, msk["7_28"], "7_28")
    r1b = correr_prueba("PRINCIPAL · capa 0-7 cm (secundaria descriptiva)",
                        rem_sp, inu, series, mu, sd, msk["0_7"], "0_7")
    # ---- PRUEBA SECUNDARIA: replicación con coordenada de terreno ---------
    r2 = correr_prueba("SECUNDARIA (replicación) · ReTeRM vs SENAPRED inundación",
                       rem_rt, inu, series, mu, sd, msk["7_28"], "7_28")
    r2b = correr_prueba("SECUNDARIA · capa 0-7 cm (secundaria descriptiva)",
                        rem_rt, inu, series, mu, sd, msk["0_7"], "0_7")
    resultados += [r1, r1b, r2, r2b]

    control_lluvia("PRINCIPAL", rem_sp, inu)
    control_region("PRINCIPAL", rem_sp, inu)
    control_lluvia("SECUNDARIA", rem_rt, inu)
    control_region("SECUNDARIA", rem_rt, inu)

    print("\n--- poder estadístico ---")
    for r in (r1, r2):
        sdz = r["nulo_sd"]
        print(f"  {r['prueba'][:40]:<42} sd nulo={sdz:.4f} → detectable "
              f"|D−nulo| ≥ {2.5758*sdz:.3f} z a p<0,01")

    # ---- ilustración (NO prueba): los casos quemados ----------------------
    print("\n--- ilustración, NO forma parte del examen ---")
    for nombre, _ in QUEMADOS:
        q = t[t.quemado == nombre]
        q = q[q.sm_7_28_ant.notna()]
        if len(q):
            print(f"  {nombre}: n={len(q)}  humedad 7-28 antecedente media="
                  f"{q.sm_7_28_ant.mean():.4f} m3/m3  z medio="
                  f"{q.z_7_28.mean():+.3f}  P48 medio={q.p48_mm.mean():.1f} mm")
        else:
            print(f"  {nombre}: sin eventos con dato en el conjunto")

    # ---- marcar en la tabla qué entró en qué prueba -----------------------
    for idx, etiqueta in ((rem_sp.index, "principal_remocion"),
                          (inu.index, "inundacion_ambas"),
                          (rem_rt.index, "secundaria_remocion")):
        t.loc[idx, "usado_en_prueba"] = True
        t.loc[idx, "prueba"] = etiqueta
    t.to_csv(SALIDA_CSV, index=False)
    print(f"\n[analizar] escrito {SALIDA_CSV} ({len(t)} filas)")
    print(f"[analizar] cobertura de la captura: {cobertura}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--etapa", choices=["bajar", "analizar"], required=True)
    a = ap.parse_args()
    if a.etapa == "bajar":
        etapa_bajar()
    else:
        etapa_analizar()


if __name__ == "__main__":
    main()

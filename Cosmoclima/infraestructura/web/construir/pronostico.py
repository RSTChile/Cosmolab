"""
PRONÓSTICO · lo único que le falta al instrumento para mirar hacia adelante
============================================================================

★ EL HUECO QUE ESTE ARCHIVO CIERRA
------------------------------------
Hasta hoy, **el proyecto entero miraba sólo hacia atrás**. Los diecinueve
adaptadores y los cinco descargadores de clima apuntan todos al mismo sitio:

    https://archive-api.open-meteo.com/v1/archive     ← REANÁLISIS HISTÓRICO

Con eso se construye un **diagnóstico** —qué pasó, dónde, con qué frecuencia— y
eso ya estaba. Un **predictor** responde otra cosa: qué va a pasar el jueves. Este
es el primer archivo del proyecto que consulta el futuro.

    https://api.open-meteo.com/v1/forecast           ← PRONÓSTICO

★ POR QUÉ ESTO ES BARATO Y EL BARRIDO HISTÓRICO NO
----------------------------------------------------
El barrido histórico se topó con un muro medido el 22-ago: `HTTP 429`, techo de
~90 celdas por hora. La causa no era el número de celdas sino **el peso de cada
petición**: Open-Meteo cobra por días × variables, y cada celda pedía 36 años,
o sea 13.383 días.

El pronóstico pide **16 días**. Es unas ochocientas veces más liviano:

    357 celdas × 16 días  =  5.712 valores por refresco
    frente a 4,78 millones de valores del archivo histórico

Por eso el predictor cuesta menos que el diagnóstico, y por eso este barrido
completo cabe holgado en el cupo gratuito.

★ AUN ASÍ: UN BARRIDO DIARIO CACHEADO, NUNCA UNA LLAMADA POR VISITA
---------------------------------------------------------------------
Si la aplicación llamara al servicio en cada consulta, cien visitas serían cien
barridos y volveríamos al 429. El pronóstico se baja **una vez al día** y se
guarda; la aplicación lee el archivo.

★ Y SI EL REFRESCO FALLA, SE DICE
-----------------------------------
El archivo guarda `generado` y `celdas_fallidas`. La aplicación **muestra la
antigüedad del pronóstico en pantalla** y no finge estar al día. Un pronóstico de
hace tres días presentado como el de hoy sería peor que no tener ninguno.

USO
---
    ../../.venv-esa/bin/python construir/pronostico.py
    ../../.venv-esa/bin/python construir/pronostico.py --prueba   # 5 celdas
"""

import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
DATOS = AQUI.parent / "publico" / "datos"
CELDAS = RAIZ / "datos" / "clima_celdas_hechas.txt"
SALIDA = DATOS / "pronostico.json"

MALLA = 0.10          # la misma rejilla de 0,1° que usa todo el proyecto
DIAS = 16             # el máximo que entrega el servicio gratuito
PAUSA_S = 0.6         # holgado: 357 celdas ≈ 4 min
REINTENTOS = 4

URL = ("https://api.open-meteo.com/v1/forecast"
       "?latitude={lat}&longitude={lon}"
       "&daily=precipitation_sum"
       "&forecast_days=" + str(DIAS) +
       "&timezone=America%2FSantiago")


def centro(clave):
    """De la clave de celda `-370_-732` al centro en grados decimales."""
    i, j = clave.split("_")
    return round(int(i) * MALLA, 4), round(int(j) * MALLA, 4)


# ★★ EL TERRITORIO INSULAR, QUE NINGUNA DE LAS DOS SERIES ALCANZA
#   ERA5-Land no cubre las islas: su máscara de tierra descarta las oceánicas
#   pequeñas y se comprobó bajando el recuadro completo — cero celdas con dato
#   en 26.001 puntos. Y el barrido histórico de Open-Meteo quedó a medias antes
#   de llegar a ellas. Sin esta lista, Juan Fernández aparece «sin cobertura»
#   para siempre, que es distinto de «no le va a llover».
#   Open-Meteo sí las sirve, porque cae a ERA5 de 0,25°, que incluye océano.
INSULARES = {
    "-336_-788": "Robinson Crusoe (Juan Fernández)",
    "-263_-801": "San Félix (Desventuradas)",
    "-271_-1094": "Isla de Pascua",
}


def celdas():
    """Las celdas para las que se pide pronóstico.

    ★★ POR QUÉ YA NO BASTA `clima_celdas_hechas.txt`
    -------------------------------------------------
    Ese archivo lista lo que Open-Meteo alcanzó a bajar del histórico, y su
    barrido nunca terminó: quedó en 681 de ~5.200 celdas. Usarlo como universo
    dejaba **16 comunas continentales sin una sola celda de pronóstico** —1.850
    activos, entre ellas Aisén, Longaví, Retiro, Colbún y Cochrane—, que en la
    aplicación aparecían como «sin cobertura» cuando en realidad SÍ tienen
    historia: la tienen en ERA5-Land, que cubre todo Chile continental.

    Así que el universo pasa a ser **las celdas con historia en ERA5-Land**, que
    es la serie completa. La regla de fondo no cambia y sigue siendo la razón de
    todo esto: sólo se pide pronóstico donde hay historia contra la cual medirlo.
    Un pronóstico sin climatología propia es un número sin referencia.
    """
    e5 = RAIZ / "datos" / "clima_diario_celdas_era5land.csv"
    vistas = set()
    if e5.exists():
        with e5.open(encoding="utf-8") as fh:
            next(fh, None)
            for linea in fh:
                c = linea.split(",", 1)[0]
                if c:
                    vistas.add(c)
    # Las de Open-Meteo se conservan: incluyen el territorio insular, que
    # ERA5-Land no cubre (su máscara de tierra descarta las islas oceánicas
    # pequeñas — comprobado: cero celdas con dato en Pascua, Juan Fernández y
    # Desventuradas). Sin ellas se perdería Isla de Pascua.
    if CELDAS.exists():
        vistas |= set(CELDAS.read_text().split())
    vistas |= set(INSULARES)
    return sorted(vistas)


def bajar(clave):
    lat, lon = centro(clave)
    for intento in range(REINTENTOS):
        try:
            with urllib.request.urlopen(URL.format(lat=lat, lon=lon),
                                        timeout=60) as r:
                d = json.loads(r.read())["daily"]
            return {"fechas": d["time"],
                    "mm": [0.0 if v is None else round(float(v), 1)
                           for v in d["precipitation_sum"]]}
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(int(e.headers.get("Retry-After", 8 * (intento + 1))))
                continue
            time.sleep(3 * (intento + 1))
        except Exception:
            time.sleep(3 * (intento + 1))
    return None


def completar():
    """Baja SÓLO las celdas que le faltan al pronóstico ya escrito.

    Rehacer las 3.427 son 90 minutos; completar tres islas son segundos. Se usa
    cuando se amplía el universo de celdas y no se quiere volver a empezar.
    """
    if not SALIDA.exists():
        print("  no hay pronostico.json que completar")
        return None
    doc = json.loads(SALIDA.read_text(encoding="utf-8"))
    faltan = [c for c in celdas() if c not in doc["celdas"]]
    print(f"  celdas en el archivo : {len(doc['celdas']):,}")
    print(f"  por completar        : {len(faltan)}")
    if not faltan:
        return {"celdas_pronostico": len(doc["celdas"]), "fallidas": 0}

    nuevas, fallidas = 0, []
    for c in faltan:
        d = bajar(c)
        if d is None:
            fallidas.append(c)
            continue
        # ⚠️ Sólo se acepta si la ventana calza con la que ya está escrita: dos
        #    celdas con fechas distintas en el mismo archivo darían un mapa donde
        #    cada comuna habla de un día diferente.
        if d["fechas"] != doc["fechas"]:
            print(f"    {c}: ventana distinta ({d['fechas'][0]}…), se omite")
            fallidas.append(c)
            continue
        doc["celdas"][c] = d["mm"]
        nuevas += 1
        print(f"    + {c} {INSULARES.get(c, '')}", flush=True)
        time.sleep(PAUSA_S)

    doc["celdas_fallidas"] = sorted(set(doc.get("celdas_fallidas", [])) | set(fallidas))
    SALIDA.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print(f"\n  añadidas {nuevas} · ahora {len(doc['celdas']):,} celdas")
    return {"celdas_pronostico": len(doc["celdas"]), "fallidas": len(fallidas)}


def construir(prueba=False):
    lista = celdas()
    if prueba:
        lista = lista[:5]
        # ★★ `--prueba` NO puede escribir sobre el archivo que consume la
        #    aplicación. Pasó el 24-ago: una prueba de 5 celdas pisó las 357 de
        #    producción y el mapa quedó COMPLETAMENTE GRIS —todas las comunas en
        #    «sin cobertura»— sin un solo error en consola. La web no distingue
        #    «no hay datos» de «los datos dicen que no llueve»: se ve igual.
        globals()["SALIDA"] = SALIDA.with_name("pronostico_prueba.json")
    if not lista:
        print("  ✗ no hay celdas. Falta datos/clima_celdas_hechas.txt")
        return None

    print(f"  celdas a consultar : {len(lista)}")
    print(f"  ventana            : {DIAS} días")
    print(f"  peso               : {len(lista)*DIAS:,} valores "
          f"(el barrido histórico eran 4.780.000)")
    print()

    t0 = time.time()
    out, fallidas = {}, []
    fechas = None
    for k, c in enumerate(lista, 1):
        d = bajar(c)
        if d is None:
            fallidas.append(c)
        else:
            fechas = fechas or d["fechas"]
            out[c] = d["mm"]
        if k % 50 == 0 or k == len(lista):
            print(f"    {k}/{len(lista)} · {len(fallidas)} fallidas · "
                  f"{k/max(time.time()-t0,1)*60:.0f} celdas/min", flush=True)
        time.sleep(PAUSA_S)

    DATOS.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps({
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fuente": "Open-Meteo · api.open-meteo.com/v1/forecast",
        "dias": DIAS,
        "fechas": fechas or [],
        "celdas": out,
        "celdas_fallidas": fallidas,
    }, ensure_ascii=False), encoding="utf-8")

    print(f"\n  celdas con pronóstico : {len(out)}")
    if fallidas:
        print(f"  ⚠️  fallidas            : {len(fallidas)} — la aplicación las "
              f"mostrará como sin dato, no como sin lluvia")
    print(f"  escrito: {SALIDA.name} ({SALIDA.stat().st_size/1e3:.0f} KB)")
    if fechas:
        print(f"  ventana: {fechas[0]} → {fechas[-1]}")
    return {"celdas_pronostico": len(out), "fallidas": len(fallidas)}


if __name__ == "__main__":
    print("=" * 70)
    print("PRONÓSTICO · el primer archivo del proyecto que mira hacia adelante")
    print("=" * 70)
    if "--completar" in sys.argv:
        r = completar()
    else:
        r = construir("--prueba" in sys.argv)
    sys.exit(0 if r else 1)

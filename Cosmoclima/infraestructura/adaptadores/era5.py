"""
Adaptador ERA5 — convierte la serie climática diaria en observaciones del esquema.

QUÉ HACE
--------
Lee `datos/clima_diario_subestaciones_erA5.csv` (521.625 filas, 39 subestaciones,
1990-2026, ya descargado) y produce dos observaciones mensuales por subestación:

  · `anomalia_precipitacion`  — cuánto se apartó el mes de lo normal PARA ESE
    LUGAR. Con signo, no en valor absoluto (hallazgo H-07).
  · `evento_precipitacion_intensa` — el máximo acumulado en 2 días del mes,
    puesto en percentil contra la propia historia del punto. Es el subíndice EOP
    de MACLIMA, y con la condición de suelo es lo que arma el EAL (aluvión).

POR QUÉ 2 DÍAS Y NO EL TOTAL DEL MES
------------------------------------
Porque un camino no se corta por llover mucho en el mes: se corta por llover
mucho de golpe. En Copiapó, marzo de 2015 acumuló 109 mm — pero lo que produjo
el aluvión fueron 104 mm en 48 horas. El total mensual y el golpe son dos cosas
distintas y hay que medir las dos.

POR QUÉ EL PERCENTIL PARA EL EVENTO
-----------------------------------
En zonas áridas la desviación típica de la lluvia es casi cero, y tipificar
contra cero hace explotar el número: cualquier gota daría «peligro máximo». El
percentil contra la propia historia del punto es robusto a eso y responde la
pregunta correcta: «de todo lo que vi acá, ¿qué tan arriba está esto?».

LÍMITE DECLARADO — no borrar
----------------------------
ERA5 es reanálisis. En la ronda 17 de Cosmoclima se comprobó que en la zona de
Illapel exagera los años secos (hasta 2,4×) y fabrica meses de lluvia que no
ocurrieron. Por eso la confianza base de esta fuente es 0,70 y no más. Cualquier
cifra de riesgo por activo que salga de acá es provisional hasta contrastarla
con estaciones DMC/DGA.
"""

import csv
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

AQUI = Path(__file__).parent.parent
sys.path.insert(0, str(AQUI))
import normalizar  # noqa: E402

CSV_DIARIO = AQUI / "datos" / "clima_diario_subestaciones_erA5.csv"
CSV_PUNTOS = AQUI / "datos" / "subestaciones_puntos.csv"

ID_FUENTE = "era5_openmeteo"
# Normal climatológica estándar de la OMM. Se declara acá para que se vea.
CLIMATOLOGIA = (1991, 2020)
# Piso de desviación típica: sin él, en el desierto cualquier gota da infinito.
PISO_SIGMA_MM = 2.0

FUENTE = dict(
    id=ID_FUENTE,
    organismo="ECMWF / Copernicus (vía Open-Meteo Archive)",
    producto="Reanálisis ERA5 diario por coordenada",
    url="https://archive-api.open-meteo.com/v1/archive",
    formato="json",
    familia="AMENAZA",
    acceso="anonimo",
    acceso_verificado=1,
    condiciones_uso="Open-Meteo: uso no comercial libre con atribución; ERA5 "
                    "bajo licencia Copernicus. Verificar antes de uso operativo.",
    permite_automatizacion="si",
    granularidad="punto",
    historia_desde="1940-01-01",
    frecuencia="diaria",
    confianza_base=0.70,
    notas="Reanálisis, no medición. Cosmoclima ronda 17: exagera años secos "
          "hasta 2,4x en zona Illapel y fabrica meses de lluvia. Provisional "
          "hasta contrastar con estaciones DMC/DGA.",
)


def leer_puntos():
    """Las 39 subestaciones con su coordenada decimal ya validada."""
    with CSV_PUNTOS.open(encoding="utf-8") as fh:
        return {f["subestacion"]: f for f in csv.DictReader(fh)}


def agregar_mensual():
    """Diario → mensual, por subestación.

    Devuelve {subestacion: {(anio, mes): {...}}} con el total del mes, el máximo
    en 1 y en 2 días, y cuántos días trajeron dato (para la confianza).
    """
    acumulado = defaultdict(lambda: defaultdict(
        lambda: {"total": 0.0, "max_1d": 0.0, "max_2d": 0.0, "dias": 0}))
    # el máximo en 2 días necesita recordar el día anterior de la misma serie
    anterior = {}

    with CSV_DIARIO.open(encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            se = fila["subestacion"]
            crudo = fila["precip_mm"]
            if crudo in ("", "None"):
                anterior[se] = None
                continue
            mm = float(crudo)
            anio, mes, _ = fila["fecha"].split("-")
            celda = acumulado[se][(int(anio), int(mes))]
            celda["total"] += mm
            celda["dias"] += 1
            celda["max_1d"] = max(celda["max_1d"], mm)
            previo = anterior.get(se)
            if previo is not None:
                celda["max_2d"] = max(celda["max_2d"], mm + previo)
            celda["max_2d"] = max(celda["max_2d"], mm)
            anterior[se] = mm
    return acumulado


def climatologia(mensual):
    """Normal y desviación por subestación y mes calendario, sobre 1991-2020.

    Se calcula por mes calendario y no sobre todo el año porque comparar un
    julio contra el promedio anual sería comparar peras con manzanas: en Chile
    central llueve en invierno y no en verano, y esa estacionalidad se lleva
    puesta cualquier anomalía si no se descuenta.
    """
    desde, hasta = CLIMATOLOGIA
    ref = {}
    for se, meses in mensual.items():
        for m in range(1, 13):
            totales = [d["total"] for (a, mm), d in meses.items()
                       if mm == m and desde <= a <= hasta]
            eventos = [d["max_2d"] for (a, mm), d in meses.items()
                       if mm == m and desde <= a <= hasta]
            if len(totales) < 15:      # menos de 15 años no es una climatología
                continue
            media = sum(totales) / len(totales)
            var = sum((x - media) ** 2 for x in totales) / (len(totales) - 1)
            ref[(se, m)] = {
                "media": media,
                "sigma": max(var ** 0.5, PISO_SIGMA_MM),
                "n": len(totales),
                "historia_eventos": eventos,
            }
    return ref


def normal_anual(ref, subestacion):
    """La lluvia normal de un año en ese punto: suma de las 12 normales
    mensuales. Es el denominador de la razón que propuso Alexis."""
    meses = [ref[(subestacion, m)]["media"] for m in range(1, 13)
             if (subestacion, m) in ref]
    if len(meses) < 12:
        return None
    return sum(meses)


def distribuciones_nacionales(mensual, ref):
    """Las DOS distribuciones nacionales contra las que se mide todo.

    · `magnitud`: acumulados de 48 h de todo el país, en milímetros.
      Responde «¿es mucha agua en términos absolutos?».
    · `razon`: los mismos eventos divididos por la normal anual de SU lugar.
      Responde «¿cuánto de lo que ese lugar recibe en un año cayó de golpe?».

    Las dos son percentiles NACIONALES, y ahí está la diferencia con la versión
    anterior. Antes la excedencia se medía contra la historia del propio punto:
    36 valores, que topan enseguida — el percentil se saturaba en 0,97 y no
    podía distinguir «muy raro» de «sin precedentes». Contra 17.000 valores del
    país entero, el extremo tiene resolución de sobra.

    Se excluyen los meses sin lluvia: incluir los miles de ceros del norte
    correría las distribuciones y haría que 5 mm parecieran altísimos. La
    pregunta es «entre las veces que llovió, ¿esto fue mucho?».
    """
    magnitudes, razones = [], []
    for se, meses in mensual.items():
        anual = normal_anual(ref, se)
        for dato in meses.values():
            if dato["max_2d"] <= 0.5:
                continue
            magnitudes.append(dato["max_2d"])
            if anual:
                razones.append(normalizar.razon_contra_normal(
                    dato["max_2d"], anual))
    magnitudes.sort(); razones.sort()
    return magnitudes, razones


def historia_nacional(mensual):
    """La distribución de acumulados en 48 h de TODO el país, ordenada.

    Es la pieza que faltaba. Preguntarle a la historia del propio punto «¿esto
    es mucho?» sólo dice si es raro ahí; en el desierto cualquier gota lo es.
    Preguntarle al país entero dice si es mucha agua de verdad.

    Se excluyen los meses sin lluvia: incluir los miles de ceros del norte
    correría la distribución y haría que 5 mm parecieran un percentil altísimo.
    La pregunta es «entre las veces que llovió, ¿esto fue mucho?».
    """
    valores = [d["max_2d"] for meses in mensual.values() for d in meses.values()
               if d["max_2d"] > 0.5]
    valores.sort()
    return valores


def traer(desde=None, hasta=None):
    """Interfaz común de los adaptadores: devuelve una lista de observaciones.

    `desde`/`hasta` son (anio, mes) inclusive; None = todo lo disponible.
    Si el CSV no está, devuelve lista vacía y el motivo — nunca inventa datos.
    """
    if not CSV_DIARIO.exists():
        return [], f"falta el archivo {CSV_DIARIO.name}"

    puntos = leer_puntos()
    mensual = agregar_mensual()
    ref = climatologia(mensual)
    dist_magnitud, dist_razon = distribuciones_nacionales(mensual, ref)
    hoy = date.today().isoformat()

    observaciones = []
    for se, meses in mensual.items():
        punto = puntos.get(se)
        if punto is None:
            continue
        lat, lon = float(punto["lat"]), float(punto["lon"])

        for (anio, mes), dato in sorted(meses.items()):
            if desde and (anio, mes) < desde:
                continue
            if hasta and (anio, mes) > hasta:
                continue
            clima = ref.get((se, mes))
            if clima is None:
                continue          # sin climatología no hay anomalía posible

            dias_del_mes = 31 if mes in (1, 3, 5, 7, 8, 10, 12) else (
                30 if mes != 2 else 28)
            cobertura = min(1.0, dato["dias"] / dias_del_mes)
            conf = normalizar.confianza(FUENTE["confianza_base"], cobertura)

            ultimo = dias_del_mes
            if mes == 2 and (anio % 4 == 0 and (anio % 100 != 0 or anio % 400 == 0)):
                ultimo = 29
            vig_ini = f"{anio:04d}-{mes:02d}-01"
            vig_fin = f"{anio:04d}-{mes:02d}-{ultimo:02d}"

            comun = dict(
                id_fuente=ID_FUENTE, familia="AMENAZA",
                vigencia_inicio=vig_ini, vigencia_fin=vig_fin,
                territorio_tipo="punto", territorio_id=se, lat=lat, lon=lon,
                confianza=conf, fecha_descarga=hoy, url_exacta=FUENTE["url"],
                ruta_crudo=str(CSV_DIARIO.relative_to(AQUI)),
            )

            # 1 · anomalía del total mensual, CON SIGNO (H-07)
            valor, metodo = normalizar.normalizar(
                dato["total"], "anomalia",
                referencia=clima["media"], desviacion=clima["sigma"])
            observaciones.append(dict(
                comun, variable="anomalia_precipitacion",
                valor_original=f"{dato['total']:.2f}", unidad_original="mm/mes",
                valor_normalizado=valor, metodo_normalizacion=metodo,
                notas=f"normal {clima['media']:.1f} mm, sigma {clima['sigma']:.1f}, "
                      f"n={clima['n']} años"))

            # 2 · golpe de lluvia: máximo en 2 días, por percentil (EOP)
            historia = clima["historia_eventos"]
            if len(historia) >= 10:
                valor, metodo = normalizar.normalizar(
                    dato["max_2d"], "percentil", muestra=historia)
                observaciones.append(dict(
                    comun, variable="evento_precipitacion_intensa",
                    valor_original=f"{dato['max_2d']:.2f}",
                    unidad_original="mm/48h",
                    valor_normalizado=valor, metodo_normalizacion=metodo,
                    notas=f"máx 1 día {dato['max_1d']:.1f} mm; percentil contra "
                          f"{len(historia)} años del mismo mes y lugar"))

            # 3 ★ PELIGRO — conjunción de las dos condiciones.
            # Versión del 16-ago-2026 con la corrección de Alexis: la excedencia
            # se mide como RAZÓN contra la lluvia normal del lugar, no como
            # percentil de su propia historia. Ver CORRECCION_RAREZA_PELIGRO.md.
            anual = normal_anual(ref, se)
            mag = normalizar.percentil_en(dato["max_2d"], dist_magnitud)
            exc = None
            razon = None
            if anual:
                razon = normalizar.razon_contra_normal(dato["max_2d"], anual)
                exc = normalizar.percentil_en(razon, dist_razon)
            if mag is not None and exc is not None:
                valor = normalizar.peligro(mag, exc)
                observaciones.append(dict(
                    comun, variable="peligro_precipitacion",
                    valor_original=f"{dato['max_2d']:.2f}",
                    unidad_original="mm/48h",
                    valor_normalizado=valor,
                    metodo_normalizacion="conjuncion_geometrica("
                                         "magnitud_nacional,razon_vs_normal_local)",
                    notas=f"{dato['max_2d']:.1f} mm en 48 h · magnitud nac "
                          f"{mag:.3f} · razón {razon:.2f} veces la normal anual "
                          f"({anual:.0f} mm) → percentil nac {exc:.3f}"))

    return observaciones, None


if __name__ == "__main__":
    obs, problema = traer()
    if problema:
        print("SIN DATO:", problema)
        raise SystemExit(1)
    print(f"{len(obs):,} observaciones generadas de {ID_FUENTE}")
    variables = defaultdict(int)
    for o in obs:
        variables[o["variable"]] += 1
    for v, n in variables.items():
        print(f"   {v}: {n:,}")

"""
Adaptador SEC — «Interrupciones en Línea»: clientes sin luz, por comuna y por hora.

QUÉ ES
------
La Superintendencia de Electricidad y Combustibles (SEC) publica, sin registro y
sin clave, cuántos clientes están sin suministro eléctrico en cada comuna del
país, hora por hora, sumando TODAS las distribuidoras. Es la capa
ESTADO-CONSECUENCIA del proyecto: no dice «puede fallar», dice «falló, acá, a
esta hora, y a tanta gente».

Endpoints usados (POST, `Content-Type: application/json`), bajo
`https://apps.sec.cl/INTONLINEv1/`:

| Endpoint                            | Cuerpo                                   | Devuelve                                                              |
|-------------------------------------|------------------------------------------|-----------------------------------------------------------------------|
| `ClientesAfectados/GetPorFecha`     | `{"anho":2025,"mes":7,"dia":10,"hora":3}` | `[{NOMBRE_REGION, NOMBRE_COMUNA, CLIENTES_AFECTADOS}]` en ESA hora     |
| `ClientesAfectados/Get`             | `{}`                                      | Serie horaria NACIONAL de los últimos ~7 días (contexto, no muestra)   |
| `ClientesAfectados/GetClientesNacional` | `{}`                                  | Total nacional de clientes — el denominador                            |
| `ClientesAfectados/GetHoraServer`   | `{}`                                      | Hora del servidor, para saber en qué reloj están las fechas            |

PARA QUÉ SE ESCRIBIÓ ESTE ADAPTADOR
-----------------------------------
Para poner a prueba una hipótesis concreta del director:

    «En pleno invierno, por el aumento de demanda eléctrica (calefacción), un
     transformador puede recalentarse e incendiarse (están llenos de aceite) y
     cortar el suministro.»

O sea: el invierno atacaría por DOS vías a la vez — el temporal daña la red
desde afuera, y la calefacción la estresa desde adentro. Si eso es cierto, los
cortes de invierno tienen que tener una FIRMA distinta de los de verano. Este
adaptador baja el dato con el que esa firma se puede medir.

LO QUE ESTE DATO **NO** ES — leer antes de usarlo
-------------------------------------------------
Esto es lo más importante del módulo, porque de acá salen todas las
limitaciones del análisis:

1. **No es un registro de eventos. Es una FOTO por hora.** La SEC no publica
   «corte tal, empezó a las 19:12, duró 3 h». Publica «en la comuna X, a las
   19:00, había N clientes sin luz». El concepto de «evento» que usa este
   módulo (ver `detectar_eventos`) es una CONSTRUCCIÓN NUESTRA sobre esas
   fotos, no un dato de la SEC.
2. **No dice POR QUÉ se cortó.** Un transformador quemado y un camión contra un
   poste se ven exactamente igual. Por eso la hipótesis no se puede confirmar
   directamente: sólo se pueden buscar sus HUELLAS (hora del día, tamaño,
   duración) y ver si aparecen.
3. **No dice QUÉ activo falló.** Da la consecuencia (clientes), no la causa
   (transformador, alimentador, línea).
4. **Cuenta clientes, no personas.** Un «cliente» es un empalme: una casa, un
   hospital y una minera cuentan 1 cada uno.
5. **La granularidad es la hora en punto.** Un corte de 40 minutos entre dos
   horas en punto puede no existir para esta fuente.

PRIVACIDAD — regla dura del proyecto
------------------------------------
Los mapas de las distribuidoras permiten consultar «por número de cliente».
Eso NO se toca. Este adaptador usa exclusivamente el agregado por comuna que
publica la SEC, que no identifica a nadie. No hay ningún campo de persona en
todo el flujo.

CONDICIONES DE USO Y RITMO
--------------------------
- `https://apps.sec.cl/robots.txt` → **404, no existe**: el dominio no declara
  ninguna restricción a robots (verificado 16-ago-2026).
- `https://www.sec.cl/robots.txt` → `User-agent: * / Allow: /`, sólo bloquea
  `/sitio-web/wp-admin/` (verificado 16-ago-2026). Nuestro endpoint no cae ahí.
- La SEC es órgano público: rige la Ley 20.285 de Transparencia.
- **Ritmo autoimpuesto: 1 petición por segundo, secuencial, un solo hilo.** El
  servidor hoy responde en ~0,15 s, pero la ficha del proyecto lo registró
  lento (30-100 s) en otras ocasiones: puede degradarse y no queremos ser la
  causa. Hay reintentos con espera creciente.
- **No se barre el calendario.** La regla del proyecto (ficha A1, punto 10) es
  no pedir año por año hora por hora — serían decenas de miles de peticiones.
  Este adaptador pide una MUESTRA de bloques, definida de antemano y con
  semilla fija (ver `bloques_de_la_muestra`).

REGLA DEL CRUDO
---------------
Toda respuesta se guarda tal como llegó, byte a byte, en
`datos/crudo/sec/<fecha-de-captura>/`, ANTES de procesar nada. Si mañana
descubrimos que el procesamiento estaba mal, el crudo sigue ahí y se reprocesa
sin volver a molestar al servidor.

USO
---
    python adaptadores/sec_cortes.py bajar      # baja la muestra y guarda el crudo
    python adaptadores/sec_cortes.py procesar   # crudo -> datos/sec_cortes.csv
    python adaptadores/sec_cortes.py analizar   # prueba de hipótesis (criterio fijo)
"""

import csv
import json
import random
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

AQUI = Path(__file__).parent.parent
BASE = "https://apps.sec.cl/INTONLINEv1/"

# Fecha de captura: nombra la carpeta del crudo. Se fija una vez, al importar,
# para que toda una corrida caiga en la misma carpeta aunque cruce medianoche.
FECHA_CAPTURA = date.today().isoformat()
CRUDO = AQUI / "datos" / "crudo" / "sec" / FECHA_CAPTURA
CSV_SALIDA = AQUI / "datos" / "sec_cortes.csv"

# Cabeceras de un navegador normal. No es evasión: es identificarse igual que
# la propia página de la SEC, que es quien llama a estos endpoints.
CABECERAS = {
    "Content-Type": "application/json",
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
    "Referer": BASE + "index.aspx",
    "Accept": "application/json, text/javascript, */*; q=0.01",
}

PAUSA_S = 1.0          # ritmo prudente entre peticiones
REINTENTOS = 4         # ante error de red o 5xx
ESPERA_REINTENTO = 5.0 # segundos, se multiplica por el número de intento


# ---------------------------------------------------------------------------
# 1. CAPA DE RED — pedir y guardar el crudo
# ---------------------------------------------------------------------------

def pedir(endpoint, cuerpo, timeout=120):
    """
    Hace un POST al endpoint y devuelve (texto_crudo, dict_de_diagnostico).

    Devuelve el TEXTO tal cual, sin parsear: el crudo se guarda antes de
    interpretarlo. Si tras los reintentos no hay respuesta, devuelve
    (None, diagnóstico con el error) — nunca inventa un valor ni devuelve una
    lista vacía haciéndola pasar por dato. «Sin dato» se declara.
    """
    datos = json.dumps(cuerpo).encode("utf-8")
    for intento in range(1, REINTENTOS + 1):
        t0 = time.time()
        try:
            req = urllib.request.Request(BASE + endpoint, data=datos, headers=CABECERAS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                texto = r.read().decode("utf-8")
            return texto, {"http": 200, "bytes": len(texto),
                           "segundos": round(time.time() - t0, 3),
                           "intentos": intento, "error": ""}
        except urllib.error.HTTPError as e:
            diag = {"http": e.code, "bytes": 0, "segundos": round(time.time() - t0, 3),
                    "intentos": intento, "error": f"HTTPError {e.code}"}
            # 4xx no se reintenta: la petición está mal, insistir no la arregla.
            if e.code < 500:
                return None, diag
        except Exception as e:
            diag = {"http": 0, "bytes": 0, "segundos": round(time.time() - t0, 3),
                    "intentos": intento, "error": f"{type(e).__name__}: {e}"}
        if intento < REINTENTOS:
            time.sleep(ESPERA_REINTENTO * intento)
    return None, diag


def guardar_crudo(ruta_relativa, texto):
    """Escribe la respuesta tal como llegó. El crudo es sagrado: no se toca."""
    destino = CRUDO / ruta_relativa
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(texto, encoding="utf-8")


# ---------------------------------------------------------------------------
# 2. DISEÑO DE LA MUESTRA — fijado ANTES de mirar ningún resultado
# ---------------------------------------------------------------------------
#
# El problema: para medir DURACIÓN y HORA DE INICIO de un corte hacen falta
# horas CONSECUTIVAS. Una foto suelta no sirve. Pero barrer el calendario
# entero son ~58.000 peticiones y eso es descarga masiva sobre un servicio
# público.
#
# La solución: BLOQUES de 3 días completos (72 horas seguidas). Dentro de un
# bloque se ve nacer, crecer y morir un corte. Se sortean bloques al azar —no
# se eligen los días de temporal— para no meter el pulgar en la balanza: si
# sólo mirásemos temporales, el invierno saldría peor por construcción.
#
# Estratos: INVIERNO austral (jun-jul-ago) y VERANO austral (dic-ene-feb).
# 2 bloques por estación y por año, para que ningún año pese de más.
#
# La semilla está fija: cualquiera reproduce exactamente la misma muestra.

SEMILLA = 20260816
BLOQUES_POR_ESTACION_Y_ANHO = 2
HORAS_POR_BLOQUE = 72  # 3 días

# Historia verificada el 16-ago-2026 con sondeos día 15 de cada trimestre:
#   2018-01/04/07 responden; 2018-10 a 2019-07 devuelven vacío (HUECO);
#   desde 2019-10 en adelante responde todo. Por eso la muestra arranca en el
#   verano 2019-20 y no antes: se usa sólo el tramo continuo y verificado.
ANHOS_INVIERNO = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
# Un «verano» lleva el año de su enero: verano 2020 = dic-2019 + ene/feb-2020.
ANHOS_VERANO = [2020, 2021, 2022, 2023, 2024, 2025, 2026]

# Hasta dónde llega la historia disponible (hoy). Ningún bloque puede pasar de acá.
ULTIMO_DIA = date(2026, 8, 13)


def _dias_candidatos(anho, estacion):
    """
    Días en que puede EMPEZAR un bloque, de modo que las 72 h caigan enteras
    dentro de la estación. Así ningún bloque mezcla invierno con primavera.
    """
    if estacion == "invierno":
        primero, ultimo = date(anho, 6, 1), date(anho, 8, 31) - timedelta(days=2)
    else:  # verano austral: 1-dic del año anterior .. fin de febrero
        primero, ultimo = date(anho - 1, 12, 1), date(anho, 2, 28) - timedelta(days=2)
    ultimo = min(ultimo, ULTIMO_DIA - timedelta(days=2))
    dias, d = [], primero
    while d <= ultimo:
        dias.append(d)
        d += timedelta(days=1)
    return dias


def bloques_de_la_muestra():
    """
    Devuelve la lista de bloques sorteados: [(fecha_inicio, estacion), ...].
    Determinista: misma semilla, misma muestra, siempre.
    """
    rng = random.Random(SEMILLA)
    bloques = []
    for estacion, anhos in (("invierno", ANHOS_INVIERNO), ("verano", ANHOS_VERANO)):
        for anho in anhos:
            candidatos = _dias_candidatos(anho, estacion)
            if not candidatos:
                continue
            # sample sin reemplazo y separando bloques para que no se solapen
            elegidos = []
            for _ in range(BLOQUES_POR_ESTACION_Y_ANHO):
                libres = [d for d in candidatos
                          if all(abs((d - e).days) >= 3 for e in elegidos)]
                if not libres:
                    break
                elegidos.append(rng.choice(libres))
            for d in sorted(elegidos):
                bloques.append((d, estacion))
    return bloques


# ---------------------------------------------------------------------------
# 3. BAJADA
# ---------------------------------------------------------------------------

def bajar():
    """
    Baja la muestra completa hora por hora y guarda TODO el crudo.
    Además guarda los tres endpoints de contexto (serie nacional, denominador
    de clientes y hora del servidor), que sirven para interpretar el resto.
    """
    CRUDO.mkdir(parents=True, exist_ok=True)
    bloques = bloques_de_la_muestra()
    total_horas = len(bloques) * HORAS_POR_BLOQUE
    print(f"Muestra: {len(bloques)} bloques de {HORAS_POR_BLOQUE} h = "
          f"{total_horas} peticiones (~{total_horas * (PAUSA_S + 0.2) / 60:.0f} min)")

    # --- contexto: se pide una vez y queda registrado con la captura ---
    for endpoint, nombre in (("ClientesAfectados/Get", "contexto_serie_nacional.json"),
                             ("ClientesAfectados/GetClientesNacional", "contexto_clientes_nacional.json"),
                             ("ClientesAfectados/GetHoraServer", "contexto_hora_servidor.json")):
        texto, diag = pedir(endpoint, {})
        if texto is not None:
            guardar_crudo(nombre, texto)
        print(f"  contexto {nombre}: {diag}")
        time.sleep(PAUSA_S)

    # --- la muestra ---
    bitacora = []
    hechas = 0
    for inicio, estacion in bloques:
        for desplazamiento in range(HORAS_POR_BLOQUE):
            t = datetime(inicio.year, inicio.month, inicio.day) + timedelta(hours=desplazamiento)
            rel = f"porfecha/{t.year:04d}-{t.month:02d}-{t.day:02d}/h{t.hour:02d}.json"
            destino = CRUDO / rel
            if destino.exists():          # reanudable: no se vuelve a pedir lo que ya está
                hechas += 1
                continue
            cuerpo = {"anho": t.year, "mes": t.month, "dia": t.day, "hora": t.hour}
            texto, diag = pedir("ClientesAfectados/GetPorFecha", cuerpo)
            if texto is not None:
                guardar_crudo(rel, texto)
            bitacora.append({"archivo": rel, "estacion": estacion,
                             "bloque": inicio.isoformat(), **diag})
            hechas += 1
            if hechas % 100 == 0:
                print(f"  {hechas}/{total_horas} ...", flush=True)
            time.sleep(PAUSA_S)

    # La bitácora deja constancia de qué se pidió, qué respondió y cuánto tardó.
    # Es parte del crudo: sin ella no se puede distinguir «no hubo cortes» de
    # «la petición falló».
    if bitacora:
        with open(CRUDO / "bitacora_peticiones.csv", "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(bitacora[0].keys()))
            if f.tell() == 0:
                w.writeheader()
            w.writerows(bitacora)
    fallidas = [b for b in bitacora if b["http"] != 200]
    print(f"Listo. {hechas} horas en {CRUDO}. Peticiones fallidas: {len(fallidas)}")


# ---------------------------------------------------------------------------
# 4. PROCESO — del crudo a la tabla
# ---------------------------------------------------------------------------

def procesar():
    """
    Convierte el crudo en `datos/sec_cortes.csv`: un panel COMUNA × HORA.

    Se deja a propósito en su forma más cruda-pero-tabular (una fila por comuna
    y hora observada, con los clientes afectados). Así cualquiera puede volver a
    derivar «eventos» con otro umbral sin tener que bajar nada de nuevo. La
    construcción de eventos vive aparte, en `detectar_eventos`.
    """
    raiz = AQUI / "datos" / "crudo" / "sec"
    # Se lee la captura más reciente que exista, no sólo la de hoy.
    capturas = sorted(p for p in raiz.glob("*/porfecha") if p.is_dir())
    if not capturas:
        print("No hay crudo. Correr primero: python adaptadores/sec_cortes.py bajar")
        return
    origen = capturas[-1]
    print(f"Procesando crudo de {origen.parent.name}")

    # Qué bloque y qué estación corresponde a cada día, según el diseño de muestra.
    estacion_de = {}
    for inicio, estacion in bloques_de_la_muestra():
        for k in range(3):
            estacion_de[(inicio + timedelta(days=k)).isoformat()] = (estacion, inicio.isoformat())

    filas = []
    horas_vacias = 0
    for dia_dir in sorted(origen.iterdir()):
        if not dia_dir.is_dir():
            continue
        dia = dia_dir.name
        estacion, bloque = estacion_de.get(dia, ("", ""))
        for archivo in sorted(dia_dir.glob("h*.json")):
            hora = int(archivo.stem[1:])
            try:
                registros = json.loads(archivo.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # El crudo llegó roto: se declara y se salta. No se rellena.
                print(f"  AVISO: JSON ilegible en {archivo}")
                continue
            if not registros:
                horas_vacias += 1
            for r in registros:
                filas.append({
                    "fecha": dia,
                    "hora": hora,
                    "anho": int(dia[:4]),
                    "mes": int(dia[5:7]),
                    "region": r.get("NOMBRE_REGION", ""),
                    "comuna": r.get("NOMBRE_COMUNA", ""),
                    "clientes_afectados": int(r.get("CLIENTES_AFECTADOS", 0)),
                    "estacion": estacion,
                    "bloque": bloque,
                })

    filas.sort(key=lambda x: (x["fecha"], x["hora"], x["region"], x["comuna"]))
    CSV_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_SALIDA, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    dias = len({x["fecha"] for x in filas})
    comunas = len({x["comuna"] for x in filas})
    print(f"{CSV_SALIDA}: {len(filas)} filas · {dias} días · {comunas} comunas "
          f"· {horas_vacias} horas sin ninguna comuna con corte")


# ---------------------------------------------------------------------------
# 5. CONSTRUCCIÓN DE «EVENTOS» — y por qué así
# ---------------------------------------------------------------------------
#
# La SEC no publica eventos, publica fotos por hora. Un «evento» acá es:
#
#     una racha máxima de horas CONSECUTIVAS en las que una MISMA comuna
#     tuvo al menos UMBRAL_CLIENTES clientes sin luz.
#
# UMBRAL_CLIENTES = 100, fijado ANTES de mirar resultados, y con razón física:
# un transformador de distribución en Chile alimenta del orden de 50-300
# clientes. Cien clientes es, a grosso modo, «al menos un transformador». Sirve
# además para lo que más ensucia esta serie: las comunas grandes casi nunca
# bajan a cero (siempre hay algún empalme cortado), y sin umbral toda la
# Región Metropolitana daría un único «corte» eterno.
#
# CENSURA: si una racha toca la primera o la última hora del bloque, no se le
# vio el principio o el final. Esos eventos se DESCARTAN de todo el análisis y
# se informa cuántos fueron. Es preferible perder casos a inventarles duración.

UMBRAL_CLIENTES = 100


def detectar_eventos(filas):
    """
    De panel comuna×hora a eventos. Devuelve lista de dicts con:
      comuna, region, estacion, hora_inicio (0-23), duracion_h, pico_clientes.
    """
    # serie por comuna dentro de cada bloque, indexada por hora absoluta
    series = defaultdict(dict)
    for f in filas:
        t = datetime.strptime(f["fecha"], "%Y-%m-%d") + timedelta(hours=f["hora"])
        series[(f["bloque"], f["estacion"], f["region"], f["comuna"])][t] = f["clientes_afectados"]

    eventos, censurados = [], 0
    for (bloque, estacion, region, comuna), serie in series.items():
        if not bloque:
            continue
        t0 = datetime.strptime(bloque, "%Y-%m-%d")
        horas = [t0 + timedelta(hours=k) for k in range(HORAS_POR_BLOQUE)]
        # Hora sin registro de la comuna = esa comuna no aparecía = 0 clientes.
        valores = [serie.get(h, 0) for h in horas]

        k = 0
        while k < len(valores):
            if valores[k] < UMBRAL_CLIENTES:
                k += 1
                continue
            j = k
            while j < len(valores) and valores[j] >= UMBRAL_CLIENTES:
                j += 1
            if k == 0 or j == len(valores):
                censurados += 1        # no se vio nacer o no se vio morir
            else:
                eventos.append({
                    "region": region, "comuna": comuna, "estacion": estacion,
                    "bloque": bloque,
                    "fecha_inicio": horas[k].date().isoformat(),
                    "hora_inicio": horas[k].hour,
                    "duracion_h": j - k,
                    "pico_clientes": max(valores[k:j]),
                })
            k = j
    return eventos, censurados


# ---------------------------------------------------------------------------
# 6. ANÁLISIS — el criterio está fijado en PRUEBA_SEC_INVIERNO.md
# ---------------------------------------------------------------------------

def analizar():
    """
    Corre las tres preguntas con el criterio pre-registrado. No decide nada por
    su cuenta: imprime los números y las pruebas; el veredicto se escribe en el
    informe.
    """
    import numpy as np
    from scipy import stats

    with open(CSV_SALIDA, encoding="utf-8") as f:
        filas = [{**r, "hora": int(r["hora"]),
                  "clientes_afectados": int(r["clientes_afectados"])}
                 for r in csv.DictReader(f)]

    eventos, censurados = detectar_eventos(filas)
    inv = [e for e in eventos if e["estacion"] == "invierno"]
    ver = [e for e in eventos if e["estacion"] == "verano"]
    print(f"\nEventos (umbral {UMBRAL_CLIENTES} clientes): "
          f"{len(eventos)} usables · {censurados} descartados por censura")
    print(f"  invierno {len(inv)} · verano {len(ver)}\n")

    def resumen(nombre, a, b, clave):
        x = np.array([e[clave] for e in a], float)
        y = np.array([e[clave] for e in b], float)
        u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        print(f"--- {nombre} ({clave}) ---")
        print(f"  invierno: n={len(x)} mediana={np.median(x):.1f} "
              f"media={x.mean():.1f} p90={np.percentile(x,90):.0f} max={x.max():.0f}")
        print(f"  verano  : n={len(y)} mediana={np.median(y):.1f} "
              f"media={y.mean():.1f} p90={np.percentile(y,90):.0f} max={y.max():.0f}")
        print(f"  razón de medianas inv/ver = {np.median(x)/max(np.median(y),1e-9):.3f}")
        print(f"  Mann-Whitney U={u:.0f}  p={p:.3g}\n")
        return p, np.median(x) / max(np.median(y), 1e-9)

    # P1 — magnitud
    resumen("P1 MAGNITUD", inv, ver, "pico_clientes")
    # P3 — duración
    resumen("P3 DURACION", inv, ver, "duracion_h")

    # P2 — hora del día. Ventana de demanda vespertina 18:00-21:59 (pre-registrada).
    print("--- P2 HORA DE INICIO ---")
    print("  hora  invierno  verano   (porcentaje de eventos de su estación)")
    for h in range(24):
        ci = sum(1 for e in inv if e["hora_inicio"] == h)
        cv = sum(1 for e in ver if e["hora_inicio"] == h)
        marca = " <-- ventana vespertina" if 18 <= h <= 21 else ""
        print(f"  {h:02d}   {ci:5d} {100*ci/max(len(inv),1):5.1f}%  "
              f"{cv:5d} {100*cv/max(len(ver),1):5.1f}%{marca}")
    vi = sum(1 for e in inv if 18 <= e["hora_inicio"] <= 21)
    vv = sum(1 for e in ver if 18 <= e["hora_inicio"] <= 21)
    tabla = np.array([[vi, len(inv) - vi], [vv, len(ver) - vv]])
    chi2, p2, _, _ = stats.chi2_contingency(tabla)
    print(f"\n  ventana 18-21h: invierno {vi}/{len(inv)} = {100*vi/len(inv):.2f}%  |  "
          f"verano {vv}/{len(ver)} = {100*vv/len(ver):.2f}%")
    print(f"  chi2={chi2:.3f}  p={p2:.3g}")

    # Perfil horario en volumen (no en conteo de eventos): útil para ver la curva
    # de demanda. Se informa como contexto, no como prueba.
    print("\n--- contexto: clientes sin luz promedio por hora del día ---")
    print("  hora  invierno   verano")
    for h in range(24):
        a = [f["clientes_afectados"] for f in filas
             if f["estacion"] == "invierno" and f["hora"] == h]
        b = [f["clientes_afectados"] for f in filas
             if f["estacion"] == "verano" and f["hora"] == h]
        print(f"  {h:02d}  {np.sum(a)/max(len({f['fecha'] for f in filas if f['estacion']=='invierno'}),1):9.0f} "
              f"{np.sum(b)/max(len({f['fecha'] for f in filas if f['estacion']=='verano'}),1):9.0f}")


if __name__ == "__main__":
    orden = sys.argv[1] if len(sys.argv) > 1 else "muestra"
    if orden == "bajar":
        bajar()
    elif orden == "procesar":
        procesar()
    elif orden == "analizar":
        analizar()
    elif orden == "muestra":
        for d, e in bloques_de_la_muestra():
            print(f"{e:9s} {d} .. {d + timedelta(days=2)}")
    else:
        print(__doc__)

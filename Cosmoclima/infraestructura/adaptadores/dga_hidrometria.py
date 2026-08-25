"""
ADAPTADOR DGA — la red hidrométrica nacional, o «la verdad instrumental del río»
================================================================================

POR QUÉ ESTE ADAPTADOR EXISTE
------------------------------
El proyecto tiene un catálogo confiable de REMOCIONES en masa (ReTeRM, 376
eventos evaluados en terreno por SERNAGEOMIN) y **ningún catálogo confiable de
DESBORDES fluviales**:

- **SENAPRED** no distingue aluvión de inundación: clasifica el aluvión de
  Copiapó del 24-mar-2015 —el caso ancla del proyecto— como «Inundación».
- **DesInventar** (prensa sistematizada) *nombra* bien pero *mide* mal: dice
  «hubo un aluvión» y no dice cuánta agua bajó.

Sin una verdad instrumental del desborde, la hipótesis central del proyecto —que
el terreno ENCAMINA la misma meteorología hacia remoción o hacia desborde— no se
puede probar contra dato: falta la mitad de la evidencia.

**La DGA es esa verdad instrumental.** Un caudal que sube 14 veces sobre su
línea base está medido por un instrumento en el cauce, no reportado por un
testigo. No opina sobre la categoría del evento: mide el agua.

QUÉ HAY, Y DE DÓNDE SALE CADA COSA (verificado el 17-ago-2026)
---------------------------------------------------------------
Tres fuentes distintas, que este adaptador cruza por código de estación:

1. **Catálogo oficial de estaciones activas** — `Informe-Nacional-*.xlsx`
   publicado por la propia DGA en `dga.mop.gob.cl/estadisticas-estaciones-dga/`.
   3.810 estaciones activas con código, nombre, fecha de instalación, región,
   provincia, comuna, cuenca / subcuenca / subsubcuenca, UTM, lat-lon en grados
   sexagesimales, altitud y tipo. Actualizado al 15-ene-2026.

2. **Red en línea (SAT)** — el mapa de tiempo real de la DGA
   (`snia.mop.gob.cl/sat/site/informes/mapas/mapas.xhtml`) trae embebido un JSON
   con **1.838 marcadores de estaciones que transmiten hoy** (1.811 códigos
   únicos: hay 27 repetidos), con lat-lon DECIMAL, altitud,
   tipo, y —cuando la DGA se lo definió— `umbral` y `nivelAlerta`, que son *los
   umbrales de crecida de la propia DGA*. Es el único lugar donde aparecen.

3. **Series diarias de caudal** — compilación del **(CR)2** (Centro de Ciencia
   del Clima y la Resiliencia, U. de Chile), `cr2_qflxDaily_2020.zip`:
   **811 estaciones, caudal medio diario, feb-1913 → mar-2020**, publicada bajo
   **Creative Commons CC0**. El propio (CR)2 declara que sus datos los bajó de
   `snia.mop.gob.cl/BNAConsultas/reportes` y de `dgasatel.mop.cl`: **es dato
   DGA**, sólo que ya consolidado en una matriz manejable en vez de repartido en
   consultas de 4 años y 10 estaciones a la vez.

★ POR QUÉ NO BAJAMOS LAS SERIES DEL PORTAL OFICIAL DE LA DGA
-------------------------------------------------------------
Porque **la DGA declaró explícitamente que no quiere que se automatice**: el
botón «Buscar» de `snia.mop.gob.cl/BNAConsultas/reportes` está detrás de un
**reCAPTCHA v2** (sitekey `6LcRYtwqAAAAACZzq5MG06b68xaZpAhVHOJaLCFt`, función
`desactivarbotonbusqueda()`), que es la forma estándar de decir «esto se usa a
mano». No se intentó eludirlo, ni se intentará. Los topes declarados por la DGA
(4 años diarios · 10 años mensuales · 40 años anuales · máx. 10 estaciones por
consulta) apuntan en la misma dirección: el portal está pensado para consultas
humanas puntuales.

La vía formal para pedir la serie completa existe y está escrita en el propio
portal: solicitud por **Ley 20.285 de transparencia** vía Plataforma SIAC. Eso
es un trámite de Alexis, no del código.

Mientras tanto, el (CR)2 publica el mismo dato bajo CC0 para investigación y
docencia. Es lo que usamos, y se cita a las dos: DGA como productor del dato,
(CR)2 como compilador.

LO QUE ESTE DATO SÍ PRUEBA, Y LO QUE NO
----------------------------------------
Probado con el caso ancla (`--copiapo`), 24-25 de marzo de 2015:

    Río Copiapó en Pastillo   1,42 → 4,91 → **19,7** m³/s   (14× la base)
    Río Pulido en Vertedero   0,95 → 2,48 → **12,2** m³/s   (13× la base)
    Río Copiapó en La Puerta  0,57 → **3,50** m³/s y la estación se apaga
    Mal Paso Aguas Abajo      el registro TERMINA el 24-03-2015

El instrumento vio el evento. **Sí sirve.**

Pero hay tres límites que hay que declarar cada vez que se use este dato:

1. **Caudal es MAGNITUD, no MECANISMO.** Copiapó 2015 fue un flujo de detritos,
   no un desborde fluvial, y el caudalímetro registra los dos igual de bien. La
   serie da el «cuánto» que al proyecto le faltaba (ver
   `CORRECCION_RAREZA_PELIGRO.md`: «la variable mide RAREZA y no PELIGRO»); el
   «cuál de las dos amenazas» lo sigue decidiendo el enrutador de terreno.

2. **Caudal alto ≠ desborde.** Para afirmar «se salió del cauce» hace falta la
   capacidad a cauce lleno de esa sección, que la DGA no publica en esta serie.
   Lo que sí se puede afirmar sin inventar nada es *«caudal excepcional respecto
   de la propia historia de esta estación»*, que es lo que calcula `crecidas()`.

3. ★ **CENSURA POR ARRIBA: los eventos más grandes rompen el instrumento.** En
   Copiapó, tres de cuatro estaciones dejan de transmitir el día siguiente al
   peak. El sesgo va justo en contra de lo que buscamos: **cuanto peor el
   evento, más probable que el dato falte**. Por eso `crecidas()` marca el
   `apagon_posterior` como una señal, no como un hueco: una estación que se
   calla después de una crecida está diciendo algo.

CONDICIONES DE USO (verificadas el 17-ago-2026, ver FUENTE_DGA.md)
-------------------------------------------------------------------
- `dga.mop.gob.cl/robots.txt`: sólo bloquea `/wp-admin/`. El xlsx es libre.
- MOP «Condiciones de Uso»: autoriza la reproducción **sin fines comerciales**,
  citando claramente la procedencia. Este proyecto es académico → cubierto.
- (CR)2: **CC0**, «para fines de investigación y docencia».
- `snia.mop.gob.cl/BNAConsultas`: reCAPTCHA → **no se automatiza**.

REGLA DE LA CASA: el crudo se guarda tal como llegó, en
`datos/crudo/dga/<fecha>/`, y este adaptador lee de ahí. Nunca vuelve a la red
salvo que se le pida explícitamente.
"""

import csv
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

AQUI = Path(__file__).resolve().parent.parent

FECHA_CAPTURA = "2026-08-17"
CRUDO = AQUI / "datos" / "crudo" / "dga" / FECHA_CAPTURA
SALIDA = AQUI / "datos" / "dga_estaciones.csv"

XLSX_OFICIAL = CRUDO / "Informe-Nacional-09012026_124944.xlsx"
JSON_EN_LINEA = CRUDO / "snia_sat_red_en_linea.json"
ZIP_SERIES = CRUDO / "cr2_qflxDaily_2020.zip"
DENTRO_SERIES = "cr2_qflxDaily_2020/cr2_qflxDaily_2020.txt"
DENTRO_ESTACIONES = "cr2_qflxDaily_2020/cr2_qflxDaily_2020_stations.txt"

SIN_DATO = -9999.0          # el centinela que usa el (CR)2

ID_FUENTE = "dga_hidrometria"

FUENTE = dict(
    id=ID_FUENTE,
    nombre="Red Hidrométrica Nacional · Dirección General de Aguas (MOP)",
    naturaleza="medición instrumental en cauce",
    variable_principal="caudal medio diario (m³/s)",
    periodo_series="1913-02 → 2020-03",
    urls=dict(
        catalogo="https://dga.mop.gob.cl/estadisticas-estaciones-dga/",
        en_linea="https://snia.mop.gob.cl/sat/site/informes/mapas/mapas.xhtml",
        portal_oficial_series="https://snia.mop.gob.cl/BNAConsultas/reportes",
        series_cr2="https://www.cr2.cl/datos-de-caudales/",
    ),
    # Confianza ALTA: es instrumento, no testimonio. Es la fuente de mayor
    # confianza que tiene el proyecto para magnitud hidrológica. Se le descuenta
    # un poco por la censura por arriba (el instrumento se rompe justo en los
    # eventos extremos) y por pasar por un compilador intermedio.
    confianza_base=0.92,
    limites=(
        "mide magnitud, no mecanismo · caudal alto no prueba desborde sin "
        "capacidad de cauce · los eventos extremos apagan la estación"
    ),
)


# =============================================================================
# 1 · UTILIDADES DE NORMALIZACIÓN
# =============================================================================

def codigo_normal(bruto):
    """Todos los códigos DGA al mismo formato: 8 dígitos con ceros a la izquierda.

    Las tres fuentes escriben el mismo código de tres maneras distintas:
        xlsx oficial   '01000005'   ya son 8 dígitos
        SAT en línea   '08382002-0' 8 dígitos + guion + dígito verificador
        (CR)2          '1201005'    sin el cero de la izquierda
    Sin unificarlos, el cruce da cero coincidencias.
    """
    if bruto is None:
        return ""
    txt = str(bruto).strip().split("-")[0]
    solo_digitos = re.sub(r"\D", "", txt)
    return solo_digitos.zfill(8) if solo_digitos else ""


def gms_a_decimal(texto, hemisferio_negativo=True):
    """Convierte «17°35'42''» a -17.595.

    El xlsx de la DGA trae las coordenadas en grados-minutos-segundos y **sin
    signo**: no dice S ni W. Chile continental está entero al sur y al oeste, así
    que el signo es negativo siempre. Se declara acá para que nadie lo deduzca
    después mirando el número.
    """
    if not texto:
        return None
    numeros = re.findall(r"\d+(?:[.,]\d+)?", str(texto))
    if not numeros:
        return None
    partes = [float(n.replace(",", ".")) for n in numeros[:3]]
    while len(partes) < 3:
        partes.append(0.0)
    grados, minutos, segundos = partes
    valor = grados + minutos / 60.0 + segundos / 3600.0
    return -valor if hemisferio_negativo else valor


def limpio(texto):
    """Aplasta espacios y saltos de línea. El xlsx trae '\\n' dentro de celdas."""
    if texto is None:
        return ""
    return re.sub(r"\s+", " ", str(texto)).strip()


def numero(bruto):
    try:
        return float(str(bruto).replace(",", "."))
    except (TypeError, ValueError):
        return None


# =============================================================================
# 2 · LAS TRES FUENTES, CADA UNA POR SEPARADO
# =============================================================================

def catalogo_oficial():
    """Lee el xlsx que publica la DGA: estaciones ACTIVAS al 15-ene-2026.

    La cabecera real está en la fila 17; arriba hay un encabezado institucional y
    el resumen de la consulta que generó el informe. Se busca la fila de
    cabecera por su contenido en vez de fijar el número, porque la DGA regenera
    este archivo cada tanto y el encabezado puede crecer o encoger.
    """
    if not XLSX_OFICIAL.exists():
        return {}, f"falta {XLSX_OFICIAL}"
    try:
        import openpyxl
    except ImportError:
        return {}, "falta el paquete openpyxl"

    hoja = openpyxl.load_workbook(
        XLSX_OFICIAL, read_only=True, data_only=True).worksheets[0]
    filas = list(hoja.iter_rows(values_only=True))

    cabecera_en = None
    for i, fila in enumerate(filas):
        valores = [limpio(c) for c in fila]
        if "Código" in valores and "Estación" in valores:
            cabecera_en = i
            break
    if cabecera_en is None:
        return {}, "el xlsx cambió de formato: no encuentro la fila de cabecera"

    campos = [limpio(c) for c in filas[cabecera_en]]
    salida = {}
    for fila in filas[cabecera_en + 1:]:
        registro = dict(zip(campos, fila))
        codigo = codigo_normal(registro.get("Código"))
        if not codigo:
            continue
        salida[codigo] = dict(
            nombre=limpio(registro.get("Estación")),
            fecha_instalacion=limpio(registro.get("Fecha Instalación")),
            region=limpio(registro.get("Región")),
            provincia=limpio(registro.get("Provincia")),
            comuna=limpio(registro.get("Comuna")),
            cuenca=limpio(registro.get("Cuenca")),
            subcuenca=limpio(registro.get("SubCuenca")),
            subsubcuenca=limpio(registro.get("SubSubCuenca")),
            latitud=gms_a_decimal(registro.get("Latitud")),
            longitud=gms_a_decimal(registro.get("Longitud")),
            altitud=numero(registro.get("Altitud m.s.n.m")),
            tipo=limpio(registro.get("Tipo de Estación")).replace("*", " ").strip(),
        )
    return salida, None


def red_en_linea():
    """Lee el JSON del mapa SAT: las estaciones que transmiten HOY.

    Aporta dos cosas que el xlsx no tiene: coordenadas decimales de fábrica (sin
    convertir nada) y los campos `umbral` / `nivelAlerta`, que son el umbral de
    crecida que la propia DGA le puso a la estación. Cuando ese umbral está
    definido, es infinitamente mejor que cualquier percentil que calculemos
    nosotros, porque lo puso quien conoce el cauce.
    """
    if not JSON_EN_LINEA.exists():
        return {}, f"falta {JSON_EN_LINEA}"
    crudo = json.loads(JSON_EN_LINEA.read_text(encoding="utf8"))
    salida = {}
    for estacion in crudo:
        codigo = codigo_normal(estacion.get("codigo"))
        if not codigo:
            continue
        umbral = estacion.get("umbral")
        salida[codigo] = dict(
            nombre=limpio(estacion.get("nombre")),
            latitud=numero(estacion.get("latitud")),
            longitud=numero(estacion.get("longitud")),
            altitud=numero(estacion.get("altitud")),
            tipo=limpio(estacion.get("tipoEstacion")),
            fuente_transmision=limpio(estacion.get("fuenteEstacion")),
            # 0.0 en este campo significa «no definido», no «umbral cero»:
            # se declara como sin dato en vez de guardar un cero engañoso.
            umbral_dga=umbral if (umbral not in (None, 0, 0.0)) else None,
        )
    return salida, None


def catalogo_series():
    """Lee el listado de estaciones que vienen DENTRO del zip de series (CR)2.

    Es el único de los tres que trae PERÍODO DE REGISTRO real (primera y última
    observación y cuántas hay). Ese dato es el que decide si una estación sirve
    para validar un evento de 1997 o sólo para uno de 2019.
    """
    if not ZIP_SERIES.exists():
        return {}, f"falta {ZIP_SERIES}"
    with zipfile.ZipFile(ZIP_SERIES) as z:
        texto = z.read(DENTRO_ESTACIONES).decode("latin-1")
    salida = {}
    for fila in csv.DictReader(texto.splitlines()):
        codigo = codigo_normal(fila.get("codigo_estacion"))
        if not codigo:
            continue
        automatica = fila.get("inicio_automatica", "-")
        salida[codigo] = dict(
            nombre=limpio(fila.get("nombre")),
            latitud=numero(fila.get("latitud")),
            longitud=numero(fila.get("longitud")),
            altitud=numero(fila.get("altura")),
            cuenca=limpio(fila.get("nombre_cuenca")),
            subcuenca=limpio(fila.get("nombre_sub_cuenca")),
            codigo_cuenca=limpio(fila.get("codigo_cuenca")),
            serie_inicio=limpio(fila.get("inicio_observaciones")),
            serie_fin=limpio(fila.get("fin_observaciones")),
            serie_n=int(numero(fila.get("cantidad_observaciones")) or 0),
            serie_inicio_automatica=("" if automatica in ("-", "") else automatica),
        )
    return salida, None


# =============================================================================
# 3 · EL CATÁLOGO CONSOLIDADO
# =============================================================================

def construir_catalogo():
    """Cruza las tres fuentes por código y devuelve una fila por estación.

    Regla de precedencia para las coordenadas, declarada:
        1º  la red en línea (decimales de fábrica, sin conversión de por medio)
        2º  el (CR)2 (decimales, pero pasados por un compilador)
        3º  el xlsx oficial (convertidas por nosotros desde grados-minutos-seg)
    Y se guarda SIEMPRE `origen_coordenadas`, para que quien mire el CSV sepa
    de dónde salió cada par sin tener que confiar en esta docstring.

    Ninguna estación se descarta por estar en una sola fuente. Estar sólo en el
    (CR)2 significa «tiene historia pero ya no está activa»; estar sólo en el
    xlsx significa «está activa pero no publica serie diaria». Las dos cosas son
    información, no error.
    """
    oficial, err1 = catalogo_oficial()
    linea, err2 = red_en_linea()
    series, err3 = catalogo_series()
    problemas = [e for e in (err1, err2, err3) if e]
    if len(problemas) == 3:
        return [], "; ".join(problemas)

    codigos = sorted(set(oficial) | set(linea) | set(series))
    filas = []
    for codigo in codigos:
        o = oficial.get(codigo, {})
        l = linea.get(codigo, {})
        s = series.get(codigo, {})

        if l.get("latitud") is not None:
            lat, lon, origen = l["latitud"], l["longitud"], "sat_en_linea"
        elif s.get("latitud") is not None:
            lat, lon, origen = s["latitud"], s["longitud"], "cr2_series"
        elif o.get("latitud") is not None:
            lat, lon, origen = o["latitud"], o["longitud"], "xlsx_oficial_gms"
        else:
            lat, lon, origen = None, None, "sin dato"

        # ★ El desacuerdo entre fuentes NO se esconde: se mide y se publica.
        # 99 % de los 1.762 pares comparables coinciden dentro de 0,01° (~1 km);
        # los que no, avisan. Dos casos conocidos: el xlsx oficial trunca el
        # dígito de las centenas en la longitud de Rapa Nui (escribe 09° donde
        # va 109°), y hay una estación con 1° exacto de diferencia que ninguna
        # de las dos fuentes permite arbitrar. Quien use la coordenada tiene
        # derecho a saberlo sin abrir el crudo.
        candidatas = [(f["latitud"], f["longitud"])
                      for f in (l, s, o) if f.get("latitud") is not None]
        if len(candidatas) > 1:
            desacuerdo = max(
                max(abs(a[0] - b[0]), abs(a[1] - b[1]))
                for a in candidatas for b in candidatas)
            discrepancia = round(desacuerdo, 6) if desacuerdo >= 0.01 else ""
        else:
            discrepancia = ""

        tipo = o.get("tipo") or l.get("tipo") or ""
        filas.append(dict(
            id_fuente=ID_FUENTE,
            codigo=codigo,
            nombre=o.get("nombre") or l.get("nombre") or s.get("nombre") or "",
            latitud=lat,
            longitud=lon,
            origen_coordenadas=origen,
            coord_discrepancia_grados=discrepancia,
            altitud=(o.get("altitud") if o.get("altitud") not in (None, 0)
                     else (l.get("altitud") or s.get("altitud"))),
            region=o.get("region", ""),
            provincia=o.get("provincia", ""),
            comuna=o.get("comuna", ""),
            cuenca=o.get("cuenca") or s.get("cuenca", ""),
            subcuenca=o.get("subcuenca") or s.get("subcuenca", ""),
            subsubcuenca=o.get("subsubcuenca", ""),
            tipo=tipo,
            mide_caudal=("FLUVIOMETRICA" in tipo.upper()
                         or "FLUVIOMETRICAS" in tipo.upper()),
            fecha_instalacion=o.get("fecha_instalacion", ""),
            # las tres banderas de presencia: dicen QUÉ se puede pedirle
            activa_2026=codigo in oficial,
            transmite_en_linea=codigo in linea,
            con_serie_diaria=codigo in series,
            fuente_transmision=l.get("fuente_transmision", ""),
            umbral_dga=l.get("umbral_dga"),
            serie_inicio=s.get("serie_inicio", ""),
            serie_fin=s.get("serie_fin", ""),
            serie_n_observaciones=s.get("serie_n", ""),
            serie_inicio_automatica=s.get("serie_inicio_automatica", ""),
        ))
    return filas, ("; ".join(problemas) if problemas else None)


# =============================================================================
# 4 · LAS SERIES DIARIAS
# =============================================================================

def _abrir_matriz():
    """Devuelve (fichero-de-texto, cabecera) del archivo ancho de caudal.

    El archivo son 218 MB con una columna por estación y una fila por día desde
    1900. Se lee en streaming desde dentro del zip: nunca se descomprime a
    disco, y nunca se carga entero en memoria.
    """
    z = zipfile.ZipFile(ZIP_SERIES)
    fh = z.open(DENTRO_SERIES)
    import io
    texto = io.TextIOWrapper(fh, encoding="latin-1")
    cabecera = texto.readline().rstrip("\n").split(",")
    return z, texto, cabecera


def serie_diaria(codigos, desde=None, hasta=None):
    """Caudal medio diario (m³/s) de una o varias estaciones.

    `codigos` una lista; `desde`/`hasta` en 'AAAA-MM-DD'. Devuelve
    {codigo: [(fecha, caudal), ...]} con los días sin registro **omitidos**, no
    rellenados: un hueco es un hueco y se declara contándolo, jamás se
    interpola.
    """
    if not ZIP_SERIES.exists():
        return {}, f"falta {ZIP_SERIES}"

    pedidos = [codigo_normal(c) for c in codigos]
    z, texto, cabecera = _abrir_matriz()
    posicion = {c: i for i, c in enumerate(cabecera)}
    columnas = {c: posicion[c] for c in pedidos if c in posicion}
    faltan = [c for c in pedidos if c not in posicion]

    # las 14 filas de metadatos que van antes de los datos
    for _ in range(14):
        texto.readline()

    resultado = {c: [] for c in columnas}
    for linea in texto:
        if not linea or linea[0] not in "0123456789":
            continue
        campos = linea.rstrip("\n").split(",")
        fecha = campos[0]
        if desde and fecha < desde:
            continue
        if hasta and fecha > hasta:
            break
        for codigo, i in columnas.items():
            valor = numero(campos[i])
            if valor is None or valor == SIN_DATO:
                continue
            resultado[codigo].append((fecha, valor))
    texto.close()
    z.close()

    aviso = None
    if faltan:
        aviso = f"sin serie diaria para: {', '.join(faltan)}"
    return resultado, aviso


def crecidas(serie, percentil=99.0, dias_base=30, minimo_observaciones=730):
    """Marca los días de caudal excepcional para la HISTORIA DE ESA ESTACIÓN.

    Qué es una crecida acá, dicho sin adornos: **un día cuyo caudal supera el
    percentil P de todo lo que esa misma estación midió alguna vez**. No es «se
    salió del cauce» —eso exige la capacidad a cauce lleno, que no tenemos— sino
    «esta estación casi nunca vio tanta agua».

    Se entrega además:
      - `razon_base`: cuántas veces el caudal del día supera la mediana de los
        `dias_base` días previos con dato. Es lo que separa una crecida súbita
        (razón 10-15×, como Copiapó 2015) de un río grande en su régimen normal.
      - `apagon_posterior`: si la estación deja de registrar dentro de los 7 días
        siguientes al peak. ★ Es señal, no hueco: el evento se llevó el
        instrumento. Ignorarlo sesga el catálogo justo contra los eventos peores.

    Devuelve [] si la estación tiene menos de `minimo_observaciones` días: un
    percentil sobre 200 días no es un percentil, es una ilusión.

    ⚠ Límite conocido del percentil-de-historia-propia: en Copiapó 2015 marca
    Pulido en Vertedero (×16,6, con apagón) pero NO marca La Puerta, cuyo peak
    de 3,50 m³/s se queda bajo su propio p99 porque ese río tuvo caudales mucho
    mayores en décadas pasadas. Una estación con historia larga y régimen
    cambiado **subestima** los eventos recientes. Para no perderlos, mirar
    también `razon_base` sin filtrar por percentil.
    """
    if len(serie) < minimo_observaciones:
        return []

    valores = sorted(v for _, v in serie)
    k = (percentil / 100.0) * (len(valores) - 1)
    bajo, alto = int(k), min(int(k) + 1, len(valores) - 1)
    umbral = valores[bajo] + (k - bajo) * (valores[alto] - valores[bajo])

    fechas = [f for f, _ in serie]
    ultima = fechas[-1]

    eventos = []
    for i, (fecha, valor) in enumerate(serie):
        if valor <= umbral:
            continue
        previos = sorted(v for _, v in serie[max(0, i - dias_base):i])
        base = previos[len(previos) // 2] if previos else None
        razon = (valor / base) if (base and base > 0) else None

        limite = (date.fromisoformat(fecha) + timedelta(days=7)).isoformat()
        hay_dato_despues = any(f for f in fechas[i + 1:i + 9] if f <= limite)
        apagon = (not hay_dato_despues) and ultima >= fecha

        eventos.append(dict(
            fecha=fecha,
            caudal=valor,
            umbral_percentil=round(umbral, 3),
            caudal_base_30d=(round(base, 3) if base is not None else None),
            razon_base=(round(razon, 2) if razon is not None else None),
            apagon_posterior=apagon,
        ))
    return eventos


# =============================================================================
# 5 · INTERFAZ COMÚN Y SALIDA
# =============================================================================

def traer():
    """Interfaz común de los adaptadores del proyecto."""
    return construir_catalogo()


def escribir_csv(filas, ruta=SALIDA):
    with open(ruta, "w", newline="", encoding="utf8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas)
    return ruta


def informe(filas):
    print("=" * 78)
    print("DGA · Red Hidrométrica Nacional — catálogo consolidado")
    print("=" * 78)
    print(f"\n  estaciones distintas (unión de las 3 fuentes): {len(filas):,}")
    print(f"      activas en el catálogo oficial 2026 : "
          f"{sum(1 for f in filas if f['activa_2026']):,}")
    print(f"      transmitiendo en línea hoy          : "
          f"{sum(1 for f in filas if f['transmite_en_linea']):,}")
    print(f"      ★ con serie diaria de caudal        : "
          f"{sum(1 for f in filas if f['con_serie_diaria']):,}")
    print(f"      con coordenadas                     : "
          f"{sum(1 for f in filas if f['latitud'] is not None):,}")
    print(f"      con umbral de crecida de la DGA     : "
          f"{sum(1 for f in filas if f['umbral_dga']):,}")

    print("\n  origen de las coordenadas:")
    for k, v in Counter(f["origen_coordenadas"] for f in filas).most_common():
        print(f"      {v:5d}  {k}")
    discrepantes = [f for f in filas if f["coord_discrepancia_grados"]]
    print(f"      {len(discrepantes):5d}  ⚠ con fuentes que se contradicen "
          "más de 0,01° (ver coord_discrepancia_grados)")

    print("\n  las que traen serie diaria, por franja de latitud:")
    franjas = defaultdict(int)
    for f in filas:
        if not f["con_serie_diaria"] or f["latitud"] is None:
            continue
        lat = f["latitud"]
        if lat > -30:
            franjas["norte árido      (>30°S)"] += 1
        elif lat > -36:
            franjas["centro           (30-36°S)"] += 1
        elif lat > -42:
            franjas["sur húmedo       (36-42°S)"] += 1
        else:
            franjas["austral          (<42°S)"] += 1
    for k in sorted(franjas):
        print(f"      {franjas[k]:5d}  {k}")

    con_serie = [f for f in filas if f["con_serie_diaria"] and f["serie_fin"]]
    print("\n  hasta cuándo llega el registro de las que tienen serie:")
    tramos = Counter()
    for f in con_serie:
        anio = int(f["serie_fin"][:4])
        tramos[">= 2018" if anio >= 2018 else
               ("2010-2017" if anio >= 2010 else
                ("1990-2009" if anio >= 1990 else "< 1990"))] += 1
    for k in (">= 2018", "2010-2017", "1990-2009", "< 1990"):
        if tramos[k]:
            print(f"      {tramos[k]:5d}  {k}")

    print("\n  cuencas distintas con serie diaria: "
          f"{len({f['cuenca'] for f in con_serie if f['cuenca']})}")

    print("\n  ⚠ la serie diaria llega hasta MARZO DE 2020. De ahí en adelante")
    print("    sólo hay tiempo real en el portal SAT, no serie consolidada.")
    print("  ⚠ caudal alto NO es lo mismo que desborde: sin la capacidad a")
    print("    cauce lleno sólo se puede afirmar «caudal excepcional».")


def demostracion_copiapo():
    """La prueba de que el instrumento vio el evento que SENAPRED clasificó mal.

    24-25 de marzo de 2015. SENAPRED lo llama «Inundación»; técnicamente fue un
    flujo de detritos. Lo que sigue no discute la etiqueta: muestra el agua.
    """
    estaciones = {
        "03430003": "Río Copiapó en Pastillo",
        "03414001": "Río Pulido en Vertedero",
        "03431001": "Río Copiapó en La Puerta",
        "03434003": "Río Copiapó en Mal Paso Aguas Abajo",
        "03430001": "Río Copiapó en Lautaro",
    }
    series, aviso = serie_diaria(list(estaciones), "2015-03-01", "2015-04-05")
    print("=" * 78)
    print("CASO ANCLA · Copiapó, 24-25 de marzo de 2015")
    print("=" * 78)
    if aviso:
        print(f"  aviso: {aviso}")
    for codigo, nombre in estaciones.items():
        datos = series.get(codigo, [])
        print(f"\n  {codigo}  {nombre}")
        if not datos:
            print("      SIN DATO en la ventana pedida")
            continue
        # el peak se marca UNA vez, y sólo si la serie tiene variación: en una
        # estación plana (caudal constante) no hay peak que marcar.
        maximo = max(v for _, v in datos)
        i_peak = ([i for i, (_, v) in enumerate(datos) if v == maximo][0]
                  if maximo > min(v for _, v in datos) else -1)
        for i, (fecha, valor) in enumerate(datos):
            marca = "  ◀ PEAK" if i == i_peak else ""
            print(f"      {fecha}  {valor:9.3f} m³/s{marca}")
        ultima = datos[-1][0]
        if ultima < "2015-04-01":
            print(f"      ★ la estación DEJA DE REGISTRAR el {ultima} — "
                  "el evento se llevó el instrumento")


def demostracion_crecidas(codigos):
    print("=" * 78)
    print("CRECIDAS · días sobre el percentil 99 de la propia estación")
    print("=" * 78)
    series, aviso = serie_diaria(codigos)
    if aviso:
        print(f"  aviso: {aviso}")
    for codigo in [codigo_normal(c) for c in codigos]:
        datos = series.get(codigo, [])
        print(f"\n  {codigo}  ({len(datos):,} días con dato)")
        if not datos:
            print("      SIN DATO")
            continue
        eventos = crecidas(datos)
        if not eventos:
            print("      serie demasiado corta para un percentil honesto")
            continue
        print(f"      {len(eventos)} días sobre el percentil 99 "
              f"(umbral {eventos[0]['umbral_percentil']} m³/s)")
        print("      los 10 de mayor salto sobre su línea base de 30 días:")
        peores = sorted(eventos, key=lambda e: -(e["razon_base"] or 0))[:10]
        for e in peores:
            apagon = "  ★ APAGÓN POSTERIOR" if e["apagon_posterior"] else ""
            print(f"        {e['fecha']}  {e['caudal']:10.2f} m³/s  "
                  f"×{e['razon_base'] or 0:6.1f} sobre base{apagon}")


if __name__ == "__main__":
    if "--copiapo" in sys.argv:
        demostracion_copiapo()
    elif "--crecidas" in sys.argv:
        pedidos = [a for a in sys.argv[1:] if not a.startswith("--")]
        demostracion_crecidas(pedidos or ["03430003", "08117001", "10134001"])
    else:
        filas, problema = traer()
        if not filas:
            print("SIN DATO:", problema)
            raise SystemExit(1)
        if problema:
            print(f"⚠ fuente(s) incompleta(s): {problema}\n")
        informe(filas)
        if "--csv" in sys.argv:
            ruta = escribir_csv(filas)
            print(f"\n  escrito: {ruta}  ({len(filas):,} filas)")

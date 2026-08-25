"""
ADAPTADOR DesInventar Chile — la tercera fuente: la prensa, sistematizada
=========================================================================

QUÉ ES
------
DesInventar es el inventario histórico de desastres de UNDRR, construido **a
partir de archivos de prensa** con una metodología estándar (LA RED). La base
chilena cubre **1970-2014** con **13.534 fichas** y cita la fuente periodística
de cada una — mayoritariamente El Mercurio y La Tercera.

POR QUÉ LA NECESITAMOS (instrucción de Alexis, 17-ago-2026)
------------------------------------------------------------
Porque las dos fuentes que teníamos no pueden separar las familias de amenaza:

- **SENAPRED** clasifica el aluvión de Copiapó del 24-mar-2015 —el caso ancla de
  este proyecto, 22.111 afectados— como `Sub evento: Inundación`. En sus 50.457
  registros la palabra «aluvión» aparece 25 veces, repartida entre CINCO
  etiquetas distintas. No distingue lo que la hipótesis necesita distinguir.
- **ReTeRM** (SERNAGEOMIN) sí es confiable, pero sólo tiene remociones en masa, y
  el norte árido aporta 23 de 329 eventos: el 7 %.

DesInventar trae **Aluvión, Inundación, Deslizamiento, Alud y Avenida torrencial
como categorías SEPARADAS**, y en el norte árido tiene **106 eventos** de las dos
familias que nos importan, contra 23 de ReTeRM.

LAS TRES CAPAS DE CLASIFICACIÓN (decisión de Alexis, 17-ago-2026)
------------------------------------------------------------------
No se elige una fuente y se descartan las otras. Se guardan las tres cosas:

1. `familia_<fuente>` — **qué dice cada fuente**, tal cual, sin traducir.
2. `familia_concordancia` — **en qué coinciden**. Si dos fuentes independientes
   dicen lo mismo, eso vale más que cualquiera por separado. Si discrepan, el
   desacuerdo NO se resuelve en silencio: queda escrito en `desacuerdo`.
3. `familia_tecnica` — **nuestra categoría formal**, asignada por regla
   declarada, aunque contradiga a las fuentes. Es la que nos cubre: permite
   decir «SENAPRED lo llama inundación, la prensa lo llama aluvión, y
   técnicamente es un flujo de detritos» sin tener que darle la razón a nadie.

La regla de la capa técnica se declara ABAJO, en TECNICA_POR_EVENTO, y los casos
que la regla no resuelve quedan como `INDETERMINADA`. Nunca se adivina.

SESGO QUE HAY QUE DECLARAR SIEMPRE
-----------------------------------
La prensa cubre lo **noticioso**: grande, urbano, cerca de Santiago, reciente.
El 90 % de las citas de esta base son de El Mercurio, diario de Santiago. Un
camino rural cortado tres días en Alto Biobío no sale en el diario.

Es un sesgo DISTINTO al de las otras dos —SERNAGEOMIN registra donde va a mirar,
SENAPRED registra lo que le reportan— y por eso las tres juntas se corrigen. Pero
ninguna es una muestra al azar del país, y eso hay que escribirlo al lado de
cualquier número que salga de aquí.

REGLA DE USO: PRENSA PARA ETIQUETAS, INSTRUMENTO PARA MAGNITUDES
-----------------------------------------------------------------
La prensa **nombra bien y mide mal**. Sirve para decir *qué fue* y *dónde*; no
para decir *cuánto llovió*. Jamás se usa la misma fuente para etiquetar y para
validar: eso es la circularidad que ya nos mordió con ReTeRM, donde el detonante
«lluvia» lo asigna el mismo organismo que registra el evento.
"""

import csv
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent.parent

XML = (AQUI / "datos" / "crudo" / "desinventar" / "2026-08-17" /
       "DI_export_chl.xml")
SALIDA = AQUI / "datos" / "desinventar_eventos.csv"

ID_FUENTE = "desinventar_undrr"

FUENTE = dict(
    id=ID_FUENTE,
    nombre="DesInventar Chile (UNDRR / LA RED)",
    naturaleza="prensa sistematizada",
    periodo="1970-2014",
    url="https://www.desinventar.net/DesInventar/download/DI_export_chl.zip",
    # Confianza más baja que una fuente instrumental: es testimonio de prensa
    # sistematizado, no medición. Alta para la ETIQUETA, baja para la magnitud.
    confianza_base=0.70,
)

# --- Capa 1 · qué familias nos interesan de las 40+ que tiene DesInventar -----

FAMILIA_POR_EVENTO = {
    "Aluvión": "flujo",              # avenida torrencial con arrastre de sólidos
    "Avenida torrencial": "flujo",
    "Deslizamiento": "remocion",
    "Alud": "remocion",              # de nieve/hielo, pero es remoción de masa
    "Inundación": "desborde",
    "Lluvias": "lluvia",             # no es familia: es el detonante suelto
}

# --- Capa 3 · LA REGLA TÉCNICA, declarada ------------------------------------
#
# Se asigna por el tipo de evento de DesInventar, que ya viene de una definición
# escrita en su guía metodológica. Donde la definición no alcanza para decidir,
# se deja INDETERMINADA — no se adivina.
#
# Las categorías salen del estudio de vectores del proyecto:
#   FLUJO_DETRITOS  agua + material sólido bajando por quebrada; pide cuenca
#                   árida o semiárida, pendiente y regolito suelto
#   DESLIZAMIENTO   masa de suelo o roca que se desplaza por una ladera
#   DESBORDE_FLUVIAL  el río sale de su cauce
#   ALUD_NIEVE      desprendimiento de nieve o hielo
#   INDETERMINADA   la fuente no permite decidir

TECNICA_POR_EVENTO = {
    "Aluvión": "FLUJO_DETRITOS",
    "Avenida torrencial": "FLUJO_DETRITOS",
    "Deslizamiento": "DESLIZAMIENTO",
    "Alud": "ALUD_NIEVE",
    "Inundación": "INDETERMINADA",   # ★ ver nota
    "Lluvias": "INDETERMINADA",
}
# ★ NOTA sobre «Inundación»: es la etiqueta más ambigua de todas y por eso NO se
# traduce automáticamente a DESBORDE_FLUVIAL. Una «inundación» puede ser el río
# saliendo del cauce (desborde), agua acumulada por drenaje urbano insuficiente
# (anegamiento), o un flujo de detritos que el cronista llamó inundación — que es
# exactamente lo que le pasó a SENAPRED con Copiapó 2015. Resolverla exige mirar
# el terreno (¿hay río aguas arriba? ¿qué pendiente?), y eso es trabajo del
# enrutador, no del adaptador. Queda INDETERMINADA a propósito.

# Regiones áridas y semiáridas: donde el flujo de detritos es el modo dominante
# y donde ReTeRM casi no tiene eventos.
NORTE_ARIDO = ("Arica y Parinacota", "Tarapacá", "Antofagasta", "Atacama",
               "Coquimbo")


def leer_fichas(ruta=XML):
    """Recorre el XML de DesInventar y devuelve las fichas como diccionarios.

    El XML mete todo en elementos <TR> genéricos, y los <TR> de las fichas
    conviven con los del catálogo de eventos, el de causas y los de metadatos.
    Se distinguen por PROFUNDIDAD: las fichas están a nivel 3, colgando del
    contenedor <fichas>. Filtrar por el nombre del contenedor es frágil porque
    los campos hijos también son elementos con nombre propio.
    """
    profundidad = 0
    for accion, elemento in ET.iterparse(ruta, events=("start", "end")):
        if accion == "start":
            profundidad += 1
            continue
        if elemento.tag == "TR" and profundidad == 3:
            ficha = {hijo.tag: (hijo.text or "").strip() for hijo in elemento}
            if "evento" in ficha and "fechano" in ficha:
                yield ficha
            elemento.clear()
        profundidad -= 1


def fecha_iso(ficha):
    """Arma la fecha. DesInventar a veces trae día o mes en blanco: se declara."""
    anio = ficha.get("fechano", "")
    mes = (ficha.get("fechames") or "").zfill(2)
    dia = (ficha.get("fechadia") or "").zfill(2)
    if not anio.isdigit():
        return None, "sin año"
    if mes in ("", "00"):
        return f"{anio}", "sólo año"
    if dia in ("", "00"):
        return f"{anio}-{mes}", "sólo mes"
    return f"{anio}-{mes}-{dia}", "día exacto"


def entero(texto):
    try:
        return int(float(texto))
    except (TypeError, ValueError):
        return 0


def traer():
    """Interfaz común de los adaptadores: lista de eventos normalizados."""
    if not XML.exists():
        return [], f"falta {XML}"

    eventos = []
    for ficha in leer_fichas():
        tipo = ficha.get("evento", "")
        if tipo not in FAMILIA_POR_EVENTO:
            continue                      # no es una familia que nos ocupe
        fecha, precision = fecha_iso(ficha)
        if fecha is None:
            continue
        region = ficha.get("name0", "")
        eventos.append(dict(
            id_fuente=ID_FUENTE,
            serial=ficha.get("serial", ""),
            fecha=fecha,
            fecha_precision=precision,
            region=region,
            provincia=ficha.get("name1", ""),
            comuna=ficha.get("name2", ""),
            lugar=ficha.get("lugar", "")[:200],
            # capa 1 · lo que dice ESTA fuente, sin traducir
            evento_fuente=tipo,
            familia_fuente=FAMILIA_POR_EVENTO[tipo],
            # capa 3 · nuestra categoría formal, por regla declarada
            familia_tecnica=TECNICA_POR_EVENTO[tipo],
            arido=any(n in region for n in NORTE_ARIDO),
            muertos=entero(ficha.get("muertos")),
            heridos=entero(ficha.get("heridos")),
            afectados=entero(ficha.get("afectados")),
            viviendas_destruidas=entero(ficha.get("vivdest")),
            viviendas_afectadas=entero(ficha.get("vivafec")),
            cita_prensa=ficha.get("fuentes", "")[:160],
        ))
    return eventos, None


def informe(eventos):
    print("=" * 76)
    print("DesInventar Chile · la prensa sistematizada, 1970-2014")
    print("=" * 76)
    print(f"\n  eventos de las familias que nos ocupan: {len(eventos):,}")

    print("\n  por tipo declarado por la fuente:")
    for k, v in Counter(e["evento_fuente"] for e in eventos).most_common():
        print(f"      {v:5d}  {k}")

    print("\n  por CATEGORÍA TÉCNICA nuestra (regla declarada):")
    for k, v in Counter(e["familia_tecnica"] for e in eventos).most_common():
        print(f"      {v:5d}  {k}")

    aridos = [e for e in eventos if e["arido"]]
    print(f"\n  ★ en el norte árido (XV a IV): {len(aridos)}")
    for k, v in Counter(e["evento_fuente"] for e in aridos).most_common():
        print(f"      {v:5d}  {k}")

    print("\n  precisión de la fecha:")
    for k, v in Counter(e["fecha_precision"] for e in eventos).most_common():
        print(f"      {v:5d}  {k}")

    print("\n  los 8 más letales del conjunto:")
    for e in sorted(eventos, key=lambda x: -x["muertos"])[:8]:
        print(f"      {e['fecha']:10s} {e['evento_fuente']:12s} "
              f"{e['muertos']:4d} muertos · {e['afectados']:7,d} afectados · "
              f"{e['region'][:20]:22s} {e['comuna'][:18]}")

    print("\n  ⚠ sesgo de la fuente: 90 % de las citas son de El Mercurio,")
    print("    diario de Santiago. Esto NO es una muestra al azar del país.")


if __name__ == "__main__":
    eventos, problema = traer()
    if problema:
        print("SIN DATO:", problema)
        raise SystemExit(1)
    informe(eventos)
    if "--csv" in sys.argv:
        with open(SALIDA, "w", newline="", encoding="utf8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(eventos[0].keys()))
            w.writeheader()
            w.writerows(eventos)
        print(f"\n  escrito: {SALIDA}  ({len(eventos):,} filas)")

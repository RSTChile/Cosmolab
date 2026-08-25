#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
construir_catalogo_modos_falla.py
=================================

QUÉ HACE
--------
Extrae, del registro real de emergencias de SENAPRED (50.457 eventos por comuna,
2015-2024), el CATÁLOGO DE MODOS DE FALLA de la infraestructura chilena.

La idea rectora del encargo: **no inventamos modos de falla, los ENCONTRAMOS**
en lo que efectivamente pasó y quedó anotado por las Unidades de Alerta Temprana
regionales.

Para cada modo de falla el script mide, con conteos:
  · frecuencia y participación en el total
  · CAUSA DECLARADA (sólo cuando el registro la declara, ver más abajo)
  · ESTACIONALIDAD mes a mes, e índice invierno/verano contra la línea base
  · DISTRIBUCIÓN TERRITORIAL (regiones y comunas que concentran)
  · TENDENCIA en los diez años
  · clasificación del vector en NATURAL / DELIBERADO / NO DISTINGUIBLE
  · qué PROPIEDAD EXPUESTA del elemento explota, en el vocabulario de
    ESTUDIO_VECTORES_DE_AMENAZA.md §4.3

PRIVACIDAD — REGLA INNEGOCIABLE
-------------------------------
La columna «Antecedentes Observaciones» del Excel contiene texto libre redactado
por los operadores, con RUT y descripciones de personas fallecidas. **Este script
la descarta en el momento de la lectura y jamás la carga en memoria de trabajo.**
Ver COLUMNAS_PROHIBIDAS más abajo: la exclusión es por nombre y ocurre dentro del
propio lector, antes de cualquier análisis.

Las demás columnas del archivo son o bien categóricas (región / comuna / tipo de
evento) o bien CONTEOS AGREGADOS de personas (afectados, damnificados, aislados).
Un conteo agregado por comuna y fecha no identifica a nadie, así que se usa. No
hay en el archivo ninguna otra columna con dato personal identificable.

POR QUÉ EL SCRIPT ES TAN CUIDADOSO CON LA TAXONOMÍA
---------------------------------------------------
El registro no tiene una columna «modo de falla». Tiene cuatro columnas anidadas
—Clase Evento › Tipo Evento › Sub Evento 1 › Sub Evento 2— que los operadores
llenaron con criterios que cambiaron a lo largo de los diez años:

  · hay 69 valores distintos de «Clase Evento» que en realidad son ~25 conceptos
    repetidos con mayúsculas, acentos y espacios finales distintos
    («Incendios» / «Incendios » / «Vientos» / «Vientos »);
  · el AÑO 2021 corrió toda la jerarquía un nivel: lo que los demás años ponen
    en «Tipo Evento» (p.ej. «Interrupción Suministro Eléctrico»), 2021 lo pone
    directamente en «Clase Evento»;
  · el uso de «Sub Evento 1» sube de 12,5 % de los eventos en 2015 a 62,7 % en
    2022: la práctica de anotar el encadenamiento causal se fue instalando.

Por eso el script normaliza (minúsculas, sin acentos, espacios colapsados) y
busca el modo de falla EN LAS CUATRO COLUMNAS a la vez, en vez de confiar en el
nivel jerárquico. Y por eso declara la tendencia como sospechosa de reflejar
cambios de registro y no sólo cambios del mundo.

CÓMO SE OBTIENE LA «CAUSA DECLARADA» — y por qué falta casi siempre
-------------------------------------------------------------------
El hallazgo central del script. Hay dos regímenes de anotación:

  (a) Evento anotado como falla suelta: Clase = «Falla de Servicios y
      Suministros», Tipo = «Interrupción Suministro Eléctrico», sub-eventos
      vacíos. **No hay causa en ninguna columna estructurada.** Lo que la
      explique, si algo la explica, está en el texto libre que tenemos prohibido
      leer.

  (b) Evento anotado como encadenamiento: Clase = «Precipitaciones», Tipo =
      «Sistema Frontal», Sub Evento 1 = «Alteración Servicio Suministro
      Eléctrico». Aquí el registro SÍ declara la causa: es el fenómeno del nivel
      padre.

El script sólo asigna causa en el caso (b), y reporta explícitamente qué
porcentaje de cada modo quedó sin causa declarada. Nunca imputa una causa.

SALIDAS
-------
  datos/modos_falla_senapred.csv   — la tabla del catálogo, una fila por modo
  (el informe legible se escribe aparte en CATALOGO_MODOS_DE_FALLA.md)

USO
---
  .venv-esa/bin/python infraestructura/construir_catalogo_modos_falla.py
"""

import csv
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import openpyxl

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
AQUI = Path(__file__).resolve().parent
XLSX = AQUI / "datos" / "crudo" / "senapred" / "2026-08-15" / "Eventos_Emergencia_2015_2024.xlsx"
HOJA = "Eventos_de_Emergencia_2015_2024"
SALIDA_CSV = AQUI / "datos" / "modos_falla_senapred.csv"

# Columnas que NUNCA se cargan. La exclusión ocurre en el lector, no después.
COLUMNAS_PROHIBIDAS = {"Antecedentes Observaciones"}

# Meses del hemisferio sur: invierno austral y verano austral.
INVIERNO = {6, 7, 8}
VERANO = {12, 1, 2}


# --------------------------------------------------------------------------
# Utilidades de normalización
# --------------------------------------------------------------------------
def norm(s):
    """Minúsculas, sin acentos, espacios colapsados.

    Imprescindible: el registro escribe el mismo concepto de muchas maneras
    («Eléctrico» / «electrico» / «Incendios »). Sin esto, un mismo modo de falla
    se cuenta como varios y todos los conteos quedan mal.
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s).strip().lower()


def num(v):
    """Conteo de personas/viviendas a float; cualquier basura vale 0."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------
# Diccionario de MODOS DE FALLA
# --------------------------------------------------------------------------
# Cada modo se reconoce por patrones sobre el texto normalizado de las cuatro
# columnas de taxonomía. El orden importa: se evalúa de arriba hacia abajo y
# gana el primero que calce, así que los modos más específicos van primero.
#
# Campos de cada modo:
#   categoria   INFRAESTRUCTURA (lo que le interesa a este proyecto) o CONTEXTO
#               (eventos que no son falla de infraestructura pero que hay que
#               contar para responder «qué falla en Chile», y que además son las
#               CAUSAS de varias fallas de infraestructura)
#   patrones    expresiones regulares sobre el texto normalizado
#   propiedades propiedades expuestas del elemento, vocabulario del §4.3 del
#               ESTUDIO_VECTORES_DE_AMENAZA.md:
#               dep_energia · dep_datos · dep_humana · exp_intemperie ·
#               extension · confinamiento · redundancia · t_reposicion
#   nota        por qué esa propiedad y no otra
MODOS = [
    # ---------------- Fallas de infraestructura ----------------
    dict(
        clave="ELEC_SUMINISTRO",
        nombre="Interrupción o alteración del suministro eléctrico",
        categoria="INFRAESTRUCTURA",
        patrones=[r"suministro (de )?electric", r"servicio suministro electric",
                  r"suministro electric", r"cortes de servicios basicos"],
        propiedades="dep_energia · extension (lineal) · exp_intemperie · redundancia ausente",
        nota=("La red de distribución es lineal y está a la intemperie en toda su "
              "longitud: cualquier punto de sus kilómetros sirve de blanco al viento, "
              "al árbol o al hielo. Todo lo que cuelga aguas abajo depende de ella "
              "(dep_energia) y en la baja tensión chilena no hay camino alterno."),
    ),
    dict(
        clave="AGUA_POTABLE",
        nombre="Interrupción o alteración del agua potable",
        categoria="INFRAESTRUCTURA",
        patrones=[r"agua potable", r"servicio (de )?agua\b", r"suministro (de )?agua\b",
                  r"sequedad de pozos"],
        propiedades="dep_energia · extension (lineal) · dep_humana · t_reposicion",
        nota=("El agua potable urbana depende de bombeo, o sea de la electricidad: es "
              "un modo de falla de segundo orden. Su matriz también es lineal y su "
              "reposición no es instantánea (hay que reponer presión y sanitizar)."),
    ),
    dict(
        clave="TELECOM",
        nombre="Interrupción de telecomunicaciones o fibra óptica",
        categoria="INFRAESTRUCTURA",
        patrones=[r"telecomunic", r"fibra optica", r"telefon", r"internet"],
        propiedades="dep_datos · dep_energia · extension (lineal) · redundancia",
        nota=("Vector típico de lo deliberado (dep_datos) pero aquí aparece casi "
              "siempre como consecuencia del corte eléctrico: sin energía en la "
              "antena, no hay red."),
    ),
    dict(
        clave="GAS",
        nombre="Interrupción o alteración del suministro de gas",
        categoria="INFRAESTRUCTURA",
        patrones=[r"servicio gas", r"suministro (de )?gas"],
        propiedades="extension (lineal) · confinamiento · t_reposicion",
        nota=("Red lineal enterrada; el confinamiento la protege del clima pero "
              "convierte cualquier fuga en un evento de materiales peligrosos."),
    ),
    dict(
        clave="ALCANTARILLADO",
        nombre="Falla del alcantarillado",
        categoria="INFRAESTRUCTURA",
        patrones=[r"alcantarill", r"aguas servidas"],
        propiedades="extension (lineal) · confinamiento · exp_intemperie (capacidad)",
        nota=("Falla por saturación: la lluvia que entra excede la capacidad de "
              "diseño. No se rompe, se desborda."),
    ),
    dict(
        clave="CONECTIVIDAD_VIAL",
        nombre="Corte o alteración de la conectividad vial",
        categoria="INFRAESTRUCTURA",
        patrones=[r"conectividad", r"accesibilidad", r"\bvial\b", r"interrupcion (de )?ruta"],
        propiedades="extension (lineal) · exp_intemperie · redundancia (rutas alternas) · t_reposicion",
        nota=("El camino es el elemento lineal por excelencia: se corta en su punto "
              "más débil, no en su promedio. La redundancia aquí es literal — "
              "¿existe otra ruta? — y es lo que separa un atraso de un aislamiento."),
    ),
    dict(
        clave="COLAPSO_ESTRUCTURAL",
        nombre="Colapso estructural",
        categoria="INFRAESTRUCTURA",
        patrones=[r"colapso estructural", r"^colapso"],
        propiedades="t_reposicion · dep_humana · confinamiento",
        nota="Pérdida total del elemento; el tiempo de reposición domina todo lo demás.",
    ),
    # ---------------- Eventos de contexto (y causas) ----------------
    dict(
        clave="INC_ESTRUCTURAL",
        nombre="Incendio estructural (residencial, comercial, público, industrial)",
        categoria="CONTEXTO",
        patrones=[r"incendio estructural", r"^estructural$", r"incendio vertedero",
                  r"incendio relleno"],
        propiedades="confinamiento · dep_humana · t_reposicion",
        nota="Es el evento más frecuente del registro, y es antrópico no deliberado.",
    ),
    dict(
        clave="INC_FORESTAL",
        nombre="Incendio forestal",
        categoria="CONTEXTO",
        patrones=[r"incendio forestal", r"^forestal$", r"incendio de interfaz", r"interfaz"],
        propiedades="extension (areal) · exp_intemperie",
        nota="Amenaza areal: no elige blanco, barre superficie. Causa de cortes eléctricos.",
    ),
    dict(
        clave="ACC_TRANSPORTE",
        nombre="Accidente de medios de transporte",
        categoria="CONTEXTO",
        patrones=[r"accidente.*transporte", r"transporte terrestre", r"transporte naviero",
                  r"transporte aereo", r"transporte ferroviario", r"incendio transporte"],
        propiedades="dep_humana",
        nota="Principal causa NO climática de corte vial; ocurre parejo todo el año.",
    ),
    dict(
        clave="MAT_PELIGROSOS",
        nombre="Incidente con materiales peligrosos (fuga, derrame, emanación)",
        categoria="CONTEXTO",
        patrones=[r"material(es)? peligros", r"^fuga", r"^derrame", r"emanacion",
                  r"intoxicacion", r"explosion"],
        propiedades="confinamiento · dep_humana",
        nota="Falla de contención; el confinamiento es a la vez la defensa y el riesgo.",
    ),
    dict(
        clave="REMOCION_MASA",
        nombre="Remoción en masa (deslizamiento, aluvión, derrumbe, caída de rocas)",
        categoria="CONTEXTO",
        patrones=[r"remocion en masa", r"aluvion", r"deslizamiento", r"derrumbe",
                  r"desprendimiento", r"desmoronamiento", r"alud", r"reblandecimiento"],
        propiedades="exp_intemperie · extension",
        nota="Vector natural del §4.2; exige suelo SECO y regolito suelto.",
    ),
    dict(
        clave="INUNDACION",
        nombre="Inundación, anegamiento o desborde",
        categoria="CONTEXTO",
        patrones=[r"inundacion", r"anegamiento", r"desborde", r"aumento (de )?caudal",
                  r"crecida"],
        propiedades="exp_intemperie · extension (areal)",
        nota="Vector natural del §4.2; exige suelo SATURADO — el signo opuesto al anterior.",
    ),
    dict(
        clave="METEO_PRECIP",
        nombre="Evento meteorológico de precipitación (sistema frontal, lluvia, nevada, núcleo frío)",
        categoria="CONTEXTO",
        patrones=[r"sistema frontal", r"precipitacion", r"lluvia", r"llovizna", r"nevada",
                  r"nieve", r"nucleo frio", r"^tormenta", r"granizo"],
        propiedades="exp_intemperie",
        nota="La causa declarada más frecuente de las fallas de infraestructura.",
    ),
    dict(
        clave="VIENTO",
        nombre="Viento fuerte",
        categoria="CONTEXTO",
        patrones=[r"^viento", r"\bvientos?\b"],
        propiedades="exp_intemperie · extension",
        nota="Segunda causa declarada de corte eléctrico; actúa sobre la línea, no sobre el nodo.",
    ),
    dict(
        clave="TEMP_EXTREMAS",
        nombre="Temperaturas extremas (ola de calor, helada)",
        categoria="CONTEXTO",
        patrones=[r"temperatura", r"ola de calor", r"helada", r"^calor"],
        propiedades="exp_intemperie · dep_energia",
        nota="Modo de falla por DEMANDA, no por daño: el elemento no se rompe, se satura.",
    ),
    dict(
        clave="MAREJADAS",
        nombre="Marejadas",
        categoria="CONTEXTO",
        patrones=[r"marejada", r"oleaje"],
        propiedades="exp_intemperie · extension (lineal costera)",
        nota="Amenaza de borde: afecta una franja, no un área.",
    ),
    dict(
        clave="SISMO_VOLCAN",
        nombre="Sismo o erupción volcánica",
        categoria="CONTEXTO",
        patrones=[r"^sismo", r"erupcion", r"volcan", r"tsunami", r"ceniza"],
        propiedades="exp_intemperie · extension (areal)",
        nota="Único vector natural del registro que NO es climático: sirve de control.",
    ),
    dict(
        clave="DEFICIT_HIDRICO",
        nombre="Déficit hídrico / sequía",
        categoria="CONTEXTO",
        patrones=[r"deficit hidrico", r"sequia", r"escasez hidrica"],
        propiedades="t_reposicion · extension (areal)",
        nota="Amenaza lenta: el registro de emergencias la ve mal porque no tiene fecha de inicio.",
    ),
    dict(
        clave="RESCATE",
        nombre="Búsqueda y rescate de personas / accidente recreacional",
        categoria="CONTEXTO",
        patrones=[r"rescate", r"busqueda", r"recreacional", r"extraviad", r"turismo",
                  r"deportiv"],
        propiedades="dep_humana",
        nota="No es falla de infraestructura; se cuenta para no inflar otros modos.",
    ),
    dict(
        clave="PLAGAS_BIO",
        nombre="Plagas y eventos biológicos",
        categoria="CONTEXTO",
        patrones=[r"plaga", r"sanitari", r"biologic", r"marea roja", r"pandemia"],
        propiedades="dep_humana",
        nota="Familia menor en el registro.",
    ),
    dict(
        clave="INCENDIO_OTRO",
        nombre="Incendio sin especificar",
        categoria="CONTEXTO",
        patrones=[r"incendio"],
        propiedades="confinamiento",
        nota="Cajón de los incendios que el operador no clasificó; va último para no robarle filas a los anteriores.",
    ),
]

# --------------------------------------------------------------------------
# Fenómenos que el registro puede declarar COMO CAUSA de un modo de falla.
# Se buscan en los niveles PADRE (Clase Evento, Tipo Evento) cuando el modo de
# falla fue detectado en un nivel HIJO (Sub Evento 1 o 2). Ése es el único caso
# en que el registro declara un encadenamiento causal.
# --------------------------------------------------------------------------
CAUSAS = [
    ("Sistema frontal / precipitaciones", [r"sistema frontal", r"precipitacion", r"lluvia",
                                           r"llovizna", r"nucleo frio"]),
    ("Nevadas", [r"nevada", r"nieve"]),
    ("Viento", [r"viento"]),
    ("Tormenta eléctrica", [r"tormenta"]),
    ("Incendio forestal", [r"incendio forestal", r"^forestal$"]),
    ("Incendio estructural", [r"incendio estructural", r"^estructural$"]),
    ("Incendio sin especificar", [r"^incendios?$", r"incendio"]),
    ("Accidente de transporte", [r"accidente.*transporte", r"transporte terrestre"]),
    ("Remoción en masa", [r"remocion en masa", r"aluvion", r"deslizamiento", r"derrumbe"]),
    ("Inundación / desborde", [r"inundacion", r"anegamiento", r"desborde"]),
    ("Sismo / erupción volcánica", [r"^sismo", r"erupcion", r"volcan"]),
    ("Marejadas", [r"marejada", r"oleaje"]),
    ("Temperaturas extremas", [r"temperatura", r"ola de calor", r"helada"]),
    ("Materiales peligrosos", [r"material(es)? peligros", r"fuga", r"derrame", r"emanacion"]),
    ("Déficit hídrico", [r"deficit hidrico", r"sequia"]),
]

# Clases que NO son un fenómeno sino un cajón administrativo: si la causa
# "declarada" resulta ser una de éstas, la causa en realidad NO está declarada.
CAJONES_SIN_CAUSA = [
    r"falla de servicios y suministros",
    r"falla de conectividad",
    r"accidentes miscelaneos",
    r"^interrupcion", r"^alteracion",
]

# Palabras que delatarían un vector DELIBERADO. Se buscan en toda la taxonomía
# para poder afirmar con números —y no de oídas— si el registro los distingue.
PALABRAS_DELIBERADO = [
    r"sabotaj", r"atentad", r"vandal", r"robo", r"hurto", r"intencional", r"delict",
    r"manifest", r"disturbi", r"terror", r"ciber", r"ataque", r"saqueo", r"protesta",
    r"corte de cable", r"sustraccion", r"malicios",
]

# Subconjunto de CAUSAS que son METEOROLÓGICAS (o disparadas por meteorología).
# Es la separación que hace falsable la pregunta invierno/verano: todo lo que NO
# está en este conjunto sirve de grupo de control.
CAUSAS_METEO = {
    "Sistema frontal / precipitaciones", "Nevadas", "Viento", "Tormenta eléctrica",
    "Inundación / desborde", "Remoción en masa", "Marejadas", "Temperaturas extremas",
}


# Una «causa» que nombra el mismo fenómeno que el modo NO es información: es la
# jerarquía repitiéndose («Incendios › Incendio Estructural Residencial»). Se
# suprime para que la columna de causas declaradas diga algo.
CAUSA_ES_EL_MISMO_MODO = {
    "Incendio estructural": {"INC_ESTRUCTURAL", "INCENDIO_OTRO"},
    "Incendio sin especificar": {"INC_ESTRUCTURAL", "INC_FORESTAL", "INCENDIO_OTRO"},
    "Incendio forestal": {"INC_FORESTAL"},
    "Accidente de transporte": {"ACC_TRANSPORTE"},
    "Materiales peligrosos": {"MAT_PELIGROSOS"},
    "Remoción en masa": {"REMOCION_MASA"},
    "Inundación / desborde": {"INUNDACION"},
    "Sistema frontal / precipitaciones": {"METEO_PRECIP"},
    "Nevadas": {"METEO_PRECIP"},
    "Tormenta eléctrica": {"METEO_PRECIP"},
    "Viento": {"VIENTO"},
    "Marejadas": {"MAREJADAS"},
    "Sismo / erupción volcánica": {"SISMO_VOLCAN"},
    "Temperaturas extremas": {"TEMP_EXTREMAS"},
    "Déficit hídrico": {"DEFICIT_HIDRICO"},
}

# Mínimo de eventos con causa declarada para atreverse a etiquetar el vector.
# Sin esto, un modo con 359 eventos y UNA causa anotada quedaría rotulado
# «NATURAL» por esa única fila. Sería inventar.
MIN_CAUSAS_PARA_ROTULAR = 20


def base_ratio_global(filas):
    """Razón invierno/verano del registro COMPLETO: la vara con la que se mide todo."""
    i = sum(1 for r in filas if r["_mes"] in INVIERNO)
    v = sum(1 for r in filas if r["_mes"] in VERANO)
    return i / v if v else float("nan")


# --------------------------------------------------------------------------
# Lectura del Excel (con la exclusión de privacidad incorporada)
# --------------------------------------------------------------------------
def leer_eventos():
    """Lee el Excel descartando en el acto las columnas prohibidas.

    Devuelve una lista de diccionarios. La columna «Antecedentes Observaciones»
    no llega a existir en la estructura devuelta: se elimina por índice antes de
    construir cada fila.
    """
    if not XLSX.exists():
        sys.exit(f"No está el archivo fuente: {XLSX}")

    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    ws = wb[HOJA]
    it = ws.iter_rows(values_only=True)
    encabezado = list(next(it))

    # Índices que se conservan: todo menos lo prohibido.
    conservar = [i for i, h in enumerate(encabezado) if h not in COLUMNAS_PROHIBIDAS]
    descartadas = [h for h in encabezado if h in COLUMNAS_PROHIBIDAS]
    print(f"[privacidad] columnas descartadas en la lectura: {descartadas}")

    nombres = [encabezado[i] for i in conservar]
    filas = []
    for fila in it:
        if fila is None or all(v is None for v in fila):
            continue
        filas.append({nombres[j]: fila[i] for j, i in enumerate(conservar)})
    wb.close()
    print(f"[lectura] {len(filas)} eventos leídos de {XLSX.name}")
    return filas


# --------------------------------------------------------------------------
# Clasificación
# --------------------------------------------------------------------------
def calza(texto, patrones):
    return any(re.search(p, texto) for p in patrones)


def clasificar(filas):
    """Añade a cada fila los campos derivados del análisis.

    _cl,_ti,_s1,_s2  taxonomía normalizada
    _modo            modo de falla canónico (clave)
    _nivel           en qué nivel de la jerarquía se detectó el modo
    _causa           causa declarada, o None
    _anio,_mes       tiempo
    """
    for r in filas:
        r["_cl"] = norm(r.get("Clase Evento"))
        r["_ti"] = norm(r.get("Tipo Evento"))
        r["_s1"] = norm(r.get("Sub Evento 1"))
        r["_s2"] = norm(r.get("Sub Evento 2"))
        r["_todo"] = " | ".join([r["_cl"], r["_ti"], r["_s1"], r["_s2"]])

        f = str(r.get("Fecha Inicio") or "")
        r["_anio"] = int(f[:4]) if f[:4].isdigit() else None
        r["_mes"] = int(f[5:7]) if f[5:7].isdigit() else None

        # --- modo de falla ---------------------------------------------
        # Se busca en los cuatro niveles. Un mismo evento puede nombrar varios
        # modos (un frente que corta la luz Y el camino); nos quedamos con el
        # PRIMERO del diccionario que aparezca en el nivel MÁS ESPECÍFICO, para
        # que una falla anotada como sub-evento no se pierda dentro del fenómeno
        # que la causó.
        r["_modo"], r["_nivel"] = None, None
        for nivel, campo in (("sub2", "_s2"), ("sub1", "_s1"), ("tipo", "_ti"), ("clase", "_cl")):
            txt = r[campo]
            if not txt:
                continue
            for m in MODOS:
                if calza(txt, m["patrones"]):
                    # Preferimos el modo de INFRAESTRUCTURA aunque esté más abajo:
                    # es el objeto de este catálogo.
                    if r["_modo"] is None or (
                        m["categoria"] == "INFRAESTRUCTURA"
                        and MODO_POR_CLAVE[r["_modo"]]["categoria"] != "INFRAESTRUCTURA"
                    ):
                        r["_modo"], r["_nivel"] = m["clave"], nivel
                    break
            if r["_modo"] and MODO_POR_CLAVE[r["_modo"]]["categoria"] == "INFRAESTRUCTURA":
                break

        # --- causa declarada -------------------------------------------
        # Sólo existe si el modo se detectó en un sub-evento: entonces los
        # niveles padre nombran el fenómeno que lo produjo.
        r["_causa"] = None
        if r["_modo"] and r["_nivel"] in ("sub1", "sub2"):
            padre = (r["_cl"] + " | " + r["_ti"]) if r["_nivel"] == "sub1" else \
                    (r["_cl"] + " | " + r["_ti"] + " | " + r["_s1"])
            for nombre, pats in CAUSAS:
                if calza(padre, pats):
                    # Si la «causa» es el mismo fenómeno que el modo, no es causa:
                    # es la jerarquía repitiéndose. Se descarta.
                    if r["_modo"] not in CAUSA_ES_EL_MISMO_MODO.get(nombre, set()):
                        r["_causa"] = nombre
                    break
        # Caso adicional: el modo se detectó en «tipo» y la CLASE nombra un
        # fenómeno real (no un cajón administrativo). También es causa declarada.
        elif r["_modo"] and r["_nivel"] == "tipo" and r["_cl"]:
            if not calza(r["_cl"], CAJONES_SIN_CAUSA):
                for nombre, pats in CAUSAS:
                    if calza(r["_cl"], pats):
                        if r["_modo"] not in CAUSA_ES_EL_MISMO_MODO.get(nombre, set()):
                            r["_causa"] = nombre
                        break
    return filas


MODO_POR_CLAVE = {m["clave"]: m for m in MODOS}


# --------------------------------------------------------------------------
# Métricas por modo
# --------------------------------------------------------------------------
def tendencia(cuentas_por_anio, anios):
    """Pendiente lineal (eventos/año) y variación entre el primer y el último trienio.

    Se reporta con la advertencia de que el vocabulario del registro cambió: una
    tendencia aquí puede ser un cambio de práctica de anotación y no del mundo.
    """
    y = np.array([cuentas_por_anio.get(a, 0) for a in anios], dtype=float)
    x = np.arange(len(anios), dtype=float)
    pend = float(np.polyfit(x, y, 1)[0]) if len(anios) > 1 else 0.0
    ini, fin = y[:3].sum(), y[-3:].sum()
    var = (100.0 * (fin - ini) / ini) if ini > 0 else float("nan")
    return pend, var


def resumen_por_modo(filas):
    anios = list(range(2015, 2025))

    # Línea base estacional de TODO el registro: sin esto, decir «el 30 % de los
    # cortes son de invierno» no significa nada, porque el registro entero podría
    # tener más eventos en invierno.
    base_inv = sum(1 for r in filas if r["_mes"] in INVIERNO)
    base_ver = sum(1 for r in filas if r["_mes"] in VERANO)
    base_ratio = base_inv / base_ver if base_ver else float("nan")

    filas_por_modo = defaultdict(list)
    for r in filas:
        filas_por_modo[r["_modo"]].append(r)

    salida = []
    for m in MODOS:
        rs = filas_por_modo.get(m["clave"], [])
        if not rs:
            continue
        n = len(rs)
        meses = Counter(r["_mes"] for r in rs if r["_mes"])
        n_inv = sum(meses.get(k, 0) for k in INVIERNO)
        n_ver = sum(meses.get(k, 0) for k in VERANO)
        ratio = (n_inv / n_ver) if n_ver else float("inf")
        # Índice estacional: cuántas veces más invernal es este modo que el
        # registro completo. 1,0 = igual que el promedio nacional de emergencias.
        indice = (ratio / base_ratio) if base_ratio and np.isfinite(ratio) else float("nan")

        por_anio = Counter(r["_anio"] for r in rs if r["_anio"])
        pend, var = tendencia(por_anio, anios)

        causas = Counter(r["_causa"] for r in rs if r["_causa"])
        n_causa = sum(causas.values())

        regiones = Counter(str(r.get("Región") or "").strip() for r in rs)
        comunas = Counter(str(r.get("Comuna") or "").strip() for r in rs)
        origen = Counter(norm(r.get("Origen Evento")) for r in rs)

        # ¿Hay alguna palabra de vector deliberado en estos eventos?
        n_delib = sum(1 for r in rs if calza(r["_todo"], PALABRAS_DELIBERADO))

        # --- clasificación del vector ----------------------------------
        # Sólo se rotula NATURAL si el registro DECLARA causas naturales en
        # cantidad suficiente. Nunca se rotula DELIBERADO, porque el registro no
        # tiene ninguna columna de intención (ver informe A).
        n_meteo = sum(v for k, v in causas.items() if k in CAUSAS_METEO)
        if n_causa < MIN_CAUSAS_PARA_ROTULAR:
            vector = "NO DISTINGUIBLE (causa declarada en muy pocos eventos)"
        elif n_meteo >= 0.5 * n_causa:
            vector = "NATURAL (mayoría de las causas declaradas es meteorológica)"
        elif n_meteo > 0:
            vector = "MIXTO NATURAL/ACCIDENTAL (nunca deliberado: el registro no lo anota)"
        else:
            vector = "NO DISTINGUIBLE (el registro no anota intención)"

        salida.append(dict(
            modo=m["clave"],
            nombre=m["nombre"],
            categoria=m["categoria"],
            n_eventos=n,
            pct_del_registro=round(100.0 * n / len(filas), 2),
            vector=vector,
            n_menciones_deliberado=n_delib,
            origen_natural=origen.get("natural", 0),
            origen_antropico=origen.get("antropico", 0),
            n_con_causa_declarada=n_causa,
            pct_con_causa_declarada=round(100.0 * n_causa / n, 1),
            causas_declaradas_top3="; ".join(f"{k} ({v})" for k, v in causas.most_common(3)) or "—",
            mes_pico=max(meses, key=meses.get) if meses else "",
            n_invierno_JJA=n_inv,
            n_verano_DEF=n_ver,
            razon_invierno_verano=round(ratio, 2) if np.isfinite(ratio) else "inf",
            indice_estacional=round(indice, 2) if np.isfinite(indice) else "",
            **{f"mes_{k:02d}": meses.get(k, 0) for k in range(1, 13)},
            **{f"n_{a}": por_anio.get(a, 0) for a in anios},
            pendiente_eventos_por_anio=round(pend, 1),
            var_pct_trienio_ini_vs_fin=(round(var, 1) if np.isfinite(var) else ""),
            top3_regiones="; ".join(f"{k} ({v})" for k, v in regiones.most_common(3)),
            top5_comunas="; ".join(f"{k} ({v})" for k, v in comunas.most_common(5)),
            personas_afectadas=int(sum(num(r.get("Total Afectados")) for r in rs)),
            personas_aisladas=int(sum(num(r.get("Total Aislados")) for r in rs)),
            propiedades_expuestas=m["propiedades"],
            nota_propiedad=m["nota"],
        ))

    salida.sort(key=lambda d: -d["n_eventos"])
    return salida, base_inv, base_ver, base_ratio


# --------------------------------------------------------------------------
# Informes auxiliares que se imprimen en consola (insumo del .md)
# --------------------------------------------------------------------------
def informes_extra(filas):
    print("\n" + "=" * 74)
    print("A · ¿EL REGISTRO DISTINGUE VECTORES DELIBERADOS?")
    print("=" * 74)
    total_delib = 0
    for p in PALABRAS_DELIBERADO:
        n = sum(1 for r in filas if re.search(p, r["_todo"]))
        if n:
            print(f"   patrón «{p}»: {n} eventos")
            total_delib += n
    print(f"   TOTAL de eventos con alguna marca de intención: {total_delib} "
          f"de {len(filas)} ({100.0*total_delib/len(filas):.4f} %)")
    og = Counter(norm(r.get("Origen Evento")) for r in filas)
    print(f"   Columna «Origen Evento»: {dict(og)}")
    print("   OJO: «Antrópico» significa ORIGEN HUMANO, no INTENCIÓN. Incluye "
          "incendios accidentales y choques.")

    print("\n" + "=" * 74)
    print("B · AISLAMIENTO DE PERSONAS (columna numérica Total Aislados)")
    print("=" * 74)
    ais = [r for r in filas if num(r.get("Total Aislados")) > 0]
    meses = Counter(r["_mes"] for r in ais)
    print(f"   {len(ais)} eventos dejaron personas aisladas; "
          f"{int(sum(num(r.get('Total Aislados')) for r in ais))} personas en total")
    print("   por mes: " + ", ".join(f"{k:02d}:{meses.get(k,0)}" for k in range(1, 13)))
    n_inv = sum(meses.get(k, 0) for k in INVIERNO)
    n_ver = sum(meses.get(k, 0) for k in VERANO)
    print(f"   invierno (JJA)={n_inv}  verano (DEF)={n_ver}  razón={n_inv/max(n_ver,1):.2f}")
    print("   causa (Clase Evento): " + "; ".join(
        f"{k} ({v})" for k, v in Counter(r["_cl"] for r in ais).most_common(6)))
    print("   comunas: " + "; ".join(
        f"{str(r):s}" for r in [", ".join(f"{k} ({v})" for k, v in
                                Counter(str(r.get('Comuna') or '') for r in ais).most_common(8))]))

    print("\n" + "=" * 74)
    print("C · CAUSA DECLARADA DE LOS CORTES ELÉCTRICOS (los que la declaran)")
    print("=" * 74)
    el = [r for r in filas if r["_modo"] == "ELEC_SUMINISTRO"]
    con = [r for r in el if r["_causa"]]
    print(f"   {len(el)} cortes eléctricos; {len(con)} con causa declarada "
          f"({100.0*len(con)/len(el):.1f} %)")
    for k, v in Counter(r["_causa"] for r in con).most_common(12):
        print(f"      {v:5d}  {k}")
    print(f"   {len(el)-len(con)} cortes ({100.0*(len(el)-len(con))/len(el):.1f} %) "
          f"quedan SIN causa en ninguna columna estructurada.")

    print("\n" + "=" * 74)
    print("D · CORTES ELÉCTRICOS CON CAUSA DECLARADA: ¿invierno o verano?")
    print("=" * 74)
    meses_con = Counter(r["_mes"] for r in con)
    meses_sin = Counter(r["_mes"] for r in el if not r["_causa"])
    print("   con causa: " + ", ".join(f"{k:02d}:{meses_con.get(k,0)}" for k in range(1, 13)))
    print("   sin causa: " + ", ".join(f"{k:02d}:{meses_sin.get(k,0)}" for k in range(1, 13)))
    ci = sum(meses_con.get(k, 0) for k in INVIERNO); cv = sum(meses_con.get(k, 0) for k in VERANO)
    si = sum(meses_sin.get(k, 0) for k in INVIERNO); sv = sum(meses_sin.get(k, 0) for k in VERANO)
    print(f"   con causa  invierno={ci} verano={cv} razón={ci/max(cv,1):.2f}")
    print(f"   sin causa  invierno={si} verano={sv} razón={si/max(sv,1):.2f}")

    print("\n" + "=" * 74)
    print("E · USO DEL SUB-EVENTO POR AÑO (la práctica de anotar el encadenamiento)")
    print("=" * 74)
    for a in range(2015, 2025):
        rs = [r for r in filas if r["_anio"] == a]
        cs = sum(1 for r in rs if r["_s1"])
        print(f"   {a}  total={len(rs):5d}  con Sub Evento 1={cs:5d}  ({100.0*cs/len(rs):.1f} %)")

    print("\n" + "=" * 74)
    print("F · REGIONES: eventos de infraestructura por región")
    print("=" * 74)
    infra = [r for r in filas
             if r["_modo"] and MODO_POR_CLAVE[r["_modo"]]["categoria"] == "INFRAESTRUCTURA"]
    for k, v in Counter(str(r.get("Región") or "").strip() for r in infra).most_common(20):
        print(f"      {v:6d}  {k}")

    print("\n" + "=" * 74)
    print("G · SIN CLASIFICAR (control de calidad del diccionario)")
    print("=" * 74)
    sin = [r for r in filas if not r["_modo"]]
    print(f"   {len(sin)} eventos ({100.0*len(sin)/len(filas):.2f} %) no calzaron ningún modo")
    for k, v in Counter(r["_todo"] for r in sin).most_common(10):
        print(f"      {v:5d}  {k[:90]}")

    # ----------------------------------------------------------------------
    # H · LA PREGUNTA CENTRAL: ¿QUÉ FALLA EN INVIERNO Y NO EN VERANO?
    # ----------------------------------------------------------------------
    # El truco metodológico: NO basta con mirar la estacionalidad de un modo de
    # falla, porque «corte eléctrico» mezcla dos poblaciones distintas. Hay que
    # partirlo por CAUSA DECLARADA en tres grupos y comparar:
    #
    #   (1) causa meteorológica declarada → si el clima mueve el riesgo, aquí
    #       tiene que verse;
    #   (2) causa NO meteorológica declarada (choque, incendio, sismo) → es el
    #       CONTROL: si aquí también hubiera exceso invernal, el exceso sería un
    #       artefacto del registro (operadores que anotan más en invierno) y no
    #       un hecho del mundo;
    #   (3) sin causa declarada → la masa del registro.
    #
    # Sin el grupo (2) el resultado sería circular: obviamente un «sistema
    # frontal» ocurre en invierno. El grupo (2) es lo que vuelve falsable la
    # afirmación del proyecto.
    print("\n" + "=" * 74)
    print("H · ★ ¿QUÉ FALLA EN INVIERNO Y NO EN VERANO?  (con control no circular)")
    print("=" * 74)

    def estacion(rs):
        c = Counter(r["_mes"] for r in rs)
        i = sum(c.get(k, 0) for k in INVIERNO)
        v = sum(c.get(k, 0) for k in VERANO)
        return len(rs), i, v, (i / v if v else float("inf")), [c.get(k, 0) for k in range(1, 13)]

    infra = [r for r in filas
             if r["_modo"] and MODO_POR_CLAVE[r["_modo"]]["categoria"] == "INFRAESTRUCTURA"]
    grupos = [
        ("(1) causa METEOROLÓGICA", [r for r in infra if r["_causa"] in CAUSAS_METEO]),
        ("(2) causa NO meteo [CONTROL]", [r for r in infra if r["_causa"] and r["_causa"] not in CAUSAS_METEO]),
        ("(3) sin causa declarada", [r for r in infra if not r["_causa"]]),
    ]
    print(f"   Universo: {len(infra)} eventos de falla de INFRAESTRUCTURA\n")
    print(f"   {'grupo':<30} {'n':>6} {'inv':>6} {'ver':>6} {'inv/ver':>8}")
    for etiqueta, g in grupos:
        n, i, v, rt, _ = estacion(g)
        print(f"   {etiqueta:<30} {n:6d} {i:6d} {v:6d} {rt:8.2f}")
    print(f"   {'LÍNEA BASE (registro entero)':<30} {len(filas):6d} "
          f"{sum(1 for r in filas if r['_mes'] in INVIERNO):6d} "
          f"{sum(1 for r in filas if r['_mes'] in VERANO):6d} "
          f"{base_ratio_global(filas):8.2f}")
    print("\n   LECTURA: si (1) es alto y (2) queda pegado a la línea base, el exceso")
    print("   invernal es del CLIMA y no del registro. Si (2) también subiera, sería")
    print("   un artefacto de anotación y la afirmación del proyecto no se sostendría.")

    print("\n   Desglose por modo de infraestructura (razón invierno/verano):")
    print(f"   {'modo':<34} {'meteo':>16} {'NO meteo (ctrl)':>18} {'sin causa':>16}")
    for m in MODOS:
        if m["categoria"] != "INFRAESTRUCTURA":
            continue
        rs = [r for r in infra if r["_modo"] == m["clave"]]
        if not rs:
            continue
        mt = [r for r in rs if r["_causa"] in CAUSAS_METEO]
        nm = [r for r in rs if r["_causa"] and r["_causa"] not in CAUSAS_METEO]
        sc = [r for r in rs if not r["_causa"]]
        def fmt(g):
            n, i, v, rt, _ = estacion(g)
            if n == 0:
                return "—"
            return f"{rt:.2f} (n={n})" if v else f"inf (n={n})"
        print(f"   {m['clave']:<34} {fmt(mt):>16} {fmt(nm):>18} {fmt(sc):>16}")

    print("\n   Curva mensual de la falla de infraestructura con causa meteorológica:")
    mt = [r for r in infra if r["_causa"] in CAUSAS_METEO]
    c = Counter(r["_mes"] for r in mt)
    for k in range(1, 13):
        n = c.get(k, 0)
        print(f"      mes {k:02d}  {n:4d}  {'█' * int(n / 5)}")

    # ----------------------------------------------------------------------
    # I · Dos anomalías que el registro muestra pero no explica
    # ----------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("I · ANOMALÍAS: concentraciones que el registro NO etiqueta")
    print("=" * 74)
    tel = [r for r in filas if r["_modo"] == "TELECOM"]
    top = Counter((r["_anio"], r["_mes"]) for r in tel).most_common(3)
    print(f"   TELECOM: {len(tel)} fallas en 10 años; los 3 meses más cargados son {top}")
    print(f"      → el mes peor concentra el {100.0*top[0][1]/len(tel):.1f} % de la década.")

    cv = [r for r in filas if r["_modo"] == "CONECTIVIDAD_VIAL"]
    print("\n   CONECTIVIDAD VIAL, octubre 2019 día a día (el registro no anota intención):")
    dia = Counter(str(r["Fecha Inicio"])[:10] for r in cv
                  if r["_anio"] == 2019 and r["_mes"] == 10)
    for d in sorted(dia):
        print(f"      {d}  {dia[d]:3d}  {'▇' * dia[d]}")
    ant = sum(dia[d] for d in dia if d <= "2019-10-18")
    des = sum(dia[d] for d in dia if d > "2019-10-18")
    print(f"      hasta el 18-oct: {ant} eventos · desde el 19-oct: {des} eventos")
    print("      El registro los clasifica como «Falla de Conectividad Vial | Accidente»,")
    print("      Origen «Antrópico», SIN causa. La taxonomía no tiene la palabra.")


# --------------------------------------------------------------------------
def main():
    filas = leer_eventos()
    filas = clasificar(filas)
    tabla, base_inv, base_ver, base_ratio = resumen_por_modo(filas)

    print("\n" + "=" * 74)
    print("LÍNEA BASE ESTACIONAL DEL REGISTRO COMPLETO")
    print("=" * 74)
    print(f"   invierno (jun-jul-ago) = {base_inv}   verano (dic-ene-feb) = {base_ver}   "
          f"razón = {base_ratio:.3f}")
    print("   Todo índice estacional de la tabla se lee contra esta razón: 1,00 = "
          "tan invernal como el registro entero.")

    print("\n" + "=" * 74)
    print("CATÁLOGO DE MODOS DE FALLA, POR FRECUENCIA")
    print("=" * 74)
    print(f"{'n':>7}  {'%':>5}  {'inv/ver':>7}  {'índ':>5}  {'%causa':>6}  modo")
    for d in tabla:
        print(f"{d['n_eventos']:7d}  {d['pct_del_registro']:5.2f}  "
              f"{str(d['razon_invierno_verano']):>7}  {str(d['indice_estacional']):>5}  "
              f"{d['pct_con_causa_declarada']:6.1f}  {d['nombre'][:52]}")

    informes_extra(filas)

    SALIDA_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tabla[0].keys()))
        w.writeheader()
        w.writerows(tabla)
    print(f"\n[salida] {SALIDA_CSV}  ({len(tabla)} modos)")


if __name__ == "__main__":
    main()

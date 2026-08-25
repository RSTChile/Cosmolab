"""
Resumen del inventario georreferenciado: cuenta lo que hay, no lo que creemos
que hay.

POR QUÉ ESTO ES UN PROGRAMA Y NO UNA TABLA ESCRITA A MANO
----------------------------------------------------------
`INVENTARIO_GEORREFERENCIADO.md` lleva la tabla del estado del inventario. Esa
tabla podría escribirse a mano copiando los números que imprime cada adaptador
al terminar. Sería más rápido y estaría mal: en cuanto alguien vuelva a correr
un adaptador y la fuente haya cambiado, la tabla del documento pasa a mentir sin
que nadie se entere.

Este módulo abre cada archivo del inventario, cuenta **filas con coordenada
usable** —no filas totales, que es otra cosa— y arma la tabla. El documento se
regenera desde acá.

QUÉ CUENTA COMO «GEORREFERENCIADO»
-----------------------------------
Una fila cuenta si tiene latitud y longitud presentes, legibles como número, y
dentro de la caja de Chile (continental, insular y antártico). Nada más.
Una fila con la coordenada vacía NO cuenta, y una fila con coordenada fuera de
Chile TAMPOCO: se cuentan aparte y se muestran, porque un activo mal ubicado es
peor que un activo sin ubicar — el primero se cruza con la amenaza equivocada y
nadie lo nota.

LO QUE NO SE SUMA, Y POR QUÉ
-----------------------------
El total no es la suma de todos los archivos. Hay tres razones para excluir, y
cada exclusión va declarada en la propia tabla:

1. **Duplicación entre fuentes.** Los puentes del Ministerio de Obras Públicas
   (6.742) y los de SENAPRED (6.628) son mayormente los mismos puentes. Los
   establecimientos de salud del Departamento de Estadísticas e Información de
   Salud (5.717) y los de SENAPRED (5.159), también. Se cuenta la fuente que
   manda y la otra queda de respaldo, marcada como `duplicado`.
2. **Permiso versus activo.** Las 52.412 antenas «autorizadas» de SUBTEL no son
   fierros instalados: son actos administrativos. Cuentan las 29.875 «en
   servicio».
3. **Proyecto versus activo.** Los 797 contratos de servicios sanitarios rurales
   son obras en curso, no sistemas operando.

USO
---
    python3 adaptadores/inventario_resumen.py
    python3 adaptadores/inventario_resumen.py --markdown   # la tabla del .md
"""

import csv
import sys
from pathlib import Path

AQUI = Path(__file__).parent.parent
DATOS = AQUI / "datos"

# `estado` es lo que decide si la fila entra al total:
#   "cuenta"     → activo georreferenciado que suma
#   "duplicado"  → el mismo activo ya viene de una fuente que manda
#   "no_activo"  → permiso, proyecto o registro que no es un fierro instalado
#   "sin_geom"   → se conserva por sus atributos, pero no aporta ubicación
#
# El orden de la lista es el orden en que sale la tabla.
ARCHIVOS = [
    # ── lo que el proyecto YA tenía antes de esta tanda ──
    dict(archivo="mop_tramos.csv", lat="lat_punto_medio", lon="lon_punto_medio",
         sector="Transporte", que="Tramos de la red vial nacional",
         fuente="MOP · Dirección de Vialidad", fecha="2026-08-17",
         estado="cuenta", previo=True,
         nota="la coordenada es el punto medio del tramo; la traza completa "
              "está en el crudo"),
    dict(archivo="mop_puentes.csv", lat="lat", lon="lon",
         sector="Transporte", que="Puentes",
         fuente="MOP · Dirección de Vialidad", fecha="2026-08-17",
         estado="cuenta", previo=True,
         nota="trae el cauce que cruza cada puente"),
    dict(archivo="subestaciones_puntos.csv", lat="lat", lon="lon",
         sector="Energía", que="Subestaciones eléctricas (piloto a mano)",
         fuente="captura manual del proyecto", fecha="2026-08-15",
         estado="duplicado", previo=True,
         nota="reemplazadas por las del Coordinador; mediana de desviación "
              "3,0 km, máxima 175 km"),
    # ── Ministerio de Obras Públicas, resto del Sistema de Información Territorial ──
    dict(archivo="inventario_agua_potable_rural.csv", lat="lat", lon="lon",
         sector="Hídrico", que="Sistemas de agua potable rural",
         fuente="MOP · Dirección de Obras Hidráulicas", fecha="2026-08-19",
         estado="cuenta",
         nota="★ trae grupo electrógeno, camión aljibe y población servida"),
    dict(archivo="inventario_aeropuertos.csv", lat="lat", lon="lon",
         sector="Transporte", que="Aeropuertos y aeródromos",
         fuente="MOP · Dirección de Aeropuertos", fecha="2026-08-19",
         estado="cuenta",
         nota="★ trae ZONA_AISLADA y CATEGORIA_AISLAMIENTO"),
    dict(archivo="inventario_obras_portuarias.csv", lat="lat", lon="lon",
         sector="Transporte", que="Obras portuarias menores",
         fuente="MOP · Dirección de Obras Portuarias", fecha="2026-08-19",
         estado="cuenta", nota="caletas, muelles, varaderos y rampas"),
    dict(archivo="inventario_embalses.csv", lat="lat", lon="lon",
         sector="Hídrico", que="Embalses",
         fuente="MOP · Dirección de Obras Hidráulicas", fecha="2026-08-19",
         estado="cuenta",
         nota="★ el servicio declara el dato actualizado a dic-2015"),
    # ── Coordinador Eléctrico Nacional ──
    dict(archivo="inventario_subestaciones_electricas.csv", lat="lat", lon="lon",
         sector="Energía", que="Subestaciones eléctricas",
         fuente="Coordinador Eléctrico Nacional · Infotécnica",
         fecha="2026-08-19", estado="cuenta",
         nota="★ desbloqueó el cuello de botella: 1.226 de 1.273 con "
              "coordenada del propio operador"),
    dict(archivo="inventario_centrales_electricas.csv", lat="lat", lon="lon",
         sector="Energía", que="Centrales de generación",
         fuente="Coordinador Eléctrico Nacional · Infotécnica",
         fecha="2026-08-19", estado="cuenta",
         nota="con potencia máxima en megavatios"),
    dict(archivo="inventario_taps_electricos.csv", lat="lat", lon="lon",
         sector="Energía", que="Derivaciones de línea (taps)",
         fuente="Coordinador Eléctrico Nacional · Infotécnica",
         fecha="2026-08-19", estado="cuenta",
         nota="sólo 58 de 277 traen coordenada"),
    dict(archivo="inventario_lineas_transmision_SIN_GEOMETRIA.csv",
         lat="lat", lon="lon",
         sector="Energía", que="Tramos de línea de transmisión",
         fuente="Coordinador Eléctrico Nacional · Infotécnica",
         fecha="2026-08-19", estado="sin_geom",
         nota="★ 40.643 km SIN traza: el campo `coordenadas` viene vacío en "
              "los 2.926 tramos"),
    # ── Salud y educación ──
    dict(archivo="inventario_salud.csv", lat="lat", lon="lon",
         sector="Salud", que="Establecimientos de salud",
         fuente="MINSAL · DEIS (datos.gob.cl, CC-Zero)", fecha="2026-08-19",
         estado="cuenta",
         nota="★ única fuente del inventario con licencia limpia; se "
              "actualiza a diario"),
    dict(archivo="inventario_educacion.csv", lat="lat", lon="lon",
         sector="Educación", que="Establecimientos educacionales",
         fuente="MINEDUC · Centro de Estudios (CC-BY)", fecha="2026-08-19",
         estado="cuenta", solo_si={"en_funcionamiento": "si"},
         nota="sólo los que están en funcionamiento; no incluye jardines "
              "infantiles; el ministerio declara la coordenada «referencial»"),
    # ── SUBTEL ──
    dict(archivo="inventario_telecom_antenas_en_servicio.csv", lat="lat", lon="lon",
         sector="Telecomunicaciones", que="Elementos de antena EN SERVICIO",
         fuente="SUBTEL · Ley de Torres", fecha="2026-08-20", estado="cuenta",
         nota="★ el ítem de telecomunicaciones estaba sin poblar"),
    dict(archivo="inventario_telecom_antenas_autorizadas.csv", lat="lat", lon="lon",
         sector="Telecomunicaciones", que="Elementos de antena AUTORIZADOS",
         fuente="SUBTEL · Ley de Torres", fecha="2026-08-20",
         estado="no_activo",
         nota="es el permiso, no el fierro: no se suma con «en servicio»"),
    dict(archivo="inventario_telecom_estaciones_base.csv", lat="lat", lon="lon",
         sector="Telecomunicaciones", que="Estaciones base móviles",
         fuente="SUBTEL · concesiones de radiocomunicación móvil",
         fecha="2026-08-20", estado="cuenta",
         nota="registro distinto del de la Ley de Torres, corte jul-2025"),
    dict(archivo="inventario_telecom_red_conectividad.csv", lat="lat", lon="lon",
         sector="Telecomunicaciones", que="Tramos de conectividad digital",
         fuente="SUBTEL", fecha="2026-08-20", estado="cuenta",
         nota="NO es la red nacional de fibra óptica completa"),
    # ── SENAPRED · Sistema Integrado de Información para Emergencias ──
    dict(archivo="inventario_senapred_energia_puntual.csv", lat="lat", lon="lon",
         sector="Energía", que="Instalaciones eléctricas puntuales",
         fuente="SENAPRED · SIIE", fecha="2026-08-19", estado="cuenta",
         nota="universo distinto del Coordinador; comparar antes de fusionar"),
    dict(archivo="inventario_senapred_energia_lineal.csv", lat="lat", lon="lon",
         sector="Energía", que="Líneas eléctricas CON traza",
         fuente="SENAPRED · SIIE", fecha="2026-08-19", estado="cuenta",
         nota="★ lo que el Coordinador no publica: geometría de línea"),
    dict(archivo="inventario_senapred_telefonia.csv", lat="lat", lon="lon",
         sector="Telecomunicaciones", que="Puntos de telefonía",
         fuente="SENAPRED · SIIE", fecha="2026-08-19", estado="duplicado",
         nota="16.669 = exactamente la capa de antenas que republica el IGM; "
              "manda SUBTEL, que es el organismo dueño del dato"),
    dict(archivo="inventario_senapred_bomberos.csv", lat="lat", lon="lon",
         sector="Emergencia", que="Cuarteles de bomberos",
         fuente="SENAPRED · SIIE", fecha="2026-08-19", estado="cuenta",
         nota="infraestructura de RESPUESTA"),
    dict(archivo="inventario_senapred_carabineros.csv", lat="lat", lon="lon",
         sector="Emergencia", que="Unidades de Carabineros",
         fuente="SENAPRED · SIIE", fecha="2026-08-19", estado="cuenta"),
    dict(archivo="inventario_senapred_pdi.csv", lat="lat", lon="lon",
         sector="Emergencia", que="Unidades de la Policía de Investigaciones",
         fuente="SENAPRED · SIIE", fecha="2026-08-19", estado="cuenta"),
    dict(archivo="inventario_senapred_servicio_medico_legal.csv", lat="lat", lon="lon",
         sector="Emergencia", que="Servicio Médico Legal",
         fuente="SENAPRED · SIIE", fecha="2026-08-19", estado="cuenta"),
    dict(archivo="inventario_senapred_direcciones_regionales.csv", lat="lat", lon="lon",
         sector="Emergencia", que="Direcciones regionales de SENAPRED",
         fuente="SENAPRED · SIIE", fecha="2026-08-19", estado="cuenta",
         nota="el nodo de mando regional del sistema al que sirve el proyecto"),
    dict(archivo="inventario_senapred_siss.csv", lat="lat", lon="lon",
         sector="Hídrico", que="Infraestructura sanitaria (SISS)",
         fuente="SENAPRED · SIIE", fecha="2026-08-19", estado="cuenta"),
    dict(archivo="inventario_senapred_aguas.csv", lat="lat", lon="lon",
         sector="Hídrico", que="Infraestructura de agua",
         fuente="SENAPRED · SIIE", fecha="2026-08-19", estado="duplicado",
         nota="mismo conteo que la capa SISS (8.463): es la misma capa "
              "publicada en dos servicios"),
    dict(archivo="inventario_senapred_suministro_alternativo_agua.csv",
         lat="lat", lon="lon",
         sector="Hídrico", que="Puntos de suministro alternativo de agua",
         fuente="SENAPRED · SIIE", fecha="2026-08-19", estado="cuenta",
         nota="★ el plan B del agua potable rural"),
    dict(archivo="inventario_senapred_educacion.csv", lat="lat", lon="lon",
         sector="Educación", que="Establecimientos educacionales",
         fuente="SENAPRED · SIIE", fecha="2026-08-19", estado="duplicado",
         nota="manda MINEDUC. Aporta los ~4.300 jardines infantiles que "
              "MINEDUC no tiene: extraerlos es tarea pendiente"),
    dict(archivo="inventario_senapred_recintos_deportivos.csv", lat="lat", lon="lon",
         sector="Emergencia", que="Recintos deportivos",
         fuente="SENAPRED · SIIE", fecha="2026-08-19", estado="cuenta",
         nota="los albergues de hecho"),
    dict(archivo="inventario_senapred_municipios.csv", lat="lat", lon="lon",
         sector="Gobierno", que="Municipalidades",
         fuente="SENAPRED · SIIE", fecha="2026-08-19", estado="cuenta",
         nota="el nivel Comunal del COGRID"),
    dict(archivo="inventario_senapred_gobernaciones.csv", lat="lat", lon="lon",
         sector="Gobierno", que="Sedes provinciales de gobierno",
         fuente="SENAPRED · SIIE", fecha="2026-08-19", estado="cuenta",
         nota="★ nombre institucional vencido: hoy son Delegaciones "
              "Presidenciales Provinciales"),
    dict(archivo="inventario_senapred_intendencias.csv", lat="lat", lon="lon",
         sector="Gobierno", que="Sedes regionales de gobierno",
         fuente="SENAPRED · SIIE", fecha="2026-08-19", estado="cuenta",
         nota="★ nombre institucional vencido"),
    dict(archivo="inventario_senapred_edificios_publicos.csv", lat="lat", lon="lon",
         sector="Gobierno", que="Edificios públicos",
         fuente="SENAPRED · SIIE", fecha="2026-08-19", estado="cuenta"),
    dict(archivo="inventario_senapred_centros_publicos.csv", lat="lat", lon="lon",
         sector="Gobierno", que="Centros públicos",
         fuente="SENAPRED · SIIE", fecha="2026-08-19", estado="cuenta"),
    dict(archivo="inventario_senapred_penitenciarios.csv", lat="lat", lon="lon",
         sector="Social", que="Centros penitenciarios",
         fuente="SENAPRED · SIIE", fecha="2026-08-19", estado="cuenta",
         nota="población que no evacúa por sus propios medios"),
    dict(archivo="inventario_senapred_mejor_ninez.csv", lat="lat", lon="lon",
         sector="Social", que="Residencias de protección de la infancia",
         fuente="SENAPRED · SIIE", fecha="2026-08-19", estado="cuenta",
         nota="población dependiente"),
    dict(archivo="inventario_senapred_senama.csv", lat="lat", lon="lon",
         sector="Social", que="Establecimientos de adulto mayor",
         fuente="SENAPRED · SIIE", fecha="2026-08-19", estado="cuenta",
         nota="población dependiente"),
    dict(archivo="inventario_senapred_sedes_universitarias.csv", lat="lat", lon="lon",
         sector="Educación", que="Sedes universitarias",
         fuente="SENAPRED · SIIE", fecha="2026-08-19", estado="cuenta"),
    dict(archivo="inventario_senapred_supermercados.csv", lat="lat", lon="lon",
         sector="Abastecimiento", que="Supermercados",
         fuente="SENAPRED · SIIE", fecha="2026-08-19", estado="cuenta",
         nota="abastecimiento de alimento en emergencia"),
    dict(archivo="inventario_senapred_puertos.csv", lat="lat", lon="lon",
         sector="Transporte", que="Puertos",
         fuente="SENAPRED · SIIE", fecha="2026-08-19", estado="cuenta"),
    dict(archivo="inventario_senapred_comunicacion_aerea.csv", lat="lat", lon="lon",
         sector="Transporte", que="Instalaciones de comunicación aérea",
         fuente="SENAPRED · SIIE", fecha="2026-08-19", estado="cuenta"),
    dict(archivo="inventario_senapred_pasos_fronterizos.csv", lat="lat", lon="lon",
         sector="Transporte", que="Pasos fronterizos",
         fuente="SENAPRED · SIIE", fecha="2026-08-19", estado="cuenta"),
]


def _numero(valor):
    if valor is None or valor == "":
        return None
    try:
        n = float(str(valor).replace(",", "."))
    except ValueError:
        return None
    return n if n == n else None


def _en_chile(lat, lon):
    return (lat is not None and lon is not None
            and -90.0 <= lat <= -17.0 and -110.0 <= lon <= -66.0)


def contar(entrada):
    """Abre un archivo del inventario y devuelve sus tres números: filas
    totales, filas con coordenada usable, y filas cuya coordenada cae fuera de
    Chile. Los tres se informan; sólo el del medio suma."""
    ruta = DATOS / entrada["archivo"]
    if not ruta.exists():
        return dict(entrada, existe=False, filas=0, ubicadas=0, fuera=0)
    filas = ubicadas = fuera = 0
    with ruta.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            filtro = entrada.get("solo_si") or {}
            if any((r.get(k) or "") != v for k, v in filtro.items()):
                continue
            filas += 1
            lat = _numero(r.get(entrada["lat"]))
            lon = _numero(r.get(entrada["lon"]))
            if lat is None or lon is None:
                continue
            if _en_chile(lat, lon):
                ubicadas += 1
            else:
                fuera += 1
    return dict(entrada, existe=True, filas=filas, ubicadas=ubicadas, fuera=fuera)


def resumen():
    return [contar(e) for e in ARCHIVOS]


def imprimir(filas):
    print(f"{'archivo':52s} {'filas':>8} {'ubicadas':>9} {'fuera':>6}  estado")
    print("-" * 92)
    for f in filas:
        if not f["existe"]:
            print(f"{f['archivo']:52s} {'—':>8} {'—':>9} {'—':>6}  NO EXISTE")
            continue
        print(f"{f['archivo']:52s} {f['filas']:8,} {f['ubicadas']:9,} "
              f"{f['fuera']:6,}  {f['estado']}")
    total = sum(f["ubicadas"] for f in filas if f["estado"] == "cuenta")
    respaldo = sum(f["ubicadas"] for f in filas if f["estado"] == "duplicado")
    no_activo = sum(f["ubicadas"] for f in filas if f["estado"] == "no_activo")
    sin_geom = sum(f["filas"] for f in filas if f["estado"] == "sin_geom")
    print("-" * 92)
    print(f"TOTAL GEORREFERENCIADO QUE CUENTA .................. {total:,}")
    print(f"   además, de respaldo (mismo activo, otra fuente) . {respaldo:,}")
    print(f"   además, permisos y proyectos (no son activos) ... {no_activo:,}")
    print(f"   además, sin geometría (se conservan por atributo) {sin_geom:,}")
    return total


def markdown(filas):
    """La tabla tal como va en INVENTARIO_GEORREFERENCIADO.md."""
    lineas = ["| Sector | Qué | Georreferenciados | Fuente | Fecha | Nota |",
              "|---|---|---:|---|---|---|"]
    for f in filas:
        if not f["existe"] or f["estado"] != "cuenta":
            continue
        nota = f.get("nota", "")
        lineas.append(f"| {f['sector']} | {f['que']} | **{f['ubicadas']:,}** | "
                      f"{f['fuente']} | {f['fecha']} | {nota} |")
    total = sum(f["ubicadas"] for f in filas if f["estado"] == "cuenta")
    lineas.append(f"| | **TOTAL** | **{total:,}** | | | |")
    return "\n".join(lineas).replace(",", ".")


def markdown_excluidos(filas):
    lineas = ["| Qué | Filas | Por qué NO se suma |", "|---|---:|---|"]
    for f in filas:
        if not f["existe"] or f["estado"] == "cuenta":
            continue
        lineas.append(f"| {f['que']} ({f['fuente']}) | {f['ubicadas']:,} | "
                      f"{f.get('nota', '')} |")
    return "\n".join(lineas).replace(",", ".")


if __name__ == "__main__":
    filas = resumen()
    if "--markdown" in sys.argv:
        print(markdown(filas))
        print()
        print(markdown_excluidos(filas))
    else:
        imprimir(filas)

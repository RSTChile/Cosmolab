"""
Trae las capas de SERNAGEOMIN y arranca el archivo histórico de la minuta.

★ LO URGENTE — POR QUÉ ESTE SCRIPT SE CORRE TODOS LOS DÍAS
-----------------------------------------------------------
La Minuta Técnica de peligro de remoción en masa **se sobrescribe a sí misma**.
No hay archivo histórico: la capa de hoy pisa la de ayer, y el campo de fecha de
emisión viene vacío. Es un pizarrón, no un cuaderno.

Eso significa que **cada día que pasa sin guardar una copia es un día de historia
perdido para siempre**. Y sin historia no se puede validar ningún modelo contra
lo que la fuente decía en su momento.

Este script guarda un snapshot fechado cada vez que corre. Empezar hoy es lo más
barato y lo más valioso que puede hacer el proyecto.

QUÉ TRAE
--------
1. `zonas morfoclimáticas` (119 polígonos) — la geografía en la que Chile declara
   la amenaza. Resuelve el problema de las dos geografías por el lado difícil.
2. `COMUNAS_2020` (345 comunas con su CUT) — la geografía administrativa.
   Resuelve el hallazgo H-13, y viene del MISMO servicio que la capa de amenaza,
   así que las dos calzan por construcción.
3. La **minuta vigente**, con el nivel de peligro por zona → snapshot fechado.
4. `ReTeRM` (380 eventos reales de remoción en masa, 1996-2026, con comuna, tipo
   y detonante) — el conjunto de validación que le faltaba al proyecto.

Fuente verificada el 15-ago-2026: servicio ArcGIS público, acceso anónimo.
"""

import json
import urllib.request
import urllib.error
from datetime import date
from pathlib import Path

AQUI = Path(__file__).parent
CAPAS = AQUI / "datos" / "capas"
CRUDO = AQUI / "datos" / "crudo" / "sernageomin"

SERVICIO = ("https://services1.arcgis.com/OyjvVdFTl5hfSdX3/arcgis/rest/"
            "services/minutasATG_Flash/FeatureServer")

# capa → (nombre de archivo, para qué sirve)
CAPAS_A_TRAER = {
    2: ("zonas_geograficas", "zonas morfoclimáticas: la geografía de la amenaza"),
    3: ("comunas", "COMUNAS_2020 con CUT: la geografía administrativa"),
    0: ("reterm_eventos", "eventos reales de remoción en masa 1996-2026"),
}

PAGINA = 1000     # el servicio pagina; se pide de a mil por las dudas

# ★ GENERALIZACIÓN DE GEOMETRÍA — decisión tomada el 15-ago-2026 tras un intento
# fallido. Pedida con todo el detalle, la capa de zonas pesó 103 MB y la de
# comunas no terminó de bajar en 15 minutos. Para lo que hace falta —saber en
# qué comuna cae una subestación— ese detalle no aporta nada: la diferencia
# entre un borde comunal dibujado al metro y dibujado a 200 m no cambia jamás
# en qué comuna está una subestación, salvo que esté justo sobre el límite, y
# en ese caso el problema es otro.
# `maxAllowableOffset` va en grados: 0,002° ≈ 200 m en esta latitud.
# El crudo de máximo detalle queda igual archivado cuando se logra bajar.
TOLERANCIA_GRADOS = 0.002


def traer_capa(n_capa, timeout=120, tolerancia=TOLERANCIA_GRADOS):
    """Descarga una capa completa como GeoJSON, paginando si hace falta.

    Devuelve (geojson, None) o (None, motivo). Nunca lanza: un fallo de red no
    debe tumbar la corrida entera, tiene que quedar anotado como hueco.
    """
    rasgos = []
    offset = 0
    while True:
        url = (f"{SERVICIO}/{n_capa}/query?where=1%3D1&outFields=*"
               f"&outSR=4326&f=geojson&resultOffset={offset}"
               f"&resultRecordCount={PAGINA}"
               f"&maxAllowableOffset={tolerancia}")
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                datos = json.load(r)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                json.JSONDecodeError) as e:
            return None, f"{type(e).__name__}: {str(e)[:120]}"

        if "error" in datos:
            return None, f"el servicio devolvió error: {datos['error']}"

        lote = datos.get("features", [])
        rasgos.extend(lote)
        if len(lote) < PAGINA:
            break
        offset += PAGINA

    return {"type": "FeatureCollection", "features": rasgos}, None


def resumir_minuta(geojson):
    """Cuenta cuántas zonas tienen cada nivel de peligro declarado.

    Importa una distinción que es fácil pasar por alto: una zona SIN nivel no es
    una zona de peligro bajo. Significa que no hay minuta vigente para ella. Son
    dos cosas distintas y confundirlas sería inventar tranquilidad.
    """
    conteo = {}
    for r in geojson["features"]:
        props = r.get("properties", {})
        nivel = None
        for campo in ("POS_OCURRENCIA", "pos_ocurrencia", "NIVEL", "nivel"):
            if props.get(campo):
                nivel = str(props[campo]).strip()
                break
        conteo[nivel or "SIN NIVEL VIGENTE"] = conteo.get(
            nivel or "SIN NIVEL VIGENTE", 0) + 1
    return conteo


def traer_solo_minuta():
    """Snapshot diario LIVIANO: sólo los atributos, sin geometría.

    Por qué así: la geometría de las 119 zonas no cambia — son las mismas
    montañas todos los días. Lo único que cambia es el NIVEL de peligro. Bajar
    4 MB de polígonos dos veces al día para capturar un puñado de etiquetas
    serían 3 GB al año de lo mismo repetido.

    Este modo pide sólo los atributos: unos pocos KB por foto. Así se puede
    correr varias veces al día sin costo, que es lo que hace falta porque la
    minuta se emite POR EVENTO y no por calendario: puede cambiar y volver
    atrás dentro del mismo día.
    """
    url = (f"{SERVICIO}/2/query?where=1%3D1"
           f"&outFields=OBJECTID,ZONA,REGION,FECHA,POS_OCURRENCIA"
           f"&returnGeometry=false&f=json")
    try:
        with urllib.request.urlopen(url, timeout=90) as r:
            datos = json.load(r)
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:120]}"
    if "error" in datos:
        return None, f"el servicio devolvió error: {datos['error']}"
    return datos, None


def snapshot_minuta():
    """Guarda una foto fechada de la minuta y avisa si cambió respecto de la
    anterior. Devuelve 0 si salió bien."""
    from datetime import datetime
    sello = datetime.now().strftime("%Y-%m-%dT%H%M")
    carpeta = CRUDO / "minuta_diaria"
    carpeta.mkdir(parents=True, exist_ok=True)

    datos, motivo = traer_solo_minuta()
    if datos is None:
        # Un fallo se ANOTA, no se silencia: mañana hay que saber que hoy no
        # hubo dato, y distinguirlo de «hoy no había peligro».
        (carpeta / f"{sello}.FALLO.txt").write_text(motivo, encoding="utf-8")
        print(f"  ✗ {sello}: {motivo}")
        return 1

    ruta = carpeta / f"{sello}.json"
    ruta.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")

    niveles = {}
    for r in datos.get("features", []):
        a = r.get("attributes", {})
        n = a.get("POS_OCURRENCIA") or "SIN NIVEL VIGENTE"
        niveles[n] = niveles.get(n, 0) + 1
    resumen_hoy = ", ".join(f"{v}×{k}" for k, v in sorted(niveles.items()))

    # ¿cambió respecto de la foto anterior? es lo único que vale la pena mirar
    previas = sorted(p for p in carpeta.glob("*.json") if p != ruta)
    cambio = "primera foto"
    if previas:
        anterior = json.loads(previas[-1].read_text(encoding="utf-8"))
        clave = lambda d: sorted(
            (r["attributes"].get("OBJECTID"),
             r["attributes"].get("POS_OCURRENCIA"))
            for r in d.get("features", []))
        cambio = "CAMBIÓ ★" if clave(anterior) != clave(datos) else "sin cambios"

    print(f"  ✓ {sello} · {len(datos.get('features', []))} zonas · {cambio}")
    print(f"    {resumen_hoy}")
    print(f"    {ruta.relative_to(AQUI)}  ({ruta.stat().st_size/1024:.0f} KB)")
    return 0


def main():
    CAPAS.mkdir(parents=True, exist_ok=True)
    hoy = date.today().isoformat()
    carpeta_hoy = CRUDO / hoy
    carpeta_hoy.mkdir(parents=True, exist_ok=True)

    print(f"SERNAGEOMIN · snapshot del {hoy}\n")
    problemas = []

    for n_capa, (nombre, para_que) in CAPAS_A_TRAER.items():
        geojson, motivo = traer_capa(n_capa)
        if geojson is None:
            print(f"  ✗ capa {n_capa} ({nombre}): {motivo}")
            problemas.append((nombre, motivo))
            continue

        n = len(geojson["features"])

        # el crudo, tal como llegó, con la fecha en la ruta: esto es el archivo
        # histórico que la fuente no tiene
        (carpeta_hoy / f"{nombre}.geojson").write_text(
            json.dumps(geojson, ensure_ascii=False), encoding="utf-8")

        # las capas de trabajo, siempre la última versión
        if nombre in ("zonas_geograficas", "comunas"):
            (CAPAS / f"{nombre}.geojson").write_text(
                json.dumps(geojson, ensure_ascii=False), encoding="utf-8")

        print(f"  ✓ capa {n_capa} · {nombre:18s} {n:4d} rasgos — {para_que}")

        if nombre == "zonas_geograficas":
            print("      niveles de peligro vigentes hoy:")
            for nivel, cuantas in sorted(resumir_minuta(geojson).items(),
                                         key=lambda x: -x[1]):
                print(f"        {cuantas:4d}  {nivel}")

    if problemas:
        print("\n  ATENCIÓN: quedaron capas sin traer. Se anotan como hueco, "
              "no como cero.")
        for nombre, motivo in problemas:
            print(f"    · {nombre}: {motivo}")

    print(f"\n  crudo fechado en: {carpeta_hoy.relative_to(AQUI)}")
    print("  ★ correr esto TODOS LOS DÍAS: la fuente se sobrescribe y no "
          "guarda historia.")
    return 1 if problemas else 0


if __name__ == "__main__":
    import sys
    # Dos modos, a propósito:
    #   --minuta  → foto liviana de los niveles. Es la que corre programada
    #               varias veces al día. Unos KB.
    #   (sin arg) → descarga completa con geometría. Se corre a mano, cuando
    #               hace falta refrescar las capas base. Unos 100 MB.
    if "--minuta" in sys.argv:
        raise SystemExit(snapshot_minuta())
    raise SystemExit(main())

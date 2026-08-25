"""
Trae clima para los lugares donde SÍ ocurrieron remociones en masa reales.

PARA QUÉ
--------
El ancla de Copiapó ya se usó para diagnosticar, así que dejó de servir para
validar: probar una corrección contra el caso que la motivó es examinarse con la
prueba que uno mismo escribió.

ReTeRM es la salida honesta: **380 eventos reales** de remoción en masa
registrados por SERNAGEOMIN entre 1996 y 2026, con fecha, comuna y detonante.
**302 de ellos fueron detonados por lluvia.** Si la medida corregida sirve, los
meses de esos eventos tienen que puntuar más alto que los meses en que no pasó
nada en el mismo lugar.

CÓMO SE ELIGEN LOS PUNTOS
-------------------------
Un punto por comuna afectada, ubicado en el promedio de las coordenadas de los
eventos reales de esa comuna — no en el centroide administrativo. La diferencia
importa: los deslizamientos ocurren en laderas, no en el centro geométrico de la
comuna, y en Chile una comuna puede ir del mar a la cordillera.

Se baja la serie completa 1990-2026 porque hace falta la climatología del lugar
para calcular la razón contra lo normal; el mes del evento solo no alcanza.
"""

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

AQUI = Path(__file__).parent
sys.path.insert(0, str(AQUI))
import traer_clima_subestaciones as base  # reutiliza el descargador ya probado

RETERM = (AQUI / "datos" / "crudo" / "sernageomin" / "2026-08-15" /
          "reterm_eventos.geojson")
SALIDA_EVENTOS = AQUI / "datos" / "reterm_eventos.csv"
SALIDA_PUNTOS = AQUI / "datos" / "reterm_puntos.csv"
SALIDA_CLIMA = AQUI / "datos" / "clima_diario_reterm_era5.csv"


def leer_eventos():
    """Eventos con fecha, comuna y coordenada utilizable."""
    datos = json.loads(RETERM.read_text(encoding="utf-8"))
    eventos = []
    for rasgo in datos["features"]:
        p = rasgo.get("properties", {})
        g = rasgo.get("geometry") or {}
        if g.get("type") != "Point" or not g.get("coordinates"):
            continue
        lon, lat = g["coordinates"][0], g["coordinates"][1]
        if not (-56 <= lat <= -17 and -76 <= lon <= -66):
            continue
        crudo = p.get("Fecha_evento")
        if not isinstance(crudo, (int, float)):
            continue
        fecha = datetime.fromtimestamp(crudo / 1000, timezone.utc)
        eventos.append({
            "id": p.get("OBJECTID"),
            "fecha": fecha.strftime("%Y-%m-%d"),
            "anio": fecha.year, "mes": fecha.month,
            "tipo": p.get("Tipo"), "detonante": p.get("Detonante"),
            "comuna": p.get("COMUNA"), "provincia": p.get("PROVINCIA"),
            "region": p.get("REGION"), "lat": round(lat, 5), "lon": round(lon, 5),
        })
    return eventos


def puntos_por_comuna(eventos):
    """Un punto por comuna, en el promedio de sus eventos reales."""
    grupos = defaultdict(list)
    for e in eventos:
        if e["comuna"]:
            grupos[e["comuna"]].append(e)
    puntos = []
    for comuna, lista in sorted(grupos.items()):
        puntos.append({
            "subestacion": f"ReTeRM · {comuna}",   # el descargador usa esta clave
            "comuna": comuna,
            "region": lista[0]["region"],
            "n_eventos": len(lista),
            "lat": round(sum(e["lat"] for e in lista) / len(lista), 5),
            "lon": round(sum(e["lon"] for e in lista) / len(lista), 5),
        })
    return puntos


def main():
    eventos = leer_eventos()
    puntos = puntos_por_comuna(eventos)
    con_lluvia = sum(1 for e in eventos
                     if e["detonante"] and "luvia" in str(e["detonante"]).lower()
                     or e["detonante"] and "recipitac" in str(e["detonante"]).lower())

    print(f"{len(eventos)} eventos con fecha y coordenada utilizables")
    print(f"   detonados por lluvia: {con_lluvia}")
    print(f"   comunas afectadas:    {len(puntos)}")

    with SALIDA_EVENTOS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(eventos[0].keys()))
        w.writeheader(); w.writerows(eventos)
    with SALIDA_PUNTOS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(puntos[0].keys()))
        w.writeheader(); w.writerows(puntos)

    # Se reutiliza el descargador de las subestaciones apuntándolo a otros
    # archivos: mismo código probado, mismos reintentos, misma capacidad de
    # reanudar si se corta.
    #
    # ★ HAY QUE REDIRIGIR **TODAS** LAS RUTAS, no sólo las obvias.
    # La primera versión de esto olvidó `SALIDA_PUNTOS` y el descargador
    # sobrescribió `subestaciones_puntos.csv` con los puntos de ReTeRM. Se
    # detectó y se restauró, pero deja la lección: al reutilizar un módulo
    # cambiándole las rutas por fuera, cualquier ruta que uno no redirija sigue
    # apuntando al archivo original y lo pisa en silencio.
    base.SALIDA = SALIDA_CLIMA
    base.SALIDA_PUNTOS = SALIDA_PUNTOS
    base.CSV_PUNTOS = SALIDA_PUNTOS
    base.leer_subestaciones = lambda: puntos
    base.HILOS = 1          # el servicio ya nos limitó con 2; con 91 puntos, de a uno
    base.PAUSA_S = 2.5
    # ★ Sólo lluvia. Open-Meteo cobra por peso (días × variables), y pidiendo
    # las tres nos cortó 60 de 91 puntos. La medida de peligro no usa
    # temperatura, así que pedirla era gastar cuota en dato que no se mira.
    base.VARIABLES = "precipitation_sum"
    print(f"\nBajando clima 1990-2026 para {len(puntos)} puntos...")
    base.main()


if __name__ == "__main__":
    main()

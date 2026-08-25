"""
LA GRILLA COMPLETA · ERA5-Land 0,1° desde Copernicus, de una sola vez
=======================================================================

✔ PROBADO el 23-ago-2026 con credencial real. La prueba de julio 2026 devolvió
396 × 101 puntos a 0,1° exactos, 31 pasos diarios, 1,46 MB, en 35 segundos.
Verificado a mano: unidades en METROS (máximo 0,1359 m = 135,9 mm/día) y los
picos del temporal caen en los días correctos.

★★ CONTRASTE CONTRA LA SERIE VIEJA (celda -308_-711, Coquimbo, julio 2026):
   Open-Meteo 343,6 mm · **ERA5-Land 408,7 mm = 119 %**. Los días grandes
   coinciden uno a uno (17-jul: 86,3 vs 89,9), así que no es un desfase de
   fechas sino una diferencia real de magnitud entre fuentes. Es exactamente el
   salto artificial que se fabricaría al empalmar las dos.

★ POR QUÉ HACE FALTA
----------------------
`barrido_climatico.py` baja celda por celda desde Open-Meteo, y ese diseño no
puede terminar: techo medido de ~90 celdas/hora y quedan ~4.800 → más de 50 h de
reloj, cortadas por el tope diario del servicio. Copernicus entrega el área
COMPLETA en cada petición, no un punto.

★ EL ATAJO QUE VUELVE VIABLE LA DESCARGA
------------------------------------------
El catálogo del CDS tiene un dataset de estadísticas diarias
(`derived-era5-land-daily-statistics`) que sería lo ideal… salvo que **excluye
justamente la precipitación**: «Accumulated variables are omitted (e.g. total
precipitation, runoff, etc)». Hay que ir a la horaria, que son 24× más datos.

Pero en ERA5-Land `total_precipitation` es un **acumulado que se reinicia cada
día a las 00 UTC**. Eso significa que el paso de las **00:00 UTC del día D+1 ya
contiene la suma de las 24 horas del día D**. Pidiendo una sola hora al día se
baja el equivalente diario sin bajar las 24:

    31.960 puntos de grilla × 13.505 días ≈ 431 millones de valores
    (contra 10.400 millones si se bajaran las 24 horas)

⚠️⚠️ TRES COSAS QUE HAY QUE DECIDIR ANTES DE MEZCLAR ESTO CON LO QUE YA HAY
-----------------------------------------------------------------------------
1. **La serie actual NO es ERA5-Land puro.** Viene de Open-Meteo, y medido el
   23-ago sus centros de celda salen separados 0,0703° — una malla más fina que
   los 0,1° de ERA5-Land. No es la misma grilla.
2. **El día no empieza a la misma hora.** La serie actual se pidió con
   `timezone=America/Santiago`; ERA5-Land acumula en UTC. Son 3-4 h de desfase,
   que en un temporal reparte la lluvia entre dos días distintos.
3. Por (1) y (2), **empalmar celdas nuevas de aquí con las 418 viejas fabricaría
   saltos artificiales** — el mismo error que el proyecto ya decidió no cometer
   al no rellenar huecos de estación con reanálisis. Lo limpio es rebajar TODAS
   las celdas desde una sola fuente, incluidas las que ya están.

USO
---
    # 1. registrarse en https://cds.climate.copernicus.eu (gratis)
    # 2. aceptar las condiciones del dataset ERA5-Land en la web
    # 3. dejar la clave en ~/.cdsapirc
    # 4. pip install cdsapi
    ../.venv-esa/bin/python bajar_era5land_cds.py --probar    # UN mes, a mano
    ../.venv-esa/bin/python bajar_era5land_cds.py --bajar     # todo, reanudable
    ../.venv-esa/bin/python bajar_era5land_cds.py --estado
"""

import sys
from datetime import date, timedelta
from pathlib import Path

AQUI = Path(__file__).resolve().parent
DESTINO = AQUI / "datos" / "crudo" / "era5land"

# Chile continental. Formato del CDS: [Norte, Oeste, Sur, Este].
# Deja fuera Pascua, Juan Fernández y la Antártica (9 celdas) a propósito:
# estiran el área a 265.000 puntos de grilla sin aportar casi nada.
AREA = [-17.0, -76.0, -56.5, -66.0]

# ★ EL TERRITORIO INSULAR, que el recuadro continental deja fuera.
#   Un solo recuadro cubre los tres archipiélagos con activos: Desventuradas
#   (San Félix, -26,3/-80,1), Juan Fernández (Robinson Crusoe, -33,6/-78,9) e
#   Isla de Pascua (-27,1/-109,4). Es casi todo océano —ERA5-Land sólo tiene
#   datos sobre tierra, el resto viene vacío— así que pesa poquísimo pese a
#   abarcar 32° de longitud. Un recuadro por isla habría triplicado las
#   peticiones sin bajar un solo dato más.
AREA_INSULAR = [-26.0, -110.0, -34.0, -78.0]

DATASET = "reanalysis-era5-land"
DESDE, HASTA = date(1990, 1, 1), date(2026, 8, 21)


def meses(desde, hasta):
    """Los (año, mes) a pedir. Uno por petición: es la unidad que el CDS acepta
    cómodamente y la que permite reanudar sin rehacer nada."""
    out, a, m = [], desde.year, desde.month
    while (a, m) <= (hasta.year, hasta.month):
        out.append((a, m))
        a, m = (a + 1, 1) if m == 12 else (a, m + 1)
    return out


def area_activa():
    """`--insular` cambia el recuadro y la carpeta de destino."""
    if "--insular" in sys.argv:
        return AREA_INSULAR, DESTINO.parent / "era5land_insular"
    return AREA, DESTINO


def peticion(a, m):
    """La hora 00:00 de cada día del mes.

    ★★ EL CORRIMIENTO DE UN DÍA — lo más fácil de equivocar de todo el script.
    En ERA5-Land `total_precipitation` acumula desde las 00 UTC, así que el paso
    de las **00:00 del día D trae la lluvia del día D−1**, no la del día D.

    Un archivo de julio (pasos 1-jul a 31-jul) cubre por tanto del 30-jun al
    30-jul. El 31 de julio no falta: aparece en el paso 1-ago, dentro del archivo
    de agosto. Bajando meses consecutivos cada día queda cubierto exactamente una
    vez, sin huecos ni repeticiones.

    (Un intento anterior agregaba «el día 1 del mes siguiente» a la lista de
    días. No servía: `month` va fijo en la petición, así que ese día 1 se pedía
    del MISMO mes y el CDS simplemente lo devolvía duplicado. Verificado en la
    prueba de julio 2026: llegaron 31 pasos, del 1 al 31 de julio.)

    ⚠️ Al convertir a serie diaria hay que RESTAR UN DÍA a la marca de tiempo.
    Y las 00:00 UTC son las 20:00 del día anterior en Chile: el «día» de ERA5 no
    coincide con el día local que usa la serie vieja de Open-Meteo.
    """
    return {
        "variable": ["total_precipitation"],
        "year": [str(a)],
        "month": [f"{m:02d}"],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": ["00:00"],
        "area": area_activa()[0],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def main():
    area, destino_dir = area_activa()
    destino_dir.mkdir(parents=True, exist_ok=True)
    globals()["DESTINO"] = destino_dir
    todos = meses(DESDE, HASTA)
    hechos = {p.stem for p in destino_dir.glob("*.nc")}
    print(f"  área: {area}  ·  destino: {destino_dir.name}")

    if "--estado" in sys.argv:
        print(f"  meses a bajar : {len(todos)}")
        print(f"  ya bajados    : {len(hechos)} "
              f"({100*len(hechos)/len(todos):.1f} %)")
        peso = sum(p.stat().st_size for p in destino_dir.glob("*.nc"))
        print(f"  en disco      : {peso/1e6:.0f} MB")
        if hechos:
            print(f"  proyección    : ~{peso/max(len(hechos),1)*len(todos)/1e9:.1f} GB al terminar")
        return 0

    try:
        import cdsapi
    except ImportError:
        print("falta cdsapi:  ../.venv-esa/bin/pip install cdsapi")
        return 1
    if not (Path.home() / ".cdsapirc").exists():
        print("falta ~/.cdsapirc — hay que registrarse en Copernicus primero.")
        print("La clave la pega Alexis; yo no la manejo ni la guardo.")
        return 1

    c = cdsapi.Client()

    if "--probar" in sys.argv:
        # ★ Un solo mes, y justo el del temporal: si algo está mal en los
        #   parámetros, se ve aquí y no después de 444 peticiones.
        a, m = 2026, 7
        destino = destino_dir / f"{a}-{m:02d}.nc"
        print(f"prueba · {a}-{m:02d} · área {AREA}")
        c.retrieve(DATASET, peticion(a, m), str(destino))
        print(f"bajado: {destino} ({destino.stat().st_size/1e6:.1f} MB)")
        print("\n★ REVISAR A MANO ANTES DE SEGUIR:")
        print("  - que la grilla sea de 0,1° y cubra el área pedida")
        print("  - que el 25-jul-2026 tenga lluvia fuerte en Coquimbo")
        print("  - que las unidades sean METROS (ERA5 acumula en m, no en mm)")
        return 0

    if "--bajar" not in sys.argv:
        print(__doc__)
        return 1

    faltan = [(a, m) for a, m in todos if f"{a}-{m:02d}" not in hechos]
    print(f"meses {len(todos)} · hechos {len(hechos)} · por bajar {len(faltan)}",
          flush=True)
    for i, (a, m) in enumerate(faltan, 1):
        destino = destino_dir / f"{a}-{m:02d}.nc"
        try:
            c.retrieve(DATASET, peticion(a, m), str(destino))
            print(f"  [{i}/{len(faltan)}] {a}-{m:02d} "
                  f"{destino.stat().st_size/1e6:.1f} MB", flush=True)
        except Exception as e:
            # No se aborta: el CDS encola y a veces expira. Se reintenta en la
            # pasada siguiente, que es barata porque los .nc ya bajados se saltan.
            print(f"  [{i}/{len(faltan)}] {a}-{m:02d} FALLÓ · {e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

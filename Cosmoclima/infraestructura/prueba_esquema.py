"""
Prueba que el esquema RECHAZA lo que tiene que rechazar.

Un esquema que sólo acepta datos buenos no prueba nada: hay que verificar que
frena los malos. Cada caso de acá corresponde a una regla que el proyecto
declaró innegociable, y existe para que nadie la afloje sin darse cuenta.

Corre sobre una base temporal; no toca `datos/consolidado.sqlite`.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import esquema  # noqa: E402

OK = "  ✓"
FALLO = "  ✗ FALLÓ"

# Una observación bien formada, que se usa de molde para ir rompiéndola.
MOLDE = dict(
    id_fuente="prueba", familia="AMENAZA", variable="peligro_remocion_masa",
    valor_original="Alta", unidad_original="categoria_3_niveles",
    valor_normalizado=1.0, metodo_normalizacion="categorica_3",
    vigencia_inicio="2026-08-15", vigencia_fin="2026-08-18",
    territorio_tipo="zona_geografica", territorio_id="Precordillera Alto Loa",
    confianza=0.9, fecha_descarga="2026-08-15T23:00:00",
    url_exacta="https://ejemplo.cl/minuta",
)


def caso(nombre, obs, debe_pasar):
    """Corre un caso y devuelve True si se comportó como se esperaba."""
    problemas = esquema.validar_observacion(obs)
    paso = not problemas
    bien = (paso == debe_pasar)
    marca = OK if bien else FALLO
    detalle = "" if paso else f" → {'; '.join(problemas)}"
    print(f"{marca}  {nombre}{detalle}")
    return bien


def main():
    print("VALIDACIÓN EN PYTHON — casos que deben pasar y casos que no\n")
    resultados = []

    resultados.append(caso("observación bien formada", dict(MOLDE), True))

    sin_vigencia = dict(MOLDE); sin_vigencia["vigencia_fin"] = ""
    resultados.append(caso("sin vigencia_fin → RECHAZA", sin_vigencia, False))

    sin_conf = dict(MOLDE); sin_conf["confianza"] = None
    resultados.append(caso("sin confianza → RECHAZA", sin_conf, False))

    sin_terr = dict(MOLDE); sin_terr["territorio_id"] = "   "
    resultados.append(caso("sin territorio_id → RECHAZA", sin_terr, False))

    mal_fam = dict(MOLDE); mal_fam["familia"] = "CLIMA"
    resultados.append(caso("familia inventada → RECHAZA", mal_fam, False))

    conf_alta = dict(MOLDE); conf_alta["confianza"] = 1.7
    resultados.append(caso("confianza 1,7 → RECHAZA", conf_alta, False))

    norm_sin_metodo = dict(MOLDE); norm_sin_metodo["metodo_normalizacion"] = None
    resultados.append(caso("normalizado sin método declarado → RECHAZA",
                           norm_sin_metodo, False))

    punto_sin_coord = dict(MOLDE)
    punto_sin_coord.update(territorio_tipo="punto", territorio_id="SE Copiapó")
    resultados.append(caso("punto sin lat/lon → RECHAZA", punto_sin_coord, False))

    vig_invertida = dict(MOLDE)
    vig_invertida.update(vigencia_inicio="2026-08-18", vigencia_fin="2026-08-15")
    resultados.append(caso("vigencia al revés → RECHAZA", vig_invertida, False))

    sin_normalizar = dict(MOLDE)
    sin_normalizar.update(valor_normalizado=None, metodo_normalizacion=None)
    resultados.append(caso("sin normalizar todavía → ACEPTA (se normaliza luego)",
                           sin_normalizar, True))

    print("\nRESTRICCIONES EN LA BASE — lo que Python no ve, lo ve SQLite\n")
    with tempfile.TemporaryDirectory() as tmp:
        con = esquema.conectar(Path(tmp) / "prueba.sqlite")
        esquema.registrar_fuente(
            con, id="prueba", organismo="Prueba", producto="Prueba",
            url="https://ejemplo.cl", formato="json", familia="AMENAZA",
            acceso="anonimo", acceso_verificado=1, permite_automatizacion="si",
            granularidad="zona_geografica", confianza_base=0.9)

        n, _ = esquema.insertar_observaciones(con, [dict(MOLDE)])
        ok = (n == 1)
        print(f"{OK if ok else FALLO}  inserta la observación válida")
        resultados.append(ok)

        # coordenada fuera de Chile: la base tiene que frenarla aunque Python
        # la deje pasar (Python no valida geografía, la base sí)
        fuera = dict(MOLDE)
        fuera.update(territorio_tipo="punto", territorio_id="Madrid",
                     lat=40.4, lon=-3.7)
        try:
            esquema.insertar_observaciones(con, [fuera])
            print(f"{FALLO}  coordenada fuera de Chile → debió rechazar")
            resultados.append(False)
        except Exception as e:
            print(f"{OK}  coordenada fuera de Chile → RECHAZA "
                  f"({type(e).__name__})")
            resultados.append(True)

        # modo no estricto: las buenas entran, las malas se devuelven
        mezcla = [dict(MOLDE), dict(sin_conf), dict(MOLDE)]
        n, rechazadas = esquema.insertar_observaciones(con, mezcla, estricto=False)
        ok = (n == 2 and len(rechazadas) == 1)
        print(f"{OK if ok else FALLO}  modo no estricto: entran 2, rechaza 1 "
              f"(entraron {n}, rechazó {len(rechazadas)})")
        resultados.append(ok)

        esquema.registrar_hueco(con, "prueba", "2026-08-15T23:10:00",
                                "sin_respuesta", "timeout", "https://ejemplo.cl")
        r = esquema.resumen(con)
        ok = (r["huecos"] == 1 and r["sin_territorio_resuelto"] == r["observaciones"])
        print(f"{OK if ok else FALLO}  hueco registrado y territorio sin resolver "
              f"({r})")
        resultados.append(ok)

    total, bien = len(resultados), sum(resultados)
    print(f"\n{'TODO EN ORDEN' if bien == total else '¡HAY FALLAS!'}: "
          f"{bien}/{total} casos se comportaron como se esperaba")
    return 0 if bien == total else 1


if __name__ == "__main__":
    sys.exit(main())

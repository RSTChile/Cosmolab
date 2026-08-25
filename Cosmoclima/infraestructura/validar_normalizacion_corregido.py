"""
La prueba dura: ¿la normalización aporta algo, o bastaba con los milímetros?

POR QUÉ HACE FALTA ESTA PRUEBA
------------------------------
Al mirar el resultado anterior caí en algo incómodo: **dentro de un mismo punto,
ordenar por `peligro` es idéntico a ordenar por milímetros**. Las dos
componentes son monótonas en el tamaño del evento cuando el lugar está fijo, así
que el orden no cambia. Es decir: la prueba «los deslizamientos caen en el decil
alto de su propio punto» confirma que llover fuerte anticipa deslizamientos
—cierto, pero nadie lo dudaba— y **no dice nada sobre si la normalización sirve**.

Lo que la normalización promete es otra cosa: hacer comparables lugares
distintos. Que 104 mm en Copiapó pesen más que 202 mm en Curicó. Esa promesa
sólo se puede probar comparando ENTRE puntos, no dentro de uno.

LA PRUEBA
---------
Se juntan todos los meses de todos los puntos en una sola bolsa y se pregunta:
tomando al azar un mes-con-deslizamiento y un mes-sin-deslizamiento, ¿con qué
probabilidad el primero puntúa más alto?

Eso es el AUC. Vale 0,5 si la medida no distingue nada y 1,0 si separa perfecto.
Se calcula dos veces:

    · con `peligro` (magnitud nacional × razón contra la normal del lugar)
    · con los milímetros crudos en 48 h

**Si el AUC del peligro no supera al de los milímetros crudos, la normalización
no está aportando** y habría que decirlo, por elegante que sea la construcción.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI / "adaptadores"))
import era5  # noqa: E402

EVENTOS = AQUI / "datos" / "reterm_eventos.csv"
# ★ Variante con las 6 coordenadas costeras corregidas a tierra
# (ver datos/correccion_puntos_costeros.csv). No excluye ningún punto:
# los 91 siguen, y con ellos los 152 eventos.
CLIMA = AQUI / "datos" / "clima_diario_reterm_era5_corregido.csv"
PUNTOS = AQUI / "datos" / "reterm_puntos.csv"


def auc(positivos, negativos):
    """Probabilidad de que un positivo al azar supere a un negativo al azar.

    Se calcula por rangos (el estadístico de Mann-Whitney), que es exacto y no
    necesita muestrear pares. Los empates cuentan medio, como corresponde.
    """
    todos = sorted([(v, 1) for v in positivos] + [(v, 0) for v in negativos])
    rango, i, suma_pos = 0, 0, 0.0
    while i < len(todos):
        j = i
        while j < len(todos) and todos[j][0] == todos[i][0]:
            j += 1
        rango_medio = (i + j - 1) / 2 + 1          # rangos desde 1, empates promediados
        for k in range(i, j):
            if todos[k][1] == 1:
                suma_pos += rango_medio
        i = j
    n1, n0 = len(positivos), len(negativos)
    if n1 == 0 or n0 == 0:
        return None
    return (suma_pos - n1 * (n1 + 1) / 2) / (n1 * n0)


def es_lluvia(d):
    t = str(d or "").lower()
    return "luvia" in t or "recipitac" in t


def main():
    era5.CSV_DIARIO = CLIMA
    era5.CSV_PUNTOS = PUNTOS
    obs, problema = era5.traer()
    if problema:
        print("SIN DATO:", problema)
        return 1

    # peligro y milímetros crudos por (punto, año, mes)
    peligro, milimetros = {}, {}
    for o in obs:
        if o["variable"] != "peligro_precipitacion":
            continue
        a, m = int(o["vigencia_inicio"][:4]), int(o["vigencia_inicio"][5:7])
        peligro[(o["territorio_id"], a, m)] = o["valor_normalizado"]
        milimetros[(o["territorio_id"], a, m)] = float(o["valor_original"])

    with EVENTOS.open(encoding="utf-8") as fh:
        eventos = [e for e in csv.DictReader(fh) if es_lluvia(e["detonante"])]

    con_evento = set()
    for e in eventos:
        clave = (f"ReTeRM · {e['comuna']}", int(e["anio"]), int(e["mes"]))
        if clave in peligro:
            con_evento.add(clave)

    if len(con_evento) < 50:
        print(f"Sólo {len(con_evento)} meses con evento. Muestra insuficiente.")
        return 1

    pos_pel = [peligro[k] for k in con_evento]
    neg_pel = [v for k, v in peligro.items() if k not in con_evento]
    pos_mm = [milimetros[k] for k in con_evento]
    neg_mm = [v for k, v in milimetros.items() if k not in con_evento]

    auc_pel = auc(pos_pel, neg_pel)
    auc_mm = auc(pos_mm, neg_mm)

    print("=" * 74)
    print("¿LA NORMALIZACIÓN APORTA?  — CON LAS 6 COORDENADAS COSTERAS CORREGIDAS")
    print("=" * 74)
    print(f"\n  meses con deslizamiento : {len(pos_pel):,}")
    print(f"  meses sin deslizamiento : {len(neg_pel):,}")
    print(f"  puntos                  : {len({k[0] for k in peligro})}")
    print(f"\n  Todos los meses de todos los puntos en una sola bolsa:\n")
    print(f"    AUC con «peligro» normalizado : {auc_pel:.4f}")
    print(f"    AUC con milímetros crudos     : {auc_mm:.4f}")
    print(f"    diferencia                    : {auc_pel - auc_mm:+.4f}")
    print(f"\n    (0,50 = no distingue nada · 1,00 = separación perfecta)")

    aporta = auc_pel > auc_mm
    dictamen = ("ordena mejor que los milímetros crudos" if aporta
                else "los milímetros crudos ordenan igual o mejor")
    print(f"\n  → La normalización {'APORTA' if aporta else 'NO APORTA'}: "
          f"{dictamen}")

    # Dónde se nota: separar por lo seco o húmedo que es cada lugar
    print("\n" + "=" * 74)
    print("DÓNDE SE NOTA — por régimen de lluvia del lugar")
    print("=" * 74)
    normales = {}
    for o in obs:
        if o["variable"] == "peligro_precipitacion" and "normal anual" in o["notas"]:
            se = o["territorio_id"]
            if se not in normales:
                trozo = o["notas"].split("normal anual (")[1].split(" mm")[0]
                normales[se] = float(trozo)
    if normales:
        corte = sorted(normales.values())[len(normales) // 2]
        print(f"\n  corte en la normal anual mediana: {corte:.0f} mm\n")
        for etiqueta, filtro in (("lugares SECOS (bajo la mediana)",
                                  lambda s: normales.get(s, 0) < corte),
                                 ("lugares HÚMEDOS (sobre la mediana)",
                                  lambda s: normales.get(s, 0) >= corte)):
            pp = [peligro[k] for k in con_evento if filtro(k[0])]
            nn = [v for k, v in peligro.items()
                  if k not in con_evento and filtro(k[0])]
            pm = [milimetros[k] for k in con_evento if filtro(k[0])]
            nm = [v for k, v in milimetros.items()
                  if k not in con_evento and filtro(k[0])]
            if len(pp) >= 20:
                print(f"  {etiqueta}  ({len(pp)} eventos)")
                print(f"     peligro {auc(pp, nn):.4f}   ·   mm crudos "
                      f"{auc(pm, nm):.4f}")

    print("\n  Nota honesta: dentro de UN punto, ordenar por peligro es idéntico")
    print("  a ordenar por milímetros — las dos componentes son monótonas con el")
    print("  lugar fijo. Todo lo que la normalización puede aportar está en la")
    print("  comparación ENTRE lugares, que es justo lo que mide esta prueba.")
    return 0 if aporta else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reconstruir_lluvia_desde_estaciones_reales.py -- reemplaza el tramo de
REANÁLISIS de la serie de lluvia del punto-reloj por una estimación derivada
de ESTACIONES REALES (13-ago-2026).

EL PROBLEMA QUE RESUELVE. La serie que usa el instrumento es estación real
(CR2 Huintil, producto MENSUAL) de 1966-01 a 2018-12, y desde 2019-01
reanálisis ERA5 corregido por sesgo -- 92 meses de 728. Ese empalme se hizo
el 12-ago porque no había estación real disponible en 60 km después de 2018.
Con las 38 estaciones de Coquimbo que llegaron el 13-ago eso dejó de ser
cierto: hay 12 estaciones DMC/FDF/INIA dentro de 60 km cubriendo 2019-2026.

Y hay un motivo fuerte para preferirlas: comparadas contra el reanálisis, el
factor multiplicativo por mes calendario COMPRIME el rango dinámico -- infla
los años secos (2019: 35,5 mm de ERA5 contra ~15 medidos; 2021: 66,4 contra
~31) y achata los húmedos (12-18% por debajo). Con un umbral de germinación
de 15 mm/mes, inflar años secos FABRICA floraciones que no ocurrieron.

EL MÉTODO, Y POR QUÉ ASÍ.
El problema es que las estaciones nuevas NO solapan con Huintil: Huintil
mensual termina 2018-12 y las nuevas empiezan 2019-01. Así que no se puede
calibrar una contra otra directamente. Se encadena:

    Huintil  <--(razón calibrada en 30-50 años)--  estación HISTÓRICA
                                                        ||  (mismo sitio)
                                          estación NUEVA DMC 2019-2026

El eslabón débil es el "mismo sitio", así que se exige explícitamente:
  - distancia entre la histórica y la nueva por debajo de UMBRAL_COLOCACION
  - al menos MIN_MESES_COMUNES meses en común entre la histórica y Huintil

La razón se calcula POR MES CALENDARIO (no anual) como cociente de sumas
--sum(Huintil de todos los eneros)/sum(histórica de todos los eneros)-- que
es lo robusto en un desierto donde muchísimos meses valen 0 y un cociente
mes a mes explotaría. Es la misma técnica que ya se usó para el empalme
ERA5, ahora aplicada a medición en vez de a reanálisis.

El resultado final de cada mes es la MEDIANA de las estimaciones de todas
las estaciones disponibles ese mes, no una sola estación: si una falla o
tiene un mes raro, no arrastra la serie.

TRES VALIDACIONES, porque un método sin error medido es una apuesta:
  V1 COLOCACIÓN  distancia real entre cada par histórica/nueva.
  V2 LOOCV       dejar-un-año-fuera sobre el registro histórico: se recalcula
                 la razón sin ese año y se predice Huintil para ese año con
                 cada estación. Da el error propio del método, en mm y en %.
  V3 CADENA      La Canela retén (estación de la tanda NUEVA) cubre 2018,
                 que SÍ solapa con Huintil mensual. Son 12 meses de
                 verificación directa de la cadena completa, el único tramo
                 donde se puede comparar predicción contra Huintil medido.

DESCARTADO A PROPÓSITO: 'Rio Illapel En Huintil', a 1.8 km -- la más cercana
de todas. En 26 meses de solape con Huintil registró 214 mm contra 479, y en
5 de 8 meses lluviosos marcó 0.0 mm mientras Huintil marcaba 38-80 mm. Es
una estación fluviométrica de la DGA, no un pluviómetro confiable. La
cercanía no la salva; el dato manda.
"""
import csv
import math
import os
import sqlite3
import statistics as st
import sys

RAIZ = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima"
DB = os.path.join(RAIZ, "investigacion/fuentes/pluviosidad_diaria_consolidada.sqlite")
CSV_FUENTES = os.path.join(RAIZ, "investigacion/fuentes/lluvia_mensual_zhcs_1900_2027.csv")
CSV_SALIDA = os.path.join(RAIZ, "investigacion/fuentes/lluvia_mensual_zhcs_reconstruida_estaciones.csv")
INFORME = os.path.join(RAIZ, "investigacion/fuentes/lluvia_reconstruccion_estaciones.txt")

HUINTIL_LAT, HUINTIL_LON = -31.5669, -70.9817
UMBRAL_COLOCACION_KM = 5.0     # histórica y nueva tienen que ser el mismo sitio
MIN_MESES_COMUNES = 120        # 10 años de registro conjunto con Huintil
RADIO_KM = 60.0
DIAS_MIN_MES = 25              # un mes con menos días medidos no se usa
PRIMER_MES_A_RECONSTRUIR = "2019-01"

BANDA = []   # el informe se arma acá y se imprime y guarda al final


def log(s=""):
    print(s)
    BANDA.append(s)


def km(la1, lo1, la2, lo2):
    return math.hypot((la1 - la2) * 111.0,
                      (lo1 - lo2) * 111.0 * math.cos(math.radians((la1 + la2) / 2)))


def cargar_huintil_mensual():
    """Serie mensual CR2 de la estación Huintil -- el objetivo a reproducir."""
    serie = {}
    for r in csv.DictReader(open(CSV_FUENTES, encoding="utf-8")):
        if "CR2 Huintil" in (r.get("fuente") or "") and r["lluvia_mm"] not in ("", None):
            serie[r["anio_mes"]] = float(r["lluvia_mm"])
    return serie


def cargar_estaciones(con):
    """Mensuales por estación real dentro del radio, sólo meses con
    DIAS_MIN_MES días medidos o más (un mes a medias sesga hacia abajo)."""
    filas = con.execute("""
        SELECT localidad, lat, lon, substr(fecha,1,7) AS mes,
               SUM(lluvia_mm) AS mm, COUNT(*) AS n
        FROM pluviosidad_diaria
        WHERE tipo_fuente = 'estacion_real'
        GROUP BY localidad, lat, lon, mes""").fetchall()
    est = {}
    for loc, la, lo, mes, mm, n in filas:
        d = km(la, lo, HUINTIL_LAT, HUINTIL_LON)
        if d > RADIO_KM or loc == "Huintil":
            continue
        if n < DIAS_MIN_MES:
            continue
        e = est.setdefault(loc, {"lat": la, "lon": lo, "km": d, "meses": {}})
        e["meses"][mes] = mm
    return est


def razones_por_mes_calendario(huintil, mensual, excluir_anio=None):
    """Cociente de sumas por mes calendario. Devuelve {1..12: razón} sólo
    para los meses donde hay masa suficiente en ambos lados."""
    num = {m: 0.0 for m in range(1, 13)}
    den = {m: 0.0 for m in range(1, 13)}
    n = {m: 0 for m in range(1, 13)}
    for ym, v in mensual.items():
        if ym not in huintil:
            continue
        if excluir_anio and ym[:4] == excluir_anio:
            continue
        m = int(ym[5:7])
        num[m] += huintil[ym]
        den[m] += v
        n[m] += 1
    # Respaldo para los meses de VERANO. En enero/febrero/diciembre casi no
    # llueve en la ZHCS, así que el cociente de sumas se queda sin masa con
    # que calibrarse (den < 20 mm) y quedaría indefinido -- dejando huecos
    # justo al principio de 2019. Para esos meses se usa la razón GLOBAL de
    # la estación (todos los meses juntos), que existe siempre. No es una
    # licencia: son meses que rondan 0 mm y quedan muy por debajo del umbral
    # de germinación con cualquier razón razonable; lo que importa es no
    # dejar el mes vacío.
    global_num = sum(num.values()); global_den = sum(den.values())
    razon_global = (global_num / global_den) if global_den > 0 else 1.0
    r = {}
    for m in range(1, 13):
        if n[m] >= 8 and den[m] >= 20.0:
            r[m] = num[m] / den[m]
        elif n[m] >= 8:
            r[m] = razon_global
    return r, n


def main():
    con = sqlite3.connect(DB)
    huintil = cargar_huintil_mensual()
    est = cargar_estaciones(con)
    log(f"Huintil mensual (CR2): {len(huintil)} meses, {min(huintil)} a {max(huintil)}")
    log(f"Estaciones reales dentro de {RADIO_KM:.0f} km: {len(est)}")

    historicas = {k: v for k, v in est.items()
                  if len(set(v["meses"]) & set(huintil)) >= MIN_MESES_COMUNES}
    nuevas = {k: v for k, v in est.items()
              if max(v["meses"], default="") >= PRIMER_MES_A_RECONSTRUIR
              and min(v["meses"], default="9999") >= "2018-01"}
    log(f"  · con solape largo con Huintil (>= {MIN_MESES_COMUNES} meses): {len(historicas)}")
    log(f"  · que cubren el tramo a reconstruir (>= {PRIMER_MES_A_RECONSTRUIR}): {len(nuevas)}")

    # ---- V1 COLOCACIÓN -----------------------------------------------------
    log("\n" + "=" * 78)
    log("V1 · COLOCACIÓN — ¿cada estación nueva tiene una histórica en el MISMO sitio?")
    log(f"{'estación nueva':<44}{'histórica pareja':<26}{'sep.':>7}{'meses':>7}")
    pares = []
    for nl, nv in sorted(nuevas.items(), key=lambda x: x[1]["km"]):
        cand = sorted(((km(nv["lat"], nv["lon"], hv["lat"], hv["lon"]), hl)
                       for hl, hv in historicas.items()))
        if not cand:
            continue
        sep, hl = cand[0]
        nm = len(set(historicas[hl]["meses"]) & set(huintil))
        marca = "" if sep <= UMBRAL_COLOCACION_KM else "   <-- DESCARTADA (no es el mismo sitio)"
        log(f"{nl[:43]:<44}{hl[:25]:<26}{sep:>6.1f}k{nm:>7}{marca}")
        if sep <= UMBRAL_COLOCACION_KM:
            pares.append((nl, hl, sep))
    log(f"\nPares utilizables: {len(pares)}")
    if not pares:
        sys.exit("Sin pares co-localizados: no se puede encadenar. No se toca nada.")

    # ---- V2 LOOCV ----------------------------------------------------------
    log("\n" + "=" * 78)
    log("V2 · LOOCV — error propio del método, dejando un año fuera de la calibración")
    log(f"{'histórica':<26}{'años':>6}{'err.mediano':>13}{'err.medio':>11}{'sesgo':>9}   (mm/mes de lluvia)")
    calidad = {}
    for _nl, hl, _sep in pares:
        if hl in calidad:
            continue
        mens = historicas[hl]["meses"]
        anios = sorted({ym[:4] for ym in mens if ym in huintil})
        errs, sesgos, usados = [], [], 0
        for a in anios:
            razones, _ = razones_por_mes_calendario(huintil, mens, excluir_anio=a)
            if not razones:
                continue
            for ym, v in mens.items():
                if ym[:4] != a or ym not in huintil:
                    continue
                m = int(ym[5:7])
                if m not in razones:
                    continue
                pred = v * razones[m]
                real = huintil[ym]
                if real + pred < 5.0:      # meses secos: el error absoluto no informa
                    continue
                errs.append(abs(pred - real))
                sesgos.append(pred - real)
                usados += 1
        if errs:
            calidad[hl] = (st.median(errs), sum(errs) / len(errs), sum(sesgos) / len(sesgos), usados)
            med, prom, ses, n = calidad[hl]
            log(f"{hl[:25]:<26}{len(anios):>6}{med:>13.1f}{prom:>11.1f}{ses:>+9.1f}")

    # ---- V3 CADENA sobre 2018 ---------------------------------------------
    log("\n" + "=" * 78)
    log("V3 · CADENA COMPLETA — el único tramo donde una estación NUEVA solapa con Huintil")
    hubo_v3 = False
    for nl, hl, sep in pares:
        meses2018 = {ym: v for ym, v in nuevas[nl]["meses"].items() if ym[:4] == "2018" and ym in huintil}
        if len(meses2018) < 6:
            continue
        hubo_v3 = True
        razones, _ = razones_por_mes_calendario(huintil, historicas[hl]["meses"])
        log(f"\n  {nl}  (pareja: {hl}, separación {sep:.1f} km)")
        log(f"    {'mes':<9}{'Huintil medido':>16}{'predicho':>11}{'dif':>9}")
        tot_r = tot_p = 0.0
        for ym in sorted(meses2018):
            m = int(ym[5:7])
            if m not in razones:
                continue
            pred = meses2018[ym] * razones[m]
            real = huintil[ym]
            tot_r += real; tot_p += pred
            log(f"    {ym:<9}{real:>16.1f}{pred:>11.1f}{pred - real:>+9.1f}")
        log(f"    {'TOTAL':<9}{tot_r:>16.1f}{tot_p:>11.1f}{tot_p - tot_r:>+9.1f}"
            f"   ({100 * (tot_p - tot_r) / tot_r:+.0f}%)" if tot_r > 0 else "")
    if not hubo_v3:
        log("  (ninguna estación nueva solapa con Huintil -- la cadena queda sin verificación directa)")

    # ---- RECONSTRUCCIÓN ----------------------------------------------------
    log("\n" + "=" * 78)
    log("RECONSTRUCCIÓN — mediana de las estaciones disponibles en cada mes")
    razones_por_par = {}
    for nl, hl, _sep in pares:
        razones, _ = razones_por_mes_calendario(huintil, historicas[hl]["meses"])
        if razones:
            razones_por_par[nl] = razones

    meses_objetivo = sorted({ym for nl in razones_por_par for ym in nuevas[nl]["meses"]
                             if ym >= PRIMER_MES_A_RECONSTRUIR})
    recon = {}
    for ym in meses_objetivo:
        m = int(ym[5:7])
        ests = [nuevas[nl]["meses"][ym] * razones_por_par[nl][m]
                for nl in razones_por_par
                if ym in nuevas[nl]["meses"] and m in razones_por_par[nl]]
        if ests:
            recon[ym] = (st.median(ests), len(ests),
                         (max(ests) - min(ests)) if len(ests) > 1 else 0.0)

    log(f"Meses reconstruidos: {len(recon)}  ({min(recon)} a {max(recon)})")
    n_est = [v[1] for v in recon.values()]
    log(f"Estaciones por mes: mediana {st.median(n_est):.0f}, mínimo {min(n_est)}, máximo {max(n_est)}")

    with open(CSV_SALIDA, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["anio_mes", "lluvia_mm", "n_estaciones", "rango_estimaciones_mm", "origen"])
        for ym in sorted(recon):
            v, n, rango = recon[ym]
            w.writerow([ym, round(v, 2), n, round(rango, 1), "estaciones_reales_encadenadas"])
    log(f"\nCSV: {CSV_SALIDA}")

    # ---- V4 · EL ESCALÓN AL CRUZAR 2018 ------------------------------------
    # El supuesto "la DMC nueva mide como la histórica co-localizada" no se
    # puede validar directamente: ninguna estación cruza el corte. Pero se
    # puede acotar usando ERA5-Huintil como VARA COMÚN -- no como verdad, sólo
    # como una referencia fija presente a ambos lados. Si la razón
    # estación/ERA5 se mantiene al cruzar, los instrumentos son comparables.
    era5 = {r["anio_mes"]: float(r["lluvia_mm"]) for r in
            csv.DictReader(open(os.path.join(RAIZ, "investigacion/fuentes/lluvia_mensual_zhcs_openmeteo.csv"),
                                encoding="utf-8"))}

    def razon_contra_era5(meses, lo, hi):
        s = e = 0.0
        for ym, v in meses.items():
            if lo <= ym <= hi and ym in era5:
                s += v; e += era5[ym]
        return (s / e) if e > 0 else None

    log("\n" + "=" * 78)
    log("V4 · ESCALÓN AL CRUZAR EL CORTE DE 2018 (vara común: ERA5-Huintil)")
    log(f"{'histórica -> nueva':<56}{'R_hist':>8}{'R_nueva':>9}{'salto':>8}")
    saltos = {}
    for nl, hl, _sep in pares:
        rh = razon_contra_era5(historicas[hl]["meses"], "1966-01", "2017-05")
        rn = razon_contra_era5(nuevas[nl]["meses"], "2019-01", "2026-12")
        if rh and rn:
            saltos[nl] = rn / rh
            n_meses = sum(1 for ym in nuevas[nl]["meses"] if ym >= "2019-01")
            log(f"{(hl[:24] + ' -> ' + nl[:26]):<56}{rh:>8.2f}{rn:>9.2f}{rn / rh:>7.2f}x"
                f"   ({n_meses} meses)")

    # La Canela retén queda FUERA de la reconstrucción: es la única con salto
    # muy lejos del grupo, tiene apenas 8 meses en el tramo nuevo, y es la que
    # hizo fallar V3 (+105%). Se excluye por sus propios diagnósticos, no por
    # conveniencia -- si se la deja, arrastra la mediana de meses de 2019.
    limpios = {nl: s for nl, s in saltos.items() if "retén" not in nl}
    salto_mediano = st.median(limpios.values()) if limpios else 1.0
    log(f"\nExcluida de la cadena: La Canela retén (salto fuera del grupo, 8 meses, reventó V3)")
    log(f"Salto mediano de las {len(limpios)} restantes: {salto_mediano:.3f}x "
        f"(rango {min(limpios.values()):.2f}-{max(limpios.values()):.2f})")
    log("NO se puede saber si el salto es del instrumento (la DMC lee más alto) o")
    log("de ERA5 (derivó en el período reciente): ninguna estación cruza el corte.")
    log("Por eso se emiten DOS series y el modelo se corre con ambas, para ver si")
    log("la conclusión depende de esta elección o aguanta en todo el rango.")

    # ---- dos series: cruda y con el escalón descontado ---------------------
    recon2 = {}
    for ym in meses_objetivo:
        m = int(ym[5:7])
        ests = [nuevas[nl]["meses"][ym] * razones_por_par[nl][m]
                for nl in razones_por_par
                if "retén" not in nl and ym in nuevas[nl]["meses"] and m in razones_por_par[nl]]
        if ests:
            recon2[ym] = (st.median(ests), len(ests), max(ests) - min(ests) if len(ests) > 1 else 0.0)

    for sufijo, divisor, nota in ((("cruda"), 1.0, "estaciones_reales_encadenadas"),
                                  (("ajustada"), salto_mediano, "estaciones_reales_encadenadas_menos_escalon")):
        ruta = CSV_SALIDA.replace(".csv", f"_{sufijo}.csv")
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["anio_mes", "lluvia_mm", "n_estaciones", "rango_estimaciones_mm", "origen"])
            for ym in sorted(recon2):
                v, n, rango = recon2[ym]
                w.writerow([ym, round(v / divisor, 2), n, round(rango / divisor, 1), nota])
        log(f"  {sufijo:<9} -> {os.path.basename(ruta)}  ({len(recon2)} meses, {min(recon2)} a {max(recon2)})")
    with open(INFORME, "w", encoding="utf-8") as f:
        f.write("\n".join(BANDA) + "\n")
    print(f"Informe: {INFORME}")


if __name__ == "__main__":
    main()

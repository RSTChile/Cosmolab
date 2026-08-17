"""
CS090 — FASE V-B (pre-Phantom): VARIANZA DE GENEALOGÍA vs. VARIANZA DE REALIZACIÓN en A2-B0-C2
=================================================================================================
QUIÉN SOY: archivo NUEVO (ningún script congelado ni ninguno de los 5 archivos de la línea F5-C2-C se
toca: `cs090_fase5_presupuesto_emergente.py`, `cs090_fase5_presupuesto_soporte.py`,
`cs090_fase5_mecanismo_aislado.py`, `cs090_fase5_presupuesto_variable.py`,
`cs090_fase5_control_azar_elastico.py`; tampoco se tocan `cs090_fase5_generador.py`,
`cs090_fase5_motor.py`, `cs090_fase5_clasificador.py`, todos pipeline congelado).

LA PREGUNTA: paso 4 del roadmap del equipo para A2-B0-C2 pide "réplicas de genealogías A2 verdaderamente
independientes (no sólo semillas de la misma genealogía)". Las 5 tareas de la línea F5-C2-C (y también
`cs090_fase5_profundizar_a2b0c2.py` para su primer lote) usaron TODAS el mismo `seed_base` (90210 para
piloto / 90211 para completo) en `cs090_fase5_generador.generar_reglas_clase("A2","B0","C2", ...)`. Ese
lote de 20 reglas tiene 20 `seed` individuales distintos (por eso hay 20 reglas), pero las 20 provienen
de la MISMA llamada/genealogía del generador. Nunca se probó si el bimodal I/III (45.0% Clase III para
C2-hard en esa genealogía) es una propiedad del MECANISMO en general, o si es sensible a esa genealogía
particular.

ANALOGÍA: una "genealogía" acá es como una red social completamente distinta y separada (otro `seed_base`
-> otro conjunto de 20 familias de parámetros K/J/noise/meandeg/kcap/seed generadas desde cero); una
"realización" son los distintos días/estados de ESA MISMA red social (las 20 semillas dentro de un mismo
`seed_base`, que ya varían: unas caen en Clase I, otras en Clase III). Esta tarea genera 3-4 redes
sociales genuinamente distintas y mide si el ~45% de "días extendidos" (Clase III) se sostiene entre
redes, o si es un rasgo de una red social particular.

GENEALOGÍAS ELEGIDAS (ver §1 abajo para la justificación completa de independencia):
  G0 = 90210    -- la genealogía YA USADA en las 5 tareas de la línea F5-C2-C (C2-hard dio 45.0% ahí,
                   documentado en FASE5_control_azar_elastico_CS.md). Se RECALCULA acá con este mismo
                   pipeline (no se copia el número) como control de consistencia del script nuevo: si
                   G0 no reproduce ~45.0%, algo en este archivo está mal antes de mirar G1-G3.
  G1 = 471829, G2 = 823001, G3 = 156644 -- tres semillas nuevas, nunca usadas en ningún script de este
                   proyecto (grep verificado sobre cs090_fase5_*.py antes de escribir este archivo),
                   elegidas bien separadas entre sí y de G0: no son desplazamientos aritméticos simples
                   unas de otras (471829-90210=381619, 823001-90210=732791, 156644-90210=66434 -- sin
                   relación de múltiplo/submúltiplo entre ninguna de las 4), y abarcan distinto rango de
                   magnitud/dígitos. `np.random.default_rng` no tiene una estructura conocida que haga
                   que semillas "parecidas numéricamente" produzcan secuencias correlacionadas (PCG64),
                   así que la separación numérica es una precaución adicional, no un requisito
                   matemático estricto -- lo que realmente garantiza independencia es que cada
                   `seed_base` dispara una cadena de generación de parámetros TOTALMENTE separada dentro
                   de `generar_reglas_clase` (cada intento usa `seed = seed_base + intento*97 + 1`, y el
                   chequeo P1-P5 usa `seed + 500_000` -- ninguna de las 4 genealogías comparte ni un solo
                   `seed` individual con otra, verificable por construcción aritmética).

Piezas reusadas TAL CUAL, sin tocar una línea:
  - `cs090_fase5_generador.generar_reglas_clase()` -- genera + filtra P1-P5, igual que todas las tareas
    anteriores de Fase V.
  - `cs090_fase5_motor.correr_regla_coarse()` con `eje_C="C2"` -- el brazo C2-hard (`_enforce_kcap`),
    IDÉNTICO al usado en las 5 tareas de F5-C2-C y en el barrido/profundización de Fase V-A.
  - `cs090_fase5_mecanismo_aislado.correr_regla_coarse_hibrido(p, modo="soporte")` -- el brazo
    C2-hibrido, segunda verificación si el tiempo alcanza (ver CORRER_HIBRIDO abajo).
  - `cs090_fase5_clasificador.clasificar_regla()` -- mismos umbrales pre-registrados, sin tocar.

No se corre Phantom. No se declara cierre ni veredicto -- se reportan números por genealogía y la
comparación de varianzas; la lectura final es de Alexis. No se hacen commits de git.
"""
from __future__ import annotations
import csv, sys, time
import numpy as np
from collections import Counter

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)

import cs090_fase5_generador as GEN                # generar_reglas_clase() -- NO SE TOCA
import cs090_fase5_motor as MOT                     # correr_regla_coarse() -- NO SE TOCA
import cs090_fase5_mecanismo_aislado as MA          # correr_regla_coarse_hibrido() -- NO SE TOCA
from cs090_fase5_clasificador import clasificar_regla

EJE_A, EJE_B, EJE_C = "A2", "B0", "C2"
N_GRANDE = 2000
ESCALAS_B = (1, 2, 4, 8, 16)
N_SWEEPS = 14
N_SEEDS_NULL_TOPO = 3
N_REGLAS_POR_GENEALOGIA = 20

# ---------------------------------------------------------------------------------------------
# Las 4 genealogías -- ver docstring arriba para la justificación de independencia de cada una.
# Etiqueta -> seed_base.
# ---------------------------------------------------------------------------------------------
GENEALOGIAS = [
    ("G0_original_90210", 90210),     # la ya usada en toda la línea F5-C2-C -- control de consistencia
    ("G1_471829", 471829),
    ("G2_823001", 823001),
    ("G3_156644", 156644),
]

CORRER_HIBRIDO = True     # si el tiempo alcanza, también se corre C2-hibrido (segundo brazo, opcional)

OUT_RAW = f"{_HERE}/cs090_fase5_genealogias_independientes_resultados.csv"
OUT_RESUMEN = f"{_HERE}/cs090_fase5_genealogias_independientes_resumen.csv"
OUT_PILOTO_RAW = f"{_HERE}/cs090_fase5_genealogias_independientes_piloto_raw.csv"
OUT_PILOTO_RESUMEN = f"{_HERE}/cs090_fase5_genealogias_independientes_piloto_resumen.csv"
OUT_GENEALOGIAS = f"{_HERE}/cs090_fase5_genealogias_independientes_por_genealogia.csv"


# ============================================================================================
# 1) GENERAR + CORRER un lote (una genealogía) para un brazo dado
# ============================================================================================
def _correr_brazo(brazo, p):
    if brazo == "C2-hard":
        return MOT.correr_regla_coarse(p, N=N_GRANDE, n_sweeps=N_SWEEPS, escalas_b=ESCALAS_B,
                                        n_seeds_null_topo=N_SEEDS_NULL_TOPO)
    elif brazo == "C2-hibrido":
        return MA.correr_regla_coarse_hibrido(p, modo="soporte")   # usa N_GRANDE/ESCALAS_B/etc de MA (idénticos)
    else:
        raise ValueError(brazo)


def correr_genealogia(etiqueta, seed_base, n_reglas, brazos, presupuesto_seg, sufijo_log=""):
    """Genera n_reglas admitidas (filtro P1-P5 real) para (A2,B0,C2) con este seed_base, las corre en
    cada brazo pedido, clasifica, y devuelve (filas_raw, filas_resumen, admitidas, descartadas)."""
    t_inicio = time.time()
    admitidas, descartadas = GEN.generar_reglas_clase(
        EJE_A, EJE_B, EJE_C, n_reglas=n_reglas, seed_base=seed_base, max_intentos=max(80, n_reglas * 4))
    print(f"[{etiqueta}{sufijo_log}] seed_base={seed_base}  admitidas={len(admitidas)}/{n_reglas}  "
          f"descartadas(P1-P5)={len(descartadas)}")
    if descartadas:
        for d in descartadas:
            print(f"    descartada {d['rule_id']}: {d['motivo_descarte']}")

    filas_raw, filas_resumen = [], []
    for p in admitidas:
        for brazo in brazos:
            t0 = time.time()
            try:
                filas = _correr_brazo(brazo, p)
            except Exception as e:
                print(f"    *** FALLO {brazo} en {p['rule_id']}: {type(e).__name__}: {e} ***")
                continue
            r = clasificar_regla(filas)
            dt = time.time() - t0
            fila_b1 = next(f for f in filas if f["escala_b"] == 1)
            grado_medio_b1 = 2 * fila_b1["n_aristas"] / fila_b1["N"] if fila_b1["N"] else float("nan")
            for f in filas:
                f2 = dict(f); f2.update(genealogia=etiqueta, seed_base=seed_base, brazo=brazo,
                                         clase_final=r["clase"])
                filas_raw.append(f2)
            filas_resumen.append(dict(
                genealogia=etiqueta, seed_base=seed_base, rule_id=p["rule_id"], brazo=brazo,
                clase=r["clase"], pendiente=r["pendiente_real"], z_agg=r["z_agg"],
                holon_ratio=r["holon_ratio"],
                n_aristas_b1=fila_b1["n_aristas"], grado_medio_b1=round(grado_medio_b1, 3),
                diam_b1=fila_b1["diam_real"], giant_b1=fila_b1["giant_real"],
                K=p["K"], J=p["J"], noise=p["noise"], meandeg=p["meandeg"], kcap=p["kcap"], seed=p["seed"],
            ))
            print(f"  [{etiqueta}] {p['rule_id']:<16} {brazo:<12} clase={r['clase']:<24} "
                  f"pendiente={r['pendiente_real']:.3f} grado_medio={grado_medio_b1:.2f} "
                  f"diam_b1={fila_b1['diam_real']:.2f} ({dt:.2f}s) [t_total={time.time()-t_inicio:.0f}s]")
        if time.time() - t_inicio > presupuesto_seg:
            print(f"  *** [{etiqueta}] SALVAGUARDA DE TIEMPO alcanzada, se corta esta genealogía ***")
            break
    return filas_raw, filas_resumen, admitidas, descartadas


# ============================================================================================
# 2) RESUMEN por genealogía × brazo
# ============================================================================================
def resumen_por_genealogia(filas_resumen, brazos):
    out = {}
    for etiqueta, seed_base in GENEALOGIAS:
        for brazo in brazos:
            fb = [f for f in filas_resumen if f["genealogia"] == etiqueta and f["brazo"] == brazo]
            if not fb:
                continue
            cnt = Counter(f["clase"] for f in fb)
            n = len(fb)
            out[(etiqueta, brazo)] = dict(
                genealogia=etiqueta, seed_base=seed_base, brazo=brazo, n=n, cnt=dict(cnt),
                frac_III=100 * cnt.get("III", 0) / n,
                frac_IIImasIV=100 * (cnt.get("III", 0) + cnt.get("IV", 0)) / n,
                grado_medio=float(np.mean([f["grado_medio_b1"] for f in fb])),
                diam_medio=float(np.mean([f["diam_b1"] for f in fb])),
                pendiente_media=float(np.mean([f["pendiente"] for f in fb])),
                pendiente_mediana=float(np.median([f["pendiente"] for f in fb])),
            )
    return out


def imprimir_resumen_genealogias(titulo, resumen, brazos):
    print("\n" + "=" * 120)
    print(titulo)
    print("=" * 120)
    for brazo in brazos:
        print(f"\n  -- brazo {brazo} --")
        fracs = []
        for etiqueta, seed_base in GENEALOGIAS:
            r = resumen.get((etiqueta, brazo))
            if not r:
                continue
            fracs.append(r["frac_III"])
            print(f"    {etiqueta:<20} seed_base={seed_base:<8} n={r['n']:<3} clases={r['cnt']}  "
                  f"%Clase_III={r['frac_III']:5.1f}%  %III+IV={r['frac_IIImasIV']:5.1f}%  "
                  f"grado_medio={r['grado_medio']:.2f}  diam_medio={r['diam_medio']:.2f}  "
                  f"pendiente_media={r['pendiente_media']:.3f}  pendiente_mediana={r['pendiente_mediana']:.3f}")
        if len(fracs) >= 2:
            media = float(np.mean(fracs)); std = float(np.std(fracs))
            print(f"    >> ENTRE genealogías ({len(fracs)}): media %Clase_III={media:.1f}%  "
                  f"std={std:.2f}pp  min={min(fracs):.1f}%  max={max(fracs):.1f}%  rango={max(fracs)-min(fracs):.1f}pp")


def guardar_csv(filas, ruta, campos=None):
    if not filas:
        print(f"(sin filas para {ruta})")
        return
    campos = campos or list(filas[0].keys())
    with open(ruta, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=campos)
        wr.writeheader()
        for f in filas:
            wr.writerow(f)
    print(f"CSV: {ruta}  ({len(filas)} filas)")


# ============================================================================================
# 3) DRIVER
# ============================================================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=["piloto", "completo"], default="piloto")
    args = ap.parse_args()

    if args.modo == "piloto":
        # Piloto chico: 3 semillas x 1 genealogía NUEVA (G1) x ambos brazos -- confirma que el
        # generador acepta seed_base arbitrarios sin romperse y que el filtro P1-P5 sigue admitiendo
        # ~20/20 con otro seed_base (acá con n_reglas=3 para que sea barato, extrapola la tasa).
        t0 = time.time()
        etiqueta, seed_base = "G1_471829", 471829
        brazos = ["C2-hard", "C2-hibrido"] if CORRER_HIBRIDO else ["C2-hard"]
        filas_raw, filas_resumen, admitidas, descartadas = correr_genealogia(
            etiqueta, seed_base, n_reglas=3, brazos=brazos, presupuesto_seg=10 * 60, sufijo_log=" PILOTO")
        resumen = resumen_por_genealogia(filas_resumen, brazos)
        imprimir_resumen_genealogias("PILOTO -- resumen por genealogía/brazo", resumen, brazos)
        guardar_csv(filas_raw, OUT_PILOTO_RAW)
        guardar_csv(filas_resumen, OUT_PILOTO_RESUMEN)
        print(f"\nPILOTO terminado en {(time.time()-t0)/60:.2f} min. "
              f"Filtro P1-P5 con seed_base={seed_base}: admitidas={len(admitidas)}/3, "
              f"descartadas={len(descartadas)} -- {'OK, filtro funciona con seed_base nuevo' if len(admitidas)==3 else 'REVISAR'}")
    else:
        t0 = time.time()
        brazos = ["C2-hard", "C2-hibrido"] if CORRER_HIBRIDO else ["C2-hard"]
        PRESUPUESTO_POR_GENEALOGIA = 12 * 60   # 4 genealogías x 12 min = 48 min, dentro del presupuesto 50-60

        todas_raw, todas_resumen = [], []
        info_filtro = []
        for etiqueta, seed_base in GENEALOGIAS:
            filas_raw, filas_resumen, admitidas, descartadas = correr_genealogia(
                etiqueta, seed_base, n_reglas=N_REGLAS_POR_GENEALOGIA, brazos=brazos,
                presupuesto_seg=PRESUPUESTO_POR_GENEALOGIA)
            todas_raw += filas_raw
            todas_resumen += filas_resumen
            info_filtro.append((etiqueta, seed_base, len(admitidas), len(descartadas)))
            if time.time() - t0 > 55 * 60:
                print("*** SALVAGUARDA DE TIEMPO GLOBAL alcanzada (55 min), se corta el barrido de genealogías ***")
                break

        print("\n" + "=" * 120)
        print("VERIFICACIÓN DEL FILTRO P1-P5 CON SEED_BASE ARBITRARIOS")
        print("=" * 120)
        for etiqueta, seed_base, n_adm, n_desc in info_filtro:
            print(f"  {etiqueta:<20} seed_base={seed_base:<8} admitidas={n_adm}/{N_REGLAS_POR_GENEALOGIA}  descartadas={n_desc}")

        resumen = resumen_por_genealogia(todas_resumen, brazos)
        imprimir_resumen_genealogias("COMPLETO -- resumen por genealogía/brazo", resumen, brazos)

        # -------- filas "por genealogía" (una fila por genealogia x brazo) a CSV aparte, para el informe --------
        filas_genealogias_csv = []
        for (etiqueta, brazo), r in resumen.items():
            filas_genealogias_csv.append(dict(
                genealogia=r["genealogia"], seed_base=r["seed_base"], brazo=r["brazo"], n=r["n"],
                n_claseI=r["cnt"].get("I", 0), n_claseII=r["cnt"].get("II", 0),
                n_claseIII=r["cnt"].get("III", 0), n_claseIV=r["cnt"].get("IV", 0),
                pct_claseIII=round(r["frac_III"], 2), pct_claseIII_mas_IV=round(r["frac_IIImasIV"], 2),
                grado_medio=round(r["grado_medio"], 3), diam_medio=round(r["diam_medio"], 3),
                pendiente_media=round(r["pendiente_media"], 4), pendiente_mediana=round(r["pendiente_mediana"], 4),
            ))

        guardar_csv(todas_raw, OUT_RAW)
        guardar_csv(todas_resumen, OUT_RESUMEN)
        guardar_csv(filas_genealogias_csv, OUT_GENEALOGIAS)

        # -------- varianza ENTRE genealogías vs. varianza YA CONOCIDA dentro de una genealogía --------
        print("\n" + "=" * 120)
        print("VARIANZA ENTRE GENEALOGÍAS (esta tarea) vs. VARIANZA DENTRO-DE-GENEALOGÍA (ya conocida, 20 semillas)")
        print("=" * 120)
        for brazo in brazos:
            fracs = [resumen[(et, brazo)]["frac_III"] for et, sb in GENEALOGIAS if (et, brazo) in resumen]
            if len(fracs) >= 2:
                print(f"  {brazo}: %Clase_III por genealogía = {[round(f,1) for f in fracs]}")
                print(f"    media entre-genealogías={np.mean(fracs):.1f}%  std entre-genealogías={np.std(fracs):.2f}pp  "
                      f"rango={max(fracs)-min(fracs):.1f}pp")
        print("  Referencia intra-genealogía YA CONOCIDA (n=20, genealogía 90210/90211, de FASE5_control_azar_elastico_CS.md "
              "y FASE5_matriz_2x2_completa_CS.md): C2-hard dio 45.0% Clase III sobre esas 20 semillas -- dentro de ESA MISMA "
              "genealogía, 9/20 seeds cayeron en III y 11/20 en I/II/otro (variación seed-a-seed ya documentada).")

        print(f"\nCOMPLETO terminado en {(time.time()-t0)/60:.1f} min")
        print("Fin. No se declara cierre ni veredicto -- números arriba, lectura final de Alexis.")

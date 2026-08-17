"""
CS090 — FASE V-A: PROFUNDIZAR A2-B0-C2 (candidato bimodal más limpio del barrido de 180 reglas)
=================================================================================================
QUIÉN SOY: archivo NUEVO (no toca ningún script congelado) que profundiza específicamente en la
combinación A2-B0-C2 (grafo dinámico co-emergente, SIN retroalimentación relación-sobre-relación,
CON límite de escala duro), que en el barrido completo (`FASE5A_completo_resultado_CS.md` §5) dio el
patrón bimodal más limpio de las 18 combinaciones: 5/10 Clase III, 5/10 Clase I, 0/10 Clase II.

Tres objetivos (pedidos por Alexis, ver tarea):
  1. ¿El bimodal 50/50 se sostiene con más muestra (15-20 reglas ADICIONALES, mismos ejes fijos,
     parámetros aleatorios nuevos)? Se generan con `cs090_fase5_generador.generar_reglas_clase()` tal
     cual (mismo método usado para las 10 originales), se corren con
     `cs090_fase5_motor.correr_regla_coarse()` tal cual, se clasifican con
     `cs090_fase5_clasificador.clasificar_regla()` tal cual.
  2. ¿Algún parámetro de los que varía el generador dentro de A2-B0-C2 (K, J, noise, meandeg, kcap)
     separa las reglas que caen en Clase III de las que caen en Clase I? Sólo se puede responder sobre
     las reglas NUEVAS de esta tarea, porque el resumen del barrido original
     (`cs090_fase5_completo_resumen.csv`) no guardó los parámetros por regla — sólo clase/pendiente/
     z_agg/holon_ratio. Se documenta esta limitación, no se inventan los parámetros originales.
  3. ¿A2-B0-C2 comparte la duda metodológica de A0 (que el motor deriva un "grafo de medición" separado
     de la dinámica para poder medir diám/giant, y ESE grafo podría estar generando por sí solo algo de
     escalamiento)? Se responde por INSPECCIÓN DE CÓDIGO de `cs090_fase5_motor.py` (función `medir()`),
     no por opinión — ver bloque de texto al final de este archivo y el informe .md.

Piezas reusadas TAL CUAL, sin tocar ningún archivo: `cs090_fase5_generador.generar_reglas_clase()`,
`cs090_fase5_motor.correr_regla_coarse()`, `cs090_fase5_clasificador.clasificar_regla()`. Mismos
parámetros de ejecución que usó el driver `cs090_fase5_completo.py` para A2-B0-C2 (N=2000, sweeps=14,
escalas_b=(1,2,4,8,16), n_seeds_null_topo=3) — para que el resultado sea directamente comparable con
las 10 reglas originales.

No se corre Phantom. No se declara cierre ni veredicto — se reportan números, la lectura final es de
Alexis. No se hacen commits de git.
"""
from __future__ import annotations
import csv, time, sys
from collections import Counter

sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
import cs090_fase5_generador as GEN
import cs090_fase5_motor as MOT
from cs090_fase5_clasificador import clasificar_regla

EJE_A, EJE_B, EJE_C = "A2", "B0", "C2"
N_NUEVAS = 20                 # reglas ADICIONALES a generar (objetivo 15-20, se apunta al techo)
SEED_BASE = 271828             # seed distinto del usado en el barrido original (que usaba hash() no
                                # reproducible entre procesos) -- reproducible siempre desde este archivo
N_GRANDE = 2000
ESCALAS_B = (1, 2, 4, 8, 16)
N_SWEEPS = 14
N_SEEDS_NULL_TOPO = 3
PRESUPUESTO_SEG = 30 * 60       # salvaguarda de tiempo generosa (el motor corrió ~1.8s/regla en el
                                 # smoke test) -- prioriza terminar el barrido de 20 reglas + reporte

RUTA_RESUMEN_ORIGINAL = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5_completo_resumen.csv"
OUT_CSV_RAW = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5_profundizar_a2b0c2_resultados.csv"
OUT_CSV_RESUMEN = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5_profundizar_a2b0c2_resumen.csv"

PARAMS_A_GUARDAR = ["K", "J", "noise", "meandeg", "kcap", "sim_thr_frac", "seed"]


def cargar_10_originales():
    """Lee las 10 reglas A2-B0-C2 YA corridas en el barrido de 180 (cs090_fase5_completo_resumen.csv).
    Sólo tiene clase/pendiente/z_agg/holon_ratio -- el barrido original no guardó los parámetros por
    regla, así que esas columnas quedan vacías para estas filas (se documenta, no se inventa)."""
    filas = []
    with open(RUTA_RESUMEN_ORIGINAL) as fh:
        for row in csv.DictReader(fh):
            if row["combo"] == f"{EJE_A}-{EJE_B}-{EJE_C}":
                filas.append(dict(
                    origen="original_barrido180", rule_id=row["rule_id"], clase=row["clase"],
                    pendiente=row["pendiente"], z_agg=row["z_agg"], holon_ratio=row["holon_ratio"],
                    K="", J="", noise="", meandeg="", kcap="", sim_thr_frac="", seed="",
                ))
    return filas


def main():
    print("=" * 100)
    print(f"CS090 — Profundizar {EJE_A}-{EJE_B}-{EJE_C}: {N_NUEVAS} reglas ADICIONALES (objetivos 1+2), "
          f"más chequeo de código (objetivo 3)")
    print("=" * 100)

    t_inicio = time.time()

    # -------------------- generar N_NUEVAS reglas admitidas (filtro P1-P5 real) --------------------
    admitidas, descartadas = GEN.generar_reglas_clase(
        EJE_A, EJE_B, EJE_C, n_reglas=N_NUEVAS, seed_base=SEED_BASE, max_intentos=80)
    print(f"\nGeneradas: admitidas={len(admitidas)}  descartadas(filtro P1-P5)={len(descartadas)}")
    if descartadas:
        for d in descartadas:
            print(f"  descartada {d['rule_id']}: {d['motivo_descarte']}")

    filas_raw = []
    filas_resumen_nuevas = []
    for p in admitidas:
        t0 = time.time()
        try:
            filas_regla = MOT.correr_regla_coarse(
                p, N=N_GRANDE, n_sweeps=N_SWEEPS, escalas_b=ESCALAS_B,
                n_seeds_null_topo=N_SEEDS_NULL_TOPO)
        except Exception as e:
            print(f"  {p['rule_id']}: *** FALLO DEL MOTOR (no se toca cs090_fase5_motor.py) *** "
                  f"{type(e).__name__}: {e}")
            continue
        r = clasificar_regla(filas_regla)
        dt = time.time() - t0
        for fr in filas_regla:
            fr2 = dict(fr); fr2.update(eje_A=EJE_A, eje_B=EJE_B, eje_C=EJE_C,
                                        clase_final=r["clase"])
            filas_raw.append(fr2)
        resumen = dict(origen="nueva_profundizar", rule_id=p["rule_id"], clase=r["clase"],
                        pendiente=r["pendiente_real"], z_agg=r["z_agg"], holon_ratio=r["holon_ratio"])
        for k in PARAMS_A_GUARDAR:
            resumen[k] = p[k]
        filas_resumen_nuevas.append(resumen)
        print(f"  {p['rule_id']}: clase={r['clase']:<24} pendiente={r['pendiente_real']:.3f} "
              f"z_agg={r['z_agg']:.2f} holon_ratio={r['holon_ratio']:.2f} "
              f"K={p['K']} J={p['J']} noise={p['noise']} meandeg={p['meandeg']} kcap={p['kcap']} "
              f"({dt:.1f}s) [t_total={time.time()-t_inicio:.0f}s]")
        if time.time() - t_inicio > PRESUPUESTO_SEG:
            print("  *** SALVAGUARDA DE TIEMPO: presupuesto alcanzado, se corta el barrido de reglas nuevas ***")
            break

    t_fin_barrido = time.time()
    print(f"\nBarrido de reglas nuevas terminado en {t_fin_barrido-t_inicio:.1f}s "
          f"({len(filas_resumen_nuevas)} reglas nuevas corridas)")

    # -------------------- CSV crudo (sólo reglas nuevas de esta tarea) --------------------
    if filas_raw:
        campos = list(filas_raw[0].keys())
        with open(OUT_CSV_RAW, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=campos)
            wr.writeheader()
            for f in filas_raw:
                wr.writerow(f)
    print(f"CSV crudo (nuevas): {OUT_CSV_RAW}  ({len(filas_raw)} filas)")

    # -------------------- CSV resumen combinado (10 originales + N nuevas) --------------------
    originales = cargar_10_originales()
    todas = originales + filas_resumen_nuevas
    campos_resumen = ["origen", "rule_id", "clase", "pendiente", "z_agg", "holon_ratio"] + PARAMS_A_GUARDAR
    with open(OUT_CSV_RESUMEN, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=campos_resumen)
        wr.writeheader()
        for f in todas:
            wr.writerow(f)
    print(f"CSV resumen combinado (originales+nuevas): {OUT_CSV_RESUMEN}  ({len(todas)} filas)")

    # ======================================================================
    # OBJETIVO 1 -- ¿se sostiene el bimodal con más muestra?
    # ======================================================================
    print("\n" + "=" * 100)
    print("OBJETIVO 1 -- patrón bimodal con muestra ampliada")
    print("=" * 100)
    cnt_originales = Counter(f["clase"] for f in originales)
    cnt_nuevas = Counter(f["clase"] for f in filas_resumen_nuevas)
    cnt_todas = Counter(f["clase"] for f in todas)
    print(f"  10 originales (barrido de 180): {dict(cnt_originales)}")
    print(f"  {len(filas_resumen_nuevas)} nuevas (esta tarea):        {dict(cnt_nuevas)}")
    print(f"  TOTAL combinado (n={len(todas)}):        {dict(cnt_todas)}")
    n_total = len(todas)
    if n_total:
        for clase in ("I", "II", "III", "IV"):
            pct = 100 * cnt_todas.get(clase, 0) / n_total
            print(f"    Clase {clase}: {cnt_todas.get(clase,0)}/{n_total} = {pct:.1f}%")

    # ======================================================================
    # OBJETIVO 2 -- ¿algún parámetro separa Clase I de Clase III? (sólo sobre reglas NUEVAS,
    # que son las únicas con parámetros guardados)
    # ======================================================================
    print("\n" + "=" * 100)
    print("OBJETIVO 2 -- ¿qué parámetro separa Clase I de Clase III? (sólo sobre reglas nuevas, con params)")
    print("=" * 100)
    grupo_I = [f for f in filas_resumen_nuevas if f["clase"] == "I"]
    grupo_III = [f for f in filas_resumen_nuevas if f["clase"] == "III"]
    otros = [f for f in filas_resumen_nuevas if f["clase"] not in ("I", "III")]
    print(f"  Clase I: n={len(grupo_I)}   Clase III: n={len(grupo_III)}   otras clases (II/IV/intermedio): n={len(otros)}")
    if otros:
        for f in otros:
            print(f"    {f['rule_id']}: clase={f['clase']}")

    def _stats(grupo, campo):
        vals = [g[campo] for g in grupo]
        if not vals:
            return None
        return (min(vals), max(vals), sum(vals) / len(vals))

    if grupo_I and grupo_III:
        print(f"\n  {'parametro':<14} {'I: min':>8} {'I: max':>8} {'I: media':>9}   "
              f"{'III: min':>9} {'III: max':>9} {'III: media':>11}   separación")
        for campo in ["K", "J", "noise", "meandeg", "kcap"]:
            si = _stats(grupo_I, campo)
            siii = _stats(grupo_III, campo)
            solapa = not (si[1] < siii[0] or siii[1] < si[0])  # rangos se solapan?
            sep = "SIN solape (separación limpia)" if not solapa else "con solape"
            print(f"  {campo:<14} {si[0]:>8.3f} {si[1]:>8.3f} {si[2]:>9.3f}   "
                  f"{siii[0]:>9.3f} {siii[1]:>9.3f} {siii[2]:>11.3f}   {sep}")
    else:
        print("  No hay suficientes reglas en ambas clases (I y III) entre las nuevas para comparar.")

    print(f"\nTiempo total del script: {(time.time()-t_inicio)/60:.1f} min")
    print("Fin. No se declara cierre ni veredicto -- números arriba, lectura final de Alexis.")


if __name__ == "__main__":
    main()

# ============================================================================================
# OBJETIVO 3 -- ¿A2-B0-C2 comparte la duda metodológica de "grafo de medición derivado" de A0?
# (documentado acá como comentario porque es una inspección de código, no un cómputo -- se resume
# también en el informe .md)
# ============================================================================================
# En `cs090_fase5_motor.py`, función `medir(sustrato, p, rng)` (línea ~328):
#   - Para kind == "A0": llama a `_grafo_medicion_A0(sustrato["S"], p["K"], p["sim_thr_frac"], rng)` --
#     un grafo CONSTRUIDO DESDE CERO sólo para poder medir diám/giant, comparando similitud de estado
#     entre pares de sitios muestreados. Este grafo NUNCA participó de la dinámica (que en A0 es
#     difusión local en anillo vía offsets left/right, sin ningún objeto de adyacencia). El parámetro
#     `sim_thr_frac` (umbral de similitud) sólo existe para esta construcción post-hoc.
#   - Para kind in ("A1", "A2"): la rama `else` usa directamente `adj = sustrato["adj"]` -- el MISMO
#     objeto de adyacencia que usó `dinamica_B0`/`dinamica_B1` para correr la dinámica real (incluida,
#     para A2, la co-evolución por `_recablear_A2` y la poda por `_enforce_kcap` bajo C2). No se
#     construye ningún grafo adicional sólo para medir.
#   - El parámetro `sim_thr_frac` SÍ se genera para toda regla (viene del espacio de parámetros
#     genérico del generador), pero para A2 nunca se lee en ningún punto del motor (grep confirma que
#     sólo aparece en `_grafo_medicion_A0`) -- es un parámetro inerte para A2, no un artefacto activo.
#
# CONCLUSIÓN (código, no opinión): A2-B0-C2 NO comparte la duda metodológica de A0. El grafo que se
# mide (diám/giant/coarse-graining) en A2 ES el mismo grafo genuino sobre el que corrió la dinámica y
# la poda por costo/kcap -- no hay una vara de medición añadida por fuera del sustrato. La duda de A0
# (que el "mundo pequeño" medido sea un artefacto de cómo se mide, no de cómo se comporta el sustrato)
# no aplica estructuralmente acá, porque no existe una construcción de grafo equivalente a
# `_grafo_medicion_A0` en la rama A1/A2 de `medir()`.

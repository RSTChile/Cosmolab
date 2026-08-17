"""
CS090 — FASE V-A: DRIVER COMPLETO (18 clases x 10 reglas = 180 reglas, N=2000, coarse-graining)
=================================================================================================
QUIÉN SOY: escala `cs090_fase5_piloto_v2.py` (que ya validó el pipeline en 5 clases x 3 reglas) a las
18 clases completas de la especificación (3 ejes A0/A1/A2 x B0/B1 x C0/C1/C2) x 10 reglas por clase,
usando EXACTAMENTE las mismas piezas ya validadas:

  - `cs090_fase5_generador.generar_reglas_clase()` -- genera + aplica filtro P1-P5, sin tocar el archivo.
  - `cs090_fase5_motor.correr_regla_coarse()` -- motor corregido (coarse-graining de UN grafo a N=2000,
    escalas b=1/2/4/8/16, reusando cajas_bfs/grafo_grueso de cs080_renormalizacion.py TAL CUAL).
  - `cs090_fase5_clasificador.clasificar_regla()` -- umbrales YA FIJADOS en la especificación §4, sin
    inventar ninguno nuevo acá.

Ningún script congelado se toca (cs080/81/82/83, cs090_fase5_generador/motor/clasificador) -- sólo se
importa. Este es el ÚNICO archivo nuevo de esta tarea (más el CSV de salida y el informe .md).

SALVAGUARDA DE TIEMPO (pedida explícitamente por Alexis en la tarea): presupuesto ~55-65 min. Si el
tiempo transcurrido se acerca al límite, el driver PRIORIZA completar las 18 clases con al menos 5
reglas cada una (90 reglas) antes que forzar las 10 completas y quedar a medias en alguna clase --
implementado corriendo primero una "pasada 1" de 5 reglas por clase para las 18 clases, y sólo si sobra
tiempo corriendo una "pasada 2" que completa hasta 10 por clase. Así, si hay que cortar, el corte cae
entre pasadas (nunca a medio completar una clase con menos de 5).

No corre Phantom. No declara cierre ni veredicto sobre S>0 -- reporta números, la lectura final y
cualquier interpretación sobre la Teoría es de Alexis.
"""
from __future__ import annotations
import csv, time, sys
from collections import Counter

sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
import cs090_fase5_generador as GEN
import cs090_fase5_motor as MOT
from cs090_fase5_clasificador import clasificar_regla

EJES_A = ("A0", "A1", "A2")
EJES_B = ("B0", "B1")
EJES_C = ("C0", "C1", "C2")
TODOS_COMBOS = [(a, b, c) for a in EJES_A for b in EJES_B for c in EJES_C]  # 18 combos

N_REGLAS_PASO1 = 5          # primero garantizar >=5 por clase (90 reglas) -- prioridad de la salvaguarda
N_REGLAS_OBJETIVO = 10       # objetivo completo por clase si el tiempo alcanza
N_GRANDE = 2000
ESCALAS_B = (1, 2, 4, 8, 16)
N_SWEEPS = 14
N_SEEDS_NULL_TOPO = 3
PRESUPUESTO_SEG = 60 * 60     # 60 min -- punto medio del rango 55-65 min pedido

OUT_CSV = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5_completo_resultados.csv"
OUT_CSV_RESUMEN = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5_completo_resumen.csv"


def correr_n_reglas_extra(eje_A, eje_B, eje_C, n_ya, n_objetivo, seed_base):
    """Genera y corre reglas ADICIONALES para llegar de n_ya a n_objetivo (usa un seed_base distinto
    por tramo para no repetir semillas ya usadas)."""
    n_faltan = n_objetivo - n_ya
    if n_faltan <= 0:
        return [], []
    admitidas, descartadas = GEN.generar_reglas_clase(
        eje_A, eje_B, eje_C, n_reglas=n_faltan, seed_base=seed_base)
    return admitidas, descartadas


def correr_regla_y_clasificar(p, eje_A, eje_B, eje_C):
    t0 = time.time()
    filas_regla = MOT.correr_regla_coarse(p, N=N_GRANDE, n_sweeps=N_SWEEPS,
                                            escalas_b=ESCALAS_B, n_seeds_null_topo=N_SEEDS_NULL_TOPO)
    r = clasificar_regla(filas_regla)
    dt = time.time() - t0
    filas_out = []
    for fr in filas_regla:
        fr2 = dict(fr); fr2.update(eje_A=eje_A, eje_B=eje_B, eje_C=eje_C,
                                    clase_final=r["clase"], pendiente_real=r["pendiente_real"],
                                    z_agg=r["z_agg"], holon_ratio=r["holon_ratio"])
        filas_out.append(fr2)
    return r, filas_out, dt


def main():
    print("=" * 100)
    print("CS090 — FASE V-A COMPLETA: 18 clases x hasta 10 reglas (objetivo 180 reglas)")
    print("=" * 100)

    t_inicio = time.time()
    filas_csv = []
    total_admitidas, total_descartadas = 0, 0
    # por combo: lista de (rule_id, clase, pendiente, z_agg, holon_ratio)
    resultados_por_combo = {c: [] for c in TODOS_COMBOS}
    # combos que el motor congelado no puede ejecutar (bug real, no se toca el motor -- se documenta)
    combos_rotos = {}  # combo -> mensaje de error (de la primera regla que falló)

    # -------------------- PASADA 1: garantizar >=5 reglas por clase (90 reglas) --------------------
    print(f"\n### PASADA 1 -- {N_REGLAS_PASO1} reglas por clase x 18 clases "
          f"({N_REGLAS_PASO1*18} reglas) ###")
    cortado_en_paso1 = False
    for (eje_A, eje_B, eje_C) in TODOS_COMBOS:
        combo_str = f"{eje_A}-{eje_B}-{eje_C}"
        seed_base = abs(hash((eje_A, eje_B, eje_C, "paso1"))) % 100000
        admitidas, descartadas = GEN.generar_reglas_clase(
            eje_A, eje_B, eje_C, n_reglas=N_REGLAS_PASO1, seed_base=seed_base)
        total_admitidas += len(admitidas); total_descartadas += len(descartadas)
        print(f"\n--- {combo_str} (paso1, admitidas={len(admitidas)} descartadas={len(descartadas)}) ---")
        fallos_seguidos = 0
        for p in admitidas:
            try:
                r, filas_out, dt = correr_regla_y_clasificar(p, eje_A, eje_B, eje_C)
            except Exception as e:
                print(f"  {p['rule_id']}: *** FALLO DEL MOTOR (no se toca cs090_fase5_motor.py, se "
                      f"documenta y se sigue) *** {type(e).__name__}: {e}")
                combos_rotos.setdefault(combo_str, f"{type(e).__name__}: {e}")
                fallos_seguidos += 1
                if fallos_seguidos >= 2:
                    print(f"  -- {combo_str}: 2 fallos seguidos, se asume bug sistemático del motor para "
                          f"esta combinación de ejes, se corta el resto de reglas de este combo --")
                    break
                continue
            fallos_seguidos = 0
            filas_csv.extend(filas_out)
            resultados_por_combo[(eje_A, eje_B, eje_C)].append(
                dict(rule_id=p["rule_id"], clase=r["clase"], pendiente=r["pendiente_real"],
                     z_agg=r["z_agg"], holon_ratio=r["holon_ratio"]))
            print(f"  {p['rule_id']}: clase={r['clase']:<24} pendiente={r['pendiente_real']:.3f} "
                  f"z_agg={r['z_agg']:.2f} holon_ratio={r['holon_ratio']:.2f} ({dt:.1f}s)"
                  f"  [t_total={time.time()-t_inicio:.0f}s]")
            if time.time() - t_inicio > PRESUPUESTO_SEG:
                print("  *** SALVAGUARDA DE TIEMPO: presupuesto alcanzado a mitad de PASADA 1 ***")
                cortado_en_paso1 = True
                break
        if cortado_en_paso1:
            break

    t_fin_paso1 = time.time()
    print(f"\n### FIN PASADA 1 -- {t_fin_paso1-t_inicio:.0f}s transcurridos "
          f"({(t_fin_paso1-t_inicio)/60:.1f} min) ###")

    # -------------------- PASADA 2: completar hasta 10 por clase, si el tiempo alcanza --------------------
    cortado_en_paso2 = False
    if cortado_en_paso1:
        print("\n### PASADA 2 OMITIDA -- ya se cortó por tiempo en pasada 1 ###")
    else:
        print(f"\n### PASADA 2 -- completar hasta {N_REGLAS_OBJETIVO} reglas por clase (si alcanza el tiempo) ###")
        for (eje_A, eje_B, eje_C) in TODOS_COMBOS:
            combo_str = f"{eje_A}-{eje_B}-{eje_C}"
            if combo_str in combos_rotos:
                print(f"\n--- {combo_str} (paso2, OMITIDO -- combo marcado roto en paso1: "
                      f"{combos_rotos[combo_str]}) ---")
                continue
            n_ya = len(resultados_por_combo[(eje_A, eje_B, eje_C)])
            if time.time() - t_inicio > PRESUPUESTO_SEG:
                print(f"  *** SALVAGUARDA DE TIEMPO: presupuesto alcanzado, se omite el resto de PASADA 2 "
                      f"(quedaron {combo_str} y siguientes con {n_ya}/10) ***")
                cortado_en_paso2 = True
                break
            seed_base = abs(hash((eje_A, eje_B, eje_C, "paso2"))) % 100000
            admitidas, descartadas = correr_n_reglas_extra(
                eje_A, eje_B, eje_C, n_ya, N_REGLAS_OBJETIVO, seed_base)
            total_admitidas += len(admitidas); total_descartadas += len(descartadas)
            if not admitidas:
                continue
            print(f"\n--- {combo_str} (paso2, +{len(admitidas)} nuevas, descartadas={len(descartadas)}) ---")
            fallos_seguidos = 0
            for p in admitidas:
                try:
                    r, filas_out, dt = correr_regla_y_clasificar(p, eje_A, eje_B, eje_C)
                except Exception as e:
                    print(f"  {p['rule_id']}: *** FALLO DEL MOTOR (no se toca cs090_fase5_motor.py) *** "
                          f"{type(e).__name__}: {e}")
                    combos_rotos.setdefault(combo_str, f"{type(e).__name__}: {e}")
                    fallos_seguidos += 1
                    if fallos_seguidos >= 2:
                        break
                    continue
                fallos_seguidos = 0
                filas_csv.extend(filas_out)
                resultados_por_combo[(eje_A, eje_B, eje_C)].append(
                    dict(rule_id=p["rule_id"], clase=r["clase"], pendiente=r["pendiente_real"],
                         z_agg=r["z_agg"], holon_ratio=r["holon_ratio"]))
                print(f"  {p['rule_id']}: clase={r['clase']:<24} pendiente={r['pendiente_real']:.3f} "
                      f"z_agg={r['z_agg']:.2f} holon_ratio={r['holon_ratio']:.2f} ({dt:.1f}s)"
                      f"  [t_total={time.time()-t_inicio:.0f}s]")
                if time.time() - t_inicio > PRESUPUESTO_SEG:
                    print("  *** SALVAGUARDA DE TIEMPO: presupuesto alcanzado a mitad de PASADA 2 ***")
                    cortado_en_paso2 = True
                    break
            if cortado_en_paso2:
                break

    # ------------------------------- CSV crudo (una fila por N/escala) -------------------------------
    if filas_csv:
        campos = list(filas_csv[0].keys())
        with open(OUT_CSV, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=campos)
            wr.writeheader()
            for f in filas_csv:
                wr.writerow(f)
    print(f"\nCSV crudo escrito: {OUT_CSV}  ({len(filas_csv)} filas)")

    # ------------------------------- CSV resumen (una fila por regla) -------------------------------
    campos_resumen = ["eje_A", "eje_B", "eje_C", "combo", "rule_id", "clase", "pendiente", "z_agg", "holon_ratio"]
    with open(OUT_CSV_RESUMEN, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=campos_resumen)
        wr.writeheader()
        for (eje_A, eje_B, eje_C), regs in resultados_por_combo.items():
            for reg in regs:
                wr.writerow(dict(eje_A=eje_A, eje_B=eje_B, eje_C=eje_C,
                                  combo=f"{eje_A}-{eje_B}-{eje_C}", **reg))
    print(f"CSV resumen (una fila por regla) escrito: {OUT_CSV_RESUMEN}")

    # ------------------------------- RESUMEN EN PANTALLA -------------------------------
    print("\n" + "=" * 100)
    print(f"RESUMEN -- filtro P1-P5: TOTAL admitidas={total_admitidas}  descartadas={total_descartadas}")
    if combos_rotos:
        print(f"\nCOMBOS QUE EL MOTOR CONGELADO NO PUDO EJECUTAR (no se tocó cs090_fase5_motor.py, "
              f"se documenta como hallazgo): {len(combos_rotos)}")
        for combo_str, err in combos_rotos.items():
            print(f"  {combo_str}: {err}")
    print("\nMAPA 18x4 -- cuántas reglas de cada combo cayeron en cada clase")
    print(f"  {'combo':<12} {'n_reglas':<10} {'I':<4} {'II':<4} {'III':<4} {'IV':<4} {'intermedio':<10}")
    for (eje_A, eje_B, eje_C) in TODOS_COMBOS:
        regs = resultados_por_combo[(eje_A, eje_B, eje_C)]
        cnt = Counter(r["clase"] for r in regs)
        interm = cnt.get("intermedio (sin clase clara)", 0)
        print(f"  {eje_A}-{eje_B}-{eje_C:<8} {len(regs):<10} {cnt.get('I',0):<4} {cnt.get('II',0):<4} "
              f"{cnt.get('III',0):<4} {cnt.get('IV',0):<4} {interm:<10}")

    todas_clases = [r["clase"] for regs in resultados_por_combo.values() for r in regs]
    conteo_global = Counter(todas_clases)
    print(f"\nDistribución global de clases entre las {len(todas_clases)} reglas corridas: {dict(conteo_global)}")

    n_combos_completos = sum(1 for regs in resultados_por_combo.values() if len(regs) >= N_REGLAS_OBJETIVO)
    n_combos_min5 = sum(1 for regs in resultados_por_combo.values() if len(regs) >= N_REGLAS_PASO1)
    print(f"\nCombos con 10/10 reglas: {n_combos_completos}/18   Combos con >=5 reglas: {n_combos_min5}/18")
    print(f"¿Se cortó por presupuesto de tiempo? paso1={cortado_en_paso1}  paso2={cortado_en_paso2}")

    print(f"\nTiempo total: {(time.time()-t_inicio)/60:.1f} min (presupuesto: {PRESUPUESTO_SEG/60:.0f} min)")
    print("Fin del barrido completo. No se declara cierre ni veredicto -- números arriba, lectura final de Alexis.")


if __name__ == "__main__":
    main()

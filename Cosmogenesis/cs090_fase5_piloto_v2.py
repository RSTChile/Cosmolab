"""
CS090 — FASE V-A: DRIVER DEL PILOTO v2 (MISMAS 5 clases x 3 reglas, MÉTODO DE PENDIENTE CORREGIDO)
============================================================================================
QUIÉN SOY: re-corre EXACTAMENTE el mismo piloto que cs090_fase5_piloto.py (mismas 5 combinaciones de
ejes, mismas 3 reglas por clase, mismo filtro P1-P5, mismo clasificador con los umbrales pre-
registrados) pero usando `cs090_fase5_motor.correr_regla_coarse()` en vez de `correr_regla()`.

POR QUÉ EXISTE ESTE ARCHIVO SEPARADO (en vez de editar cs090_fase5_piloto.py in-place): el hallazgo
metodológico documentado en FASE5A_piloto_resultado_CS.md §4 encontró que el piloto v1 medía la
pendiente log(diám)-vs-log(N) sobre grafos INDEPENDIENTES regenerados a cada N (500/1000/1500/2000),
método distinto del usado para calibrar los umbrales 0.35-0.45/0.7-0.8 en cs080_renormalizacion.py
(coarse-graining de UN sólo grafo). Se deja cs090_fase5_piloto.py y su CSV intactos como referencia
histórica de "qué medía el método viejo" y se corre este v2 aparte con el método corregido -- así
ambos resultados quedan comparables lado a lado, sin sobrescribir el diagnóstico que los distingue.

Escribe cs090_fase5_piloto_v2_resultados.csv. No corre Phantom. No declara cierre/veredicto -- reporta
números, la lectura final es de Alexis.
"""
from __future__ import annotations
import csv, time, sys
import numpy as np

sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
import cs090_fase5_generador as GEN
import cs090_fase5_motor as MOT
from cs090_fase5_clasificador import clasificar_regla

COMBOS_PILOTO = [
    ("A0", "B0", "C0", "control no-grafo"),
    ("A1", "B0", "C0", "línea base grafo fijo, mundo-pequeño esperado"),
    ("A2", "B1", "C1", "combinación más prometedora -> ¿clase III/IV?"),
    ("A1", "B1", "C0", "aísla retroalimentación sin costo"),
    ("A2", "B0", "C2", "aísla límite de escala duro sin retroalimentación"),
]
N_REGLAS_POR_CLASE = 3
N_GRANDE = 2000                      # UN grafo por regla, a este N (antes: 4 grafos independientes)
ESCALAS_B = (1, 2, 4, 8, 16)         # mismas escalas de coarse-graining que cs080_renormalizacion.py
N_SWEEPS = 14
OUT_CSV = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5_piloto_v2_resultados.csv"


def main():
    print("=" * 100)
    print("CS090 — FASE V-A: PILOTO v2 (mismas 5 clases x 3 reglas, pendiente por coarse-graining)")
    print("=" * 100)

    filas_csv = []
    resumen_clases = []
    total_admitidas, total_descartadas = 0, 0
    t_inicio = time.time()

    for eje_A, eje_B, eje_C, nota in COMBOS_PILOTO:
        print(f"\n--- clase {eje_A}-{eje_B}-{eje_C} ({nota}) ---")
        admitidas, descartadas = GEN.generar_reglas_clase(
            eje_A, eje_B, eje_C, n_reglas=N_REGLAS_POR_CLASE,
            seed_base=abs(hash((eje_A, eje_B, eje_C))) % 100000)
        total_admitidas += len(admitidas); total_descartadas += len(descartadas)
        print(f"  filtro P1-P5: admitidas={len(admitidas)}  descartadas={len(descartadas)}"
              + (f"  [motivos: {[d['motivo_descarte'] for d in descartadas]}]" if descartadas else ""))

        clases_de_esta_combo = []
        for p in admitidas:
            t0 = time.time()
            filas_regla = MOT.correr_regla_coarse(p, N=N_GRANDE, n_sweeps=N_SWEEPS,
                                                    escalas_b=ESCALAS_B, n_seeds_null_topo=3)
            r = clasificar_regla(filas_regla)
            dt = time.time() - t0
            clases_de_esta_combo.append(r["clase"])
            print(f"  {p['rule_id']}: clase={r['clase']:<24} pendiente={r['pendiente_real']:.3f} "
                  f"z_agg={r['z_agg']:.2f} holon_ratio={r['holon_ratio']:.2f} ({dt:.1f}s)")
            for fr in filas_regla:
                fr2 = dict(fr); fr2.update(eje_A=eje_A, eje_B=eje_B, eje_C=eje_C,
                                            clase_final=r["clase"], pendiente_real=r["pendiente_real"],
                                            z_agg=r["z_agg"], holon_ratio=r["holon_ratio"])
                filas_csv.append(fr2)

        resumen_clases.append(dict(combo=f"{eje_A}-{eje_B}-{eje_C}", nota=nota,
                                    clases=clases_de_esta_combo,
                                    n_admitidas=len(admitidas), n_descartadas=len(descartadas)))

    # ------------------------------- CSV -------------------------------
    if filas_csv:
        campos = list(filas_csv[0].keys())
        with open(OUT_CSV, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=campos)
            wr.writeheader()
            for f in filas_csv:
                wr.writerow(f)
    print(f"\nCSV escrito: {OUT_CSV}  ({len(filas_csv)} filas)")

    # ------------------------------- RESUMEN -------------------------------
    print("\n" + "=" * 100)
    print("RESUMEN — filtro P1-P5")
    print(f"  TOTAL admitidas={total_admitidas}  descartadas={total_descartadas}")
    print("\nRESUMEN — clasificación de las 5 clases piloteadas (método v2: coarse-graining)")
    print(f"  {'combo':<14} {'nota':<52} {'clases (3 reglas)':<22}")
    for r in resumen_clases:
        print(f"  {r['combo']:<14} {r['nota']:<52} {r['clases']}")

    todas_clases = [c for r in resumen_clases for c in r["clases"]]
    from collections import Counter
    conteo = Counter(todas_clases)
    print(f"\nDistribución global de clases entre las 5 combos piloteadas: {dict(conteo)}")
    colapso = len(conteo) <= 1
    print(f"¿el clasificador colapsa todo en una sola clase? {'SÍ -- señal de bug' if colapso else 'NO'}")

    print(f"\nTiempo total del piloto v2: {(time.time()-t_inicio)/60:.1f} min")
    print("Fin del piloto v2. No se declara cierre ni veredicto -- números arriba, lectura final de Alexis.")


if __name__ == "__main__":
    main()

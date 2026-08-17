"""
CS090 — FASE VI / O1-C (complemento): ¿ES REPRODUCIBLE LA CLASE DE UNA REGLA A0?
====================================================================================================
QUIÉN SOY
---------
Script NUEVO, complemento de `cs090_fase6_o1c_cierre_a0.py`. Hace la prueba más directa posible de la
sospecha del artefacto, y no requiere ninguna métrica nueva:

    Se congela EL CAMPO (mismo sustrato A0, misma dinámica, mismo array `S` bit a bit) y se repite
    SÓLO LA MEDICIÓN varias veces, cambiando únicamente las semillas de la medición:
      - qué `n_cand=15` candidatos al azar mira cada sitio al armar el grafo derivado
        (`_grafo_medicion_A0`),
      - qué cajas arma el BFS del coarse-graining (`cajas_bfs` de cs080),
      - qué grafos Erdős-Rényi salen como NULL_topo.
    El campo NO cambia entre réplicas. Ni un solo valor de `S`.

LÓGICA DE LA PRUEBA (por qué esto decide algo)
-----------------------------------------------
- Si "Clase II" fuera una propiedad DEL CAMPO, todas las réplicas de una misma regla deberían dar la
  misma clase: el campo es idéntico, y lo único que cambió es con qué muestra al azar se lo miró.
- Si "Clase II" fuera una propiedad DEL MÉTODO DE MEDICIÓN (el muestreo al azar de candidatos), la
  clase debería bailar entre réplicas del mismo campo.
No hay una tercera opción cómoda: es una prueba de test-retest, la misma que se le pide a cualquier
instrumento antes de creerle.

ANALOGÍA SIMPLE
----------------
Es como pesar la MISMA bolsa de harina seis veces en la misma balanza. Si la balanza dice 1 kg, 1 kg,
1 kg..., la bolsa pesa 1 kg. Si dice 1 kg, 3 kg, 1 kg, 4 kg..., no aprendimos nada sobre la bolsa:
aprendimos algo sobre la balanza.

QUÉ SE REUSA (sólo import, cero ediciones en congelados)
---------------------------------------------------------
  `cs090_fase5_motor` (construir_A0, dinamica_B0, medir, medir_null_valor, _diam, _muestrear_triangulos),
  `cs080_renormalizacion` (cajas_bfs, grafo_grueso), `cg003_diagnostico_gromov` (aleatorio),
  `cs090_fase5_clasificador` (clasificar_regla), `cs090_fase5_generador` (generar_regla).
La cadena de medición es la MISMA de `cs090_fase5_motor.correr_regla_coarse`, copiada en
`medir_una_vez()` con un solo cambio explícito y documentado: las semillas de medición salen de
`semilla_medicion` en vez de derivarse de `p["seed"]`, que es justamente la variable que se quiere
mover. La réplica no pretende reproducir bit a bit la corrida histórica (para eso está
`cs090_fase6_o1c_cierre_a0.py`); pretende ser un sorteo más, estadísticamente equivalente, del mismo
procedimiento de medición.

No declara cierre ni veredicto: reporta números.
"""
from __future__ import annotations

import csv
import sys
import time

import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cg003_diagnostico_gromov as GR
import cs080_renormalizacion as CS80
import cs090_fase5_clasificador as CLS
import cs090_fase5_generador as GEN
import cs090_fase5_motor as MOT

N_SITIOS = 2000
N_SWEEPS = 14
ESCALAS_B = (1, 2, 4, 8, 16)
N_REPLICAS = 6           # cuántas veces se vuelve a medir EL MISMO campo
N_POR_GRUPO = 15         # cuántas reglas Clase II y cuántas Clase I se re-miden


def construir_campo(p):
    """Construye y evoluciona el campo EXACTAMENTE como el pipeline (misma semilla `seed*5000+N`):
    esto es lo que queda CONGELADO entre réplicas."""
    rng = np.random.default_rng(p["seed"] * 5000 + N_SITIOS)
    sustrato = MOT.construir_A0(N_SITIOS, rng, p)
    sustrato = MOT.dinamica_B0(sustrato, p, rng, N_SWEEPS, p["eje_C"])
    return sustrato


def medir_una_vez(sustrato, p, semilla_medicion):
    """La cadena de medición completa de `correr_regla_coarse`, pero con TODAS sus semillas derivadas
    de `semilla_medicion`. Devuelve las filas con el esquema que espera el clasificador congelado."""
    rng_med = np.random.default_rng(semilla_medicion)
    m = MOT.medir(sustrato, p, rng_med)          # <- acá vive el muestreo al azar de candidatos
    adj_real = m["adj_final"]
    nv = MOT.medir_null_valor(m, p, np.random.default_rng(semilla_medicion + 11))

    meandeg_equiv = max(0.5, 2.0 * m["n_aristas"] / max(1, N_SITIOS))
    adjs_null = []
    for s in range(3):
        adj0, _ = GR.aleatorio(N_SITIOS, meandeg=meandeg_equiv, seed=int(semilla_medicion + 101 + s))
        adjs_null.append([set(a.tolist()) for a in adj0])

    filas = []
    for b in ESCALAS_B:
        if b == 1:
            adj_g, n_cajas = adj_real, N_SITIOS
        else:
            asign, n_cajas = CS80.cajas_bfs(adj_real, N_SITIOS, b,
                                            np.random.default_rng(semilla_medicion + 1000 + b))
            adj_g = CS80.grafo_grueso(adj_real, N_SITIOS, asign, n_cajas)
        diam_g = float(MOT._diam(adj_g, n_cajas)) if n_cajas > 1 else float("nan")
        d_nulls = []
        for k, adj_n in enumerate(adjs_null):
            if b == 1:
                adj_ng, n_cajas_n = adj_n, N_SITIOS
            else:
                asign_n, n_cajas_n = CS80.cajas_bfs(adj_n, N_SITIOS, b,
                                                    np.random.default_rng(semilla_medicion + 2000 + b * 7 + k))
                adj_ng = CS80.grafo_grueso(adj_n, N_SITIOS, asign_n, n_cajas_n)
            d_nulls.append(float(MOT._diam(adj_ng, n_cajas_n)) if n_cajas_n > 1 else float("nan"))
        filas.append(dict(
            rule_id=p["rule_id"], N=n_cajas, escala_b=b, diam_real=diam_g,
            giant_real=float(MOT._giant(adj_g, n_cajas)) if n_cajas > 1 else 0.0,
            holon_real=m["holonomia"], n_aristas=sum(len(a) for a in adj_g) // 2,
            n_triangulos=m["n_triangulos"],
            diam_null_topo=float(np.nanmean(d_nulls)),
            diam_null_topo_std=float(np.nanstd(d_nulls)) + 1e-9,
            holon_null_valor=nv["holonomia"],
        ))
    return filas


def main(ruta_resumen=f"{_HERE}/cs090_fase6_o1c_a0_resumen.csv"):
    t_ini = time.time()
    with open(ruta_resumen) as fh:
        todas = list(csv.DictReader(fh))
    clase_II = [r for r in todas if r["clase_vieja"] == "II"][:N_POR_GRUPO]
    clase_I = [r for r in todas if r["clase_vieja"] == "I"][:N_POR_GRUPO]
    print("=" * 100)
    print("O1-C complemento — test-retest: MISMO campo, medición repetida con otras semillas")
    print("=" * 100)
    print(f"Se re-miden {len(clase_II)} reglas etiquetadas Clase II y {len(clase_I)} etiquetadas "
          f"Clase I, {N_REPLICAS} réplicas cada una.\n")

    filas_out = []
    for etiqueta_origen, grupo in (("II", clase_II), ("I", clase_I)):
        for r in grupo:
            p = GEN.generar_regla("A0", "B0", "C0", idx=0, seed=int(r["seed"]))
            p["rule_id"] = r["rule_id"]
            sustrato = construir_campo(p)
            # el campo queda congelado: se guarda una huella para probar que NO cambia entre réplicas
            huella = float(np.sum(sustrato["S"]))
            clases, pendientes = [], []
            for rep in range(N_REPLICAS):
                filas = medir_una_vez(sustrato, p, semilla_medicion=int(r["seed"]) * 97 + rep * 7919 + 3)
                res = CLS.clasificar_regla(filas)
                clases.append(res["clase"]); pendientes.append(res["pendiente_real"])
            n_II = sum(1 for c in clases if c == "II")
            filas_out.append(dict(
                rule_id=r["rule_id"], seed=int(r["seed"]), clase_original=etiqueta_origen,
                huella_campo=huella,
                n_replicas=N_REPLICAS, n_replicas_clase_II=n_II,
                fraccion_II=round(n_II / N_REPLICAS, 4),
                clases="|".join(clases),
                pendiente_media=round(float(np.mean(pendientes)), 4),
                pendiente_min=round(float(np.min(pendientes)), 4),
                pendiente_max=round(float(np.max(pendientes)), 4),
                pendiente_sd=round(float(np.std(pendientes)), 4),
            ))
            print(f"  [{etiqueta_origen:>2}] {r['rule_id']:<16} réplicas Clase II: {n_II}/{N_REPLICAS}"
                  f"   pendiente {np.min(pendientes):.3f}–{np.max(pendientes):.3f} "
                  f"(sd={np.std(pendientes):.3f})   clases: {'|'.join(clases)}", flush=True)

    with open(f"{_HERE}/cs090_fase6_o1c_reproducibilidad.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas_out[0].keys()))
        w.writeheader(); w.writerows(filas_out)

    orig_II = [f for f in filas_out if f["clase_original"] == "II"]
    orig_I = [f for f in filas_out if f["clase_original"] == "I"]
    print(f"\n{'='*100}\nRESUMEN\n{'='*100}")
    for etq, g in (("etiquetadas Clase II", orig_II), ("etiquetadas Clase I", orig_I)):
        fr = np.array([f["fraccion_II"] for f in g])
        estables = sum(1 for f in g if f["n_replicas_clase_II"] in (0, N_REPLICAS))
        print(f"  {etq:<22} n={len(g)}  fracción de réplicas que dan Clase II: media={fr.mean():.3f} "
              f"[{fr.min():.2f}, {fr.max():.2f}]  |  reglas 100% estables (6/6 o 0/6): "
              f"{estables}/{len(g)}")
    sd = np.array([f["pendiente_sd"] for f in filas_out])
    print(f"  variación de la pendiente entre réplicas del MISMO campo: sd media={sd.mean():.4f} "
          f"(máx={sd.max():.4f}); el ancho de la banda que define Clase II es 0.45-0.35 = 0.10")
    print(f"\n  -> cs090_fase6_o1c_reproducibilidad.csv ({len(filas_out)} filas)")
    print(f"TOTAL {(time.time()-t_ini)/60:.1f} min")


if __name__ == "__main__":
    main()

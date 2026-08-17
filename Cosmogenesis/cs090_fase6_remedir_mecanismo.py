"""
cs090_fase6_remedir_mecanismo.py — PASO 3 de la adopción del diámetro corregido: re-etiquetado de la
LÍNEA DEL MECANISMO de Fase V (F5-C2-C → C5) y de las genealogías independientes.
====================================================================================================

QUÉ TAREAS CUBRE (las que usaron "%Clase III" como endpoint principal)
-----------------------------------------------------------------------
  FASE5_presupuesto_emergente_CS.md      -> cs090_fase5_presupuesto_emergente     (4 brazos × 20 reglas)
  FASE5_presupuesto_soporte_local_CS.md  -> cs090_fase5_presupuesto_soporte       (5 brazos × 20)
  FASE5_mecanismo_aislado_CS.md          -> cs090_fase5_mecanismo_aislado         (5 brazos × 20)
  FASE5_matriz_2x2_completa_CS.md        -> cs090_fase5_presupuesto_variable      (5 brazos × 20)
  FASE5_control_azar_elastico_CS.md      -> cs090_fase5_control_azar_elastico     (5 brazos × 20)
  FASE5_genealogias_independientes_CS.md -> cs090_fase5_genealogias_independientes(2 brazos × 4 gen. × 20)

CÓMO SE RE-MIDE SIN TOCAR NINGÚN SCRIPT CONGELADO
--------------------------------------------------
Los seis scripts calculan el diámetro llamando SIEMPRE a `cs090_fase5_motor._diam(...)` — una búsqueda
de atributo de módulo que se resuelve en el momento de la llamada. Entonces alcanza con **sustituir ese
atributo en memoria** (`MOT._diam = cs090_diam_corregido.diam_gigante`) antes de invocar la función de
brazo de cada script, y restaurarlo después. Ningún archivo se modifica en disco; cada brazo corre con
su propia cadena exacta (mismos rng, mismos NULL, mismo coarse-graining), sólo que apoyando el metro en
la componente gigante. Es el equivalente en memoria de "la medición corregida reemplaza a la oficial".

El lado VIEJO no se re-corre: se reconstruye directamente desde los CSV históricos
`cs090_fase5_*_resultados.csv`, que ya guardan diam_real / diam_null_topo / diam_null_topo_std / N por
escala — exactamente los campos que consume `cs090_fase5_clasificador.clasificar_regla`. Como control
de que esa reconstrucción es fiel, se exige que la clase recalculada desde el CSV coincida con la
`clase_final` guardada en el propio CSV.

SALIDA
------
`cs090_fase6_remedicion_mecanismo.csv` — una fila por (tarea, regla, brazo) con clase vieja, clase
corregida, pendientes y el diagnóstico. El resumen impreso da el %Clase III por brazo ANTES y DESPUÉS,
que es el endpoint sobre el que se apoyaron los informes de esas 6 tareas.

No corre Phantom. No toca scripts congelados. No declara veredicto.
"""
from __future__ import annotations
import csv
import sys
import time
import collections

import numpy as np

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)

import cs090_fase5_generador as GEN
import cs090_fase5_motor as MOT
import cs090_diam_corregido as DC
from cs090_fase5_clasificador import clasificar_regla

# --------------------------------------------------------------------------------------------
# Catálogo de tareas: (etiqueta, módulo con _correr_brazo, csv_resumen, csv_resultados, cols_id)
# --------------------------------------------------------------------------------------------
import cs090_fase5_presupuesto_emergente as PE
import cs090_fase5_presupuesto_soporte as PS
import cs090_fase5_mecanismo_aislado as MA
import cs090_fase5_presupuesto_variable as PV
import cs090_fase5_control_azar_elastico as CA
import cs090_fase5_genealogias_independientes as GE

TAREAS = [
    ("presupuesto_emergente", PE, "cs090_fase5_presupuesto_emergente"),
    ("presupuesto_soporte",   PS, "cs090_fase5_presupuesto_soporte"),
    ("mecanismo_aislado",     MA, "cs090_fase5_mecanismo_aislado"),
    ("presupuesto_variable",  PV, "cs090_fase5_presupuesto_variable"),
    ("control_azar_elastico", CA, "cs090_fase5_control_azar_elastico"),
    ("genealogias",           GE, "cs090_fase5_genealogias_independientes"),
]


def filas_historicas(base):
    """Rearma, desde el CSV de resultados históricos, el `filas` por (rule_id, brazo[, genealogía])
    con el esquema que consume el clasificador. Devuelve dict clave -> (filas, clase_guardada)."""
    ruta = f"{HERE}/{base}_resultados.csv"
    with open(ruta) as f:
        rows = list(csv.DictReader(f))
    out = collections.defaultdict(list)
    clase_guardada = {}
    for r in rows:
        k = (r.get("genealogia", ""), r["rule_id"], r["brazo"])
        out[k].append(dict(
            rule_id=r["rule_id"], N=int(float(r["N"])), escala_b=int(r["escala_b"]),
            diam_real=float(r["diam_real"]), giant_real=float(r["giant_real"]),
            holon_real=float(r["holon_real"]), n_aristas=int(float(r["n_aristas"])),
            diam_null_topo=float(r["diam_null_topo"]),
            diam_null_topo_std=float(r["diam_null_topo_std"]),
            holon_null_valor=float(r["holon_null_valor"]),
        ))
        clase_guardada[k] = r["clase_final"]
    for k in out:
        out[k].sort(key=lambda d: d["escala_b"])
    return out, clase_guardada


def parametros_por_regla(base):
    """Del CSV de resumen: clave -> (seed, K, J, noise, meandeg, kcap, clase, pendiente)."""
    with open(f"{HERE}/{base}_resumen.csv") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        k = (r.get("genealogia", ""), r["rule_id"], r["brazo"])
        out[k] = r
    return out


def main(solo=None, limite=None):
    t_ini = time.time()
    salida = []
    for etiqueta, mod, base in TAREAS:
        if solo and etiqueta not in solo:
            continue
        hist, clase_csv = filas_historicas(base)
        params = parametros_por_regla(base)
        claves = sorted(set(hist) & set(params))
        if limite:
            claves = claves[:limite]
        print(f"\n### {etiqueta}: {len(claves)} (regla × brazo)", flush=True)

        n_fiel = 0
        for i, k in enumerate(claves):
            gen, rule_id, brazo = k
            r = params[k]
            seed = int(r["seed"])
            p = GEN.generar_regla("A2", "B0", "C2", idx=0, seed=seed)
            p["rule_id"] = rule_id
            # control: los parámetros regenerados desde el seed deben coincidir con el CSV
            coincide_p = all(abs(float(p[c]) - float(r[c])) < 1e-9
                             for c in ("K", "J", "noise", "meandeg", "kcap"))

            c_viejo = clasificar_regla(hist[k])
            fiel_hist = (c_viejo["clase"] == clase_csv[k])
            n_fiel += fiel_hist

            # ---- re-corrida del brazo con el diámetro CORREGIDO (parche en memoria) ----
            _orig = MOT._diam
            try:
                MOT._diam = DC.diam_gigante
                filas_corr = mod._correr_brazo(brazo, p)
            except Exception as e:
                MOT._diam = _orig
                print(f"   *** FALLO {rule_id}/{brazo}: {type(e).__name__}: {e}", flush=True)
                continue
            finally:
                MOT._diam = _orig
            c_nuevo = clasificar_regla(filas_corr)

            salida.append(dict(
                tarea=etiqueta, genealogia=gen, rule_id=rule_id, brazo=brazo, seed=seed,
                params_coinciden=coincide_p, clase_csv_reproducida=fiel_hist,
                clase_vieja=c_viejo["clase"], clase_corregida=c_nuevo["clase"],
                cambia_clase=(c_viejo["clase"] != c_nuevo["clase"]),
                pendiente_vieja=c_viejo["pendiente_real"], pendiente_corregida=c_nuevo["pendiente_real"],
                z_agg_vieja=c_viejo["z_agg"], z_agg_corregida=c_nuevo["z_agg"],
                diam_viejo=[f["diam_real"] for f in hist[k]],
                diam_corregido=[f["diam_real"] for f in filas_corr],
                n_cajas_viejo=[f["N"] for f in hist[k]],
                n_cajas_corregido=[f["N"] for f in filas_corr],
            ))
            if (i + 1) % 25 == 0:
                print(f"   ...{i+1}/{len(claves)}  ({time.time()-t_ini:.0f}s)", flush=True)
        print(f"   clase histórica reproducida desde el CSV: {n_fiel}/{len(claves)}")

    campos = list(salida[0].keys())
    with open(f"{HERE}/cs090_fase6_remedicion_mecanismo.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for d in salida:
            w.writerow(d)
    print(f"\n[csv] cs090_fase6_remedicion_mecanismo.csv ({len(salida)} filas, "
          f"{time.time()-t_ini:.0f}s)")

    # ---------------- %Clase III por brazo, antes y después ----------------
    print("\n" + "=" * 100)
    print("%CLASE III POR BRAZO — ENDPOINT PRINCIPAL DE LA LÍNEA DEL MECANISMO")
    print("=" * 100)
    for etiqueta, _, _ in TAREAS:
        sub = [d for d in salida if d["tarea"] == etiqueta]
        if not sub:
            continue
        print(f"\n--- {etiqueta} ---")
        clave_grupo = (lambda d: (d["genealogia"], d["brazo"])) if etiqueta == "genealogias" \
            else (lambda d: d["brazo"])
        grupos = collections.defaultdict(list)
        for d in sub:
            grupos[clave_grupo(d)].append(d)
        for g, v in sorted(grupos.items(), key=lambda kv: str(kv[0])):
            n = len(v)
            iii_v = sum(1 for d in v if d["clase_vieja"] == "III")
            iii_c = sum(1 for d in v if d["clase_corregida"] == "III")
            cam = sum(1 for d in v if d["cambia_clase"])
            print(f"   {str(g):<45} n={n:<4} %III viejo={100.0*iii_v/n:5.1f}  "
                  f"%III corregido={100.0*iii_c/n:5.1f}   (cambian de clase: {cam})")

    cam = [d for d in salida if d["cambia_clase"]]
    print(f"\n[TOTAL] cambian de clase: {len(cam)}/{len(salida)}")
    for (a, b), n in collections.Counter((d["clase_vieja"], d["clase_corregida"]) for d in cam).most_common():
        print(f"   {a} -> {b} : {n}")


if __name__ == "__main__":
    solo = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] != "-" else None
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(solo, lim)

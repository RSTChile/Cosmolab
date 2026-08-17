"""
cs090_fase6_o3e_correr.py — FASE VI, tarea O3-E: batería emparejada CON MEMORIA vs SIN MEMORIA
=========================================================================================================

QUÉ HACE
--------
Toma un puñado de reglas A2-B0-C2 YA generadas y clasificadas (las 430 de
`cs090_fase6_remedicion_430.csv`, elegidas de forma determinista y estratificada por pendiente
corregida, para no cherry-pickear) y, para CADA UNA, construye DOS universos que difieren en una sola
cosa: si la poda final recuerda la historia de inestabilidad de cada relación (`-mem`) o si sólo mira el
conflicto del presente (`-nomem`). Ver `cs090_fase6_o3e_memoria.py` para el detalle exacto de qué se
quitó.

Cada uno de los dos brazos pasa por EL MISMO pipeline que toda la línea Fase V-B, reusado sin tocarlo:

    grafo (cs090_fase6_o3e_memoria) → layout de resortes + masa fija + turbulencia
    (`cs090_fase5b_phantom_adaptador.generar_ic_masa_fija_desde_grafo`, N=2000, masa total 18800,
     lado de caja fijo 2000^(1/3), seed_layout=12345, Mach=3 seed=42)
    → Phantom (`cs090_fase5b_correr.correr_una`: icreate_sinks=1, rho_crit=1000, r_crit=0.6, h_acc=0.3,
       tmax=0.500, dtmax=0.001 — los MISMOS parámetros de toda la jerarquía CS073)
    → métricas (`cs090_fase5b_analizar.analizar_carpeta`: fracción de masa en sumideros, κ_V, etc.)

y además se le mide al grafo de cada brazo la PENDIENTE CORREGIDA (coarse-graining b=1,2,4,8,16 con
`cs090_diam_corregido.diam_gigante`, la medición oficial desde el 11-ago-2026), para poder separar dos
explicaciones distintas si aparece una diferencia de masa: que la memoria cambie primero la GEOMETRÍA
del grafo (y la masa la siga), o que actúe por otra vía.

VERIFICACIÓN CRUZADA (obligatoria en esta línea, por la lección del bug de colisión de nombres)
------------------------------------------------------------------------------------------------
Después de escribir cada `meta_regla.json` se relee del disco y se exige que `seed`, `K`, `kcap` y
`brazo` coincidan con lo que la tarea pidió, y que el nombre de la carpeta coincida con el `rule_id` de
adentro. Además, para cada par se exige que los dos `meta_regla.json` (mem y nomem) declaren EXACTAMENTE
los mismos parámetros de regla (seed, K, J, noise, meandeg, kcap): si el emparejamiento se rompiera por
un cruce de nombres, eso lo detecta. Cualquier discrepancia aborta con AssertionError.

Prefijo de lote NUEVO, sin colisión con ninguno previo (`r0-r19`, `r0-r39`, `batch3-*`, `batch4-*`,
`*v1fix`, `*v2fix`, `*pendNEG`): **`A2-B0-C2-o3e-s{seed}-mem` / `-nomem`**.

No modifica ningún archivo existente. No hace commits. No declara cierre ni veredicto: escribe números.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)

BASE_SALIDA = Path("/Users/alexis/phantom_cs073/bateria_fase6_o3e_memoria")
RUTA_ENTRADA_430 = f"{_HERE}/cs090_fase6_remedicion_430.csv"
RUTA_CSV_CRUDO = f"{_HERE}/cs090_fase6_o3e_memoria_crudo.csv"

N_PILOTO = 2000
N_SWEEPS = 14
SEED_LAYOUT = 12345          # MISMO seed de layout que toda la Fase V-B (no se cambia)
PREFIJO = "A2-B0-C2-o3e-s"   # prefijo de lote nuevo, sin colisión
N_PARES_OBJETIVO = 15
N_WORKERS = 6                # la máquina tiene 16 núcleos pero está compartida con otras tareas
# COSTO MEDIDO (piloto de 2 pares, 11-ago-2026, máquina cargada con otras tareas del equipo):
#   grafo+geometría ≈ 5 s · layout+IC ≈ 275 s · Phantom ≈ 30 s por corrida.
# El layout de resortes domina; por eso el paso 1 va en paralelo y el resto serial.


# =========================================================================================
# 1) SELECCIÓN DETERMINISTA Y ESTRATIFICADA DE LAS REGLAS BASE
# =========================================================================================
def elegir_seeds(n_pares=N_PARES_OBJETIVO):
    """Ordena las 430 reglas A2-B0-C2 ya remedidas por PENDIENTE CORREGIDA y toma `n_pares` posiciones
    igualmente espaciadas. No se elige por resultado en Phantom (ninguna de estas 430 se eligió por su
    masa), ni al azar sin registro: la receta es reproducible con una línea y cubre todo el rango de
    geometría, que es justamente la variable que después hay que descontar."""
    with open(RUTA_ENTRADA_430) as f:
        filas = list(csv.DictReader(f))
    filas = [r for r in filas if r["pendiente_corregida"] not in ("", "nan")]
    filas.sort(key=lambda r: (float(r["pendiente_corregida"]), int(r["seed"])))
    idxs = np.linspace(0, len(filas) - 1, n_pares).round().astype(int)
    elegidas = []
    vistos = set()
    for i in idxs:
        r = filas[int(i)]
        if int(r["seed"]) in vistos:
            continue
        vistos.add(int(r["seed"]))
        elegidas.append(dict(seed=int(r["seed"]), K=int(r["K"]), kcap=int(r["kcap"]),
                             clase_corregida=r["clase_corregida"],
                             pendiente_corregida_historica=float(r["pendiente_corregida"]),
                             rule_id_original=r["rule_id"]))
    return elegidas


# =========================================================================================
# 2) UNA CORRIDA (un brazo de un par): grafo → pendiente corregida → IC de Phantom
# =========================================================================================
def preparar_brazo(tarea):
    """Ejecutable en un proceso worker. `tarea` = dict(seed, K, kcap, brazo). brazo ∈ {"mem","nomem"}."""
    import cs090_fase6_o3e_memoria as O3E
    from cs090_fase5b_phantom_adaptador import generar_ic_masa_fija_desde_grafo

    seed, brazo = tarea["seed"], tarea["brazo"]
    ventana = None if brazo == "mem" else 1
    rule_id = f"{PREFIJO}{seed}-{brazo}"
    carpeta = BASE_SALIDA / rule_id
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_ic = carpeta / "cosmogenesis_ic.txt"

    if (carpeta / "meta_regla.json").exists() and ruta_ic.exists():
        return json.loads((carpeta / "meta_regla.json").read_text())

    t0 = time.time()
    p, m, diag = O3E.reconstruir(seed, ventana_memoria=ventana, N=N_PILOTO, n_sweeps=N_SWEEPS)
    t_grafo = time.time() - t0

    t1 = time.time()
    geo = O3E.pendiente_corregida(m["adj_final"], N_PILOTO, seed, m["n_aristas"])
    t_geo = time.time() - t1

    t2 = time.time()
    info_ic = generar_ic_masa_fija_desde_grafo(m["adj_final"], N=N_PILOTO, seed_layout=SEED_LAYOUT,
                                               ruta_salida=str(ruta_ic))
    t_ic = time.time() - t2

    meta = dict(
        rule_id=rule_id, brazo=brazo, ventana_memoria=ventana, clase=brazo,   # `clase` lo lee analizar_carpeta
        seed=seed, N=N_PILOTO, n_sweeps=N_SWEEPS, seed_layout=SEED_LAYOUT,
        K=p["K"], J=p["J"], noise=p["noise"], meandeg=p["meandeg"], kcap=p["kcap"],
        sim_thr_frac=p["sim_thr_frac"],
        n_aristas_grafo_final=m["n_aristas"], grado_medio_grafo_final=2.0 * m["n_aristas"] / N_PILOTO,
        diam_grafo_final=m["diam"], giant_grafo_final=m["giant"], holon_grafo_final=m["holonomia"],
        pendiente_corregida=geo["pendiente_corregida"], pendiente_null=geo["pendiente_null"],
        z_agg_corregido=geo["z_agg"], z_sostenido=bool(geo["z_sostenido"]),
        diam_corr_por_escala=geo["diam_por_escala"], n_cajas_por_escala=geo["n_cajas_por_escala"],
        # diagnóstico del contraste de memoria (ver docstring de cs090_fase6_o3e_memoria)
        aristas_antes_de_podar=diag["n_edges"], frac_conservada=diag["frac_conservada"],
        hist_usado_suma=diag["hist_suma"], hist_usado_no_cero=diag["hist_no_cero"],
        hist_acumulado_suma=diag["hist_acumulado_suma"], n_costos_distintos=diag["n_costos_distintos"],
        masa_total_ic=info_ic["masa_total"], carpeta=str(carpeta), ruta_ic=str(ruta_ic),
        t_grafo_s=round(t_grafo, 2), t_geometria_s=round(t_geo, 2), t_ic_s=round(t_ic, 2),
    )
    with open(carpeta / "meta_regla.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def verificar_meta(meta, tarea):
    """Relee del disco y compara contra lo pedido — detección de colisión de nombres."""
    carpeta = Path(meta["carpeta"])
    en_disco = json.loads((carpeta / "meta_regla.json").read_text())
    assert carpeta.name == en_disco["rule_id"], f"{carpeta}: nombre de carpeta != rule_id del meta"
    assert en_disco["seed"] == tarea["seed"], f"{carpeta}: seed {en_disco['seed']} != {tarea['seed']}"
    assert en_disco["K"] == tarea["K"] and en_disco["kcap"] == tarea["kcap"], (
        f"{carpeta}: K/kcap {en_disco['K']}/{en_disco['kcap']} != {tarea['K']}/{tarea['kcap']}")
    assert en_disco["brazo"] == tarea["brazo"], f"{carpeta}: brazo {en_disco['brazo']} != {tarea['brazo']}"
    esperado_ventana = None if tarea["brazo"] == "mem" else 1
    assert en_disco["ventana_memoria"] == esperado_ventana, f"{carpeta}: ventana_memoria mal"
    return en_disco


# =========================================================================================
# 3) PIPELINE COMPLETO
# =========================================================================================
def main(n_pares=N_PARES_OBJETIVO, solo_preparar=False):
    t_inicio = time.time()
    BASE_SALIDA.mkdir(parents=True, exist_ok=True)
    elegidas = elegir_seeds(n_pares)
    print(f"[O3-E] {len(elegidas)} reglas base elegidas (estratificadas por pendiente corregida) — "
          f"seeds={[e['seed'] for e in elegidas]}", flush=True)

    tareas = []
    for e in elegidas:
        for brazo in ("mem", "nomem"):
            tareas.append(dict(seed=e["seed"], K=e["K"], kcap=e["kcap"], brazo=brazo))

    # ---- Paso 1: grafos + geometría + IC (en paralelo, cada tarea es independiente) ----
    t0 = time.time()
    metas = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for tarea, meta in zip(tareas, ex.map(preparar_brazo, tareas)):
            meta_disco = verificar_meta(meta, tarea)
            metas[(tarea["seed"], tarea["brazo"])] = meta_disco
            print(f"  IC {meta_disco['rule_id']}: aristas={meta_disco['n_aristas_grafo_final']} "
                  f"pend_corr={meta_disco['pendiente_corregida']:.3f} "
                  f"(t_grafo={meta_disco['t_grafo_s']}s t_ic={meta_disco['t_ic_s']}s)", flush=True)
    print(f"[TOTAL IC] {len(tareas)} condiciones iniciales en {time.time()-t0:.0f}s", flush=True)

    # ---- Verificación cruzada de emparejamiento: los dos brazos de un par son LA MISMA regla ----
    for e in elegidas:
        a, b = metas[(e["seed"], "mem")], metas[(e["seed"], "nomem")]
        for campo in ("seed", "K", "J", "noise", "meandeg", "kcap", "sim_thr_frac", "seed_layout", "N"):
            assert a[campo] == b[campo], (
                f"par seed={e['seed']}: {campo} difiere entre brazos ({a[campo]} vs {b[campo]}) — "
                f"el par NO está emparejado, abortando")
    print(f"[verificación cruzada] {len(elegidas)} pares: los dos brazos declaran la MISMA regla "
          f"(seed/K/J/noise/meandeg/kcap/seed_layout/N idénticos) — OK", flush=True)

    if solo_preparar:
        return elegidas, metas, None

    # ---- Paso 2: Phantom (serial, reusando correr_una sin tocarlo) ----
    from cs090_fase5b_correr import correr_una
    t0 = time.time()
    for k, tarea in enumerate(tareas):
        carpeta = BASE_SALIDA / f"{PREFIJO}{tarea['seed']}-{tarea['brazo']}"
        info = correr_una(carpeta)
        if info.get("ya_corrida"):
            print(f"  [{k+1}/{len(tareas)}] {carpeta.name}: ya corrida, se salta", flush=True)
        else:
            print(f"  [{k+1}/{len(tareas)}] {carpeta.name}: exit_run={info['exit_run']} "
                  f"t_run={info['t_run']}s", flush=True)
            if info["exit_run"] != 0:
                print(f"    AVISO: exit_run != 0", flush=True)
    print(f"[TOTAL PHANTOM] {time.time()-t0:.0f}s", flush=True)

    # ---- Paso 3: métricas (reusando analizar_carpeta sin tocarlo) ----
    from cs090_fase5b_analizar import analizar_carpeta
    filas = []
    for e in elegidas:
        for brazo in ("mem", "nomem"):
            carpeta = BASE_SALIDA / f"{PREFIJO}{e['seed']}-{brazo}"
            fila = analizar_carpeta(carpeta)
            meta = metas[(e["seed"], brazo)]
            fila["brazo"] = brazo
            fila["par_seed"] = e["seed"]
            fila["clase_corregida_historica"] = e["clase_corregida"]
            fila["rule_id_original"] = e["rule_id_original"]
            for campo in ("pendiente_corregida", "pendiente_null", "z_agg_corregido", "z_sostenido",
                          "grado_medio_grafo_final", "giant_grafo_final", "holon_grafo_final",
                          "aristas_antes_de_podar", "frac_conservada", "hist_usado_suma",
                          "hist_usado_no_cero", "hist_acumulado_suma", "n_costos_distintos"):
                fila[campo] = meta[campo]
            filas.append(fila)

    campos = list(filas[0].keys())
    for f in filas:
        for c in f:
            if c not in campos:
                campos.append(c)
    with open(RUTA_CSV_CRUDO, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    print(f"\n[TOTAL] {len(elegidas)} pares ({len(filas)} corridas) -> {RUTA_CSV_CRUDO} "
          f"— tiempo total {time.time()-t_inicio:.0f}s", flush=True)
    return elegidas, metas, filas


# =========================================================================================
# 4) CONSOLIDAR DESDE DISCO — arma el CSV crudo con los pares que YA terminaron
# =========================================================================================
def consolidar_desde_disco():
    """Recorre las carpetas de la batería y arma el CSV crudo SÓLO con los pares en que los DOS brazos
    tienen dump final de Phantom. Sirve para leer resultados parciales sin esperar a que termine la
    batería entera (la máquina está compartida con el resto del equipo y el tiempo de pared no depende
    sólo de esta tarea). Aplica la MISMA verificación cruzada de emparejamiento que `main`."""
    from cs090_fase5b_analizar import analizar_carpeta
    from leer_volcado_phantom import listar_dumps

    por_seed = {}
    for carpeta in sorted(BASE_SALIDA.iterdir()):
        if not carpeta.is_dir() or not (carpeta / "meta_regla.json").exists():
            continue
        meta = json.loads((carpeta / "meta_regla.json").read_text())
        assert carpeta.name == meta["rule_id"], f"{carpeta}: carpeta != rule_id"
        if not listar_dumps(carpeta):
            continue
        por_seed.setdefault(meta["seed"], {})[meta["brazo"]] = (carpeta, meta)

    historico = {int(e["seed"]): e for e in elegir_seeds(430)}
    filas = []
    completos = 0
    for seed, d in sorted(por_seed.items()):
        if "mem" not in d or "nomem" not in d:
            continue
        (_, ma), (_, mb) = d["mem"], d["nomem"]
        for campo in ("seed", "K", "J", "noise", "meandeg", "kcap", "sim_thr_frac", "seed_layout", "N"):
            assert ma[campo] == mb[campo], f"par seed={seed}: {campo} difiere entre brazos — abortando"
        completos += 1
        for brazo in ("mem", "nomem"):
            carpeta, meta = d[brazo]
            fila = analizar_carpeta(carpeta)
            fila["brazo"] = brazo
            fila["par_seed"] = seed
            h = historico.get(seed, {})
            fila["clase_corregida_historica"] = h.get("clase_corregida")
            fila["rule_id_original"] = h.get("rule_id_original")
            for campo in ("pendiente_corregida", "pendiente_null", "z_agg_corregido", "z_sostenido",
                          "grado_medio_grafo_final", "giant_grafo_final", "holon_grafo_final",
                          "aristas_antes_de_podar", "frac_conservada", "hist_usado_suma",
                          "hist_usado_no_cero", "hist_acumulado_suma", "n_costos_distintos"):
                fila[campo] = meta.get(campo)
            filas.append(fila)

    campos = list(filas[0].keys())
    for f in filas:
        for c in f:
            if c not in campos:
                campos.append(c)
    with open(RUTA_CSV_CRUDO, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    print(f"[consolidar] {completos} pares completos ({len(filas)} corridas) -> {RUTA_CSV_CRUDO}")
    return filas


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "consolidar":
        consolidar_desde_disco()
        sys.exit(0)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_PARES_OBJETIVO
    solo = len(sys.argv) > 2 and sys.argv[2] == "solo_preparar"
    main(n, solo_preparar=solo)

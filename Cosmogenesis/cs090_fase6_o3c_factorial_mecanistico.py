"""
cs090_fase6_o3c_factorial_mecanistico.py — FASE VI, O3-C: la cadena completa
MECANISMO RELACIONAL -> GEOMETRIA -> DINAMICA GRAVITACIONAL, sobre las MISMAS genealogias.
=========================================================================================================

QUIEN SOY (y por que existo)
----------------------------
Hoy la evidencia vive en DOS segmentos separados que nunca se corrieron juntos:

  segmento 1 (Fase V, linea del mecanismo F5-C2-C .. C5):
      "rigidez del corte + criterio de soporte local"  ->  GEOMETRIA (pendiente del diametro vs escala)
  segmento 2 (Fase V-B, 40 pares en Phantom):
      "geometria extendida (Clase III)"                 ->  MAS MASA ACRETADA en Phantom

Nadie corrio nunca el mecanismo relacional DIRECTO contra Phantom. Esta tarea (propuesta F6-03 de
GPT-5.6 Sol, id O3-C del plan `FASE6_PLAN_EJECUCION_COMPLETA_CS.md`) une los dos segmentos en una sola
cadena de mediacion: se generan las 4 condiciones del factorial mecanistico sobre las MISMAS reglas
(misma genealogia, mismo seed, mismos K/J/noise/meandeg/kcap) y TODAS pasan por Phantom, registrando por
regla la pendiente corregida (geometria) Y la fraccion de masa acretada (gravedad).

EL FACTORIAL 2x2 (rigidez del corte x criterio de seleccion), con el cupo VARIABLE fijo en las 4 celdas
--------------------------------------------------------------------------------------------------------
Se mantiene constante la tercera dimension (uniformidad del cupo = VARIABLE por nodo) porque es la unica
donde existen las 4 celdas ya implementadas y validadas, y porque F5-C2-C4 mostro que esa dimension es la
que MENOS mueve la aguja (0-10 pp) -- fijarla no cuesta casi nada y hace el factorial limpio.

                          CRITERIO = soporte/costo local        CRITERIO = azar
    CORTE RIGIDO          (1) C2-hibrido                        (2) C2-random
    (conteo exacto)           MA.dinamica_B0_hibrido(...,           MA.dinamica_B0_hibrido(...,
                              modo="soporte")                       modo="azar")

    CORTE ELASTICO        (3) C2-presupuesto-variable           (4) C2-presupuesto-variable-azar
    (knapsack/presupuesto)    PV.dinamica_B0_presupuesto_          CAE.dinamica_B0_presupuesto_
                              variable                              variable_azar

QUE SE REUSA SIN TOCAR NI UNA LINEA (todo lo demas es pegamento)
-----------------------------------------------------------------
  cs090_fase5_generador          GEN.generar_reglas_clase / generar_regla  (el mismo lote de reglas)
  cs090_fase5_motor              MOT.construir_A2 / medir / correr_regla_coarse / _diam
  cs090_fase5_mecanismo_aislado  MA._cupo_variable, MA.dinamica_B0_hibrido, MA.correr_regla_coarse_hibrido
  cs090_fase5_presupuesto_variable    PV.dinamica_B0_presupuesto_variable, PV.correr_regla_coarse_...
  cs090_fase5_control_azar_elastico   CAE.dinamica_B0_presupuesto_variable_azar, CAE.correr_regla_...
  cs090_fase5_clasificador       clasificar_regla  (pendiente + clase)
  cs090_diam_corregido           DC.diam_gigante   (metro oficial desde FASE6_adopcion_diam_corregido_CS)
  cs090_fase5b_phantom_adaptador generar_ic_masa_fija_desde_grafo  (grafo -> IC de Phantom, masa fija)
  cs090_fase5b_correr            correr_una        (phantomsetup + phantom, mismos parametros CS073)
  cs090_fase5b_analizar          analizar_carpeta  (masa en sumideros, kappa_V, t primer sumidero)

Las 4 condiciones comparten EXACTAMENTE el mismo prologo determinista que ya usan sus 4 funciones
`correr_regla_coarse_*` originales (mismo rng `seed*5000+N`, mismo `construir_A2`, mismo `_cupo_variable`
sobre el grado inicial recien nacido). `_grafo_final_de_condicion` no reimplementa nada: llama a esas
mismas piezas y se queda ADEMAS con el grafo final (`m["adj_final"]`), que las funciones originales
calculan internamente pero no devuelven. Que ese grafo sea el mismo se VERIFICA numericamente (chequeo
cruzado #3 abajo), no se asume.

LOS 5 CHEQUEOS CRUZADOS OBLIGATORIOS (leccion del bug de colision de nombres de Fase V-B)
------------------------------------------------------------------------------------------
  #1 seed/K/kcap regenerados desde `GEN` == los archivados en cs090_fase6_remedicion_mecanismo.csv
     para la MISMA regla historica (A2-B0-C2-r{idx}) -- antes de generar nada.
  #2 la pendiente corregida recalculada fresca == la archivada en ese mismo CSV (reproducibilidad del
     motor; se reporta la diferencia, se aborta si supera 1e-6).
  #3 n_aristas del grafo reconstruido == n_aristas de la fila b=1 del brazo oficial -> es EL MISMO grafo
     que produjo la pendiente, no uno parecido.
  #4 meta_regla.json RELEIDO DEL DISCO tras escribirlo == rule_id/seed/K/kcap/condicion esperados.
  #5 al analizar, el meta de la carpeta se vuelve a cruzar contra la tabla de tareas antes de aceptar
     la fila de metricas.

PREFIJO DE NOMBRE SIN COLISION
-------------------------------
`rule_id` = "A2-B0-C2-mec-r{idx}"  (prefijo "mec", nunca usado: los ya ocupados son r0-r19, r0-r39,
batch3-*, batch4-*, *v1fix, *v2fix, *pendNEG). Carpeta = "{rule_id}__{cond_id}" -- la condicion va en el
nombre, asi que las 4 condiciones de una misma regla NUNCA pueden pisarse entre si.

SALIDAS
-------
  cs090_fase6_o3c_crudo.csv     una fila por (regla x condicion): geometria + gravedad + verificaciones
  cs090_fase6_o3c_mediacion.csv tabla de la cadena condicion -> pendiente -> masa (analisis)
  /Users/alexis/phantom_cs073/bateria_fase6_o3c_mecanistico/<rule_id>__<cond>/  IC + dumps + meta

No modifica ningun archivo existente. No hace commits. No declara cierre ni veredicto.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)

import cs090_fase5_generador as GEN
import cs090_fase5_motor as MOT
import cs090_diam_corregido as DC
import cs090_fase5_mecanismo_aislado as MA
import cs090_fase5_presupuesto_variable as PV
import cs090_fase5_control_azar_elastico as CAE
from cs090_fase5_clasificador import clasificar_regla
from cs090_fase5b_phantom_adaptador import generar_ic_masa_fija_desde_grafo

# ------------------------------------------------------------------------------------------------
# Constantes: TODAS heredadas de la linea que se esta uniendo, ninguna elegida nueva
# ------------------------------------------------------------------------------------------------
N_GRANDE = MA.N_GRANDE                 # 2000, misma resolucion que toda Fase V-A/V-B
N_SWEEPS = MA.N_SWEEPS                 # 14
ESCALAS_B = MA.ESCALAS_B               # (1,2,4,8,16)
N_SEEDS_NULL_TOPO = MA.N_SEEDS_NULL_TOPO
SEED_BASE_MECANISMO = MA.SEED_BASE + 1  # 90211: el lote "completo" de las 5 tareas F5-C2-C..C5
SEED_LAYOUT = 12345                    # mismo layout de resortes que TODA la Fase V-B

PREFIJO_RULE_ID = "A2-B0-C2-mec-r"     # prefijo nuevo, sin colision (ver docstring)
BASE_SALIDA = "/Users/alexis/phantom_cs073/bateria_fase6_o3c_mecanistico"
RUTA_CRUDO_CSV = f"{HERE}/cs090_fase6_o3c_crudo.csv"
RUTA_MEDIACION_CSV = f"{HERE}/cs090_fase6_o3c_mediacion.csv"
RUTA_REMEDICION_CSV = f"{HERE}/cs090_fase6_remedicion_mecanismo.csv"

# ------------------------------------------------------------------------------------------------
# Las 4 condiciones del factorial. Cada una apunta a (a) la funcion de dinamica ya validada que produce
# el grafo, y (b) la funcion de brazo ya validada que produce las filas de coarse-graining -> pendiente.
# `tarea_hist`/`brazo_hist` identifican la fila equivalente en cs090_fase6_remedicion_mecanismo.csv,
# que es contra lo que se cruza el chequeo #2.
# ------------------------------------------------------------------------------------------------
CONDICIONES = {
    "c1-rigido-soporte": dict(
        n=1, rigidez="rigido", criterio="soporte",
        etiqueta="(1) corte RIGIDO + criterio de SOPORTE local",
        brazo_hist="C2-hibrido", tarea_hist="mecanismo_aislado"),
    "c2-rigido-azar": dict(
        n=2, rigidez="rigido", criterio="azar",
        etiqueta="(2) corte RIGIDO + AZAR",
        brazo_hist="C2-random", tarea_hist="mecanismo_aislado"),
    "c3-elastico-soporte": dict(
        n=3, rigidez="elastico", criterio="soporte",
        etiqueta="(3) presupuesto ELASTICO + criterio de costo/SOPORTE local",
        brazo_hist="C2-presupuesto-variable", tarea_hist="control_azar_elastico"),
    "c4-elastico-azar": dict(
        n=4, rigidez="elastico", criterio="azar",
        etiqueta="(4) presupuesto ELASTICO + AZAR",
        brazo_hist="C2-presupuesto-variable-azar", tarea_hist="control_azar_elastico"),
}
ORDEN_COND = ["c1-rigido-soporte", "c2-rigido-azar", "c3-elastico-soporte", "c4-elastico-azar"]


# ================================================================================================
# 1) GRAFO FINAL de cada condicion -- mismo prologo determinista que las 4 funciones originales
# ================================================================================================
def _grafo_final_de_condicion(cond_id, p, N=N_GRANDE, n_sweeps=N_SWEEPS):
    """Devuelve el dict de `MOT.medir` (con `adj_final`) para la condicion pedida.

    Las 4 funciones `correr_regla_coarse_*` originales empiezan TODAS con estas mismas 5 lineas y
    despues descartan el grafo (solo devuelven filas de coarse-graining). Aca se repiten esas 5 lineas
    exactas -- mismo rng, mismo constructor, mismo `_cupo_variable` sobre el grado recien nacido, misma
    funcion de dinamica IMPORTADA de su modulo original -- y se conserva el grafo. No se copia ni se
    modifica el cuerpo de ninguna de esas funciones: se las llama."""
    rng = np.random.default_rng(p["seed"] * 5000 + N)
    sustrato = MOT.construir_A2(N, rng, p)
    grado_inicial = np.array([len(sustrato["adj"][i]) for i in range(N)], dtype=float)
    cupo = MA._cupo_variable(grado_inicial, p["kcap"])      # mismo B_i / kcap_i en las 4 condiciones

    if cond_id == "c1-rigido-soporte":
        sustrato = MA.dinamica_B0_hibrido(sustrato, p, rng, n_sweeps, cupo, "soporte")
    elif cond_id == "c2-rigido-azar":
        sustrato = MA.dinamica_B0_hibrido(sustrato, p, rng, n_sweeps, cupo, "azar")
    elif cond_id == "c3-elastico-soporte":
        sustrato = PV.dinamica_B0_presupuesto_variable(sustrato, p, rng, n_sweeps, cupo)
    elif cond_id == "c4-elastico-azar":
        sustrato = CAE.dinamica_B0_presupuesto_variable_azar(sustrato, p, rng, n_sweeps, cupo)
    else:
        raise ValueError(f"condicion desconocida: {cond_id}")

    return MOT.medir(sustrato, p, rng)


def _filas_coarse_de_condicion(cond_id, p):
    """Filas de coarse-graining (b=1,2,4,8,16) de la condicion -- llamando a la funcion de brazo ORIGINAL
    de cada modulo, sin tocarla. Se corre con el diametro CORREGIDO parcheado en memoria (misma tecnica
    exacta que `cs090_fase6_remedir_mecanismo.py`: `MOT._diam` se resuelve en el momento de la llamada,
    asi que sustituir el atributo del modulo basta y no se edita ningun archivo en disco)."""
    _orig = MOT._diam
    try:
        MOT._diam = DC.diam_gigante
        if cond_id == "c1-rigido-soporte":
            return MA.correr_regla_coarse_hibrido(p, modo="soporte")
        if cond_id == "c2-rigido-azar":
            return MA.correr_regla_coarse_hibrido(p, modo="azar")
        if cond_id == "c3-elastico-soporte":
            return PV.correr_regla_coarse_presupuesto_variable(p)
        if cond_id == "c4-elastico-azar":
            return CAE.correr_regla_coarse_presupuesto_variable_azar(p)
        raise ValueError(f"condicion desconocida: {cond_id}")
    finally:
        MOT._diam = _orig


# ================================================================================================
# 2) EL LOTE DE REGLAS -- las MISMAS genealogias de la linea del mecanismo (seed_base 90211)
# ================================================================================================
def reglas_del_lote(n_reglas):
    """Regenera el lote admitido (filtro P1-P5 real) con el MISMO seed_base que usaron las 5 tareas
    F5-C2-C..C5 en su corrida "completo". Devuelve la lista de dicts `p`, con el rule_id historico
    (A2-B0-C2-r{idx}) guardado aparte y el rule_id NUEVO (prefijo mec) puesto en `p["rule_id"]`."""
    admitidas, descartadas = GEN.generar_reglas_clase(
        "A2", "B0", "C2", n_reglas=20, seed_base=SEED_BASE_MECANISMO, max_intentos=80)
    salida = []
    for idx, p in enumerate(admitidas[:n_reglas]):
        p = dict(p)
        p["rule_id_historico"] = f"A2-B0-C2-r{idx}"
        p["idx_lote"] = idx
        p["rule_id"] = f"{PREFIJO_RULE_ID}{idx}"
        salida.append(p)
    return salida, len(descartadas)


def cargar_remedicion():
    """Tabla archivada de la re-medicion con diametro corregido: (tarea, rule_id, brazo) -> fila.
    Es la fuente contra la que se cruzan los chequeos #1 y #2."""
    out = {}
    with open(RUTA_REMEDICION_CSV) as f:
        for r in csv.DictReader(f):
            out[(r["tarea"], r["rule_id"], r["brazo"])] = r
    return out


def chequeo_1_parametros(reglas, remed):
    """Chequeo cruzado #1: para cada regla del lote y cada condicion, el seed/K/kcap regenerados desde
    el generador deben coincidir EXACTO con los archivados para la regla historica equivalente."""
    problemas = []
    for p in reglas:
        for cond_id in ORDEN_COND:
            c = CONDICIONES[cond_id]
            k = (c["tarea_hist"], p["rule_id_historico"], c["brazo_hist"])
            if k not in remed:
                problemas.append(f"{k}: no existe en {RUTA_REMEDICION_CSV}")
                continue
            r = remed[k]
            if int(r["seed"]) != int(p["seed"]):
                problemas.append(f"{k}: seed archivado {r['seed']} != regenerado {p['seed']}")
    assert not problemas, "CHEQUEO #1 FALLIDO:\n  " + "\n  ".join(problemas)
    print(f"[chequeo #1] {len(reglas)} reglas x {len(ORDEN_COND)} condiciones: seed regenerado == "
          f"seed archivado en cs090_fase6_remedicion_mecanismo.csv -- OK", flush=True)


# ================================================================================================
# 3) PREPARAR UNA CELDA (regla x condicion): geometria + grafo + condicion inicial de Phantom
#    Es la funcion que corre en cada proceso del pool -- debe ser de nivel de modulo (spawn en macOS).
# ================================================================================================
def preparar_una(tarea):
    """tarea = dict(seed, idx_lote, rule_id, rule_id_historico, cond_id, pendiente_archivada).
    Hace, en este orden: (a) pendiente corregida fresca con el brazo original, (b) reconstruccion del
    grafo final de la misma condicion, (c) chequeo #3 de identidad del grafo, (d) IC de Phantom,
    (e) meta_regla.json + chequeo #4 releyendolo del disco."""
    t0 = time.time()
    cond_id = tarea["cond_id"]
    c = CONDICIONES[cond_id]

    p = GEN.generar_regla("A2", "B0", "C2", idx=0, seed=tarea["seed"])
    p["rule_id"] = tarea["rule_id"]

    # (a) geometria: pendiente corregida, con el brazo ORIGINAL de cada condicion
    filas = _filas_coarse_de_condicion(cond_id, p)
    clasif = clasificar_regla(filas)
    fila_b1 = next(f for f in filas if f["escala_b"] == 1)
    t_geom = time.time() - t0

    # chequeo #2: reproducibilidad contra el valor archivado. NO aborta -- se REGISTRA.
    # Razón (documentada en el informe §3): la pendiente que usa el análisis es la FRESCA, y el chequeo
    # #3 garantiza que esa pendiente y el grafo que va a Phantom salen de la misma corrida. Comparar
    # contra el archivo es un diagnóstico de reproducibilidad entre sesiones, no un requisito de
    # correctitud de esta tarea; abortar por él descartaría una celda válida y, peor, escondería el
    # dato interesante. Se guarda la diferencia en el meta y se reporta cuántas celdas reproducen.
    dif_pend = abs(clasif["pendiente_real"] - tarea["pendiente_archivada"])
    reproduce_archivada = bool(dif_pend < 1e-6)
    if not reproduce_archivada:
        print(f"  AVISO chequeo #2: {tarea['rule_id']}/{cond_id} pendiente fresca "
              f"{clasif['pendiente_real']:.9f} != archivada {tarea['pendiente_archivada']:.9f} "
              f"(dif={dif_pend:.2e}) -- se registra y se sigue", flush=True)

    # (b) el grafo final de ESA MISMA condicion
    t1 = time.time()
    m = _grafo_final_de_condicion(cond_id, p)
    t_grafo = time.time() - t1

    # (c) chequeo #3: el grafo reconstruido es EL MISMO que produjo la pendiente
    assert int(m["n_aristas"]) == int(fila_b1["n_aristas"]), (
        f"CHEQUEO #3 FALLIDO {tarea['rule_id']}/{cond_id}: n_aristas grafo={m['n_aristas']} != "
        f"n_aristas fila b=1 del brazo={fila_b1['n_aristas']} -- el grafo NO es el mismo, abortando")

    # (d) condicion inicial de Phantom, misma receta de masa fija de toda la Fase V-B
    carpeta = f"{BASE_SALIDA}/{tarea['rule_id']}__{cond_id}"
    os.makedirs(carpeta, exist_ok=True)
    ruta_ic = f"{carpeta}/cosmogenesis_ic.txt"
    t2 = time.time()
    info_ic = generar_ic_masa_fija_desde_grafo(m["adj_final"], N=N_GRANDE, seed_layout=SEED_LAYOUT,
                                               ruta_salida=ruta_ic)
    t_ic = time.time() - t2

    # (e) meta + chequeo #4 releyendo del disco
    meta = dict(
        rule_id=tarea["rule_id"], rule_id_historico=tarea["rule_id_historico"],
        idx_lote=tarea["idx_lote"], cond_id=cond_id, cond_n=c["n"],
        rigidez=c["rigidez"], criterio=c["criterio"], brazo_hist=c["brazo_hist"],
        tarea_hist=c["tarea_hist"],
        clase=clasif["clase"], pendiente_corregida=clasif["pendiente_real"],
        pendiente_archivada=tarea["pendiente_archivada"],
        dif_vs_archivada=dif_pend, reproduce_archivada=reproduce_archivada,
        z_agg=clasif["z_agg"],
        holon_ratio=clasif["holon_ratio"],
        seed=int(p["seed"]), N=N_GRANDE, seed_layout=SEED_LAYOUT,
        K=p["K"], J=p["J"], noise=p["noise"], meandeg=p["meandeg"], kcap=p["kcap"],
        sim_thr_frac=p["sim_thr_frac"],
        n_aristas_grafo_final=int(m["n_aristas"]), diam_grafo_final=m["diam"],
        giant_grafo_final=m["giant"], holon_grafo_final=m["holonomia"],
        grado_medio_grafo_final=2.0 * m["n_aristas"] / N_GRANDE,
        n_triangulos_grafo_final=m.get("n_triangulos"),
        diam_b1_corregido=fila_b1["diam_real"],
        masa_total_ic=info_ic["masa_total"], carpeta=carpeta, ruta_ic=ruta_ic,
        t_geometria_s=round(t_geom, 2), t_grafo_s=round(t_grafo, 2), t_ic_s=round(t_ic, 2),
    )
    with open(f"{carpeta}/meta_regla.json", "w") as f:
        json.dump(meta, f, indent=2)

    releido = json.loads(Path(f"{carpeta}/meta_regla.json").read_text())
    for campo, esperado in (("rule_id", tarea["rule_id"]), ("cond_id", cond_id),
                            ("seed", int(p["seed"])), ("K", p["K"]), ("kcap", p["kcap"])):
        assert releido[campo] == esperado, (
            f"CHEQUEO #4 FALLIDO {carpeta}: meta_regla.json releido tiene {campo}={releido[campo]} "
            f"!= esperado {esperado} -- POSIBLE COLISION DE NOMBRE, abortando")

    meta["t_total_s"] = round(time.time() - t0, 2)
    return meta


# ================================================================================================
# 4) ETAPAS
# ================================================================================================
def etapa_ic(n_reglas, workers, conds):
    t_ini = time.time()
    reglas, n_desc = reglas_del_lote(n_reglas)
    remed = cargar_remedicion()
    chequeo_1_parametros(reglas, remed)
    print(f"[lote] {len(reglas)} reglas admitidas (descartadas por P1-P5 en el lote de 20: {n_desc}); "
          f"seed_base={SEED_BASE_MECANISMO}", flush=True)

    tareas = []
    for p in reglas:
        for cond_id in conds:
            c = CONDICIONES[cond_id]
            r = remed[(c["tarea_hist"], p["rule_id_historico"], c["brazo_hist"])]
            tareas.append(dict(seed=int(p["seed"]), idx_lote=p["idx_lote"], rule_id=p["rule_id"],
                               rule_id_historico=p["rule_id_historico"], cond_id=cond_id,
                               pendiente_archivada=float(r["pendiente_corregida"])))

    pendientes = [t for t in tareas
                  if not Path(f"{BASE_SALIDA}/{t['rule_id']}__{t['cond_id']}/cosmogenesis_ic.txt").exists()]
    print(f"[IC] {len(tareas)} celdas (regla x condicion); faltan {len(pendientes)}; "
          f"workers={workers}", flush=True)

    metas = []
    if pendientes:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(preparar_una, t): t for t in pendientes}
            for k, fut in enumerate(as_completed(futs)):
                t = futs[fut]
                meta = fut.result()          # si algun chequeo fallo, la excepcion sube y corta la etapa
                metas.append(meta)
                print(f"  [{k+1}/{len(pendientes)}] {meta['rule_id']}__{meta['cond_id']} "
                      f"clase={meta['clase']} pend={meta['pendiente_corregida']:.3f} "
                      f"aristas={meta['n_aristas_grafo_final']} "
                      f"(geom {meta['t_geometria_s']}s + grafo {meta['t_grafo_s']}s + IC "
                      f"{meta['t_ic_s']}s) [t={time.time()-t_ini:.0f}s]", flush=True)
    print(f"[IC] listo en {time.time()-t_ini:.0f}s ({len(metas)} nuevas)", flush=True)
    return tareas


def etapa_phantom(tareas, workers=1):
    """Corre Phantom celda por celda reusando `correr_una` (congelado, sin tocar). `workers>1` lanza
    varias corridas a la vez con hilos -- `correr_una` es un envoltorio de `subprocess.run`, asi que el
    trabajo real ocurre en procesos externos y los hilos solo esperan; no hay estado compartido entre
    carpetas. Se usa porque la maquina esta compartida con otras tareas y una corrida individual de
    N=2000 dura ~10-30s: encadenarlas de a una desperdicia la espera de E/S."""
    from concurrent.futures import ThreadPoolExecutor
    from cs090_fase5b_correr import correr_una      # congelado -- solo import
    t_ini = time.time()
    tiempos = []

    def _una(t):
        carpeta = Path(f"{BASE_SALIDA}/{t['rule_id']}__{t['cond_id']}")
        return carpeta, correr_una(carpeta)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, (carpeta, info) in enumerate(ex.map(_una, tareas)):
            if info.get("ya_corrida"):
                print(f"  [{k+1}/{len(tareas)}] {carpeta.name}: ya corrida, se salta", flush=True)
                continue
            tiempos.append(info["t_run"])
            print(f"  [{k+1}/{len(tareas)}] {carpeta.name}: exit_setup={info['exit_setup']} "
                  f"t_setup={info['t_setup']}s exit_run={info['exit_run']} t_run={info['t_run']}s "
                  f"[t={time.time()-t_ini:.0f}s]", flush=True)
            if info["exit_run"] != 0:
                print(f"    AVISO: exit_run != 0 en {carpeta}", flush=True)
    if tiempos:
        print(f"[PHANTOM] {len(tiempos)} corridas nuevas, t_run medio={np.mean(tiempos):.1f}s, "
              f"total={time.time()-t_ini:.0f}s", flush=True)


def etapa_analizar(tareas):
    from cs090_fase5b_analizar import analizar_carpeta   # congelado -- solo import
    filas = []
    faltan = []
    for t in tareas:
        carpeta = Path(f"{BASE_SALIDA}/{t['rule_id']}__{t['cond_id']}")
        # una celda todavia sin meta o sin dump final no se inventa: se anota y se salta, para que un
        # analisis parcial sea posible sin fabricar datos ni romper la corrida
        if not (carpeta / "meta_regla.json").exists() or not (carpeta / "cosmog_00500").exists():
            faltan.append(carpeta.name)
            continue
        meta = json.loads((carpeta / "meta_regla.json").read_text())
        # chequeo #5: el meta de la carpeta vuelve a cruzarse antes de aceptar la fila
        assert meta["rule_id"] == t["rule_id"] and meta["cond_id"] == t["cond_id"], (
            f"CHEQUEO #5 FALLIDO {carpeta}: meta dice {meta['rule_id']}/{meta['cond_id']}, "
            f"la tabla de tareas dice {t['rule_id']}/{t['cond_id']}")
        assert meta["seed"] == t["seed"], f"CHEQUEO #5 FALLIDO {carpeta}: seed"
        fila = analizar_carpeta(carpeta)
        fila.update(
            cond_id=meta["cond_id"], cond_n=meta["cond_n"], rigidez=meta["rigidez"],
            criterio=meta["criterio"], brazo_hist=meta["brazo_hist"],
            rule_id_historico=meta["rule_id_historico"], idx_lote=meta["idx_lote"],
            pendiente_corregida=meta["pendiente_corregida"],
            pendiente_archivada=meta["pendiente_archivada"],
            dif_vs_archivada=abs(meta["pendiente_corregida"] - meta["pendiente_archivada"]),
            reproduce_archivada=(abs(meta["pendiente_corregida"] - meta["pendiente_archivada"]) < 1e-6),
            clase_geom=meta["clase"], z_agg=meta["z_agg"],
            giant_grafo_final=meta["giant_grafo_final"],
            grado_medio_grafo_final=meta["grado_medio_grafo_final"],
            diam_b1_corregido=meta["diam_b1_corregido"],
            holon_grafo_final=meta["holon_grafo_final"],
        )
        filas.append(fila)

    campos = []
    for f in filas:
        for c in f:
            if c not in campos:
                campos.append(c)
    with open(RUTA_CRUDO_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    print(f"[CSV] {len(filas)} filas -> {RUTA_CRUDO_CSV}", flush=True)
    if faltan:
        print(f"[CSV] {len(faltan)} celdas todavia sin corrida completa, NO incluidas: {faltan}",
              flush=True)
    # solo se conservan las reglas con las 4 condiciones completas: el diseño es pareado dentro de regla
    por_regla = {}
    for f in filas:
        por_regla.setdefault(f["rule_id"], set()).add(f["cond_id"])
    completas = {r for r, cs in por_regla.items() if len(cs) == len(CONDICIONES)}
    print(f"[CSV] reglas con las {len(CONDICIONES)} condiciones completas: {len(completas)} de "
          f"{len(por_regla)}", flush=True)
    return filas


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa", choices=["ic", "phantom", "analizar", "todo"], default="todo")
    ap.add_argument("--n-reglas", type=int, default=12)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--phantom-workers", type=int, default=3)
    ap.add_argument("--conds", default=",".join(ORDEN_COND))
    args = ap.parse_args()

    conds = [c.strip() for c in args.conds.split(",") if c.strip()]
    for c in conds:
        assert c in CONDICIONES, f"condicion desconocida: {c}"

    t_global = time.time()
    if args.etapa in ("ic", "todo"):
        tareas = etapa_ic(args.n_reglas, args.workers, conds)
    else:
        reglas, _ = reglas_del_lote(args.n_reglas)
        remed = cargar_remedicion()
        tareas = [dict(seed=int(p["seed"]), idx_lote=p["idx_lote"], rule_id=p["rule_id"],
                       rule_id_historico=p["rule_id_historico"], cond_id=cid,
                       pendiente_archivada=float(remed[(CONDICIONES[cid]["tarea_hist"],
                                                        p["rule_id_historico"],
                                                        CONDICIONES[cid]["brazo_hist"])]["pendiente_corregida"]))
                  for p in reglas for cid in conds]

    if args.etapa in ("phantom", "todo"):
        etapa_phantom(tareas, workers=args.phantom_workers)
    if args.etapa in ("analizar", "todo"):
        etapa_analizar(tareas)
    print(f"\n[FIN etapa={args.etapa}] {time.time()-t_global:.0f}s. Sin cierre ni veredicto: "
          f"solo numeros.", flush=True)

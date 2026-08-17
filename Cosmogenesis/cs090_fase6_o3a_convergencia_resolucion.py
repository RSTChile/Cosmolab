"""
cs090_fase6_o3a_convergencia_resolucion.py — FASE VI, tarea O3-A (11-ago-2026)
==============================================================================================

QUÉ PREGUNTA CONTESTA ESTE ARCHIVO
----------------------------------
En Fase V-B se midió, sobre 40 pares de reglas A2-B0-C2 (una Clase III contra una Clase I con MISMO K y
MISMO kcap), que la Clase III termina con más masa en sumideros que su pareja Clase I en 31 de 40 pares
(p=0.00068, `FASE5B_escala_40pares_CS.md`). TODAS esas corridas fueron a N=2000 partículas.

Después, en `FASE5B_investigacion_8sumideros_y_escala_CS.md`, se descubrió que el NÚMERO de sumideros no
lo manda el grafo sino la RESOLUCIÓN: a masa total fija (18800), N=2000→4000→8000 dio 8→29→122 sumideros.
Eso deja abierta la pregunta que el equipo marcó como obligatoria:

    ¿la ventaja de la Clase III en FRACCIÓN DE MASA (la métrica principal) también depende de N?
    ¿crece, se mantiene, o se va a cero al subir la resolución?

Si se va a cero o se invierte, el resultado de Fase V-B sería un artefacto numérico de baja resolución.
Si se mantiene o crece, es un efecto que escala. Este archivo GENERA LOS NÚMEROS para decidir eso; NO
declara cierre ni veredicto (regla permanente del proyecto: la lectura final es de Alexis).

EN SIMPLE, CON ANALOGÍA
-----------------------
Es como fotografiar el mismo paisaje con una cámara de 2 megapíxeles, después con una de 4 y después con
una de 8, siempre con la misma luz, el mismo encuadre y el mismo trípode. La foto de 8 MP no va a tener
los mismos píxeles que la de 2 MP — eso es obvio y no se le exige. Lo que se le exige es que si en la
foto chica el árbol de la izquierda se veía MÁS ALTO que el de la derecha, en la foto grande siga siendo
el más alto. Si al subir megapíxeles los árboles se emparejan o se dan vuelta, entonces "el árbol más
alto" era un efecto de la mala resolución, no del paisaje.

QUÉ CAMBIA Y QUÉ NO ENTRE RESOLUCIONES (leer con atención — hay una sutileza honesta)
--------------------------------------------------------------------------------------
Se mantienen EXACTAMENTE iguales: la regla (mismo `seed`, mismos K/J/noise/meandeg/kcap), la masa física
total (18800, `MASA_TOTAL_OBJETIVO`), el lado de caja (fijo, no escala con N), la semilla de layout
(12345), la semilla de turbulencia (42, Mach=3), los 14 sweeps del motor, los 100 iters de
`layout_resortes`, y TODO el protocolo de Phantom (icreate_sinks=1, rho_crit_cgs=1000, r_crit=0.6,
h_acc=0.3, tmax=0.500, dtmax=0.001).

Lo que NO puede mantenerse literalmente idéntico, y hay que decirlo: en este pipeline **los nodos del
grafo SON las partículas SPH**. `reconstruir_regla_a2b0c2(seed, N)` construye el grafo con N nodos y
`rng = default_rng(seed*5000+N)`. Entonces subir N no reetiqueta un grafo fijo: reconstruye la MISMA
REGLA sobre un grafo MÁS GRANDE (otra realización de la misma ley generativa). Es exactamente el mismo
sentido de "subir resolución" que usó `ON77_sistemaA_cierre.py` para el barrido 2000/4000/8000 que
produjo el 8→29→122. La invariante entre resoluciones es la REGLA, no la lista de aristas.

Consecuencia práctica que hay que vigilar (y que este script mide y reporta, no asume): la etiqueta de
clase (I / III) se la pone `cs090_fase5_clasificador.clasificar_regla` a partir de métricas del grafo
medidas a una N dada. Si al subir N una regla "Clase III" dejara de clasificar como III, el par perdería
sentido. Por eso cada corrida guarda las métricas del grafo a esa N (n_aristas, grado medio, diámetro de
la componente gigante, holonomía) para poder auditar si el par sigue siendo el par.

MEDICIÓN DE DIÁMETRO: se usa `cs090_diam_corregido.diam_gigante` (medición oficial desde el 11-ago-2026),
NO el `_diam` de `cs055_proceso_acoplado` — regla vigente del proyecto. El `diam` que devuelve el motor
se guarda igual, en una columna aparte, sólo como testigo del valor histórico.

CÓMO SE USA (dos modos)
-----------------------
    # modo worker: hace UNA corrida completa (grafo -> IC -> Phantom -> métricas) y escribe su JSON
    ./venv/bin/python cs090_fase6_o3a_convergencia_resolucion.py worker <rule_id> <seed> <clase> <N>

    # modo tabla: junta todos los JSON escritos por los workers + el CSV de N=2000 y arma el CSV final
    ./venv/bin/python cs090_fase6_o3a_convergencia_resolucion.py tabla

El modo worker existe para poder lanzar varias corridas EN PARALELO desde el shell (el paso caro,
`layout_resortes`, es O(N²) y de un solo hilo; la máquina tiene 16 núcleos). Cada worker escribe en su
propia carpeta bajo `BASE_SALIDA`, así que no hay colisión de nombres — la lección del bug de la tarea
anterior (`FASE5B_investigacion_8sumideros_y_escala_CS.md`): el nombre de carpeta lleva rule_id + clase +
N, y además cada worker VERIFICA su `meta_regla.json` contra el CSV de origen antes de dar el dato por
bueno (`verificar_meta_contra_csv`).

Ningún script congelado se modifica — `cs090_fase5b_phantom_adaptador.py`, `cs090_fase5b_correr.py`,
`cs090_fase5b_analizar.py`, `cs090_fase5_motor.py`, `cs090_fase5_generador.py`, `cs090_diam_corregido.py`
sólo se IMPORTAN. No se hacen commits de git. No se declara cierre.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")

# --- todo lo de abajo es import puro de scripts congelados: nada se edita ---
from cs090_fase5b_phantom_adaptador import (reconstruir_regla_a2b0c2,
                                            generar_ic_masa_fija_desde_grafo,
                                            MASA_TOTAL_OBJETIVO, LADO_FIJO, TURB_SEED)
from cs090_fase5b_correr import correr_una           # setup + edicion de cosmog.in + run de Phantom
from cs090_fase5b_analizar import analizar_carpeta   # lee dump final + .sink -> fraccion de masa, kappa_V
import cs090_diam_corregido as DIAMC                 # medicion OFICIAL de diametro (regla 11-ago-2026)

BASE_SALIDA = Path("/Users/alexis/phantom_cs073/bateria_fase6_o3a_resolucion")
CSV_ORIGEN_N2000 = Path("/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5b_TOTAL_40pares.csv")
CSV_SALIDA = Path("/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase6_o3a_resolucion_crudo.csv")

SEED_LAYOUT = 12345      # misma semilla de layout que TODA la Fase V-B
N_SWEEPS = 14            # mismos 14 sweeps del motor que en Fase V-A/V-B
ITERS_LAYOUT = 100       # mismos 100 iters que en Fase V-B (NO se baja a 25 como hizo ON77: bajarlo
                         # cambiaria el protocolo respecto del punto N=2000 con el que se compara)


def carpeta_de(rule_id: str, clase: str, N: int) -> Path:
    """Nombre de carpeta con rule_id + clase + N: los tres, para que sea imposible que una corrida de
    N=4000 pise la de N=2000 de la misma regla (lección del bug de colisión de nombres)."""
    return BASE_SALIDA / f"N{N}" / f"{rule_id}_{clase}"


def verificar_meta_contra_csv(meta: dict, rule_id: str, seed: int, clase: str) -> dict:
    """VERIFICACIÓN CRUZADA OBLIGATORIA (lección del bug de colisión de nombres de la tarea anterior):
    antes de confiar en una corrida, se relee el CSV de origen (`cs090_fase5b_TOTAL_40pares.csv`) y se
    comprueba que el `seed`, la `clase`, el `K` y el `kcap` que quedaron escritos en `meta_regla.json`
    son los MISMOS que los de la fila de esa regla en el CSV. Si no coinciden, se levanta AssertionError
    y esa corrida no entra en el análisis."""
    import csv as _csv
    with open(CSV_ORIGEN_N2000) as fh:
        filas = [r for r in _csv.DictReader(fh) if r["rule_id"] == rule_id]
    assert filas, f"{rule_id}: no aparece en {CSV_ORIGEN_N2000.name}"
    ref = filas[0]
    assert int(ref["seed"]) == int(seed) == int(meta["seed"]), \
        f"{rule_id}: seed CSV={ref['seed']} pedido={seed} meta={meta['seed']} -- NO coinciden"
    assert ref["rol"] == clase == meta["clase"], \
        f"{rule_id}: rol CSV={ref['rol']} pedido={clase} meta={meta['clase']} -- NO coinciden"
    assert int(ref["K"]) == int(meta["K"]), f"{rule_id}: K CSV={ref['K']} meta={meta['K']}"
    assert int(ref["kcap"]) == int(meta["kcap"]), f"{rule_id}: kcap CSV={ref['kcap']} meta={meta['kcap']}"
    return dict(K_csv=int(ref["K"]), kcap_csv=int(ref["kcap"]),
                fraccion_masa_n2000=float(ref["fraccion_masa_en_sumideros"]),
                n_sumideros_n2000=int(ref["n_sumideros"]),
                kappa_v_n2000=float(ref["kappa_v_agregado"]),
                par_n2000=ref["par"])


def worker(rule_id: str, seed: int, clase: str, N: int, solo_ic: bool = False) -> dict:
    """UNA corrida completa a resolución N: reconstruye el grafo de la regla, escribe la condición
    inicial de masa fija, corre Phantom y extrae las métricas. Cronometra cada etapa por separado
    (`t_grafo_s`, `t_ic_s`, `t_phantom_s`) porque medir el costo real ANTES de comprometer la batería es
    parte explícita de esta tarea. `solo_ic=True` corta después de escribir la IC — se usa para el
    benchmark de costo de `layout_resortes` sin gastar Phantom."""
    carpeta = carpeta_de(rule_id, clase, N)
    carpeta.mkdir(parents=True, exist_ok=True)
    t_ini = time.time()

    ruta_ic = carpeta / "cosmogenesis_ic.txt"
    meta_path = carpeta / "meta_regla.json"

    if ruta_ic.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        t0 = time.time()
        p, m = reconstruir_regla_a2b0c2(seed=seed, N=N, n_sweeps=N_SWEEPS)
        t_grafo = time.time() - t0

        diam_ofi = DIAMC.diam_gigante(m["adj_final"], N)   # medicion OFICIAL (cs090_diam_corregido)

        t1 = time.time()
        info_ic = generar_ic_masa_fija_desde_grafo(m["adj_final"], N=N, seed_layout=SEED_LAYOUT,
                                                   ruta_salida=str(ruta_ic),
                                                   iters_layout=ITERS_LAYOUT)
        t_ic = time.time() - t1

        meta = dict(rule_id=rule_id, clase=clase, seed=seed, N=N, seed_layout=SEED_LAYOUT,
                    iters_layout=ITERS_LAYOUT, n_sweeps=N_SWEEPS, turb_seed=TURB_SEED,
                    masa_total_objetivo=MASA_TOTAL_OBJETIVO, lado_fijo=LADO_FIJO,
                    K=p["K"], J=p["J"], noise=p["noise"], meandeg=p["meandeg"], kcap=p["kcap"],
                    sim_thr_frac=p["sim_thr_frac"],
                    n_aristas_grafo_final=m["n_aristas"],
                    grado_medio_grafo_final=2.0 * m["n_aristas"] / N,
                    diam_grafo_final_OFICIAL=diam_ofi,
                    diam_grafo_final_motor_cs055=m["diam"],
                    giant_grafo_final=m["giant"], holon_grafo_final=m["holonomia"],
                    masa_total_ic=info_ic["masa_total"], masa_particula=info_ic["masa_particula"],
                    t_grafo_s=round(t_grafo, 2), t_ic_s=round(t_ic, 2))
        meta_path.write_text(json.dumps(meta, indent=2))

    ref = verificar_meta_contra_csv(meta, rule_id, seed, clase)
    meta.update(ref)

    if solo_ic:
        meta["t_total_s"] = round(time.time() - t_ini, 2)
        return meta

    # ---- Phantom (se REUSA correr_una tal cual; su timeout interno es 20 min por corrida) ----
    t2 = time.time()
    try:
        info_run = correr_una(carpeta)
        meta["exit_run"] = info_run.get("exit_run")
        meta["ya_corrida"] = bool(info_run.get("ya_corrida", False))
        meta["abortado_por_timeout"] = False
    except Exception as e:                      # incluye subprocess.TimeoutExpired del limite de 20 min
        meta["exit_run"] = None
        meta["abortado_por_timeout"] = True
        meta["error_run"] = f"{type(e).__name__}: {e}"
    meta["t_phantom_s"] = round(time.time() - t2, 2)

    # ---- métricas (se REUSA analizar_carpeta tal cual; lee el ÚLTIMO dump disponible) ----
    try:
        fila = analizar_carpeta(carpeta)
        meta.update({f"metrica_{k}": v for k, v in fila.items()})
    except Exception as e:
        meta["error_analisis"] = f"{type(e).__name__}: {e}"

    meta["t_total_s"] = round(time.time() - t_ini, 2)
    (carpeta / "resultado_o3a.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta


def tabla() -> None:
    """Junta todos los `resultado_o3a.json` de `BASE_SALIDA` en un CSV plano, una fila por
    (rule_id, N). El punto N=2000 no se recomputa: sale del CSV congelado de Fase V-B."""
    import csv as _csv
    filas = []

    # --- punto N=2000: se TOMA del CSV de Fase V-B, no se vuelve a correr ---
    with open(CSV_ORIGEN_N2000) as fh:
        for r in _csv.DictReader(fh):
            filas.append(dict(rule_id=r["rule_id"], clase=r["rol"], seed=int(r["seed"]), N=2000,
                              par_n2000=r["par"], K=int(r["K"]), kcap=int(r["kcap"]),
                              fraccion_masa_en_sumideros=float(r["fraccion_masa_en_sumideros"]),
                              n_sumideros=int(r["n_sumideros"]),
                              kappa_v_agregado=float(r["kappa_v_agregado"]),
                              t_primer_sumidero=r["t_primer_sumidero"],
                              masa_acretada_total=float(r["masa_acretada_total"]),
                              n_aristas_grafo_final="", grado_medio_grafo_final="",
                              diam_grafo_final_OFICIAL="", holon_grafo_final="",
                              n_dump_final="", abortado_por_timeout=False,
                              t_grafo_s="", t_ic_s="", t_phantom_s="", fuente="FaseVB_TOTAL_40pares"))

    # --- puntos N>2000: de los JSON escritos por los workers ---
    for j in sorted(BASE_SALIDA.glob("N*/*/resultado_o3a.json")):
        m = json.loads(j.read_text())
        filas.append(dict(rule_id=m["rule_id"], clase=m["clase"], seed=m["seed"], N=m["N"],
                          par_n2000=m.get("par_n2000", ""), K=m["K"], kcap=m["kcap"],
                          fraccion_masa_en_sumideros=m.get("metrica_fraccion_masa_en_sumideros"),
                          n_sumideros=m.get("metrica_n_sumideros"),
                          kappa_v_agregado=m.get("metrica_kappa_v_agregado"),
                          t_primer_sumidero=m.get("metrica_t_primer_sumidero"),
                          masa_acretada_total=m.get("metrica_masa_acretada_total"),
                          n_aristas_grafo_final=m.get("n_aristas_grafo_final"),
                          grado_medio_grafo_final=m.get("grado_medio_grafo_final"),
                          diam_grafo_final_OFICIAL=m.get("diam_grafo_final_OFICIAL"),
                          holon_grafo_final=m.get("holon_grafo_final"),
                          n_dump_final=m.get("metrica_n_dump_final"),
                          abortado_por_timeout=m.get("abortado_por_timeout"),
                          t_grafo_s=m.get("t_grafo_s"), t_ic_s=m.get("t_ic_s"),
                          t_phantom_s=m.get("t_phantom_s"), fuente="O3A_worker"))

    campos = list(filas[0].keys())
    with open(CSV_SALIDA, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    print(f"[tabla] {len(filas)} filas -> {CSV_SALIDA}")


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "tabla"
    if modo == "worker":
        rule_id, seed, clase, N = sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
        solo_ic = len(sys.argv) > 6 and sys.argv[6] == "solo_ic"
        try:
            r = worker(rule_id, seed, clase, N, solo_ic=solo_ic)
            print(f"[OK] {rule_id} clase={clase} N={N} t_grafo={r.get('t_grafo_s')}s "
                  f"t_ic={r.get('t_ic_s')}s t_phantom={r.get('t_phantom_s')}s "
                  f"frac={r.get('metrica_fraccion_masa_en_sumideros')} "
                  f"nsink={r.get('metrica_n_sumideros')} kV={r.get('metrica_kappa_v_agregado')} "
                  f"dump={r.get('metrica_n_dump_final')} timeout={r.get('abortado_por_timeout')}",
                  flush=True)
        except Exception:
            print(f"[FALLA] {rule_id} clase={clase} N={N}\n{traceback.format_exc()}", flush=True)
            sys.exit(1)
    elif modo == "tabla":
        tabla()
    else:
        print(__doc__)

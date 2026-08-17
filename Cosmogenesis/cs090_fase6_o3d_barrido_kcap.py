"""
cs090_fase6_o3d_barrido_kcap.py — FASE VI, tarea O3-D: BARRIDO DE `kcap` DIRECTAMENTE EN PHANTOM
=================================================================================================

QUIÉN SOY / QUÉ HAGO
--------------------
`FASE6_O2B_genealogias_escaladas_CS.md` (800 corridas de motor) encontró que el tope duro de vecinos
por nodo, `kcap`, es la palanca más fuerte hallada en todo el proyecto SOBRE LA GEOMETRÍA DEL GRAFO:
kcap=4 -> 98.4 % Clase III, 5 -> 71.7 %, 6 -> 4.9 %, 7 -> 0 % (η² kcap = 0.619 vs η² genealogía = 0.078).
Pero eso se midió SÓLO en el grafo — nunca con masa y gravedad de verdad.

Este script lleva ese barrido a Phantom: para kcap ∈ {4,5,6,7} toma 8 reglas A2-B0-C2 cada uno
(generador congelado + filtro P1-P5 real), mide su PENDIENTE CORREGIDA (variable continua, con
`cs090_diam_corregido.diam_gigante`, según `FASE6_adopcion_diam_corregido_CS.md`), les construye la
condición inicial de masa fija (18800, N=2000, misma receta de toda la jerarquía CS073) y las suelta en
Phantom con los mismos parámetros de siempre. Después mide fracción de masa acretada y κ_V.

Preguntas que la batería tiene que poder responder con números (no se contestan acá, se miden):
  1. ¿La fracción de masa acretada varía monótonamente con kcap, igual que la geometría?
  2. Usando la PENDIENTE CONTINUA (no las clases): ¿la relación kcap -> masa pasa A TRAVÉS de la
     geometría, o hay un efecto directo de kcap sobre la masa?
  3. Con kcap=7 dando 0 % de geometría extendida, ¿la masa acretada cae al nivel del control sin
     estructura (grafo Erdős-Rényi con la misma receta de masa fija)?

DISEÑO — POR QUÉ ESTÁ HECHO ASÍ
-------------------------------
* **La selección de reglas NO mira el resultado.** Se generan candidatas, se agrupan por su `kcap`
  (que es un parámetro del sorteo, conocido ANTES de correr nada) y se eligen las primeras 8 de cada
  grupo, balanceando K (mitad K bajo, mitad K alto) para poder cruzar el eje secundario que pide la
  tarea. En ningún momento se elige por pendiente, por clase ni por masa: eso sería sesgo de selección.
* **`seed_base` nuevo, prefijo nuevo.** `SEED_BASE_O3D = 9314159`, prefijo de rule_id
  `A2-B0-C2-kbar-r{idx}`. `_verificar_separacion()` comprueba ANTES de gastar cómputo que la cadena de
  semillas individuales de esta tarea (`seed = seed_base + intento*97 + 1`) no toca ninguna de las
  cadenas ya usadas en el proyecto — la lección del bug de colisión de nombres de Fase V-B.
* **Verificación cruzada obligatoria** (misma disciplina que `cs090_fase5b_correr_v3.py`): el
  `meta_regla.json` escrito en disco tras generar la IC se compara contra lo que el CSV de candidatas
  decía ANTES; si K/kcap/seed no coinciden, se aborta con AssertionError.
* **Diámetro corregido**: se sustituye el atributo en memoria (`MOT._diam = DC.diam_gigante`), el mismo
  mecanismo ya usado y verificado por `cs090_fase6_remedir_mecanismo.py`. Ningún archivo cambia en disco.
* **Phantom serial**: el binario está compilado con OpenMP y usa los 16 hilos de la máquina; correr
  varias instancias a la vez cambiaría el orden de las reducciones respecto de la línea base (misma
  razón que documenta `cs090_fase6_o3a_correr_phantom.sh`). La generación de IC en Python, en cambio,
  SÍ va en paralelo (es monohilo y es el 85 % del costo).
* **Poda de dumps**: el disco de esta Mac tiene ~16 GiB libres y cada corrida escribe ~27 MB en 500
  dumps de los que el análisis sólo lee el primero, el último y el `.sink`. Se podan DESPUÉS de
  analizar y guardar el resultado, nunca antes.

NADA EXISTENTE SE MODIFICA. Se importan (nunca se editan): `cs090_fase5_generador`,
`cs090_fase5_motor`, `cs090_fase5_clasificador`, `cs090_diam_corregido`,
`cs090_fase5b_phantom_adaptador`, `cs090_fase5b_correr`, `cs090_fase5b_analizar`,
`grafo_random_layout_generar_ic_masa_fija`, `campo_velocidad_turbulento`. No se declara cierre ni
veredicto: sólo se producen números. No se hacen commits.

MODOS (línea de comando)
------------------------
  candidatas [n]        genera n reglas candidatas (filtro P1-P5 real) y escribe el CSV de candidatas
  seleccionar           elige 8 por kcap (balanceando K) y escribe el archivo de trabajos
  worker <rid> <seed>   una regla: motor (pendiente corregida) + reconstrucción de grafo + IC de Phantom
  control <seed> <nar>  una condición inicial de CONTROL sin estructura (Erdős-Rényi, mismas aristas)
  phantom               corre Phantom serial sobre todas las carpetas que tengan IC y no estén corridas
  analizar              consolida el CSV crudo final
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)

import cs090_fase5_generador as GEN
import cs090_fase5_motor as MOT
import cs090_diam_corregido as DC
from cs090_fase5_clasificador import clasificar_regla
from cs090_fase5b_phantom_adaptador import reconstruir_regla_a2b0c2, generar_ic_masa_fija_desde_grafo

# --------------------------------------------------------------------------------------------------
# CONSTANTES DE LA TAREA
# --------------------------------------------------------------------------------------------------
EJE_A, EJE_B, EJE_C = "A2", "B0", "C2"
N_PILOTO = 2000              # piso de resolución SPH válido (MISTERIO_N500_vs_N2000_CS.md)
N_SWEEPS = 14
SEED_LAYOUT = 12345          # misma realización espacial que toda la línea Fase V-B
ESCALAS_B = (1, 2, 4, 8, 16)
N_SEEDS_NULL_TOPO = 3

SEED_BASE_O3D = 9314159
PREFIJO_RULE_ID = "A2-B0-C2-kbar-r"
KCAPS_OBJETIVO = (4, 5, 6, 7)     # los cuatro del rango calibrado RANGO_KCAP=(4,7) del generador
N_REGLAS_POR_KCAP = 8
N_CANDIDATAS_DEFECTO = 140        # kcap sale de round(uniform(4,7)): 4 y 7 pesan 1/6 cada uno, así que
                                  # para 8 reglas del kcap más raro hacen falta ~48 candidatas; 140 da
                                  # holgura de sobra y permite balancear K dentro de cada grupo.

BASE_SALIDA = Path("/Users/alexis/phantom_cs073/bateria_fase6_o3d_kcap")
RUTA_CANDIDATAS_CSV = Path(f"{_HERE}/cs090_fase6_o3d_candidatas.csv")
RUTA_TRABAJOS = Path(f"{_HERE}/cs090_fase6_o3d_trabajos.txt")
RUTA_SELECCION_JSON = Path(f"{_HERE}/cs090_fase6_o3d_seleccion.json")
RUTA_CRUDO_CSV = Path(f"{_HERE}/cs090_fase6_o3d_crudo.csv")

# Semillas base YA usadas en el proyecto (misma lista que verificó O2-B, más las 20 genealogías de O2-B
# y las dos de A0 nativo). Ninguna cadena de esta tarea puede tocar ninguna de ellas.
SEEDS_YA_USADAS = [
    90210, 156644, 271828, 371828, 471828, 471829, 571828, 823001,
    113477, 218903, 344251, 662819, 741037, 905683, 1128409, 1357061, 1604923, 1889347,
    2043761, 2296589, 2571043, 2814697, 3102859, 3389417, 3670213, 3948071, 4213589, 4507921,
    20260810, 20260811,
]


def _ancho_cadena(n_candidatas):
    """seed = seed_base + intento*97 + 1, con max_intentos = n_candidatas*4 -> ancho máximo de la
    cadena de semillas individuales que esta tarea puede llegar a consumir."""
    return n_candidatas * 4 * 97 + 1


def _verificar_separacion(n_candidatas):
    """Falla ANTES de gastar cómputo si la cadena de semillas de esta tarea puede solaparse con la de
    cualquier tarea previa. Es mucho más barato abortar acá que descubrir después que dos 'reglas
    distintas' eran la misma."""
    ancho = _ancho_cadena(n_candidatas)
    d_min = min(abs(SEED_BASE_O3D - s) for s in SEEDS_YA_USADAS)
    print(f"[verificación] seed_base={SEED_BASE_O3D}; ancho de cadena de esta tarea = {ancho}; "
          f"separación mínima contra las {len(SEEDS_YA_USADAS)} semillas base ya usadas = {d_min}")
    assert d_min > ancho, (
        f"la cadena [{SEED_BASE_O3D+1}, {SEED_BASE_O3D+ancho}] puede tocar una semilla ya usada "
        f"(separación {d_min} <= ancho {ancho}) — abortando antes de generar nada")
    print(f"[verificación] OK: ninguna regla de O3-D puede compartir semilla con tareas previas "
          f"(holgura {d_min/ancho:.1f}x)")


# --------------------------------------------------------------------------------------------------
# MODO 1 — candidatas: generar reglas y quedarse con sus PARÁMETROS (no corre el motor todavía)
# --------------------------------------------------------------------------------------------------
def modo_candidatas(n_candidatas=N_CANDIDATAS_DEFECTO):
    _verificar_separacion(n_candidatas)
    t0 = time.time()
    print(f"Generando {n_candidatas} reglas candidatas {EJE_A}-{EJE_B}-{EJE_C} "
          f"(filtro P1-P5 REAL del generador congelado)...", flush=True)
    admitidas, descartadas = GEN.generar_reglas_clase(
        EJE_A, EJE_B, EJE_C, n_reglas=n_candidatas, seed_base=SEED_BASE_O3D,
        max_intentos=n_candidatas * 4)
    print(f"  admitidas={len(admitidas)}  descartadas(P1-P5)={len(descartadas)}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    filas = []
    for idx, p in enumerate(admitidas):
        p["rule_id"] = f"{PREFIJO_RULE_ID}{idx}"
        filas.append(dict(rule_id=p["rule_id"], seed=p["seed"], K=p["K"], J=p["J"], noise=p["noise"],
                          meandeg=p["meandeg"], kcap=p["kcap"], sim_thr_frac=p["sim_thr_frac"]))

    with open(RUTA_CANDIDATAS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    cnt = defaultdict(int)
    for fila in filas:
        cnt[fila["kcap"]] += 1
    print(f"  reparto por kcap: {dict(sorted(cnt.items()))}")
    print(f"  -> {RUTA_CANDIDATAS_CSV}")
    return filas


# --------------------------------------------------------------------------------------------------
# MODO 2 — seleccionar: 8 reglas por kcap, balanceando K (eje secundario), SIN mirar ningún resultado
# --------------------------------------------------------------------------------------------------
def modo_seleccionar(n_por_kcap=N_REGLAS_POR_KCAP):
    filas = list(csv.DictReader(open(RUTA_CANDIDATAS_CSV)))
    for fila in filas:
        fila["kcap"] = int(fila["kcap"]); fila["K"] = int(fila["K"]); fila["seed"] = int(fila["seed"])

    por_kcap = defaultdict(list)
    for fila in filas:
        por_kcap[fila["kcap"]].append(fila)

    seleccion = []
    for kcap in KCAPS_OBJETIVO:
        grupo = por_kcap.get(kcap, [])
        assert len(grupo) >= n_por_kcap, (
            f"kcap={kcap}: sólo {len(grupo)} candidatas, hacen falta {n_por_kcap} "
            f"— generar más candidatas antes de seguir")
        # Eje secundario K: mitad del cupo con K bajo (<=5), mitad con K alto (>=7); si un lado no
        # alcanza, se completa con los que haya, EN ORDEN DE APARICIÓN (nunca por resultado).
        bajos = [f for f in grupo if f["K"] <= 5]
        altos = [f for f in grupo if f["K"] >= 7]
        medios = [f for f in grupo if f["K"] == 6]
        mitad = n_por_kcap // 2
        elegidas = bajos[:mitad] + altos[:mitad]
        resto = [f for f in (bajos[mitad:] + altos[mitad:] + medios)
                 if f["rule_id"] not in {e["rule_id"] for e in elegidas}]
        elegidas += resto[:n_por_kcap - len(elegidas)]
        elegidas = elegidas[:n_por_kcap]
        for f in elegidas:
            f["grupo_K"] = "K_bajo" if f["K"] <= 5 else ("K_alto" if f["K"] >= 7 else "K_medio")
        n_b = sum(1 for f in elegidas if f["grupo_K"] == "K_bajo")
        n_a = sum(1 for f in elegidas if f["grupo_K"] == "K_alto")
        print(f"  kcap={kcap}: {len(elegidas)} reglas de {len(grupo)} disponibles "
              f"(K_bajo={n_b} K_alto={n_a} K_medio={len(elegidas)-n_b-n_a}) "
              f"K={[f['K'] for f in elegidas]}")
        seleccion += elegidas

    RUTA_SELECCION_JSON.write_text(json.dumps(seleccion, indent=2))
    with open(RUTA_TRABAJOS, "w") as f:
        for s in seleccion:
            f.write(f"{s['rule_id']} {s['seed']} {s['kcap']}\n")
    print(f"[selección] {len(seleccion)} reglas -> {RUTA_TRABAJOS}")
    return seleccion


# --------------------------------------------------------------------------------------------------
# MODO 3 — worker: UNA regla. Motor (pendiente corregida) + grafo final + IC de Phantom.
# --------------------------------------------------------------------------------------------------
def _con_diam_corregido(fn, *a, **kw):
    """Ejecuta `fn` con `MOT._diam` sustituido en memoria por `DC.diam_gigante` (doble-BFS arrancando
    en la componente conexa MÁS GRANDE). Mecanismo idéntico al de cs090_fase6_remedir_mecanismo.py;
    ningún archivo cambia en disco y el atributo se restaura pase lo que pase."""
    _orig = MOT._diam
    try:
        MOT._diam = DC.diam_gigante
        return fn(*a, **kw)
    finally:
        MOT._diam = _orig


def modo_worker(rule_id, seed, kcap_esperado):
    seed = int(seed); kcap_esperado = int(kcap_esperado)
    carpeta = BASE_SALIDA / rule_id
    carpeta.mkdir(parents=True, exist_ok=True)
    if (carpeta / "meta_regla.json").exists() and (carpeta / "cosmogenesis_ic.txt").exists():
        print(f"[{rule_id}] ya tiene IC y meta — no se recomputa")
        return

    t0 = time.time()
    # (a) parámetros de la regla — verificación cruzada #1 contra lo que pidió el selector
    p = GEN.generar_regla(EJE_A, EJE_B, EJE_C, idx=0, seed=seed)
    assert p["kcap"] == kcap_esperado, (
        f"{rule_id}: generar_regla(seed={seed}) da kcap={p['kcap']}, se esperaba {kcap_esperado} "
        f"— POSIBLE COLISIÓN DE NOMBRE/SEMILLA, abortando")

    # (b) motor con coarse-graining -> pendiente CORREGIDA (variable continua) + clase
    filas_motor = _con_diam_corregido(MOT.correr_regla_coarse, p, N=N_PILOTO, n_sweeps=N_SWEEPS,
                                      escalas_b=ESCALAS_B, n_seeds_null_topo=N_SEEDS_NULL_TOPO)
    r = clasificar_regla(filas_motor)
    t_motor = time.time() - t0

    # (c) grafo final reconstruido bit a bit (mismo rng que usó el motor) + IC de masa fija
    t1 = time.time()
    p2, m = _con_diam_corregido(reconstruir_regla_a2b0c2, seed=seed, N=N_PILOTO, n_sweeps=N_SWEEPS)
    assert p2["seed"] == p["seed"] and p2["kcap"] == p["kcap"] and p2["K"] == p["K"], \
        f"{rule_id}: la reconstrucción no coincide con los parámetros de la regla"
    t_grafo = time.time() - t1

    t2 = time.time()
    ruta_ic = carpeta / "cosmogenesis_ic.txt"
    info_ic = generar_ic_masa_fija_desde_grafo(m["adj_final"], N=N_PILOTO, seed_layout=SEED_LAYOUT,
                                                ruta_salida=str(ruta_ic))
    t_ic = time.time() - t2

    meta = dict(rule_id=rule_id, tarea="O3-D", clase=r["clase"], seed=seed, N=N_PILOTO,
                seed_layout=SEED_LAYOUT, K=p["K"], J=p["J"], noise=p["noise"], meandeg=p["meandeg"],
                kcap=p["kcap"], sim_thr_frac=p["sim_thr_frac"],
                pendiente_corregida=r["pendiente_real"], pendiente_null=r["pendiente_null"],
                z_agg=r["z_agg"], holon_ratio=r["holon_ratio"], motivo_clase=r["motivo"],
                n_aristas_grafo_final=m["n_aristas"], diam_grafo_final=m["diam"],
                giant_grafo_final=m["giant"], holon_grafo_final=m["holonomia"],
                grado_medio_grafo_final=2.0 * m["n_aristas"] / N_PILOTO,
                masa_total_ic=info_ic["masa_total"], carpeta=str(carpeta), ruta_ic=str(ruta_ic),
                t_motor_s=round(t_motor, 1), t_grafo_s=round(t_grafo, 1), t_ic_s=round(t_ic, 1))
    (carpeta / "meta_regla.json").write_text(json.dumps(meta, indent=2))
    # verificación cruzada #2: lo escrito en disco debe coincidir con lo que el CSV pedía
    leido = json.loads((carpeta / "meta_regla.json").read_text())
    assert leido["seed"] == seed and leido["kcap"] == kcap_esperado, \
        f"{rule_id}: meta_regla.json en disco no coincide con lo pedido — abortando"
    print(f"[{rule_id}] kcap={p['kcap']} K={p['K']} clase={r['clase']} "
          f"pend={r['pendiente_real']:.3f} aristas={m['n_aristas']} diam={m['diam']} "
          f"| motor {t_motor:.0f}s grafo {t_grafo:.0f}s ic {t_ic:.0f}s", flush=True)


# --------------------------------------------------------------------------------------------------
# MODO 4 — control: condición inicial SIN ESTRUCTURA (Erdős-Rényi) con el mismo n_aristas
# --------------------------------------------------------------------------------------------------
def modo_control(seed_random, n_aristas):
    """Control sin estructura EMPAREJADO EN ARISTAS: mismo pipeline de masa fija, misma caja, misma
    turbulencia, misma cantidad de aristas — pero el grafo es Erdős-Rényi puro, sin dinámica, sin
    recableo y sin `kcap`. Es la referencia de la pregunta 3 (¿kcap=7 cae al nivel de 'sin estructura'?)."""
    import grafo_random_layout_generar_ic_masa_fija as GRM
    from campo_velocidad_turbulento import factory as factory_turb, MACH_OBJETIVO
    from cs090_fase5b_phantom_adaptador import CS_SONORA, TURB_SEED

    seed_random = int(seed_random); n_aristas = int(n_aristas)
    rule_id = f"CONTROL-ER-a{n_aristas}-s{seed_random}"
    carpeta = BASE_SALIDA / rule_id
    carpeta.mkdir(parents=True, exist_ok=True)
    if (carpeta / "meta_regla.json").exists() and (carpeta / "cosmogenesis_ic.txt").exists():
        print(f"[{rule_id}] ya tiene IC — no se recomputa")
        return

    t0 = time.time()
    # MISMA llamada exacta que usan el adaptador de Fase V-B y grafo_random_masa_fija_generar.py
    vel_gen = factory_turb(CS_SONORA, seed=TURB_SEED, mach=MACH_OBJETIVO)
    dens_bar = np.ones(N_PILOTO)                       # ignorado a propósito por el generador (ver docstring)
    info = GRM.generar_control_random_masa_fija(
        n=N_PILOTO, dens_bar=dens_bar, n_aristas=n_aristas, seed_random=seed_random,
        seed_layout=seed_random, vel_generador=vel_gen,
        ruta_salida=str(carpeta / "cosmogenesis_ic.txt"))
    meta = dict(rule_id=rule_id, tarea="O3-D", clase="CONTROL_ER", seed=seed_random, N=N_PILOTO,
                seed_layout=seed_random, K=None, J=None, noise=None, meandeg=None, kcap=None,
                pendiente_corregida=None, n_aristas_grafo_final=n_aristas,
                grado_medio_grafo_final=2.0 * n_aristas / N_PILOTO,
                masa_total_ic=info["masa_total"], carpeta=str(carpeta),
                t_ic_s=round(time.time() - t0, 1))
    (carpeta / "meta_regla.json").write_text(json.dumps(meta, indent=2))
    print(f"[{rule_id}] IC lista en {time.time()-t0:.0f}s (aristas={n_aristas})", flush=True)


# --------------------------------------------------------------------------------------------------
# MODO 5 — phantom: corrida SERIAL + análisis + poda de dumps intermedios
# --------------------------------------------------------------------------------------------------
def modo_phantom(limite=None):
    from cs090_fase5b_correr import correr_una
    from cs090_fase5b_analizar import analizar_carpeta

    # Se exige `meta_regla.json` ADEMÁS del ASCII: el worker escribe primero la IC y sólo después el
    # meta, así que la presencia del meta garantiza que el ASCII está completo. Esto permite ir
    # corriendo Phantom sobre las reglas ya listas mientras otras todavía se están generando, sin
    # riesgo de leer un archivo a medio escribir.
    carpetas = sorted(c for c in BASE_SALIDA.iterdir()
                      if c.is_dir() and (c / "cosmogenesis_ic.txt").exists()
                      and (c / "meta_regla.json").exists())
    pendientes = [c for c in carpetas if not (c / "resultado_o3d.json").exists()]
    if limite:
        pendientes = pendientes[:int(limite)]
    print(f"[phantom] {len(pendientes)} corridas pendientes de {len(carpetas)} con IC", flush=True)

    t_inicio = time.time()
    for i, carpeta in enumerate(pendientes):
        t0 = time.time()
        info = correr_una(carpeta)
        fila = analizar_carpeta(carpeta)
        meta = json.loads((carpeta / "meta_regla.json").read_text())
        fila.update({k: meta.get(k) for k in
                     ("pendiente_corregida", "pendiente_null", "z_agg", "holon_ratio",
                      "giant_grafo_final", "grado_medio_grafo_final", "tarea")})
        fila["t_setup_s"] = info.get("t_setup"); fila["t_run_s"] = info.get("t_run")
        (carpeta / "resultado_o3d.json").write_text(json.dumps(fila, indent=2, default=str))
        # poda: se conservan el primer dump, el último y el .sink (el análisis no lee los del medio)
        dumps = sorted(carpeta.glob("cosmog_0*"))
        for d in dumps[1:-1]:
            d.unlink()
        print(f"[{i+1}/{len(pendientes)}] {carpeta.name}: frac_masa="
              f"{fila.get('fraccion_masa_en_sumideros')} n_sinks={fila.get('n_sumideros')} "
              f"kappa_v={fila.get('kappa_v_agregado')} ({time.time()-t0:.0f}s, "
              f"acumulado {(time.time()-t_inicio)/60:.1f} min)", flush=True)
    print(f"[phantom] FIN — {len(pendientes)} corridas en {(time.time()-t_inicio)/60:.1f} min", flush=True)


# --------------------------------------------------------------------------------------------------
# MODO 6 — analizar: consolidar el CSV crudo
# --------------------------------------------------------------------------------------------------
RUTA_CONTROL_HIST_CSV = Path(f"{_HERE}/cs090_fase6_o3d_control_historico.csv")
CARPETAS_CONTROL_HISTORICO = [
    Path("/Users/alexis/phantom_cs073/bateria_grafo_random_masa_fija/ic_masaFija_N2000_s1"),
    Path("/Users/alexis/phantom_cs073/bateria_grafo_random_masa_fija/ic_masaFija_N2000_s2"),
]


def modo_control_historico():
    """Lee (SÓLO LEE — no escribe nada dentro de esas carpetas, son de un experimento ya cerrado) las
    dos corridas de control Erdős-Rényi con masa fija a N=2000 que ya existían en el proyecto
    (`bateria_grafo_random_masa_fija`, 4945 aristas cada una). Son referencia gratis: mismo pipeline,
    misma caja, misma masa, misma turbulencia, corridas hace tiempo con los mismos parámetros de
    Phantom. Se vuelcan a un CSV propio en el directorio del proyecto."""
    from cs090_fase5b_analizar import analizar_carpeta
    filas = []
    for carpeta in CARPETAS_CONTROL_HISTORICO:
        if not carpeta.exists():
            print(f"  (falta {carpeta})"); continue
        fila = analizar_carpeta(carpeta)
        fila["rule_id"] = f"CONTROL-ER-HIST-{carpeta.name}"
        fila["clase"] = "CONTROL_ER_HIST"
        fila["n_aristas_grafo_final"] = 4945          # del encabezado del propio cosmogenesis_ic.txt
        fila["grado_medio_grafo_final"] = 2.0 * 4945 / 2000
        filas.append(fila)
        print(f"  {fila['rule_id']}: frac_masa={fila['fraccion_masa_en_sumideros']:.4f} "
              f"n_sinks={fila['n_sumideros']} kappa_v={fila['kappa_v_agregado']}")
    with open(RUTA_CONTROL_HIST_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader(); w.writerows(filas)
    print(f"[control histórico] {len(filas)} filas -> {RUTA_CONTROL_HIST_CSV}")


CAMPOS_CRUDO = ["rule_id", "clase", "kcap", "K", "J", "noise", "meandeg", "seed",
                "pendiente_corregida", "pendiente_null", "z_agg", "holon_ratio",
                "n_aristas_grafo_final", "grado_medio_grafo_final", "diam_grafo_final",
                "giant_grafo_final", "n_gas_inicial", "n_sumideros", "masa_gas_final",
                "masa_sumideros_final", "masa_total_final", "fraccion_masa_en_sumideros",
                "t_primer_sumidero", "masa_acretada_total", "kappa_v_agregado",
                "kappa_v_medio_valido", "n_kappa_indefinidos", "n_dump_final",
                "t_setup_s", "t_run_s", "carpeta"]


def modo_analizar():
    filas = []
    for carpeta in sorted(BASE_SALIDA.iterdir()):
        res = carpeta / "resultado_o3d.json"
        if not res.exists():
            continue
        d = json.loads(res.read_text())
        d["carpeta"] = carpeta.name
        filas.append({k: d.get(k) for k in CAMPOS_CRUDO})
    with open(RUTA_CRUDO_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_CRUDO)
        w.writeheader()
        w.writerows(filas)
    print(f"[analizar] {len(filas)} filas -> {RUTA_CRUDO_CSV}")
    return filas


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "candidatas"
    if modo == "candidatas":
        modo_candidatas(int(sys.argv[2]) if len(sys.argv) > 2 else N_CANDIDATAS_DEFECTO)
    elif modo == "seleccionar":
        modo_seleccionar()
    elif modo == "worker":
        modo_worker(sys.argv[2], sys.argv[3], sys.argv[4])
    elif modo == "control":
        modo_control(sys.argv[2], sys.argv[3])
    elif modo == "phantom":
        modo_phantom(sys.argv[2] if len(sys.argv) > 2 else None)
    elif modo == "control_historico":
        modo_control_historico()
    elif modo == "analizar":
        modo_analizar()
    else:
        raise SystemExit(f"modo desconocido: {modo}")

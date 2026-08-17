"""
cs090_fase7_f701_factorial.py — FASE VII, tarea F7-01: FACTORIAL ORTOGONAL `kcap` x NÚMERO DE ARISTAS (M)
==========================================================================================================

QUIÉN SOY / QUÉ HAGO
--------------------
`FASE6_O3D_barrido_kcap_phantom_CS.md` midió el barrido de `kcap` en Phantom y encontró el efecto más
limpio de toda la línea (η² = 0,949, ρ = −0,969, cuatro grupos sin solape). Pero también encontró el
problema que hace que ese número no se pueda interpretar: `kcap`, el grado medio del grafo y la
pendiente son casi la MISMA variable (r(kcap, grado medio) = +0,984 · VIF hasta 47,8). Con esa
colinealidad, "repartir" el efecto entre `kcap` y densidad es pedirle a los datos una distinción que
no contienen.

Esta tarea rompe la colinealidad por DISEÑO: un factorial ortogonal donde `kcap` y el número de
aristas del grafo final (`M`) se mueven por separado. Dos preguntas independientes:

  (1) Manteniendo `kcap` fijo, ¿qué pasa con la masa acretada al cambiar sólo M?
  (2) Manteniendo M fijo, ¿qué queda del efecto de `kcap`?

CÓMO SE IGUALA M — LA DECISIÓN DE DISEÑO CRÍTICA
------------------------------------------------
La forma obvia de igualar M sería podar al azar las aristas que sobran. **No se hace.** "Podar al azar"
es exactamente uno de los brazos del experimento F7-04 que corre en paralelo en esta misma sesión;
usarlo acá mezclaría el control de una tarea con el tratamiento de la otra.

M se iguala **POR SELECCIÓN**: dentro de un mismo `kcap`, M no es constante — depende de los otros
parámetros del sorteo (sobre todo `meandeg`, el grado medio del grafo Erdős-Rényi de partida, y de la
poda por costo). Se generan MUCHAS reglas por `kcap`, se mide el M de cada una (barato: 0,6 s, no hace
falta ni el coarse-graining ni Phantom) y se conservan sólo las que caen naturalmente en el M objetivo
dentro de una tolerancia declarada. Más caro en generación, pero el grafo que entra a Phantom es un
grafo que la regla produjo por su cuenta, sin ninguna intervención externa.

La selección mira M — que es una VARIABLE DE DISEÑO (el factor que se quiere ortogonalizar), conocida
antes de correr Phantom y antes de medir cualquier resultado. NO mira pendiente, ni clase, ni
holonomía, ni masa. Ésos son los desenlaces.

MODOS (línea de comando)
------------------------
  mapear [n]          genera n candidatas (filtro P1-P5 real) y mide el M de cada una -> mapa de M por kcap
  seleccionar         arma la grilla factorial a partir del mapa y del plan de celdas -> archivo de trabajos
  worker <rid> <seed> <kcap> <M>   una regla: motor (pendiente corregida) + grafo + clustering + IC
  phantom             corre Phantom SERIAL sobre todas las carpetas con IC y sin resultado
  analizar            consolida el CSV crudo + estadística de los dos R²

NADA EXISTENTE SE MODIFICA — todo se importa: `cs090_fase5_generador`, `cs090_fase5_motor`,
`cs090_fase5_clasificador`, `cs090_diam_corregido`, `cs090_fase5b_phantom_adaptador`,
`cs090_fase5b_correr`, `cs090_fase5b_analizar`. No se declara cierre ni veredicto: sólo números.
No se hacen commits.
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
# CONSTANTES DE LA TAREA (mismas que toda la línea Fase V-B / Fase VI, para poder comparar)
# --------------------------------------------------------------------------------------------------
EJE_A, EJE_B, EJE_C = "A2", "B0", "C2"
N_PILOTO = 2000              # piso de resolución SPH válido (MISTERIO_N500_vs_N2000_CS.md)
N_SWEEPS = 14
SEED_LAYOUT = 12345          # misma realización espacial que toda la línea Fase V-B
ESCALAS_B = (1, 2, 4, 8, 16)
N_SEEDS_NULL_TOPO = 3

SEED_BASE_F701 = 70701000
PREFIJO_RULE_ID = "A2-B0-C2-f701-r"

BASE_SALIDA = Path("/Users/alexis/phantom_cs073/bateria_fase7_f701_kcapM")
RUTA_MAPA_CSV = Path(f"{_HERE}/cs090_fase7_f701_mapa_M.csv")
RUTA_PLAN_JSON = Path(f"{_HERE}/cs090_fase7_f701_plan.json")
RUTA_TRABAJOS = Path(f"{_HERE}/cs090_fase7_f701_trabajos.txt")
RUTA_CRUDO_CSV = Path(f"{_HERE}/cs090_fase7_f701_crudo.csv")

# Semillas base YA usadas en el proyecto (heredado de cs090_fase6_o3d_barrido_kcap.py + la propia O3-D).
SEEDS_YA_USADAS = [
    90210, 156644, 271828, 371828, 471828, 471829, 571828, 823001,
    113477, 218903, 344251, 662819, 741037, 905683, 1128409, 1357061, 1604923, 1889347,
    2043761, 2296589, 2571043, 2814697, 3102859, 3389417, 3670213, 3948071, 4213589, 4507921,
    20260810, 20260811, 9314159, 3000001, 3000002, 3000003, 3000011, 3000012, 3000013,
]


def _ancho_cadena(n_candidatas):
    """generar_reglas_clase usa seed = seed_base + intento*97 + 1, y el filtro P1-P5 usa además
    seed + 500_000 como semilla de chequeo -> el ancho REAL de la cadena consumida incluye ese offset
    (la verificación de O3-D no lo contaba; acá sí, es la opción conservadora)."""
    return n_candidatas * 4 * 97 + 500_000 + 1


def _verificar_separacion(n_candidatas):
    ancho = _ancho_cadena(n_candidatas)
    d_min = min(abs(SEED_BASE_F701 - s) for s in SEEDS_YA_USADAS)
    print(f"[verificación] seed_base={SEED_BASE_F701}; ancho de cadena = {ancho}; "
          f"separación mínima contra las {len(SEEDS_YA_USADAS)} semillas base ya usadas = {d_min}")
    assert d_min > ancho, (
        f"la cadena puede tocar una semilla ya usada (separación {d_min} <= ancho {ancho}) — abortando")
    print(f"[verificación] OK: holgura {d_min/ancho:.1f}x")


# --------------------------------------------------------------------------------------------------
# MEDICIÓN BARATA DE M — el corazón de la selección
# --------------------------------------------------------------------------------------------------
def medir_M(seed, N=N_PILOTO, n_sweeps=N_SWEEPS):
    """Número de aristas del grafo FINAL de la regla, sin medir nada más.

    Reproduce bit a bit el mismo camino que `reconstruir_regla_a2b0c2` (mismo `p`, mismo rng derivado
    de `seed*5000+N`, mismo construir_A2 + dinamica_B0) pero SE SALTEA `medir()`, que es la parte cara
    (diámetro, componente gigante, muestreo de triángulos, holonomía) y que NO cambia ni una arista.
    El M que devuelve es idénticamente el `m["n_aristas"]` que devolvería la reconstrucción completa —
    y el worker lo verifica cruzadamente después, así que no es una promesa: se comprueba.

    Costo medido: ~0,6 s por regla a N=2000. Esto es lo que hace viable igualar M POR SELECCIÓN en vez
    de por poda."""
    p = GEN.generar_regla(EJE_A, EJE_B, EJE_C, idx=0, seed=int(seed))
    rng = np.random.default_rng(p["seed"] * 5000 + N)
    sustrato = MOT.construir_A2(N, rng, p)
    sustrato = MOT.dinamica_B0(sustrato, p, rng, n_sweeps, EJE_C)
    adj = sustrato["adj"]
    M = sum(len(a) for a in adj) // 2
    return M, p


def clustering_medio(adj):
    """Coeficiente de clustering local promedio (Watts-Strogatz), sobre los nodos con grado >= 2.

    Es uno de los dos ENDPOINTS geométricos congelados de esta línea (el otro es la pendiente
    continua). Mide 'qué fracción de los pares de amigos de un nodo son amigos entre sí' — la
    tendencia de la red a cerrar triángulos. Se calcula exacto (no muestreado): con grados de 2-5 y
    N=2000 el costo es despreciable."""
    tot, n = 0.0, 0
    for i, vec in enumerate(adj):
        k = len(vec)
        if k < 2:
            continue
        vl = list(vec)
        enlaces = 0
        for a in range(len(vl)):
            va = adj[vl[a]]
            for b in range(a + 1, len(vl)):
                if vl[b] in va:
                    enlaces += 1
        tot += 2.0 * enlaces / (k * (k - 1))
        n += 1
    return tot / n if n else 0.0


def _tarea_mapa(args):
    seed, idx = args
    t0 = time.time()
    M, p = medir_M(seed)
    return dict(rule_id=f"{PREFIJO_RULE_ID}{idx}", seed=int(seed), kcap=int(p["kcap"]), K=int(p["K"]),
                J=p["J"], noise=p["noise"], meandeg=p["meandeg"], M=int(M),
                grado_medio=round(2.0 * M / N_PILOTO, 4), t_s=round(time.time() - t0, 2))


def modo_mapear(n_candidatas=1200, n_proc=8):
    """Paso obligatorio ANTES de comprometer la grilla: mapear qué rango de M produce naturalmente
    cada kcap, para poder elegir la INTERSECCIÓN (que es donde vive la pregunta)."""
    from multiprocessing import Pool
    _verificar_separacion(n_candidatas)
    t0 = time.time()
    print(f"Generando {n_candidatas} candidatas {EJE_A}-{EJE_B}-{EJE_C} (filtro P1-P5 real)...", flush=True)
    admitidas, descartadas = GEN.generar_reglas_clase(
        EJE_A, EJE_B, EJE_C, n_reglas=n_candidatas, seed_base=SEED_BASE_F701,
        max_intentos=n_candidatas * 4)
    print(f"  admitidas={len(admitidas)} descartadas(P1-P5)={len(descartadas)} ({time.time()-t0:.0f}s)",
          flush=True)

    args = [(p["seed"], i) for i, p in enumerate(admitidas)]
    t1 = time.time()
    with Pool(n_proc) as pool:
        filas = pool.map(_tarea_mapa, args, chunksize=8)
    print(f"  M medido en {len(filas)} reglas en {time.time()-t1:.0f}s "
          f"({(time.time()-t1)/len(filas):.2f} s/regla x {n_proc} procesos)", flush=True)

    with open(RUTA_MAPA_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader(); w.writerows(filas)
    print(f"  -> {RUTA_MAPA_CSV}")
    resumen_mapa(filas)
    return filas


def resumen_mapa(filas=None):
    if filas is None:
        filas = [dict(r, kcap=int(r["kcap"]), M=int(r["M"])) for r in csv.DictReader(open(RUTA_MAPA_CSV))]
    por = defaultdict(list)
    for r in filas:
        por[int(r["kcap"])].append(int(r["M"]))
    print("\nRANGO DE M ALCANZABLE POR kcap")
    print(f"{'kcap':>5} {'n':>5} {'min':>6} {'p5':>6} {'p25':>6} {'mediana':>8} {'p75':>6} {'p95':>6} {'max':>6}")
    for k in sorted(por):
        v = np.array(sorted(por[k]))
        print(f"{k:>5} {len(v):>5} {v.min():>6} {int(np.percentile(v,5)):>6} {int(np.percentile(v,25)):>6} "
              f"{int(np.median(v)):>8} {int(np.percentile(v,75)):>6} {int(np.percentile(v,95)):>6} {v.max():>6}")
    ks = sorted(por)
    print("\nINTERSECCIONES (rango de M cubierto por CADA uno de los kcap del subconjunto)")
    for i in range(len(ks)):
        for j in range(i + 1, len(ks) + 1):
            sub = ks[i:j]
            if len(sub) < 2:
                continue
            lo = max(min(por[k]) for k in sub)
            hi = min(max(por[k]) for k in sub)
            if hi > lo:
                print(f"  kcap {sub}: M ∈ [{lo}, {hi}]  (ancho {hi-lo})")
    return por


# --------------------------------------------------------------------------------------------------
# MODO 2 — seleccionar: la grilla factorial, por SELECCIÓN (nunca por poda)
# --------------------------------------------------------------------------------------------------
def modo_seleccionar(plan_path=None):
    """Lee el plan (celdas kcap x M objetivo + tolerancia + reglas por celda) y elige, del mapa, las
    reglas cuyo M cae naturalmente dentro de la tolerancia. Si una celda no se puede llenar, se
    declara vacía — nunca se rellena podando."""
    plan = json.loads(Path(plan_path or RUTA_PLAN_JSON).read_text())
    filas = [dict(r, kcap=int(r["kcap"]), M=int(r["M"]), seed=int(r["seed"]), K=int(r["K"]))
             for r in csv.DictReader(open(RUTA_MAPA_CSV))]
    por = defaultdict(list)
    for r in filas:
        por[r["kcap"]].append(r)

    tol = float(plan["tolerancia_frac"])
    n_por_celda = int(plan["n_por_celda"])
    seleccion, vacias = [], []
    for celda in plan["celdas"]:
        kcap, M_obj = int(celda["kcap"]), int(celda["M_objetivo"])
        lo, hi = M_obj * (1 - tol), M_obj * (1 + tol)
        cand = [r for r in por.get(kcap, []) if lo <= r["M"] <= hi]
        # criterio de desempate DECLARADO Y CIEGO AL RESULTADO: los más cercanos al M objetivo,
        # desempate por seed ascendente. Nunca por pendiente/clase/masa.
        cand.sort(key=lambda r: (abs(r["M"] - M_obj), r["seed"]))
        elegidas = cand[:n_por_celda]
        if len(elegidas) < n_por_celda:
            vacias.append((kcap, M_obj, len(elegidas)))
        for r in elegidas:
            r = dict(r); r["M_objetivo"] = M_obj; r["celda"] = f"k{kcap}_M{M_obj}"
            seleccion.append(r)
        Ms = [r['M'] for r in elegidas]
        print(f"  celda kcap={kcap} M_obj={M_obj}: {len(elegidas)}/{n_por_celda} "
              f"(de {len(cand)} candidatas en rango)  M={Ms}")
    if vacias:
        print(f"  ATENCIÓN — celdas incompletas: {vacias}")

    with open(RUTA_TRABAJOS, "w") as f:
        for s in seleccion:
            f.write(f"{s['rule_id']} {s['seed']} {s['kcap']} {s['M']} {s['M_objetivo']}\n")
    Path(f"{_HERE}/cs090_fase7_f701_seleccion.json").write_text(json.dumps(seleccion, indent=2))
    print(f"[selección] {len(seleccion)} reglas -> {RUTA_TRABAJOS}")
    return seleccion


# --------------------------------------------------------------------------------------------------
# MODO 3 — worker: UNA regla. Motor (pendiente corregida) + grafo final + clustering + IC de Phantom.
# --------------------------------------------------------------------------------------------------
def _con_diam_corregido(fn, *a, **kw):
    """Ejecuta `fn` con `MOT._diam` sustituido en memoria por `DC.diam_gigante`. Mismo mecanismo ya
    usado y verificado por cs090_fase6_remedir_mecanismo.py y O3-D; ningún archivo cambia en disco."""
    _orig = MOT._diam
    try:
        MOT._diam = DC.diam_gigante
        return fn(*a, **kw)
    finally:
        MOT._diam = _orig


def modo_worker(rule_id, seed, kcap_esperado, M_esperado):
    seed = int(seed); kcap_esperado = int(kcap_esperado); M_esperado = int(M_esperado)
    carpeta = BASE_SALIDA / rule_id
    carpeta.mkdir(parents=True, exist_ok=True)
    if (carpeta / "meta_regla.json").exists() and (carpeta / "cosmogenesis_ic.txt").exists():
        print(f"[{rule_id}] ya tiene IC y meta — no se recomputa")
        return

    t0 = time.time()
    p = GEN.generar_regla(EJE_A, EJE_B, EJE_C, idx=0, seed=seed)
    assert p["kcap"] == kcap_esperado, (
        f"{rule_id}: generar_regla(seed={seed}) da kcap={p['kcap']}, se esperaba {kcap_esperado} "
        f"— POSIBLE COLISIÓN DE NOMBRE/SEMILLA, abortando")

    filas_motor = _con_diam_corregido(MOT.correr_regla_coarse, p, N=N_PILOTO, n_sweeps=N_SWEEPS,
                                      escalas_b=ESCALAS_B, n_seeds_null_topo=N_SEEDS_NULL_TOPO)
    r = clasificar_regla(filas_motor)
    t_motor = time.time() - t0

    t1 = time.time()
    p2, m = _con_diam_corregido(reconstruir_regla_a2b0c2, seed=seed, N=N_PILOTO, n_sweeps=N_SWEEPS)
    assert p2["seed"] == p["seed"] and p2["kcap"] == p["kcap"] and p2["K"] == p["K"], \
        f"{rule_id}: la reconstrucción no coincide con los parámetros de la regla"
    # verificación cruzada del atajo de mapeo: el M barato tiene que ser EXACTAMENTE el M completo
    assert m["n_aristas"] == M_esperado, (
        f"{rule_id}: M del grafo reconstruido = {m['n_aristas']} pero el mapa decía {M_esperado} "
        f"— el atajo medir_M() no reproduce la reconstrucción, abortando")
    clust = clustering_medio(m["adj_final"])
    t_grafo = time.time() - t1

    t2 = time.time()
    ruta_ic = carpeta / "cosmogenesis_ic.txt"
    info_ic = generar_ic_masa_fija_desde_grafo(m["adj_final"], N=N_PILOTO, seed_layout=SEED_LAYOUT,
                                               ruta_salida=str(ruta_ic))
    t_ic = time.time() - t2
    assert info_ic["n_aristas"] == M_esperado, f"{rule_id}: la IC escribió otro M"

    meta = dict(rule_id=rule_id, tarea="F7-01", clase=r["clase"], seed=seed, N=N_PILOTO,
                seed_layout=SEED_LAYOUT, K=p["K"], J=p["J"], noise=p["noise"], meandeg=p["meandeg"],
                kcap=p["kcap"], sim_thr_frac=p["sim_thr_frac"], M_objetivo=None,
                igualacion_M="seleccion",
                pendiente_corregida=r["pendiente_real"], pendiente_null=r["pendiente_null"],
                z_agg=r["z_agg"], holon_ratio=r["holon_ratio"], motivo_clase=r["motivo"],
                clustering_medio=round(clust, 6),
                n_aristas_grafo_final=m["n_aristas"], diam_grafo_final=m["diam"],
                giant_grafo_final=m["giant"], holon_grafo_final=m["holonomia"],
                grado_medio_grafo_final=2.0 * m["n_aristas"] / N_PILOTO,
                masa_total_ic=info_ic["masa_total"], carpeta=str(carpeta), ruta_ic=str(ruta_ic),
                t_motor_s=round(t_motor, 1), t_grafo_s=round(t_grafo, 1), t_ic_s=round(t_ic, 1))
    (carpeta / "meta_regla.json").write_text(json.dumps(meta, indent=2))
    leido = json.loads((carpeta / "meta_regla.json").read_text())
    assert leido["seed"] == seed and leido["kcap"] == kcap_esperado \
        and leido["n_aristas_grafo_final"] == M_esperado, \
        f"{rule_id}: meta_regla.json en disco no coincide con lo pedido — abortando"
    print(f"[{rule_id}] kcap={p['kcap']} M={m['n_aristas']} clase={r['clase']} "
          f"pend={r['pendiente_real']:.3f} clust={clust:.4f} "
          f"| motor {t_motor:.0f}s grafo {t_grafo:.0f}s ic {t_ic:.0f}s", flush=True)


# --------------------------------------------------------------------------------------------------
# MODO 4 — phantom: corrida SERIAL + análisis + poda de dumps intermedios
# --------------------------------------------------------------------------------------------------
def modo_phantom(limite=None):
    from cs090_fase5b_correr import correr_una
    from cs090_fase5b_analizar import analizar_carpeta

    carpetas = sorted(c for c in BASE_SALIDA.iterdir()
                      if c.is_dir() and (c / "cosmogenesis_ic.txt").exists()
                      and (c / "meta_regla.json").exists())
    pendientes = [c for c in carpetas if not (c / "resultado_f701.json").exists()]
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
                      "giant_grafo_final", "grado_medio_grafo_final", "clustering_medio",
                      "kcap", "K", "J", "noise", "meandeg", "seed", "clase", "igualacion_M",
                      "n_aristas_grafo_final", "diam_grafo_final", "tarea")})
        fila["t_setup_s"] = info.get("t_setup"); fila["t_run_s"] = info.get("t_run")
        (carpeta / "resultado_f701.json").write_text(json.dumps(fila, indent=2, default=str))
        dumps = sorted(carpeta.glob("cosmog_0*"))
        for d in dumps[1:-1]:
            d.unlink()
        print(f"[{i+1}/{len(pendientes)}] {carpeta.name}: frac_masa="
              f"{fila.get('fraccion_masa_en_sumideros')} n_sinks={fila.get('n_sumideros')} "
              f"kappa_v={fila.get('kappa_v_agregado')} ({time.time()-t0:.0f}s, "
              f"acumulado {(time.time()-t_inicio)/60:.1f} min)", flush=True)
    print(f"[phantom] FIN — {len(pendientes)} corridas en {(time.time()-t_inicio)/60:.1f} min", flush=True)


# --------------------------------------------------------------------------------------------------
# MODO 5 — analizar: consolidar el CSV crudo
# --------------------------------------------------------------------------------------------------
CAMPOS_CRUDO = ["rule_id", "clase", "igualacion_M", "kcap", "n_aristas_grafo_final",
                "grado_medio_grafo_final", "clustering_medio", "pendiente_corregida",
                "pendiente_null", "z_agg", "holon_ratio", "diam_grafo_final", "giant_grafo_final",
                "K", "J", "noise", "meandeg", "seed",
                "n_gas_inicial", "n_sumideros", "masa_gas_final", "masa_sumideros_final",
                "masa_total_final", "fraccion_masa_en_sumideros", "t_primer_sumidero",
                "masa_acretada_total", "kappa_v_agregado", "kappa_v_medio_valido",
                "n_kappa_indefinidos", "n_dump_final", "t_setup_s", "t_run_s", "carpeta"]


def modo_analizar():
    filas = []
    for carpeta in sorted(BASE_SALIDA.iterdir()):
        res = carpeta / "resultado_f701.json"
        if not res.exists():
            continue
        d = json.loads(res.read_text())
        d["carpeta"] = carpeta.name
        filas.append({k: d.get(k) for k in CAMPOS_CRUDO})
    with open(RUTA_CRUDO_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_CRUDO)
        w.writeheader(); w.writerows(filas)
    print(f"[analizar] {len(filas)} filas -> {RUTA_CRUDO_CSV}")
    return filas


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "mapear"
    if modo == "mapear":
        modo_mapear(int(sys.argv[2]) if len(sys.argv) > 2 else 1200,
                    int(sys.argv[3]) if len(sys.argv) > 3 else 8)
    elif modo == "resumen_mapa":
        resumen_mapa()
    elif modo == "seleccionar":
        modo_seleccionar(sys.argv[2] if len(sys.argv) > 2 else None)
    elif modo == "worker":
        modo_worker(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif modo == "phantom":
        modo_phantom(sys.argv[2] if len(sys.argv) > 2 else None)
    elif modo == "analizar":
        modo_analizar()
    else:
        raise SystemExit(f"modo desconocido: {modo}")

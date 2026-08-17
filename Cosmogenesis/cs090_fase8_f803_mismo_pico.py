"""
cs090_fase8_f803_mismo_pico.py — FASE VIII, tarea F8-03: MISMO PICO LOCAL, DISTINTA TOPOLOGÍA
==============================================================================================

LA PREGUNTA
-----------
F7-03/F8-01 dejaron esto: con N, aristas, grados nodo por nodo y número de triángulos clavados, el brazo
`solap` (triángulos apiñados, soporte chico) junta **+28 partículas** más de masa en sumideros que el
brazo `disp` (triángulos repartidos, soporte grande). F8-01 además mostró que la variable que queda en
pie es **el tamaño del soporte** (el Gini por nodo quedó excluido por intervención, +0.0 ± 3.1).

F7-05 mostró, por otro camino, que lo único que sobrevive a condicionar es **el pico local de densidad
inicial** (p90/mediana de la densidad a 8 vecinos, medida sobre la condición inicial).

Falta el control que puede CERRAR el mecanismo:

    ¿el apiñamiento actúa A TRAVÉS del pico local, o ADEMÁS de él?

- Si dos grafos con apiñamiento muy distinto pero **el mismo pico local** dan **la misma masa**, la
  cadena `topología → pico local → masa` se cierra.
- Si Phantom **todavía los distingue**, hay un segundo canal que no pasa por la concentración espacial.

CÓMO SE IGUALA EL PICO — **ESTRATEGIA (a), POR SELECCIÓN** (la preferida por la tarea)
--------------------------------------------------------------------------------------
No se toca ni el grafo ni el layout para forzar nada: **no hay ninguna intervención**. De cada grafo
base se fabrican `R_REALIZACIONES` realizaciones **independientes** de cada brazo (mismo criterio de
aceptación, misma mecánica, **distinta semilla de azar del brazo**). Todas las realizaciones de los dos
brazos se rebobinan al MISMO `T*` (el mínimo de todos los techos), así que:

  - todas las realizaciones de `solap` tienen apiñamiento alto (soporte chico) **por construcción**,
  - todas las de `disp` tienen apiñamiento bajo (soporte grande) **por construcción**,
  - pero cada realización cae en un sitio distinto del espacio, y su **pico local sale distinto**.

Con `R` realizaciones por brazo hay `R×R` emparejamientos posibles dentro de cada grafo base. Se elige
**el que minimiza |Δpico|**, y esa elección se hace mirando SÓLO el pico de la condición inicial —
**antes de correr Phantom, y sin haber visto ni una sola masa** (`cs090_fase8_f803_elegir_pares.py`).
El resto de las realizaciones también se corre: el promedio de los `R` `solap` contra el promedio de
los `R` `disp` reproduce el contraste SIN controlar, o sea el ancla de +28 partículas, **en la misma
batería y sobre los mismos grafos**. Ancla y control salen del mismo tubo.

Un control fallido también es información: si el mejor emparejamiento de un grafo sigue teniendo un
|Δpico| grande, ese grafo se declara **no igualado** y se dice, no se esconde.

QUÉ SE MANTIENE FIJO (verificado, no asumido)
---------------------------------------------
N=2000, masa total 18800, nº de aristas, **secuencia de grados nodo por nodo** (`np.array_equal`,
con `assert` que aborta), nº de triángulos (`dif_max` debe dar 0), **mismo `layout_resortes` con el
mismo `seed_layout=12345` en TODAS las realizaciones de TODOS los brazos** (regla dura de Fase VIII),
misma dilatación, misma turbulencia, mismo protocolo de sumideros.

EL PICO LOCAL, DEFINIDO ANTES DE CORRER
----------------------------------------
`pico_p90_med` = percentil 90 / mediana de la densidad local estimada con 8 vecinos
(`rho_i = 8 / r_{i,8}^3`), medida sobre las posiciones de `cosmogenesis_ic.txt`. Es **exactamente** la
vara `geoIC_knn8_p90_med` de `cs090_fase7_f705_geometria_ic_todas.py` (se copia la fórmula, 4 líneas, en
vez de importar el módulo entero que barre todo el disco). Se reportan además `pico_cv` (CV de la misma
densidad) y `pico_max_med` (máximo/mediana), para poder decir si la igualación aguanta con otras varas.

OBSERVABLE PRINCIPAL, DECLARADO DE ANTEMANO
--------------------------------------------
**Fracción de masa en sumideros de Phantom** (`icreate_sinks=1`, `rho_crit_cgs=1000`), tal como la lee
`cs090_fase5b_analizar.analizar_carpeta`. NO se usa FoF laxo (aviso de F8-05: el observable puede
invertir el signo si se cambia la vara). Secundarios: pico local en la IC, tamaño del soporte de cada
brazo, tiempo al primer sumidero y κ_V.

QUÉ REUSA Y QUÉ NO TOCA
------------------------
Sólo importa, jamás modifica: `cs090_fase7_f703_organizacion` (motor de los brazos `solap`/`disp`,
`replay`, `techo_conectado`, `medir_organizacion`), `cs090_fase7_f702_escalera` (swap que conserva
grados + selección de grafos base), `cs090_fase8_f801_desacople` (`medir_carga_aristas`, `guardar_grafo`
npz), `cs090_fase8_f800_grafos` (formato canónico + sha256), `cs090_fase6_o3b_rewiring`,
`cs090_fase5b_phantom_adaptador`. Prefijo `f803`, jamás usado antes. No corre Phantom (eso es
`cs090_fase8_f803_correr.py`). No declara cierre ni veredicto.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)

import cs090_fase6_o3b_rewiring as O3B                   # sólo import
import cs090_fase7_f702_escalera as F702                 # sólo import (swap que conserva grados)
import cs090_fase7_f703_organizacion as F703             # sólo import (motor de brazos de F7-03)
import cs090_fase8_f801_desacople as F801                # sólo import (medidas de carga + npz)
import cs090_fase8_f800_grafos as F800                   # sólo import (formato canónico + sha256)
from cs090_fase5b_phantom_adaptador import (             # sólo import (pipeline validado Fase V-B)
    reconstruir_regla_a2b0c2, generar_ic_masa_fija_desde_grafo,
)

# ---------------------------------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------------------------------
N_NODOS = F703.N_NODOS               # 2000
N_SWEEPS = F703.N_SWEEPS             # 14
SEED_LAYOUT = F703.SEED_LAYOUT       # 12345 — MISMO layout en todas las realizaciones (regla de la fase)

MULT_SEED_DES = 10800                # multiplicador NUEVO. Ya usados en la línea:
                                     # 1000/2000/5000/6000/7000/7500/8000/9100/9700/9800/10300.
PASO_REALIZACION = 7919              # primo; separa las semillas de una realización a la siguiente

FACTOR_INTENTOS_BAJAR = F702.FACTOR_INTENTOS_BAJAR       # mismo piso que F7-02/F7-03/F8-01
FACTOR_INTENTOS_SUBIR = F703.FACTOR_INTENTOS_SUBIR       # 60 × aristas
FRAC_GIGANTE_MINIMA = F702.FRAC_GIGANTE_MINIMA           # 0.97

BRAZOS = ("solap", "disp")           # los dos de F7-03, IMPORTADOS TAL CUAL (+28 partículas de gap)
R_REALIZACIONES = 3                  # realizaciones independientes por brazo -> R×R emparejamientos

BASE_SALIDA = "/Users/alexis/phantom_cs073/bateria_fase8_f803_mismo_pico"
RUTA_ESTRUCTURA = f"{HERE}/cs090_fase8_f803_estructura.csv"


# =============================================================================================
# 1) EL PICO LOCAL — misma vara que F7-05 (`geoIC_knn8_p90_med`)
# =============================================================================================
def medir_pico_ic(ruta_ic, k=8):
    """Pico local de densidad de una condición inicial ya escrita en disco.

    Se lee el archivo tal como lo dejó el generador (no las posiciones en memoria) para que la vara sea
    literalmente la misma que la de `cs090_fase7_f705_geometria_ic_todas.py`, que barre el disco.

    Devuelve p90/mediana (el `pico` de F7-05), el CV de la misma densidad y máximo/mediana."""
    d = np.loadtxt(ruta_ic, skiprows=2)
    pos = d[:, :3]
    arbol = cKDTree(pos)
    dist, _ = arbol.query(pos, k=k + 1)
    r_k = dist[:, k]
    rho = k / (r_k ** 3 + 1e-300)
    med = float(np.median(rho))
    return dict(pico_p90_med=float(np.percentile(rho, 90) / med),
                pico_cv=float(rho.std() / rho.mean()),
                pico_max_med=float(rho.max() / med),
                pico_p99_med=float(np.percentile(rho, 99) / med),
                pico_mediana_rho=med)


# =============================================================================================
# 2) PROCESAR un grafo base: piso común -> 2 brazos × R realizaciones -> mismo T* -> IC -> pico
# =============================================================================================
def procesar_una(sel, generar_ic=True, r_realizaciones=R_REALIZACIONES):
    rid, seed = sel["rule_id"], sel["seed"]
    t_ini = time.time()

    p, m = reconstruir_regla_a2b0c2(seed=seed, N=N_NODOS, n_sweeps=N_SWEEPS)
    adj_orig = O3B._a_lista(m["adj_final"], N_NODOS)
    E_orig = O3B.aristas_set(adj_orig, N_NODOS)
    g_orig = O3B.grados(adj_orig, N_NODOS)
    gig_orig = O3B.tam_gigante(adj_orig, N_NODOS)
    c_orig, tr_orig, t_orig = O3B.clustering(adj_orig, N_NODOS)
    piso_gigante = FRAC_GIGANTE_MINIMA * gig_orig

    # ---- piso común: se destruyen los triángulos (misma función y dosis que F7-02/F7-03/F8-01) ----
    semilla_base = int(seed) * MULT_SEED_DES + 17
    rng_piso = np.random.default_rng(semilla_base)
    adj_p, edges_p, idx_p = F702._estado_desde_adj(adj_orig, N_NODOS)
    n_aristas = len(edges_p)
    t0 = time.time()
    d_baja, acc_baja = F702.bajar_clustering(adj_p, edges_p, idx_p, N_NODOS, rng_piso,
                                             FACTOR_INTENTOS_BAJAR * n_aristas)
    t_piso = t_orig + d_baja
    adj_piso = [set(s) for s in adj_p]
    t_bajar_s = time.time() - t0
    print(f"    piso: tri {t_orig} -> {t_piso}  ({acc_baja} swaps, {t_bajar_s:.1f}s)", flush=True)

    # ---- cada (brazo, realización) desde el MISMO piso, hasta su techo ----
    variantes = {}
    for j, modo in enumerate(BRAZOS):
        for r in range(r_realizaciones):
            clave = f"{modo}{r}"
            rng_b = np.random.default_rng(semilla_base + 101 * (j + 1) + PASO_REALIZACION * r)
            adj_b, edges_b, idx_b = F702._estado_desde_adj(adj_piso, N_NODOS)
            tb = time.time()
            swaps, t_por_swap, hist = F703.cerrar_triangulos(
                adj_b, edges_b, idx_b, N_NODOS, rng_b, modo, t_piso,
                FACTOR_INTENTOS_SUBIR * n_aristas)
            k_techo, t_techo = F703.techo_conectado(hist, piso_gigante)
            variantes[clave] = dict(brazo=modo, realizacion=r, swaps=swaps, t_por_swap=t_por_swap,
                                    k_techo=k_techo, t_techo=t_techo, n_aceptados=len(swaps),
                                    t_swaps_s=round(time.time() - tb, 1))
            print(f"    {clave:>7}: techo_conectado tri={t_techo} (k={k_techo}/{len(swaps)}) "
                  f"[{variantes[clave]['t_swaps_s']}s]", flush=True)

    # ---- T* = el mínimo de TODOS los techos; cada variante se rebobina al swap más cercano ----
    claves = list(variantes)
    T_obj = min(variantes[c]["t_techo"] for c in claves)
    tris_conseguidos = {}
    for c in claves:
        v = variantes[c]
        tps, k_max = v["t_por_swap"], v["k_techo"]
        k_mejor = 0 if k_max == 0 else min(range(1, k_max + 1),
                                           key=lambda k: (abs(tps[k - 1] - T_obj), k))
        v["k_elegido"] = k_mejor
        v["adj"] = F703.replay(adj_piso, v["swaps"], k_mejor)
        tris_conseguidos[c] = (tps[k_mejor - 1] if k_mejor else t_piso)
    dif_max = max(tris_conseguidos.values()) - min(tris_conseguidos.values())
    print(f"    T*={T_obj}  conseguidos={tris_conseguidos}  dif_max={dif_max}", flush=True)

    adjs_null = O3B.nulls_topo_de(seed, N_NODOS, len(E_orig))
    rng_med = np.random.default_rng(semilla_base + 7)

    filas = []
    for c in claves:
        v = variantes[c]
        adj_v = v["adj"]
        # --- VERIFICACIÓN NUMÉRICA nodo por nodo (no se asume: es el control que valida todo) ---
        g_v = O3B.grados(adj_v, N_NODOS)
        iguales = bool(np.array_equal(g_orig, g_v))
        n_dif = int((g_orig != g_v).sum())
        assert iguales, f"{rid}/{c}: la secuencia de grados NO se conservó ({n_dif} nodos difieren)"
        E_v = O3B.aristas_set(adj_v, N_NODOS)
        assert len(E_v) == len(E_orig), f"{rid}/{c}: cambió el nº de aristas"
        assert all(i not in adj_v[i] for i in range(N_NODOS)), f"{rid}/{c}: hay un bucle i-i"

        cl, tr, ntri = O3B.clustering(adj_v, N_NODOS)
        org = F703.medir_organizacion(adj_v, N_NODOS, rng_med)
        assert org["n_triangulos_medido"] == ntri, f"{rid}/{c}: dos conteos de triángulos distintos"
        carga = F801.medir_carga_aristas(adj_v, N_NODOS)
        pc = O3B.pendiente_corregida_de_grafo(O3B.canonicalizar(adj_v, N_NODOS), N_NODOS, seed, adjs_null)

        fila = dict(
            rule_id=rid, seed=seed, lote=sel["lote"], K=sel["K"], kcap=sel["kcap"],
            variante=c, brazo=v["brazo"], realizacion=v["realizacion"],
            n_nodos=N_NODOS, n_aristas=len(E_v), grado_medio=2.0 * len(E_v) / N_NODOS,
            grados_identicos=iguales, n_nodos_grado_distinto=n_dif,
            solapamiento_aristas=len(E_orig & E_v) / len(E_orig),
            clustering_local=cl, transitividad=tr, n_triangulos=ntri,
            T_objetivo=T_obj, dif_max_triangulos=dif_max,
            gigante=O3B.tam_gigante(adj_v, N_NODOS),
            n_componentes=F702.n_componentes(adj_v, N_NODOS),
            asortatividad=F702.asortatividad_grados(adj_v, N_NODOS),
            pendiente_corr=pc["pendiente"], z_agg=pc["z_agg"], diams=pc["diams"], n_cajas=pc["n_cajas"],
            clustering_original=c_orig, transitividad_original=tr_orig, n_triangulos_original=t_orig,
            t_piso=t_piso, t_techo_variante=v["t_techo"],
            n_swaps_aceptados=v["n_aceptados"], k_elegido=v["k_elegido"],
            t_swaps_s=v["t_swaps_s"], acc_baja=acc_baja, t_bajar_s=round(t_bajar_s, 1),
            gigante_original=gig_orig, semilla_base=semilla_base,
            pendiente_corr_csv_ref=sel["pendiente_corregida"],
            frac_masa_fase5b=sel["frac_masa_fase5b"],
        )
        fila.update(org)
        fila.update(carga)

        if generar_ic:
            carpeta = f"{BASE_SALIDA}/{rid}_s{seed}_f803_{c}"
            os.makedirs(carpeta, exist_ok=True)
            F801.guardar_grafo(adj_v, N_NODOS, f"{carpeta}/grafo_f803.npz")     # npz (formato F8-01)
            ruta_can = F800.guardar_grafo(adj_v, f"{carpeta}/grafo_f803.grafo.gz", N=N_NODOS,
                                          meta=dict(tarea="FASE8_F803", variante=c, rule_id=rid,
                                                    seed=seed))                 # canónico (formato F8-00)
            t1 = time.time()
            ruta_ic = f"{carpeta}/cosmogenesis_ic.txt"
            info = generar_ic_masa_fija_desde_grafo(adj_v, N=N_NODOS, seed_layout=SEED_LAYOUT,
                                                    ruta_salida=ruta_ic)
            fila["t_ic_s"] = round(time.time() - t1, 2)
            pico = medir_pico_ic(ruta_ic)          # <-- el pico local, medido sobre la IC ya escrita
            fila.update(pico)
            fila["masa_total_ic"] = info["masa_total"]
            meta = dict(
                tarea="FASE8_F803_mismo_pico_distinta_topologia", brazo=v["brazo"], variante=c,
                realizacion=v["realizacion"], rule_id=rid, clase="III",
                seed=seed, lote=sel["lote"], N=N_NODOS, seed_layout=SEED_LAYOUT,
                K=p["K"], J=p["J"], noise=p["noise"], meandeg=p["meandeg"], kcap=p["kcap"],
                sim_thr_frac=p["sim_thr_frac"],
                n_aristas_grafo_final=len(E_v), grado_medio_grafo_final=2.0 * len(E_v) / N_NODOS,
                grados_identicos_al_original=iguales,
                semilla_base=semilla_base, solapamiento_aristas=fila["solapamiento_aristas"],
                clustering_local=cl, transitividad=tr, n_triangulos=ntri,
                T_objetivo=T_obj, dif_max_triangulos=dif_max,
                gigante=fila["gigante"], pendiente_corregida=pc["pendiente"],
                frac_aristas_multi_tri=org["frac_aristas_multi_tri"],
                tri_por_arista_media=org["tri_por_arista_media"],
                tri_por_arista_max=carga["tri_por_arista_max"],
                frac_aristas_en_triangulo=org["frac_aristas_en_triangulo"],
                gini_tri_nodo=org["gini_tri_nodo"], modularidad_tri=org["modularidad_tri"],
                masa_total_ic=info["masa_total"], carpeta=carpeta,
                grafo_guardado="grafo_f803.npz", grafo_canonico="grafo_f803.grafo.gz",
                **pico,
            )
            with open(f"{carpeta}/meta_regla.json", "w") as f:
                json.dump(meta, f, indent=2)
            F800.anotar_hash_en_meta(f"{carpeta}/meta_regla.json", adj_v, N=N_NODOS,
                                     archivo_grafo=os.path.basename(str(ruta_can)))
            fila["carpeta"] = carpeta

        filas.append(fila)
        print(f"    {c:>7}  tri={ntri:<5} A={org['tri_por_arista_media']:.3f} "
              f"D={org['frac_aristas_en_triangulo']:.4f} multi={org['frac_aristas_multi_tri']:.3f} "
              f"gini_n={org['gini_tri_nodo']:.3f} gig={fila['gigante']} "
              f"pend={pc['pendiente']:.4f}"
              + (f" PICO={fila.get('pico_p90_med'):.4f} ic={fila.get('t_ic_s')}s" if generar_ic else ""),
              flush=True)

    for f in filas:
        f["t_total_grafo_s"] = round(time.time() - t_ini, 1)
    return filas


def main(indices=None, generar_ic=True, sufijo_csv="", n_por_lote=3, r_real=R_REALIZACIONES):
    elegidos = F702.seleccionar_grafos(n_por_lote=n_por_lote)   # mismos grafos base de F7-02/F7-03/F8-01
    if indices is not None:
        elegidos = [elegidos[i] for i in indices]
    print(f"[f803] {len(elegidos)} grafos base (generar_ic={generar_ic}) brazos={BRAZOS} "
          f"R={r_real}", flush=True)

    todas, t0 = [], time.time()
    for k, sel in enumerate(elegidos):
        print(f"[{k+1}/{len(elegidos)}] {sel['rule_id']} seed={sel['seed']} lote={sel['lote']}",
              flush=True)
        todas.extend(procesar_una(sel, generar_ic=generar_ic, r_realizaciones=r_real))

    ruta = RUTA_ESTRUCTURA.replace(".csv", f"{sufijo_csv}.csv")
    with open(ruta, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(todas[0].keys()))
        w.writeheader()
        w.writerows(todas)
    print(f"\n[f803] {len(todas)} filas -> {ruta.split('/')[-1]}  (total {time.time()-t0:.0f}s)")
    return todas


if __name__ == "__main__":
    # uso: python3.9 cs090_fase8_f803_mismo_pico.py [idx0,idx1,...] [--sin-ic] [--sufijo=_s0]
    #                                               [--n-lote=3] [--r=3]
    idxs, gen_ic, suf, npl, rr = None, True, "", 3, R_REALIZACIONES
    for arg in sys.argv[1:]:
        if arg == "--sin-ic":
            gen_ic = False
        elif arg.startswith("--sufijo="):
            suf = arg.split("=", 1)[1]
        elif arg.startswith("--n-lote="):
            npl = int(arg.split("=", 1)[1])
        elif arg.startswith("--r="):
            rr = int(arg.split("=", 1)[1])
        else:
            idxs = [int(x) for x in arg.split(",")]
    main(indices=idxs, generar_ic=gen_ic, sufijo_csv=suf, n_por_lote=npl, r_real=rr)

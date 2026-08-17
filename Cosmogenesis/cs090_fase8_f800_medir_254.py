#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F8-00 · RUNNER — regenerar y medir los grafos de las 254 corridas del dataset unificado
=======================================================================================

QUÉ HACE ESTE SCRIPT (a nivel módulo)
--------------------------------------
`cs090_fase7_f705_dataset_unificado.csv` tiene 254 corridas de Phantom, pero el apiñamiento de
triángulos está medido en sólo 24 de ellas (las de O3-B): los grafos no se guardaron. Este script
**los vuelve a construir** — son deterministas desde `(seed, N)` — los **mide en todas** las variables
de apiñamiento que Fase VII señaló como relevantes, los **guarda en disco** con sello verificable
(`cs090_fase8_f800_grafos.py`) y escribe un CSV **nuevo** con las 254 filas enriquecidas.

Analogía: cada corrida era una construcción de la que sólo se anotó el resultado final. Los planos se
tiraron, pero la receta para volver a levantarla, paso a paso, quedó escrita. Acá se vuelve a levantar
cada una, se la mide con la misma cinta métrica en todas, se archivan los planos y se comprueba —
contra las 24 que sí se habían medido — que la reconstrucción da exactamente lo mismo.

LA RECETA DE CADA FAMILIA (de dónde sale el grafo de cada fila)
----------------------------------------------------------------
Ninguna de estas recetas es nueva: cada una se copia de la llamada exacta que hizo el runner original,
importando su módulo sin tocarlo.

  F5B_40pares / O3A_N4000 / OUT_pendNEG / O3D_kcap(brazo=regla) / O3B_rewiring(brazo=orig)
      `reconstruir_regla_a2b0c2(seed, N, n_sweeps=14)`  (cs090_fase5b_phantom_adaptador)
  O3B_rewiring (brazo=rewire)
      el original + `barajar_aristas(adj, N, seed=seed*9100+7, factor_swaps=10)`
      (mismas constantes MULT_SEED_REWIRE / FACTOR_SWAPS de cs090_fase6_o3b_rewiring)
  O3C_factorial (brazo = c1..c4)
      `generar_regla(A2,B0,C2, idx=0, seed)` + `O3C._grafo_final_de_condicion(brazo, p, N)`
  O3E_memoria (brazo = mem / nomem)
      `O3E.reconstruir(seed, ventana_memoria = None si mem, 1 si nomem)`
  O3D_kcap (brazo = control_ER)
      Erdős-Rényi puro: `generar_grafo_erdos_renyi(N, n_aristas, seed)` con (n_aristas, seed) leídos
      del propio `rule_id` = `CONTROL-ER-a{aristas}-s{seed}` (así lo nombró el runner de O3-D)
  O3D_control_hist
      Erdős-Rényi histórico de `grafo_random_masa_fija_generar.py`, cuya semilla es `1000*N + k`
      (k = 1 ó 2 del nombre `ic_masaFija_N2000_s{k}`); se VERIFICA contra la cabecera del
      `cosmogenesis_ic.txt` de esa carpeta antes de aceptarla.

QUÉ SE MIDE (todas las variables de apiñamiento pedidas por F8-00)
-------------------------------------------------------------------
  Apiñamiento sobre las ARISTAS
    `f800_tri_ar_media_sop`, `_mediana_sop`, `_max`, `_p99_sop`  — triángulos por arista, entre las
        aristas que sostienen al menos un triángulo (misma convención que `tri_por_arista_media` de
        F7-03, para poder comparar con lo ya publicado)
    `f800_tri_ar_media_todas` — el mismo promedio pero sobre TODAS las aristas (sube el peso de las
        aristas vacías; se reporta aparte porque no es la misma pregunta)
    `f800_frac_aristas_en_triangulo` — fracción de aristas que sostienen al menos un triángulo
    `f800_frac_aristas_multi_tri` — SOLAPAMIENTO: de las aristas con triángulo, cuántas están en ≥2
  Apiñamiento sobre los NODOS
    `f800_gini_tri_nodo` — concentración de triángulos por nodo (0 = parejo, →1 = todo en un nodo)
    `f800_tri_por_nodo_max`, `f800_frac_nodos_en_triangulo`
  Forma del conjunto de triángulos
    `f800_n_comp_tri` (cúmulos por NODO compartido, definición de F7-03), `f800_n_comp_tri_arista`
        (cúmulos por ARISTA compartida — el eje de solapamiento), `f800_frac_mayor_comp_tri`,
        `f800_tam_medio_comp_tri`, `f800_modularidad_tri`, `f800_dist_media_tri`, `f800_dist_media_azar`
  Como DATO, nunca como variable explicativa (F7-03: falla en el signo con triángulos fijos)
    `f800_clustering`, `f800_transitividad`, `f800_n_triangulos`
  Estructurales de siempre
    `f800_n_aristas`, `f800_grado_medio`, `f800_giant` (fracción), `f800_diam` (diámetro corregido de
    la gigante, `cs090_diam_corregido`), `f800_asortatividad`, `f800_pendiente_nativa` y
    `f800_pendiente_canon`

POR QUÉ HAY DOS PENDIENTES
---------------------------
`cs080_renormalizacion.cajas_bfs` recorre `for v in adj[u]`: el ORDEN DE ITERACIÓN de los `set` de
Python cambia la partición en cajas y mueve la pendiente en la 2ª-3ª decimal (documentado en
`cs090_fase6_o3b_rewiring._a_lista`). `f800_pendiente_nativa` se mide sobre el objeto TAL CUAL sale del
motor — reproduce el histórico bit a bit. `f800_pendiente_canon` se mide sobre la forma canónica
(vecinos insertados de menor a mayor), que es la que se recupera al leer un grafo del disco. Se
publican las dos y su diferencia, para que nadie confunda una cosa con la otra.

VERIFICACIONES QUE CORRE SOLO
------------------------------
  1. contra las 24 filas de O3-B con clustering ya medido: clustering, transitividad y nº de
     triángulos tienen que dar IGUAL (tolerancia 1e-12).
  2. contra las 254 filas: `n_aristas` reconstruido vs. `n_aristas` histórico.
  3. contra las 217 filas con pendiente: `pendiente` reconstruida vs. histórica.
  4. sello sha256: se guarda el grafo, se vuelve a leer del disco y se comprueba que el conjunto de
     aristas es idéntico.

SALIDAS
-------
  cs090_fase8_f800_dataset_enriquecido.csv   las mismas 254 filas + columnas `f800_*`
  cs090_fase8_f800_correlaciones.csv         matriz de correlaciones entre medidas de apiñamiento
  cs090_fase8_f800_verificacion.csv          fila a fila, reconstruido vs. histórico
  cs090_fase8_f800_medir_254.log             registro completo
  grafos_f800/<exp>/<rule_id>__s<seed>__<brazo>__N<N>.grafo.gz     los grafos, con sello

QUÉ NO HACE
-----------
No corre Phantom. No modifica ningún script, CSV ni `meta_regla.json` existente. No declara cierre.

USO
---
    python3.9 cs090_fase8_f800_medir_254.py                 # todo (usa caché si ya hay)
    python3.9 cs090_fase8_f800_medir_254.py --limite 5      # prueba corta
    python3.9 cs090_fase8_f800_medir_254.py --workers 1     # serial (depuración)
    python3.9 cs090_fase8_f800_medir_254.py --solo-analisis # rehace CSV/correlaciones desde la caché
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)

import cs090_fase8_f800_grafos as G8                       # módulo nuevo de persistencia (F8-00)

RUTA_DATASET = f"{HERE}/cs090_fase7_f705_dataset_unificado.csv"
RUTA_ENRIQUECIDO = f"{HERE}/cs090_fase8_f800_dataset_enriquecido.csv"
RUTA_CORRELACIONES = f"{HERE}/cs090_fase8_f800_correlaciones.csv"
RUTA_VERIFICACION = f"{HERE}/cs090_fase8_f800_verificacion.csv"
RUTA_CACHE = f"{HERE}/cs090_fase8_f800_cache.jsonl"
RUTA_LOG = f"{HERE}/cs090_fase8_f800_medir_254.log"
DIR_GRAFOS = f"{HERE}/grafos_f800"

N_SWEEPS = 14
MULT_SEED_MEDICION = 9900        # multiplicador NUEVO para el rng de las mediciones que muestrean
                                 # (distancias entre triángulos). Ya usados en la línea:
                                 # 1000/2000/5000/6000/7000/7500/8000/9100/9700/9800 — 9900 no colisiona.

_log = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _log.append(s)


# =============================================================================================
# 1) RECETAS DE RECONSTRUCCIÓN — una por familia, cada una copiada del runner original
# =============================================================================================
def _int(x, defecto=None):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return defecto
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return defecto
        return int(float(s))
    except (TypeError, ValueError):
        return defecto


def reconstruir_grafo(fila):
    """Devuelve (adj_nativo, N, receta) para una fila del dataset unificado.

    `adj_nativo` es el objeto TAL CUAL lo devuelve el motor (o el generador ER), sin canonicalizar:
    hace falta así para poder reproducir la pendiente histórica bit a bit."""
    import cs090_fase5_generador as GEN
    from cs090_fase5b_phantom_adaptador import reconstruir_regla_a2b0c2

    exp = fila["exp"]
    brazo = str(fila.get("brazo", "") or "")
    rid = str(fila["rule_id"])
    N = _int(fila["N_nodos"])
    seed = _int(fila["seed"])

    # --- familia 1: la reconstrucción base A2-B0-C2 -------------------------------------------
    if exp in ("F5B_40pares", "O3A_N4000", "OUT_pendNEG") or \
       (exp == "O3D_kcap" and brazo == "regla") or \
       (exp == "O3B_rewiring" and brazo == "orig"):
        p, m = reconstruir_regla_a2b0c2(seed=seed, N=N, n_sweeps=N_SWEEPS)
        return m["adj_final"], N, f"reconstruir_regla_a2b0c2(seed={seed}, N={N})"

    # --- familia 2: el gemelo reconfigurado de O3-B --------------------------------------------
    if exp == "O3B_rewiring" and brazo == "rewire":
        import cs090_fase6_o3b_rewiring as O3B
        from cs072_modulos.piezas.p_semilla_causal import barajar_aristas
        p, m = reconstruir_regla_a2b0c2(seed=seed, N=N, n_sweeps=N_SWEEPS)
        adj_orig = O3B._a_lista(m["adj_final"], N)
        s_rw = int(seed * O3B.MULT_SEED_REWIRE + 7)
        adj_rw = O3B._a_lista(barajar_aristas(adj_orig, N, seed=s_rw,
                                              factor_swaps=O3B.FACTOR_SWAPS), N)
        return adj_rw, N, f"barajar_aristas(orig(seed={seed}), seed_rewire={s_rw}, x{O3B.FACTOR_SWAPS})"

    # --- familia 3: las 4 condiciones del factorial mecanístico de O3-C -------------------------
    if exp == "O3C_factorial":
        import cs090_fase6_o3c_factorial_mecanistico as O3C
        p = GEN.generar_regla("A2", "B0", "C2", idx=0, seed=seed)
        m = O3C._grafo_final_de_condicion(brazo, p, N=N, n_sweeps=N_SWEEPS)
        return m["adj_final"], N, f"O3C._grafo_final_de_condicion({brazo}, seed={seed}, N={N})"

    # --- familia 4: memoria / sin memoria de O3-E ----------------------------------------------
    if exp == "O3E_memoria":
        import cs090_fase6_o3e_memoria as O3E
        ventana = None if brazo == "mem" else 1
        p, m, _diag = O3E.reconstruir(seed, ventana_memoria=ventana, N=N, n_sweeps=N_SWEEPS)
        return m["adj_final"], N, f"O3E.reconstruir(seed={seed}, ventana={ventana}, N={N})"

    # --- familia 5: controles Erdős-Rényi de O3-D (mismo nº de aristas, sin dinámica) -----------
    if exp == "O3D_kcap" and brazo == "control_ER":
        from grafo_random_layout_generar_ic import generar_grafo_erdos_renyi
        # el runner de O3-D nombró la carpeta CONTROL-ER-a{n_aristas}-s{seed_random}
        cuerpo = rid.replace("CONTROL-ER-a", "")
        n_ar_txt, s_txt = cuerpo.split("-s")
        n_ar, s_er = int(n_ar_txt), int(s_txt)
        if seed is not None and seed != s_er:
            raise ValueError(f"{rid}: seed del CSV ({seed}) != seed del nombre ({s_er})")
        adj_d, _e, _r = generar_grafo_erdos_renyi(N, n_ar, seed=s_er)
        return [set(adj_d[i]) for i in range(N)], N, \
            f"generar_grafo_erdos_renyi(N={N}, aristas={n_ar}, seed={s_er})"

    # --- familia 6: los 2 controles ER históricos (masa fija) ------------------------------------
    if exp == "O3D_control_hist":
        from grafo_random_layout_generar_ic import generar_grafo_erdos_renyi
        k = int(rid.rsplit("_s", 1)[1])                     # ic_masaFija_N2000_s{k}
        n_ar = _int(fila["n_aristas"])
        s_er = 1000 * N + k                                 # grafo_random_masa_fija_generar.py línea 82
        # verificación: la cabecera del IC guardado declara su seed_random
        ruta_ic = f"/Users/alexis/phantom_cs073/bateria_grafo_random_masa_fija/{rid.split('HIST-')[1]}/cosmogenesis_ic.txt"
        if os.path.exists(ruta_ic):
            with open(ruta_ic) as f:
                cab = f.readline()
            declarado = [t for t in cab.split() if t.startswith("seed_random=")]
            dec_ar = [t for t in cab.split() if t.startswith("n_aristas=")]
            if declarado and int(declarado[0].split("=")[1]) != s_er:
                raise ValueError(f"{rid}: seed_random declarado {declarado[0]} != derivado {s_er}")
            if dec_ar and int(dec_ar[0].split("=")[1]) != n_ar:
                raise ValueError(f"{rid}: n_aristas declarado {dec_ar[0]} != CSV {n_ar}")
        adj_d, _e, _r = generar_grafo_erdos_renyi(N, n_ar, seed=s_er)
        return [set(adj_d[i]) for i in range(N)], N, \
            f"generar_grafo_erdos_renyi(N={N}, aristas={n_ar}, seed={s_er}) [ER histórico]"

    raise ValueError(f"sin receta de reconstrucción para exp={exp} brazo={brazo} rule_id={rid}")


# =============================================================================================
# 2) MEDICIÓN — todas las variables de apiñamiento sobre un grafo ya construido
# =============================================================================================
def _p99(v):
    return float(np.percentile(np.asarray(v, dtype=float), 99)) if len(v) else float("nan")


def medir_todo(adj_nativo, N, seed_medicion):
    """Mide TODO lo pedido por F8-00 sobre un grafo. Reusa, sin tocarlas, las funciones ya validadas:
    `O3B.clustering` (conteo exacto de triángulos), `F703.medir_organizacion` y
    `F703.enumerar_triangulos` (apiñamiento), `F702.asortatividad_grados`, `DC.diam_gigante` /
    `DC.componentes` (medición oficial de diámetro), `O3B.pendiente_corregida_de_grafo`."""
    import cs090_diam_corregido as DC
    import cs090_fase6_o3b_rewiring as O3B
    import cs090_fase7_f702_escalera as F702
    import cs090_fase7_f703_organizacion as F703

    adj_lista = O3B._a_lista(adj_nativo, N)          # lista de sets (no canonicaliza el orden)
    adj_canon = O3B.canonicalizar(adj_lista, N)      # forma canónica: la que se recupera del disco

    out = {}
    m_ar = sum(len(adj_lista[i]) for i in range(N)) // 2
    out["n_aristas"] = m_ar
    out["grado_medio"] = 2.0 * m_ar / N

    comps = DC.componentes(adj_canon, N)
    tam_gig = max((len(c) for c in comps), default=0)
    out["giant"] = tam_gig / N
    out["n_componentes"] = len(comps)
    out["diam"] = float(DC.diam_gigante(adj_canon, N))
    out["asortatividad"] = float(F702.asortatividad_grados(adj_canon, N))

    # --- clustering / transitividad / nº de triángulos (DATO, no variable explicativa) ---
    c_loc, trans, n_tri = O3B.clustering(adj_lista, N)
    out["clustering"] = c_loc
    out["transitividad"] = trans
    out["n_triangulos"] = n_tri

    # --- apiñamiento: batería completa de F7-03 ---
    rng = np.random.default_rng(seed_medicion)
    org = F703.medir_organizacion(adj_lista, N, rng)
    for k in ("frac_nodos_en_triangulo", "frac_aristas_en_triangulo", "frac_aristas_multi_tri",
              "tri_por_arista_media", "tri_por_nodo_max", "gini_tri_nodo",
              "n_comp_tri", "tam_mayor_comp_tri", "frac_mayor_comp_tri", "tam_medio_comp_tri",
              "modularidad_tri", "dist_media_tri", "dist_media_azar"):
        out[k] = org.get(k, float("nan"))
    out["tri_ar_media_sop"] = out.pop("tri_por_arista_media")     # nombre explícito: sobre el soporte

    # --- apiñamiento: distribución completa de triángulos por arista (lo que F7-03 no daba) ---
    tris = F703.enumerar_triangulos(adj_lista, N)
    tri_ar = {}
    for (a, b, c) in tris:
        for e in ((a, b), (a, c), (b, c)):
            tri_ar[e] = tri_ar.get(e, 0) + 1
    v_sop = np.array(sorted(tri_ar.values()), dtype=float)        # sólo aristas con ≥1 triángulo
    out["tri_ar_mediana_sop"] = float(np.median(v_sop)) if len(v_sop) else float("nan")
    out["tri_ar_max"] = float(v_sop.max()) if len(v_sop) else 0.0
    out["tri_ar_p99_sop"] = _p99(v_sop)
    out["tri_ar_media_todas"] = (float(v_sop.sum()) / m_ar) if m_ar else float("nan")

    # --- cúmulos de triángulos por ARISTA compartida (el eje de solapamiento, complementa n_comp_tri
    #     de F7-03 que agrupa por NODO compartido) ---
    if tris:
        uf = G8_UF(len(tris))
        por_arista = {}
        for ti, (a, b, c) in enumerate(tris):
            for e in ((a, b), (a, c), (b, c)):
                por_arista.setdefault(e, []).append(ti)
        for e, lst in por_arista.items():
            for ti in lst[1:]:
                uf.union(lst[0], ti)
        raices = {}
        for ti in range(len(tris)):
            raices.setdefault(uf.find(ti), 0)
            raices[uf.find(ti)] += 1
        tam = sorted(raices.values(), reverse=True)
        out["n_comp_tri_arista"] = len(tam)
        out["frac_mayor_comp_tri_arista"] = tam[0] / len(tris)
    else:
        out["n_comp_tri_arista"] = float("nan")
        out["frac_mayor_comp_tri_arista"] = float("nan")

    return out, adj_lista, adj_canon


class G8_UF:
    """Union-find mínimo (misma idea que `_UF` de F7-03; se reimplementa acá sólo para no depender de
    un símbolo privado de otro módulo)."""

    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


# =============================================================================================
# 3) UNA FILA COMPLETA: reconstruir -> medir -> guardar -> releer y verificar el sello
# =============================================================================================
def procesar_fila(tarea):
    import cs090_fase6_o3b_rewiring as O3B

    fila, idx = tarea["fila"], tarea["idx"]
    t0 = time.time()
    res = dict(idx=idx, exp=fila["exp"], rule_id=str(fila["rule_id"]), brazo=str(fila.get("brazo") or ""),
               seed=_int(fila["seed"]), N=_int(fila["N_nodos"]))
    try:
        adj_nativo, N, receta = reconstruir_grafo(fila)
        t_rec = time.time() - t0

        seed_med = int((res["seed"] if res["seed"] is not None else abs(hash(res["rule_id"])) % 10**7)
                       * MULT_SEED_MEDICION + N)
        t1 = time.time()
        med, adj_lista, adj_canon = medir_todo(adj_nativo, N, seed_med)
        t_med = time.time() - t1

        # --- pendiente corregida: nativa (histórica) y canónica (la que se recupera del disco) ---
        t2 = time.time()
        seed_pend = res["seed"] if res["seed"] is not None else 1000 * N + int(res["rule_id"][-1] if
                                                                              res["rule_id"][-1].isdigit() else 1)
        nulls = O3B.nulls_topo_de(seed_pend, N, med["n_aristas"])
        pc_nat = O3B.pendiente_corregida_de_grafo(adj_nativo, N, seed_pend, nulls)
        pc_can = O3B.pendiente_corregida_de_grafo(adj_canon, N, seed_pend, nulls)
        med["pendiente_nativa"] = pc_nat["pendiente"]
        med["pendiente_canon"] = pc_can["pendiente"]
        t_pend = time.time() - t2

        # --- guardar el grafo y releerlo para comprobar el sello ---
        nombre = (f"{res['rule_id']}__s{res['seed']}__{res['brazo'] or 'unico'}__N{N}"
                  f"{G8.SUFIJO}")
        ruta = os.path.join(DIR_GRAFOS, str(fila["exp"]), nombre)
        sello = G8.hash_grafo(adj_lista, N)
        G8.guardar_grafo(adj_lista, ruta, N=N,
                         meta=dict(rule_id=res["rule_id"], seed=res["seed"], brazo=res["brazo"] or "unico",
                                   exp=res["exp"], receta=receta.replace(" ", "")))
        adj_leido, n_leido, meta_leido = G8.cargar_grafo(ruta)          # verifica el sello solo
        ida_vuelta = (n_leido == N) and all(adj_leido[i] == adj_canon[i] for i in range(N))

        res.update(med)
        res.update(ok=True, receta=receta, sha256=sello, archivo=os.path.relpath(ruta, HERE),
                   ida_vuelta_ok=bool(ida_vuelta), bytes=os.path.getsize(ruta),
                   t_reconstruir_s=round(t_rec, 2), t_medir_s=round(t_med, 2),
                   t_pendiente_s=round(t_pend, 2), t_total_s=round(time.time() - t0, 2))
    except Exception as e:                                              # noqa: BLE001
        res.update(ok=False, motivo=f"{type(e).__name__}: {e}",
                   traza=traceback.format_exc()[-600:], t_total_s=round(time.time() - t0, 2))
    return res


# =============================================================================================
# 4) ORQUESTACIÓN
# =============================================================================================
def cargar_cache():
    if not os.path.exists(RUTA_CACHE):
        return {}
    out = {}
    with open(RUTA_CACHE) as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            d = json.loads(linea)
            out[d["idx"]] = d
    return out


def main(limite=None, workers=6, solo_analisis=False):
    t_ini = time.time()
    log("=" * 90)
    log("F8-00 · regenerar, medir y GUARDAR los grafos de las 254 corridas del dataset unificado")
    log("=" * 90)

    D = pd.read_csv(RUTA_DATASET)
    log(f"\nDataset unificado: {len(D)} filas x {len(D.columns)} columnas")
    log("Composición (exp x brazo):")
    log(D.groupby(["exp", "brazo"], dropna=False).size().to_string())
    log(f"\nFilas con clustering YA medido en el dataset: {int(D['clustering'].notna().sum())}")

    filas = D.to_dict("records")
    cache = cargar_cache()
    log(f"Caché previa: {len(cache)} filas ya medidas")

    pendientes = [dict(fila=f, idx=i) for i, f in enumerate(filas) if i not in cache]
    if limite:
        pendientes = pendientes[:limite]
    if solo_analisis:
        pendientes = []

    if pendientes:
        log(f"\n--- Midiendo {len(pendientes)} filas ({workers} proceso(s)) ---")
        fh = open(RUTA_CACHE, "a")
        hechos = 0
        if workers <= 1:
            it = (procesar_fila(t) for t in pendientes)
        else:
            pool = mp.Pool(workers)
            it = pool.imap_unordered(procesar_fila, pendientes)
        for r in it:
            hechos += 1
            cache[r["idx"]] = r
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if r.get("ok"):
                log(f"  [{hechos:3d}/{len(pendientes)}] {r['exp']:16s} {r['rule_id'][:34]:34s} "
                    f"{r['brazo'][:12]:12s} aristas={r['n_aristas']:5d} tri={r['n_triangulos']:5d} "
                    f"gini={r['gini_tri_nodo']:.3f} tri/ar_max={r['tri_ar_max']:.0f} "
                    f"({r['t_total_s']:.1f}s)")
            else:
                log(f"  [{hechos:3d}/{len(pendientes)}] FALLA {r['exp']} {r['rule_id']} "
                    f"{r['brazo']}: {r.get('motivo')}")
        if workers > 1:
            pool.close(); pool.join()
        fh.close()

    # -----------------------------------------------------------------------------------------
    # 4a) armar el dataset enriquecido
    # -----------------------------------------------------------------------------------------
    COLS_MED = ["n_aristas", "grado_medio", "giant", "n_componentes", "diam", "asortatividad",
                "clustering", "transitividad", "n_triangulos",
                "tri_ar_media_sop", "tri_ar_mediana_sop", "tri_ar_max", "tri_ar_p99_sop",
                "tri_ar_media_todas", "frac_aristas_en_triangulo", "frac_aristas_multi_tri",
                "gini_tri_nodo", "tri_por_nodo_max", "frac_nodos_en_triangulo",
                "n_comp_tri", "tam_mayor_comp_tri", "frac_mayor_comp_tri", "tam_medio_comp_tri",
                "n_comp_tri_arista", "frac_mayor_comp_tri_arista",
                "modularidad_tri", "dist_media_tri", "dist_media_azar",
                "pendiente_nativa", "pendiente_canon"]

    E = D.copy()
    for c in COLS_MED:
        E["f800_" + c] = [cache.get(i, {}).get(c, np.nan) if cache.get(i, {}).get("ok") else np.nan
                          for i in range(len(E))]
    E["f800_ok"] = [bool(cache.get(i, {}).get("ok", False)) for i in range(len(E))]
    E["f800_motivo"] = [cache.get(i, {}).get("motivo", "") if not cache.get(i, {}).get("ok") else ""
                        for i in range(len(E))]
    E["f800_sha256"] = [cache.get(i, {}).get("sha256", "") for i in range(len(E))]
    E["f800_archivo"] = [cache.get(i, {}).get("archivo", "") for i in range(len(E))]
    E["f800_receta"] = [cache.get(i, {}).get("receta", "") for i in range(len(E))]
    E.to_csv(RUTA_ENRIQUECIDO, index=False)
    n_ok = int(E["f800_ok"].sum())
    log(f"\nESCRITO: {os.path.basename(RUTA_ENRIQUECIDO)}  ({len(E)} filas x {len(E.columns)} columnas)")
    log(f"Medidas con éxito: {n_ok} / {len(E)}")
    if n_ok < len(E):
        log("Filas NO medidas (con motivo):")
        for _, r in E[~E["f800_ok"]].iterrows():
            log(f"  {r['exp']:16s} {str(r['rule_id'])[:40]:40s} {str(r['brazo'])[:14]:14s} "
                f"-> {r['f800_motivo'][:110]}")

    # -----------------------------------------------------------------------------------------
    # 4b) VERIFICACIONES contra lo histórico
    # -----------------------------------------------------------------------------------------
    log("\n" + "=" * 90)
    log("VERIFICACIÓN 1 — las 24 filas de O3-B que YA tenían clustering medido")
    log("=" * 90)
    v = E[E["clustering"].notna() & E["f800_ok"]].copy()
    filas_v = []
    for _, r in v.iterrows():
        d_c = abs(r["clustering"] - r["f800_clustering"])
        d_t = abs(r["transitividad"] - r["f800_transitividad"])
        d_n = abs(r["n_triangulos"] - r["f800_n_triangulos"])
        filas_v.append(dict(rule_id=r["rule_id"], seed=r["seed"], brazo=r["brazo"],
                            clustering_hist=r["clustering"], clustering_f800=r["f800_clustering"],
                            dif_clustering=d_c,
                            transitividad_hist=r["transitividad"], transitividad_f800=r["f800_transitividad"],
                            dif_transitividad=d_t,
                            n_tri_hist=r["n_triangulos"], n_tri_f800=r["f800_n_triangulos"],
                            dif_n_tri=d_n,
                            pend_hist=r["pendiente"], pend_nativa=r["f800_pendiente_nativa"],
                            pend_canon=r["f800_pendiente_canon"]))
        log(f"  {str(r['rule_id'])[:26]:26s} {str(r['brazo']):7s} "
            f"clus {r['clustering']:.10f} vs {r['f800_clustering']:.10f} (Δ={d_c:.2e})  "
            f"trans Δ={d_t:.2e}  tri {int(r['n_triangulos'])} vs {int(r['f800_n_triangulos'])}")
    if filas_v:
        V = pd.DataFrame(filas_v)
        V.to_csv(RUTA_VERIFICACION, index=False)
        log(f"\n  n verificadas = {len(V)}")
        log(f"  max |Δ clustering|    = {V.dif_clustering.max():.3e}")
        log(f"  max |Δ transitividad| = {V.dif_transitividad.max():.3e}")
        log(f"  max |Δ nº triángulos| = {V.dif_n_tri.max():.0f}")
        exacto = (V.dif_clustering.max() < 1e-12 and V.dif_transitividad.max() < 1e-12
                  and V.dif_n_tri.max() == 0)
        log(f"  -> COINCIDENCIA EXACTA (tol 1e-12): {exacto}")
        pv = V.dropna(subset=["pend_hist"])
        if len(pv):
            log(f"  pendiente: max |Δ nativa-histórica| = {(pv.pend_hist - pv.pend_nativa).abs().max():.3e} "
                f"· max |Δ canónica-histórica| = {(pv.pend_hist - pv.pend_canon).abs().max():.3e}")
        log(f"  ESCRITO: {os.path.basename(RUTA_VERIFICACION)}")

    log("\n" + "=" * 90)
    log("VERIFICACIÓN 2 — nº de aristas reconstruido vs. histórico (las 254)")
    log("=" * 90)
    a = E[E["f800_ok"] & E["n_aristas"].notna()].copy()
    a["_d"] = (a["n_aristas"] - a["f800_n_aristas"]).abs()
    log(f"  filas comparadas: {len(a)} · coinciden exacto: {int((a['_d'] < 1e-9).sum())} · "
        f"max |Δ| = {a['_d'].max():.0f}")
    malas = a[a["_d"] >= 1e-9]
    for _, r in malas.iterrows():
        log(f"    DISCREPA {r['exp']:16s} {str(r['rule_id'])[:32]:32s} {str(r['brazo']):12s} "
            f"hist={r['n_aristas']:.0f} f800={r['f800_n_aristas']:.0f}")

    log("\n" + "=" * 90)
    log("VERIFICACIÓN 3 — pendiente corregida reconstruida vs. histórica")
    log("=" * 90)
    q = E[E["f800_ok"] & E["pendiente"].notna()].copy()
    q["_dn"] = (q["pendiente"] - q["f800_pendiente_nativa"]).abs()
    q["_dc"] = (q["pendiente"] - q["f800_pendiente_canon"]).abs()
    log(f"  filas con pendiente histórica: {len(q)}")
    log(f"  nativa   : idénticas (<1e-12) {int((q['_dn'] < 1e-12).sum())} · "
        f"|Δ|<0.01 {int((q['_dn'] < 0.01).sum())} · max |Δ| {q['_dn'].max():.4f} · "
        f"mediana |Δ| {q['_dn'].median():.2e}")
    log(f"  canónica : idénticas (<1e-12) {int((q['_dc'] < 1e-12).sum())} · "
        f"|Δ|<0.01 {int((q['_dc'] < 0.01).sum())} · max |Δ| {q['_dc'].max():.4f} · "
        f"mediana |Δ| {q['_dc'].median():.2e}")
    for e_, g in q.groupby("exp"):
        log(f"    {e_:18s} n={len(g):3d}  nativa: idénticas {int((g['_dn'] < 1e-12).sum()):3d}, "
            f"max|Δ| {g['_dn'].max():.4f}")

    log("\n" + "=" * 90)
    log("VERIFICACIÓN 4 — sello sha256 e ida y vuelta a disco")
    log("=" * 90)
    iv = [cache[i].get("ida_vuelta_ok") for i in range(len(E)) if cache.get(i, {}).get("ok")]
    tot_bytes = sum(cache[i].get("bytes", 0) for i in range(len(E)) if cache.get(i, {}).get("ok"))
    log(f"  grafos guardados: {len(iv)} · releídos idénticos arista por arista: {sum(bool(x) for x in iv)}")
    log(f"  sellos únicos: {len(set(E.loc[E['f800_ok'], 'f800_sha256']))} de {n_ok} "
        f"(sellos repetidos = grafos genuinamente idénticos entre filas)")
    log(f"  espacio total en disco: {tot_bytes / 1024:.0f} KB "
        f"({tot_bytes / max(1, len(iv)):.0f} bytes por grafo)")

    # -----------------------------------------------------------------------------------------
    # 4c) MATRIZ DE CORRELACIONES entre medidas de apiñamiento (insumo directo de F8-01)
    # -----------------------------------------------------------------------------------------
    log("\n" + "=" * 90)
    log("MATRIZ DE CORRELACIONES entre las medidas de apiñamiento (n = filas medidas)")
    log("=" * 90)
    VARS = ["f800_tri_ar_media_sop", "f800_tri_ar_mediana_sop", "f800_tri_ar_max", "f800_tri_ar_p99_sop",
            "f800_tri_ar_media_todas", "f800_frac_aristas_en_triangulo", "f800_frac_aristas_multi_tri",
            "f800_gini_tri_nodo", "f800_tri_por_nodo_max", "f800_frac_nodos_en_triangulo",
            "f800_n_triangulos", "f800_transitividad", "f800_clustering",
            "f800_n_comp_tri", "f800_n_comp_tri_arista", "f800_frac_mayor_comp_tri",
            "f800_tam_medio_comp_tri", "f800_modularidad_tri", "f800_dist_media_tri",
            "f800_n_aristas", "f800_grado_medio", "f800_giant", "f800_asortatividad",
            "f800_pendiente_nativa", "f800_diam"]
    VARS = [v for v in VARS if v in E.columns]
    M = E.loc[E["f800_ok"], VARS]

    rs = M.corr(method="spearman")
    rp = M.corr(method="pearson")
    salida = []
    for i, a_ in enumerate(VARS):
        for b_ in VARS[i + 1:]:
            sub = M[[a_, b_]].dropna()
            salida.append(dict(var_a=a_.replace("f800_", ""), var_b=b_.replace("f800_", ""),
                               n=len(sub), spearman=rs.loc[a_, b_], pearson=rp.loc[a_, b_],
                               abs_spearman=abs(rs.loc[a_, b_])))
    C = pd.DataFrame(salida).sort_values("abs_spearman", ascending=False)
    C.to_csv(RUTA_CORRELACIONES, index=False)
    log(f"  n usado: {len(M)} filas")
    log(f"  ESCRITO (pares, ordenados por |Spearman|): {os.path.basename(RUTA_CORRELACIONES)}")

    APIN = [v for v in VARS if v.replace("f800_", "") in
            ("tri_ar_media_sop", "tri_ar_mediana_sop", "tri_ar_max", "tri_ar_p99_sop",
             "tri_ar_media_todas", "frac_aristas_en_triangulo", "frac_aristas_multi_tri",
             "gini_tri_nodo", "tri_por_nodo_max", "n_triangulos", "transitividad", "clustering")]
    log("\n  Spearman entre las medidas de apiñamiento nucleares:")
    sub = rs.loc[APIN, APIN].round(3)
    sub.index = [i.replace("f800_", "") for i in sub.index]
    sub.columns = [c.replace("f800_", "")[:14] for c in sub.columns]
    log(sub.to_string())

    log("\n  Pares MÁS colineales (|Spearman| >= 0.90):")
    for _, r in C[C.abs_spearman >= 0.90].iterrows():
        log(f"    {r.var_a:28s} ~ {r.var_b:28s} rho={r.spearman:+.3f}  (Pearson {r.pearson:+.3f}, n={r.n})")

    log("\n  Pares MENOS acoplados entre medidas de apiñamiento (|Spearman| <= 0.35):")
    nucleo = set(v.replace("f800_", "") for v in APIN)
    for _, r in C[(C.abs_spearman <= 0.35) & C.var_a.isin(nucleo) & C.var_b.isin(nucleo)].iterrows():
        log(f"    {r.var_a:28s} ~ {r.var_b:28s} rho={r.spearman:+.3f}  (n={r.n})")

    log(f"\nTiempo total: {time.time() - t_ini:.0f}s")
    with open(RUTA_LOG, "w") as f:
        f.write("\n".join(_log) + "\n")
    print(f"\nlog -> {RUTA_LOG}")


if __name__ == "__main__":
    limite, workers, solo = None, 6, False
    for arg in sys.argv[1:]:
        if arg.startswith("--limite"):
            limite = int(arg.split("=")[1]) if "=" in arg else None
        elif arg.startswith("--workers"):
            workers = int(arg.split("=")[1]) if "=" in arg else 6
        elif arg == "--solo-analisis":
            solo = True
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--limite" and i + 1 < len(args):
            limite = int(args[i + 1])
        if a == "--workers" and i + 1 < len(args):
            workers = int(args[i + 1])
    main(limite=limite, workers=workers, solo_analisis=solo)

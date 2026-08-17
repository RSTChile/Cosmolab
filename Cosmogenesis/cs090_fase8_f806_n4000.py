#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
cs090_fase8_f806_n4000.py — FASE VIII, tarea F8-06: el contraste `solap` vs `disj` de F7-03,
                            REGENERADO A N=4000 (y sólo a N=4000; N=8000 quedó descartado)
=================================================================================================

QUÉ PREGUNTA CONTESTA (a nivel módulo)
--------------------------------------
`FASE7_F703_grados_y_triangulos_fijos_CS.md` midió, a N=2000, que con la secuencia de grados clavada
NODO POR NODO y el NÚMERO DE TRIÁNGULOS igualado, apilar los triángulos sobre las mismas aristas
(`solap`) junta más masa en sumideros que repartirlos sin que se toquen (`disj`):
**+0.01433 de fracción de masa (+13.8 %), 12/12 grafos, Wilcoxon p = 4.9e-04**.

La pregunta de esta tarea es una sola: **¿ese +13.8 % se mantiene, crece o se diluye al duplicar la
resolución?** Nada más. No se cambia el observable, no se cambia el protocolo de Phantom, no se
cambia el generador de grafos.

POR QUÉ N=4000 Y NO N=8000
--------------------------
`FASE8_F804_grano_n8000_CS.md` midió dos cosas que cierran la puerta de N=8000 con el protocolo actual:
  1. **0 de 14 corridas a N=8000 llegan a `tmax=0.500`** — el costo por dump se duplica cada ~6 dumps
     (colapso del paso de tiempo). El endpoint de la serie *no existe* a esa resolución.
  2. **El grano del instrumento no lo pone N, lo ponen los sumideros**: σ ∝ (nº de sumideros)^1.24.
     A N=8000 nacen ~15× más sumideros que a N=2000, así que subir la resolución EMPEORA la medición.
A N=4000, en cambio, 8/8 corridas de O3-A llegaron completas (18-57 s) con ~29 sumideros, y el ajuste
de F8-04 predice σ ≈ 0.0027 — contra el cual el efecto de F7-03 (0.0143) son **5.3 σ**.

LA REGLA DURA DE LA FASE, Y CÓMO SE CUMPLE ACÁ: **NO MEZCLAR LAYOUTS**
----------------------------------------------------------------------
El sesgo que introduce el layout Barnes-Hut con θ=0.3 (+0.0025 a +0.0071 de fracción de masa) es MAYOR
que los residuales que persigue esta línea, así que un punto medido con la suma N² no se puede comparar
con uno medido con Barnes-Hut.

**Este script usa `layout_resortes` — la suma exacta O(N²) — en LOS DOS BRAZOS**, sin tocar el
adaptador: `generar_ic_masa_fija_desde_grafo` lo llama tal cual, no se le monkeypatchea el layout.
Eso da tres comparaciones limpias a la vez:
  * `solap` contra `disj` dentro de cada par (la regla literal de la fase);
  * N=4000 contra los puntos de N=2000 de F7-03, que también son N²;
  * N=4000 contra los puntos de N=4000 de O3-A, que también son N² (`cs090_fase6_o3a_convergencia_resolucion.py`).
θ no aplica: no hay árbol, no hay criterio de apertura.

El costo se midió en esta máquina antes de decidir (una evaluación de fuerza a N=4000, ×100 iteraciones):
  suma N² = 233 s/layout · Barnes-Hut θ=0.3 = 110 s/layout · Barnes-Hut θ=0.5 = 40 s/layout.
Barnes-Hut θ=0.3 es ~2× más rápido, pero **elegir velocidad acá costaría la comparación con N=2000**,
que es justamente la pregunta de la tarea. Con 24 layouts en paralelo el N² entra en presupuesto, así
que se paga el precio y se conserva la comparabilidad. (El número queda registrado en el informe.)

QUÉ SE REUSA Y QUÉ NO SE TOCA
------------------------------
`cs090_fase7_f703_organizacion.py` se **importa tal cual** y hace TODO el trabajo pesado: piso común de
triángulos destruidos, motor de swaps dirigidos que conserva el grado de los cuatro nodos, igualación
del nº de triángulos por rebobinado, verificación `np.array_equal` de la secuencia de grados, batería de
medidas de organización, escritura de la condición inicial y del `meta_regla.json`.

Este script sólo le cambia, DESDE AFUERA (atributos de módulo, ni una línea del archivo original):

  | atributo             | F7-03 (N=2000)                  | acá (F8-06)                              |
  |----------------------|---------------------------------|------------------------------------------|
  | `N_NODOS`            | 2000                            | **4000**                                 |
  | `BRAZOS`             | libre, conc, disp, solap, disj  | **solap, disj** (el contraste a replicar) |
  | `BASE_SALIDA`        | bateria_fase7_f703_organizacion | **bateria_fase8_f806_n4000**             |
  | `RUTA_ESTRUCTURA`    | cs090_fase7_f703_estructura.csv | cs090_fase8_f806_estructura.csv          |

**Las dos consecuencias de recortar `BRAZOS`, dichas de frente:**
  a) `T*` (el nº de triángulos común) pasa a ser el mínimo de DOS techos en vez de cinco. En F7-03 el
     techo mínimo lo puso `solap` en los 12 grafos (apilar triángulos sobre la misma arista se agota
     rápido con grados ≤8), así que se espera el mismo `T*`; el script reporta los dos techos por grafo
     para que se vea si eso se cumplió también a N=4000.
  b) Las semillas de los brazos se derivan del ÍNDICE dentro de `BRAZOS` (`semilla_base + 101*(j+1)`),
     así que `solap` y `disj` reciben acá los flujos 1 y 2 en vez de los 4 y 5. No es una repetición
     con la misma semilla — es una realización independiente del mismo procedimiento. A N=4000 el grafo
     base ya es otro de todas formas (el motor depende de N), así que la igualdad de semillas no era
     alcanzable ni deseable.

LO QUE ESTE SCRIPT AGREGA POR SU CUENTA
----------------------------------------
1. **Guarda el grafo de cada brazo** (`grafo_final.grafo.gz`, módulo `cs090_fase8_f800_grafos`) y anota
   su `sha256` en el `meta_regla.json` — regla vigente de la Fase VIII. Lo hace envolviendo la función
   de condición inicial, que es la única que recibe el grafo ya listo.
2. **Cronometra el layout** de cada brazo por separado (el paso caro) y lo deja en el CSV.
3. **Escribe `meta_f806.json`** en cada carpeta con el layout usado (`layout_resortes`, N², `theta=None`)
   y con qué brazo es — para que la verificación cruzada posterior no dependa de recordar nada.

USO
---
    python3.9 cs090_fase8_f806_n4000.py costo            # mide 1 grafo entero y reporta el costo
    python3.9 cs090_fase8_f806_n4000.py 0,3,6 --sufijo=_shard0
    python3.9 cs090_fase8_f806_n4000.py --todos --sufijo=_shardX

No corre Phantom (eso es `cs090_fase8_f806_correr.py`). No declara cierre ni veredicto. Sin commits.
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)

import cs090_fase7_f703_organizacion as F703      # SÓLO import: el generador, sin modificar
import cs090_fase8_f800_grafos as G8              # SÓLO import: persistencia de grafos (F8-00)

# ---------------------------------------------------------------------------------------------
# Parámetros de ESTA tarea (se aplican como atributos del módulo importado, no editando el archivo)
# ---------------------------------------------------------------------------------------------
N_OBJETIVO = 4000
BRAZOS_F806 = ("solap", "disj")
BASE_SALIDA_F806 = "/Users/alexis/phantom_cs073/bateria_fase8_f806_n4000"
RUTA_ESTRUCTURA_F806 = f"{HERE}/cs090_fase8_f806_estructura.csv"

LAYOUT_USADO = "layout_resortes_N2_exacto"   # el layout original O(N²) del proyecto
THETA_USADO = None                            # no aplica: no hay árbol de Barnes-Hut

F703.N_NODOS = N_OBJETIVO
F703.BRAZOS = BRAZOS_F806
F703.BASE_SALIDA = BASE_SALIDA_F806
F703.RUTA_ESTRUCTURA = RUTA_ESTRUCTURA_F806


# =============================================================================================
# 1) Envoltura de la condición inicial: guarda el grafo y cronometra el layout
# =============================================================================================
_ic_original = F703.generar_ic_masa_fija_desde_grafo
_ultimos_tiempos = {}          # ruta de carpeta -> segundos de layout+IC


def _ic_guardando_grafo(adj_lista, N, seed_layout, ruta_salida, **kw):
    """Misma llamada de siempre, con dos cosas alrededor:

    (a) antes de generar nada se escribe el grafo del brazo en `grafo_final.grafo.gz` con su sello
        sha256 — así el grafo de esta corrida existe en disco aunque nadie lo vuelva a generar;
    (b) se cronometra la generación de la condición inicial, que a N=4000 es ~el 95 % del costo
        (la suma N² del layout).

    No se toca ni un parámetro de la receta: `adj_lista`, `N`, `seed_layout`, `ruta_salida` y los
    kwargs pasan tal cual a la función original del adaptador congelado."""
    carpeta = os.path.dirname(ruta_salida)
    os.makedirs(carpeta, exist_ok=True)
    G8.guardar_grafo(adj_lista, os.path.join(carpeta, "grafo_final" + G8.SUFIJO), N=N,
                     meta=dict(tarea="FASE8_F806_f703_a_N4000", N=N, seed_layout=seed_layout))
    t0 = time.time()
    info = _ic_original(adj_lista, N=N, seed_layout=seed_layout, ruta_salida=ruta_salida, **kw)
    _ultimos_tiempos[carpeta] = round(time.time() - t0, 2)
    info["t_layout_ic_s"] = _ultimos_tiempos[carpeta]
    return info


F703.generar_ic_masa_fija_desde_grafo = _ic_guardando_grafo


# =============================================================================================
# 2) Post-proceso por carpeta: sello del grafo en el meta + meta propio de F8-06
# =============================================================================================
def _sellar_carpeta(fila):
    """Anota en el `meta_regla.json` que escribió F7-03 el sha256 del grafo guardado, y deja un
    `meta_f806.json` con lo que esta tarea necesita verificar después (layout, θ, brazo, N)."""
    carpeta = fila.get("carpeta")
    if not carpeta or not os.path.isdir(carpeta):
        return
    ruta_grafo = os.path.join(carpeta, "grafo_final" + G8.SUFIJO)
    if not os.path.exists(ruta_grafo):
        return
    adj, n, meta_g = G8.cargar_grafo(ruta_grafo)          # verifica el sello al leer
    G8.anotar_hash_en_meta(os.path.join(carpeta, "meta_regla.json"), adj, n,
                           archivo_grafo="grafo_final" + G8.SUFIJO)
    with open(os.path.join(carpeta, "meta_f806.json"), "w") as f:
        json.dump(dict(
            tarea="FASE8_F806_f703_a_N4000", brazo=fila["brazo"], rule_id=fila["rule_id"],
            seed=fila["seed"], lote=fila["lote"], N=N_OBJETIVO,
            layout=LAYOUT_USADO, theta=THETA_USADO, seed_layout=F703.SEED_LAYOUT,
            iters_layout=100, brazos_de_esta_tanda=list(BRAZOS_F806),
            n_aristas=fila["n_aristas"], n_triangulos=fila["n_triangulos"],
            T_objetivo=fila["T_objetivo"], dif_max_triangulos=fila["dif_max_triangulos"],
            grados_identicos=bool(fila["grados_identicos"]),
            grafo_sha256=meta_g["sha256"], grafo_archivo="grafo_final" + G8.SUFIJO,
            t_layout_ic_s=_ultimos_tiempos.get(carpeta),
        ), f, indent=2)


# =============================================================================================
# 3) Driver
# =============================================================================================
def correr(indices, sufijo):
    import csv
    elegidos = F703.F702.seleccionar_grafos(n_por_lote=3)      # LOS MISMOS 12 grafos base de F7-03
    if indices is not None:
        elegidos = [elegidos[i] for i in indices]
    print(f"[f806] N={F703.N_NODOS} brazos={F703.BRAZOS} layout={LAYOUT_USADO} theta={THETA_USADO}",
          flush=True)
    print(f"[f806] {len(elegidos)} grafos: {[e['rule_id'] for e in elegidos]}", flush=True)

    todas, t0 = [], time.time()
    for k, sel in enumerate(elegidos):
        t_g = time.time()
        print(f"[{k+1}/{len(elegidos)}] {sel['rule_id']} seed={sel['seed']} lote={sel['lote']}",
              flush=True)
        filas = F703.procesar_una(sel, generar_ic=True)
        for fila in filas:
            fila["N_objetivo"] = N_OBJETIVO
            fila["layout"] = LAYOUT_USADO
            fila["theta"] = THETA_USADO
            fila["t_layout_ic_s"] = _ultimos_tiempos.get(fila.get("carpeta"))
            _sellar_carpeta(fila)
        todas.extend(filas)
        print(f"    [grafo listo en {time.time()-t_g:.0f}s]", flush=True)

    ruta = RUTA_ESTRUCTURA_F806.replace(".csv", f"{sufijo}.csv")
    with open(ruta, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(todas[0].keys()))
        w.writeheader()
        w.writerows(todas)
    print(f"\n[f806] {len(todas)} filas -> {os.path.basename(ruta)}  (total {time.time()-t0:.0f}s)",
          flush=True)
    return todas


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "costo":
        # mide UN grafo entero antes de comprometer la batería (lo que pide la consigna)
        correr([0], "_costo")
        sys.exit(0)
    idxs, suf = None, ""
    for arg in args:
        if arg == "--todos":
            idxs = None
        elif arg.startswith("--sufijo="):
            suf = arg.split("=", 1)[1]
        else:
            idxs = [int(x) for x in arg.split(",")]
    correr(idxs, suf)

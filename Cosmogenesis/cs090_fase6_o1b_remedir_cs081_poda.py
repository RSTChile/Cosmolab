"""
cs090_fase6_o1b_remedir_cs081_poda.py — ¿el bug de `_diam` toca a FASE III Experimento 2 (poda, cs081)?
=======================================================================================================

QUÉ ES ESTO (tarea O1-B, Parte 1)
---------------------------------
La auditoría de adopción del diámetro corregido (`FASE6_adopcion_diam_corregido_CS.md`) re-midió
Fase III **Experimento 1** (`cs080_renormalizacion.py`, 0/54 descarrilamientos) y toda la Fase V, pero
dejó fuera Fase III **Experimento 2** (`cs081_poda_dinamica.py`, la poda por costo de enlace). cs081 es,
junto a cs080, el ÚNICO script del rango cs076-cs089 que llama a `C7._diam` — la función congelada de
cs055 con el bug del arranque en fragmento suelto. Y la conclusión publicada de ese experimento
(`FASE3_poda_dinamica_resultado_CS.md`: costo_P50 = 0.786 vs azar_P50 = 0.655 vs sin_poda = 0.421) ES una pendiente
log(diám) vs log(N_cajas) — exactamente el tipo de número que el bug puede torcer.

EL BUG, EN UNA LÍNEA
--------------------
`_diam(adj,N)` arranca el doble-BFS en `next(i for i in range(N) if adj[i])` — el primer nodo POR ÍNDICE
con aristas. Si ese nodo cayó en un pedacito suelto (típico: un par de 2 nodos), mide el diámetro DEL
PEDACITO. Analogía: apoyar el metro en el buzón de la vereda para medir la altura del edificio.

QUÉ HACE ESTE SCRIPT
--------------------
Reconstruye la cadena EXACTA de `cs081_poda_dinamica.corre_semilla` (mismo motor
`proceso066_instrumentado`, mismo `costo_por_arista`, mismas variantes sin_poda / costo_P{50,70,90} /
azar_P{50,70,90}, mismas escalas b=1,2,4,8,16,32 con `cajas_bfs`/`grafo_grueso` de cs080) y en CADA
escala mide el diámetro DE LAS DOS MANERAS:

  - `diam_viejo`      = `cs055._diam` tal cual (extraído por AST en `cs090_diam_corregido`)
  - `diam_corregido`  = mismo doble-BFS pero arrancando en la componente conexa MÁS GRANDE

y guarda el diagnóstico que decide la cuestión: tamaño de la componente donde cayó la medición vieja
frente al tamaño de la gigante. Después recalcula la pendiente log-log con las dos varas.

NO se llama a `metricas_escala` completa (que además calcula d_s por crecimiento de bola y la holonomía
de Burgers, lo caro): esta auditoría pregunta SÓLO por el diámetro, y d_s/holonomía no usan `_diam`.

LIMITACIÓN DECLARADA (misma que `cs090_fase6_remedir_fase3.py`, no se disimula)
-------------------------------------------------------------------------------
cs081 deriva semillas de rng con `hash(str)` (p.ej. `RNG(seed*991 + hash(nombre) % 7919)`). El hash de
strings en Python está aleatorizado por proceso salvo `PYTHONHASHSEED` fijo, y el valor de la corrida
histórica no quedó registrado. Este script fija `PYTHONHASHSEED=0` para ser reproducible de acá en
adelante; sus grafos son del MISMO tipo pero no necesariamente la misma realización que los del CSV
histórico `cs081_poda_dinamica.csv`. Eso no afecta la pregunta que se hace ("¿este tipo de sustrato se
fragmenta al punto de descarrilar la medición?", que es propiedad del tipo de sustrato, no de una
realización), pero SÍ impide comparar diámetro contra diámetro con el CSV histórico — por eso no se hace
esa comparación acá, y en cambio el detector barato se aplica por separado, directo sobre el CSV
histórico, en `cs090_fase6_o1b_auditoria_diam_fases_restantes.py`.

No toca ningún script congelado. No corre Phantom. No declara cierre ni veredicto: reporta números.
"""
from __future__ import annotations
import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)   # re-arranca con hash determinista

import csv
import time

import numpy as np

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)

import cs080_renormalizacion as C80          # cajas_bfs, grafo_grueso -- SIN tocar
import cs081_poda_dinamica as C81            # proceso066_instrumentado, costo_por_arista, podar_* -- SIN tocar
import cs064_sistema_completo as C64         # _cataloga -- SIN tocar
import cs090_diam_corregido as DC            # diam_original / diam_gigante / diagnostico

RNG = np.random.default_rng

# Config: EXACTAMENTE la de cs081 (mismas semillas, escalas y percentiles).
SEMILLAS    = tuple(C81.SEEDS)          # (80100, 80200, 80300)
ESCALAS_B   = tuple(C81.ESCALAS_B)      # (2,4,8,16,32) -- b=1 se mide aparte, como en cs081
PERCENTILES = tuple(C81.PERCENTILES)    # (50,70,90)
N           = C81.N_NODOS               # 8000
K_LOCAL     = C81.K_LOCAL               # 6

OUT_CSV = os.path.join(HERE, "cs090_fase6_o1b_remedicion_cs081.csv")

CAMPOS = ["seed", "variante", "n_podadas", "b", "n_cajas",
          "diam_viejo", "diam_corregido", "tam_comp_medida", "tam_gigante",
          "n_componentes", "n_aislados", "descarrila"]


def pendiente_loglog(n_cajas, diams):
    """Pendiente del ajuste lineal de log(diám) vs log(nº de cajas) — la métrica de veredicto de
    Fase III (Exp.1 y Exp.2). Se descartan puntos con diám<=0 (log indefinido)."""
    x, y = [], []
    for nc, d in zip(n_cajas, diams):
        if nc > 0 and d > 0:
            x.append(np.log(nc)); y.append(np.log(d))
    if len(x) < 2:
        return float("nan")
    return float(np.polyfit(np.asarray(x), np.asarray(y), 1)[0])


def main():
    print("=" * 100, flush=True)
    print("O1-B / Parte 1 — re-medición del diámetro (viejo vs corregido) en FASE III Exp.2 (cs081, poda)", flush=True)
    print(f"N={N}  k_local={K_LOCAL}  semillas={SEMILLAS}  escalas b=1,{ESCALAS_B}  percentiles P={PERCENTILES}",
          flush=True)
    print("=" * 100, flush=True)

    t0 = time.time()
    filas = []

    for seed in SEMILLAS:
        print(f"\n--- semilla {seed} ---", flush=True)
        # === cadena idéntica a C81.corre_semilla ===
        rng = RNG(seed)
        cat = C64._cataloga(N, rng)
        r2 = RNG(seed * 137 + hash("local") % 9973 + 5)
        ts = time.time()
        adj, V, flip_count = C81.proceso066_instrumentado(N, cat, K_LOCAL, r2)
        print(f"  sustrato local construido: aristas={sum(len(a) for a in adj)//2} ({time.time()-ts:.1f}s)",
              flush=True)

        rng_costo = RNG(seed * 911 + 3)
        edges, costo, _ = C81.costo_por_arista(adj, N, V, flip_count, K_LOCAL, rng_costo)

        variantes = {"sin_poda": (adj, 0)}
        for P in PERCENTILES:
            na, n_pod = C81.podar_por_costo(adj, N, edges, costo, P)
            variantes[f"costo_P{P}"] = (na, n_pod)
            rng_rand = RNG(seed * 733 + P)
            variantes[f"azar_P{P}"] = (C81.podar_aleatorio(adj, N, edges, n_pod, rng_rand), n_pod)

        for nombre, (adj_v, n_podadas) in variantes.items():
            tb = time.time()
            nc_serie, dv_serie, dc_serie = [], [], []

            # b = 1 (nativo, sin agrupar)
            dg = DC.diagnostico(adj_v, N)
            filas.append(dict(seed=seed, variante=nombre, n_podadas=n_podadas, b=1, n_cajas=N,
                              diam_viejo=dg["diam_orig"], diam_corregido=dg["diam_corr"],
                              tam_comp_medida=dg["tam_comp_medida"], tam_gigante=dg["tam_gigante"],
                              n_componentes=dg["n_componentes"], n_aislados=dg["n_aislados"],
                              descarrila=dg["descarrila"]))
            nc_serie.append(N); dv_serie.append(dg["diam_orig"]); dc_serie.append(dg["diam_corr"])

            # escalas agrupadas (MISMA derivación de semillas que cs081)
            for b in ESCALAS_B:
                rng_b = RNG(seed * 733 + b * 31 + hash(nombre) % 4999)
                asign, n_cajas = C80.cajas_bfs(adj_v, N, b, rng_b)
                adj_g = C80.grafo_grueso(adj_v, N, asign, n_cajas)
                dgb = DC.diagnostico(adj_g, n_cajas)
                filas.append(dict(seed=seed, variante=nombre, n_podadas=n_podadas, b=b, n_cajas=n_cajas,
                                  diam_viejo=dgb["diam_orig"], diam_corregido=dgb["diam_corr"],
                                  tam_comp_medida=dgb["tam_comp_medida"], tam_gigante=dgb["tam_gigante"],
                                  n_componentes=dgb["n_componentes"], n_aislados=dgb["n_aislados"],
                                  descarrila=dgb["descarrila"]))
                nc_serie.append(n_cajas); dv_serie.append(dgb["diam_orig"]); dc_serie.append(dgb["diam_corr"])

            pv = pendiente_loglog(nc_serie, dv_serie)
            pc = pendiente_loglog(nc_serie, dc_serie)
            print(f"  [{nombre:<10}] podadas={n_podadas:<6} pend VIEJA={pv:+.4f}  CORREGIDA={pc:+.4f}  "
                  f"(dif {pc-pv:+.4f})  diám viejo={dv_serie} corr={dc_serie}  ({time.time()-tb:.1f}s)",
                  flush=True)

    with open(OUT_CSV, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=CAMPOS)
        wr.writeheader()
        for fila in filas:
            wr.writerow(fila)

    n_desc = sum(1 for f in filas if f["descarrila"])
    n_dif = sum(1 for f in filas if f["diam_viejo"] != f["diam_corregido"])
    print(f"\n[csv] {OUT_CSV} ({len(filas)} filas, {time.time()-t0:.0f}s)", flush=True)
    print(f"[resultado] escalas donde la medición vieja DESCARRILA: {n_desc}/{len(filas)}", flush=True)
    print(f"[resultado] escalas donde viejo != corregido:           {n_dif}/{len(filas)}", flush=True)


if __name__ == "__main__":
    main()

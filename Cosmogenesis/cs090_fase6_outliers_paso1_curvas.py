"""
cs090_fase6_outliers_paso1_curvas.py -- FASE VI, investigacion de los "casos raros" (PASO 1).

QUIEN SOY: reconstruyo el grafo final de un punado de reglas A2-B0-C2 y vuelvo a medir su curva de
coarse-graining ENTERA (los 5 puntos crudos b=1,2,4,8,16), en vez de mirar solo la pendiente ajustada.

Por que hace falta: la "pendiente" que clasifica cada regla es UN ajuste lineal sobre 5 puntos
(log(diametro) contra log(nº de cajas)). Una pendiente muy negativa puede venir de dos cosas muy
distintas:
   (a) una curva genuinamente decreciente y monotona -- el diametro CRECE al agrupar nodos, y
   (b) una curva NO-MONOTONA (sube y baja) donde el ajuste lineal es puro artefacto del promedio.
Solo mirando los 5 puntos crudos se distinguen. El proyecto ya vio al menos un caso del tipo (b)
(regla `r16`, "intermedio, pendiente=-1.137", FASE5_matriz_2x2_completa_CS.md).

Ademas mido tres cosas que la pendiente NO reporta y que pueden explicar diametros raros:
   - `giant`   : fraccion de nodos en la componente conexa mas grande (ya lo mide el motor).
   - `n_comp`  : cuantas componentes conexas (no aisladas) tiene el grafo en esa escala.
   - `tam_comp_fuente`: TAMANO de la componente donde efectivamente cae la medicion de diametro.
     Esto importa mucho: `_diam(adj,N)` de cs055 (congelado) arranca su doble-BFS en el PRIMER nodo
     no-aislado (`src = next(i for i in range(N) if adj[i])`), asi que en un grafo fragmentado NO mide
     el diametro del grafo entero sino el de la componente donde cae ese nodo -- y esa componente puede
     ser distinta en cada escala, porque el indice 0 despues de agrupar es otra caja. Si el grafo esta
     roto en pedazos, la "curva" puede estar saltando de componente en componente.

Como reproduzco el grafo: EXACTAMENTE la misma cadena de `cs090_fase5_motor.correr_regla_coarse()`
(lineas 433-436 y 454-475 del motor) -- mismo `p` via `cs090_fase5_generador.generar_regla`, mismo rng
`seed*5000+N`, mismo `construir_A2` + `dinamica_B0` + `medir`, mismas cajas `cs080.cajas_bfs` con rng
`seed*7000+b*31`. Los tres modulos se IMPORTAN, no se tocan ni se copian. La verificacion de que la
reproduccion es fiel es dura: recalculo la pendiente con el clasificador congelado y exijo que coincida
con la del CSV de origen hasta 1e-9; si no coincide, el script lo dice y no disfraza nada.

Salidas: cs090_fase6_outliers_curvas.csv (una fila por regla x escala, datos crudos),
cs090_fase6_outliers_curvas.png. No modifica nada existente. No corre Phantom. No declara veredicto.
"""
from __future__ import annotations
import csv
import sys
import time
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)

import cs090_fase5_generador as GEN
import cs090_fase5_motor as MOT
import cs080_renormalizacion as CS80
from cs090_fase5_clasificador import _pendiente_loglog

N_PILOTO = 2000
N_SWEEPS = 14
ESCALAS = (1, 2, 4, 8, 16)

RUTA_430 = f"{HERE}/cs090_fase6_outliers_430_todas.csv"


# ------------------------------------------------------------------------------------------------
# diagnostico de componentes (nuevo, no lo hace el motor): cuantas hay, y en cual cae `_diam`
# ------------------------------------------------------------------------------------------------
def _componentes(adj, N):
    """Devuelve (n_componentes_no_triviales, tamano de la componente donde arranca _diam).
    Replica la eleccion de nodo fuente de `_diam` de cs055 (primer indice con vecinos) para poder
    decir sobre QUE pedazo del grafo se calculo el diametro reportado."""
    src = next((i for i in range(N) if adj[i]), None)
    seen = np.zeros(N, bool)
    n_comp = 0
    tam_src = 0
    for s in range(N):
        if seen[s] or not adj[s]:
            continue
        n_comp += 1
        q = deque([s]); seen[s] = True; c = 0; contiene_src = False
        while q:
            u = q.popleft(); c += 1
            if u == src:
                contiene_src = True
            for w in adj[u]:
                if not seen[w]:
                    seen[w] = True; q.append(int(w))
        if contiene_src:
            tam_src = c
    return n_comp, tam_src


def curva_detallada(seed, rule_id, N=N_PILOTO, n_sweeps=N_SWEEPS, escalas_b=ESCALAS):
    """Mismos pasos que correr_regla_coarse, con mediciones extra. Devuelve (p, filas, m_nativa)."""
    p = GEN.generar_regla("A2", "B0", "C2", idx=0, seed=seed)
    p["rule_id"] = rule_id
    rng = np.random.default_rng(p["seed"] * 5000 + N)
    sustrato = MOT.construir_A2(N, rng, p)
    t0 = time.time()
    sustrato = MOT.dinamica_B0(sustrato, p, rng, n_sweeps, "C2")
    m = MOT.medir(sustrato, p, rng)
    adj_real = m["adj_final"]
    t_din = time.time() - t0

    filas = []
    for b in escalas_b:
        if b == 1:
            adj_g, n_cajas = adj_real, N
        else:
            rng_b = np.random.default_rng(p["seed"] * 7000 + b * 31)
            asign, n_cajas = CS80.cajas_bfs(adj_real, N, b, rng_b)
            adj_g = CS80.grafo_grueso(adj_real, N, asign, n_cajas)
        diam_g = float(MOT._diam(adj_g, n_cajas)) if n_cajas > 1 else float("nan")
        giant_g = float(MOT._giant(adj_g, n_cajas)) if n_cajas > 1 else 0.0
        n_aristas_g = sum(len(a) for a in adj_g) // 2
        n_comp, tam_src = _componentes(adj_g, n_cajas)
        n_aislados = sum(1 for i in range(n_cajas) if not adj_g[i])
        filas.append(dict(rule_id=rule_id, seed=seed, escala_b=b, n_cajas=n_cajas,
                          diam=diam_g, giant=giant_g, n_aristas=n_aristas_g,
                          grado_medio=2.0 * n_aristas_g / max(1, n_cajas),
                          n_componentes=n_comp, tam_comp_del_diam=tam_src, n_aislados=n_aislados))
    return p, filas, m, t_din


def cargar_430():
    filas = {}
    with open(RUTA_430) as f:
        for row in csv.DictReader(f):
            filas[row["rule_id"]] = row
    return filas


def main(rule_ids):
    tabla = cargar_430()
    salida = []
    resumen = []
    for rid in rule_ids:
        info = tabla[rid]
        seed = int(info["seed"])
        pend_csv = float(info["pendiente"])
        p, filas, m, t_din = curva_detallada(seed, rid)
        # verificacion dura: la pendiente recalculada debe reproducir la del CSV de origen
        pend_repro = _pendiente_loglog([f["n_cajas"] for f in filas], [f["diam"] for f in filas])
        ok = abs(pend_repro - pend_csv) < 1e-9
        print(f"\n=== {rid}  seed={seed}  clase='{info['clase']}'  (K={p['K']} J={p['J']:.3f} "
              f"noise={p['noise']:.3f} meandeg={p['meandeg']:.2f} kcap={p['kcap']}) [{t_din:.0f}s]")
        print(f"    pendiente CSV={pend_csv:+.6f}  recalculada={pend_repro:+.6f}  "
              f"REPRODUCE={'SI' if ok else 'NO <-- ATENCION'}")
        print(f"    grafo nativo b=1: n_aristas={m['n_aristas']} grado_medio="
              f"{2*m['n_aristas']/N_PILOTO:.2f} giant={m['giant']:.4f} diam={m['diam']:.0f}")
        print("     b  n_cajas   diam   giant   n_aristas  gr.medio  n_comp  tam_comp(diam)  aislados")
        for f in filas:
            print(f"    {f['escala_b']:2d}  {f['n_cajas']:7d}  {f['diam']:5.0f}  {f['giant']:.4f}  "
                  f"{f['n_aristas']:9d}  {f['grado_medio']:7.2f}  {f['n_componentes']:6d}  "
                  f"{f['tam_comp_del_diam']:13d}  {f['n_aislados']:8d}")
        # monotonia: la curva log(diam) contra log(n_cajas), en orden de b creciente
        diams = [f["diam"] for f in filas]
        monot = ("decreciente (diam baja al agrupar)" if all(diams[i] >= diams[i+1] for i in range(4))
                 else "creciente (diam SUBE al agrupar)" if all(diams[i] <= diams[i+1] for i in range(4))
                 else "NO MONOTONA")
        print(f"    forma de la curva de diametro (b=1->16): {diams} -> {monot}")
        for f in filas:
            f["clase_csv"] = info["clase"]; f["pendiente_csv"] = pend_csv
            f["pendiente_repro"] = pend_repro; f["reproduce"] = ok
            f["K"] = p["K"]; f["J"] = p["J"]; f["noise"] = p["noise"]
            f["meandeg"] = p["meandeg"]; f["kcap"] = p["kcap"]
            f["monotonia"] = monot
        salida.extend(filas)
        resumen.append((rid, info["clase"], pend_csv, monot, diams,
                        [f["giant"] for f in filas], [f["n_componentes"] for f in filas]))

    campos = list(salida[0].keys())
    with open(f"{HERE}/cs090_fase6_outliers_curvas.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos); w.writeheader(); w.writerows(salida)
    print(f"\n[csv] cs090_fase6_outliers_curvas.csv ({len(salida)} filas)")

    # ---- grafico: curva cruda de cada regla ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ids = [r[0] for r in resumen]
    colores = plt.cm.tab10(np.linspace(0, 1, max(10, len(ids))))
    for k, rid in enumerate(ids):
        sub = [f for f in salida if f["rule_id"] == rid]
        x = [f["n_cajas"] for f in sub]
        est = "-o" if sub[0]["pendiente_csv"] < 0 else "--s"
        lab = f"{rid} (p={sub[0]['pendiente_csv']:+.2f}, {sub[0]['clase_csv'][:12]})"
        axes[0].plot(x, [f["diam"] for f in sub], est, color=colores[k], label=lab, ms=5)
        axes[1].plot(x, [f["giant"] for f in sub], est, color=colores[k], ms=5)
        axes[2].plot(x, [f["n_componentes"] for f in sub], est, color=colores[k], ms=5)
    for ax, t, yl in ((axes[0], "diametro vs nº de cajas (log-log)", "diametro"),
                      (axes[1], "componente gigante", "giant (fraccion)"),
                      (axes[2], "nº de componentes conexas", "n_componentes")):
        ax.set_xscale("log"); ax.set_xlabel("nº de cajas (N_b)"); ax.set_ylabel(yl); ax.set_title(t)
        ax.grid(alpha=0.25)
    axes[0].set_yscale("log")
    axes[2].set_yscale("log")
    axes[0].legend(fontsize=6.5, loc="best")
    fig.suptitle("Curva de coarse-graining cruda: outliers de pendiente muy negativa vs reglas de referencia",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{HERE}/cs090_fase6_outliers_curvas.png", dpi=130)
    print(f"[png] cs090_fase6_outliers_curvas.png")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        main(["A2-B0-C2-batch3-r100", "A2-B0-C2-batch4-r51", "A2-B0-C2-batch3-r143"])

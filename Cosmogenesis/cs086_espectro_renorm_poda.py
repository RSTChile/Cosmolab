"""
CS086 — ¿SUENA distinto bajo renormalización o bajo poda? Espectro del laplaciano aplicado a Fase III
========================================================================================================
Este script cruza dos líneas que ya existían por separado y las junta con UNA sola herramienta de
medición:

  - CS084 (`cs084_espectro_laplaciano.py`) construyó el diagnóstico ESPECTRAL completo (forma de la
    densidad de eigenvalues, dimensión espectral por núcleo de calor Tr(e^{-tL}), estadística de
    espaciado de niveles Poisson-vs-GOE) sobre el tejido CRUDO de CS066 (b=1, sin agrupar, sin podar).
  - Fase III (`cs080_renormalizacion.py` Exp.1, `cs081_poda_dinamica.py` Exp.2) aplicó dos
    intervenciones al MISMO tejido -- agrupar en supernodos (renormalización) y podar enlaces por costo
    -- y las juzgó con UN SOLO número: la pendiente log-log del DIÁMETRO. Ese juez no distinguió real de
    NULL bajo renormalización (Exp.1: 0.376 vs 0.420 vs 0.406, solapadas) y sólo distinguió costo de
    azar de forma modesta bajo poda (Exp.2: costo_P50=0.786 vs azar_P50=0.655 vs sin_poda=0.421).

La pregunta de HOY (pedida por Alexis): el diámetro es UN número que resume el grafo. El espectro
completo del laplaciano lleva mucha más información (todos los "tonos" del tambor, no sólo su
"tamaño"). ¿Ese instrumento más fino ve algo que el diámetro no vio -- en renormalización, en poda, o en
ninguna de las dos?

QUÉ HACE ESTE SCRIPT, EN DOS PARTES:
  PARTE A -- espectro bajo renormalización: para cada escala b en {1,4,16} (subconjunto de las b=1,2,4,
    8,16,32 de CS080, elegido para acotar el costo -- ver nota de tiempo abajo), se construye el tejido
    crudo de CS066 (motor `C80.construir_sustrato`, brazo local/local_barajado/er_null, MISMAS 3 semillas
    que Fase III: 80100,80200,80300) y se agrupa con el MISMO coarse-graining por cajas BFS de CS080
    (`C80.cajas_bfs`+`C80.grafo_grueso`+`C80.propagar_spins`, sin tocar). Sobre el grafo YA AGRUPADO
    (no el original) se corren los 3 diagnósticos espectrales de CS084.
  PARTE B -- espectro bajo poda: para el tejido SIN agrupar (b=1) pero PODADO con el motor exacto de
    CS081 (`C81.proceso066_instrumentado` + `C81.costo_por_arista` + `C81.podar_por_costo` /
    `C81.podar_aleatorio`), MISMAS 3 semillas, variantes sin_poda / costo_P50 / azar_P50 (P50 es donde
    Fase III vio la brecha costo-vs-azar más grande: +0.131 en la pendiente de diámetro). Se corren los
    mismos 3 diagnósticos espectrales sobre cada variante podada.

FUNCIONES REUSADAS TAL CUAL (ningún archivo congelado se toca, sólo import):
  - `cs080_renormalizacion.py`: construir_sustrato, cajas_bfs, grafo_grueso, propagar_spins.
  - `cs081_poda_dinamica.py`: proceso066_instrumentado, costo_por_arista, podar_por_costo,
    podar_aleatorio.
  - `cs084_espectro_laplaciano.py`: dimension_espectral (núcleo de calor), unfolding_local, cdf_goe,
    estadisticas_espaciado (espaciado de niveles), T_GRID.
  - `cs064_smoke.py`: adj_sparse, _cataloga vía cs064_sistema_completo.
  NOTA HONESTA sobre `laplaciano_denso` de CS084: esa función está atada a construir el tejido DESDE
  CERO vía `C80.construir_sustrato(N,seed,arm)` -- sólo sabe fabricar el tejido b=1 sin podar. Acá hace
  falta el espectro de un grafo YA CONSTRUIDO (agrupado o podado desde afuera), así que se define
  `espectro_desde_adj(adj,N)` abajo: es el MISMO patrón de 5 líneas de `laplaciano_denso` (adj_sparse ->
  L=D-A -> eigh denso -> clip negativos numéricos -> sort), aplicado a una adyacencia que ya existe en
  vez de construirla. No es una reescritura del método, es la misma receta aplicada a otro insumo.

NOTA DE TIEMPO / N usado: se mantiene N=8000 (la N EXACTA de Fase III, para comparar manzana con
manzana) y diagonalización DENSA completa (igual que CS084, necesaria para el diagnóstico de espaciado
de niveles). El costo dominante es SIEMPRE la escala b=1 (N_b=8000, denso ~60-70s por matriz en esta
máquina, medido por CS084); b=4 y b=16 son mucho más baratos porque N_b baja a ~2000-2900 y ~500-1500 --
la diagonalización densa escala ~N^3, así que son órdenes de magnitud más rápidas. Para acotar el
presupuesto de ~50-60 min se usan 3 semillas (no 5) y sólo P50 en la Parte B (donde Fase III vio la
brecha costo-vs-azar más grande) -- ambas reducciones documentadas, no ocultas.

No se toca `cs080_renormalizacion.py`, `cs081_poda_dinamica.py`, `cs084_espectro_laplaciano.py` ni
`cs066_localidad_geometrogenesis.py`. No se declara cierre ni veredicto de arco -- se reportan números,
la lectura final es de Alexis.

Codea/ejecuta: CC (Claude).
"""
from __future__ import annotations
import os, sys, time, csv
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.linalg import eigh

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)

import cs080_renormalizacion as C80        # construir_sustrato, cajas_bfs, grafo_grueso, propagar_spins -- SIN tocar
import cs081_poda_dinamica as C81          # proceso066_instrumentado, costo_por_arista, podar_* -- SIN tocar
import cs084_espectro_laplaciano as C84    # dimension_espectral, unfolding_local, estadisticas_espaciado, T_GRID -- SIN tocar
import cs064_smoke as SM                   # adj_sparse -- SIN tocar
import cs064_sistema_completo as C64       # _cataloga, DMAX_INT -- SIN tocar

RNG = np.random.default_rng

N_NODOS    = int(os.environ.get("CS086_N", 8000))                     # misma N que Fase III (cs080/cs081)
K_LOCAL    = int(os.environ.get("CS086_KLOC", 6))                     # mismo k_local que Fase III
SEEDS      = [int(x) for x in os.environ.get("CS086_SEEDS", "80100,80200,80300").split(",")]  # MISMAS 3 semillas de Fase III
ESCALAS_A  = [int(x) for x in os.environ.get("CS086_B", "1,4,16").split(",")]                 # subconjunto de b de CS080
ARMS_A     = ("local", "local_barajado", "er_null")
PERCENTIL_B = int(os.environ.get("CS086_P", 50))                      # dónde Fase III vio la brecha costo-vs-azar más grande
OUT_A      = os.environ.get("CS086_OUT_A", os.path.join(_HERE, "cs086_espectro_renorm.csv"))
OUT_B      = os.environ.get("CS086_OUT_B", os.path.join(_HERE, "cs086_espectro_poda.csv"))


# ============================ ESPECTRO de un grafo YA CONSTRUIDO (agrupado o podado) ============================
def espectro_desde_adj(adj, N):
    """Mismo patrón de cálculo que C84.laplaciano_denso (adj_sparse -> L=D-A -> eigh denso -> clip a 0
    -> sort), aplicado a una adyacencia que YA EXISTE (coarse-grained por C80 o podada por C81) en vez
    de construirla desde cero -- laplaciano_denso de CS084 sólo sabe fabricar tejido b=1 sin podar
    (está atado a construir_sustrato). Devuelve eigenvalues ordenados, nº de componentes, frac. gigante."""
    if N < 4:
        return np.array([0.0]), 1, float("nan")
    A = SM.adj_sparse(adj, N)
    n_comp, labels = connected_components(A, directed=False)
    giant = float(np.max(np.bincount(labels))) / N
    deg = np.asarray(A.sum(axis=1)).ravel()
    L = sparse.diags(deg) - A
    Ld = L.toarray()
    w = eigh(Ld, eigvals_only=True)
    w = np.clip(w, 0.0, None)
    w.sort()
    return w, int(n_comp), giant


def resumir_fila(eigvals, n_comp, giant, meta):
    """Aplica los 3 diagnósticos de CS084 (dimension_espectral, unfolding_local+estadisticas_espaciado)
    a un espectro ya calculado y arma una fila de resumen -- misma estructura de campos que
    cs084_espectro_laplaciano.csv, para poder comparar directo con esa tabla."""
    lam2 = float(eigvals[n_comp]) if n_comp < len(eigvals) else float("nan")
    lam_max = float(eigvals[-1])
    d_s_curve, _ = C84.dimension_espectral(eigvals)
    s = C84.unfolding_local(eigvals, n_comp)
    stats_esp = C84.estadisticas_espaciado(s)
    fila = dict(meta)
    fila.update(dict(
        N=len(eigvals), n_componentes=n_comp, giant_frac=round(giant, 4) if giant == giant else giant,
        lambda2=round(lam2, 6), lambda_max=round(lam_max, 3),
        mean_eig=round(float(np.mean(eigvals)), 4), std_eig=round(float(np.std(eigvals)), 4),
    ))
    for tt in (0.05, 0.2, 1.0, 5.0):
        j = int(np.argmin(np.abs(C84.T_GRID - tt)))
        fila[f"d_s_t{tt}"] = round(float(d_s_curve[j]), 3)
    fila.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats_esp.items()})
    return fila


# ============================ PARTE A -- espectro bajo renormalización (b=1,4,16) ============================
def parte_a():
    print("=" * 100, flush=True)
    print("CS086 PARTE A -- espectro del laplaciano de los grafos COARSE-GRAINED de Fase III Exp.1", flush=True)
    print(f"N={N_NODOS}  k_local={K_LOCAL}  escalas b={ESCALAS_A}  semillas={SEEDS}  brazos={ARMS_A}", flush=True)
    print("=" * 100, flush=True)
    t0 = time.time()
    filas = []
    for arm in ARMS_A:
        for seed in SEEDS:
            tb0 = time.time()
            adj, V = C80.construir_sustrato(N_NODOS, seed, arm)
            print(f"  [{arm:<15}] seed={seed}  tejido b=1 construido ({time.time()-tb0:.1f}s)", flush=True)
            for b in ESCALAS_A:
                tb = time.time()
                if b == 1:
                    adj_b, N_b = adj, N_NODOS
                else:
                    rng_b = RNG(seed * 733 + b * 31 + hash(arm) % 4999)
                    asign, n_cajas = C80.cajas_bfs(adj, N_NODOS, b, rng_b)
                    adj_b = C80.grafo_grueso(adj, N_NODOS, asign, n_cajas)
                    N_b = n_cajas
                eigvals, n_comp, giant = espectro_desde_adj(adj_b, N_b)
                fila = resumir_fila(eigvals, n_comp, giant, dict(seed=seed, arm=arm, b=b))
                filas.append(fila)
                print(f"    b={b:<3} N_b={N_b:<5} n_comp={n_comp:<4} giant={giant:.3f}  "
                      f"lambda2={fila['lambda2']:.5f}  lambda_max={fila['lambda_max']:.2f}  "
                      f"d_s(t=1.0)={fila['d_s_t1.0']}  <s^2>={fila['mean_s2']}  "
                      f"KS_Poiss={fila['ks_D_poisson']}  KS_GOE={fila['ks_D_goe']}  ({time.time()-tb:.1f}s)",
                      flush=True)
    campos = list(filas[0].keys())
    with open(OUT_A, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=campos)
        wr.writeheader()
        for fila in filas:
            wr.writerow(fila)
    print(f"\nPARTE A completa en {(time.time()-t0)/60:.1f} min -> {OUT_A}", flush=True)
    return filas


# ============================ PARTE B -- espectro bajo poda (b=1, sin_poda/costo_P50/azar_P50) ============================
def parte_b():
    print("=" * 100, flush=True)
    print("CS086 PARTE B -- espectro del laplaciano de los grafos PODADOS de Fase III Exp.2", flush=True)
    print(f"N={N_NODOS}  k_local={K_LOCAL}  semillas={SEEDS}  percentil={PERCENTIL_B}  "
          f"variantes=sin_poda,costo_P{PERCENTIL_B},azar_P{PERCENTIL_B}", flush=True)
    print("=" * 100, flush=True)
    t0 = time.time()
    filas = []
    for seed in SEEDS:
        tb0 = time.time()
        rng = RNG(seed)
        cat = C64._cataloga(N_NODOS, rng)
        r2 = RNG(seed * 137 + hash("local") % 9973 + 5)   # MISMA derivación de semilla que C81.corre_semilla
        adj, V, flip_count = C81.proceso066_instrumentado(N_NODOS, cat, K_LOCAL, r2)
        n_edges0 = sum(len(a) for a in adj) // 2
        print(f"  seed={seed}  tejido instrumentado construido: aristas={n_edges0} ({time.time()-tb0:.1f}s)",
              flush=True)

        rng_costo = RNG(seed * 911 + 3)
        edges, costo, _ = C81.costo_por_arista(adj, N_NODOS, V, flip_count, K_LOCAL, rng_costo)
        print(f"  costo calculado sobre {len(edges)} aristas ({time.time()-tb0:.1f}s acumulado)", flush=True)

        na_costo, n_pod = C81.podar_por_costo(adj, N_NODOS, edges, costo, PERCENTIL_B)
        rng_rand = RNG(seed * 733 + PERCENTIL_B)
        na_azar = C81.podar_aleatorio(adj, N_NODOS, edges, n_pod, rng_rand)

        variantes = {"sin_poda": (adj, 0), f"costo_P{PERCENTIL_B}": (na_costo, n_pod),
                     f"azar_P{PERCENTIL_B}": (na_azar, n_pod)}
        for nombre, (adj_v, n_podadas) in variantes.items():
            tv = time.time()
            eigvals, n_comp, giant = espectro_desde_adj(adj_v, N_NODOS)
            fila = resumir_fila(eigvals, n_comp, giant,
                                 dict(seed=seed, variante=nombre, n_podadas=n_podadas, b=1))
            filas.append(fila)
            print(f"    [{nombre:<12}] podadas={n_podadas:<6} n_comp={n_comp:<4} giant={giant:.3f}  "
                  f"lambda2={fila['lambda2']:.5f}  lambda_max={fila['lambda_max']:.2f}  "
                  f"d_s(t=1.0)={fila['d_s_t1.0']}  <s^2>={fila['mean_s2']}  "
                  f"KS_Poiss={fila['ks_D_poisson']}  KS_GOE={fila['ks_D_goe']}  ({time.time()-tv:.1f}s)",
                  flush=True)
    campos = list(filas[0].keys())
    with open(OUT_B, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=campos)
        wr.writeheader()
        for fila in filas:
            wr.writerow(fila)
    print(f"\nPARTE B completa en {(time.time()-t0)/60:.1f} min -> {OUT_B}", flush=True)
    return filas


def main():
    t0 = time.time()
    filas_a = parte_a()
    filas_b = parte_b()
    print(f"\nCS086 COMPLETO -- Parte A + Parte B en {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"  CSV Parte A (renormalización): {OUT_A}", flush=True)
    print(f"  CSV Parte B (poda):            {OUT_B}", flush=True)
    return filas_a, filas_b


if __name__ == "__main__":
    main()

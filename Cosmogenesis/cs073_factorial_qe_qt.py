"""
cs073_factorial_qe_qt.py — diseño factorial q_E (identidad de aristas) x q_T (orden de inserción),
variados INDEPENDIENTEMENTE, medido con el espectro del laplaciano (sin Phantom).

POR QUÉ (encargo de Alexis, 10-ago-2026): NULL-3 (double-edge-swap + filtro de longitud) y NULL-4
(mismo conjunto de aristas, orden de inserción rebarajado) cambian DOS cosas a la vez de forma no
independiente -- NULL-3 cambia identidad de aristas Y de hecho también revuelve el orden (el barajado
reconstruye el dict de adyacencia desde un set, sin preservar ningún orden original); NULL-4 cambia
SÓLO el orden. Ninguno de los dos aísla completamente su eje. Este script construye un continuo en
cada eje por separado -- q_E: fracción de aristas ORIGINALES conservadas; q_T: fracción de POSICIONES
del orden de inserción que quedan intactas -- para poder cruzarlos en una grilla factorial y ver cuál
de los dos (o su interacción) mueve la distancia espectral a REAL.

REGLA DE LA CASA (recordatorio, no se repite en cada función): NO se corre Phantom. Todo lo que sigue
mide el grafo/malla ANTES del colapso gravitacional, con el mismo instrumento espectral de CS084/CS085.

QUÉ SE REUSA, SIN TOCAR NINGÚN ARCHIVO CONGELADO (sólo import/lectura) -- mismas piezas que CS085:
  - `cs084_espectro_laplaciano.dimension_espectral` / `unfolding_local` / `estadisticas_espaciado` /
    `T_GRID` -- el diagnóstico de forma/dimensión/espaciado, tal cual.
  - `cs085_espectro_jerarquia_cs073.laplaciano_denso_desde_adj` -- construcción L=D-A desde un adj tipo
    dict-de-sets (formato nativo de CS073), tal cual.
  - `cs072_modulos.piezas.p_semilla_causal.malla_causal_atomos` -- reconstruye la malla causal REAL.
  - `null3_investigacion_preliminar.barajar_aristas_preservando_longitud` / `aristas_de` -- el
    double-edge-swap con filtro de longitud que define NULL-3 (aquí sólo se le hace variar
    `factor_swaps`, el propio parámetro que el archivo congelado ya expone -- no se edita el archivo).
  - `null4_verificar_invarianza_orden.adj_a_lista_aristas` / `construir_adj_en_orden` -- cómo se
    reconstruye un `adj` a partir de una lista de aristas en un ORDEN dado (lo que define NULL-4).
  - `null1_generar_ic.leer_ic_txt` -- para cargar pos_real (referencia de longitud del swap).

QUÉ ES NUEVO (no existía en ningún archivo previo, es la lógica propia de esta tarea):
  - `q_E -> factor_swaps`: parametrización continua del "cuánto" de NULL-3 (ver docstring de
    `adj_con_qE` más abajo).
  - `q_T -> orden parcialmente barajado`: parametrización continua del "cuánto" de NULL-4 (ver
    docstring de `orden_con_qT` más abajo) -- NULL-4 tal como está congelado sólo conoce el barajado
    COMPLETO (`rng.permutation` entero); acá se necesita un dial continuo, así que se construye una
    función nueva que interpola entre "orden intacto" y "orden completamente al azar".

HALLAZGO ESPERADO A PRIORI, NO EMPÍRICO (léase antes de correr): L=D-A (el laplaciano) es una función
del CONJUNTO de aristas únicamente -- nunca del orden en que ese conjunto se insertó en las estructuras
de adyacencia. CS085 ya lo confirmó numéricamente para el caso extremo (NULL-4 puro: max|Δeigenvalue|
vs REAL = 0.000e+00 exacto). Por lo tanto, para CUALQUIER q_E, se espera que TODO el eje q_T dé el
espectro IDÉNTICO (hasta ruido de punto flotante) -- no una tendencia empírica que haya que descubrir,
sino una identidad algebraica que este script re-confirma con una grilla más fina en vez de asumirla.
Si el resultado sale distinto de esto, hay un bug (posible ruptura del assert de conjunto de aristas
idéntico) -- se audita, no se reporta como hallazgo.

No se declara cierre ni veredicto -- sólo se reportan números. La lectura final es de Alexis.

Codea/ejecuta: CC (Claude).
"""
from __future__ import annotations
import os, sys, time, csv
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)

from cs084_espectro_laplaciano import dimension_espectral, unfolding_local, estadisticas_espaciado, T_GRID
from cs085_espectro_jerarquia_cs073 import laplaciano_denso_desde_adj
from cs072_modulos.piezas.p_semilla_causal import malla_causal_atomos
from null1_generar_ic import leer_ic_txt
from null3_investigacion_preliminar import barajar_aristas_preservando_longitud, aristas_de
from null4_verificar_invarianza_orden import adj_a_lista_aristas, construir_adj_en_orden

RUTA_POOL = "/Users/alexis/phantom_cs073/bateria_n2000"
RUTA_IC_REAL = f"{RUTA_POOL}/ic_real/cosmogenesis_ic.txt"
D_CAUSAL, K_CAUSAL = 3, 4
SEED_EJES_CANONICO = 2000        # misma malla REAL canónica que usó CS085/toda la jerarquía

# -------- grilla factorial (documentada, ver informe para la justificación de la asimetría) --------
Q_E_GRID = [1.0, 0.75, 0.5, 0.25, 0.0]     # 5 niveles -- eje "identidad de aristas" (NULL-3)
Q_T_GRID = [1.0, 0.5, 0.0]                 # 3 niveles -- eje "orden de inserción" (NULL-4)
N_SEMILLAS = 5                              # semillas de swap por celda de q_E (5, igual que CS085/NULL-3)
FACTOR_SWAPS_MAX = 10.0                     # factor_swaps de NULL-3 "puro" (mismo valor que usó CS085)
SEED_SWAP_BASE = 6000                       # semillas de double-edge-swap: 6000+i
SEED_ORDEN_BASE = 7000                      # semillas del barajado parcial de orden: 7000+i*10+j

OUT_CSV = os.path.join(_HERE, "cs073_factorial_qe_qt.csv")


def adj_con_qE(adj_real, n, pos_real, q_E, seed_swap, tol_relativa=0.2, factor_swaps_max=FACTOR_SWAPS_MAX):
    """Parametrización continua de NULL-3: factor_swaps = factor_swaps_max * (1 - q_E). q_E=1.0 ->
    factor_swaps=0 -> CERO intentos de swap -> grafo original intacto (edge_set = aristas de REAL,
    0 aceptados). q_E=0.0 -> factor_swaps=factor_swaps_max=10, EXACTAMENTE el mismo parámetro que usó
    NULL-3 "puro" en CS085/NULL3_resultado_CS.md (~12-13% de aristas distintas de REAL). Valores
    intermedios de q_E dan un número de INTENTOS de swap proporcional a (1-q_E) -- el número de
    ACEPTADOS depende además del filtro de longitud (aceptación no garantizada), así que la relación
    entre q_E nominal y % de aristas realmente distintas se mide y se reporta (columna
    `pct_aristas_distintas`), no se asume lineal."""
    factor_swaps_nominal = factor_swaps_max * (1.0 - q_E)
    # barajar_aristas_preservando_longitud hace `range(factor_swaps * len(edges))` -- exige factor_swaps
    # entero (n_intentos entero). Se redondea al entero más cercano (q_E=1.0 -> 0 exacto -> 0 intentos,
    # q_E=0.0 -> 10 exacto -> idéntico al factor_swaps=10 "puro" de NULL-3/CS085; los 3 valores
    # intermedios (7.5, 5.0, 2.5) redondean a 8, 5, 2 -- se reporta el valor nominal (float) Y el
    # redondeado en el CSV para que quede auditable).
    factor_swaps_int = int(round(factor_swaps_nominal))
    adj_qe, aceptados, intentos = barajar_aristas_preservando_longitud(
        adj_real, n, pos_real, seed=seed_swap, factor_swaps=factor_swaps_int, tol_relativa=tol_relativa)
    return adj_qe, aceptados, intentos, factor_swaps_nominal, factor_swaps_int


def orden_con_qT(n_aristas, q_T, seed_orden):
    """Parametrización continua de NULL-4: arranca de la identidad (orden = 0..n_aristas-1, "orden
    original intacto"). Elige al azar una fracción (1-q_T) de POSICIONES del array de orden y revuelve
    (shuffle) SOLO esas posiciones entre sí -- el resto de las posiciones queda exactamente en su lugar
    original. q_T=1.0 -> 0 posiciones tocadas -> orden idéntico al de construcción natural de
    `malla_causal_atomos` (equivalente a "orden real intacto"). q_T=0.0 -> el 100% de las posiciones
    entra en el sorteo -> equivale a un `rng.permutation` completo sobre todo el array, el mismo
    mecanismo que usa NULL-4 "puro" (`null4_generar_ic.generar_null4`/`null4_verificar_invarianza_orden`
    con q_T=0 como caso límite). Valores intermedios interpolan continuamente la FRACCIÓN de posiciones
    perturbadas, no la intensidad del revuelto en sí (una vez que una posición entra en el sorteo, se
    reasigna con la misma libertad que en el barajado completo)."""
    rng = np.random.default_rng(seed_orden)
    orden = np.arange(n_aristas)
    frac_tocada = 1.0 - q_T
    n_tocadas = int(round(frac_tocada * n_aristas))
    if n_tocadas >= 2:
        idx_tocadas = rng.choice(n_aristas, size=n_tocadas, replace=False)
        valores = orden[idx_tocadas].copy()
        rng.shuffle(valores)
        orden[idx_tocadas] = valores
    return orden.tolist(), n_tocadas


def diagnostico_fila(q_E, q_T, seed_swap, seed_orden, adj, n, extra=None):
    """Misma estructura de columnas que `cs085_espectro_jerarquia_cs073.diagnostico_fila` (reusa las
    mismas funciones de cálculo), con las columnas q_E/q_T añadidas al frente."""
    t0 = time.time()
    w, n_comp, giant = laplaciano_denso_desde_adj(adj, n)
    n_aristas = int(round(float(np.sum([len(adj.get(i, ())) for i in range(n)])) / 2))
    lam2 = float(w[n_comp]) if n_comp < len(w) else float("nan")
    d_s_curve, _tr = dimension_espectral(w)
    s = unfolding_local(w, n_comp)
    stats_esp = estadisticas_espaciado(s)
    fila = dict(q_E=q_E, q_T=q_T, seed_swap=seed_swap, seed_orden=seed_orden, n=n, n_aristas=n_aristas,
                n_componentes=n_comp, giant_frac=round(giant, 4),
                lambda2=round(lam2, 6), lambda_max=round(float(w[-1]), 3),
                mean_eig=round(float(np.mean(w)), 4), std_eig=round(float(np.std(w)), 4))
    for tt in (0.05, 0.2, 1.0, 5.0):
        j = int(np.argmin(np.abs(T_GRID - tt)))
        fila[f"d_s_t{tt}"] = round(float(d_s_curve[j]), 3)
    fila.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats_esp.items()})
    if extra:
        fila.update(extra)
    fila["_eigvals"] = w
    fila["_tiempo_s"] = round(time.time() - t0, 2)
    return fila


def main():
    t_inicio = time.time()
    print("=" * 100, flush=True)
    print("CS073 factorial q_E x q_T -- espectro del laplaciano, SIN Phantom", flush=True)
    print("=" * 100, flush=True)

    dens_bar = np.load(f"{RUTA_POOL}/dens_bar.npy")
    n = len(dens_bar)
    pos_real = leer_ic_txt(RUTA_IC_REAL)[0]
    print(f"pool cargado: n={n}\n", flush=True)

    print("reconstruyendo malla causal REAL canónica (seed_ejes=2000, misma que CS085)...", flush=True)
    adj_real, _m = malla_causal_atomos(dens_bar, D=D_CAUSAL, k=K_CAUSAL, seed_ejes=SEED_EJES_CANONICO)
    lista_real = adj_a_lista_aristas(adj_real, n)
    n_aristas_real = len(lista_real)
    print(f"  n_aristas REAL = {n_aristas_real}\n", flush=True)

    # -------------------- punto de referencia REAL puro (q_E=1.0, q_T=1.0) --------------------
    fila_real = diagnostico_fila(1.0, 1.0, None, None, adj_real, n,
                                  extra=dict(pct_aristas_distintas=0.0, n_tocadas_orden=0,
                                             factor_swaps_nominal=0.0, factor_swaps_int=0,
                                             es_referencia="REAL"))
    filas = [fila_real]
    print(f"[REF REAL] lambda2={fila_real['lambda2']} lambda_max={fila_real['lambda_max']} "
          f"std_eig={fila_real['std_eig']} ({fila_real['_tiempo_s']}s)", flush=True)

    # -------------------- grilla factorial q_E x q_T x semillas --------------------
    total_celdas = len(Q_E_GRID) * len(Q_T_GRID) * N_SEMILLAS
    contador = 0
    print(f"\ngrilla: {len(Q_E_GRID)} (q_E) x {len(Q_T_GRID)} (q_T) x {N_SEMILLAS} (semillas de swap) "
          f"= {total_celdas} celdas\n", flush=True)

    for q_E in Q_E_GRID:
        for i_seed in range(N_SEMILLAS):
            seed_swap = SEED_SWAP_BASE + i_seed
            # el swap de aristas (eje q_E) se hace UNA vez por (q_E, semilla) -- después se le aplican
            # los distintos q_T (barajados de orden) sobre el MISMO conjunto de aristas resultante, así
            # el eje q_T queda genuinamente aislado (misma identidad de aristas, sólo cambia el orden).
            adj_qe, aceptados, intentos, factor_swaps_nominal, factor_swaps_int = adj_con_qE(
                adj_real, n, pos_real, q_E, seed_swap)
            lista_qe = adj_a_lista_aristas(adj_qe, n)
            n_aristas_qe = len(lista_qe)
            edges_real_set = set(tuple(e) for e in lista_real)
            edges_qe_set = set(tuple(e) for e in lista_qe)
            pct_distintas = 100.0 * len(edges_qe_set - edges_real_set) / len(edges_real_set)

            for q_T in Q_T_GRID:
                seed_orden = SEED_ORDEN_BASE + i_seed * 10 + Q_T_GRID.index(q_T)
                orden, n_tocadas = orden_con_qT(n_aristas_qe, q_T, seed_orden)
                adj_qt = construir_adj_en_orden(lista_qe, n, orden)
                # sanidad: el conjunto de aristas NO debe cambiar por el barajado de orden (por
                # construcción de construir_adj_en_orden) -- se verifica, no se asume.
                assert sorted(adj_a_lista_aristas(adj_qt, n)) == sorted(lista_qe), (
                    f"¡conjunto de aristas cambió al aplicar q_T={q_T}! bug en orden_con_qT/"
                    f"construir_adj_en_orden -- auditar antes de reportar.")

                fila = diagnostico_fila(
                    q_E, q_T, seed_swap, seed_orden, adj_qt, n,
                    extra=dict(pct_aristas_distintas=round(pct_distintas, 2),
                                swap_aceptados=aceptados, swap_intentos=intentos,
                                factor_swaps_nominal=round(factor_swaps_nominal, 3),
                                factor_swaps_int=factor_swaps_int,
                                n_tocadas_orden=n_tocadas, n_aristas_qe=n_aristas_qe))
                filas.append(fila)
                contador += 1
                print(f"  [{contador}/{total_celdas}] q_E={q_E:.2f} q_T={q_T:.2f} seed_swap={seed_swap} "
                      f"aristas_distintas={pct_distintas:.1f}% lambda2={fila['lambda2']:.5f} "
                      f"lambda_max={fila['lambda_max']} ({fila['_tiempo_s']}s)", flush=True)

    # -------------------- guarda CSV (sin la columna de eigenvalues crudos) --------------------
    for f in filas:
        f.pop("_eigvals", None)
    campos = list(filas[0].keys())
    for f in filas[1:]:
        for k in f:
            if k not in campos:
                campos.append(k)
    with open(OUT_CSV, "w", newline="") as fcsv:
        wr = csv.DictWriter(fcsv, fieldnames=campos, restval="")
        wr.writeheader()
        for f in filas:
            wr.writerow(f)
    print(f"\nCSV -> {OUT_CSV}", flush=True)
    print(f"COMPLETO en {time.time()-t_inicio:.1f}s ({len(filas)} filas)", flush=True)


if __name__ == "__main__":
    main()

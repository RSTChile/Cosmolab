"""
cs073_factorial_qe_qt_posicional.py — el MISMO factorial q_E (identidad de aristas) × q_T (orden de
inserción) que ya corrió `cs073_factorial_qe_qt.py`, pero medido con un instrumento que SÍ puede ver el
orden de formación: las POSICIONES que devuelve `layout_resortes`.

--------------------------------------------------------------------------------------------------
POR QUÉ EXISTE ESTE ARCHIVO (encargo O2-A, 11-ago-2026)
--------------------------------------------------------------------------------------------------
`CS073_factorial_qE_qT_CS.md` reportó que en las 75 celdas del factorial el eje q_T dio efecto
EXACTAMENTE cero — pero no como hallazgo empírico sino por identidad matemática: el laplaciano L = D - A
se construye desde la matriz de adyacencia FINAL, que es una función del CONJUNTO de aristas y no tiene
ninguna columna de "quién se conectó primero". El espectro topológico es literalmente ciego a q_T.

Ese mismo informe señaló el camino no tomado, que es exactamente esta tarea: `layout_resortes`
(Fruchterman-Reingold) SÍ es sensible al orden de inserción — `null4_verificar_invarianza_orden.py` ya
lo había medido sobre el caso extremo (mismo conjunto de aristas, 4 órdenes distintos → hasta 38% de la
escala típica de coordenada en el peor |Δpos| individual). Acá se repite la grilla factorial completa
corriendo `layout_resortes` en cada celda y midiendo observables POSICIONALES, para responder con
números: ¿de qué tamaño es la huella de q_T comparada con la de q_E, cuando el instrumento puede verla?

REGLA DE LA CASA: NO se corre Phantom. Todo lo de abajo se mide sobre la nube de posiciones ANTES del
colapso gravitacional. No se declara cierre ni veredicto — sólo se reportan números.

--------------------------------------------------------------------------------------------------
QUÉ SE REUSA, SIN TOCAR NINGÚN ARCHIVO CONGELADO (sólo import/lectura)
--------------------------------------------------------------------------------------------------
  - `cs073_factorial_qe_qt.adj_con_qE` / `orden_con_qT` — LA PARAMETRIZACIÓN YA VALIDADA de los dos ejes
    (q_E → factor_swaps = round(10*(1-q_E)) sobre el double-edge-swap con filtro de longitud;
    q_T → fracción (1-q_T) de posiciones del orden de inserción que entran en un sorteo y se revuelven
    entre sí). Se importan tal cual, NO se reescriben ni se modifica ese archivo.
  - `null4_verificar_invarianza_orden.adj_a_lista_aristas` / `construir_adj_en_orden` — cómo se
    reconstruye un `adj` a partir de una lista de aristas en un ORDEN dado.
  - `cs072_modulos.piezas.p_semilla_causal.malla_causal_atomos` / `layout_resortes` — la malla causal
    REAL y el layout físico, con los MISMOS parámetros de toda la jerarquía (D=3, k=4, seed_ejes=2000,
    iters=100, seed_layout=12345).
  - `null1_generar_ic.leer_ic_txt` — carga `pos_real`, la referencia de longitud que usa el swap.

QUÉ ES NUEVO (la lógica propia de esta tarea):
  - correr `layout_resortes` en cada celda del factorial (el paso caro, ~52 s por grafo a N=2000);
  - los OBSERVABLES POSICIONALES (ver `observables_posicionales`, con la justificación de cada uno);
  - la descomposición de varianza de dos vías (q_E × q_T con réplicas) sobre cada observable;
  - dos controles de escala que el factorial anterior no necesitaba (ver `CONTROLES` más abajo).

--------------------------------------------------------------------------------------------------
LOS OBSERVABLES POSICIONALES Y POR QUÉ CADA UNO
--------------------------------------------------------------------------------------------------
Todos se calculan sobre `pos` (N×3) y sobre la configuración de REFERENCIA `pos_ref` (la celda
q_E=1.0, q_T=1.0: grafo original intacto, orden de inserción intacto).

  (A) DISTANCIA A LA REFERENCIA — "¿cuánto se movió cada partícula respecto de la configuración de
      referencia?". Es el observable más directo del encargo. Los nodos conservan su etiqueta y
      `layout_resortes` arranca de la MISMA nube inicial (`rng.uniform(seed=12345)` depende sólo de
      `n`, no del grafo), así que comparar coordenada a coordenada tiene sentido y no hace falta
      alinear/rotar nada.
        - `rms_a_ref`     = sqrt(promedio_i |pos_i - pos_ref_i|^2)   [unidades de coordenada]
        - `rms_a_ref_rel` = rms_a_ref / radio_giro_ref               [adimensional, fracción del tamaño
                             de la nube — así "4%" o "80%" se leen directo]
        - `max_a_ref`     = max_i |pos_i - pos_ref_i|                [el peor caso individual, la misma
                             cifra que reportó `null4_verificar_invarianza_orden`]

  (B) ESTADÍSTICA DE LA NUBE (no necesita referencia; describe la forma que quedó)
        - `radio_giro`   = sqrt(promedio_i |pos_i - centroide|^2) — tamaño global de la nube.
        - `anisotropia`  = sqrt(λ1/λ3) de la matriz de covarianza de las posiciones — cuán alargada
                            quedó (1.0 = esférica). Es invariante a traslación y a rotación, así que
                            captura FORMA, no sólo "se movió".
        - `d_nn_media` / `d_nn_std` — distancia al vecino más cercano (cKDTree): densidad LOCAL, la
                            escala que después decide si Phantom resuelve o no el grumo.
        - `d_knn8_media` — distancia al 8º vecino: la misma idea pero suavizada, menos sensible al par
                            casi-coincidente aislado.
        - `long_arista_media` / `long_arista_std` — longitud GEOMÉTRICA media de las aristas del grafo:
                            el único observable que acopla explícitamente topología (qué aristas hay,
                            eje q_E) con geometría (dónde quedaron los nodos, sensible a q_T).

  (C) ESPECTRO LAPLACIANO PONDERADO POR DISTANCIA — el análogo directo del instrumento del factorial
      anterior, pero con memoria de la geometría. En vez de A_ij ∈ {0,1} se usa
          w_ij = exp( -d_ij^2 / (2 σ^2) ),   σ = mediana de la longitud de arista de la REFERENCIA
      (σ fijo para toda la grilla, calculado una sola vez, para que las celdas sean comparables entre
      sí y no cada una con su propia escala). L_w = D_w - W, diagonalización densa completa. A
      diferencia de L = D - A, este operador SÍ cambia si los nodos terminan en otro lado, aunque el
      conjunto de aristas sea idéntico bit a bit — es la versión "que puede ver q_T" del mismo
      diagnóstico. Se reportan `lambda2_w`, `lambda_max_w`, `mean_eig_w`, `std_eig_w`.

--------------------------------------------------------------------------------------------------
CONTROLES DE ESCALA (para no leer un número sin vara de medir)
--------------------------------------------------------------------------------------------------
  1. `TECHO_seed_layout` — la MISMA configuración de referencia relajada desde 3 nubes iniciales
     distintas (`seed_layout` = 12346/12347/12348). Da la escala de "dos layouts completamente
     decorrelacionados del mismo grafo": el techo contra el cual comparar cualquier efecto de q_E/q_T.
     Sin esta vara, un `rms_a_ref` de 0.26 no se sabe si es mucho o poco.
  2. Determinismo: la celda (q_E=1.0, q_T=1.0) se calcula con las 5 semillas. Como `factor_swaps=0` no
     hace ningún intento de swap y `n_tocadas=0` no toca ninguna posición del orden, las 5 réplicas
     deben dar `rms_a_ref` EXACTAMENTE 0.0. Si no lo dan, hay una fuente de azar no contabilizada y se
     audita antes de reportar nada (misma disciplina de `feedback-verificar-determinismo-antes-de-
     asumir-azar`).

--------------------------------------------------------------------------------------------------
GRILLA Y PRESUPUESTO (dimensionada con un piloto cronometrado, no a ojo)
--------------------------------------------------------------------------------------------------
Dos mediciones de costo, las dos hechas ANTES de comprometer la grilla (y la segunda obligó a recortarla
— se documenta el recorte, no se esconde):
  1. Piloto serial, N=2000: `layout_resortes(iters=100)` = 52.0 s / grafo.
  2. Throughput REAL con 8 procesos (8 núcleos físicos): NO da 52/8 = 6.5 s/celda sino ~34 s/celda —
     el layout es denso O(n^2) con arrays (2000,2000,3) y satura el ancho de banda de memoria, así que
     8 procesos rinden ~1.5x, no 8x. Con ese número medido (no estimado), la grilla 5×5×5 = 125 celdas
     habría costado ~82 min de reloj, por encima del presupuesto de 60-75 min del encargo.

**Decisión documentada:** se mantiene N=2000 (NO se baja, para que la grilla sea directamente
comparable con el factorial topológico anterior, que corrió sobre exactamente la misma malla de 4945
aristas) y se recorta la GRILLA al mínimo que pide el encargo, 4×4×5 = 80 celdas (~48 min). Los niveles
que se sacrifican son los redundantes de cada eje, no los extremos ni los diagnósticos:

  q_E ∈ {1.0, 0.75, 0.5, 0.0}    — 4 niveles. Se saca q_E=0.25 (del anterior 5-niveles); quedan los dos
        extremos (grafo intacto / NULL-3 puro) y dos intermedios.
  q_T ∈ {1.0, 0.99, 0.9, 0.0}    — 4 niveles. Espaciado NO uniforme y concentrado cerca de q_T=1 A
        PROPÓSITO, decidido antes de correr: si el orden dejara huella por amplificación caótica del
        redondeo de punto flotante (`np.add.at` suma en distinto orden → diferencia ~1e-16 → 100
        iteraciones de una relajación no lineal), bastaría tocar el 1% de las posiciones para SATURAR
        el efecto, y la curva q_T sería un ESCALÓN. Si en cambio hay una dependencia estructurada del
        GRADO de desorden, la curva sería un GRADIENTE. El factorial anterior usaba {1.0, 0.5, 0.0},
        que no distingue esas dos posibilidades; acá se cubren 1%, 10% y 100% de posiciones revueltas.
        (Se saca q_T=0.5, el punto menos informativo: si 0.9 y 0.0 ya coinciden, 0.5 no agrega nada.)
  semillas: 5 (`seed_swap` = 6000..6004, `seed_orden` = 7000 + i*10 + j) — mismas familias de semilla
        que el factorial anterior. NO se recortan: son la vara de ruido contra la que se mide todo.

Aislamiento de los ejes, igual que antes: para cada (q_E, semilla) el swap de aristas se hace UNA vez, y
sobre ESE MISMO conjunto de aristas se aplican los distintos q_T — así el eje q_T compara exactamente
el mismo conjunto de aristas con distinto orden. Como todo es determinista dado (q_E, seed), cada
worker reconstruye su propia celda desde cero (barato: la malla son 0.14 s) en vez de pasar grafos
grandes entre procesos.

Codea/ejecuta: CC (Claude). Entregable del encargo O2-A.
"""
from __future__ import annotations
import os, sys, time, csv, itertools
import multiprocessing as mp

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import eigh

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)

from cs072_modulos.piezas.p_semilla_causal import malla_causal_atomos, layout_resortes
from null1_generar_ic import leer_ic_txt
from null4_verificar_invarianza_orden import adj_a_lista_aristas, construir_adj_en_orden
# la parametrización ya validada de los dos ejes -- se importa, no se reinventa
from cs073_factorial_qe_qt import adj_con_qE, orden_con_qT

RUTA_POOL = "/Users/alexis/phantom_cs073/bateria_n2000"
RUTA_IC_REAL = f"{RUTA_POOL}/ic_real/cosmogenesis_ic.txt"
D_CAUSAL, K_CAUSAL = 3, 4
SEED_EJES_CANONICO = 2000       # misma malla REAL canónica de toda la jerarquía
SEED_LAYOUT = 12345             # misma semilla de layout de toda la jerarquía
ITERS_LAYOUT = 100              # mismos iters de toda la jerarquía

Q_E_GRID = [1.0, 0.75, 0.5, 0.0]
Q_T_GRID = [1.0, 0.99, 0.9, 0.0]
N_SEMILLAS = 5
SEED_SWAP_BASE = 6000
SEED_ORDEN_BASE = 7000
SEEDS_LAYOUT_CONTROL = [12346, 12347, 12348]   # techo de decorrelación (control 1)

N_WORKERS = int(os.environ.get("N_WORKERS", "8"))
OUT_CSV = os.path.join(_HERE, "cs073_factorial_qe_qt_posicional.csv")
OUT_PNG = os.path.join(_HERE, "cs073_factorial_qe_qt_posicional.png")

# ---- estado global de cada worker (se llena una vez por proceso en `_init_worker`) ----
_G = {}


# =====================================================================================
# construcción de una celda (determinista dado (q_E, q_T, i_seed))
# =====================================================================================
def construir_adj_celda(q_E, q_T, i_seed, adj_real, n, pos_real):
    """Devuelve el `adj` de la celda + metadatos. Determinista: mismas entradas → mismo grafo, bit a
    bit (por eso cada worker puede reconstruir su celda sin que haya que serializar grafos)."""
    seed_swap = SEED_SWAP_BASE + i_seed
    adj_qe, aceptados, intentos, fs_nom, fs_int = adj_con_qE(adj_real, n, pos_real, q_E, seed_swap)
    lista_qe = adj_a_lista_aristas(adj_qe, n)
    seed_orden = SEED_ORDEN_BASE + i_seed * 10 + Q_T_GRID.index(q_T)
    orden, n_tocadas = orden_con_qT(len(lista_qe), q_T, seed_orden)
    adj_qt = construir_adj_en_orden(lista_qe, n, orden)
    # el conjunto de aristas NO puede cambiar por el barajado de ORDEN -- se verifica, no se asume
    assert sorted(adj_a_lista_aristas(adj_qt, n)) == sorted(lista_qe), (
        f"¡el conjunto de aristas cambió al aplicar q_T={q_T}! auditar antes de reportar")
    meta = dict(seed_swap=seed_swap, seed_orden=seed_orden, n_aristas=len(lista_qe),
                swap_aceptados=aceptados, swap_intentos=intentos,
                factor_swaps_nominal=round(fs_nom, 3), factor_swaps_int=fs_int,
                n_tocadas_orden=n_tocadas)
    return adj_qt, lista_qe, meta


# =====================================================================================
# observables posicionales (la justificación de cada uno está en el docstring del módulo)
# =====================================================================================
def observables_posicionales(pos, aristas, pos_ref, radio_giro_ref, sigma_w):
    """Todos los observables de un layout. `pos_ref` puede ser None (cuando se está calculando la
    propia referencia): en ese caso se omiten los tres de distancia a la referencia."""
    out = {}
    n = len(pos)

    # ---- (A) distancia a la configuración de referencia ----
    if pos_ref is not None:
        d = np.linalg.norm(pos - pos_ref, axis=1)
        rms = float(np.sqrt(np.mean(d ** 2)))
        out["rms_a_ref"] = round(rms, 6)
        out["rms_a_ref_rel"] = round(rms / radio_giro_ref, 6)
        out["max_a_ref"] = round(float(d.max()), 6)

    # ---- (B) estadística de la nube ----
    cen = pos.mean(axis=0)
    rel = pos - cen
    rg = float(np.sqrt(np.mean(np.sum(rel ** 2, axis=1))))
    out["radio_giro"] = round(rg, 6)
    cov = (rel.T @ rel) / n
    ev = np.linalg.eigvalsh(cov)          # ascendente: ev[0] <= ev[1] <= ev[2]
    out["anisotropia"] = round(float(np.sqrt(ev[2] / max(ev[0], 1e-300))), 5)

    arbol = cKDTree(pos)
    dk, _ = arbol.query(pos, k=9)          # k=1 es el propio punto
    d_nn = dk[:, 1]
    out["d_nn_media"] = round(float(d_nn.mean()), 6)
    out["d_nn_std"] = round(float(d_nn.std()), 6)
    out["d_knn8_media"] = round(float(dk[:, 8].mean()), 6)

    ii = np.array([e[0] for e in aristas]); jj = np.array([e[1] for e in aristas])
    l_ar = np.linalg.norm(pos[ii] - pos[jj], axis=1)
    out["long_arista_media"] = round(float(l_ar.mean()), 6)
    out["long_arista_std"] = round(float(l_ar.std()), 6)

    # ---- (C) espectro laplaciano PONDERADO POR DISTANCIA geométrica ----
    w_ij = np.exp(-(l_ar ** 2) / (2.0 * sigma_w ** 2))
    W = np.zeros((n, n))
    W[ii, jj] = w_ij
    W[jj, ii] = w_ij
    Lw = np.diag(W.sum(axis=1)) - W
    ew = eigh(Lw, eigvals_only=True)
    ew = np.clip(ew, 0.0, None); ew.sort()
    # lambda2 = primer autovalor por encima del núcleo de componentes conexas (el grafo de esta
    # jerarquía es conexo; se localiza igual el nº de ceros por robustez, con la misma tolerancia
    # relativa que usa el resto del arco)
    n_ceros = int(np.sum(ew < 1e-9 * max(ew[-1], 1.0)))
    n_ceros = max(n_ceros, 1)
    out["lambda2_w"] = round(float(ew[n_ceros]) if n_ceros < n else float("nan"), 6)
    out["lambda_max_w"] = round(float(ew[-1]), 5)
    out["mean_eig_w"] = round(float(ew.mean()), 6)
    out["std_eig_w"] = round(float(ew.std()), 6)
    out["n_ceros_w"] = n_ceros
    return out


# =====================================================================================
# workers
# =====================================================================================
def _init_worker(pos_ref, radio_giro_ref, sigma_w):
    dens_bar = np.load(f"{RUTA_POOL}/dens_bar.npy")
    n = len(dens_bar)
    _G["n"] = n
    _G["lado"] = float(n) ** (1.0 / 3.0)
    _G["pos_real"] = leer_ic_txt(RUTA_IC_REAL)[0]
    _G["adj_real"], _ = malla_causal_atomos(dens_bar, D=D_CAUSAL, k=K_CAUSAL, seed_ejes=SEED_EJES_CANONICO)
    _G["pos_ref"] = pos_ref
    _G["rg_ref"] = radio_giro_ref
    _G["sigma_w"] = sigma_w


def _celda(args):
    q_E, q_T, i_seed = args
    t0 = time.time()
    n, lado = _G["n"], _G["lado"]
    adj, aristas, meta = construir_adj_celda(q_E, q_T, i_seed, _G["adj_real"], n, _G["pos_real"])
    pos = layout_resortes(adj, n, lado=lado, iters=ITERS_LAYOUT, seed=SEED_LAYOUT)
    obs = observables_posicionales(pos, aristas, _G["pos_ref"], _G["rg_ref"], _G["sigma_w"])
    fila = dict(tipo="grilla", q_E=q_E, q_T=q_T, i_seed=i_seed, n=n, seed_layout=SEED_LAYOUT)
    fila.update(meta); fila.update(obs)
    fila["_tiempo_s"] = round(time.time() - t0, 2)
    return fila


def _control_techo(seed_layout):
    """Control 1: MISMO grafo de referencia (q_E=1, q_T=1), relajado desde otra nube inicial."""
    t0 = time.time()
    n, lado = _G["n"], _G["lado"]
    adj, aristas, meta = construir_adj_celda(1.0, 1.0, 0, _G["adj_real"], n, _G["pos_real"])
    pos = layout_resortes(adj, n, lado=lado, iters=ITERS_LAYOUT, seed=seed_layout)
    obs = observables_posicionales(pos, aristas, _G["pos_ref"], _G["rg_ref"], _G["sigma_w"])
    fila = dict(tipo="control_techo_seedlayout", q_E=1.0, q_T=1.0, i_seed=0, n=n, seed_layout=seed_layout)
    fila.update(meta); fila.update(obs)
    fila["_tiempo_s"] = round(time.time() - t0, 2)
    return fila


# =====================================================================================
# descomposición de varianza de dos vías (q_E × q_T, con réplicas)
# =====================================================================================
def anova_dos_vias(filas, col):
    """SS_total = SS_qE + SS_qT + SS_interaccion + SS_error, sobre el diseño balanceado
    5 (q_E) × 5 (q_T) × 5 (réplicas de semilla). No es un F-test formal: es la misma "descomposición
    simple de varianza" que reportó el factorial anterior (fracción de la suma de cuadrados que
    corresponde a cada eje), para que los dos informes sean comparables número a número. Se añaden
    los F y los grados de libertad porque son gratis y ayudan a leer si una fracción chica es o no
    distinguible del ruido entre semillas."""
    y = {}
    for f in filas:
        if f["tipo"] != "grilla":
            continue
        v = f.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        y[(f["q_E"], f["q_T"], f["i_seed"])] = float(v)
    A, B = Q_E_GRID, Q_T_GRID
    nrep = N_SEMILLAS
    M = np.array([[[y[(a, b, r)] for r in range(nrep)] for b in B] for a in A])  # (|A|,|B|,nrep)
    gran = M.mean()
    ss_tot = float(((M - gran) ** 2).sum())
    mA = M.mean(axis=(1, 2)); mB = M.mean(axis=(0, 2)); mAB = M.mean(axis=2)
    ss_A = float(len(B) * nrep * ((mA - gran) ** 2).sum())
    ss_B = float(len(A) * nrep * ((mB - gran) ** 2).sum())
    ss_AB = float(nrep * ((mAB - mA[:, None] - mB[None, :] + gran) ** 2).sum())
    ss_err = float(((M - mAB[:, :, None]) ** 2).sum())
    df_A, df_B = len(A) - 1, len(B) - 1
    df_AB, df_err = df_A * df_B, len(A) * len(B) * (nrep - 1)
    def frac(x):
        return 0.0 if ss_tot <= 0 else 100.0 * x / ss_tot
    ms_err = ss_err / df_err if df_err else float("nan")
    def F(ss, df):
        return (ss / df) / ms_err if ms_err > 0 else float("nan")
    return dict(col=col, ss_total=ss_tot,
                pct_qE=frac(ss_A), pct_qT=frac(ss_B), pct_inter=frac(ss_AB), pct_error=frac(ss_err),
                F_qE=F(ss_A, df_A), F_qT=F(ss_B, df_B), F_inter=F(ss_AB, df_AB),
                df=(df_A, df_B, df_AB, df_err))


OBS_COLS = ["rms_a_ref", "rms_a_ref_rel", "max_a_ref", "radio_giro", "anisotropia",
            "d_nn_media", "d_nn_std", "d_knn8_media", "long_arista_media", "long_arista_std",
            "lambda2_w", "lambda_max_w", "mean_eig_w", "std_eig_w"]


# =====================================================================================
def main():
    t_ini = time.time()
    print("=" * 104, flush=True)
    print("CS073 factorial q_E x q_T -- instrumento POSICIONAL (layout_resortes). SIN Phantom.", flush=True)
    print("=" * 104, flush=True)

    dens_bar = np.load(f"{RUTA_POOL}/dens_bar.npy")
    n = len(dens_bar); lado = float(n) ** (1.0 / 3.0)
    pos_real = leer_ic_txt(RUTA_IC_REAL)[0]
    adj_real, _ = malla_causal_atomos(dens_bar, D=D_CAUSAL, k=K_CAUSAL, seed_ejes=SEED_EJES_CANONICO)
    n_aristas_real = len(adj_a_lista_aristas(adj_real, n))
    print(f"pool: n={n}  lado={lado:.6f}  aristas malla REAL={n_aristas_real}", flush=True)

    # ---------- fase 1: configuración de REFERENCIA (q_E=1, q_T=1) ----------
    print("\n[fase 1] layout de la configuración de REFERENCIA (q_E=1.0, q_T=1.0)...", flush=True)
    t0 = time.time()
    adj_ref, aristas_ref, meta_ref = construir_adj_celda(1.0, 1.0, 0, adj_real, n, pos_real)
    pos_ref = layout_resortes(adj_ref, n, lado=lado, iters=ITERS_LAYOUT, seed=SEED_LAYOUT)
    cen = pos_ref.mean(axis=0)
    rg_ref = float(np.sqrt(np.mean(np.sum((pos_ref - cen) ** 2, axis=1))))
    ii = np.array([e[0] for e in aristas_ref]); jj = np.array([e[1] for e in aristas_ref])
    sigma_w = float(np.median(np.linalg.norm(pos_ref[ii] - pos_ref[jj], axis=1)))
    print(f"  referencia lista en {time.time()-t0:.1f}s -- radio_giro_ref={rg_ref:.6f}  "
          f"sigma_w (mediana long. arista)={sigma_w:.6f}", flush=True)

    # ---------- fase 2: grilla en paralelo ----------
    tareas = [(qE, qT, s) for qE in Q_E_GRID for s in range(N_SEMILLAS) for qT in Q_T_GRID]
    print(f"\n[fase 2] grilla {len(Q_E_GRID)}(q_E) x {len(Q_T_GRID)}(q_T) x {N_SEMILLAS}(semillas) = "
          f"{len(tareas)} celdas, + {len(SEEDS_LAYOUT_CONTROL)} controles de techo, "
          f"con {N_WORKERS} procesos", flush=True)
    filas = []
    ctx = mp.get_context("fork")
    with ctx.Pool(N_WORKERS, initializer=_init_worker, initargs=(pos_ref, rg_ref, sigma_w)) as pool:
        r_ctrl = pool.map_async(_control_techo, SEEDS_LAYOUT_CONTROL)
        hecho = 0
        for fila in pool.imap_unordered(_celda, tareas):
            filas.append(fila); hecho += 1
            print(f"  [{hecho}/{len(tareas)}] q_E={fila['q_E']:.2f} q_T={fila['q_T']:.2f} "
                  f"s={fila['i_seed']} | rms_a_ref={fila['rms_a_ref']:.4f} "
                  f"({100*fila['rms_a_ref_rel']:.1f}% de r_g) | lam2_w={fila['lambda2_w']:.5f} "
                  f"| d_nn={fila['d_nn_media']:.4f} ({fila['_tiempo_s']}s)", flush=True)
        filas_ctrl = r_ctrl.get()
    filas.extend(filas_ctrl)
    filas.sort(key=lambda f: (f["tipo"], f["q_E"], f["q_T"], f["i_seed"], f["seed_layout"]))

    # ---------- fase 3: chequeos de sanidad ----------
    print("\n[fase 3] chequeos de sanidad", flush=True)
    base = [f for f in filas if f["tipo"] == "grilla" and f["q_E"] == 1.0 and f["q_T"] == 1.0]
    peor = max(abs(f["rms_a_ref"]) for f in base)
    print(f"  determinismo (q_E=1,q_T=1, 5 semillas): peor rms_a_ref = {peor:.3e} "
          f"{'OK (0 exacto)' if peor == 0.0 else '<-- AUDITAR: hay azar no contabilizado'}", flush=True)
    for f in filas_ctrl:
        print(f"  TECHO seed_layout={f['seed_layout']}: rms_a_ref={f['rms_a_ref']:.4f} "
              f"({100*f['rms_a_ref_rel']:.1f}% de r_g)  max={f['max_a_ref']:.3f}", flush=True)

    # ---------- fase 4: CSV ----------
    campos = []
    for f in filas:
        for k in f:
            if k not in campos:
                campos.append(k)
    with open(OUT_CSV, "w", newline="") as fcsv:
        wr = csv.DictWriter(fcsv, fieldnames=campos, restval="")
        wr.writeheader()
        for f in filas:
            wr.writerow(f)
    print(f"\nCSV -> {OUT_CSV}", flush=True)

    # ---------- fase 5: descomposición de varianza ----------
    print("\n" + "=" * 104, flush=True)
    print("DESCOMPOSICIÓN DE VARIANZA (2 vías con réplicas) -- % de la suma de cuadrados total", flush=True)
    print("=" * 104, flush=True)
    print(f"{'observable':<20}{'SS_total':>12}{'q_E %':>9}{'q_T %':>9}{'inter %':>9}{'error %':>9}"
          f"{'F_qE':>10}{'F_qT':>10}{'F_int':>9}", flush=True)
    resumen = []
    for col in OBS_COLS:
        r = anova_dos_vias(filas, col)
        if r is None:
            continue
        resumen.append(r)
        print(f"{col:<20}{r['ss_total']:>12.4g}{r['pct_qE']:>9.2f}{r['pct_qT']:>9.2f}"
              f"{r['pct_inter']:>9.2f}{r['pct_error']:>9.2f}{r['F_qE']:>10.1f}{r['F_qT']:>10.1f}"
              f"{r['F_inter']:>9.1f}", flush=True)
    with open(OUT_CSV.replace(".csv", "_anova.csv"), "w", newline="") as fcsv:
        wr = csv.DictWriter(fcsv, fieldnames=["col", "ss_total", "pct_qE", "pct_qT", "pct_inter",
                                              "pct_error", "F_qE", "F_qT", "F_inter", "df"])
        wr.writeheader()
        for r in resumen:
            wr.writerow(r)

    # ---------- fase 6: tablas marginales (¿escalón o gradiente?) ----------
    def marg(eje, otro, col):
        print(f"\n  {col} -- medias por nivel de {eje} (colapsando {otro} y semillas):", flush=True)
        niveles = Q_E_GRID if eje == "q_E" else Q_T_GRID
        for v in niveles:
            vals = [f[col] for f in filas if f["tipo"] == "grilla" and f[eje] == v]
            print(f"    {eje}={v:<5} media={np.mean(vals):.6f}  std={np.std(vals):.6f}  "
                  f"min={np.min(vals):.6f}  max={np.max(vals):.6f}", flush=True)

    print("\n" + "=" * 104, flush=True)
    print("PERFILES MARGINALES", flush=True)
    print("=" * 104, flush=True)
    for col in ["rms_a_ref_rel", "lambda2_w", "d_nn_media", "long_arista_media"]:
        marg("q_E", "q_T", col)
        marg("q_T", "q_E", col)

    # la pregunta central, aislada: q_T con el grafo INTACTO (q_E=1) vs q_E con el orden INTACTO (q_T=1)
    print("\n" + "=" * 104, flush=True)
    print("EJE PURO: q_T con grafo intacto (q_E=1.0)  |  q_E con orden intacto (q_T=1.0)", flush=True)
    print("=" * 104, flush=True)
    for qT in Q_T_GRID:
        vals = [f["rms_a_ref_rel"] for f in filas
                if f["tipo"] == "grilla" and f["q_E"] == 1.0 and f["q_T"] == qT]
        print(f"  q_E=1.0, q_T={qT:<5} rms_a_ref_rel media={np.mean(vals)*100:6.2f}%  "
              f"(min {np.min(vals)*100:.2f}% / max {np.max(vals)*100:.2f}%)", flush=True)
    for qE in Q_E_GRID:
        vals = [f["rms_a_ref_rel"] for f in filas
                if f["tipo"] == "grilla" and f["q_T"] == 1.0 and f["q_E"] == qE]
        print(f"  q_T=1.0, q_E={qE:<5} rms_a_ref_rel media={np.mean(vals)*100:6.2f}%  "
              f"(min {np.min(vals)*100:.2f}% / max {np.max(vals)*100:.2f}%)", flush=True)
    techo = np.mean([f["rms_a_ref_rel"] for f in filas_ctrl])
    print(f"\n  TECHO (mismo grafo, otra nube inicial): rms_a_ref_rel = {techo*100:.2f}%", flush=True)

    # ---------- fase 7: figura ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        for ax, col, tit in zip(axes,
                                ["rms_a_ref_rel", "lambda2_w", "d_nn_media"],
                                ["distancia RMS a la referencia (frac. del radio de giro)",
                                 "lambda2 del laplaciano PONDERADO por distancia",
                                 "distancia media al vecino más cercano"]):
            for qE in Q_E_GRID:
                ys = [np.mean([f[col] for f in filas if f["tipo"] == "grilla"
                               and f["q_E"] == qE and f["q_T"] == qT]) for qT in Q_T_GRID]
                ax.plot(range(len(Q_T_GRID)), ys, "o-", label=f"q_E={qE}")
            ax.set_xticks(range(len(Q_T_GRID)))
            ax.set_xticklabels([str(v) for v in Q_T_GRID])
            ax.set_xlabel("q_T (fracción del orden de inserción conservada)")
            ax.set_title(tit, fontsize=9)
            ax.grid(alpha=0.3)
        axes[0].axhline(techo, ls="--", c="k", lw=1, label="techo (otra nube inicial)")
        axes[0].legend(fontsize=7)
        fig.suptitle("CS073 factorial q_E x q_T con instrumento POSICIONAL (layout_resortes, N=2000, "
                     "5 semillas/celda) -- sin Phantom", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT_PNG, dpi=130)
        print(f"\nPNG -> {OUT_PNG}", flush=True)
    except Exception as e:
        print(f"\n(figura omitida: {e})", flush=True)

    print(f"\nCOMPLETO en {time.time()-t_ini:.1f}s ({len(filas)} filas)", flush=True)


if __name__ == "__main__":
    main()

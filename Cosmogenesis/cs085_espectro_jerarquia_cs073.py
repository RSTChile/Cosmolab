"""
cs085_espectro_jerarquia_cs073.py — el diagnóstico espectral de CS084 (¿se oye la forma del tejido?)
aplicado a la jerarquía de 6 controles de CS073 (NULL-0..NULL-5, línea de Phantom/gravedad), no a la
línea de CS066 donde nació.

POR QUÉ ESTE SCRIPT (encargo de Alexis, 9-ago-2026): CS084 encontró que el espectro COMPLETO del
laplaciano de grafo (L=D-A) separa con una limpieza que el diámetro nunca logró (λ_max y dispersión de
eigenvalues 3-17× mayores en el tejido real que en sus controles, sin solape en 5 semillas) — pero eso
se midió sobre el tejido puro de CS066 (proceso066, sin gravedad, sin Phantom). CS073 es una línea
DISTINTA y más antigua: ahí SÍ se sabe con precisión, escalón por escalón, qué preserva y qué destruye
cada control, y SÍ se corrió Phantom de verdad para medir formación de sumideros. La pregunta: ¿el
mismo instrumento espectral, aplicado a ESA jerarquía (donde ya hay números de sumideros con los que
comparar), escala de forma gradual con cuánta estructura real se preserva, y correlaciona con esos
números de sumideros ya medidos?

QUÉ SE REUSA, SIN TOCAR NINGÚN ARCHIVO CONGELADO (sólo import/lectura):
  - `cs084_espectro_laplaciano.py`: las funciones de CÁLCULO del diagnóstico -- `dimension_espectral`
    (núcleo de calor), `unfolding_local` + `cdf_goe` + `estadisticas_espaciado` (espaciado de niveles),
    `T_GRID`. Se reusan tal cual; lo único que cambia es la FUENTE del grafo (ahí era `proceso066` de
    CS066, acá es la malla causal de CS073) y la construcción densa del laplaciano se reimplementa aquí
    en 10 líneas porque el formato de adyacencia de CS073 es un dict de sets (no una lista como en
    cs064_smoke.adj_sparse) -- misma matemática (L=D-A, eigh denso), sin reescribir el diagnóstico.
  - `null4_generar_ic.py` / `null4_verificar_invarianza_orden.py`: cómo se reconstruye la malla causal
    REAL exacta (N=2000, `malla_causal_atomos`, D=3/k=4/seed_ejes=2000) y cómo se rebaraja el ORDEN de
    inserción de aristas preservando el conjunto exacto (para el chequeo de sanidad de NULL-4).
  - `null3_generar_ic.py` / `null3_investigacion_preliminar.py`: `barajar_aristas_preservando_longitud`
    (double-edge-swap con filtro de longitud, el grafo de NULL-3).
  - `grafo_random_layout_generar_ic.py`: `generar_grafo_erdos_renyi` + `contar_aristas_malla_real` (el
    control Erdős-Rényi puro, sin relación con REAL).
  - `null1_generar_ic.py`: `leer_ic_txt` (para cargar las posiciones REAL ya en disco, que NULL-3
    necesita como referencia de longitud).

QUÉ NO SE HACE (documentado como limitación, no forzado): NULL-1 (radio exacto/ángulo aleatorio) y
NULL-2 (Zel'dovich/espectro de potencia) NO tienen un grafo de vecindad de fondo -- son construcciones
puramente estadísticas sobre posiciones, nunca pasan por `layout_resortes` ni por ningún `adj`. No se
inventa un grafo artificial para ellos: se documentan explícitamente como "N/A, sin grafo". NULL-4 y
NULL-5 SÍ tienen grafo, pero es -- por construcción ya verificada en `NULL4_resultado_CS.md` y
`NULL5_resultado_CS.md` -- el MISMO conjunto exacto de aristas que REAL (NULL-4 sólo cambia el orden de
inserción antes del layout; NULL-5 ni siquiera toca la adyacencia, sólo permuta qué nodo ocupa cuál
posición en el archivo de Phantom). Como el laplaciano L=D-A depende sólo del CONJUNTO de aristas (no
del orden en que se insertaron, no de qué "nombre" se le puso a cada nodo más allá de su índice), el
espectro de NULL-4 y NULL-5 tiene que ser IDÉNTICO al de REAL por construcción matemática, no por un
hallazgo empírico nuevo -- este script lo CONFIRMA numéricamente (no lo asume) y lo reporta como chequeo
de sanidad, no como resultado nuevo.

NOTA SOBRE SEMILLAS DE REAL (honestidad, pedida explícitamente por Alexis): la malla causal REAL de
CS073 usa `seed_ejes=2000` FIJO en toda la jerarquía -- las "6 semillas REAL" de sumideros que ya existen
en disco (`bateria_real_extra_n2000`, etc.) sólo variaban `seed_layout` (la física/relajación de
resortes), NUNCA la TOPOLOGÍA del grafo -- comparten la MISMA malla causal, por lo tanto el MISMO
espectro exacto. Para este diagnóstico (que depende sólo de la topología, no del layout físico) eso
significa n=1 real para el grafo canónico. Para tener variabilidad GENUINA se generan además 4 mallas
REALES adicionales con `seed_ejes` distinto (mismo método `malla_causal_atomos`, D=3/k=4, sólo cambia la
semilla del sembrado de ejes) -- son grafos "REAL" en el mismo sentido metodológico (mismo mecanismo de
construcción), pero NO son la malla canónica que corrió Phantom en el resto de la jerarquía; se marcan
aparte en la tabla.

No se declara cierre ni veredicto -- sólo se reportan números. La lectura final es de Alexis.

Codea/ejecuta: CC (Claude).
"""
from __future__ import annotations
import os, sys, time, csv
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)

# ---- diagnóstico espectral: funciones de CÁLCULO reusadas tal cual de CS084, sin reescribir método ----
from cs084_espectro_laplaciano import dimension_espectral, unfolding_local, estadisticas_espaciado, T_GRID

# ---- piezas de la jerarquía CS073, todas ya congeladas/validadas, sólo importadas ----
from cs072_modulos.piezas.p_semilla_causal import malla_causal_atomos
from null1_generar_ic import leer_ic_txt
from null3_investigacion_preliminar import barajar_aristas_preservando_longitud, aristas_de
from null4_verificar_invarianza_orden import adj_a_lista_aristas, construir_adj_en_orden
from grafo_random_layout_generar_ic import generar_grafo_erdos_renyi, contar_aristas_malla_real

RUTA_POOL = "/Users/alexis/phantom_cs073/bateria_n2000"
RUTA_IC_REAL = f"{RUTA_POOL}/ic_real/cosmogenesis_ic.txt"
D_CAUSAL, K_CAUSAL = 3, 4
SEED_EJES_CANONICO = 2000          # la malla REAL exacta que corrió Phantom en toda la jerarquía CS073
SEEDS_EJES_REAL_EXTRA = [2001, 2002, 2003, 2004]  # variabilidad GENUINA adicional (mismo método, seed distinta)
SEEDS_NULL3 = [501, 502, 503, 504, 505]            # double-edge-swap + filtro de longitud, tol=0.2
SEEDS_RANDOM = [9001, 9002, 9003, 9004, 9005]      # Erdős-Rényi independiente
SEED_REORDEN_NULL4 = 4001                          # un reorden de inserción para el chequeo de sanidad
OUT_CSV = os.path.join(_HERE, "cs085_espectro_jerarquia_cs073.csv")


# ============================ construcción del laplaciano denso desde un adj tipo CS073 (dict de sets) ============================
def laplaciano_denso_desde_adj(adj, n):
    """Misma matemática que `laplaciano_denso` de CS084 (L=D-A, eigh denso completo, clip a >=0 por
    ruido numérico) pero partiendo de un `adj` en formato dict-de-sets (el formato nativo de
    `malla_causal_atomos`/`barajar_aristas_preservando_longitud`/`generar_grafo_erdos_renyi`), en vez de
    la lista que produce `construir_sustrato` de CS080/CS066. No se reescribe el diagnóstico -- sólo la
    conversión adj->matriz dispersa, que en CS084 vivía en `cs064_smoke.adj_sparse` (que espera una LISTA
    indexable, no un dict, así que no es reusable tal cual aquí)."""
    rows, cols = [], []
    for i in range(n):
        for j in adj.get(i, ()):
            rows.append(i); cols.append(j)
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    n_comp, labels = connected_components(A, directed=False)
    giant = float(np.max(np.bincount(labels))) / n
    deg = np.asarray(A.sum(axis=1)).ravel()
    L = sparse.diags(deg) - A
    w = eigh(L.toarray(), eigvals_only=True)
    w = np.clip(w, 0.0, None)
    w.sort()
    return w, int(n_comp), giant


def diagnostico_fila(grupo, etiqueta, seed, adj, n, extra=None):
    """Corre los 3 diagnósticos de CS084 (forma espectral, dimensión por núcleo de calor, espaciado de
    niveles) sobre UN grafo y arma una fila de tabla -- misma estructura de columnas que `cs084...csv`
    para poder comparar directo."""
    t0 = time.time()
    w, n_comp, giant = laplaciano_denso_desde_adj(adj, n)
    n_aristas = int(round(float(np.sum([len(adj.get(i, ())) for i in range(n)])) / 2))
    lam2 = float(w[n_comp]) if n_comp < len(w) else float("nan")
    d_s_curve, _tr = dimension_espectral(w)
    s = unfolding_local(w, n_comp)
    stats_esp = estadisticas_espaciado(s)
    fila = dict(grupo=grupo, etiqueta=etiqueta, seed=seed, n=n, n_aristas=n_aristas,
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
    print("CS085 — espectro del laplaciano aplicado a la jerarquía CS073 (NULL-0..NULL-5)", flush=True)
    print("=" * 100, flush=True)

    dens_bar = np.load(f"{RUTA_POOL}/dens_bar.npy")
    n = len(dens_bar)
    pos_real = leer_ic_txt(RUTA_IC_REAL)[0]
    print(f"pool cargado: n={n}  (dens_bar.npy, pos_real de ic_real/cosmogenesis_ic.txt)\n", flush=True)

    filas = []

    # -------------------- REAL (canónica seed_ejes=2000 + 4 semillas adicionales genuinas) --------------------
    print("[REAL] reconstruyendo malla(s) causal(es) real(es)...", flush=True)
    adj_real_canon, _m = malla_causal_atomos(dens_bar, D=D_CAUSAL, k=K_CAUSAL, seed_ejes=SEED_EJES_CANONICO)
    fila = diagnostico_fila("REAL", "REAL (canónica, seed_ejes=2000 -- la que corrió Phantom en CS073)",
                             SEED_EJES_CANONICO, adj_real_canon, n, extra=dict(canonica=True))
    filas.append(fila)
    print(f"  seed_ejes={SEED_EJES_CANONICO} (CANÓNICA): n_aristas={fila['n_aristas']} "
          f"lambda_max={fila['lambda_max']} std_eig={fila['std_eig']} ({fila['_tiempo_s']}s)", flush=True)
    for se in SEEDS_EJES_REAL_EXTRA:
        adj_r, _m = malla_causal_atomos(dens_bar, D=D_CAUSAL, k=K_CAUSAL, seed_ejes=se)
        fila = diagnostico_fila("REAL", f"REAL (variante genuina, seed_ejes={se})", se, adj_r, n,
                                 extra=dict(canonica=False))
        filas.append(fila)
        print(f"  seed_ejes={se} (variante): n_aristas={fila['n_aristas']} "
              f"lambda_max={fila['lambda_max']} std_eig={fila['std_eig']} ({fila['_tiempo_s']}s)", flush=True)

    # -------------------- NULL-3 (double-edge-swap + filtro de longitud, sobre la malla canónica) --------------------
    print("\n[NULL-3] double-edge-swap + filtro de longitud (tol_relativa=0.2) sobre malla canónica...", flush=True)
    edges_real_canon = set(tuple(e) for e in aristas_de(adj_real_canon, n))
    for sn in SEEDS_NULL3:
        adj_n3, aceptados, intentos = barajar_aristas_preservando_longitud(
            adj_real_canon, n, pos_real, seed=sn, factor_swaps=10, tol_relativa=0.2)
        edges_n3 = set(tuple(e) for e in aristas_de(adj_n3, n))
        pct_distintas = 100.0 * len(edges_n3 - edges_real_canon) / len(edges_real_canon)
        fila = diagnostico_fila("NULL-3", f"NULL-3 (swap+filtro long., seed={sn})", sn, adj_n3, n,
                                 extra=dict(pct_aristas_distintas=round(pct_distintas, 2),
                                            swap_aceptados=aceptados, swap_intentos=intentos))
        filas.append(fila)
        print(f"  seed={sn}: n_aristas={fila['n_aristas']} aristas_distintas={pct_distintas:.1f}% "
              f"lambda_max={fila['lambda_max']} std_eig={fila['std_eig']} ({fila['_tiempo_s']}s)", flush=True)

    # -------------------- Erdős-Rényi random (independiente de REAL) --------------------
    print("\n[RANDOM] Erdős-Rényi G(n,m) independiente (m = nº aristas de la malla causal REAL canónica)...",
          flush=True)
    n_chk, n_aristas_obj, _adj_chk = contar_aristas_malla_real(dens_bar, D=D_CAUSAL, k=K_CAUSAL,
                                                                 seed_ejes=SEED_EJES_CANONICO)
    assert n_chk == n
    for sr in SEEDS_RANDOM:
        adj_er, _edge_set, _intentos = generar_grafo_erdos_renyi(n, n_aristas_obj, seed=sr)
        fila = diagnostico_fila("RANDOM", f"Erdős-Rényi (seed={sr})", sr, adj_er, n)
        filas.append(fila)
        print(f"  seed={sr}: n_aristas={fila['n_aristas']} "
              f"lambda_max={fila['lambda_max']} std_eig={fila['std_eig']} ({fila['_tiempo_s']}s)", flush=True)

    # -------------------- NULL-4: chequeo de sanidad (topología 100% idéntica a REAL, orden rebarajado) --------------------
    print("\n[NULL-4] chequeo de sanidad -- topología 100% idéntica a REAL, sólo cambia el orden de "
          "inserción de aristas antes del layout (que este diagnóstico ni siquiera usa)...", flush=True)
    lista_real = adj_a_lista_aristas(adj_real_canon, n)
    rng4 = np.random.default_rng(SEED_REORDEN_NULL4)
    orden4 = rng4.permutation(len(lista_real)).tolist()
    adj_null4 = construir_adj_en_orden(lista_real, n, orden4)
    assert sorted(adj_a_lista_aristas(adj_null4, n)) == sorted(lista_real), (
        "¡el conjunto de aristas de NULL-4 difiere del de REAL! -- no debería pasar, construir_adj_en_orden "
        "preserva el conjunto exacto.")
    w_null4, n_comp4, giant4 = laplaciano_denso_desde_adj(adj_null4, n)
    w_real_canon = filas[0]["_eigvals"]
    diff_null4 = float(np.max(np.abs(w_null4 - w_real_canon)))
    print(f"  conjunto de aristas verificado idéntico a REAL (assert paso). "
          f"max|Δeigenvalue| NULL-4 vs REAL canónica = {diff_null4:.3e}", flush=True)

    # -------------------- NULL-5: chequeo de sanidad analítico (ni siquiera toca la adyacencia) --------------------
    print("\n[NULL-5] chequeo de sanidad -- NULL-5 permuta qué NODO ocupa cuál POSICIÓN en el archivo de "
          "Phantom, pero (ver NULL5_resultado_CS.md, Paso 0) nunca vuelve a tocar `adj`: la adyacencia "
          "por índice de nodo es LITERALMENTE la misma matriz que REAL, no una reconstrucción aproximada "
          "-- diff=0.0 exacto por definición, no hay nada que computar de nuevo.", flush=True)

    # -------------------- NULL-1 / NULL-2: sin grafo de fondo, documentado explícitamente --------------------
    print("\n[NULL-1 / NULL-2] NO tienen grafo de vecindad de fondo -- NULL-1 (radio exacto/ángulo "
          "aleatorio) y NULL-2 (Zel'dovich/espectro de potencia) actúan directo sobre posiciones, nunca "
          "pasan por `malla_causal_atomos` ni por `layout_resortes`. N/A para este diagnóstico -- no se "
          "fuerza un grafo artificial.", flush=True)

    # -------------------- guarda CSV (sin la columna de eigenvalues crudos, sólo el resumen) --------------------
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

    print(f"\nNULL-4 sanidad: max|Δeigenvalue| vs REAL canónica = {diff_null4:.3e}", flush=True)
    print(f"COMPLETO en {time.time()-t_inicio:.1f}s", flush=True)


if __name__ == "__main__":
    main()

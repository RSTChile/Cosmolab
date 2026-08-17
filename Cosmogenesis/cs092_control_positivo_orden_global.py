"""
CS092 — CONTROL POSITIVO del instrumento de "orden global" (nodo C-N2.6.4)
================================================================================================
PREGUNTA (no es rescatar el nodo): los tres jueces con los que el arco CS067B / CS068-MundoB /
CS069B / Fase III declararon "no hay organización a gran escala", ¿son capaces de DETECTAR orden
global cuando SABEMOS que lo hay?

LOS TRES JUECES — se importan TAL CUAL, no se reimplementa ninguno:
  J-A  Q.cedazo_pi(D,N,rng)          -> pi_cv    (bajo = "pi se congela" = metrico plano)
  J-B  _pendiente_loglog(logN,logdiam) -> pendiente (>0.3 = "hay lejos real"; umbral de CS068 2b)
  J-B' magnitud absoluta del diametro a N fijo (el juez que CS068 declaro DECISIVO)
  J-C  Q.juez_gap_espectral(D,N)     -> n_ejes + pico_medio>0.85 (certificado)

DOS VIAS DE APLICACION (el observable no se modifica en ninguna):
  Via Q: Q.brazo_completo(adj,N,rng) -> D_q (integral de camino) -> J-A, J-B, J-C  == cs069_tanda
  Via M: distancias BFS del grafo    -> D_bfs                    -> J-A, J-C ; J-B con H._diam_robusto
La via M separa "el observable es ciego" de "la caneria cuantica que lo alimenta es ciega".

SUSTRATOS: positivos con orden global conocido (anillo 1D, reticula 2D/3D/4D, 2D anisotropa,
grafo en capas con flujo sembrado) y negativos (Erdos-Renyi, barajado de grados del real, real).

Pre-registro completo (criterios de deteccion, guardas) en:
  CONTROL_POSITIVO_orden_global_CN264_CS.md  -- escrito ANTES de correr esto.

Codea/ejecuta: CC. No declara cierre.
"""
from __future__ import annotations
import os, sys, time, json, math, itertools
from collections import deque
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs069_quantum_graph as Q          # J-A (cedazo_pi), J-C (juez_gap_espectral), D_q
import cs067_habitacion_completa as H    # _diam_robusto, cuenta_ejes_gap, picado_por_nodo
import cs068_paso2_mundo_ab as P2        # _double_edge_swap (barajado de grados)
import cs068_inflacion_estirar_enfriar as E   # _sustrato (el grafo REAL del arco)
import cs090_diam_corregido as C90       # diam_gigante (GUARDA 3: componente gigante, no el nodo 0)

RNG = np.random.default_rng

# ---- criterios de deteccion PRE-REGISTRADOS (no se tocan) ----
UMBRAL_PENDIENTE = 0.3      # J-B, literal de cs068_paso2b_diametro.PENDIENTE_UMBRAL
UMBRAL_PI_CV     = 0.5      # J-A, "menos de la mitad de la banda 1.0-1.1 del sustrato real"
UMBRAL_PICO      = 0.85     # J-C, literal de Q.juez_gap_espectral
DIAM_SW_REF_2500 = 13.0     # J-B', referencia small-world de CS068 paso2b a N=2500
FACTOR_JB_PRIMA  = 3.0      # J-B' detecta si diam(N grande) >= 3x la referencia small-world


# ================================ SUSTRATOS ================================
def _vacia(N):
    return [set() for _ in range(N)]

def _add(adj, i, j):
    if i != j:
        adj[i].add(j); adj[j].add(i)


def sub_anillo1d(N, seed):
    """Anillo periodico 1D: orden global maximo, dimension 1. diam = N/2."""
    adj = _vacia(N)
    for i in range(N):
        _add(adj, i, (i + 1) % N)
    return adj, N


def sub_reticula2d(N_obj, seed):
    """Retícula cuadrada periodica (toro). Orden global evidente, d=2. diam ~ L = sqrt(N)."""
    L = int(round(math.sqrt(N_obj))); N = L * L
    adj = _vacia(N)
    idx = lambda x, y: (x % L) * L + (y % L)
    for x in range(L):
        for y in range(L):
            _add(adj, idx(x, y), idx(x + 1, y))
            _add(adj, idx(x, y), idx(x, y + 1))
    return adj, N


def sub_reticula3d(N_obj, seed):
    """Retícula cubica periodica, d=3. diam ~ N^(1/3)."""
    L = int(round(N_obj ** (1 / 3.0))); N = L ** 3
    adj = _vacia(N)
    idx = lambda x, y, z: ((x % L) * L + (y % L)) * L + (z % L)
    for x in range(L):
        for y in range(L):
            for z in range(L):
                _add(adj, idx(x, y, z), idx(x + 1, y, z))
                _add(adj, idx(x, y, z), idx(x, y + 1, z))
                _add(adj, idx(x, y, z), idx(x, y, z + 1))
    return adj, N


def sub_reticula4d(N_obj, seed):
    """Retícula hipercubica periodica, d=4. diam ~ N^(1/4) -> pendiente esperada 0.25 < 0.3.
    Se incluye a proposito para probar la GUARDA 1 (¿el umbral 0.3 es alcanzable en d alto?)."""
    L = int(round(N_obj ** 0.25)); N = L ** 4
    adj = _vacia(N)
    def idx(a, b, c, d):
        return (((a % L) * L + (b % L)) * L + (c % L)) * L + (d % L)
    for a in range(L):
        for b in range(L):
            for c in range(L):
                for d in range(L):
                    _add(adj, idx(a, b, c, d), idx(a + 1, b, c, d))
                    _add(adj, idx(a, b, c, d), idx(a, b + 1, c, d))
                    _add(adj, idx(a, b, c, d), idx(a, b, c + 1, d))
                    _add(adj, idx(a, b, c, d), idx(a, b, c, d + 1))
    return adj, N


def sub_aniso2d(N_obj, seed, p_y=0.25):
    """Retícula 2D con DIRECCION GLOBAL sembrada a mano: los enlaces en x siempre estan, los enlaces
    en y sobreviven solo con p_y. El tejido queda estirado a lo largo de x -- anisotropia macroscopica
    inequivoca (no es 'orden por construccion trivial': la topologia sigue siendo 2D, lo que se siembra
    es una DIRECCION preferente)."""
    L = int(round(math.sqrt(N_obj))); N = L * L
    rng = RNG(seed)
    adj = _vacia(N)
    idx = lambda x, y: (x % L) * L + (y % L)
    for x in range(L):
        for y in range(L):
            _add(adj, idx(x, y), idx(x + 1, y))
            if rng.random() < p_y:
                _add(adj, idx(x, y), idx(x, y + 1))
    return adj, N


def sub_flujo_capas(N_obj, seed, k_out=3):
    """Grafo en CAPAS con flujo neto sembrado a proposito: capa l -> capa l+1 unicamente (sin retorno,
    sin atajos). El indice de capa ES un gradiente global impuesto a mano. diam ~ n_capas ~ sqrt(N)."""
    L = int(round(math.sqrt(N_obj))); M = L; N = L * M
    rng = RNG(seed)
    adj = _vacia(N)
    for l in range(L - 1):
        for m in range(M):
            i = l * M + m
            for _ in range(k_out):
                j = (l + 1) * M + int(rng.integers(0, M))
                _add(adj, i, j)
    return adj, N


def sub_er(N_obj, seed, k_medio=6):
    """Erdos-Renyi: control negativo absoluto, sin ninguna estructura."""
    N = int(N_obj)
    rng = RNG(seed)
    adj = _vacia(N)
    n_edges = int(k_medio * N / 2)
    for _ in range(n_edges):
        i, j = int(rng.integers(0, N)), int(rng.integers(0, N))
        _add(adj, i, j)
    return adj, N


_CACHE_REAL = {}
def sub_real(N_obj, seed):
    """El sustrato del arco: motor CS067 'completo' via E._sustrato (lo que CS068/CS069 midieron)."""
    key = (int(N_obj), int(seed))
    if key not in _CACHE_REAL:
        _CACHE_REAL[key] = E._sustrato(int(N_obj), int(seed))
    return [set(s) for s in _CACHE_REAL[key]], int(N_obj)


def sub_real_barajado(N_obj, seed):
    """Barajado con GRADOS PRESERVADOS del real (double edge swap). GUARDA 4: se verifica aparte que
    destruye algo medible (clustering/triangulos) y que NO es isomorfo al real."""
    adj, N = sub_real(N_obj, seed)
    n_edges = sum(len(a) for a in adj) // 2
    adj2 = P2._double_edge_swap(adj, N, 10 * n_edges, RNG(seed + 777))
    return adj2, N


SUSTRATOS = {
    # nombre            : (funcion, escalera de N objetivo, orden_global_conocido)
    "anillo1d":     (sub_anillo1d,     [900, 1600, 2500], True),
    "reticula2d":   (sub_reticula2d,   [900, 1600, 2500], True),
    "reticula3d":   (sub_reticula3d,   [729, 1331, 2197], True),
    "reticula4d":   (sub_reticula4d,   [625, 1296, 2401], True),
    "aniso2d":      (sub_aniso2d,      [900, 1600, 2500], True),
    "flujo_capas":  (sub_flujo_capas,  [900, 1600, 2500], True),
    "er":           (sub_er,           [900, 1600, 2500], False),
    "real":         (sub_real,         [900, 1500, 2500], False),
    "real_barajado": (sub_real_barajado, [900, 1500, 2500], False),
}


# ================================ MEDICION ================================
def _dist_bfs(adj, N):
    """Matriz de distancias BFS (todos los pares) via scipy. Pares no alcanzables -> NaN, igual que
    Q._K_y_Dq marca los pares fuera de alcance."""
    filas, cols = [], []
    for i in range(N):
        for j in adj[i]:
            filas.append(i); cols.append(j)
    A = sp.csr_matrix((np.ones(len(filas)), (filas, cols)), shape=(N, N))
    D = shortest_path(A, method="D", unweighted=True, directed=False)
    D[np.isinf(D)] = np.nan
    np.fill_diagonal(D, 0.0)
    return D


def _clustering_global(adj, N):
    """Coeficiente de clustering medio (para la GUARDA 4)."""
    cs = []
    for i in range(N):
        v = list(adj[i]); k = len(v)
        if k < 2:
            continue
        t = sum(1 for a, b in itertools.combinations(v, 2) if b in adj[a])
        cs.append(2.0 * t / (k * (k - 1)))
    return float(np.mean(cs)) if cs else 0.0


def _n_triangulos(adj, N):
    t = 0
    for i in range(N):
        for a, b in itertools.combinations(sorted(adj[i]), 2):
            if b in adj[a]:
                t += 1
    return t // 3


def _mide_un_caso(nombre, N_obj, seed, con_via_q=True):
    fn, _, _ = SUSTRATOS[nombre]
    t0 = time.time()
    adj, N = fn(N_obj, seed)
    out = dict(sustrato=nombre, N_objetivo=N_obj, N=N, seed=seed)
    out["n_aristas"] = sum(len(a) for a in adj) // 2
    out["grado_medio"] = 2.0 * out["n_aristas"] / N

    # ---------- VIA M: observables sobre la metrica desnuda (BFS) ----------
    D = _dist_bfs(adj, N)
    piM, cvM = Q.cedazo_pi(D, N, RNG(seed + 3))
    out["M_pi_media"], out["M_pi_cv"] = piM, cvM
    jcM = Q.juez_gap_espectral(D, N)
    out["M_n_ejes"] = jcM["n_ejes"]; out["M_PR"] = jcM["PR"]
    out["M_gap_interno"] = jcM["gap_interno"]; out["M_pico_medio"] = jcM["pico_medio"]
    out["M_certificado"] = bool(jcM["certificado"])
    out["M_diam_robusto"] = H._diam_robusto(adj, N, RNG(seed + 4))     # el usado por CS068 2b
    out["M_diam_gigante"] = float(C90.diam_gigante(adj, N))            # GUARDA 3
    fin = D[np.isfinite(D)]
    out["M_frac_alcanzable"] = float(len(fin) / (N * N))

    # ---------- VIA Q: el instrumento completo, tal cual cs069_tanda ----------
    if con_via_q:
        Dq = Q.brazo_completo(adj, N, RNG(seed + 1))
        out["Q_diam"] = Q.diam_q_robusto(Dq, N, RNG(seed + 2))
        piQ, cvQ = Q.cedazo_pi(Dq, N, RNG(seed + 3))
        out["Q_pi_media"], out["Q_pi_cv"] = piQ, cvQ
        jcQ = Q.juez_gap_espectral(Dq, N)
        out["Q_n_ejes"] = jcQ["n_ejes"]; out["Q_PR"] = jcQ["PR"]
        out["Q_gap_interno"] = jcQ["gap_interno"]; out["Q_pico_medio"] = jcQ["pico_medio"]
        out["Q_certificado"] = bool(jcQ["certificado"])
        del Dq
    del D
    out["seg"] = time.time() - t0
    return out


# ================================ GUARDA 1: identidad algebraica de J-A ================================
def guarda1_perfil_pi(adjs_por_d, salida_csv):
    """pi_local(r) = |S(r)|/(2r) medido cascaron por cascaron sobre la metrica BFS de retículas de
    dimension conocida. Si pi(r) ~ r^(d-2), entonces pi es CONSTANTE si y solo si d=2 -- degeneracion
    algebraica: el juez no mide 'planitud', mide 'bidimensionalidad'."""
    filas = []
    for d, (adj, N) in adjs_por_d.items():
        D = _dist_bfs(adj, N)
        rng = RNG(1234)
        fuentes = [int(s) for s in rng.integers(0, N, size=min(8, N))]
        maxr = int(np.nanmax(D))
        for r in range(1, maxr + 1):
            cnt = np.mean([np.sum(D[s] == r) for s in fuentes])
            if cnt > 0:
                filas.append(dict(d=d, N=N, r=r, cascaron=float(cnt), pi=float(cnt / (2.0 * r))))
    import csv
    with open(salida_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["d", "N", "r", "cascaron", "pi"])
        w.writeheader(); w.writerows(filas)
    return filas


def main():
    solo = os.environ.get("CS092_SOLO")          # p.ej. "reticula2d,anillo1d"
    semillas = [int(x) for x in os.environ.get("CS092_SEEDS", "91001,91002").split(",")]
    con_q = os.environ.get("CS092_SIN_Q", "0") != "1"
    nombres = list(SUSTRATOS.keys()) if not solo else solo.split(",")

    print("=" * 110, flush=True)
    print("CS092 — CONTROL POSITIVO del instrumento de orden global (C-N2.6.4)", flush=True)
    print(f"sustratos={nombres}  semillas={semillas}  via_Q={'si' if con_q else 'no'}", flush=True)
    print("=" * 110, flush=True)
    t0 = time.time()
    filas = []
    for nom in nombres:
        _, escalera, _ = SUSTRATOS[nom]
        for N_obj in escalera:
            for s in semillas:
                r = _mide_un_caso(nom, N_obj, s, con_via_q=con_q)
                filas.append(r)
                print(f"  [{nom:15s} N={r['N']:5d} s={s}] "
                      f"M: diam={r['M_diam_robusto']:6.1f}/{r['M_diam_gigante']:6.1f} "
                      f"pi_cv={r['M_pi_cv']:.3f} n_ejes={r['M_n_ejes']} pico={r['M_pico_medio']:.3f} | "
                      + (f"Q: diam={r.get('Q_diam', float('nan')):6.2f} pi_cv={r.get('Q_pi_cv', float('nan')):.3f} "
                         f"n_ejes={r.get('Q_n_ejes','-')} pico={r.get('Q_pico_medio', float('nan')):.3f}" if con_q else "")
                      + f"  ({r['seg']:.1f}s, t={(time.time()-t0)/60:.1f}min)", flush=True)
        with open(os.path.join(_HERE, "cs092_control_positivo_crudo.json"), "w") as f:
            json.dump(filas, f, indent=1, default=float)
    print(f"\ntiempo total {(time.time()-t0)/60:.2f} min", flush=True)


if __name__ == "__main__":
    main()

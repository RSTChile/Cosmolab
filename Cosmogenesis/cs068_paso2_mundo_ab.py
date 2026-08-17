"""
CS068 Paso 2 — Mundo A vs Mundo B: ¿el blob real de CS067 tiene tejido métrico latente?
==============================================================================
Ruling de CS (ADJUDICACION_CS068_paso1_resultado_CS.md, 16-jul-2026): Paso 1 des-arriesgó la maquinaria
(el proceso estirar-enfriar SÍ produce gradiente ordenado con verdad de fondo). Ahora la pregunta viva del
arco: ¿el blob real de CS067 (mundo-pequeño, SIN retícula base conocida) tiene tejido métrico latente que
el enfriamiento pueda revelar (Mundo A), o es mundo-pequeño hasta el fondo, sin nada debajo que revelar
(Mundo B)? Ambas salidas son resultados reales -- no hay resultado "malo".

Test (ADJUDICACION_CS068_etapa1_tejido_latente_CS.md, "Paso 2"): comparar el blob real contra su propio
NULL de reconexión que preserva la secuencia de grados (configuration model, vía double-edge-swap). Si el
blob real tiene MÁS soporte local (vecinos comunes por arista) que su versión reconectada al azar -> hay
estructura métrica de vecindario por encima de lo que la mera secuencia de grados explica -> Mundo A. Si es
indistinguible de su propia reconexión aleatoria -> Mundo B: el sustrato nunca tuvo geometría latente bajo
los atajos, y CS066/CS067 nunca encendieron direcciones porque no había piso que encender, no porque
faltara una pieza.

Estadístico: soporte medio (vecinos comunes) sobre TODAS las aristas del grafo, en el blob real vs en K
realizaciones de reconexión completa que preservan la secuencia de grados EXACTA del mismo blob (double-
edge-swap, no una construcción nueva). z-score y percentil del valor real dentro de la distribución NULL
(K muestras independientes de la estadística, no aristas agrupadas -- evita pseudo-replicación).

Regla de decisión PRE-INSCRITA (antes de correr, no se ajusta después de ver el número):
Mundo A si z > 2.0 (equivalente a ~p<0.025 a una cola) en TODAS las semillas corridas (blindaje). Mundo B
si no. Zona gris (0 < z <= 2.0 en alguna semilla) se reporta tal cual, sin forzar veredicto.

Codea/ejecuta: CC. Diseño/ruling: CS.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs068_inflacion_estirar_enfriar as E

RNG = np.random.default_rng

FACTOR_SWAPS = 10  # nº de swaps exitosos por realización = FACTOR_SWAPS * |E| -- estándar para "mezclar bien"
                    # preservando grados (regla de dedo de la literatura de configuration-model MCMC). Fijo.


def _edges_list(adj, N):
    return [(i, j) for i in range(N) for j in adj[i] if i < j]


def _soporte(adj, i, j):
    return len(adj[i] & adj[j])


def _soporte_medio(adj, edges):
    if not edges:
        return 0.0
    return float(np.mean([_soporte(adj, i, j) for (i, j) in edges]))


def _double_edge_swap(adj, N, n_swaps, rng):
    """Reconexión que preserva la secuencia de grados EXACTA (configuration model vía cadena de Markov de
    double-edge-swap): toma dos aristas (a,b),(c,d) y las reemplaza por (a,d),(c,b) si el resultado no crea
    self-loop ni arista duplicada. Repite hasta lograr n_swaps ÉXITOS (no intentos). No usa librerías
    externas (no hay networkx en el venv) -- implementación directa sobre sets."""
    a2 = [set(s) for s in adj]
    edges = _edges_list(a2, N)
    edge_set = set(edges)
    n = len(edges)
    if n < 2:
        return a2
    exitos = 0
    intentos = 0
    tope = max(n_swaps, 1) * 30
    while exitos < n_swaps and intentos < tope:
        intentos += 1
        i1, i2 = rng.integers(0, n, size=2)
        if i1 == i2:
            continue
        a, b = edges[i1]
        c, d = edges[i2]
        if len({a, b, c, d}) < 4:
            continue
        n1 = (a, d) if a < d else (d, a)
        n2 = (c, b) if c < b else (b, c)
        if n1[0] == n1[1] or n2[0] == n2[1]:
            continue
        if n1 in edge_set or n2 in edge_set:
            continue
        old1, old2 = edges[i1], edges[i2]
        a2[old1[0]].discard(old1[1]); a2[old1[1]].discard(old1[0])
        a2[old2[0]].discard(old2[1]); a2[old2[1]].discard(old2[0])
        a2[n1[0]].add(n1[1]); a2[n1[1]].add(n1[0])
        a2[n2[0]].add(n2[1]); a2[n2[1]].add(n2[0])
        edge_set.discard(old1); edge_set.discard(old2)
        edge_set.add(n1); edge_set.add(n2)
        edges[i1] = n1; edges[i2] = n2
        exitos += 1
    return a2


# ============================ Clasificador config-model (para USAR solo si Mundo A) ============================
def _null_soporte_por_bin(adj, N, rng, n_realizaciones=30, factor_swaps=FACTOR_SWAPS, n_bins=10):
    """K realizaciones de reconexión completa (double-edge-swap, preserva grados EXACTOS). Acumula el
    soporte de TODAS las aristas de cada realización, agrupado por bin de grado (decil de deg_i+deg_j sobre
    la distribución REAL de grados, fija -- los grados no cambian por el rewiring). Da media/std del NULL
    por bin, para comparar cada arista real contra aristas 'parecidas en grado' bajo el null, no contra el
    pool global (que mezclaría hubs con nodos de grado bajo)."""
    grados = np.array([len(s) for s in adj])
    edges_reales = _edges_list(adj, N)
    suma_reales = np.array([grados[i] + grados[j] for (i, j) in edges_reales], float)
    cortes = np.quantile(suma_reales, np.linspace(0, 1, n_bins + 1))
    cortes[0] -= 1; cortes[-1] += 1

    def _bin_de(dg):
        return int(min(n_bins - 1, max(0, np.searchsorted(cortes, dg, side="right") - 1)))

    acumulado = [[] for _ in range(n_bins)]
    n_swaps = factor_swaps * len(edges_reales)
    for r in range(n_realizaciones):
        rk = RNG(int(rng.integers(0, 2**31)))
        a2 = _double_edge_swap(adj, N, n_swaps, rk)
        for (i, j) in _edges_list(a2, N):
            acumulado[_bin_de(grados[i] + grados[j])].append(_soporte(a2, i, j))

    stats = []
    for b in range(n_bins):
        v = np.array(acumulado[b], float) if acumulado[b] else np.array([0.0])
        mu = float(v.mean())
        sd = float(v.std(ddof=1)) if len(v) > 1 else 1e-9
        stats.append((mu, max(sd, 1e-9)))
    return grados, cortes, stats, _bin_de


def clasifica_config_model(adj, N, rng, n_realizaciones=30, factor_swaps=FACTOR_SWAPS):
    """Adjudicado por CS (ADJUDICACION_CS068_etapa1_tejido_latente_CS.md): una arista es TEJIDO si su
    soporte EXCEDE el esperado bajo reconexión que preserva grados (z>0 respecto al NULL de su bin de
    grado); ATAJO si no. El NULL fija el umbral -- no hay número elegido a mano. Reemplaza la mediana de
    _clasifica() en cs068_inflacion_estirar_enfriar.py (criticada porque el blob real no tiene retícula
    base y la mediana no mide nada contra un fondo)."""
    grados, cortes, stats, bin_de = _null_soporte_por_bin(adj, N, rng, n_realizaciones, factor_swaps)
    edges = _edges_list(adj, N)
    local_edges, atajos, zs = [], [], []
    for (i, j) in edges:
        mu, sd = stats[bin_de(grados[i] + grados[j])]
        z = (_soporte(adj, i, j) - mu) / sd
        zs.append(z)
        (local_edges if z > 0 else atajos).append((i, j))
    adj_local = [set() for _ in range(N)]
    for (i, j) in local_edges:
        adj_local[i].add(j); adj_local[j].add(i)
    return adj_local, atajos, np.array(zs)


def _mundo_ab_una_semilla(N, seed, K, rng_global):
    adj = E._sustrato(N, seed)
    edges = _edges_list(adj, N)
    mean_real = _soporte_medio(adj, edges)

    n_swaps = FACTOR_SWAPS * len(edges)
    nulls = []
    for k in range(K):
        rk = RNG(seed * 1000 + k + 1)
        a2 = _double_edge_swap(adj, N, n_swaps, rk)
        e2 = _edges_list(a2, N)
        nulls.append(_soporte_medio(a2, e2))
    nulls = np.array(nulls)
    null_mean = float(nulls.mean())
    null_std = float(nulls.std(ddof=1)) if K > 1 else 1e-9
    z = (mean_real - null_mean) / max(null_std, 1e-9)
    percentil = float((nulls < mean_real).mean())
    return dict(N=N, seed=seed, n_edges=len(edges), mean_real=mean_real, null_mean=null_mean,
                null_std=null_std, z=z, percentil=percentil, K=K)


def main():
    N = int(os.environ.get("CS068_N", 1500))
    K = int(os.environ.get("CS068_K", 30))
    n_seeds = int(os.environ.get("CS068_SEEDS", 5))
    print("=" * 108, flush=True)
    print("CS068 PASO 2 — MUNDO A vs MUNDO B: ¿el blob real de CS067 tiene tejido métrico latente", flush=True)
    print("(más soporte local que su propia reconexión que preserva grados) o es mundo-pequeño hasta el fondo?",
          flush=True)
    print(f"N={N} · K_reconexiones={K} · semillas={n_seeds} · factor_swaps={FACTOR_SWAPS}", flush=True)
    print("=" * 108, flush=True)
    t0 = time.time()
    resultados = []
    for s in range(n_seeds):
        seed = 68200 + 97 * s
        r = _mundo_ab_una_semilla(N, seed, K, RNG(seed))
        resultados.append(r)
        print(f"\n--- semilla {s} (seed={seed}) ---", flush=True)
        print(f"  n_edges={r['n_edges']}  soporte_medio REAL={r['mean_real']:.4f}", flush=True)
        print(f"  soporte_medio NULL(config-model, n={K}): media={r['null_mean']:.4f} std={r['null_std']:.4f}",
              flush=True)
        print(f"  z={r['z']:+.3f}  percentil_real_en_null={r['percentil']:.3f}", flush=True)
        print(f"  tiempo acumulado: {(time.time()-t0)/60:.2f} min", flush=True)

    zs = np.array([r["z"] for r in resultados])
    print("\n" + "=" * 108, flush=True)
    print(f"z por semilla: {[round(float(z),3) for z in zs]}", flush=True)
    print(f"z medio={zs.mean():.3f}  min={zs.min():.3f}  max={zs.max():.3f}", flush=True)
    if np.all(zs > 2.0):
        print("\nVEREDICTO: MUNDO A. El blob real tiene MÁS soporte local que su propia reconexión que", flush=True)
        print("preserva grados, en TODAS las semillas (z>2.0). Hay estructura métrica de vecindario por", flush=True)
        print("encima de lo que la secuencia de grados sola explica -> hay tejido latente que el", flush=True)
        print("enfriamiento puede intentar revelar. Proceder con el clasificador config-model + re-correr", flush=True)
        print("Etapa 1 sobre el tejido/atajo así clasificado en el blob real.", flush=True)
    elif np.all(zs <= 0.5):
        print("\nVEREDICTO: MUNDO B. El blob real es INDISTINGUIBLE de su propia reconexión al azar que", flush=True)
        print("preserva grados. El sustrato de CS067 nunca tuvo geometría latente bajo los atajos -- no hay", flush=True)
        print("tejido que el enfriamiento pueda revelar porque no hay nada debajo. Esto re-ata con el arco:", flush=True)
        print("CS066/CS067 nunca encendieron direcciones porque el sustrato jamás fue métrico, no porque", flush=True)
        print("faltara una pieza. Resultado honesto, no un fracaso.", flush=True)
    else:
        print("\nVEREDICTO: ZONA GRIS. No cumple el criterio pre-inscrito limpio en ambas direcciones.", flush=True)
        print("Reportar tal cual a CS, sin forzar Mundo A ni Mundo B.", flush=True)
    print(f"\ntiempo total: {(time.time()-t0)/60:.2f} min", flush=True)


if __name__ == "__main__":
    main()

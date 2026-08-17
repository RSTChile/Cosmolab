"""
CS072 v6 — EXPLORATORIA (§9): ¿el contacto-por-roce fuerza dimensión, o la dimensión emerge medida?
==============================================================================
Verifica, ANTES de plegar las 10 leyes y correr la tanda de veredicto:
(a) el contacto-por-roce NO fuerza dimensión (barre ε/N, comprueba que β/d_s NO son un artefacto de
    construcción -- a diferencia de v5, donde el orden-por-escalar daba β≈d=1 SIEMPRE, algebraicamente).
(b) no colapsa (T diverge) ni lava trivialmente (CV->0) -- ya verificado en el informe anterior que CV
    crece desde ε=1e-6 hasta 1e-2, acotado (T>=0 por construcción).
(c) el NULL_BARAJADO y el control positivo (retícula sembrada) se comportan como deben.

Codea/ejecuta: CC. Diseño/ruling: CS.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs072_v6_nucleo as V6
import cs071_histeresis as S71
import cs071_tanda as S71T
import cs064_smoke as SM
import cs068_paso1_sintetico as S1

RNG = np.random.default_rng
NS_BETA = [400, 900, 1600]
PASOS = 80


def _judges(adj, N, rng):
    diam = S71._diam_robusto(adj, N, rng)
    delta_g = S71._delta_gromov(adj, N, rng)
    ds = SM.dim_volumen(adj, N, rng=rng)
    frac_gig = S71._frac_gigante(adj, N, rng)
    return dict(diam=diam, delta_gromov=delta_g, d_s=ds, frac_gigante=frac_gig)


def control_positivo_bootstrap(N, seed):
    """Sustituye SOLO el bootstrap: retícula 2D limpia (métrica de verdad) en vez de grafo aleatorio --
    mismo proceso (gravedad+flujo+memoria) encima, para validar que el juez detecta metricidad y que el
    proceso no la destruye (como en CS071)."""
    side = int(round(N ** 0.5))
    adj0, N2 = S1._reticula_2d(side)
    return adj0, N2


def corre_v6_con_bootstrap(N, n_focos, delta, pasos, seed, adj_inicial_fn=None, barajado=False):
    """Igual que V6.corre_nucleo_v6 pero permite inyectar un bootstrap distinto (control positivo)."""
    rng = V6.RNG(seed)
    T = V6.V5.campo_inicial(N, n_focos, delta, rng)
    if adj_inicial_fn is None:
        adj = V6._bootstrap(N, int(rng.integers(1 << 30)))
    else:
        adj, N2 = adj_inicial_fn(N, int(rng.integers(1 << 30)))
        assert N2 == N
    deg0_hist = np.array([len(a) for a in adj], float)
    w_hist = {(i, j): 1.0 for i in range(N) for j in adj[i] if i < j}
    cvs = [V6.V5.cv_heterogeneidad(T)]
    for _ in range(pasos):
        V6._gravedad(adj, N, T, rng)
        T_antes = T.copy()
        edges_vivos = [(i, j) for i in range(N) for j in adj[i] if i < j]
        T = V6.flujo_enfriamiento(T, edges_vivos, N)
        w_hist = V6._memoria_roce(adj, w_hist, N, deg0_hist, T_antes, T, rng, barajado=barajado)
        T = V6.V5.enfria(T)
        cvs.append(V6.V5.cv_heterogeneidad(T))
    return dict(T_final=T, adj=adj, cvs=cvs)


def main():
    t0 = time.time()
    print("=" * 100, flush=True)
    print("CS072 v6 EXPLORATORIA -- (a) invarianza a ε, (b) β sobre 3 N (TODO/NULL/control)", flush=True)
    print("=" * 100, flush=True)

    print("\n--- (a) invarianza a ε (N=400, 1 foco) ---", flush=True)
    for delta in [1e-2, 1e-4, 1e-6]:
        r = corre_v6_con_bootstrap(400, 1, delta, PASOS, 72800)
        j = _judges(r["adj"], 400, RNG(1))
        print(f"  delta={delta:.0e}: CV[-1]={r['cvs'][-1]:.3f}  diam={j['diam']:.2f}  d_s={j['d_s']:.2f}  "
              f"delta_gromov={j['delta_gromov']:.2f}  frac_gigante={j['frac_gigante']:.3f}", flush=True)

    print("\n--- (b) β sobre 3 N: TODO vs NULL_BARAJADO vs CONTROL_POSITIVO (ε: 1 foco, delta=1e-4) ---",
          flush=True)
    for nombre, kwargs in [("TODO", dict(barajado=False)),
                            ("NULL_BARAJADO", dict(barajado=True)),
                            ("CONTROL_POSITIVO", dict(barajado=False, adj_inicial_fn=control_positivo_bootstrap))]:
        diams = []
        for N in NS_BETA:
            r = corre_v6_con_bootstrap(N, 1, 1e-4, PASOS, 72900, **kwargs)
            j = _judges(r["adj"], N, RNG(2))
            diams.append(j["diam"])
            print(f"  [{nombre}] N={N}: CV[-1]={r['cvs'][-1]:.3f} diam={j['diam']:.2f} d_s={j['d_s']:.2f} "
                  f"delta_g={j['delta_gromov']:.2f} frac_gig={j['frac_gigante']:.3f} "
                  f"(t={(time.time()-t0)/60:.1f}min)", flush=True)
        Ns_validos = [N for N, d in zip(NS_BETA, diams) if np.isfinite(d) and d > 0]
        diams_validos = [d for d in diams if np.isfinite(d) and d > 0]
        if len(Ns_validos) >= 2:
            beta, _ = S71T._pendiente_loglog(Ns_validos, diams_validos)
            print(f"  [{nombre}] β (pendiente log-log diam vs N) = {beta:.3f}", flush=True)
        else:
            print(f"  [{nombre}] β: no computable (diam no finito)", flush=True)

    print(f"\ntiempo total: {(time.time()-t0)/60:.2f} min", flush=True)


if __name__ == "__main__":
    main()

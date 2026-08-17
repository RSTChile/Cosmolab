"""
CS072 -- LA ÚNICA CORRIDA DE VEREDICTO. Motor de perillas (cs072_fold_completo.py, ya corregido y validado:
las 21 piezas confirmadas corriendo cada paso, NULL con permutaciones independientes, poda que baja el peso
real, sin try/except silenciosos). REAL vs NULL_CATALOGO, curvas de filtración COMPLETAS + β (multi-N,
onset-de-persistencia) + segundo sello (δ-Gromov, dos transformaciones). Un solo veredicto.

Codea/ejecuta: CC. Diseño/ruling: CS + director + Codex.
"""
from __future__ import annotations
import sys, time, json
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs072_fold_completo as FOLD
import cs072_ii_filtracion as FIL
import cs071_histeresis as S71
import cs064_smoke as SM

RNG = np.random.default_rng
NS_BETA = [100, 200, 400, 800]
N_CURVA = 400          # N representativo para publicar la curva de filtración completa
PODA_TASA = 0.05        # heredada (mismo orden que v7)
PASOS = FOLD.STEPS
SEED_BASE = 72180


def _onset_persistencia(W, N, frac_umbral, rng):
    bloques = FIL._bloques_de_empate(W, N)
    uf = FIL._UnionFind(N)
    adj = [set() for _ in range(N)]
    total = N * (N - 1) // 2
    incl = 0
    for _, pares in bloques:
        for (i, j) in pares:
            uf.union(i, j); adj[i].add(j); adj[j].add(i)
        incl += len(pares)
        if uf.tam_max() / N >= frac_umbral:
            diam = S71._diam_robusto(adj, N, rng)
            ds = SM.dim_volumen(adj, N, rng=rng)
            return dict(frac_pares=incl / total, diam=diam, d_s=ds, frac_gigante=uf.tam_max() / N)
    return dict(frac_pares=float("nan"), diam=float("nan"), d_s=float("nan"), frac_gigante=uf.tam_max() / N)


def _beta_de_brazo(arm, seed_off):
    diams = []
    detalle = []
    for N in NS_BETA:
        r = FOLD.corre_fold(N=N, arm=arm, seed=SEED_BASE + seed_off + N, pasos=PASOS, poda_tasa=PODA_TASA)
        rng = RNG(SEED_BASE + seed_off + N + 1)
        onset = _onset_persistencia(r["W"], N, frac_umbral=0.9, rng=rng)
        diams.append(onset["diam"])
        detalle.append(dict(N=N, **onset))
    Ns_v = [N for N, d in zip(NS_BETA, diams) if np.isfinite(d) and d > 0]
    diams_v = [d for d in diams if np.isfinite(d) and d > 0]
    if len(Ns_v) >= 2:
        x = np.log(Ns_v); y = np.log(diams_v)
        A = np.vstack([x, np.ones_like(x)]).T
        beta, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    else:
        beta = float("nan")
    return beta, detalle


def main():
    t0 = time.time()
    print("=" * 100, flush=True)
    print("CS072 -- TANDA DE VEREDICTO: REAL vs NULL_CATALOGO (motor de perillas, 21 piezas confirmadas)",
          flush=True)
    print("=" * 100, flush=True)

    resultados = {}
    for arm, seed_off in [("real", 0), ("null_catalogo", 10000)]:
        print(f"\n--- brazo: {arm} ---", flush=True)
        beta, detalle = _beta_de_brazo(arm, seed_off)
        for d in detalle:
            print(f"  N={d['N']}: diam(onset)={d['diam']} frac_gigante={d['frac_gigante']:.3f} "
                  f"d_s={d['d_s']} frac_pares_onset={d['frac_pares']:.4f}  (t={(time.time()-t0)/60:.1f}min)",
                  flush=True)
        print(f"  >>> beta({arm}) = {beta:.3f}", flush=True)
        resultados[arm] = dict(beta=beta, detalle=detalle)

    print("\n" + "=" * 100, flush=True)
    print(f"CURVA DE FILTRACIÓN COMPLETA en N={N_CURVA} (REAL vs NULL) + segundo sello", flush=True)
    print("=" * 100, flush=True)
    for arm, seed_off in [("real", 0), ("null_catalogo", 10000)]:
        r = FOLD.corre_fold(N=N_CURVA, arm=arm, seed=SEED_BASE + seed_off + N_CURVA, pasos=PASOS,
                             poda_tasa=PODA_TASA)
        rng = RNG(SEED_BASE + seed_off + 99)
        curva, adj = FIL.curva_filtracion(r["W"], N_CURVA, n_checkpoints_judges=15, rng_judges=rng)
        print(f"\n  [{arm}] curva completa (n_niveles totales={len(curva)}):", flush=True)
        for item in curva:
            if "diam" in item:
                print(f"    frac_pares={item['frac_pares']:.3f}  frac_gigante={item['frac_gigante']:.3f}  "
                      f"diam={item['diam']}  d_s={item.get('d_s')}", flush=True)
        jueces = FIL.jueces_continuos_sin_umbral(r["W"], N_CURVA)
        sello = FIL.segundo_sello(r["W"], N_CURVA, rng, n_landmarks=40, n_quad=300)
        print(f"  [{arm}] jueces continuos: {jueces}", flush=True)
        print(f"  [{arm}] segundo sello: {sello}", flush=True)
        resultados.setdefault("curvas", {})[arm] = dict(
            n_niveles=len(curva), jueces=jueces, sello=sello,
            curva_checkpoints=[it for it in curva if "diam" in it])

    with open("cs072_fold_tanda_resultados.json", "w") as f:
        json.dump(resultados, f, indent=2, default=str)

    print(f"\ntiempo total: {(time.time()-t0)/60:.2f} min", flush=True)


if __name__ == "__main__":
    main()

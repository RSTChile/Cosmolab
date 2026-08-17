"""
CS071 — TANDA BLINDADA: 4 brazos, N∈{400,900,1600}, ≥8 semillas/brazo
==============================================================================
Cronograma FIJO (REFUERZO=0.04, DECAY=0.99, PRUNE_FRAC=0.15, ver cs071_histeresis.py -- calibrado en
corrida exploratoria declarada, criterio = evitar degeneración de conectividad, NUNCA "acercarse a √N").

Juez: pendiente log-log de diam(N) en 3 escalas (β en diam~N^β). β≈0.5=métrico, β≈0=mundo-pequeño,
diam→2-3 con grado_max disparado=HUB (degeneración, NO cuenta como métrico -- G-ANTI-HUB).

Lectura pre-inscrita (DISENO_CS071_histeresis_memoria_enlace_CS.md):
(A) β≈0.5 SOLO en HISTERESIS, no en NULL_BARAJADO, sin hub -> la memoria fabrica métrica.
(B) HISTERESIS ≈ NULL_BARAJADO ≈ SIN_PROCESO en log N -> la memoria no metriciza. Cuarta ruta al muro.
(C) HISTERESIS colapsa a hub, NULL no -> el proceso hace algo real pero en dirección equivocada.

Codea/ejecuta: CC. Diseño/ruling: CS.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs071_histeresis as H

RNG = np.random.default_rng
NS = [400, 900, 1600]
N_SEEDS = 8
ARMS = ["histeresis", "null_barajado", "sin_proceso", "histeresis_sobre_reticula"]


def _pendiente_loglog(Ns, vals):
    x = np.log(np.array(Ns, float)); y = np.log(np.array(vals, float))
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m), float(b)


def _ic95(vals):
    x = np.array([v for v in vals if np.isfinite(v)], float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(x.mean())
    sem = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    return m, m - 1.96 * sem, m + 1.96 * sem


def main():
    t0 = time.time()
    print("=" * 108, flush=True)
    print("CS071 TANDA BLINDADA", flush=True)
    print(f"brazos={ARMS}  N={NS}  semillas/brazo={N_SEEDS}", flush=True)
    print(f"cronograma: REFUERZO={H.REFUERZO} DECAY={H.DECAY} PRUNE_FRAC={H.PRUNE_FRAC} "
          f"PASOS={H.PASOS_PROCESO} WS_K={H.WS_K} WS_P={H.WS_P}", flush=True)
    print("=" * 108, flush=True)
    resultados = []
    for arm in ARMS:
        fn = H.BRAZOS[arm]
        for i, N in enumerate(NS):
            for s in range(N_SEEDS):
                seed = 71200 + 1000 * ARMS.index(arm) + 97 * i + 13 * s
                adj = fn(N, seed)
                r = H.evalua(adj, N, seed)
                r["arm"] = arm; r["N"] = N; r["seed"] = seed
                resultados.append(r)
                print(f"  [{arm:26s} N={N:5d} s={s}] diam={r['diam']} frac_gig={r['frac_gigante']:.3f} "
                      f"delta_g={r['delta_gromov']:.3f} grado_max={r['grado_max']} grado_medio={r['grado_medio']:.2f} "
                      f"(t={(time.time()-t0)/60:.1f}min)", flush=True)
    with open(os.path.join(_HERE, "cs071_tanda_resultados.json"), "w") as f:
        json.dump(resultados, f, indent=1, default=float)
    print(f"\ntiempo total: {(time.time()-t0)/60:.2f} min -- resultados en cs071_tanda_resultados.json", flush=True)

    print("\n" + "=" * 108, flush=True)
    print("AGREGADO por brazo (blindaje de semillas)", flush=True)
    print("=" * 108, flush=True)
    resumen = {}
    for arm in ARMS:
        rs = [r for r in resultados if r["arm"] == arm]
        diam_por_N = []
        for N in NS:
            rn = [r["diam"] for r in rs if r["N"] == N and np.isfinite(r["diam"])]
            diam_por_N.append(float(np.mean(rn)) if rn else float("nan"))
        Ns_validos = [N for N, d in zip(NS, diam_por_N) if np.isfinite(d)]
        diams_validos = [d for d in diam_por_N if np.isfinite(d)]
        beta, _b0 = _pendiente_loglog(Ns_validos, diams_validos) if len(Ns_validos) >= 2 else (float("nan"), 0)
        frac_gig_m, frac_gig_lo, frac_gig_hi = _ic95([r["frac_gigante"] for r in rs])
        delta_g_m, delta_g_lo, delta_g_hi = _ic95([r["delta_gromov"] for r in rs])
        gmax_m, _, _ = _ic95([r["grado_max"] for r in rs])
        gmed_m, _, _ = _ic95([r["grado_medio"] for r in rs])
        hub = gmax_m > 3 * (H.WS_K if arm != "histeresis_sobre_reticula" else 4)
        resumen[arm] = dict(beta=beta, diam_por_N=list(zip(NS, diam_por_N)), frac_gigante=frac_gig_m,
                             delta_gromov=delta_g_m, grado_max_medio=gmax_m, grado_medio_medio=gmed_m, hub=hub)
        print(f"\n  {arm}:", flush=True)
        print(f"    diam(N): {list(zip(NS, [round(d,2) for d in diam_por_N]))}", flush=True)
        print(f"    β (pendiente log-log) = {beta:.3f}  (0.5=métrico, 0=mundo-pequeño)", flush=True)
        print(f"    frac_gigante media={frac_gig_m:.3f} IC95%=[{frac_gig_lo:.3f},{frac_gig_hi:.3f}]", flush=True)
        print(f"    δ-Gromov media={delta_g_m:.3f}  grado_max medio={gmax_m:.2f}  grado_medio medio={gmed_m:.2f}  "
              f"HUB={'SÍ' if hub else 'no'}", flush=True)

    print("\n" + "=" * 108, flush=True)
    print("VEREDICTO", flush=True)
    print("=" * 108, flush=True)
    rh = resumen["histeresis"]; rb = resumen["null_barajado"]; rs_ = resumen["sin_proceso"]
    rl = resumen["histeresis_sobre_reticula"]
    metrico_h = rh["beta"] > 0.35 and not rh["hub"] and rh["frac_gigante"] > 0.5
    metrico_b = rb["beta"] > 0.35 and not rb["hub"]
    print(f"  histeresis: β={rh['beta']:.3f} hub={rh['hub']} frac_gigante={rh['frac_gigante']:.3f} "
          f"-> {'MÉTRICO' if metrico_h else 'no métrico'}", flush=True)
    print(f"  null_barajado: β={rb['beta']:.3f} hub={rb['hub']} -> {'MÉTRICO' if metrico_b else 'no métrico'}",
          flush=True)
    print(f"  sin_proceso: β={rs_['beta']:.3f}", flush=True)
    print(f"  histeresis_sobre_reticula (control positivo): β={rl['beta']:.3f} frac_gigante={rl['frac_gigante']:.3f} "
          f"hub={rl['hub']}", flush=True)

    if metrico_h and not metrico_b:
        print("\n(A) β≈0.5 en HISTERESIS, no en NULL_BARAJADO, sin hub -> LA MEMORIA FABRICA MÉTRICA.", flush=True)
        print("Refuta la predicción pre-registrada de CS. Primer mecanismo del arco que rompe el muro por", flush=True)
        print("auto-organización. Replicar y estresar antes de cantar victoria.", flush=True)
    elif rh["hub"] and not rb["hub"]:
        print("\n(C) HISTERESIS colapsa a hub, NULL_BARAJADO no -> el proceso hace algo real (se distingue", flush=True)
        print("del azar) pero en la dirección equivocada -- concentra en vez de distribuir. Positivo de", flush=True)
        print("mecanismo, veredicto de geometría negativo (como CS054-v2).", flush=True)
    else:
        print("\n(B) HISTERESIS ≈ NULL_BARAJADO ≈ SIN_PROCESO en log N -> la memoria NO metriciza. Cuarta", flush=True)
        print("ruta al mismo muro. Consistente con la predicción pre-registrada de CS y con el mecanismo", flush=True)
        print("medido (el tránsito ciego carga los atajos).", flush=True)
    print(f"\ntiempo total: {(time.time()-t0)/60:.2f} min", flush=True)


if __name__ == "__main__":
    main()

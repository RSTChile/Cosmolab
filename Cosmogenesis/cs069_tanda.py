"""
CS069 — TANDA BLINDADA: los 4 brazos × los 3 jueces, N∈{900,1500,2500}, ≥8 semillas/brazo
==============================================================================
Autorizada por CS (ADJUDICACION_CS069_smoke_regla_fase_CS.md): las 3 anclas de smoke pasan con la regla de
fase v2 (frustración entre extremos, Kuramoto por nodo ponderado por ρ). Cuerda decisiva: COMPLETO vs
NULL_FASE_TOPO en los 3 jueces.

Lectura pre-inscrita (DISENO_CS069_frente_cuantico_CS.md):
(A) COMPLETO gana a NULL_FASE_TOPO en los 3 jueces → la superposición enciende la dirección.
(B) COMPLETO ≈ NULL_FASE_TOPO → Mundo B se extiende al régimen cuántico.
(C) COMPLETO gana solo a NULL_FASE_AZAR (no a NULL_FASE_TOPO) → era clustering, no cuántico.

Codea/ejecuta: CC. Diseño/ruling: CS.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs069_quantum_graph as Q
import cs068_inflacion_estirar_enfriar as E

RNG = np.random.default_rng

NS = [900, 1500, 2500]
N_SEEDS = 8
ARMS = ["completo", "null_fase_topo", "null_fase_azar", "null_clasico"]


def _ic95(vals):
    x = np.array(vals, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(x.mean())
    sem = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    return m, m - 1.96 * sem, m + 1.96 * sem


def _corre_un_caso(arm, N, seed):
    adj = E._sustrato(N, seed)
    rng = RNG(seed + 1)
    Dq = Q.BRAZOS[arm](adj, N, rng)
    diamq = Q.diam_q_robusto(Dq, N, RNG(seed + 2))
    pi_media, pi_cv = Q.cedazo_pi(Dq, N, RNG(seed + 3))
    juezc = Q.juez_gap_espectral(Dq, N)
    return dict(arm=arm, N=N, seed=seed, diamq=diamq, pi_media=pi_media, pi_cv=pi_cv, **juezc)


def main():
    t0 = time.time()
    print("=" * 108, flush=True)
    print("CS069 TANDA BLINDADA", flush=True)
    print(f"brazos={ARMS}  N={NS}  semillas/brazo={N_SEEDS}", flush=True)
    print("=" * 108, flush=True)
    resultados = []
    for arm in ARMS:
        for i, N in enumerate(NS):
            for s in range(N_SEEDS):
                seed = 69300 + 1000 * ARMS.index(arm) + 97 * i + 13 * s
                r = _corre_un_caso(arm, N, seed)
                resultados.append(r)
                print(f"  [{arm:15s} N={N:5d} s={s}] diamq={r['diamq']:.2f} pi_cv={r['pi_cv']:.3f} "
                      f"n_ejes={r['n_ejes']} pico_medio={r['pico_medio']:.3f} certificado={r['certificado']} "
                      f"(t={((time.time()-t0)/60):.1f}min)", flush=True)
    with open(os.path.join(_HERE, "cs069_tanda_resultados.json"), "w") as f:
        json.dump(resultados, f, indent=1, default=float)
    print(f"\ntiempo total: {(time.time()-t0)/60:.2f} min -- resultados en cs069_tanda_resultados.json", flush=True)

    # ============================ AGREGACION por brazo x N ============================
    print("\n" + "=" * 108, flush=True)
    print("AGREGADO por brazo (media+IC95% sobre las 3 escalas de N, blindaje de semillas)", flush=True)
    print("=" * 108, flush=True)
    resumen = {}
    for arm in ARMS:
        rs = [r for r in resultados if r["arm"] == arm]
        pi_cvs = [r["pi_cv"] for r in rs]
        n_ejes = [r["n_ejes"] for r in rs]
        pico_medios = [r["pico_medio"] for r in rs]
        certificados = [r["certificado"] for r in rs]
        Ns_arm = sorted(set(r["N"] for r in rs))
        diamq_por_N = [float(np.mean([r["diamq"] for r in rs if r["N"] == N])) for N in Ns_arm]
        pendiente, _ = Q._pendiente_loglog(Ns_arm, diamq_por_N)
        m_cv, lo_cv, hi_cv = _ic95(pi_cvs)
        m_pico, lo_pico, hi_pico = _ic95(pico_medios)
        frac_certificado = float(np.mean(certificados))
        m_nejes, lo_nejes, hi_nejes = _ic95(n_ejes)
        resumen[arm] = dict(pendiente_diamq=pendiente, pi_cv_media=m_cv, pi_cv_ic95=(lo_cv, hi_cv),
                             pico_medio_media=m_pico, pico_medio_ic95=(lo_pico, hi_pico),
                             frac_certificado=frac_certificado, n_ejes_media=m_nejes)
        print(f"\n  {arm}:", flush=True)
        print(f"    Juez B -- pendiente log-log diam_q(N) = {pendiente:.3f}", flush=True)
        print(f"    Juez A -- pi_cv media={m_cv:.3f} IC95%=[{lo_cv:.3f},{hi_cv:.3f}]  "
              f"(bajo=converge/'se congela', alto=estalla)", flush=True)
        print(f"    Juez C -- n_ejes media={m_nejes:.2f}  pico_medio media={m_pico:.3f} "
              f"IC95%=[{lo_pico:.3f},{hi_pico:.3f}]  frac_certificado(>0.85)={frac_certificado:.2f}", flush=True)

    # ============================ VEREDICTO — cuerda decisiva COMPLETO vs NULL_FASE_TOPO ============================
    print("\n" + "=" * 108, flush=True)
    print("VEREDICTO -- cuerda decisiva: COMPLETO vs NULL_FASE_TOPO en los 3 jueces", flush=True)
    print("=" * 108, flush=True)
    rc, rt, ra = resumen["completo"], resumen["null_fase_topo"], resumen["null_fase_azar"]
    gana_B = rc["pendiente_diamq"] > 0.3 and rt["pendiente_diamq"] <= 0.3
    gana_A = (rc["pi_cv_ic95"][1] < rt["pi_cv_ic95"][0])  # IC95% de completo por debajo del de topo, sin solape
    gana_C = rc["frac_certificado"] > rt["frac_certificado"] + 0.2  # margen no trivial, no un empate
    gana_C_vs_azar = rc["frac_certificado"] > ra["frac_certificado"] + 0.2

    print(f"  Juez A (pi converge): completo CV IC95% sup={rc['pi_cv_ic95'][1]:.3f} "
          f"{'<' if gana_A else '>='} topo CV IC95% inf={rt['pi_cv_ic95'][0]:.3f}  -> {'GANA' if gana_A else 'no gana'}",
          flush=True)
    print(f"  Juez B (pendiente diam_q): completo={rc['pendiente_diamq']:.3f} (>0.3) vs "
          f"topo={rt['pendiente_diamq']:.3f} (<=0.3)  -> {'GANA' if gana_B else 'no gana'}", flush=True)
    print(f"  Juez C (certificado picado>0.85): completo={rc['frac_certificado']:.2f} vs "
          f"topo={rt['frac_certificado']:.2f}  -> {'GANA' if gana_C else 'no gana'}", flush=True)

    if gana_A and gana_B and gana_C:
        print("\n(A) COMPLETO gana a NULL_FASE_TOPO en los 3 jueces -> LA SUPERPOSICIÓN ENCIENDE LA DIRECCIÓN.",
              flush=True)
        print("Primer 'sí' direccional del arco. π se congela al colapsar.", flush=True)
    elif gana_C_vs_azar and not gana_C:
        print("\n(C) COMPLETO gana solo a NULL_FASE_AZAR pero no a NULL_FASE_TOPO -> ERA CLUSTERING, no", flush=True)
        print("cuántico (la lección de CS068 repetida en el régimen cuántico).", flush=True)
    else:
        print("\n(B) COMPLETO ≈ NULL_FASE_TOPO -> Mundo B se extiende al régimen cuántico. El muro es más", flush=True)
        print("profundo que lo clásico. Resultado fuerte, honesto, pre-registrado.", flush=True)


if __name__ == "__main__":
    main()
